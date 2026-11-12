"""Offline benchmark for joint adaptive profiling and risk-aware plan selection."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from semanticflow.optimizer.adaptive import sampling_order, simulate_adaptive_profiling
from semanticflow.optimizer.bootstrap import collect_paired_bootstrap, order_key
from semanticflow.optimizer.economics import decide_adaptive_profiling_start
from semanticflow.optimizer.joint import (
    JOINT_SCHEMA_VERSION,
    analyze_joint_checkpoint,
    choose_joint_action,
    collect_checkpoint_uncertainty,
)
from semanticflow.optimizer.models import OptimizerSettings, ProfilingResult, SplitManifest
from semanticflow.optimizer.risk import select_risk_aware_plan
from semanticflow.optimizer.risk_benchmark import (
    DEFAULT_WORKLOADS,
    load_observations,
    observe_selection,
)
from semanticflow.optimizer.workflow import (
    _load_json,
    _profiling_report_path,
    load_profile_matrix,
)
from semanticflow.query.models import load_query
from semanticflow.storage.results import ResultStore, git_commit, git_worktree_dirty

JOINT_BENCHMARK_SCHEMA_VERSION = "m4-joint-adaptive-risk-benchmark-v1"
DEFAULT_HORIZONS = (1, 5, 10, 25, 50, 100, 250, 1000)
ABLATION_HORIZON = 10
ACTIVE_EXPECTED_REGRET_CAP_USD = 0.0001
ACTIVE_CVAR_LOSS_CAP_USD = 0.0


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _profile_cost(matrix, input_rate: float | None, output_rate: float | None) -> float | None:
    if input_rate is None or output_rate is None:
        return None
    total = 0.0
    for row in matrix.rows:
        for input_tokens, output_tokens in zip(row.input_tokens, row.output_tokens, strict=True):
            if input_tokens is None or output_tokens is None:
                return None
            total += (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
    return total


def _context(spec: Mapping[str, str], optimizer: OptimizerSettings) -> dict[str, Any]:
    query = load_query(spec["query"])
    profile_dir = Path(spec["profile_dir"])
    profile = ProfilingResult.model_validate(_load_json(_profiling_report_path(profile_dir)))
    split = SplitManifest.model_validate(_load_json(profile_dir / "split_manifest.json"))
    matrix = load_profile_matrix(profile_dir, profile, split)
    analysis = _load_json(Path(spec["analysis"]))
    original = [item.id for item in query.predicates]
    observations = load_observations(analysis)
    if observations["fixed_order"] != order_key(original):
        raise ValueError(f"{spec['id']} fixed order does not match its query")
    pricing = profile.pricing or {}
    input_rate = pricing.get("input_per_million_usd")
    output_rate = pricing.get("output_per_million_usd")
    profile_cost = float(profile.estimated_cost_usd or 0.0)
    historical = _load_json(Path(spec["historical_strategies"]))
    average_decision_cost = profile_cost / max(1, profile.completed_decisions)
    structural_bound = split.evaluation_size * (len(original) - 1) * average_decision_cost
    return {
        "id": spec["id"],
        "query": query,
        "profile": profile,
        "split": split,
        "matrix": matrix,
        "analysis": analysis,
        "observations": observations,
        "original": original,
        "input_rate": input_rate,
        "output_rate": output_rate,
        "profile_cost": profile_cost,
        "average_decision_cost": average_decision_cost,
        "structural_bound": structural_bound,
        "optimizer": optimizer,
        "historical_strategies": historical,
    }


def _checkpoint_packs(ctx: Mapping[str, Any]) -> list[dict[str, Any]]:
    optimizer: OptimizerSettings = ctx["optimizer"]
    matrix = ctx["matrix"]
    profile = ctx["profile"]
    split = ctx["split"]
    original = ctx["original"]
    sizes = [size for size in optimizer.adaptive.checkpoints if size <= matrix.size]
    ranked = sampling_order([row.review_id for row in matrix.rows], optimizer.adaptive.seed)
    packs = []
    for index, size in enumerate(sizes):
        evidence = matrix.select_reviews(ranked[:size])
        next_size = sizes[index + 1] if index + 1 < len(sizes) else None
        incurred = _profile_cost(evidence, ctx["input_rate"], ctx["output_rate"])
        increment = (
            incurred / size * (next_size - size)
            if incurred is not None and next_size is not None
            else None
        )
        common = dict(
            estimator_name="correlation_aware",
            original_order=original,
            objective=optimizer.planner.objective,
            evaluation_record_count=split.evaluation_size,
            question_hashes=profile.predicate_hashes,
            bootstrap=optimizer.bootstrap,
            correlation=optimizer.correlation,
            joint=optimizer.joint,
            input_price_per_million=ctx["input_rate"],
            output_price_per_million=ctx["output_rate"],
            tie_relative_tolerance=optimizer.planner.tie_relative_tolerance,
            seed=optimizer.joint.seed + size,
        )
        posterior = collect_checkpoint_uncertainty(evidence, **common)
        empirical = collect_checkpoint_uncertainty(
            evidence, sparse_posterior_predictive=False, **common
        )
        independent = collect_checkpoint_uncertainty(
            evidence,
            **{
                **common,
                "estimator_name": "independent",
                "seed": optimizer.joint.seed + 1000 + size,
            },
        )
        analyze_common = dict(
            evidence=evidence,
            next_sample_size=next_size,
            incremental_profiling_cost_usd=increment,
            expected_executions=1,
            estimator_name="correlation_aware",
            original_order=original,
            objective=optimizer.planner.objective,
            evaluation_record_count=split.evaluation_size,
            question_hashes=profile.predicate_hashes,
            bootstrap=optimizer.bootstrap,
            correlation=optimizer.correlation,
            economics=optimizer.economics.model_copy(update={"expected_executions": 1}),
            joint=optimizer.joint,
            input_price_per_million=ctx["input_rate"],
            output_price_per_million=ctx["output_rate"],
            tie_relative_tolerance=optimizer.planner.tie_relative_tolerance,
            seed=optimizer.joint.seed + size,
        )
        policies = {
            "joint": analyze_joint_checkpoint(
                risk=optimizer.risk,
                checkpoint_collected=posterior,
                **analyze_common,
            ),
            "empirical_sparse_ablation": analyze_joint_checkpoint(
                risk=optimizer.risk,
                checkpoint_collected=empirical,
                sparse_posterior_predictive=False,
                **analyze_common,
            ),
            "active_regret": analyze_joint_checkpoint(
                risk=optimizer.risk.model_copy(
                    update={"maximum_expected_regret": ACTIVE_EXPECTED_REGRET_CAP_USD}
                ),
                checkpoint_collected=posterior,
                **analyze_common,
            ),
            "active_cvar": analyze_joint_checkpoint(
                risk=optimizer.risk.model_copy(
                    update={"maximum_cvar_loss": ACTIVE_CVAR_LOSS_CAP_USD}
                ),
                checkpoint_collected=posterior,
                **analyze_common,
            ),
            "robust_disagreement": analyze_joint_checkpoint(
                risk=optimizer.risk,
                checkpoint_collected=posterior,
                robust=True,
                independent_collected=independent,
                **analyze_common,
            ),
        }
        packs.append(
            {
                "size": size,
                "next_size": next_size,
                # In-memory validation inputs. These are intentionally omitted from serialized
                # workload reports; the controller still receives only ``evidence``.
                "evidence": evidence,
                "posterior_collection": posterior,
                "empirical_collection": empirical,
                "incurred_cost_usd": incurred,
                "increment_cost_usd": increment,
                "policies": policies,
                "empirical_risk_collection": collect_paired_bootstrap(
                    evidence,
                    estimator_name="correlation_aware",
                    original_order=original,
                    objective=optimizer.planner.objective,
                    evaluation_record_count=split.evaluation_size,
                    question_hashes=profile.predicate_hashes,
                    settings=optimizer.bootstrap,
                    correlation=optimizer.correlation,
                    input_price_per_million=ctx["input_rate"],
                    output_price_per_million=ctx["output_rate"],
                    tie_relative_tolerance=optimizer.planner.tie_relative_tolerance,
                ),
            }
        )
    return packs


def _rescale(checkpoint: Mapping[str, Any], horizon: int, *, charge_cost: bool) -> dict[str, Any]:
    result = copy.deepcopy(checkpoint)
    value = result.get("value_of_additional_profiling")
    if value is not None:
        reduction = value["expected_reduction_in_decision_loss_per_execution"]
        cost = value["incremental_profiling_cost_usd"] if charge_cost else 0.0
        value["expected_executions"] = horizon
        value["gross_value_of_information_usd"] = horizon * reduction
        value["incremental_profiling_cost_usd"] = cost
        value["net_value_of_information_usd"] = horizon * reduction - cost
        confidence = value.get("monte_carlo_confidence") or {}
        lower_reduction = confidence.get("lower_reduction_per_execution")
        if lower_reduction is not None:
            confidence["lower_gross_value_usd"] = horizon * lower_reduction
            confidence["lower_net_value_usd"] = horizon * lower_reduction - cost
        value["continuation_value_usd"] = (
            value["net_value_of_information_usd"]
            if value.get("continuation_value_basis", "point_estimate") == "point_estimate"
            else confidence.get("lower_net_value_usd")
        )
    return result


def _route_joint(
    ctx: Mapping[str, Any],
    packs: Sequence[Mapping[str, Any]],
    *,
    horizon: int,
    policy: str = "joint",
    use_voi: bool = True,
    charge_profiling_economics: bool = True,
    profiling_budget_usd: float | None = None,
    maximum_profiling_decisions: int | None = None,
) -> dict[str, Any]:
    optimizer: OptimizerSettings = ctx["optimizer"]
    original = ctx["original"]
    first = packs[0]
    start = decide_adaptive_profiling_start(
        expected_executions=horizon,
        first_checkpoint_decisions=first["size"] * len(original),
        first_checkpoint_cost=ctx["average_decision_cost"] * first["size"] * len(original),
        settings=optimizer.economics.model_copy(update={"expected_executions": horizon}),
        savings_upper_bound_per_execution=ctx["structural_bound"],
        savings_bound_basis="structural_upper_bound",
        statistics_available=False,
        maximum_profiling_decisions=maximum_profiling_decisions,
        adaptive_budget_usd=profiling_budget_usd,
    )
    if start["action"] != "start_adaptive_profiling":
        return {
            "action": "SKIP_PROFILING",
            "selected_order": original,
            "profiling_reviews_consumed": 0,
            "profiling_decisions_consumed": 0,
            "profiling_cost_usd": 0.0,
            "reason": ";".join(start["reasons"]),
            "trajectory": [],
        }
    trajectory = []
    for pack in packs:
        checkpoint = _rescale(
            pack["policies"][policy], horizon, charge_cost=charge_profiling_economics
        )
        if not use_voi and checkpoint["risk_decision"] not in {
            "ADOPT_REORDER",
            "REUSE_EXISTING_PLAN",
        }:
            action, selected, reasons = (
                "KEEP_ORIGINAL",
                original,
                ["value_of_information_ablation_disabled"],
            )
        else:
            action, selected, reasons = choose_joint_action(
                checkpoint,
                original_order=original,
                profiling_cost_incurred_usd=pack["incurred_cost_usd"],
                profiling_budget_usd=profiling_budget_usd,
                maximum_profiling_decisions=maximum_profiling_decisions,
                minimum_voi_usd=optimizer.joint.minimum_voi_usd,
            )
        checkpoint.update(
            {
                "controller_action": action,
                "controller_selected_order": selected,
                "controller_reasons": reasons,
                "profiling_cost_incurred_usd": pack["incurred_cost_usd"],
                "controller_explanation": {
                    "action": action,
                    "reason_codes": reasons,
                    "minimum_net_voi_usd": optimizer.joint.minimum_voi_usd,
                    "estimated_net_voi_usd": (
                        (checkpoint.get("value_of_additional_profiling") or {}).get(
                            "net_value_of_information_usd"
                        )
                    ),
                    "continuation_value_basis": (
                        (checkpoint.get("value_of_additional_profiling") or {}).get(
                            "continuation_value_basis"
                        )
                    ),
                    "continuation_value_usd": (
                        (checkpoint.get("value_of_additional_profiling") or {}).get(
                            "continuation_value_usd"
                        )
                    ),
                    "profiling_budget_usd": profiling_budget_usd,
                    "maximum_profiling_decisions": maximum_profiling_decisions,
                },
            }
        )
        trajectory.append(checkpoint)
        if action != "PROFILE_MORE":
            return {
                "action": action,
                "selected_order": selected,
                "profiling_reviews_consumed": pack["size"],
                "profiling_decisions_consumed": pack["size"] * len(original),
                "profiling_cost_usd": pack["incurred_cost_usd"],
                "reason": ";".join(reasons),
                "trajectory": trajectory,
            }
    raise AssertionError("final checkpoint must terminate the joint controller")


def _route_existing_adaptive(
    ctx: Mapping[str, Any],
    packs: Sequence[Mapping[str, Any]],
    old: Mapping[str, Any],
    *,
    horizon: int,
    apply_risk_gate: bool,
) -> dict[str, Any]:
    optimizer: OptimizerSettings = ctx["optimizer"]
    original = ctx["original"]
    stop_index = len(packs) - 1
    for index, entry in enumerate(old["trajectory"]):
        final = entry["next_checkpoint_reviews"] is None
        stable = entry["bootstrap_selection_frequency_of_selected"] >= (
            optimizer.adaptive.stability_threshold
        )
        increment = entry["next_increment_cost_usd_estimate"]
        uneconomic = (
            increment is not None
            and horizon * entry["bootstrap_mean_regret_per_execution"] <= increment
        )
        if final or stable or uneconomic:
            stop_index = index
            break
    pack = packs[stop_index]
    selected = list(old["trajectory"][stop_index]["selected_order"])
    explanation = None
    if apply_risk_gate:
        selected, explanation = select_risk_aware_plan(
            None,
            original_order=original,
            objective=optimizer.planner.objective,
            evaluation_record_count=ctx["split"].evaluation_size,
            question_hashes=ctx["profile"].predicate_hashes,
            bootstrap=optimizer.bootstrap,
            correlation=optimizer.correlation,
            risk=optimizer.risk,
            economics=optimizer.economics.model_copy(update={"expected_executions": horizon}),
            input_price_per_million=ctx["input_rate"],
            output_price_per_million=ctx["output_rate"],
            tie_relative_tolerance=optimizer.planner.tie_relative_tolerance,
            primary_estimator="correlation_aware",
            statistics_state="available",
            primary_collected=pack["empirical_risk_collection"],
        )
    return {
        "action": (explanation or {}).get("decision") or "SELECT_ADAPTIVE_MINIMUM",
        "selected_order": selected,
        "profiling_reviews_consumed": pack["size"],
        "profiling_decisions_consumed": pack["size"] * len(original),
        "profiling_cost_usd": pack["incurred_cost_usd"],
        "reason": (explanation or {}).get("reason") or "existing_adaptive_stop",
        "explanation": explanation,
    }


def _score(ctx: Mapping[str, Any], result: Mapping[str, Any], horizon: int) -> dict[str, Any]:
    observed = observe_selection(
        ctx["observations"],
        result["selected_order"],
        predicted_saving=None,
        profiling_cost_usd=float(result["profiling_cost_usd"] or 0.0),
    )
    saving = observed.get("observed_saving_usd")
    observed["net_saving_usd"] = (
        None
        if saving is None
        else horizon * saving - float(result["profiling_cost_usd"] or 0.0)
    )
    observed["action"] = result["action"]
    observed["profiling_reviews_consumed"] = result["profiling_reviews_consumed"]
    observed["profiling_decisions_consumed"] = result["profiling_decisions_consumed"]
    observed["stopping_reason"] = result["reason"]
    return observed


def evaluate_joint_workload(
    spec: Mapping[str, str], optimizer: OptimizerSettings
) -> dict[str, Any]:
    ctx = _context(spec, optimizer)
    packs = _checkpoint_packs(ctx)
    old = simulate_adaptive_profiling(
        ctx["matrix"],
        estimator_name="correlation_aware",
        original_order=ctx["original"],
        objective=optimizer.planner.objective,
        evaluation_record_count=ctx["split"].evaluation_size,
        question_hashes=ctx["profile"].predicate_hashes,
        settings=optimizer.adaptive,
        bootstrap=optimizer.bootstrap,
        correlation=optimizer.correlation,
        input_price_per_million=ctx["input_rate"],
        output_price_per_million=ctx["output_rate"],
        expected_executions=10**9,
        tie_relative_tolerance=optimizer.planner.tie_relative_tolerance,
        estimated_cost_per_decision=ctx["average_decision_cost"],
    )
    horizons = {}
    for horizon in DEFAULT_HORIZONS:
        joint = _route_joint(ctx, packs, horizon=horizon)
        adaptive = _route_existing_adaptive(ctx, packs, old, horizon=horizon, apply_risk_gate=False)
        adaptive_risk = _route_existing_adaptive(
            ctx, packs, old, horizon=horizon, apply_risk_gate=True
        )
        horizons[str(horizon)] = {
            "joint": {**joint, "score": _score(ctx, joint, horizon)},
            "existing_adaptive": {
                **adaptive,
                "score": _score(ctx, adaptive, horizon),
            },
            "existing_adaptive_risk": {
                **adaptive_risk,
                "score": _score(ctx, adaptive_risk, horizon),
            },
        }
    ablation_routes = {
        "joint": _route_joint(ctx, packs, horizon=ABLATION_HORIZON),
        "independent_stopping_then_risk": _route_existing_adaptive(
            ctx, packs, old, horizon=ABLATION_HORIZON, apply_risk_gate=True
        ),
        "empirical_sparse_ablation": _route_joint(
            ctx, packs, horizon=ABLATION_HORIZON, policy="empirical_sparse_ablation"
        ),
        "voi_disabled": _route_joint(
            ctx, packs, horizon=ABLATION_HORIZON, use_voi=False
        ),
        "active_expected_regret": _route_joint(
            ctx, packs, horizon=ABLATION_HORIZON, policy="active_regret"
        ),
        "active_cvar": _route_joint(
            ctx, packs, horizon=ABLATION_HORIZON, policy="active_cvar"
        ),
        "robust_disagreement": _route_joint(
            ctx, packs, horizon=ABLATION_HORIZON, policy="robust_disagreement"
        ),
        "profiling_cost_ignored": _route_joint(
            ctx, packs, horizon=ABLATION_HORIZON, charge_profiling_economics=False
        ),
    }
    for cap in (75, 150, 300, 600):
        ablation_routes[f"decision_budget_{cap}"] = _route_joint(
            ctx,
            packs,
            horizon=ABLATION_HORIZON,
            maximum_profiling_decisions=cap,
        )
    ablations = {
        name: {**route, "score": _score(ctx, route, ABLATION_HORIZON)}
        for name, route in ablation_routes.items()
    }
    historical = ctx["historical_strategies"]["strategies"]
    static_orders = {
        "fixed": ctx["original"],
        "full_correlation_aware": historical["correlation_aware"]["selected_order"],
        "full_confidence_aware": historical["correlation_aware_confidence"]["selected_order"],
    }
    economic_scenarios = {}
    for horizon in DEFAULT_HORIZONS:
        rows = {}
        for name, order in static_orders.items():
            profile_cost = 0.0 if name == "fixed" else ctx["profile_cost"]
            observed = observe_selection(
                ctx["observations"],
                order,
                predicted_saving=None,
                profiling_cost_usd=profile_cost,
            )
            saving = observed.get("observed_saving_usd")
            rows[name] = {
                "selected_order": order,
                "profiling_cost_usd": profile_cost,
                "observed_saving_usd": saving,
                "cold_start_net_saving_usd": (
                    None if saving is None else horizon * saving - profile_cost
                ),
                "warm_start_net_saving_usd": None if saving is None else horizon * saving,
            }
        for name in ("existing_adaptive", "existing_adaptive_risk", "joint"):
            result = horizons[str(horizon)][name]
            score = result["score"]
            saving = score.get("observed_saving_usd")
            rows[name] = {
                "selected_order": result["selected_order"],
                "profiling_cost_usd": result["profiling_cost_usd"],
                "observed_saving_usd": saving,
                "cold_start_net_saving_usd": score["net_saving_usd"],
                "warm_start_net_saving_usd": None if saving is None else horizon * saving,
            }
        economic_scenarios[str(horizon)] = rows

    def comparison_row(strategy: str) -> dict[str, Any]:
        result = horizons["1"][strategy]
        score = result["score"]
        explanation = result.get("explanation") or {}
        selected_risk = explanation.get("selected_risk") or {}
        if strategy == "joint" and result.get("trajectory"):
            selected_risk = result["trajectory"][-1].get("selected_risk") or {}
        return {
            "profiling_reviews_consumed": result["profiling_reviews_consumed"],
            "profiling_calls": result["profiling_decisions_consumed"],
            "profiling_cost_usd": result["profiling_cost_usd"],
            "selected_order": result["selected_order"],
            "execution_cost_saving_usd": score.get("observed_saving_usd"),
            "execution_cost_saving_percent": score.get("observed_saving_percent"),
            "expected_regret_usd": (
                selected_risk.get("expected_regret_vs_modeled_minimum") or {}
            ).get("mean"),
            "observed_regret_usd": score.get("observed_regret_usd"),
            "net_saving_at_r_1_usd": score.get("net_saving_usd"),
            "net_saving_at_r_10_usd": horizons["10"][strategy]["score"].get(
                "net_saving_usd"
            ),
            "net_saving_at_r_100_usd": horizons["100"][strategy]["score"].get(
                "net_saving_usd"
            ),
            "stopping_reason": result["reason"],
        }
    return {
        "workload_id": ctx["id"],
        "original_order": ctx["original"],
        "profiling_fingerprint": ctx["split"].profiling_fingerprint,
        "evaluation_fingerprint": ctx["split"].evaluation_fingerprint,
        "controller_evidence_only": True,
        "held_out_outcomes_used_for_selection": False,
        "horizons": horizons,
        "ablations_at_r_10": ablations,
        "economic_scenarios": economic_scenarios,
        "primary_r_1_comparison": {
            "existing_adaptive_risk": comparison_row("existing_adaptive_risk"),
            "joint": comparison_row("joint"),
        },
        "checkpoint_evidence": [pack["policies"]["joint"] for pack in packs],
    }


def _aggregate(
    workloads: Sequence[Mapping[str, Any]], horizon: int, strategy: str
) -> dict[str, Any]:
    rows = [workload["horizons"][str(horizon)][strategy]["score"] for workload in workloads]
    measured = [row for row in rows if row["observed_result"] == "measured"]
    savings = [row["observed_saving_percent"] for row in measured]
    regrets = [row["observed_regret_usd"] for row in measured]
    return {
        "mean_execution_saving_percent": sum(savings) / len(savings),
        "median_execution_saving_percent": median(savings),
        "mean_observed_regret_usd": sum(regrets) / len(regrets),
        "maximum_observed_regret_usd": max(regrets),
        "beneficial_reorderings_accepted": sum(
            row["observed_saving_usd"] > 0 and row["action"] == "ADOPT_REORDER"
            for row in measured
        ),
        "beneficial_reorderings_rejected": sum(
            row["observed_regret_usd"] > 0 and row["action"] != "ADOPT_REORDER"
            for row in measured
        ),
        "harmful_reorderings_accepted": sum(
            row["observed_saving_usd"] < 0 and row["action"] == "ADOPT_REORDER"
            for row in measured
        ),
        "average_profiling_decisions_consumed": sum(
            row["profiling_decisions_consumed"] for row in rows
        )
        / len(rows),
        "total_profiling_cost_usd": sum(row["profiling_cost_usd"] for row in rows),
        "total_net_saving_usd": sum(row["net_saving_usd"] for row in measured),
        "leave_one_workload_out_mean_execution_saving_percent": {
            workload["workload_id"]: sum(
                other["horizons"][str(horizon)][strategy]["score"][
                    "observed_saving_percent"
                ]
                for other in workloads
                if other["workload_id"] != workload["workload_id"]
            )
            / (len(workloads) - 1)
            for workload in workloads
        },
    }


def run_joint_benchmark(
    optimizer: OptimizerSettings,
    output_dir: Path,
    *,
    workloads: Sequence[Mapping[str, str]] = DEFAULT_WORKLOADS,
    config_path: str | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    evaluated = [evaluate_joint_workload(spec, optimizer) for spec in workloads]
    aggregates = {
        str(horizon): {
            strategy: _aggregate(evaluated, horizon, strategy)
            for strategy in ("existing_adaptive", "existing_adaptive_risk", "joint")
        }
        for horizon in DEFAULT_HORIZONS
    }
    economic_totals = {
        str(horizon): {
            strategy: {
                scenario: sum(
                    workload["economic_scenarios"][str(horizon)][strategy][scenario]
                    for workload in evaluated
                    if workload["economic_scenarios"][str(horizon)][strategy][scenario]
                    is not None
                )
                for scenario in (
                    "cold_start_net_saving_usd",
                    "warm_start_net_saving_usd",
                )
            }
            for strategy in (
                "fixed",
                "full_correlation_aware",
                "full_confidence_aware",
                "existing_adaptive",
                "existing_adaptive_risk",
                "joint",
            )
        }
        for horizon in DEFAULT_HORIZONS
    }
    report = {
        "schema_version": JOINT_BENCHMARK_SCHEMA_VERSION,
        "controller_schema_version": JOINT_SCHEMA_VERSION,
        "created_at": _now(),
        "git_commit": git_commit(),
        "git_worktree_dirty": git_worktree_dirty(),
        "config_path": config_path,
        "label": "exploratory_retrospective_joint_sequential_replay",
        "not_a_confirmatory_test": True,
        "live_jev_calls": 0,
        "paid_api_calls": 0,
        "held_out_outcomes_used_for_selection": False,
        "horizons": list(DEFAULT_HORIZONS),
        "ablation_horizon": ABLATION_HORIZON,
        "predetermined_active_regret_cap_usd": ACTIVE_EXPECTED_REGRET_CAP_USD,
        "predetermined_active_cvar_loss_cap_usd": ACTIVE_CVAR_LOSS_CAP_USD,
        "workloads": evaluated,
        "aggregates": aggregates,
        "economic_totals_by_horizon": economic_totals,
        "uncertainty_scope": {
            "predicate_selectivity": "smoothed posterior-predictive outcome patterns",
            "predicate_correlation": "joint pattern distribution with finite-sample uncertainty",
            "execution_cost": "candidate cost vectors refit on every predictive draw",
            "jev_response_variability": (
                "not included in the predictive distribution; stored repeats are sensitivity only"
            ),
        },
        "caveat": (
            "four related workloads previously used for development; results are exploratory "
            "and do not establish statistical significance or the globally cheapest permutation"
        ),
    }
    ResultStore.write_json(output_dir / "benchmark.json", report)
    for workload in evaluated:
        ResultStore.write_json(output_dir / f"{workload['workload_id']}.json", workload)
    write_joint_summary(report, output_dir / "summary.md")
    return report


def write_joint_summary(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Joint adaptive-risk offline benchmark",
        "",
        "Strict sequential replay over stored profiling matrices; no Jev calls.",
        "",
        "| R | Existing adaptive+risk net | Joint net | Joint profile cost | Joint mean saving |",
        "|---:|---:|---:|---:|---:|",
    ]
    for horizon in report["horizons"]:
        old = report["aggregates"][str(horizon)]["existing_adaptive_risk"]
        joint = report["aggregates"][str(horizon)]["joint"]
        lines.append(
            f"| {horizon} | ${old['total_net_saving_usd']:.6f} | "
            f"${joint['total_net_saving_usd']:.6f} | "
            f"${joint['total_profiling_cost_usd']:.6f} | "
            f"{joint['mean_execution_saving_percent']:.2f}% |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
