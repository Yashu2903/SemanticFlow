"""Offline calibration diagnostics for the Milestone 4 joint VOI controller.

This module never constructs a provider.  Stored profiling rows are revealed only through the
same nested checkpoint packs used by the joint benchmark; held-out live costs are joined after
each checkpoint decision solely to label retrospective, observable decision changes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any

from semanticflow.optimizer.adaptive import sampling_order
from semanticflow.optimizer.joint import analyze_joint_checkpoint, collect_checkpoint_uncertainty
from semanticflow.optimizer.joint_benchmark import (
    DEFAULT_HORIZONS,
    _context,
    _profile_cost,
    _route_joint,
)
from semanticflow.optimizer.models import JointAdaptiveSettings, OptimizerSettings, RiskSettings
from semanticflow.optimizer.risk_benchmark import DEFAULT_WORKLOADS, observe_selection
from semanticflow.storage.results import ResultStore, git_commit, git_worktree_dirty

VOI_VALIDATION_SCHEMA_VERSION = "m4-voi-validation-v1"
CONTRACTION_EXPONENTS = (0.25, 0.375, 0.5, 0.625, 0.75)
PRIOR_STRENGTHS = (0.25, 0.5, 1.0, 2.0, 5.0)
VOI_SCENARIO_COUNTS = (8, 16, 32, 64, 128)
MINIMUM_VOI_THRESHOLDS = (0.0, 0.0001, 0.00025, 0.0005, 0.001)
SAVING_PROBABILITIES = (0.9, 0.95, 0.99)
DECISION_BUDGETS = (75, 150, 300, 450, 600)
CHECKPOINT_SCHEDULES = (
    (25, 50, 100, 150, 200),
    (25, 50, 100),
    (25, 100, 200),
    (50, 100, 200),
)
_USE_PACK_NEXT = object()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _error(predicted: float | None, realized: float | None) -> dict[str, float | None]:
    if predicted is None or realized is None:
        return {"absolute": None, "relative_to_realized": None}
    absolute = predicted - realized
    return {
        "absolute": absolute,
        "relative_to_realized": (
            None if realized == 0.0 else absolute / abs(realized)
        ),
    }


def _metric_transition(
    *,
    current: float | None,
    predicted_next: float | None,
    realized_next: float | None,
) -> dict[str, Any]:
    predicted_reduction = (
        None if current is None or predicted_next is None else current - predicted_next
    )
    realized_reduction = (
        None if current is None or realized_next is None else current - realized_next
    )
    return {
        "current": current,
        "predicted_next": predicted_next,
        "realized_next": realized_next,
        "predicted_reduction": predicted_reduction,
        "realized_reduction": realized_reduction,
        "next_value_error": _error(predicted_next, realized_next),
        "reduction_error": _error(predicted_reduction, realized_reduction),
    }


def _uncertainty_transition(
    current: Mapping[str, Any], following: Mapping[str, Any]
) -> dict[str, Any]:
    current_stats = current["statistical_uncertainty"]
    next_stats = following["statistical_uncertainty"]
    projection = current_stats["root_n_projection_to_next_checkpoint"]
    contraction = projection["contraction"]
    marginals = {
        key: _metric_transition(
            current=value["posterior_std"],
            predicted_next=value["posterior_std"] * contraction,
            realized_next=next_stats["marginal_selectivity"][key]["posterior_std"],
        )
        for key, value in current_stats["marginal_selectivity"].items()
    }
    conditionals = {
        key: {
            **_metric_transition(
                current=value["posterior_std"],
                predicted_next=value["posterior_std"] * contraction,
                realized_next=next_stats["conditional_selectivity"][key]["posterior_std"],
            ),
            "current_support": value["trials"],
            "realized_next_support": next_stats["conditional_selectivity"][key]["trials"],
        }
        for key, value in current_stats["conditional_selectivity"].items()
    }
    costs = {
        key: _metric_transition(
            current=value,
            predicted_next=value * contraction,
            realized_next=next_stats["candidate_cost_std_by_order"][key],
        )
        for key, value in current_stats["candidate_cost_std_by_order"].items()
    }
    value = current.get("value_of_additional_profiling") or {}
    ranking = _metric_transition(
        current=current_stats["plan_ranking_uncertainty"],
        predicted_next=value.get("projected_plan_ranking_uncertainty"),
        realized_next=next_stats["plan_ranking_uncertainty"],
    )
    regret = _metric_transition(
        current=current_stats["selected_plan_expected_regret"]["std"],
        predicted_next=value.get("projected_selected_plan_regret_std"),
        realized_next=next_stats["selected_plan_expected_regret"]["std"],
    )
    return {
        "contraction": contraction,
        "marginal_selectivity": marginals,
        "conditional_selectivity": conditionals,
        "cost_estimate": costs,
        "plan_ranking": ranking,
        "expected_regret": regret,
    }


def _observed_cost(ctx: Mapping[str, Any], order: Sequence[str]) -> float | None:
    result = observe_selection(
        ctx["observations"], order, predicted_saving=None, profiling_cost_usd=0.0
    )
    if result["observed_result"] != "measured":
        return None
    return float(result["observed_cost_usd"])


def _transition_diagnostics(
    ctx: Mapping[str, Any], packs: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    trace = [pack["policies"]["joint"] for pack in packs]
    rows = []
    for index, (current, following) in enumerate(zip(trace, trace[1:], strict=False)):
        value = current.get("value_of_additional_profiling") or {}
        current_order = current["risk_selected_order"]
        next_order = following["risk_selected_order"]
        current_cost = _observed_cost(ctx, current_order)
        next_cost = _observed_cost(ctx, next_order)
        realized_benefit = (
            None if current_cost is None or next_cost is None else current_cost - next_cost
        )
        actual_increment = packs[index + 1]["incurred_cost_usd"] - packs[index][
            "incurred_cost_usd"
        ]
        rows.append(
            {
                "from_reviews": current["profiled_reviews"],
                "to_reviews": following["profiled_reviews"],
                "predicted_incremental_profiling_cost_usd": value.get(
                    "incremental_profiling_cost_usd"
                ),
                "realized_stored_incremental_profiling_cost_usd": actual_increment,
                "profiling_cost_prediction_error_usd": (
                    None
                    if value.get("incremental_profiling_cost_usd") is None
                    else value["incremental_profiling_cost_usd"] - actual_increment
                ),
                "current_minimum_predicted_plan": current["minimum_objective_order"],
                "next_minimum_predicted_plan": following["minimum_objective_order"],
                "minimum_predicted_plan_changed": current["minimum_objective_order"]
                != following["minimum_objective_order"],
                "current_risk_selected_plan": current_order,
                "next_risk_selected_plan": next_order,
                "risk_selected_plan_changed": current_order != next_order,
                "current_risk_decision": current["risk_decision"],
                "next_risk_decision": following["risk_decision"],
                "reorder_decision_changed": current["risk_decision"]
                != following["risk_decision"],
                "predicted_probability_plan_changes": value.get(
                    "probability_next_checkpoint_changes_selected_plan"
                ),
                "predicted_decision_benefit_per_execution_usd": value.get(
                    "expected_reduction_in_decision_loss_per_execution"
                ),
                "realized_observed_decision_benefit_per_execution_usd": realized_benefit,
                "decision_benefit_prediction_error_usd": (
                    None
                    if realized_benefit is None
                    or value.get("expected_reduction_in_decision_loss_per_execution") is None
                    else value["expected_reduction_in_decision_loss_per_execution"]
                    - realized_benefit
                ),
                "observability": (
                    "live_executed_orders"
                    if realized_benefit is not None
                    else "unavailable_counterfactual"
                ),
                "uncertainty": _uncertainty_transition(current, following),
            }
        )
    return rows


def _analyze_pack(
    ctx: Mapping[str, Any],
    pack: Mapping[str, Any],
    *,
    joint: JointAdaptiveSettings,
    risk: RiskSettings | None = None,
    collection: Mapping[str, Any] | None = None,
    next_sample_size: int | None | object = _USE_PACK_NEXT,
) -> dict[str, Any]:
    optimizer: OptimizerSettings = ctx["optimizer"]
    next_size = (
        pack["next_size"] if next_sample_size is _USE_PACK_NEXT else next_sample_size
    )
    assert next_size is None or isinstance(next_size, int)
    increment = (
        None
        if next_size is None
        else pack["incurred_cost_usd"] / pack["size"] * (next_size - pack["size"])
    )
    return analyze_joint_checkpoint(
        pack["evidence"],
        next_sample_size=next_size,
        incremental_profiling_cost_usd=increment,
        expected_executions=1,
        estimator_name="correlation_aware",
        original_order=ctx["original"],
        objective=optimizer.planner.objective,
        evaluation_record_count=ctx["split"].evaluation_size,
        question_hashes=ctx["profile"].predicate_hashes,
        bootstrap=optimizer.bootstrap,
        correlation=optimizer.correlation,
        risk=risk or optimizer.risk,
        economics=optimizer.economics.model_copy(update={"expected_executions": 1}),
        joint=joint,
        input_price_per_million=ctx["input_rate"],
        output_price_per_million=ctx["output_rate"],
        tie_relative_tolerance=optimizer.planner.tie_relative_tolerance,
        seed=joint.seed + pack["size"],
        checkpoint_collected=collection or pack["posterior_collection"],
    )


def _packs_with_policy(
    packs: Sequence[Mapping[str, Any]], analyses: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {**pack, "policies": {"joint": analysis}}
        for pack, analysis in zip(packs, analyses, strict=True)
    ]


def _validation_checkpoint_packs(ctx: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compute only the joint posterior path needed for VOI validation."""
    optimizer: OptimizerSettings = ctx["optimizer"]
    matrix = ctx["matrix"]
    sizes = [size for size in optimizer.adaptive.checkpoints if size <= matrix.size]
    ranked = sampling_order([row.review_id for row in matrix.rows], optimizer.adaptive.seed)
    packs = []
    for index, size in enumerate(sizes):
        evidence = matrix.select_reviews(ranked[:size])
        next_size = sizes[index + 1] if index + 1 < len(sizes) else None
        incurred = _profile_cost(evidence, ctx["input_rate"], ctx["output_rate"])
        collection = collect_checkpoint_uncertainty(
            evidence,
            estimator_name="correlation_aware",
            original_order=ctx["original"],
            objective=optimizer.planner.objective,
            evaluation_record_count=ctx["split"].evaluation_size,
            question_hashes=ctx["profile"].predicate_hashes,
            bootstrap=optimizer.bootstrap,
            correlation=optimizer.correlation,
            joint=optimizer.joint,
            input_price_per_million=ctx["input_rate"],
            output_price_per_million=ctx["output_rate"],
            tie_relative_tolerance=optimizer.planner.tie_relative_tolerance,
            seed=optimizer.joint.seed + size,
        )
        pack = {
            "size": size,
            "next_size": next_size,
            "evidence": evidence,
            "posterior_collection": collection,
            "incurred_cost_usd": incurred,
        }
        analysis = _analyze_pack(
            ctx, pack, joint=optimizer.joint, collection=collection
        )
        pack["policies"] = {"joint": analysis}
        packs.append(pack)
    return packs


def _route_summary(route: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "action": route["action"],
        "selected_order": route["selected_order"],
        "profiling_reviews_consumed": route["profiling_reviews_consumed"],
        "profiling_decisions_consumed": route["profiling_decisions_consumed"],
        "profiling_cost_usd": route["profiling_cost_usd"],
        "reason": route["reason"],
        "checkpoints_visited": [item["profiled_reviews"] for item in route["trajectory"]],
    }


def _family_sensitivity(
    ctx: Mapping[str, Any], packs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    optimizer: OptimizerSettings = ctx["optimizer"]
    result: dict[str, Any] = {}

    contraction = {}
    for exponent in CONTRACTION_EXPONENTS:
        settings = optimizer.joint.model_copy(
            update={"uncertainty_contraction_exponent": exponent}
        )
        analyses = [_analyze_pack(ctx, pack, joint=settings) for pack in packs]
        route = _route_joint(
            ctx, _packs_with_policy(packs, analyses), horizon=10, policy="joint"
        )
        contraction[str(exponent)] = _route_summary(route)
    result["uncertainty_contraction_exponent"] = contraction

    scenarios = {}
    for count in VOI_SCENARIO_COUNTS:
        started = perf_counter()
        settings = optimizer.joint.model_copy(update={"voi_scenarios": count})
        analyses = [_analyze_pack(ctx, pack, joint=settings) for pack in packs]
        elapsed = perf_counter() - started
        route = _route_joint(
            ctx, _packs_with_policy(packs, analyses), horizon=10, policy="joint"
        )
        scenarios[str(count)] = {
            **_route_summary(route),
            "analysis_runtime_seconds": elapsed,
        }
    result["voi_scenarios"] = scenarios

    thresholds = {}
    for threshold in MINIMUM_VOI_THRESHOLDS:
        adjusted_optimizer = optimizer.model_copy(
            update={
                "joint": optimizer.joint.model_copy(update={"minimum_voi_usd": threshold})
            }
        )
        adjusted_ctx = {**ctx, "optimizer": adjusted_optimizer}
        route = _route_joint(adjusted_ctx, packs, horizon=10, policy="joint")
        thresholds[str(threshold)] = _route_summary(route)
    result["minimum_net_voi_usd"] = thresholds

    saving_probability = {}
    for probability in SAVING_PROBABILITIES:
        risk = optimizer.risk.model_copy(update={"minimum_saving_probability": probability})
        analyses = [
            _analyze_pack(ctx, pack, joint=optimizer.joint, risk=risk) for pack in packs
        ]
        route = _route_joint(
            ctx, _packs_with_policy(packs, analyses), horizon=10, policy="joint"
        )
        saving_probability[str(probability)] = _route_summary(route)
    result["minimum_saving_probability"] = saving_probability

    risk_constraints = {}
    for name, updates in {
        "disabled": {"maximum_expected_regret": None, "maximum_cvar_loss": None},
        "expected_regret_0.0001": {"maximum_expected_regret": 0.0001},
        "cvar_0": {"maximum_cvar_loss": 0.0},
    }.items():
        risk = optimizer.risk.model_copy(update=updates)
        analyses = [
            _analyze_pack(ctx, pack, joint=optimizer.joint, risk=risk) for pack in packs
        ]
        route = _route_joint(
            ctx, _packs_with_policy(packs, analyses), horizon=10, policy="joint"
        )
        risk_constraints[name] = _route_summary(route)
    result["risk_constraints"] = risk_constraints

    result["decision_budget"] = {
        str(budget): _route_summary(
            _route_joint(ctx, packs, horizon=10, maximum_profiling_decisions=budget)
        )
        for budget in DECISION_BUDGETS
    }

    schedules = {}
    by_size = {pack["size"]: pack for pack in packs}
    for schedule in CHECKPOINT_SCHEDULES:
        selected_packs = [by_size[size] for size in schedule if size in by_size]
        analyses = []
        for index, pack in enumerate(selected_packs):
            following = (
                selected_packs[index + 1]["size"]
                if index + 1 < len(selected_packs)
                else None
            )
            analyses.append(
                _analyze_pack(
                    ctx, pack, joint=optimizer.joint, next_sample_size=following
                )
            )
        route = _route_joint(
            ctx,
            _packs_with_policy(selected_packs, analyses),
            horizon=10,
            policy="joint",
        )
        schedules[",".join(map(str, schedule))] = _route_summary(route)
    result["checkpoint_schedule"] = schedules
    return result


def _prior_sensitivity(
    ctx: Mapping[str, Any], packs: Sequence[Mapping[str, Any]], *, iterations: int = 500
) -> dict[str, Any]:
    optimizer: OptimizerSettings = ctx["optimizer"]
    output = {}
    for strength in PRIOR_STRENGTHS:
        started = perf_counter()
        settings = optimizer.joint.model_copy(
            update={
                "predictive_prior_strength": strength,
                "bootstrap_iterations": iterations,
            }
        )
        analyses = []
        for pack in packs:
            collection = collect_checkpoint_uncertainty(
                pack["evidence"],
                estimator_name="correlation_aware",
                original_order=ctx["original"],
                objective=optimizer.planner.objective,
                evaluation_record_count=ctx["split"].evaluation_size,
                question_hashes=ctx["profile"].predicate_hashes,
                bootstrap=optimizer.bootstrap,
                correlation=optimizer.correlation,
                joint=settings,
                input_price_per_million=ctx["input_rate"],
                output_price_per_million=ctx["output_rate"],
                tie_relative_tolerance=optimizer.planner.tie_relative_tolerance,
                seed=settings.seed + pack["size"],
            )
            analyses.append(
                _analyze_pack(
                    ctx, pack, joint=settings, collection=collection
                )
            )
        route = _route_joint(
            ctx, _packs_with_policy(packs, analyses), horizon=10, policy="joint"
        )
        output[str(strength)] = {
            **_route_summary(route),
            "bootstrap_iterations": iterations,
            "analysis_runtime_seconds": perf_counter() - started,
        }
    return output


def _reuse_horizons(ctx: Mapping[str, Any], packs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = {}
    for horizon in DEFAULT_HORIZONS:
        route = _route_joint(ctx, packs, horizon=horizon)
        score = observe_selection(
            ctx["observations"],
            route["selected_order"],
            predicted_saving=None,
            profiling_cost_usd=route["profiling_cost_usd"],
        )
        saving = score.get("observed_saving_usd")
        rows[str(horizon)] = {
            **_route_summary(route),
            "initial_profiling_decision": "START_PROFILING",
            "execution_cost_per_run_usd": score.get("observed_cost_usd"),
            "gross_execution_saving_usd": (
                None if saving is None else horizon * saving
            ),
            "cold_start_net_saving_usd": (
                None if saving is None else horizon * saving - route["profiling_cost_usd"]
            ),
            "warm_start_net_saving_usd": None if saving is None else horizon * saving,
        }
    return rows


def _aggregate_error(rows: Sequence[Mapping[str, Any]], path: Sequence[str]) -> dict[str, Any]:
    values = []
    for row in rows:
        node: Any = row
        for key in path:
            node = node[key]
        if node is not None:
            values.append(abs(float(node)))
    return {
        "count": len(values),
        "mean_absolute_error": None if not values else mean(values),
        "median_absolute_error": None if not values else median(values),
        "maximum_absolute_error": None if not values else max(values),
    }


def _calibration_summary(workloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    transitions = [row for workload in workloads for row in workload["transitions"]]
    benefit_errors = [
        abs(row["decision_benefit_prediction_error_usd"])
        for row in transitions
        if row["decision_benefit_prediction_error_usd"] is not None
    ]
    return {
        "observable_transition_count": len(benefit_errors),
        "decision_benefit_prediction_error_usd": {
            "mean_absolute_error": mean(benefit_errors) if benefit_errors else None,
            "median_absolute_error": median(benefit_errors) if benefit_errors else None,
            "maximum_absolute_error": max(benefit_errors) if benefit_errors else None,
        },
        "plan_ranking_next_uncertainty_error": _aggregate_error(
            transitions, ("uncertainty", "plan_ranking", "next_value_error", "absolute")
        ),
        "expected_regret_next_std_error": _aggregate_error(
            transitions, ("uncertainty", "expected_regret", "next_value_error", "absolute")
        ),
        "profiling_cost_prediction_error_usd": _aggregate_error(
            transitions, ("profiling_cost_prediction_error_usd",)
        ),
    }


def _canonical_behavior_matches(
    canonical: Mapping[str, Any], workloads: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = {item["workload_id"]: item for item in canonical["workloads"]}
    mismatches = []
    for workload in workloads:
        prior = expected[workload["workload_id"]]
        for horizon, actual in workload["reuse_horizons"].items():
            old = prior["horizons"][horizon]["joint"]
            for field in (
                "action",
                "selected_order",
                "profiling_reviews_consumed",
                "profiling_decisions_consumed",
                "profiling_cost_usd",
            ):
                if actual[field] != old[field]:
                    mismatches.append(
                        {
                            "workload_id": workload["workload_id"],
                            "horizon": int(horizon),
                            "field": field,
                            "canonical": old[field],
                            "reproduced": actual[field],
                        }
                    )
    return {"matches": not mismatches, "mismatches": mismatches}


def run_voi_validation(
    optimizer: OptimizerSettings,
    output_dir: Path,
    *,
    canonical_dir: Path,
    workloads: Sequence[Mapping[str, str]] = DEFAULT_WORKLOADS,
    config_path: str | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    canonical_path = canonical_dir / "benchmark.json"
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    evaluated = []
    for spec in workloads:
        ctx = _context(spec, optimizer)
        packs = _validation_checkpoint_packs(ctx)
        evaluated.append(
            {
                "workload_id": ctx["id"],
                "checkpoint_trace": [pack["policies"]["joint"] for pack in packs],
                "transitions": _transition_diagnostics(ctx, packs),
                "reuse_horizons": _reuse_horizons(ctx, packs),
                "sensitivity": _family_sensitivity(ctx, packs),
                "prior_strength_sensitivity": _prior_sensitivity(ctx, packs),
            }
        )
    canonical_hashes = {
        path.name: _sha256(path) for path in sorted(canonical_dir.iterdir()) if path.is_file()
    }
    report = {
        "schema_version": VOI_VALIDATION_SCHEMA_VERSION,
        "created_at": _now(),
        "git_commit": git_commit(),
        "git_worktree_dirty": git_worktree_dirty(),
        "config_path": config_path,
        "canonical_directory": str(canonical_dir),
        "canonical_artifact_sha256": canonical_hashes,
        "canonical_behavior_reproduced": _canonical_behavior_matches(canonical, evaluated),
        "live_jev_calls": 0,
        "paid_api_calls": 0,
        "synthetic_draws_presented_as_jev_evidence": False,
        "controller_uses_future_rows": False,
        "workloads": evaluated,
        "calibration_summary": _calibration_summary(evaluated),
        "methodology": {
            "base_bootstrap_iterations": optimizer.joint.bootstrap_iterations,
            "prior_sensitivity_bootstrap_iterations": 500,
            "sensitivity_policy": "one parameter family at a time; routes reported at R=10",
            "observed_decision_quality": (
                "live costs only when both selected orders were historically executed"
            ),
            "counterfactual_limitation": (
                "unexecuted plan costs remain unavailable and are never replaced with estimates"
            ),
        },
    }
    ResultStore.write_json(output_dir / "validation.json", report)
    for workload in evaluated:
        ResultStore.write_json(output_dir / f"{workload['workload_id']}.json", workload)
    return report


def write_validation_comparison(
    *,
    canonical_benchmark_path: Path,
    corrected_benchmark_path: Path,
    current_validation_path: Path,
    corrected_validation_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Join completed offline artifacts without rerunning selection or reading provider data."""
    comparison_path = output_dir / "comparison.json"
    summary_path = output_dir / "summary.md"
    if comparison_path.exists() or summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite comparison artifacts in {output_dir}")
    canonical = json.loads(canonical_benchmark_path.read_text(encoding="utf-8"))
    corrected = json.loads(corrected_benchmark_path.read_text(encoding="utf-8"))
    current_validation = json.loads(current_validation_path.read_text(encoding="utf-8"))
    corrected_validation = json.loads(corrected_validation_path.read_text(encoding="utf-8"))
    old_by_workload = {item["workload_id"]: item for item in canonical["workloads"]}
    new_by_workload = {item["workload_id"]: item for item in corrected["workloads"]}
    workload_rows = {}
    for workload_id, old in old_by_workload.items():
        new = new_by_workload[workload_id]
        workload_rows[workload_id] = {}
        for horizon in corrected["horizons"]:
            key = str(horizon)
            point = old["horizons"][key]["joint"]
            lower = new["horizons"][key]["joint"]
            workload_rows[workload_id][key] = {
                "point_estimate": {
                    "reviews": point["profiling_reviews_consumed"],
                    "action": point["action"],
                    "selected_order": point["selected_order"],
                    "profiling_cost_usd": point["profiling_cost_usd"],
                    "observed_saving_usd": point["score"]["observed_saving_usd"],
                    "observed_regret_usd": point["score"]["observed_regret_usd"],
                    "cold_start_net_saving_usd": point["score"]["net_saving_usd"],
                },
                "lower_confidence_bound": {
                    "reviews": lower["profiling_reviews_consumed"],
                    "action": lower["action"],
                    "selected_order": lower["selected_order"],
                    "profiling_cost_usd": lower["profiling_cost_usd"],
                    "observed_saving_usd": lower["score"]["observed_saving_usd"],
                    "observed_regret_usd": lower["score"]["observed_regret_usd"],
                    "cold_start_net_saving_usd": lower["score"]["net_saving_usd"],
                },
            }
    aggregate_rows = {}
    for horizon in corrected["horizons"]:
        key = str(horizon)
        old_aggregate = canonical["aggregates"][key]
        new_aggregate = corrected["aggregates"][key]
        economics = corrected["economic_totals_by_horizon"][key]
        aggregate_rows[key] = {
            "fixed": economics["fixed"],
            "full_correlation_aware": economics["full_correlation_aware"],
            "full_confidence_aware": economics["full_confidence_aware"],
            "existing_adaptive": economics["existing_adaptive"],
            "existing_adaptive_risk": old_aggregate["existing_adaptive_risk"],
            "point_estimate_joint": old_aggregate["joint"],
            "lower_confidence_bound_joint": new_aggregate["joint"],
            "corrected_minus_point_net_saving_usd": (
                new_aggregate["joint"]["total_net_saving_usd"]
                - old_aggregate["joint"]["total_net_saving_usd"]
            ),
        }
    canonical_hashes_now = {
        path.name: _sha256(path)
        for path in sorted(canonical_benchmark_path.parent.iterdir())
        if path.is_file()
    }
    recorded_hashes = current_validation["canonical_artifact_sha256"]
    report = {
        "schema_version": "m4-voi-validation-comparison-v1",
        "created_at": _now(),
        "live_jev_calls": 0,
        "paid_api_calls": 0,
        "canonical_behavior_reproduced_before_correction": current_validation[
            "canonical_behavior_reproduced"
        ],
        "canonical_artifact_integrity": {
            "unchanged": canonical_hashes_now == recorded_hashes,
            "recorded_before": recorded_hashes,
            "verified_after": canonical_hashes_now,
        },
        "aggregate_by_horizon": aggregate_rows,
        "per_workload_by_horizon": workload_rows,
        "calibration_summary": current_validation["calibration_summary"],
        "corrected_sensitivity": {
            item["workload_id"]: {
                "sensitivity": item["sensitivity"],
                "prior_strength_sensitivity": item["prior_strength_sensitivity"],
            }
            for item in corrected_validation["workloads"]
        },
        "evidence_scope": {
            "selection": "profiling prefixes and statistical simulations only",
            "evaluation": "historical live costs for executed orders only",
            "unexecuted_counterfactuals": "unavailable",
        },
    }
    ResultStore.write_json(comparison_path, report)
    lines = [
        "# VOI validation comparison",
        "",
        "Offline stored-data evaluation; zero Jev calls.",
        "",
        "| R | Adaptive+risk | Point joint | Corrected joint | Corrected - point |",
        "|---:|---:|---:|---:|---:|",
    ]
    for horizon in corrected["horizons"]:
        row = aggregate_rows[str(horizon)]
        lines.append(
            f"| {horizon} | "
            f"${row['existing_adaptive_risk']['total_net_saving_usd']:.6f} | "
            f"${row['point_estimate_joint']['total_net_saving_usd']:.6f} | "
            f"${row['lower_confidence_bound_joint']['total_net_saving_usd']:.6f} | "
            f"${row['corrected_minus_point_net_saving_usd']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "Canonical artifacts unchanged: "
            + str(report["canonical_artifact_integrity"]["unchanged"]),
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
