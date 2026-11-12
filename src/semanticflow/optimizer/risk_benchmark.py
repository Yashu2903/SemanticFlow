"""Offline Milestone 4 benchmark: risk-aware selection over stored Milestone 3 evidence.

Plan selection uses the 200-review profiling split only. Held-out outcomes are used for
retrospective scoring and never for fitting thresholds. Observed costs are taken only from
executed live orders; unexecuted permutations are marked unavailable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from semanticflow.optimizer.adaptive import sampling_order, simulate_adaptive_profiling
from semanticflow.optimizer.bootstrap import collect_paired_bootstrap, order_key
from semanticflow.optimizer.models import OptimizerSettings, ProfilingResult, SplitManifest
from semanticflow.optimizer.planner import CostBasedPlanner
from semanticflow.optimizer.risk import select_risk_aware_plan
from semanticflow.optimizer.uncertainty import response_variability_sensitivity
from semanticflow.optimizer.workflow import (
    _load_json,
    _profiling_report_path,
    load_profile_matrix,
)
from semanticflow.query.models import load_query
from semanticflow.storage.results import ResultStore, git_commit, git_worktree_dirty

BENCHMARK_SCHEMA_VERSION = "m4-offline-benchmark-v1"
DEFAULT_HORIZONS = (1, 10, 100, 1000)
COST_MODEL = {
    "independent": "independent-and-v1",
    "correlation_aware": "correlation-aware-and-v1",
}

DEFAULT_WORKLOADS: tuple[dict[str, str], ...] = (
    {
        "id": "product_quality",
        "query": "configs/queries/product_quality.yaml",
        "profile_dir": "results/milestone_3/jev_product_quality_benchmark",
        "analysis": "results/milestone_3/jev_product_quality_benchmark/analysis.json",
        "historical_strategies": (
            "results/milestone_3/jev_product_quality_benchmark/strategies.json"
        ),
    },
    {
        "id": "listing_mismatch",
        "query": "configs/queries/listing_mismatch.yaml",
        "profile_dir": "results/milestone_3/jev_listing_mismatch/profile",
        "analysis": "results/milestone_3/jev_listing_mismatch/analysis.json",
        "historical_strategies": "results/milestone_3/jev_listing_mismatch/strategies.json",
    },
    {
        "id": "positive_experience",
        "query": "configs/queries/positive_experience.yaml",
        "profile_dir": "results/milestone_3/jev_positive_experience/profile",
        "analysis": "results/milestone_3/jev_positive_experience/analysis.json",
        "historical_strategies": "results/milestone_3/jev_positive_experience/strategies.json",
    },
    {
        "id": "mixed_feedback",
        "query": "configs/queries/mixed_feedback.yaml",
        "profile_dir": "results/milestone_3/jev_mixed_feedback/profile",
        "analysis": "results/milestone_3/jev_mixed_feedback/analysis.json",
        "historical_strategies": "results/milestone_3/jev_mixed_feedback/strategies.json",
    },
)

MAJOR_STRATEGIES = (
    "fixed",
    "independent",
    "correlation_aware",
    "correlation_aware_confidence",
    "risk_aware",
)
ABLATIONS = (
    "independence_ungated",
    "correlation_ungated",
    "correlation_confidence",
    "risk_aware_correlation",
    "risk_aware_robust",
    "risk_aware_economics",
    "risk_aware_adaptive",
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def load_observations(analysis: Mapping[str, Any]) -> dict[str, Any]:
    orders = analysis["orders"]
    return {
        "fixed_order": analysis["fixed_order"],
        "observed_cheapest_order": analysis["observed_cheapest_order"],
        "executed_orders": sorted(orders),
        "mean_cost_usd": {
            key: float(payload["mean_cost_usd"]) for key, payload in orders.items()
        },
        "all_orders_executed": bool(analysis.get("all_orders_executed")),
        "live_analysis": analysis,
    }


def observe_selection(
    observations: Mapping[str, Any],
    selected_order: Sequence[str],
    *,
    predicted_saving: float | None,
    profiling_cost_usd: float,
) -> dict[str, Any]:
    selected_key = order_key(selected_order)
    original_key = observations["fixed_order"]
    costs = observations["mean_cost_usd"]
    if selected_key not in costs or original_key not in costs:
        return {
            "observed_result": "unavailable",
            "selected_order": list(selected_order),
            "observed_cost_usd": None,
            "observed_saving_usd": None,
            "observed_saving_percent": None,
            "observed_regret_usd": None,
            "predicted_saving_usd": predicted_saving,
            "predicted_minus_observed_saving_usd": None,
            "selected_is_observed_cheapest": None,
            "profiling_cost_usd": profiling_cost_usd,
            "note": (
                "no matching live execution; predicted cost is not substituted for a measurement"
            ),
        }
    selected_cost = costs[selected_key]
    original_cost = costs[original_key]
    cheapest = costs[observations["observed_cheapest_order"]]
    saving = original_cost - selected_cost
    percent = 100.0 * saving / original_cost if original_cost else 0.0
    return {
        "observed_result": "measured",
        "selected_order": list(selected_order),
        "observed_cost_usd": selected_cost,
        "observed_saving_usd": saving,
        "observed_saving_percent": percent,
        "observed_regret_usd": selected_cost - cheapest,
        "predicted_saving_usd": predicted_saving,
        "predicted_minus_observed_saving_usd": (
            None if predicted_saving is None else predicted_saving - saving
        ),
        "selected_is_observed_cheapest": selected_key == observations["observed_cheapest_order"],
        "profiling_cost_usd": profiling_cost_usd,
        "net_saving_by_horizon_usd": {
            str(horizon): horizon * saving - profiling_cost_usd for horizon in DEFAULT_HORIZONS
        },
        "forward_net_saving_sunk_excluded_by_horizon_usd": {
            str(horizon): horizon * saving for horizon in DEFAULT_HORIZONS
        },
    }


def _planner_settings(
    optimizer: OptimizerSettings,
    *,
    estimator: str,
    policy: str,
    **risk_updates: Any,
) -> OptimizerSettings:
    risk = optimizer.risk.model_copy(update=risk_updates) if risk_updates else optimizer.risk
    return optimizer.model_copy(
        update={
            "planner": optimizer.planner.model_copy(
                update={
                    "cardinality_estimator": estimator,
                    "cost_model_version": COST_MODEL[estimator],
                    "selection_policy": policy,
                }
            ),
            "risk": risk,
        }
    )


def _predicted_saving(plan: Mapping[str, Any]) -> float:
    original = plan["original_estimated_objective"]
    selected = plan["selected_estimated_objective"]
    return float(original - selected)


def _risk_saving(explanation: Mapping[str, Any]) -> float | None:
    selected = explanation.get("selected_risk") or {}
    payload = selected.get("predicted_saving_vs_original") or {}
    mean = payload.get("mean")
    return None if mean is None else float(mean)


def _compact(explanation: Mapping[str, Any]) -> dict[str, Any]:
    disagreement = explanation.get("disagreement") or {}
    selected_risk = explanation.get("selected_risk") or {}
    return {
        "decision": explanation.get("decision"),
        "reason": explanation.get("reason"),
        "reasons": explanation.get("reasons"),
        "minimum_objective_order": explanation.get("minimum_objective_order"),
        "bootstrap_probability_is_not_calibrated": explanation.get(
            "bootstrap_probability_is_not_calibrated", True
        ),
        "probability_cheaper_than_original": selected_risk.get(
            "probability_cheaper_than_original"
        ),
        "expected_saving_usd": (selected_risk.get("predicted_saving_vs_original") or {}).get(
            "mean"
        ),
        "expected_relative_saving": (
            selected_risk.get("predicted_relative_saving_vs_original") or {}
        ).get("mean"),
        "expected_regret_usd": (
            selected_risk.get("expected_regret_vs_modeled_minimum") or {}
        ).get("mean"),
        "cvar_loss_usd": (selected_risk.get("cvar_loss_vs_original") or {}).get("cvar"),
        "select_same_order": disagreement.get("select_same_order"),
        "agree_original_is_worse": disagreement.get("agree_original_is_worse"),
        "net_benefit": explanation.get("net_benefit"),
        "profiling_economics": explanation.get("profiling_economics"),
        "pre_profiling_economics": explanation.get("pre_profiling_economics"),
    }


def evaluate_workload(
    spec: Mapping[str, str],
    optimizer: OptimizerSettings,
    *,
    include_adaptive: bool = True,
) -> dict[str, Any]:
    query = load_query(spec["query"])
    profile_dir = Path(spec["profile_dir"])
    profile = ProfilingResult.model_validate(_load_json(_profiling_report_path(profile_dir)))
    split = SplitManifest.model_validate(_load_json(profile_dir / "split_manifest.json"))
    if set(split.evaluation_ids) & set(split.profiling_ids):
        raise ValueError(f"{spec['id']} split leaks evaluation IDs into profiling")
    matrix = load_profile_matrix(profile_dir, profile, split)
    analysis = _load_json(Path(spec["analysis"]))
    observations = load_observations(analysis)
    original = [item.id for item in query.predicates]
    if observations["fixed_order"] != order_key(original):
        raise ValueError(f"{spec['id']} analysis fixed order does not match the query file")
    pricing = profile.pricing or {}
    input_rate = pricing.get("input_per_million_usd")
    output_rate = pricing.get("output_per_million_usd")
    profiling_cost = float(profile.estimated_cost_usd or 0.0)
    historical = None
    historical_path = spec.get("historical_strategies")
    if historical_path and Path(historical_path).is_file():
        historical = _load_json(Path(historical_path))

    independent = collect_paired_bootstrap(
        matrix,
        estimator_name="independent",
        original_order=original,
        objective=optimizer.planner.objective,
        evaluation_record_count=split.evaluation_size,
        question_hashes=profile.predicate_hashes,
        settings=optimizer.bootstrap,
        correlation=optimizer.correlation,
        input_price_per_million=input_rate,
        output_price_per_million=output_rate,
        tie_relative_tolerance=optimizer.planner.tie_relative_tolerance,
    )
    correlation = collect_paired_bootstrap(
        matrix,
        estimator_name="correlation_aware",
        original_order=original,
        objective=optimizer.planner.objective,
        evaluation_record_count=split.evaluation_size,
        question_hashes=profile.predicate_hashes,
        settings=optimizer.bootstrap,
        correlation=optimizer.correlation,
        input_price_per_million=input_rate,
        output_price_per_million=output_rate,
        tie_relative_tolerance=optimizer.planner.tie_relative_tolerance,
    )

    def plan(estimator: str, policy: str, **risk_updates: Any) -> Any:
        settings = _planner_settings(
            optimizer, estimator=estimator, policy=policy, **risk_updates
        )
        return CostBasedPlanner(settings).plan(query, profile, split, matrix=matrix)

    def risk_select(
        *,
        robust: bool,
        estimator: str = "correlation_aware",
        primary_collected: Mapping[str, Any] | None = None,
        **risk_updates: Any,
    ) -> tuple[list[str], dict[str, Any]]:
        settings = _planner_settings(
            optimizer, estimator=estimator, policy="risk_aware", **risk_updates
        )
        cached = primary_collected
        if cached is None:
            cached = correlation if estimator == "correlation_aware" else independent
        return select_risk_aware_plan(
            matrix,
            original_order=original,
            objective=optimizer.planner.objective,
            evaluation_record_count=split.evaluation_size,
            question_hashes=profile.predicate_hashes,
            bootstrap=settings.bootstrap,
            correlation=settings.correlation,
            risk=settings.risk,
            economics=settings.economics,
            input_price_per_million=input_rate,
            output_price_per_million=output_rate,
            tie_relative_tolerance=settings.planner.tie_relative_tolerance,
            primary_estimator=estimator,
            robust=robust,
            statistics_state="available",
            new_profiling_cost=profiling_cost,
            sunk_profiling_cost=profiling_cost,
            primary_collected=cached,
            independent_collected=independent,
            correlation_collected=correlation,
        )

    independent_plan = plan("independent", "minimum_expected_objective")
    correlation_plan = plan("correlation_aware", "minimum_expected_objective")
    confidence_plan = plan("correlation_aware", "confidence_aware")
    risk_order, risk_explanation = risk_select(robust=False)
    robust_order, robust_explanation = risk_select(robust=True)
    economics_order, economics_explanation = risk_select(
        robust=False, apply_profiling_economics=True
    )

    selections: dict[str, dict[str, Any]] = {
        "fixed": {
            "selected_order": original,
            "predicted_saving_usd": 0.0,
            "profiling_cost_usd": 0.0,
            "explanation": {
                "decision": "KEEP_ORIGINAL",
                "reason": "Query-file order; no optimizer.",
            },
        },
        "independent": {
            "selected_order": independent_plan.selected_predicate_order,
            "predicted_saving_usd": _predicted_saving(independent_plan.artifact_dict()),
            "profiling_cost_usd": profiling_cost,
            "explanation": {"decision": "ADOPT_REORDER"}
            if independent_plan.reordered
            else {"decision": "KEEP_ORIGINAL"},
        },
        "correlation_aware": {
            "selected_order": correlation_plan.selected_predicate_order,
            "predicted_saving_usd": _predicted_saving(correlation_plan.artifact_dict()),
            "profiling_cost_usd": profiling_cost,
            "explanation": correlation_plan.selection_explanation,
        },
        "correlation_aware_confidence": {
            "selected_order": confidence_plan.selected_predicate_order,
            "predicted_saving_usd": _predicted_saving(confidence_plan.artifact_dict()),
            "profiling_cost_usd": profiling_cost,
            "explanation": confidence_plan.selection_explanation,
        },
        "risk_aware": {
            "selected_order": risk_order,
            "predicted_saving_usd": _risk_saving(risk_explanation),
            "profiling_cost_usd": profiling_cost,
            "explanation": risk_explanation,
        },
        "independence_ungated": {
            "selected_order": independent_plan.selected_predicate_order,
            "predicted_saving_usd": _predicted_saving(independent_plan.artifact_dict()),
            "profiling_cost_usd": profiling_cost,
            "explanation": {"ablation_of": "independent"},
        },
        "correlation_ungated": {
            "selected_order": correlation_plan.selected_predicate_order,
            "predicted_saving_usd": _predicted_saving(correlation_plan.artifact_dict()),
            "profiling_cost_usd": profiling_cost,
            "explanation": {"ablation_of": "correlation_aware"},
        },
        "correlation_confidence": {
            "selected_order": confidence_plan.selected_predicate_order,
            "predicted_saving_usd": _predicted_saving(confidence_plan.artifact_dict()),
            "profiling_cost_usd": profiling_cost,
            "explanation": {"ablation_of": "correlation_aware_confidence"},
        },
        "risk_aware_correlation": {
            "selected_order": risk_order,
            "predicted_saving_usd": _risk_saving(risk_explanation),
            "profiling_cost_usd": profiling_cost,
            "explanation": risk_explanation,
        },
        "risk_aware_robust": {
            "selected_order": robust_order,
            "predicted_saving_usd": _risk_saving(robust_explanation),
            "profiling_cost_usd": profiling_cost,
            "explanation": robust_explanation,
        },
        "risk_aware_economics": {
            "selected_order": economics_order,
            "predicted_saving_usd": _risk_saving(economics_explanation),
            "profiling_cost_usd": profiling_cost,
            "explanation": economics_explanation,
        },
    }

    adaptive_payload = None
    if include_adaptive:
        width = len(original)
        prior = profiling_cost / max(1, profile.completed_decisions)
        bound = split.evaluation_size * (width - 1) * prior
        adaptive_payload = simulate_adaptive_profiling(
            matrix,
            estimator_name="correlation_aware",
            original_order=original,
            objective=optimizer.planner.objective,
            evaluation_record_count=split.evaluation_size,
            question_hashes=profile.predicate_hashes,
            settings=optimizer.adaptive,
            bootstrap=optimizer.bootstrap,
            correlation=optimizer.correlation,
            input_price_per_million=input_rate,
            output_price_per_million=output_rate,
            expected_executions=optimizer.economics.horizon(),
            tie_relative_tolerance=optimizer.planner.tie_relative_tolerance,
            economics=optimizer.economics,
            savings_upper_bound_per_execution=bound,
            savings_bound_basis="structural_upper_bound",
            estimated_cost_per_decision=prior,
            statistics_available=False,
        )
        adaptive_cost = float(adaptive_payload.get("profiling_cost_usd_offline_estimate") or 0.0)
        if adaptive_payload.get("profiling_started") and adaptive_payload.get(
            "stopping_checkpoint"
        ):
            adaptive_cost = float(
                adaptive_payload["stopping_checkpoint"]["profiling_cost_usd_offline_estimate"]
                or 0.0
            )
            size = int(adaptive_payload["stopping_checkpoint"]["profiled_reviews"])
            ranked = sampling_order([row.review_id for row in matrix.rows], optimizer.adaptive.seed)
            prefix = matrix.select_reviews(ranked[:size])
            prefix_corr = collect_paired_bootstrap(
                prefix,
                estimator_name="correlation_aware",
                original_order=original,
                objective=optimizer.planner.objective,
                evaluation_record_count=split.evaluation_size,
                question_hashes=profile.predicate_hashes,
                settings=optimizer.bootstrap,
                correlation=optimizer.correlation,
                input_price_per_million=input_rate,
                output_price_per_million=output_rate,
                tie_relative_tolerance=optimizer.planner.tie_relative_tolerance,
            )
            adaptive_order, adaptive_explanation = select_risk_aware_plan(
                prefix,
                original_order=original,
                objective=optimizer.planner.objective,
                evaluation_record_count=split.evaluation_size,
                question_hashes=profile.predicate_hashes,
                bootstrap=optimizer.bootstrap,
                correlation=optimizer.correlation,
                risk=optimizer.risk,
                economics=optimizer.economics,
                input_price_per_million=input_rate,
                output_price_per_million=output_rate,
                tie_relative_tolerance=optimizer.planner.tie_relative_tolerance,
                primary_estimator="correlation_aware",
                robust=False,
                statistics_state="available",
                new_profiling_cost=adaptive_cost,
                sunk_profiling_cost=adaptive_cost,
                primary_collected=prefix_corr,
            )
        else:
            adaptive_order = original
            adaptive_explanation = {
                "decision": "SKIP_PROFILING",
                "reason": "Adaptive controller skipped profiling.",
                "adaptive": adaptive_payload,
            }
            adaptive_cost = 0.0
        selections["risk_aware_adaptive"] = {
            "selected_order": adaptive_order,
            "predicted_saving_usd": _risk_saving(adaptive_explanation),
            "profiling_cost_usd": adaptive_cost,
            "explanation": adaptive_explanation,
        }
    else:
        selections["risk_aware_adaptive"] = {
            "selected_order": original,
            "predicted_saving_usd": 0.0,
            "profiling_cost_usd": 0.0,
            "explanation": {"decision": "SKIP_PROFILING", "reason": "adaptive ablation disabled"},
        }

    scored: dict[str, Any] = {}
    for name, payload in selections.items():
        observed = observe_selection(
            observations,
            payload["selected_order"],
            predicted_saving=payload["predicted_saving_usd"],
            profiling_cost_usd=payload["profiling_cost_usd"],
        )
        historical_order = None
        historical_names = {"independent", "correlation_aware", "correlation_aware_confidence"}
        if historical and name in historical_names:
            historical_order = (historical.get("strategies") or {}).get(name, {}).get(
                "selected_order"
            )
        scored[name] = {
            **observed,
            "decision": (payload.get("explanation") or {}).get("decision"),
            "reason": (payload.get("explanation") or {}).get("reason"),
            "explanation": _compact(payload.get("explanation") or {}),
            "matches_historical_m3_selection": (
                None
                if historical_order is None
                else list(payload["selected_order"]) == list(historical_order)
            ),
        }

    return {
        "workload_id": spec["id"],
        "query_id": query.query_id,
        "dataset_fingerprint": split.dataset_fingerprint,
        "profiling_fingerprint": split.profiling_fingerprint,
        "evaluation_fingerprint": split.evaluation_fingerprint,
        "profiling_size": split.profiling_size,
        "evaluation_size": split.evaluation_size,
        "original_order": original,
        "executed_orders": observations["executed_orders"],
        "observed_cheapest_order": observations["observed_cheapest_order"],
        "profiling_cost_usd": profiling_cost,
        "held_out_outcomes_used_for_selection": False,
        "response_variability": response_variability_sensitivity(analysis),
        "independent_full_sample_order": list(independent["full_sample_selected_order"]),
        "correlation_aware_full_sample_order": list(correlation["full_sample_selected_order"]),
        "selections": scored,
        "adaptive": {
            "profiling_started": None if adaptive_payload is None else adaptive_payload.get(
                "profiling_started"
            ),
            "stopping_checkpoint": None
            if adaptive_payload is None
            else (adaptive_payload.get("stopping_checkpoint") or {}).get("profiled_reviews"),
            "selected_order_before_risk_gate": None
            if adaptive_payload is None
            else adaptive_payload.get("selected_order"),
        },
        "label": "exploratory_retrospective_benchmark",
    }


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def summarize_strategy(
    workloads: Sequence[Mapping[str, Any]], name: str
) -> dict[str, Any]:
    rows = []
    for workload in workloads:
        payload = workload["selections"][name]
        rows.append({"workload_id": workload["workload_id"], **payload})
    measured = [row for row in rows if row["observed_result"] == "measured"]
    percents = [row["observed_saving_percent"] for row in measured]
    usd = [row["observed_saving_usd"] for row in measured]
    regret = [row["observed_regret_usd"] for row in measured]
    keep = sum(
        1
        for row in rows
        if row["selected_order"]
        == next(
            item["original_order"]
            for item in workloads
            if item["workload_id"] == row["workload_id"]
        )
    )
    improved = sum(1 for row in measured if row["observed_saving_usd"] > 0)
    worse = sum(1 for row in measured if row["observed_saving_usd"] < 0)
    leave_one_out = {}
    for left in rows:
        others = [
            row["observed_saving_percent"]
            for row in measured
            if row["workload_id"] != left["workload_id"]
            and row["observed_saving_percent"] is not None
        ]
        leave_one_out[left["workload_id"]] = _mean(others)
    errors = [
        abs(row["predicted_minus_observed_saving_usd"])
        for row in measured
        if row["predicted_minus_observed_saving_usd"] is not None
    ]
    abstentions = sum(
        1
        for row in rows
        if row.get("decision") in {"KEEP_ORIGINAL", "SKIP_PROFILING"}
    )
    return {
        "strategy": name,
        "workloads": {row["workload_id"]: row for row in rows},
        "mean_saving_percent": _mean(percents),
        "median_saving_percent": median(percents) if percents else None,
        "mean_saving_usd": _mean(usd),
        "total_saving_usd": sum(usd) if usd else None,
        "mean_regret_usd": _mean(regret),
        "maximum_regret_usd": max(regret) if regret else None,
        "workloads_improved": improved,
        "workloads_worse": worse,
        "workloads_retaining_original": keep,
        "workloads_measured": len(measured),
        "workloads_unavailable": [
            row["workload_id"] for row in rows if row["observed_result"] != "measured"
        ],
        "leave_one_workload_out_mean_saving_percent": leave_one_out,
        "mean_absolute_predicted_minus_observed_saving_usd": _mean(errors),
        "total_profiling_cost_usd": sum(row["profiling_cost_usd"] or 0.0 for row in rows),
        "abstention_or_fallback_count": abstentions,
        "selection_accuracy": (
            sum(1 for row in measured if row["selected_is_observed_cheapest"]) / len(measured)
            if measured
            else None
        ),
        "net_saving_by_horizon_usd": {
            str(horizon): sum(
                (row.get("net_saving_by_horizon_usd") or {}).get(str(horizon), 0.0)
                for row in measured
            )
            for horizon in DEFAULT_HORIZONS
        },
    }


def run_offline_benchmark(
    optimizer: OptimizerSettings,
    output_dir: Path,
    *,
    workloads: Sequence[Mapping[str, str]] = DEFAULT_WORKLOADS,
    include_adaptive: bool = True,
    config_path: str | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    evaluated = [
        evaluate_workload(spec, optimizer, include_adaptive=include_adaptive)
        for spec in workloads
    ]
    strategy_names = list(MAJOR_STRATEGIES) + [
        name for name in ABLATIONS if name not in MAJOR_STRATEGIES
    ]
    summaries = {name: summarize_strategy(evaluated, name) for name in strategy_names}
    report = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "created_at": _now(),
        "git_commit": git_commit(),
        "git_worktree_dirty": git_worktree_dirty(),
        "label": "exploratory_retrospective_benchmark",
        "not_a_confirmatory_test": True,
        "live_jev_calls": 0,
        "held_out_outcomes_used_for_selection": False,
        "bootstrap_iterations": optimizer.bootstrap.iterations,
        "bootstrap_seed": optimizer.bootstrap.seed,
        "bootstrap_probability_is_not_calibrated": True,
        "config_path": config_path,
        "risk_settings": optimizer.risk.model_dump(mode="json"),
        "workloads": evaluated,
        "strategies": {name: summaries[name] for name in MAJOR_STRATEGIES},
        "ablations": {name: summaries[name] for name in ABLATIONS},
        "caveat": (
            "four semantically related SemCEB workloads; bootstrap frequencies are "
            "resampling diagnostics, not calibrated production probabilities; "
            "do not claim statistical significance from this sample"
        ),
    }
    ResultStore.write_json(output_dir / "benchmark.json", report)
    for workload in evaluated:
        ResultStore.write_json(
            output_dir / f"{workload['workload_id']}.json",
            workload,
        )
    return report


def write_text_summary(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Milestone 4 offline benchmark",
        "",
        "Exploratory retrospective evaluation on stored Milestone 3 live traces.",
        "No new Jev requests. Bootstrap probabilities are resampling diagnostics.",
        "",
        f"Bootstrap iterations: {report['bootstrap_iterations']}",
        f"Seed: {report['bootstrap_seed']}",
        "",
        "## Major strategies (observed execution-cost saving percent)",
        "",
        "| Strategy | PQ | LM | PE | MF | mean | median | worse | keep original |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in MAJOR_STRATEGIES:
        summary = report["strategies"][name]
        cells = []
        for workload in (
            "product_quality",
            "listing_mismatch",
            "positive_experience",
            "mixed_feedback",
        ):
            row = summary["workloads"].get(workload, {})
            value = row.get("observed_saving_percent")
            cells.append("n/a" if value is None else f"{value:.2f}")
        mean = summary["mean_saving_percent"]
        median_value = summary["median_saving_percent"]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    *cells,
                    "n/a" if mean is None else f"{mean:.2f}",
                    "n/a" if median_value is None else f"{median_value:.2f}",
                    str(summary["workloads_worse"]),
                    str(summary["workloads_retaining_original"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
