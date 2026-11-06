from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from semanticflow.optimizer.bootstrap import (
    order_costs,
    order_key,
    paired_bootstrap,
    select_minimum,
)
from semanticflow.optimizer.economics import decide_adaptive_profiling_start
from semanticflow.optimizer.matrix import ProfileMatrix
from semanticflow.optimizer.models import (
    AdaptiveProfilingSettings,
    BootstrapSettings,
    CorrelationSettings,
    EconomicsSettings,
    OptimizationMetric,
)

ADAPTIVE_SCHEMA_VERSION = "m3-adaptive-profiling-v2"

SAMPLING_DESIGN = {
    "design": "uniform_without_replacement_complete_rows",
    "order": "sha256(seed:review_id) rank fixed before any outcome is observed",
    "per_row_evaluation": "every predicate is evaluated for every sampled review",
    "selection_bias": (
        "None from row or predicate selection: inclusion never depends on observed "
        "outcomes, so each checkpoint is a simple random sample of fixed size. The stopping "
        "time does depend on data, so estimates reported at a data-dependent stop are not "
        "guaranteed unbiased. Outcome-dependent (predicate-selective) profiling is not "
        "implemented; adding it would require inverse-probability weighting."
    ),
}


def sampling_order(review_ids: Sequence[str], seed: int) -> list[str]:
    return sorted(
        review_ids,
        key=lambda review_id: (
            hashlib.sha256(f"{seed}:{review_id}".encode()).hexdigest(),
            review_id,
        ),
    )


def _row_cost(
    matrix: ProfileMatrix,
    rows: Sequence[int],
    input_price: float | None,
    output_price: float | None,
) -> float | None:
    if input_price is None or output_price is None:
        return None
    total = 0.0
    for index in rows:
        row = matrix.rows[index]
        for input_tokens, output_tokens in zip(row.input_tokens, row.output_tokens, strict=True):
            if input_tokens is None or output_tokens is None:
                return None
            total += (input_tokens * input_price + output_tokens * output_price) / 1_000_000
    return total


def simulate_adaptive_profiling(
    matrix: ProfileMatrix,
    *,
    estimator_name: str,
    original_order: Sequence[str],
    objective: OptimizationMetric,
    evaluation_record_count: int,
    question_hashes: Mapping[str, str],
    settings: AdaptiveProfilingSettings,
    bootstrap: BootstrapSettings,
    correlation: CorrelationSettings,
    input_price_per_million: float | None,
    output_price_per_million: float | None,
    expected_executions: int,
    tie_relative_tolerance: float = 1e-9,
    economics: EconomicsSettings | None = None,
    savings_upper_bound_per_execution: float | None = None,
    savings_bound_basis: str | None = None,
    estimated_cost_per_decision: float | None = None,
    statistics_available: bool = False,
) -> dict[str, Any]:
    """Replay sequential profiling offline over an already complete profiling matrix.

    No provider is contacted. Consumed decisions and costs are the recorded usage of the
    rows that the controller would have profiled, so savings are offline estimates.

    Before the first checkpoint, ``decide_adaptive_profiling_start`` may skip profiling
    entirely; the original order is then executed and no decision is consumed. The
    first-checkpoint cost uses ``estimated_cost_per_decision`` when given (what a live
    controller knows beforehand) and otherwise the recorded usage of those rows.
    """
    if not matrix.complete:
        raise ValueError("adaptive simulation requires a complete profiling matrix")
    cost_unit = {
        "estimated_cost_usd": "usd",
        "evaluations": "evaluations",
        "total_tokens": "tokens",
        "input_tokens": "tokens",
    }[objective]
    original = tuple(original_order)
    order = sampling_order([row.review_id for row in matrix.rows], settings.seed)
    position = {row.review_id: index for index, row in enumerate(matrix.rows)}
    ordered_indices = [position[review_id] for review_id in order]
    width = len(matrix.predicate_ids)
    common = {
        "estimator_name": estimator_name,
        "original_order": original,
        "objective": objective,
        "evaluation_record_count": evaluation_record_count,
        "question_hashes": question_hashes,
        "correlation": correlation,
        "input_price_per_million": input_price_per_million,
        "output_price_per_million": output_price_per_million,
    }
    full_costs = order_costs(matrix, **common)
    full_plan = select_minimum(full_costs, original, tie_relative_tolerance)
    full_cost = _row_cost(
        matrix, ordered_indices, input_price_per_million, output_price_per_million
    )
    checkpoints = [value for value in settings.checkpoints if value <= matrix.size]
    if not checkpoints:
        raise ValueError("no adaptive checkpoint fits within the profiling matrix")
    first_decisions = checkpoints[0] * width
    if estimated_cost_per_decision is not None:
        first_cost: float | None = estimated_cost_per_decision * first_decisions
        first_cost_source = "estimated_cost_per_decision_times_decisions"
    else:
        first_cost = _row_cost(
            matrix,
            ordered_indices[: checkpoints[0]],
            input_price_per_million,
            output_price_per_million,
        )
        first_cost_source = "recorded_usage_of_first_checkpoint_rows_offline"
    start = decide_adaptive_profiling_start(
        expected_executions=expected_executions,
        first_checkpoint_decisions=first_decisions,
        first_checkpoint_cost=first_cost,
        settings=economics,
        savings_upper_bound_per_execution=savings_upper_bound_per_execution,
        savings_bound_basis=savings_bound_basis,
        statistics_available=statistics_available,
        maximum_profiling_decisions=settings.maximum_profiling_decisions,
        adaptive_budget_usd=settings.profiling_budget_usd,
    )
    start["first_checkpoint_cost_source"] = first_cost_source
    header = {
        "schema_version": ADAPTIVE_SCHEMA_VERSION,
        "evidence": "offline_simulation_over_recorded_complete_profiling_matrix",
        "live_provider_calls": 0,
        "sampling": {**SAMPLING_DESIGN, "seed": settings.seed},
        "estimator": estimator_name,
        "objective": objective,
        "objective_unit": cost_unit,
        "expected_executions": expected_executions,
        "checkpoints": checkpoints,
        "stability_threshold": settings.stability_threshold,
        "bootstrap_iterations_per_checkpoint": settings.bootstrap_iterations,
        "value_of_information_bound": (
            "expected_executions * bootstrap mean regret of the selected plan; an EVPI-style "
            "upper bound on what further profiling of this workload could save"
        ),
        "full_profile": {
            "profiled_reviews": matrix.size,
            "profiling_decisions": matrix.size * width,
            "profiling_cost_usd": full_cost,
            "selected_order": list(full_plan),
            "predicted_saving_per_execution_vs_original": (
                full_costs[original] - full_costs[full_plan]
            ),
            "note": "diagnostic computed from every profiled row; unknown to the controller",
        },
        "start_decision": start,
    }
    if start["action"] != "start_adaptive_profiling":
        return {
            **header,
            "profiling_started": False,
            "stopping_checkpoint": None,
            "selected_order": list(original),
            "selected_order_source": (
                "existing_statistics"
                if start["action"] == "skip_profiling_use_existing_statistics"
                else "original_order_without_profiling"
            ),
            "profiling_decisions_used": 0,
            "profiling_decisions_saved": matrix.size * width,
            "profiling_cost_saved_usd_offline_estimate": full_cost,
            "stopped_plan_agrees_with_full_profile": original == full_plan,
            "trajectory": [],
        }
    trajectory: list[dict[str, Any]] = []
    stop: dict[str, Any] | None = None
    for number, size in enumerate(checkpoints):
        rows = ordered_indices[:size]
        sample = matrix.take(rows)
        costs = order_costs(sample, **common)
        selected = select_minimum(costs, original, tie_relative_tolerance)
        report = paired_bootstrap(
            sample,
            settings=bootstrap,
            tie_relative_tolerance=tie_relative_tolerance,
            iterations=settings.bootstrap_iterations,
            seed=settings.seed + size,
            **common,
        )
        consumed_cost = _row_cost(matrix, rows, input_price_per_million, output_price_per_million)
        decisions = size * width
        next_size = checkpoints[number + 1] if number + 1 < len(checkpoints) else None
        increment_cost = None
        if next_size is not None and consumed_cost is not None:
            increment_cost = consumed_cost / size * (next_size - size)
        value_bound = expected_executions * report["reference_regret"]["mean"]
        stability = report["selection_frequency"][order_key(selected)]
        reasons: list[str] = []
        if next_size is None:
            reasons.append("maximum_checkpoint_reached")
        else:
            if (
                settings.profiling_budget_usd is not None
                and consumed_cost is not None
                and increment_cost is not None
                and consumed_cost + increment_cost > settings.profiling_budget_usd
            ):
                reasons.append("profiling_budget_exhausted_before_next_checkpoint")
            if (
                settings.maximum_profiling_decisions is not None
                and next_size * width > settings.maximum_profiling_decisions
            ):
                reasons.append("decision_budget_exhausted_before_next_checkpoint")
            if stability >= settings.stability_threshold:
                reasons.append("selected_plan_stable")
            if (
                objective == "estimated_cost_usd"
                and increment_cost is not None
                and value_bound <= increment_cost
            ):
                reasons.append("expected_value_of_information_below_increment_cost")
        decision = "stop" if reasons else "continue"
        entry = {
            "checkpoint": number + 1,
            "profiled_reviews": size,
            "profiling_decisions": decisions,
            "profiling_cost_usd_offline_estimate": consumed_cost,
            "selected_order": list(selected),
            "selected_predicted_objective": costs[selected],
            "original_predicted_objective": costs[original],
            "agrees_with_full_profile_plan": selected == full_plan,
            "bootstrap_selection_frequency_of_selected": stability,
            "bootstrap_modal_order": report["modal_order"],
            "bootstrap_probability_selected_cheaper_than_original": report[
                "probability_reference_cheaper_than_original"
            ],
            "bootstrap_mean_regret_per_execution": report["reference_regret"]["mean"],
            "expected_value_of_information_bound": value_bound,
            "next_checkpoint_reviews": next_size,
            "next_increment_cost_usd_estimate": increment_cost,
            "controller_decision": decision if stop is None else "after_stop_diagnostic",
            "reasons": reasons,
        }
        trajectory.append(entry)
        if stop is None and decision == "stop":
            stop = entry
    assert stop is not None
    saved_cost = (
        full_cost - stop["profiling_cost_usd_offline_estimate"]
        if full_cost is not None and stop["profiling_cost_usd_offline_estimate"] is not None
        else None
    )
    return {
        **header,
        "profiling_started": True,
        "stopping_checkpoint": stop,
        "selected_order": stop["selected_order"],
        "selected_order_source": "adaptive_profile_at_stopping_checkpoint",
        "profiling_decisions_used": stop["profiling_decisions"],
        "profiling_decisions_saved": matrix.size * width - stop["profiling_decisions"],
        "profiling_cost_saved_usd_offline_estimate": saved_cost,
        "stopped_plan_agrees_with_full_profile": stop["agrees_with_full_profile_plan"],
        "trajectory": trajectory,
    }
