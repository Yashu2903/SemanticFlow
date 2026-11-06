from __future__ import annotations

from collections.abc import Sequence

from semanticflow.optimizer.cardinality import (
    CardinalityEstimator,
    IndependentEstimator,
    StageEstimate,
)
from semanticflow.optimizer.models import (
    CandidatePlan,
    OptimizationMetric,
    PredicateStatistics,
)


def _objective_value(
    objective: OptimizationMetric,
    *,
    evaluations: float,
    input_tokens: float | None,
    total_tokens: float | None,
    cost_usd: float | None,
) -> float:
    if objective == "evaluations":
        return evaluations
    if objective == "input_tokens":
        if input_tokens is None:
            raise ValueError("input-token objective requires measured input-token usage")
        return input_tokens
    if objective == "total_tokens":
        if total_tokens is None:
            raise ValueError("total-token objective requires measured token usage")
        return total_tokens
    if cost_usd is None:
        raise ValueError("monetary objective requires verified pricing and token usage")
    return cost_usd


def estimate_plan(
    predicate_order: Sequence[str],
    statistics: dict[str, PredicateStatistics],
    *,
    input_record_count: int,
    objective: OptimizationMetric,
    original_order: Sequence[str] | None = None,
    estimator: CardinalityEstimator | None = None,
) -> CandidatePlan:
    if input_record_count <= 0:
        raise ValueError("input_record_count must be positive")
    order = list(predicate_order)
    if not 2 <= len(order) <= 6:
        raise ValueError("cost model supports 2 to 6 predicates")
    if len(order) != len(set(order)):
        raise ValueError("candidate order contains duplicate predicates")
    missing = set(order) - set(statistics)
    if missing:
        raise ValueError(f"missing statistics for predicates: {sorted(missing)}")
    selected_estimator = estimator or IndependentEstimator(statistics)
    return plan_from_stage_estimates(
        order,
        selected_estimator.stage_estimates(order),
        input_record_count=input_record_count,
        objective=objective,
        original_order=original_order,
        estimator_name=selected_estimator.name,
    )


def plan_from_stage_estimates(
    order: list[str],
    stages: list[StageEstimate],
    *,
    input_record_count: int,
    objective: OptimizationMetric,
    original_order: Sequence[str] | None,
    estimator_name: str,
) -> CandidatePlan:
    if [item.predicate_id for item in stages] != order:
        raise ValueError("stage estimates do not match the candidate order")
    estimated_evaluations = 0.0
    estimated_input_tokens = 0.0
    estimated_output_tokens = 0.0
    estimated_cost_usd = 0.0
    estimated_latency = 0.0
    token_usage_known = True
    monetary_cost_known = True
    stage_inputs: list[float] = []

    for item in stages:
        evaluations = input_record_count * item.reach_probability
        stage_inputs.append(evaluations)
        estimated_evaluations += evaluations
        estimated_latency += evaluations * item.average_latency_seconds
        if item.average_input_tokens is None or item.average_output_tokens is None:
            token_usage_known = False
        else:
            estimated_input_tokens += evaluations * item.average_input_tokens
            estimated_output_tokens += evaluations * item.average_output_tokens
        if item.average_cost_usd is None:
            monetary_cost_known = False
        else:
            estimated_cost_usd += evaluations * item.average_cost_usd

    input_tokens = estimated_input_tokens if token_usage_known else None
    output_tokens = estimated_output_tokens if token_usage_known else None
    total_tokens = estimated_input_tokens + estimated_output_tokens if token_usage_known else None
    cost_usd = estimated_cost_usd if monetary_cost_known else None
    objective_value = _objective_value(
        objective,
        evaluations=estimated_evaluations,
        input_tokens=input_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
    )
    confidences = {item.confidence for item in stages if item.confidence is not None}
    return CandidatePlan(
        rank=1,
        predicate_order=order,
        estimated_evaluations=estimated_evaluations,
        estimated_input_tokens=input_tokens,
        estimated_output_tokens=output_tokens,
        estimated_total_tokens=total_tokens,
        estimated_cost_usd=cost_usd,
        estimated_summed_latency_seconds=estimated_latency,
        objective=objective,
        objective_value=objective_value,
        estimated_reduction_vs_original_percent=None,
        is_original_order=original_order is not None and order == list(original_order),
        cardinality_estimator=estimator_name,  # type: ignore[arg-type]
        estimated_stage_input_records=stage_inputs,
        stage_estimates=[item.to_dict() for item in stages],
        confidence=("low" if "low" in confidences else "high") if confidences else None,
    )


def percentage_reduction(original: float, candidate: float) -> float | None:
    if original < 0 or candidate < 0:
        raise ValueError("costs cannot be negative")
    if original == 0:
        return 0.0 if candidate == 0 else None
    return (original - candidate) / original * 100
