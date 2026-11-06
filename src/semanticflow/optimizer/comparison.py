from __future__ import annotations

from typing import Any

from semanticflow.optimizer.economics import break_even_executions
from semanticflow.optimizer.models import PlanSelection, ProfilingResult


def _percent_reduction(original: float, optimized: float) -> float | None:
    if original == 0:
        return 0.0 if optimized == 0 else None
    return (original - optimized) / original * 100


def _objective_value(report: dict[str, Any], objective: str) -> float | None:
    field = {
        "estimated_cost_usd": "estimated_cost_usd",
        "total_tokens": "total_tokens",
        "input_tokens": "input_tokens",
        "evaluations": "completed_decisions",
    }[objective]
    value = report.get(field)
    return float(value) if value is not None else None


def _prediction_error(predicted: float, observed: float | None) -> dict[str, float | None]:
    if observed is None:
        return {
            "predicted": predicted,
            "observed": None,
            "absolute_error": None,
            "relative_error_percent": None,
        }
    absolute = abs(predicted - observed)
    relative = None if observed == 0 else absolute / observed * 100
    return {
        "predicted": predicted,
        "observed": observed,
        "absolute_error": absolute,
        "relative_error_percent": relative,
    }


def compare_execution_reports(
    fixed: dict[str, Any],
    optimized: dict[str, Any],
    profile: ProfilingResult,
    plan: PlanSelection,
    *,
    fixed_match_ids: set[str],
    optimized_match_ids: set[str],
) -> dict[str, Any]:
    if fixed["total_input_reviews"] != optimized["total_input_reviews"]:
        raise ValueError("fixed and optimized runs must use the same number of records")
    fixed_objective = _objective_value(fixed, plan.objective)
    optimized_objective = _objective_value(optimized, plan.objective)
    profile_objective = {
        "estimated_cost_usd": profile.estimated_cost_usd,
        "total_tokens": float(profile.total_tokens),
        "input_tokens": float(profile.input_tokens),
        "evaluations": float(profile.completed_decisions),
    }[plan.objective]
    execution_savings = (
        fixed_objective - optimized_objective
        if fixed_objective is not None and optimized_objective is not None
        else None
    )
    net_savings = (
        execution_savings - profile_objective
        if execution_savings is not None and profile_objective is not None
        else None
    )
    break_even = None
    if execution_savings is not None and execution_savings > 0 and profile_objective is not None:
        break_even = break_even_executions(profile_objective, execution_savings)

    fixed_only = sorted(fixed_match_ids - optimized_match_ids)
    optimized_only = sorted(optimized_match_ids - fixed_match_ids)
    return {
        "schema_version": "m2-comparison-v1",
        "comparison_status": (
            "complete"
            if fixed["completion_status"] == optimized["completion_status"] == "complete"
            else "incomplete"
        ),
        "records_comparable": fixed["total_input_reviews"] == optimized["total_input_reviews"],
        "evaluation_record_count": fixed["total_input_reviews"],
        "original_predicate_order": plan.original_predicate_order,
        "selected_predicate_order": plan.selected_predicate_order,
        "reordered": plan.reordered,
        "objective": plan.objective,
        "fixed": {
            "completed_decisions": fixed["completed_decisions"],
            "provider_attempts": fixed["provider_attempts"],
            "http_attempts": fixed["http_attempts"],
            "input_tokens": fixed["input_tokens"],
            "output_tokens": fixed["output_tokens"],
            "total_tokens": fixed["total_tokens"],
            "estimated_cost_usd": fixed["estimated_cost_usd"],
            "wall_time_seconds": fixed["wall_time_seconds"],
            "final_matching_reviews": fixed["final_matching_reviews"],
            "completion_status": fixed["completion_status"],
        },
        "optimized": {
            "completed_decisions": optimized["completed_decisions"],
            "provider_attempts": optimized["provider_attempts"],
            "http_attempts": optimized["http_attempts"],
            "input_tokens": optimized["input_tokens"],
            "output_tokens": optimized["output_tokens"],
            "total_tokens": optimized["total_tokens"],
            "estimated_cost_usd": optimized["estimated_cost_usd"],
            "wall_time_seconds": optimized["wall_time_seconds"],
            "final_matching_reviews": optimized["final_matching_reviews"],
            "completion_status": optimized["completion_status"],
        },
        "request_reduction_percent": _percent_reduction(
            float(fixed["completed_decisions"]), float(optimized["completed_decisions"])
        ),
        "provider_attempt_reduction_percent": _percent_reduction(
            float(fixed["provider_attempts"]), float(optimized["provider_attempts"])
        ),
        "http_attempt_reduction_percent": _percent_reduction(
            float(fixed["http_attempts"]), float(optimized["http_attempts"])
        ),
        "token_reduction_percent": _percent_reduction(
            float(fixed["total_tokens"]), float(optimized["total_tokens"])
        ),
        "cost_reduction_percent": (
            _percent_reduction(
                float(fixed["estimated_cost_usd"]), float(optimized["estimated_cost_usd"])
            )
            if fixed["estimated_cost_usd"] is not None
            and optimized["estimated_cost_usd"] is not None
            else None
        ),
        "latency_reduction_percent": _percent_reduction(
            float(fixed["wall_time_seconds"]), float(optimized["wall_time_seconds"])
        ),
        "wall_time_comparison": "measured_same_process_same_configuration",
        "prediction_error": {
            "original": _prediction_error(plan.original_estimated_objective, fixed_objective),
            "selected": _prediction_error(plan.selected_estimated_objective, optimized_objective),
        },
        "profiling_overhead": {
            "requests": profile.completed_decisions,
            "provider_attempts": profile.provider_attempts,
            "http_attempts": profile.http_attempts,
            "input_tokens": profile.input_tokens,
            "output_tokens": profile.output_tokens,
            "total_tokens": profile.total_tokens,
            "estimated_cost_usd": profile.estimated_cost_usd,
            "wall_time_seconds": profile.wall_time_seconds,
            "objective_value": profile_objective,
        },
        "execution_only_objective_savings": execution_savings,
        "net_objective_savings_after_profiling": net_savings,
        "break_even_execution_count": break_even,
        "break_even_savings_basis": "observed_execution_difference",
        "output_equivalence": {
            "equivalent": not fixed_only and not optimized_only,
            "fixed_only_review_ids": fixed_only,
            "optimized_only_review_ids": optimized_only,
        },
        "candidate_ranking_accuracy": {
            "status": "partial_two_plan_observation",
            "predicted_selected_better_or_equal": plan.selected_estimated_objective
            <= plan.original_estimated_objective,
            "observed_selected_better_or_equal": (
                optimized_objective <= fixed_objective
                if fixed_objective is not None and optimized_objective is not None
                else None
            ),
        },
    }
