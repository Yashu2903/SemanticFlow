from __future__ import annotations

import math
from typing import Any, Literal

from semanticflow.optimizer.models import EconomicsSettings

ECONOMICS_SCHEMA_VERSION = "m3-optimization-economics-v2"

StatisticsState = Literal["none", "available"]
DecisionAction = Literal[
    "execute_original_without_profiling",
    "collect_additional_statistics",
    "optimize_with_existing_statistics",
    "reuse_existing_plan",
    "decline_optimization_execute_original",
]
AdaptiveStartAction = Literal[
    "skip_profiling_execute_original",
    "skip_profiling_use_existing_statistics",
    "start_adaptive_profiling",
]

# What a per-execution saving was derived from. Only "observed" comes from executions, and
# even that compares runs made at different times; none of them is a causal measurement.
SAVINGS_BASES = {
    "observed_confounded": (
        "difference between two recorded executions; includes response variability and "
        "time effects, so it is not an ordering effect"
    ),
    "replay_estimate": (
        "offline replay of both orders over one resolved set of stored decisions; an "
        "estimate under stationarity and order-independence assumptions"
    ),
    "predicted": "planner prediction from profiling statistics",
    "bootstrap_interval": "percentile interval of predicted savings over profiling resamples",
    "structural_upper_bound": (
        "no order can save more than N times (sum of per-evaluation costs minus the cheapest); "
        "holds for any selectivities"
    ),
    "transferred_prediction": (
        "predicted saving of a comparable, already profiled workload, assumed to transfer"
    ),
}


def break_even_executions(profiling_cost: float, savings_per_execution: float) -> int | None:
    """Smallest reuse count whose cumulative savings cover profiling; None if never."""
    if profiling_cost < 0:
        raise ValueError("profiling cost cannot be negative")
    if profiling_cost == 0:
        return 0
    if savings_per_execution <= 0:
        return None
    ratio = profiling_cost / savings_per_execution
    nearest = round(ratio)
    # Savings are differences of float costs; do not round 80.0000000000005 up to 81.
    if math.isclose(ratio, nearest, rel_tol=1e-9):
        return int(nearest)
    return math.ceil(ratio)


def break_even_range(
    profiling_cost: float, savings_lower: float, savings_upper: float
) -> dict[str, int | None]:
    """Break-even across a savings interval; ``None`` means that end never breaks even."""
    if savings_lower > savings_upper:
        raise ValueError("savings interval lower bound exceeds upper bound")
    return {
        "fewest_executions": break_even_executions(profiling_cost, savings_upper),
        "most_executions": break_even_executions(profiling_cost, savings_lower),
    }


def _validate_basis(basis: str | None) -> None:
    if basis is not None and basis not in SAVINGS_BASES:
        raise ValueError(f"unknown savings basis: {basis}")


def profiling_cost_accounting(
    *,
    required_decisions: int,
    reusable_decisions: int,
    average_cost_per_decision: float | None,
    incurred_cost: float = 0.0,
) -> dict[str, Any]:
    """Split profiling into sunk cost, reusable (cached) decisions, and new spend.

    Decisions already stored for the same review, predicate hash, and model are not
    purchased again, so only the remainder is a prospective cost.
    """
    if required_decisions < 0 or reusable_decisions < 0:
        raise ValueError("decision counts cannot be negative")
    reusable = min(reusable_decisions, required_decisions)
    new_decisions = required_decisions - reusable
    new_cost = (
        new_decisions * average_cost_per_decision
        if average_cost_per_decision is not None
        else None
    )
    return {
        "required_decisions": required_decisions,
        "reusable_cached_decisions": reusable,
        "new_decisions": new_decisions,
        "average_cost_per_decision_usd": average_cost_per_decision,
        "new_profiling_cost_usd": new_cost,
        "sunk_profiling_cost_usd": incurred_cost,
    }


def decide_optimization(
    *,
    settings: EconomicsSettings,
    original_cost_per_execution: float | None,
    optimized_cost_per_execution: float | None = None,
    statistics_state: StatisticsState,
    new_profiling_cost: float | None = None,
    sunk_profiling_cost: float = 0.0,
    existing_plan_available: bool = False,
    maximum_savings_per_execution: float | None = None,
    savings_interval: tuple[float, float] | None = None,
    savings_probability: float | None = None,
    savings_basis: str | None = None,
) -> dict[str, Any]:
    """Decide whether optimizing is worth its forward-looking cost.

    Forward-looking totals exclude sunk profiling: money already spent cannot be recovered
    by declining to use the statistics. Retrospective totals that include sunk cost are
    reported separately and never drive the decision. ``savings_basis`` records where the
    per-execution saving came from (see ``SAVINGS_BASES``).
    """
    _validate_basis(savings_basis)
    horizon = settings.horizon()
    reasons: list[str] = []
    explanation: dict[str, Any] = {
        "schema_version": ECONOMICS_SCHEMA_VERSION,
        "savings_basis": savings_basis,
        "expected_executions": horizon,
        "expected_executions_known": settings.expected_executions is not None,
        "statistics_state": statistics_state,
        "statistics_reusable": settings.statistics_reusable,
        "existing_plan_available": existing_plan_available,
        "original_cost_per_execution_usd": original_cost_per_execution,
        "optimized_cost_per_execution_usd": optimized_cost_per_execution,
        "new_profiling_cost_usd": new_profiling_cost,
        "sunk_profiling_cost_usd": sunk_profiling_cost,
        "optimization_overhead_usd": settings.optimization_overhead_usd,
        "profiling_budget_usd": settings.profiling_budget_usd,
        "minimum_net_savings_usd": settings.minimum_net_savings_usd,
        "fixed_total_cost_usd": (
            horizon * original_cost_per_execution
            if original_cost_per_execution is not None
            else None
        ),
    }

    def finish(action: DecisionAction, status: str = "decided") -> dict[str, Any]:
        explanation.update({"action": action, "status": status, "reasons": reasons})
        return explanation

    if original_cost_per_execution is None:
        reasons.append("pricing_or_usage_unavailable")
        return finish("execute_original_without_profiling", "insufficient_information")

    if statistics_state == "none":
        if new_profiling_cost is None:
            reasons.append("profiling_cost_unknown")
            return finish("execute_original_without_profiling", "insufficient_information")
        explanation["optimized_total_cost_upper_bound_usd"] = (
            new_profiling_cost + settings.optimization_overhead_usd
            + horizon * original_cost_per_execution
        )
        if (
            settings.profiling_budget_usd is not None
            and new_profiling_cost > settings.profiling_budget_usd
        ):
            reasons.append("profiling_budget_exceeded")
            return finish("execute_original_without_profiling")
        if horizon == 0:
            reasons.append("no_expected_executions")
            return finish("execute_original_without_profiling")
        if maximum_savings_per_execution is not None:
            best_case_net = (
                horizon * maximum_savings_per_execution
                - new_profiling_cost
                - settings.optimization_overhead_usd
            )
            explanation["best_case_net_savings_usd"] = best_case_net
            explanation["best_case_break_even_executions"] = break_even_executions(
                new_profiling_cost + settings.optimization_overhead_usd,
                maximum_savings_per_execution,
            )
            if best_case_net <= settings.minimum_net_savings_usd:
                reasons.append("profiling_cannot_break_even_within_horizon")
                return finish("execute_original_without_profiling")
        else:
            reasons.append("savings_upper_bound_unknown")
        reasons.append("potential_savings_may_exceed_profiling_cost")
        return finish("collect_additional_statistics")

    if optimized_cost_per_execution is None:
        reasons.append("optimized_cost_unavailable")
        return finish("decline_optimization_execute_original", "insufficient_information")

    reuse = existing_plan_available and settings.statistics_reusable
    overhead = 0.0 if reuse else settings.optimization_overhead_usd
    savings = original_cost_per_execution - optimized_cost_per_execution
    net_forward = horizon * savings - overhead
    explanation.update(
        {
            "savings_per_execution_usd": savings,
            "forward_optimized_total_cost_usd": horizon * optimized_cost_per_execution
            + overhead,
            "forward_net_savings_usd": net_forward,
            "retrospective_optimized_total_including_sunk_profiling_usd": (
                sunk_profiling_cost + horizon * optimized_cost_per_execution + overhead
            ),
            "retrospective_net_savings_including_sunk_profiling_usd": (
                net_forward - sunk_profiling_cost
            ),
            # Each break-even charges the same overhead its net-savings total charges.
            "break_even_executions_for_sunk_profiling": break_even_executions(
                sunk_profiling_cost + overhead, savings
            ),
            "break_even_executions_for_new_comparable_profile": (
                break_even_executions(
                    new_profiling_cost + settings.optimization_overhead_usd, savings
                )
                if new_profiling_cost is not None
                else None
            ),
            "savings_interval_usd": list(savings_interval) if savings_interval else None,
            "savings_probability": savings_probability,
        }
    )
    if horizon == 0:
        reasons.append("no_expected_executions")
    elif savings <= 0:
        reasons.append("nonpositive_expected_savings")
    elif net_forward <= settings.minimum_net_savings_usd:
        reasons.append("net_savings_not_above_minimum")
    if settings.maximum_relative_uncertainty is not None:
        if savings_interval is None:
            reasons.append("savings_uncertainty_unavailable")
        elif savings > 0:
            relative = (savings_interval[1] - savings_interval[0]) / (2 * savings)
            explanation["relative_savings_uncertainty"] = relative
            if relative > settings.maximum_relative_uncertainty:
                reasons.append("savings_uncertainty_too_high")
    if settings.minimum_savings_probability is not None:
        if savings_probability is None:
            reasons.append("savings_probability_unavailable")
        elif savings_probability < settings.minimum_savings_probability:
            reasons.append("savings_probability_below_minimum")
    if reasons:
        return finish("decline_optimization_execute_original")
    reasons.append("forward_net_savings_positive")
    return finish("reuse_existing_plan" if reuse else "optimize_with_existing_statistics")


def decide_adaptive_profiling_start(
    *,
    expected_executions: int,
    first_checkpoint_decisions: int,
    first_checkpoint_cost: float | None,
    settings: EconomicsSettings | None = None,
    savings_upper_bound_per_execution: float | None = None,
    savings_bound_basis: str | None = None,
    statistics_available: bool = False,
    maximum_profiling_decisions: int | None = None,
    adaptive_budget_usd: float | None = None,
) -> dict[str, Any]:
    """Decide, before any profiling call, whether adaptive profiling should start at all.

    Adaptive profiling commits to at least its first checkpoint. The decision cap and
    budgets are always enforced on that commitment. With ``settings``, profiling is also
    skipped when reusable statistics exist, when no execution is expected, when cost or the
    savings bound is unknown, or when even the bounded saving over the whole horizon cannot
    cover the first checkpoint. Without ``settings`` only the caps apply. The horizon is
    ``expected_executions``; ``settings.expected_executions`` is not consulted.
    """
    if expected_executions < 0 or first_checkpoint_decisions < 0:
        raise ValueError("executions and decisions cannot be negative")
    _validate_basis(savings_bound_basis)
    overhead = settings.optimization_overhead_usd if settings is not None else 0.0
    reasons: list[str] = []
    explanation: dict[str, Any] = {
        "schema_version": ECONOMICS_SCHEMA_VERSION,
        "economics_gate_configured": settings is not None,
        "expected_executions": expected_executions,
        "first_checkpoint_decisions": first_checkpoint_decisions,
        "first_checkpoint_cost_usd": first_checkpoint_cost,
        "savings_upper_bound_per_execution_usd": savings_upper_bound_per_execution,
        "savings_bound_basis": savings_bound_basis,
        "optimization_overhead_usd": overhead,
    }

    def finish(action: AdaptiveStartAction, status: str = "decided") -> dict[str, Any]:
        explanation.update({"action": action, "status": status, "reasons": reasons})
        return explanation

    if settings is not None and statistics_available and settings.statistics_reusable:
        reasons.append("reusable_statistics_available")
        return finish("skip_profiling_use_existing_statistics")
    if settings is not None and expected_executions == 0:
        reasons.append("no_expected_executions")
        return finish("skip_profiling_execute_original")
    if maximum_profiling_decisions is not None and (
        first_checkpoint_decisions > maximum_profiling_decisions
    ):
        reasons.append("first_checkpoint_exceeds_decision_cap")
    budgets = [
        value
        for value in (
            adaptive_budget_usd,
            settings.profiling_budget_usd if settings is not None else None,
        )
        if value is not None
    ]
    if budgets and first_checkpoint_cost is not None and first_checkpoint_cost > min(budgets):
        reasons.append("first_checkpoint_exceeds_profiling_budget")
    if reasons:
        return finish("skip_profiling_execute_original")
    if settings is None:
        reasons.append("economics_gate_not_configured")
        return finish("start_adaptive_profiling")
    if first_checkpoint_cost is None:
        reasons.append("profiling_cost_unknown")
        return finish("skip_profiling_execute_original", "insufficient_information")
    if savings_upper_bound_per_execution is None:
        reasons.append("savings_upper_bound_unknown")
        return finish("skip_profiling_execute_original", "insufficient_information")
    best_case_net = (
        expected_executions * savings_upper_bound_per_execution - first_checkpoint_cost - overhead
    )
    explanation["best_case_net_savings_usd"] = best_case_net
    explanation["best_case_break_even_executions"] = break_even_executions(
        first_checkpoint_cost + overhead, savings_upper_bound_per_execution
    )
    if best_case_net <= settings.minimum_net_savings_usd:
        reasons.append("profiling_cannot_break_even_within_horizon")
        return finish("skip_profiling_execute_original")
    reasons.append("bounded_savings_may_exceed_first_checkpoint_cost")
    return finish("start_adaptive_profiling")
