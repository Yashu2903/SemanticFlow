"""Expected regret and downside-risk metrics over paired bootstrap draws.

Regret is always relative to the cheapest modeled candidate on that resample, not the
unknown true optimum. Downside loss is relative to the original query-file order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from semanticflow.optimizer.bootstrap import _summary, order_key, quantile, sample_std

REGRET_SCHEMA_VERSION = "m4-regret-v1"


def cvar(values: Sequence[float], level: float) -> dict[str, object]:
    """Upper-tail Conditional Value at Risk: mean of values at or above the ``level`` quantile.

    For loss L = C(plan) - C(original), CVaR_0.95 is the mean of the worst 5% of draws.
    A positive value means the candidate is more expensive than the original in that tail.
    """
    if not values:
        raise ValueError("cvar requires values")
    if not 0.0 < level < 1.0:
        raise ValueError("cvar level must lie in (0, 1)")
    ordered = sorted(values)
    var = quantile(ordered, level)
    tail = [item for item in ordered if item >= var]
    if not tail:
        tail = [ordered[-1]]
    return {
        "level": level,
        "var": var,
        "cvar": sum(tail) / len(tail),
        "tail_count": len(tail),
        "definition": (
            f"mean of draws at or above the {level:.2f} quantile "
            "(upper tail; higher loss is worse)"
        ),
    }


def _aligned_costs(
    draws: Mapping[tuple[str, ...], Sequence[float]],
) -> tuple[tuple[tuple[str, ...], ...], int]:
    orders = tuple(sorted(draws))
    if not orders:
        raise ValueError("regret requires at least one candidate")
    count = len(draws[orders[0]])
    if count == 0:
        raise ValueError("regret requires bootstrap draws")
    if any(len(draws[order]) != count for order in orders):
        raise ValueError("bootstrap draws must be aligned across candidate plans")
    return orders, count


def plan_risk_metrics(
    draws: Mapping[tuple[str, ...], Sequence[float]],
    *,
    original_order: Sequence[str],
    candidate_order: Sequence[str],
    interval_level: float,
    cvar_level: float,
) -> dict[str, object]:
    """Risk of one candidate relative to the modeled set and to the original order."""
    original = tuple(original_order)
    candidate = tuple(candidate_order)
    orders, count = _aligned_costs(draws)
    if original not in draws or candidate not in draws:
        raise ValueError("candidate and original orders must appear in the bootstrap draws")
    original_costs = list(draws[original])
    candidate_costs = list(draws[candidate])
    savings = [original_costs[index] - candidate_costs[index] for index in range(count)]
    losses = [-value for value in savings]
    relative = [
        savings[index] / original_costs[index] if original_costs[index] else 0.0
        for index in range(count)
    ]
    modeled_min = [
        min(draws[order][index] for order in orders) for index in range(count)
    ]
    regret = [candidate_costs[index] - modeled_min[index] for index in range(count)]
    original_regret = [original_costs[index] - modeled_min[index] for index in range(count)]
    return {
        "schema_version": REGRET_SCHEMA_VERSION,
        "candidate_order": list(candidate),
        "original_order": list(original),
        "iterations": count,
        "predicted_cost": _summary(candidate_costs, interval_level),
        "predicted_saving_vs_original": _summary(savings, interval_level),
        "predicted_relative_saving_vs_original": _summary(relative, interval_level),
        "probability_positive_saving": sum(value > 0 for value in savings) / count,
        "probability_cheaper_than_original": sum(
            candidate_costs[index] < original_costs[index] for index in range(count)
        )
        / count,
        "expected_regret_vs_modeled_minimum": _summary(regret, interval_level),
        "probability_positive_regret": sum(value > 0 for value in regret) / count,
        "maximum_bootstrap_regret": max(regret),
        "original_regret_vs_modeled_minimum": _summary(original_regret, interval_level),
        "downside_loss_vs_original": _summary(losses, interval_level),
        "cvar_loss_vs_original": cvar(losses, cvar_level),
        "regret_scope": (
            "cheapest candidate among the modeled permutations on each resample; "
            "not regret against an unknown true optimum or unmodeled orders"
        ),
        "interpretation": "sampling_uncertainty_diagnostic",
        "cost_std": sample_std(candidate_costs),
    }


def metrics_by_order(
    draws: Mapping[tuple[str, ...], Sequence[float]],
    *,
    original_order: Sequence[str],
    interval_level: float,
    cvar_level: float,
) -> dict[str, dict[str, object]]:
    return {
        order_key(order): plan_risk_metrics(
            draws,
            original_order=original_order,
            candidate_order=order,
            interval_level=interval_level,
            cvar_level=cvar_level,
        )
        for order in sorted(draws)
    }
