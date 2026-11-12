"""Independence vs correlation-aware disagreement as a decision-uncertainty signal.

Agreement does not prove correctness. Disagreement does not prove either estimator is wrong.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from semanticflow.optimizer.bootstrap import _summary, order_key

DISAGREEMENT_SCHEMA_VERSION = "m4-estimator-disagreement-v1"


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _interval_overlap(left: Mapping[str, float], right: Mapping[str, float]) -> bool:
    return not (left["upper"] < right["lower"] or right["upper"] < left["lower"])


def compare_estimators(
    *,
    independent: Mapping[tuple[str, ...], Sequence[float]],
    correlation_aware: Mapping[tuple[str, ...], Sequence[float]],
    original_order: Sequence[str],
    independent_selected: Sequence[str],
    correlation_selected: Sequence[str],
    interval_level: float,
) -> dict[str, Any]:
    """Compare two paired-bootstrap cost distributions that share resample indices."""
    original = tuple(original_order)
    independent_order = tuple(independent_selected)
    correlation_order = tuple(correlation_selected)
    if set(independent) != set(correlation_aware):
        raise ValueError("estimator draws must cover the same candidate orders")
    count = len(independent[original])
    if any(len(correlation_aware[order]) != count for order in independent):
        raise ValueError("estimator draws must be aligned by bootstrap iteration")
    by_order: dict[str, Any] = {}
    for order in sorted(independent):
        left = list(independent[order])
        right = list(correlation_aware[order])
        left_summary = _summary(left, interval_level)
        right_summary = _summary(right, interval_level)
        difference = [left[index] - right[index] for index in range(count)]
        left_saving = [independent[original][index] - left[index] for index in range(count)]
        right_saving = [
            correlation_aware[original][index] - right[index] for index in range(count)
        ]
        by_order[order_key(order)] = {
            "independent_cost": left_summary,
            "correlation_aware_cost": right_summary,
            "expected_cost_difference_independent_minus_correlation": _mean(difference),
            "cost_intervals_overlap": _interval_overlap(left_summary, right_summary),
            "independent_expected_saving_vs_original": _mean(left_saving),
            "correlation_aware_expected_saving_vs_original": _mean(right_saving),
            "agree_on_saving_sign": (
                (_mean(left_saving) > 0) == (_mean(right_saving) > 0)
            ),
            "one_predicts_loss_other_gain": (
                (_mean(left_saving) > 0) != (_mean(right_saving) > 0)
            ),
        }
    same_order = independent_order == correlation_order
    return {
        "schema_version": DISAGREEMENT_SCHEMA_VERSION,
        "interpretation": (
            "estimator disagreement is a diagnostic, not a guarantee of error or of safety"
        ),
        "independent_selected_order": list(independent_order),
        "correlation_aware_selected_order": list(correlation_order),
        "select_same_order": same_order,
        "agree_original_is_minimum": independent_order == original
        and correlation_order == original,
        "agree_original_is_worse": independent_order != original
        and correlation_order != original,
        "by_order": by_order,
        "interval_level": interval_level,
        "iterations": count,
    }
