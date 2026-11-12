"""Cost-distribution summaries and labeled uncertainty sources.

Sampling uncertainty is estimated by paired bootstrap. Model-response variability is reported
only as a sensitivity analysis when repeated traces exist. The two are not treated as
independent additive noise, and a response distribution is never inferred from a handful of
threshold crossings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from semanticflow.optimizer.bootstrap import _summary, order_key
from semanticflow.optimizer.regret import metrics_by_order

UNCERTAINTY_SCHEMA_VERSION = "m4-uncertainty-v1"
VARIABILITY_SCHEMA_VERSION = "m4-response-variability-v1"


def cost_distributions(
    draws: Mapping[tuple[str, ...], Sequence[float]],
    *,
    original_order: Sequence[str],
    interval_level: float,
    cvar_level: float,
) -> dict[str, Any]:
    original = tuple(original_order)
    summaries = {
        order_key(order): _summary(values, interval_level)
        for order, values in sorted(draws.items())
    }
    return {
        "schema_version": UNCERTAINTY_SCHEMA_VERSION,
        "interpretation": "sampling_uncertainty_diagnostic",
        "not_a_calibrated_production_probability": True,
        "interval_level": interval_level,
        "original_order": list(original),
        "by_order": summaries,
        "risk_by_order": metrics_by_order(
            draws,
            original_order=original,
            interval_level=interval_level,
            cvar_level=cvar_level,
        ),
        "sources": {
            "sampling_across_reviews": (
                "paired bootstrap of complete profiling rows, preserving all predicate "
                "outcomes and usage for each review"
            ),
            "selectivity_estimation": (
                "each resample refits marginals, joints, and prefix conditionals; "
                "this is statistical uncertainty of the profile, "
                "not a posterior over the population"
            ),
            "per_request_cost_and_latency": (
                "token and latency observations are resampled with their review row; "
                "an average token cost is not treated as certain"
            ),
            "model_response_variability": (
                "not folded into these draws; see response_variability_sensitivity"
            ),
        },
    }


def response_variability_sensitivity(
    live_analysis: Mapping[str, Any] | None,
    *,
    extra_repeated_sources: int = 0,
) -> dict[str, Any]:
    """Report repeated-Jev sensitivity without inventing a generative noise model."""
    report: dict[str, Any] = {
        "schema_version": VARIABILITY_SCHEMA_VERSION,
        "mode": "sensitivity_analysis",
        "not_a_probabilistic_response_model": True,
        "sources_are_not_assumed_independent": True,
    }
    if not live_analysis or "orders" not in live_analysis:
        report.update(
            {
                "status": "insufficient_repeated_trace_evidence",
                "reason": "no contemporaneous replicate runs were supplied",
                "extra_repeated_sources": extra_repeated_sources,
            }
        )
        return report
    rows = []
    for order, payload in live_analysis["orders"].items():
        variability = payload.get("variability") or {}
        pairs = variability.get("pairs") or []
        crossings = [item.get("threshold_crossings") for item in pairs]
        rows.append(
            {
                "order": order,
                "replicates": payload.get("replicates"),
                "replicate_costs_usd": variability.get("replicate_costs_usd"),
                "replicate_standard_deviation_usd": variability.get(
                    "replicate_standard_deviation_usd"
                ),
                "threshold_crossings": crossings,
                "decision_differences": [item.get("decision_difference") for item in pairs],
                "cost_differences_usd": [item.get("cost_difference_usd") for item in pairs],
            }
        )
    total_crossings = sum(
        value for row in rows for value in row["threshold_crossings"] if isinstance(value, int)
    )
    report.update(
        {
            "status": "observed_replicate_sensitivity",
            "orders": rows,
            "total_recorded_threshold_crossings": total_crossings,
            "limitation": (
                "replicate differences mix response variability with residual time effects; "
                "they are not used as an independent noise term in the bootstrap"
            ),
        }
    )
    if total_crossings < 20:
        report["small_sample_warning"] = (
            "too few threshold crossings to fit a response-flip distribution"
        )
    return report
