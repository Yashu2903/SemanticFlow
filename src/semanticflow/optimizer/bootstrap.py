from __future__ import annotations

import itertools
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from semanticflow.optimizer.cardinality import build_estimator, predicate_statistics_from_matrix
from semanticflow.optimizer.cost_model import estimate_plan
from semanticflow.optimizer.matrix import ProfileMatrix
from semanticflow.optimizer.models import (
    BootstrapSettings,
    CorrelationSettings,
    OptimizationMetric,
)

BOOTSTRAP_SCHEMA_VERSION = "m3-paired-bootstrap-v1"


def order_key(order: Sequence[str]) -> str:
    return " -> ".join(order)


def quantile(values: Sequence[float], probability: float) -> float:
    """Linear-interpolation quantile (type 7); deterministic for sorted input."""
    if not values:
        raise ValueError("quantile requires values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def select_minimum(
    costs: Mapping[tuple[str, ...], float],
    original: tuple[str, ...],
    tie_relative_tolerance: float,
) -> tuple[str, ...]:
    """The planner's rule: minimum objective, original order on ties, then lexicographic."""
    minimum = min(costs.values())
    tolerance = tie_relative_tolerance * max(1.0, abs(minimum))
    tied = [
        order
        for order, value in costs.items()
        if math.isclose(value, minimum, rel_tol=tie_relative_tolerance, abs_tol=tolerance)
    ]
    return original if original in tied else min(tied)


def order_costs(
    matrix: ProfileMatrix,
    *,
    estimator_name: str,
    original_order: Sequence[str],
    objective: OptimizationMetric,
    evaluation_record_count: int,
    question_hashes: Mapping[str, str],
    correlation: CorrelationSettings,
    input_price_per_million: float | None,
    output_price_per_million: float | None,
) -> dict[tuple[str, ...], float]:
    statistics = predicate_statistics_from_matrix(
        matrix,
        question_hashes=question_hashes,
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
    )
    estimator = build_estimator(
        estimator_name,
        matrix,
        question_hashes=question_hashes,
        correlation=correlation,
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
    )
    return {
        order: estimate_plan(
            order,
            statistics,
            input_record_count=evaluation_record_count,
            objective=objective,
            original_order=original_order,
            estimator=estimator,
        ).objective_value
        for order in itertools.permutations(original_order)
    }


def _summary(values: Sequence[float], level: float) -> dict[str, float]:
    tail = (1 - level) / 2
    return {
        "mean": sum(values) / len(values),
        "lower": quantile(values, tail),
        "median": quantile(values, 0.5),
        "upper": quantile(values, 1 - tail),
    }


def paired_bootstrap(
    matrix: ProfileMatrix,
    *,
    estimator_name: str,
    original_order: Sequence[str],
    objective: OptimizationMetric,
    evaluation_record_count: int,
    question_hashes: Mapping[str, str],
    settings: BootstrapSettings,
    correlation: CorrelationSettings,
    input_price_per_million: float | None,
    output_price_per_million: float | None,
    tie_relative_tolerance: float = 1e-9,
    reference_order: Sequence[str] | None = None,
    iterations: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Resample whole review rows with replacement and re-plan on every resample.

    Rows keep all of their predicate outcomes and usage together, preserving correlations.
    Results describe sampling uncertainty of the profile only; they do not establish that a
    plan generalizes to other data, prompts, or model versions.
    """
    if not matrix.complete:
        raise ValueError("paired bootstrap requires a complete profiling matrix")
    original = tuple(original_order)
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
    full_selected = select_minimum(full_costs, original, tie_relative_tolerance)
    reference = tuple(reference_order) if reference_order is not None else full_selected
    count = iterations if iterations is not None else settings.iterations
    rng = random.Random(settings.seed if seed is None else seed)
    size = matrix.size
    selections: Counter[tuple[str, ...]] = Counter()
    per_order: dict[tuple[str, ...], list[float]] = {order: [] for order in full_costs}
    savings: list[float] = []
    relative_savings: list[float] = []
    regrets: list[float] = []
    reference_beats_original = 0
    for _ in range(count):
        # random() is the only generator output Python guarantees across versions.
        indices = [min(int(rng.random() * size), size - 1) for _ in range(size)]
        costs = order_costs(matrix.take(indices), **common)
        best = select_minimum(costs, original, tie_relative_tolerance)
        selections[best] += 1
        for order, value in costs.items():
            per_order[order].append(value)
        saving = costs[original] - costs[reference]
        savings.append(saving)
        relative_savings.append(saving / costs[original] if costs[original] else 0.0)
        regrets.append(costs[reference] - costs[best])
        reference_beats_original += costs[reference] < costs[original]
    frequency = {
        order_key(order): selections.get(order, 0) / count for order in sorted(full_costs)
    }
    modal_order, modal_count = max(
        sorted(selections.items()), key=lambda item: item[1]
    )
    stable = (
        modal_order == reference and modal_count / count >= settings.stability_threshold
    )
    return {
        "schema_version": BOOTSTRAP_SCHEMA_VERSION,
        "interpretation": "sampling_uncertainty_diagnostic",
        "not_evidence_of": "real-world generalization to other data, prompts, or model versions",
        "resampling_unit": "review_row_with_all_predicate_outcomes_and_usage",
        "estimator": estimator_name,
        "objective": objective,
        "iterations": count,
        "seed": settings.seed if seed is None else seed,
        "random_generator": "python_random_mersenne_twister_random_only",
        "sample_size": size,
        "evaluation_record_count": evaluation_record_count,
        "interval_level": settings.interval_level,
        "original_order": list(original),
        "full_sample_selected_order": list(full_selected),
        "reference_order": list(reference),
        "full_sample_objective_by_order": {
            order_key(order): value for order, value in sorted(full_costs.items())
        },
        "selection_frequency": frequency,
        "probability_cheapest": frequency,
        "modal_order": list(modal_order),
        "modal_frequency": modal_count / count,
        "stability_threshold": settings.stability_threshold,
        "stable_selection": stable,
        "unstable_selection_evidence": None
        if stable
        else (
            f"reference order wins {selections.get(reference, 0) / count:.3f} of resamples; "
            f"threshold is {settings.stability_threshold}"
        ),
        "reference_savings_vs_original": _summary(savings, settings.interval_level),
        "reference_relative_savings_vs_original": _summary(
            relative_savings, settings.interval_level
        ),
        "probability_reference_cheaper_than_original": reference_beats_original / count,
        "probability_reference_not_more_expensive": sum(value >= 0 for value in savings) / count,
        "reference_regret": {
            **_summary(regrets, settings.interval_level),
            "probability_positive": sum(value > 0 for value in regrets) / count,
            "definition": "objective(reference) - minimum objective, per resample",
        },
        "objective_distribution_by_order": {
            order_key(order): _summary(values, settings.interval_level)
            for order, values in sorted(per_order.items())
        },
    }


def confidence_aware_choice(
    *,
    minimum_order: Sequence[str],
    original_order: Sequence[str],
    minimum_objective: float,
    original_objective: float,
    bootstrap: Mapping[str, Any],
    settings: BootstrapSettings,
) -> tuple[list[str], dict[str, Any]]:
    """Keep the original order unless the cheaper order is both credible and material.

    The cheaper order is adopted only when (1) it is cheaper than the original in at least
    ``confidence_level`` of paired resamples (a one-sided percentile bootstrap criterion) and
    (2) its full-sample relative saving is at least ``minimum_relative_saving``.
    """
    minimum = list(minimum_order)
    original = list(original_order)
    explanation: dict[str, Any] = {
        "policy": "confidence_aware",
        "confidence_level": settings.confidence_level,
        "minimum_relative_saving": settings.minimum_relative_saving,
        "minimum_objective_order": minimum,
        "original_order": original,
    }
    if minimum == original:
        explanation.update({"decision": "original_is_minimum", "selected_order": original})
        return original, explanation
    if list(bootstrap["reference_order"]) != minimum:
        raise ValueError("bootstrap reference order must be the minimum-objective order")
    probability = float(bootstrap["probability_reference_cheaper_than_original"])
    relative = (
        (original_objective - minimum_objective) / original_objective
        if original_objective
        else 0.0
    )
    reasons = []
    if probability < settings.confidence_level:
        reasons.append("savings_probability_below_confidence_level")
    if relative < settings.minimum_relative_saving:
        reasons.append("relative_saving_below_practical_threshold")
    selected = original if reasons else minimum
    explanation.update(
        {
            "probability_minimum_cheaper_than_original": probability,
            "full_sample_relative_saving": relative,
            "decision": "retain_original" if reasons else "adopt_minimum",
            "reasons": reasons,
            "selected_order": selected,
        }
    )
    return selected, explanation
