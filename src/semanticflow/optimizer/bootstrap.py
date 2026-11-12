from __future__ import annotations

import itertools
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from semanticflow.optimizer.cardinality import build_estimator, predicate_statistics_from_matrix
from semanticflow.optimizer.cost_model import estimate_plan
from semanticflow.optimizer.matrix import MatrixRow, ProfileMatrix
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


def sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((item - mean) ** 2 for item in values) / (len(values) - 1))


def _summary(values: Sequence[float], level: float) -> dict[str, float]:
    tail = (1 - level) / 2
    return {
        "mean": sum(values) / len(values),
        "median": quantile(values, 0.5),
        "std": sample_std(values),
        "percentile_05": quantile(values, 0.05),
        "percentile_95": quantile(values, 0.95),
        "lower": quantile(values, tail),
        "upper": quantile(values, 1 - tail),
        "min": min(values),
        "max": max(values),
    }


def collect_paired_bootstrap(
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
    iterations: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Resample complete review rows and score every candidate order on each draw.

    Predicate columns are never resampled independently. Token and cost observations
    travel with the row, so usage variation is part of the same sampling distribution.
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
    count = iterations if iterations is not None else settings.iterations
    used_seed = settings.seed if seed is None else seed
    rng = random.Random(used_seed)
    size = matrix.size
    orders = tuple(sorted(full_costs))
    draws: dict[tuple[str, ...], list[float]] = {order: [] for order in orders}
    selected_orders: list[tuple[str, ...]] = []
    for _ in range(count):
        indices = [min(int(rng.random() * size), size - 1) for _ in range(size)]
        costs = order_costs(matrix.take(indices), **common)
        selected_orders.append(select_minimum(costs, original, tie_relative_tolerance))
        for order in orders:
            draws[order].append(costs[order])
    return {
        "estimator": estimator_name,
        "objective": objective,
        "iterations": count,
        "seed": used_seed,
        "sample_size": size,
        "evaluation_record_count": evaluation_record_count,
        "interval_level": settings.interval_level,
        "original_order": original,
        "orders": orders,
        "full_sample_costs": full_costs,
        "full_sample_selected_order": full_selected,
        "draws": draws,
        "selected_orders": selected_orders,
        "tie_relative_tolerance": tie_relative_tolerance,
        "interpretation": "sampling_uncertainty_diagnostic",
        "not_evidence_of": "real-world generalization to other data, prompts, or model versions",
        "resampling_unit": "review_row_with_all_predicate_outcomes_and_usage",
        "random_generator": "python_random_mersenne_twister_random_only",
    }


def collect_sparse_posterior_bootstrap(
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
    prior_strength: float = 1.0,
    tie_relative_tolerance: float = 1e-9,
    iterations: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Posterior-predictive row bootstrap with support for unseen binary outcomes.

    A plain empirical bootstrap is degenerate for a predicate observed to pass zero times.
    This approximation places a Dirichlet prior over the complete binary outcome-pattern
    distribution.  Its base measure is the product of Jeffreys-smoothed marginal means, so
    every pattern has positive mass while observed joint patterns still dominate as evidence
    accumulates.  Pattern draws are statistical simulations, never recorded Jev responses.

    Usage is sampled from an observed row with the same pattern when one exists and otherwise
    from any currently observed row.  Consequently the method represents uncertainty in
    selectivity and observed correlations, but it does not identify response variability and
    only approximates outcome/usage dependence for unseen patterns.
    """
    if not matrix.complete:
        raise ValueError("sparse posterior bootstrap requires a complete profiling matrix")
    if prior_strength <= 0:
        raise ValueError("posterior predictive prior strength must be positive")
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
    count = iterations if iterations is not None else settings.iterations
    used_seed = settings.seed if seed is None else seed
    rng = random.Random(used_seed)
    size = matrix.size
    width = len(matrix.predicate_ids)
    patterns = tuple(itertools.product((False, True), repeat=width))
    observed_counts = matrix.pattern_counts()
    marginal_means = []
    for index in range(width):
        passing, support = matrix.count(1 << index)
        marginal_means.append((passing + 0.5) / (support + 1.0))
    base_mass = []
    for pattern in patterns:
        probability = 1.0
        for value, marginal in zip(pattern, marginal_means, strict=True):
            probability *= marginal if value else 1.0 - marginal
        base_mass.append(probability)
    alphas = [
        observed_counts.get(pattern, 0) + prior_strength * base
        for pattern, base in zip(patterns, base_mass, strict=True)
    ]
    rows_by_pattern: dict[tuple[bool, ...], list[MatrixRow]] = {
        pattern: [] for pattern in patterns
    }
    for row in matrix.rows:
        pattern = tuple(bool(value) for value in row.outcomes)
        rows_by_pattern[pattern].append(row)

    orders = tuple(sorted(full_costs))
    draws: dict[tuple[str, ...], list[float]] = {order: [] for order in orders}
    selected_orders: list[tuple[str, ...]] = []
    for draw_index in range(count):
        weights = [rng.gammavariate(alpha, 1.0) for alpha in alphas]
        synthetic_rows = []
        sampled_patterns = rng.choices(patterns, weights=weights, k=size)
        for row_index, pattern in enumerate(sampled_patterns):
            pool = rows_by_pattern[pattern] or list(matrix.rows)
            usage = pool[rng.randrange(len(pool))]
            synthetic_rows.append(
                MatrixRow(
                    review_id=f"posterior-{draw_index}-{row_index}",
                    outcomes=pattern,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    latency_seconds=usage.latency_seconds,
                )
            )
        synthetic = ProfileMatrix(matrix.predicate_ids, synthetic_rows, resampled=True)
        costs = order_costs(synthetic, **common)
        selected_orders.append(select_minimum(costs, original, tie_relative_tolerance))
        for order in orders:
            draws[order].append(costs[order])
    return {
        "estimator": estimator_name,
        "objective": objective,
        "iterations": count,
        "seed": used_seed,
        "sample_size": size,
        "evaluation_record_count": evaluation_record_count,
        "interval_level": settings.interval_level,
        "original_order": original,
        "orders": orders,
        "full_sample_costs": full_costs,
        "full_sample_selected_order": full_selected,
        "draws": draws,
        "selected_orders": selected_orders,
        "tie_relative_tolerance": tie_relative_tolerance,
        "interpretation": "sparse_posterior_predictive_uncertainty_approximation",
        "not_evidence_of": (
            "actual unseen Jev responses, response variability, or generalization to other data"
        ),
        "resampling_unit": "simulated_complete_outcome_pattern_with_observed_usage_row",
        "random_generator": "python_random_mersenne_twister_random_only",
        "sparse_data_treatment": {
            "method": "dirichlet_outcome_patterns",
            "prior_strength": prior_strength,
            "base_measure": "product_of_jeffreys_smoothed_marginal_means",
            "observed_pattern_count": len(observed_counts),
            "possible_pattern_count": len(patterns),
            "draws_are_actual_provider_responses": False,
            "limitations": (
                "captures finite-profile selectivity and joint-pattern uncertainty; "
                "unseen-pattern usage is backed off to all observed rows; model-response "
                "variability is not identified"
            ),
        },
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
    collected = collect_paired_bootstrap(
        matrix,
        estimator_name=estimator_name,
        original_order=original_order,
        objective=objective,
        evaluation_record_count=evaluation_record_count,
        question_hashes=question_hashes,
        settings=settings,
        correlation=correlation,
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
        tie_relative_tolerance=tie_relative_tolerance,
        iterations=iterations,
        seed=seed,
    )
    original = collected["original_order"]
    full_costs = collected["full_sample_costs"]
    full_selected = collected["full_sample_selected_order"]
    reference = tuple(reference_order) if reference_order is not None else full_selected
    count = collected["iterations"]
    draws = collected["draws"]
    selections: Counter[tuple[str, ...]] = Counter(collected["selected_orders"])
    savings = [
        draws[original][index] - draws[reference][index] for index in range(count)
    ]
    relative_savings = [
        value / draws[original][index] if draws[original][index] else 0.0
        for index, value in enumerate(savings)
    ]
    regrets = [
        draws[reference][index] - min(draws[order][index] for order in collected["orders"])
        for index in range(count)
    ]
    reference_beats_original = sum(
        draws[reference][index] < draws[original][index] for index in range(count)
    )
    frequency = {
        order_key(order): selections.get(order, 0) / count for order in collected["orders"]
    }
    modal_order, modal_count = max(sorted(selections.items()), key=lambda item: item[1])
    stable = (
        modal_order == reference and modal_count / count >= settings.stability_threshold
    )
    return {
        "schema_version": BOOTSTRAP_SCHEMA_VERSION,
        "interpretation": collected["interpretation"],
        "not_evidence_of": collected["not_evidence_of"],
        "resampling_unit": collected["resampling_unit"],
        "estimator": estimator_name,
        "objective": objective,
        "iterations": count,
        "seed": collected["seed"],
        "random_generator": collected["random_generator"],
        "sample_size": collected["sample_size"],
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
            for order, values in sorted(draws.items())
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
