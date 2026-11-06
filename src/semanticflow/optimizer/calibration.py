from __future__ import annotations

import hashlib
import itertools
import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

from semanticflow.optimizer.bootstrap import order_key, quantile
from semanticflow.optimizer.cardinality import (
    CorrelationAwareEstimator,
    IndependentEstimator,
    build_estimator,
    exact_matrix_cardinalities,
)
from semanticflow.optimizer.matrix import ProfileMatrix
from semanticflow.optimizer.models import CorrelationSettings
from semanticflow.optimizer.selectivity import wilson_interval

CALIBRATION_SCHEMA_VERSION = "m3-cardinality-calibration-v1"
INTERVAL_LEVELS = (0.5, 0.9)
ESTIMATOR_NAMES = ("independent", "correlation_aware")

PREDICTIVE_METHOD = {
    "parameter_uncertainty": "refit the estimator on paired bootstrap resamples of profiling rows",
    "outcome_sampling": (
        "sequential binomial draws of stage survivors on the target record count, using the "
        "resampled chain conditionals"
    ),
    "pit": "mid-PIT: P(draw < observed) + 0.5 * P(draw = observed); uniform when calibrated",
    "caveat": (
        "Checks share records and orders, so they are not independent; coverage rates are "
        "descriptive and carry wide sampling error."
    ),
}


def chain_conditionals(
    estimator: IndependentEstimator | CorrelationAwareEstimator, order: Sequence[str]
) -> list[float]:
    """P(stage k passes | every earlier stage passed), for every stage including the last.

    Products of these values reproduce the estimator's stage reach probabilities.
    """
    if isinstance(estimator, IndependentEstimator):
        values = []
        for predicate_id in order:
            selectivity = estimator.statistics[predicate_id].estimated_selectivity
            if selectivity is None:
                raise ValueError(f"unknown selectivity for predicate {predicate_id}")
            values.append(float(selectivity))
        return values
    indices = [estimator.matrix.index(predicate_id) for predicate_id in order]
    return [
        estimator.conditional(index, indices[:position])[0]
        for position, index in enumerate(indices)
    ]


def quantity_names(order: Sequence[str]) -> list[str]:
    """Stage 1 always receives every record, so only later stages and matches are random."""
    return [f"stage_{stage}_input_records" for stage in range(2, len(order) + 1)] + [
        "final_matches"
    ]


def expected_counts(conditionals: Sequence[float], record_count: int) -> list[float]:
    counts = []
    current = float(record_count)
    for value in conditionals:
        current *= value
        counts.append(current)
    return counts


def observed_matrix_counts(matrix: ProfileMatrix, order: Sequence[str]) -> list[int]:
    """Stage inputs after stage 1 and final matches from exact replay of a complete matrix."""
    return exact_matrix_cardinalities(matrix, order)[1:] + [matrix.count(matrix.full_mask)[0]]


def _binomial(rng: random.Random, trials: int, probability: float) -> int:
    if probability <= 0.0:
        return 0
    if probability >= 1.0:
        return trials
    return sum(rng.random() < probability for _ in range(trials))


def _estimator(
    matrix: ProfileMatrix,
    name: str,
    question_hashes: Mapping[str, str],
    correlation: CorrelationSettings,
) -> IndependentEstimator | CorrelationAwareEstimator:
    return build_estimator(
        name,
        matrix,
        question_hashes=question_hashes,
        correlation=correlation,
        input_price_per_million=None,
        output_price_per_million=None,
    )


def predictive_draws(
    matrix: ProfileMatrix,
    orders: Sequence[Sequence[str]],
    *,
    estimator_name: str,
    question_hashes: Mapping[str, str],
    correlation: CorrelationSettings,
    record_count: int,
    iterations: int,
    seed: int,
) -> dict[tuple[str, ...], list[list[int]]]:
    """Posterior-predictive-style draws of stage counts on ``record_count`` new records."""
    if not matrix.complete:
        raise ValueError("calibration requires a complete profiling matrix")
    if iterations < 1:
        raise ValueError("calibration requires at least one iteration")
    rng = random.Random(seed)
    size = matrix.size
    keys = [tuple(order) for order in orders]
    draws: dict[tuple[str, ...], list[list[int]]] = {key: [] for key in keys}
    for _ in range(iterations):
        # random() is the only generator output Python guarantees across versions.
        indices = [min(int(rng.random() * size), size - 1) for _ in range(size)]
        estimator = _estimator(matrix.take(indices), estimator_name, question_hashes, correlation)
        for key in keys:
            current = record_count
            counts = []
            for value in chain_conditionals(estimator, key):
                current = _binomial(rng, current, value)
                counts.append(current)
            draws[key].append(counts)
    return draws


def _mid_pit(values: Sequence[int], observed: float) -> float:
    below = sum(value < observed for value in values)
    equal = sum(value == observed for value in values)
    return (below + 0.5 * equal) / len(values)


def _level_key(level: float) -> str:
    return str(round(level * 100))


def calibrate_orders(
    matrix: ProfileMatrix,
    observations: Mapping[tuple[str, ...], Sequence[float]],
    *,
    estimator_name: str,
    question_hashes: Mapping[str, str],
    correlation: CorrelationSettings,
    record_count: int,
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Compare predictive distributions fitted on ``matrix`` with observed counts.

    ``observations`` maps an order to its observed stage inputs after stage 1 followed by
    the final match count, on ``record_count`` records that ``matrix`` does not contain.
    """
    orders = list(observations)
    fitted = _estimator(matrix, estimator_name, question_hashes, correlation)
    draws = predictive_draws(
        matrix,
        orders,
        estimator_name=estimator_name,
        question_hashes=question_hashes,
        correlation=correlation,
        record_count=record_count,
        iterations=iterations,
        seed=seed,
    )
    records = []
    for order in orders:
        observed = [float(value) for value in observations[order]]
        names = quantity_names(order)
        if len(observed) != len(names):
            raise ValueError(f"order {order_key(order)} needs {len(names)} observed counts")
        conditionals = chain_conditionals(fitted, order)
        points = expected_counts(conditionals, record_count)
        quantities = []
        for position, name in enumerate(names):
            values = [draw[position] for draw in draws[order]]
            intervals = {}
            covered = {}
            for level in INTERVAL_LEVELS:
                tail = (1 - level) / 2
                lower, upper = quantile(values, tail), quantile(values, 1 - tail)
                intervals[_level_key(level)] = [lower, upper]
                covered[_level_key(level)] = lower <= observed[position] <= upper
            quantities.append(
                {
                    "quantity": name,
                    "point_prediction": points[position],
                    "predictive_mean": sum(values) / len(values),
                    "predictive_intervals": intervals,
                    "observed": observed[position],
                    "signed_error_of_point_prediction": points[position] - observed[position],
                    "pit": _mid_pit(values, observed[position]),
                    "covered": covered,
                }
            )
        transitions = []
        survivors = [float(record_count)] + observed
        for stage, (predicate_id, predicted) in enumerate(
            zip(order, conditionals, strict=True), start=1
        ):
            reaching, passing = survivors[stage - 1], survivors[stage]
            interval = wilson_interval(int(passing), int(reaching)) if reaching else None
            transitions.append(
                {
                    "stage": stage,
                    "predicate_id": predicate_id,
                    "predicted_conditional_pass_rate": predicted,
                    "observed_records_reaching_stage": reaching,
                    "observed_records_passing_stage": passing,
                    "observed_conditional_pass_rate": passing / reaching if reaching else None,
                    "observed_wilson_95": list(interval) if interval else None,
                    "predicted_within_observed_wilson_95": (
                        interval[0] <= predicted <= interval[1] if interval else None
                    ),
                }
            )
        records.append(
            {
                "order": list(order),
                "quantities": quantities,
                "stage_conditionals": transitions,
            }
        )
    return records


def coverage_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate interval coverage and PIT over every quantity in ``records``."""
    checks = [quantity for record in records for quantity in record["quantities"]]

    def summarize(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not items:
            return {"checks": 0}
        pits = [item["pit"] for item in items]
        return {
            "checks": len(items),
            **{
                f"coverage_{_level_key(level)}": sum(
                    item["covered"][_level_key(level)] for item in items
                )
                / len(items)
                for level in INTERVAL_LEVELS
            },
            "mean_pit": sum(pits) / len(pits),
            "pit_below_0_05": sum(value < 0.05 for value in pits) / len(pits),
            "pit_above_0_95": sum(value > 0.95 for value in pits) / len(pits),
            "mean_signed_error_of_point_prediction": sum(
                item["signed_error_of_point_prediction"] for item in items
            )
            / len(items),
        }

    by_quantity = {
        name: summarize([item for item in checks if item["quantity"] == name])
        for name in sorted({item["quantity"] for item in checks})
    }
    transitions = [
        transition
        for record in records
        for transition in record["stage_conditionals"]
        if transition["predicted_within_observed_wilson_95"] is not None
    ]
    return {
        "all_quantities": summarize(checks),
        "by_quantity": by_quantity,
        "nominal_coverage": {_level_key(level): level for level in INTERVAL_LEVELS},
        "conditional_rates_within_observed_wilson_95": (
            sum(item["predicted_within_observed_wilson_95"] for item in transitions)
            / len(transitions)
            if transitions
            else None
        ),
        "conditional_rate_checks": len(transitions),
    }


def cross_validated_calibration(
    matrix: ProfileMatrix,
    *,
    question_hashes: Mapping[str, str],
    correlation: CorrelationSettings,
    folds: int,
    fold_seed: int,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Calibration of both estimators on K-fold splits of the profiling matrix only."""
    if folds < 2:
        raise ValueError("cross-validation requires at least two folds")
    ranked = sorted(
        (row.review_id for row in matrix.rows),
        key=lambda item: (hashlib.sha256(f"{fold_seed}:{item}".encode()).hexdigest(), item),
    )
    assignments = {review_id: index % folds for index, review_id in enumerate(ranked)}
    orders = list(itertools.permutations(matrix.predicate_ids))
    records: dict[str, list[dict[str, Any]]] = {name: [] for name in ESTIMATOR_NAMES}
    for fold in range(folds):
        train = matrix.select_reviews([rid for rid in ranked if assignments[rid] != fold])
        test = matrix.select_reviews([rid for rid in ranked if assignments[rid] == fold])
        observations = {order: observed_matrix_counts(test, order) for order in orders}
        for name in ESTIMATOR_NAMES:
            for record in calibrate_orders(
                train,
                observations,
                estimator_name=name,
                question_hashes=question_hashes,
                correlation=correlation,
                record_count=test.size,
                iterations=iterations,
                seed=seed + fold,
            ):
                records[name].append({"fold": fold, **record})
    return {
        "method": "k_fold_cross_validation_within_profiling_split",
        "folds": folds,
        "fold_assignment": f"sha256({fold_seed}:review_id) rank modulo folds",
        "iterations_per_fold": iterations,
        "seed": seed,
        "summary": {name: coverage_summary(items) for name, items in records.items()},
        "records": records,
    }


def two_proportion_test(
    passing_a: int, total_a: int, passing_b: int, total_b: int
) -> dict[str, Any]:
    """Pooled two-sided z test for a difference in pass rates (normal approximation)."""
    if total_a <= 0 or total_b <= 0:
        raise ValueError("both samples must be non-empty")
    rate_a, rate_b = passing_a / total_a, passing_b / total_b
    pooled = (passing_a + passing_b) / (total_a + total_b)
    variance = pooled * (1 - pooled) * (1 / total_a + 1 / total_b)
    z = (rate_a - rate_b) / math.sqrt(variance) if variance > 0 else 0.0
    return {
        "rate_a": rate_a,
        "rate_b": rate_b,
        "difference": rate_a - rate_b,
        "z": z,
        "two_sided_p_value": math.erfc(abs(z) / math.sqrt(2)),
        "method": "pooled two-proportion z test, normal approximation",
    }
