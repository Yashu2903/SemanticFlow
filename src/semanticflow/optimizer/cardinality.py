from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol

from semanticflow.optimizer.matrix import ProfileMatrix
from semanticflow.optimizer.models import CorrelationSettings, PredicateStatistics
from semanticflow.optimizer.selectivity import wilson_interval

JEFFREYS_PRIOR_MEAN = 0.5


@dataclass(frozen=True, slots=True)
class StageEstimate:
    """Predicted reach probability and per-evaluation usage for one stage of an order."""

    stage: int
    predicate_id: str
    reach_probability: float
    average_input_tokens: float | None
    average_output_tokens: float | None
    average_latency_seconds: float
    average_cost_usd: float | None
    method: str
    sample_size: int | None = None
    prefix_passing: int | None = None
    minimum_chain_support: int | None = None
    reach_wilson_lower: float | None = None
    reach_wilson_upper: float | None = None
    confidence: Literal["high", "low"] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CardinalityEstimator(Protocol):
    name: str

    def stage_estimates(self, order: Sequence[str]) -> list[StageEstimate]: ...


class IndependentEstimator:
    """Milestone 2 model: stage reach is the product of unconditional selectivities."""

    name = "independent"

    def __init__(self, statistics: Mapping[str, PredicateStatistics]) -> None:
        self.statistics = statistics

    def stage_estimates(self, order: Sequence[str]) -> list[StageEstimate]:
        reach = 1.0
        estimates: list[StageEstimate] = []
        for stage, predicate_id in enumerate(order, start=1):
            item = self.statistics[predicate_id]
            if item.estimated_selectivity is None:
                raise ValueError(f"unknown selectivity for predicate {predicate_id}")
            estimates.append(
                StageEstimate(
                    stage=stage,
                    predicate_id=predicate_id,
                    reach_probability=reach,
                    average_input_tokens=item.average_input_tokens,
                    average_output_tokens=item.average_output_tokens,
                    average_latency_seconds=item.average_inference_latency_seconds,
                    average_cost_usd=item.average_estimated_cost_usd,
                    method="independent_product",
                    sample_size=item.completed_decisions,
                )
            )
            reach *= item.estimated_selectivity
        return estimates


class CorrelationAwareEstimator:
    """Estimates stage reach as P(all preceding predicates pass) from the joint profile.

    Reach is built by the chain rule over the *entire* prefix. Each conditional
    P(P_j | P_1..P_{j-1}) = x / m is shrunk with a Beta prior of strength ``prior_strength``
    toward a backoff target: the Jeffreys mean 0.5 for marginals, the smoothed marginal for
    one conditioning predicate, and the support-weighted mean of smoothed pairwise
    conditionals for longer prefixes. With ``prior_strength = 0`` and observed prefixes, the
    estimate equals exact replay over the profiling matrix.
    """

    name = "correlation_aware"

    def __init__(
        self,
        matrix: ProfileMatrix,
        settings: CorrelationSettings | None = None,
        *,
        input_price_per_million: float | None = None,
        output_price_per_million: float | None = None,
    ) -> None:
        self.settings = settings or CorrelationSettings()
        if not matrix.complete:
            if not self.settings.allow_incomplete_matrix:
                raise ValueError(
                    f"correlation-aware estimation requires a complete profiling matrix; "
                    f"{matrix.missing_cells} decisions are missing"
                )
            matrix = matrix.complete_rows()
        self.matrix = matrix
        self.input_price = input_price_per_million
        self.output_price = output_price_per_million
        self._usage_cache: dict[tuple[int, int], tuple[float | None, float | None, float]] = {}

    @property
    def sample_size(self) -> int:
        return self.matrix.size

    def _smoothed(self, passing: int, support: int, target: float) -> float:
        alpha = self.settings.prior_strength
        if support + alpha == 0:
            return target
        return (passing + alpha * target) / (support + alpha)

    def _marginal(self, index: int) -> float:
        passing, support = self.matrix.count(1 << index)
        return self._smoothed(passing, support, JEFFREYS_PRIOR_MEAN)

    def _backoff_target(self, index: int, prefix: Sequence[int]) -> float:
        marginal = self._marginal(index)
        if len(prefix) <= 1:
            return marginal
        weighted = weight = 0.0
        for condition in prefix:
            joint, _ = self.matrix.count((1 << index) | (1 << condition))
            support, _ = self.matrix.count(1 << condition)
            if support == 0:
                continue
            weighted += support * self._smoothed(joint, support, marginal)
            weight += support
        return weighted / weight if weight else marginal

    def conditional(self, index: int, prefix: Sequence[int]) -> tuple[float, int, int, str]:
        """Return (estimate, passing, support, method) for P(index | all prefix pass)."""
        prefix_mask = 0
        for condition in prefix:
            prefix_mask |= 1 << condition
        support, _ = self.matrix.count(prefix_mask) if prefix else (self.matrix.size, 0)
        passing, _ = self.matrix.count(prefix_mask | (1 << index))
        target = self._backoff_target(index, prefix) if prefix else JEFFREYS_PRIOR_MEAN
        if support == 0:
            return target, 0, 0, "backoff_unseen_prefix"
        if self.settings.prior_strength == 0:
            return passing / support, passing, support, "empirical"
        return self._smoothed(passing, support, target), passing, support, "smoothed"

    def _usage(self, index: int, prefix_mask: int) -> tuple[float | None, float | None, float]:
        key = (index, prefix_mask)
        cached = self._usage_cache.get(key)
        if cached is not None:
            return cached
        rows = self.matrix.rows

        def mean(values: list[float]) -> float:
            return sum(values) / len(values)

        all_input = [row.input_tokens[index] for row in rows]
        all_output = [row.output_tokens[index] for row in rows]
        all_latency = [row.latency_seconds[index] or 0.0 for row in rows]
        tokens_known = None not in all_input and None not in all_output
        base_input = mean([float(v) for v in all_input if v is not None]) if tokens_known else None
        base_output = (
            mean([float(v) for v in all_output if v is not None]) if tokens_known else None
        )
        base_latency = mean(all_latency)
        if not self.settings.conditional_usage or prefix_mask == 0:
            result = (base_input, base_output, base_latency)
            self._usage_cache[key] = result
            return result
        selected = [
            row
            for row in rows
            if all(
                row.outcomes[bit]
                for bit in range(len(self.matrix.predicate_ids))
                if prefix_mask & (1 << bit)
            )
        ]
        count = len(selected)
        alpha = self.settings.usage_prior_strength

        def shrink(values: list[float], base: float) -> float:
            if count + alpha == 0:
                return base
            return (sum(values) + alpha * base) / (count + alpha)

        conditional_input = (
            shrink([float(row.input_tokens[index] or 0) for row in selected], base_input)
            if base_input is not None
            else None
        )
        conditional_output = (
            shrink([float(row.output_tokens[index] or 0) for row in selected], base_output)
            if base_output is not None
            else None
        )
        conditional_latency = shrink(
            [row.latency_seconds[index] or 0.0 for row in selected], base_latency
        )
        result = (conditional_input, conditional_output, conditional_latency)
        self._usage_cache[key] = result
        return result

    def _cost(self, input_tokens: float | None, output_tokens: float | None) -> float | None:
        if (
            input_tokens is None
            or output_tokens is None
            or self.input_price is None
            or self.output_price is None
        ):
            return None
        return (input_tokens * self.input_price + output_tokens * self.output_price) / 1_000_000

    def stage_estimates(self, order: Sequence[str]) -> list[StageEstimate]:
        indices = [self.matrix.index(predicate_id) for predicate_id in order]
        minimum_support = self.settings.minimum_conditional_support
        estimates: list[StageEstimate] = []
        reach = 1.0
        prefix: list[int] = []
        prefix_mask = 0
        prefix_passing = self.matrix.size
        chain_support = self.matrix.size
        chained = (
            "prefix_empirical" if self.settings.prior_strength == 0 else "prefix_chain_smoothed"
        )
        for stage, (predicate_id, index) in enumerate(zip(order, indices, strict=True), start=1):
            input_tokens, output_tokens, latency = self._usage(index, prefix_mask)
            interval = (
                wilson_interval(prefix_passing, self.matrix.size) if prefix else (1.0, 1.0)
            )
            estimates.append(
                StageEstimate(
                    stage=stage,
                    predicate_id=predicate_id,
                    reach_probability=reach,
                    average_input_tokens=input_tokens,
                    average_output_tokens=output_tokens,
                    average_latency_seconds=latency,
                    average_cost_usd=self._cost(input_tokens, output_tokens),
                    method="certain_first_stage" if not prefix else chained,
                    sample_size=self.matrix.size,
                    prefix_passing=prefix_passing,
                    minimum_chain_support=chain_support,
                    reach_wilson_lower=interval[0] if interval else None,
                    reach_wilson_upper=interval[1] if interval else None,
                    confidence="high" if chain_support >= minimum_support else "low",
                )
            )
            estimate, passing, support, _ = self.conditional(index, prefix)
            chain_support = min(chain_support, support)
            reach *= estimate
            prefix.append(index)
            prefix_mask |= 1 << index
            prefix_passing = passing
        return estimates


def predicate_statistics_from_matrix(
    matrix: ProfileMatrix,
    *,
    question_hashes: Mapping[str, str],
    input_price_per_million: float | None,
    output_price_per_million: float | None,
    model_versions: list[str] | None = None,
) -> dict[str, PredicateStatistics]:
    """Unconditional per-predicate statistics, computed exactly as the profiler does."""
    statistics: dict[str, PredicateStatistics] = {}
    for index, predicate_id in enumerate(matrix.predicate_ids):
        observed = [row for row in matrix.rows if row.outcomes[index] is not None]
        completed = len(observed)
        passing = sum(bool(row.outcomes[index]) for row in observed)
        interval = wilson_interval(passing, completed) if completed else None
        tokens_known = completed > 0 and all(
            row.input_tokens[index] is not None and row.output_tokens[index] is not None
            for row in observed
        )
        average_input = (
            sum(row.input_tokens[index] or 0 for row in observed) / completed
            if tokens_known
            else None
        )
        average_output = (
            sum(row.output_tokens[index] or 0 for row in observed) / completed
            if tokens_known
            else None
        )
        average_cost = None
        if (
            average_input is not None
            and average_output is not None
            and input_price_per_million is not None
            and output_price_per_million is not None
        ):
            average_cost = (
                average_input * input_price_per_million
                + average_output * output_price_per_million
            ) / 1_000_000
        selectivity = passing / completed if completed else None
        statistics[predicate_id] = PredicateStatistics(
            predicate_id=predicate_id,
            question_hash=question_hashes[predicate_id],
            sample_size=matrix.size,
            completed_decisions=completed,
            passing_decisions=passing,
            rejected_decisions=completed - passing,
            estimated_selectivity=selectivity,
            estimated_rejection_rate=1 - selectivity if selectivity is not None else None,
            wilson_lower=interval[0] if interval else None,
            wilson_upper=interval[1] if interval else None,
            average_input_tokens=average_input,
            average_output_tokens=average_output,
            average_total_tokens=(
                average_input + average_output
                if average_input is not None and average_output is not None
                else None
            ),
            average_inference_latency_seconds=(
                sum(row.latency_seconds[index] or 0.0 for row in observed) / completed
                if completed
                else 0.0
            ),
            average_estimated_cost_usd=average_cost,
            provider_attempts=0,
            http_attempts=0,
            successful_api_calls=0,
            failed_http_attempts=0,
            retry_attempts=0,
            provider_errors=0,
            budget_blocked_evaluations=0,
            model_versions=model_versions or [],
        )
    return statistics


def build_estimator(
    name: str,
    matrix: ProfileMatrix,
    *,
    question_hashes: Mapping[str, str],
    correlation: CorrelationSettings,
    input_price_per_million: float | None,
    output_price_per_million: float | None,
) -> IndependentEstimator | CorrelationAwareEstimator:
    if name == "independent":
        return IndependentEstimator(
            predicate_statistics_from_matrix(
                matrix,
                question_hashes=question_hashes,
                input_price_per_million=input_price_per_million,
                output_price_per_million=output_price_per_million,
            )
        )
    if name == "correlation_aware":
        return CorrelationAwareEstimator(
            matrix,
            correlation,
            input_price_per_million=input_price_per_million,
            output_price_per_million=output_price_per_million,
        )
    raise ValueError(f"unknown cardinality estimator: {name}")


def exact_matrix_cardinalities(matrix: ProfileMatrix, order: Sequence[str]) -> list[int]:
    """Stage input counts from sequential early rejection over a complete matrix."""
    if not matrix.complete:
        raise ValueError("exact matrix replay requires a complete matrix")
    counts = [matrix.size]
    mask = 0
    for predicate_id in order[:-1]:
        mask |= 1 << matrix.index(predicate_id)
        counts.append(matrix.count(mask)[0])
    return counts
