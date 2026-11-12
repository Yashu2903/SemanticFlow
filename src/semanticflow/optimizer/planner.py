from __future__ import annotations

import hashlib
import itertools
import math
from typing import Any

from semanticflow.optimizer.bootstrap import confidence_aware_choice, paired_bootstrap
from semanticflow.optimizer.cardinality import (
    CardinalityEstimator,
    CorrelationAwareEstimator,
    IndependentEstimator,
)
from semanticflow.optimizer.cost_model import estimate_plan, percentage_reduction
from semanticflow.optimizer.matrix import ProfileMatrix
from semanticflow.optimizer.models import (
    CandidatePlan,
    OptimizerSettings,
    PlanSelection,
    ProfilingResult,
    SplitManifest,
)
from semanticflow.optimizer.risk import select_risk_aware_plan
from semanticflow.query.models import QuerySpec


def predicate_hash(predicate_id: str, instructions: str) -> str:
    return hashlib.sha256(f"noul\0{predicate_id}\0{instructions}".encode()).hexdigest()


def _pricing_rates(profile: ProfilingResult) -> tuple[float | None, float | None]:
    input_rate = profile.pricing.get("input_per_million_usd")
    output_rate = profile.pricing.get("output_per_million_usd")
    return (
        float(input_rate) if input_rate is not None else None,
        float(output_rate) if output_rate is not None else None,
    )


class CostBasedPlanner:
    def __init__(self, settings: OptimizerSettings) -> None:
        self.settings = settings

    def validate_compatibility(
        self,
        query: QuerySpec,
        profile: ProfilingResult,
        split: SplitManifest,
        *,
        provider_model: str | None = None,
    ) -> None:
        profile.require_complete()
        order = [predicate.id for predicate in query.predicates]
        if not 2 <= len(order) <= 6:
            raise ValueError("optimizer requires 2 to 6 predicates")
        if profile.query_id != query.query_id:
            raise ValueError("profiling query ID does not match query")
        if profile.dataset_fingerprint != split.dataset_fingerprint:
            raise ValueError("profiling dataset fingerprint does not match split")
        if profile.profiling_fingerprint != split.profiling_fingerprint:
            raise ValueError("profiling record fingerprint does not match split")
        if profile.threshold != query.threshold:
            raise ValueError("profiling threshold does not match query")
        if profile.predicate_order != order:
            raise ValueError("profiling predicate IDs do not match query")
        expected_hashes = {
            item.id: predicate_hash(item.id, item.instructions) for item in query.predicates
        }
        if profile.predicate_hashes != expected_hashes:
            raise ValueError("profiling predicate definitions do not match query")
        if profile.statistics_version != self.settings.planner.statistics_version:
            raise ValueError("profiling statistics version is incompatible")
        if provider_model is not None and profile.requested_model != provider_model:
            raise ValueError("profiling model does not match execution model")

    def validate_matrix(
        self, profile: ProfilingResult, split: SplitManifest, matrix: ProfileMatrix
    ) -> None:
        if list(matrix.predicate_ids) != profile.predicate_order:
            raise ValueError("profile matrix predicates do not match profiling order")
        review_ids = {row.review_id for row in matrix.rows}
        if review_ids != set(split.profiling_ids):
            raise ValueError("profile matrix rows must be exactly the profiling split")
        if review_ids & set(split.evaluation_ids):
            raise ValueError("profile matrix contains held-out evaluation reviews")
        if matrix.size != profile.sample_size:
            raise ValueError("profile matrix size does not match profiling sample size")
        statistics = {item.predicate_id: item for item in profile.statistics}
        for predicate_id in matrix.predicate_ids:
            passing, support = matrix.count(matrix.mask([predicate_id]))
            item = statistics[predicate_id]
            if (passing, support) != (item.passing_decisions, item.completed_decisions):
                raise ValueError(
                    f"profile matrix counts for {predicate_id} disagree with statistics"
                )

    def estimator(
        self,
        profile: ProfilingResult,
        split: SplitManifest,
        matrix: ProfileMatrix | None,
    ) -> CardinalityEstimator:
        name = self.settings.planner.cardinality_estimator
        if name == "independent":
            return IndependentEstimator({item.predicate_id: item for item in profile.statistics})
        if matrix is None:
            raise ValueError("correlation-aware planning requires the profiling decision matrix")
        self.validate_matrix(profile, split, matrix)
        input_rate, output_rate = _pricing_rates(profile)
        return CorrelationAwareEstimator(
            matrix,
            self.settings.correlation,
            input_price_per_million=input_rate,
            output_price_per_million=output_rate,
        )

    def _rank(
        self,
        raw_candidates: list[CandidatePlan],
        selected: CandidatePlan,
        original: CandidatePlan,
    ) -> list[CandidatePlan]:
        ordered = [selected] + sorted(
            (item for item in raw_candidates if item is not selected),
            key=lambda item: (
                item.objective_value,
                0 if item.is_original_order else 1,
                tuple(item.predicate_order),
            ),
        )
        return [
            item.model_copy(
                update={
                    "rank": rank,
                    "selected": item is selected,
                    "estimated_reduction_vs_original_percent": percentage_reduction(
                        original.objective_value, item.objective_value
                    ),
                }
            )
            for rank, item in enumerate(ordered, start=1)
        ]

    def plan(
        self,
        query: QuerySpec,
        profile: ProfilingResult,
        split: SplitManifest,
        *,
        provider_model: str | None = None,
        matrix: ProfileMatrix | None = None,
    ) -> PlanSelection:
        self.validate_compatibility(query, profile, split, provider_model=provider_model)
        original_order = [predicate.id for predicate in query.predicates]
        statistics = {item.predicate_id: item for item in profile.statistics}
        estimator = self.estimator(profile, split, matrix)
        raw_candidates = [
            estimate_plan(
                order,
                statistics,
                input_record_count=split.evaluation_size,
                objective=self.settings.planner.objective,
                original_order=original_order,
                estimator=estimator,
            )
            for order in itertools.permutations(original_order)
        ]
        original = next(item for item in raw_candidates if item.is_original_order)
        minimum_value = min(item.objective_value for item in raw_candidates)
        tolerance = self.settings.planner.tie_relative_tolerance * max(1.0, abs(minimum_value))
        tied = [
            item
            for item in raw_candidates
            if math.isclose(
                item.objective_value,
                minimum_value,
                rel_tol=self.settings.planner.tie_relative_tolerance,
                abs_tol=tolerance,
            )
        ]
        if original in tied:
            minimum = original
        else:
            minimum = min(tied, key=lambda item: tuple(item.predicate_order))

        selected = minimum
        explanation: dict[str, Any] | None = None
        policy = self.settings.planner.selection_policy
        if policy == "confidence_aware":
            selected, explanation = self._confidence_aware(
                profile, split, matrix, original, minimum, raw_candidates
            )
        elif policy in {"risk_aware", "risk_aware_robust"}:
            selected, explanation = self._risk_aware(
                profile, split, matrix, original, raw_candidates
            )

        candidates = self._rank(raw_candidates, selected, original)
        selected_candidate = candidates[0]
        correlation_aware = self.settings.planner.cardinality_estimator == "correlation_aware"
        return PlanSelection(
            query_id=query.query_id,
            dataset_fingerprint=profile.dataset_fingerprint,
            profiling_fingerprint=profile.profiling_fingerprint,
            threshold=query.threshold,
            requested_model=profile.requested_model,
            predicate_hashes=profile.predicate_hashes,
            statistics_version=self.settings.planner.statistics_version,
            cost_model_version=self.settings.planner.cost_model_version,
            optimizer_version=self.settings.planner.optimizer_version,
            objective=self.settings.planner.objective,
            evaluation_record_count=split.evaluation_size,
            original_predicate_order=original_order,
            selected_predicate_order=selected_candidate.predicate_order,
            original_estimated_objective=original.objective_value,
            selected_estimated_objective=selected_candidate.objective_value,
            estimated_reduction_percent=percentage_reduction(
                original.objective_value, selected_candidate.objective_value
            ),
            reordered=selected_candidate.predicate_order != original_order,
            candidate_count=len(candidates),
            candidates=candidates,
            cardinality_estimator=self.settings.planner.cardinality_estimator,
            joint_statistics_version=(
                self.settings.correlation.joint_statistics_version if correlation_aware else None
            ),
            joint_matrix_fingerprint=(
                matrix.fingerprint() if correlation_aware and matrix is not None else None
            ),
            selection_policy=self.settings.planner.selection_policy,
            minimum_objective_predicate_order=minimum.predicate_order,
            selection_explanation=explanation,
        )

    def _confidence_aware(
        self,
        profile: ProfilingResult,
        split: SplitManifest,
        matrix: ProfileMatrix | None,
        original: CandidatePlan,
        minimum: CandidatePlan,
        raw_candidates: list[CandidatePlan],
    ) -> tuple[CandidatePlan, dict[str, Any]]:
        if minimum is original:
            _, explanation = confidence_aware_choice(
                minimum_order=minimum.predicate_order,
                original_order=original.predicate_order,
                minimum_objective=minimum.objective_value,
                original_objective=original.objective_value,
                bootstrap={},
                settings=self.settings.bootstrap,
            )
            return original, explanation
        if matrix is None:
            raise ValueError("confidence-aware selection requires the profiling decision matrix")
        self.validate_matrix(profile, split, matrix)
        input_rate, output_rate = _pricing_rates(profile)
        report = paired_bootstrap(
            matrix,
            estimator_name=self.settings.planner.cardinality_estimator,
            original_order=original.predicate_order,
            objective=self.settings.planner.objective,
            evaluation_record_count=split.evaluation_size,
            question_hashes=profile.predicate_hashes,
            settings=self.settings.bootstrap,
            correlation=self.settings.correlation,
            input_price_per_million=input_rate,
            output_price_per_million=output_rate,
            tie_relative_tolerance=self.settings.planner.tie_relative_tolerance,
            reference_order=minimum.predicate_order,
        )
        order, explanation = confidence_aware_choice(
            minimum_order=minimum.predicate_order,
            original_order=original.predicate_order,
            minimum_objective=minimum.objective_value,
            original_objective=original.objective_value,
            bootstrap=report,
            settings=self.settings.bootstrap,
        )
        explanation["bootstrap_iterations"] = report["iterations"]
        explanation["bootstrap_seed"] = report["seed"]
        explanation["interpretation"] = report["interpretation"]
        selected = next(item for item in raw_candidates if item.predicate_order == order)
        return selected, explanation

    def _risk_aware(
        self,
        profile: ProfilingResult,
        split: SplitManifest,
        matrix: ProfileMatrix | None,
        original: CandidatePlan,
        raw_candidates: list[CandidatePlan],
    ) -> tuple[CandidatePlan, dict[str, Any]]:
        if matrix is None:
            raise ValueError("risk-aware selection requires the profiling decision matrix")
        self.validate_matrix(profile, split, matrix)
        input_rate, output_rate = _pricing_rates(profile)
        policy = self.settings.planner.selection_policy
        order, explanation = select_risk_aware_plan(
            matrix,
            original_order=original.predicate_order,
            objective=self.settings.planner.objective,
            evaluation_record_count=split.evaluation_size,
            question_hashes=profile.predicate_hashes,
            bootstrap=self.settings.bootstrap,
            correlation=self.settings.correlation,
            risk=self.settings.risk,
            economics=self.settings.economics,
            input_price_per_million=input_rate,
            output_price_per_million=output_rate,
            tie_relative_tolerance=self.settings.planner.tie_relative_tolerance,
            primary_estimator=self.settings.planner.cardinality_estimator,
            robust=policy == "risk_aware_robust",
            statistics_state="available",
            new_profiling_cost=profile.estimated_cost_usd,
            sunk_profiling_cost=profile.estimated_cost_usd or 0.0,
        )
        selected = next(item for item in raw_candidates if item.predicate_order == order)
        return selected, explanation
