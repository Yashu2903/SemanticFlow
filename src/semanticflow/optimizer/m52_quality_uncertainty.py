from __future__ import annotations

import math
from statistics import NormalDist

from pydantic import BaseModel, ConfigDict, Field

from semanticflow.optimizer.m52_evidence import (
    EvidenceTrack,
    ProportionBounds,
    QualificationState,
    QualityEstimate,
    QualityEvidence,
)


class SequentialQualificationConfig(BaseModel):
    """Frozen family-wise sequential qualification protocol.

    A Bonferroni allocation covers every candidate, checkpoint, and simultaneously
    inspected one-sided bound. It is conservative but remains valid under adaptive
    stopping because every possible inspection has an allocation fixed in advance.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "m5.2-sequential-quality-bonferroni-v1"
    minimum_records: int = Field(default=40, ge=1)
    minimum_reference_positives: int = Field(default=3, ge=0)
    minimum_agreement: float = Field(default=0.98, ge=0, le=1)
    minimum_recall: float = Field(default=0.95, ge=0, le=1)
    maximum_false_negative_rate: float = Field(default=0.05, ge=0, le=1)
    maximum_changed_decision_rate: float = Field(default=0.02, ge=0, le=1)
    family_alpha: float = Field(default=0.05, gt=0, lt=1)
    maximum_checkpoints: int = Field(default=8, ge=1)
    family_candidate_count: int = Field(default=4, ge=1)
    simultaneous_bounds: int = Field(default=4, ge=1)
    descriptive_interval_level: float = Field(default=0.95, gt=0, lt=1)
    require_complete_physical_plan_evidence: bool = True

    @property
    def allocated_one_sided_alpha(self) -> float:
        return self.family_alpha / (
            self.maximum_checkpoints * self.family_candidate_count * self.simultaneous_bounds
        )


def wilson_interval(successes: int, trials: int, level: float = 0.95) -> tuple[float, float]:
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("invalid binomial counts")
    if trials == 0:
        raise ValueError("a Wilson interval requires at least one trial")
    if not 0 < level < 1:
        raise ValueError("interval level must be between zero and one")
    z = NormalDist().inv_cdf(0.5 + level / 2)
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    radius /= denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _binomial_cdf(k: int, trials: int, probability: float) -> float:
    if k < 0:
        return 0.0
    if k >= trials:
        return 1.0
    if probability <= 0:
        return 1.0
    if probability >= 1:
        return 0.0
    q = 1 - probability
    term = q**trials
    total = term
    for value in range(k):
        term *= (trials - value) / (value + 1) * probability / q
        total += term
    return min(1.0, max(0.0, total))


def exact_one_sided_bounds(successes: int, trials: int, alpha: float) -> tuple[float, float]:
    """Clopper-Pearson one-sided bounds, computed without optional SciPy."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("invalid binomial counts")
    if trials == 0:
        raise ValueError("exact bounds require at least one trial")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if successes == 0:
        lower = 0.0
    else:
        low, high = 0.0, 1.0
        for _ in range(80):
            mid = (low + high) / 2
            # P_p(X >= successes) = P_(1-p)(failures <= n-successes).
            # This avoids catastrophic cancellation when the upper tail is tiny.
            upper_tail = _binomial_cdf(trials - successes, trials, 1 - mid)
            if upper_tail < alpha:
                low = mid
            else:
                high = mid
        lower = (low + high) / 2
    if successes == trials:
        upper = 1.0
    else:
        low, high = 0.0, 1.0
        for _ in range(80):
            mid = (low + high) / 2
            cdf = _binomial_cdf(successes, trials, mid)
            if cdf > alpha:
                low = mid
            else:
                high = mid
        upper = (low + high) / 2
    return lower, upper


def _bounds(successes: int, trials: int, config: SequentialQualificationConfig) -> ProportionBounds:
    if trials == 0:
        return ProportionBounds(successes=successes, trials=trials)
    wilson = wilson_interval(successes, trials, config.descriptive_interval_level)
    sequential = exact_one_sided_bounds(successes, trials, config.allocated_one_sided_alpha)
    return ProportionBounds(
        estimate=successes / trials,
        wilson_lower=wilson[0],
        wilson_upper=wilson[1],
        sequential_lower=sequential[0],
        sequential_upper=sequential[1],
        successes=successes,
        trials=trials,
    )


def qualify_evidence(
    evidence: QualityEvidence,
    config: SequentialQualificationConfig,
    *,
    checkpoint: int,
) -> QualityEstimate:
    if checkpoint < 0 or checkpoint >= config.maximum_checkpoints:
        raise ValueError("checkpoint is outside the preregistered sequential schedule")
    counts = evidence.counts(EvidenceTrack.CERTIFICATION)
    agreement = _bounds(counts.true_positive + counts.true_negative, counts.observations, config)
    recall = _bounds(counts.true_positive, counts.reference_positives, config)
    fnr = _bounds(counts.false_negative, counts.reference_positives, config)
    changed = _bounds(counts.false_positive + counts.false_negative, counts.observations, config)
    failures = [row for row in evidence.measurements if row.provider_failure]
    reasons: list[str] = []
    if failures:
        state = QualificationState.PROVIDER_FAILURE
        reasons.append("provider_failure_in_required_evidence")
    elif (
        config.require_complete_physical_plan_evidence
        and not evidence.complete_physical_plan_evidence
    ):
        state = QualificationState.UNSUPPORTED
        reasons.append("incomplete_physical_plan_evidence")
    elif counts.observations < config.minimum_records:
        state = QualificationState.INSUFFICIENT_SUPPORT
        reasons.append("insufficient_representative_records")
    elif counts.reference_positives < config.minimum_reference_positives:
        state = QualificationState.INSUFFICIENT_SUPPORT
        reasons.append("insufficient_reference_positives")
    else:
        qualifies = (
            recall.sequential_lower is not None
            and recall.sequential_lower >= config.minimum_recall
            and fnr.sequential_upper is not None
            and fnr.sequential_upper <= config.maximum_false_negative_rate
            and agreement.sequential_lower is not None
            and agreement.sequential_lower >= config.minimum_agreement
            and changed.sequential_upper is not None
            and changed.sequential_upper <= config.maximum_changed_decision_rate
        )
        rejects = (
            (
                recall.sequential_upper is not None
                and recall.sequential_upper < config.minimum_recall
            )
            or (
                fnr.sequential_lower is not None
                and fnr.sequential_lower > config.maximum_false_negative_rate
            )
            or (
                agreement.sequential_upper is not None
                and agreement.sequential_upper < config.minimum_agreement
            )
            or (
                changed.sequential_lower is not None
                and changed.sequential_lower > config.maximum_changed_decision_rate
            )
        )
        if qualifies:
            state = QualificationState.QUALIFIED
            reasons.append("all_sequential_bounds_satisfy_quality_constraints")
        elif rejects:
            state = QualificationState.REJECTED
            reasons.append("sequential_bound_establishes_quality_violation")
        else:
            state = QualificationState.UNCERTAIN
            reasons.append("bounds_neither_certify_nor_reject")
    return QualityEstimate(
        plan_id=evidence.plan_id,
        checkpoint=checkpoint,
        state=state,
        reasons=tuple(reasons),
        counts=counts,
        agreement=agreement,
        recall=recall,
        false_negative_rate=fnr,
        changed_decision_rate=changed,
        family_alpha=config.family_alpha,
        allocated_one_sided_alpha=config.allocated_one_sided_alpha,
        maximum_checkpoints=config.maximum_checkpoints,
        family_candidate_count=config.family_candidate_count,
        complete_physical_plan_evidence=evidence.complete_physical_plan_evidence,
        semantic_accuracy_established=evidence.reference_method == "human_labels",
    )
