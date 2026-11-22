from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field

from semanticflow.optimizer.m52_quality_uncertainty import wilson_interval
from semanticflow.optimizer.m521_feasibility import (
    binomial_tail_probability,
    lower_bound_satisfies,
)


class PrevalenceEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    positives: int = Field(ge=0)
    observations: int = Field(gt=0)
    point_estimate: float = Field(ge=0, le=1)
    lower: float = Field(ge=0, le=1)
    upper: float = Field(ge=0, le=1)
    method: str
    provenance: str


class PowerEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trials: int = Field(ge=0)
    true_quality: float = Field(ge=0, le=1)
    threshold: float = Field(gt=0, lt=1)
    alpha: float = Field(gt=0, lt=1)
    minimum_successes: int | None
    maximum_errors: int | None
    certification_probability: float = Field(ge=0, le=1)
    exact_binomial: bool = True


def estimate_prevalence(
    positives: int, observations: int, *, provenance: str
) -> PrevalenceEstimate:
    if observations <= 0 or positives < 0 or positives > observations:
        raise ValueError("invalid prevalence counts")
    lower, upper = wilson_interval(positives, observations)
    return PrevalenceEstimate(
        positives=positives,
        observations=observations,
        point_estimate=positives / observations,
        lower=lower,
        upper=upper,
        method="Wilson score 95%",
        provenance=provenance,
    )


def binomial_cdf(successes_at_most: int, trials: int, probability: float) -> float:
    if successes_at_most < 0:
        return 0.0
    if successes_at_most >= trials:
        return 1.0
    return 1 - binomial_tail_probability(successes_at_most + 1, trials, probability)


def positive_support_probability(
    records: int, prevalence: float, required_positives: int
) -> float:
    if required_positives <= 0:
        return 1.0
    if records < required_positives:
        return 0.0
    return binomial_tail_probability(required_positives, records, prevalence)


def expected_records_for_positives(required_positives: int, prevalence: float) -> int | None:
    if required_positives < 0 or not 0 <= prevalence <= 1:
        raise ValueError("invalid expected-support inputs")
    if required_positives == 0:
        return 0
    if prevalence == 0:
        return None
    return math.ceil(required_positives / prevalence)


def minimum_records_for_support_probability(
    required_positives: int, prevalence: float, target_probability: float
) -> int | None:
    if required_positives < 0 or not 0 <= prevalence <= 1:
        raise ValueError("invalid support-design inputs")
    if not 0 < target_probability < 1:
        raise ValueError("target probability must be in (0, 1)")
    if required_positives == 0:
        return 0
    if prevalence == 0:
        return None
    low = required_positives
    high = max(low, math.ceil(required_positives / prevalence))
    while positive_support_probability(high, prevalence, required_positives) < target_probability:
        high *= 2
    while low < high:
        middle = (low + high) // 2
        probability = positive_support_probability(
            middle, prevalence, required_positives
        )
        if probability >= target_probability:
            high = middle
        else:
            low = middle + 1
    return low


def certification_power(
    *, trials: int, true_quality: float, threshold: float, alpha: float
) -> PowerEstimate:
    if trials < 0 or not 0 <= true_quality <= 1:
        raise ValueError("invalid power inputs")
    minimum: int | None = None
    low, high = 0, trials
    if lower_bound_satisfies(trials, trials, threshold, alpha):
        while low < high:
            middle = (low + high) // 2
            if lower_bound_satisfies(middle, trials, threshold, alpha):
                high = middle
            else:
                low = middle + 1
        minimum = low
    probability = (
        0.0
        if minimum is None
        else binomial_tail_probability(minimum, trials, true_quality)
    )
    return PowerEstimate(
        trials=trials,
        true_quality=true_quality,
        threshold=threshold,
        alpha=alpha,
        minimum_successes=minimum,
        maximum_errors=None if minimum is None else trials - minimum,
        certification_probability=probability,
    )


def joint_certification_power_approximation(
    *,
    representative_records: int,
    positive_prevalence: float,
    required_positive_trials: int,
    true_recall: float,
    recall_threshold: float,
    recall_alpha: float,
    true_agreement: float,
    agreement_threshold: float,
    agreement_alpha: float,
) -> dict[str, float | str]:
    support = positive_support_probability(
        representative_records, positive_prevalence, required_positive_trials
    )
    recall = certification_power(
        trials=required_positive_trials,
        true_quality=true_recall,
        threshold=recall_threshold,
        alpha=recall_alpha,
    ).certification_probability
    agreement = certification_power(
        trials=representative_records,
        true_quality=true_agreement,
        threshold=agreement_threshold,
        alpha=agreement_alpha,
    ).certification_probability
    return {
        "positive_support_probability": support,
        "conditional_recall_certification_power": recall,
        "agreement_certification_power": agreement,
        "joint_independence_approximation": support * recall * agreement,
        "assumption": (
            "multiplicative approximation; positive support, recall errors, and total "
            "agreement are not generally independent"
        ),
    }
