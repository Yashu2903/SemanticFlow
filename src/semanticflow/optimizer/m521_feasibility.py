from __future__ import annotations

import hashlib
import json
import math
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from semanticflow.optimizer.m52_evidence import EvidenceCounts
from semanticflow.optimizer.m521_error_budget import QualificationRequirements


class FeasibilityVerdict(StrEnum):
    FEASIBLE_NOW = "FEASIBLE_NOW"
    FEASIBLE_WITH_MORE_EVIDENCE = "FEASIBLE_WITH_MORE_EVIDENCE"
    IMPOSSIBLE_WITHIN_RECORD_BUDGET = "IMPOSSIBLE_WITHIN_RECORD_BUDGET"
    IMPOSSIBLE_WITHIN_CALL_BUDGET = "IMPOSSIBLE_WITHIN_CALL_BUDGET"
    IMPOSSIBLE_WITHIN_TOKEN_BUDGET = "IMPOSSIBLE_WITHIN_TOKEN_BUDGET"
    INSUFFICIENT_POSITIVE_SUPPORT = "INSUFFICIENT_POSITIVE_SUPPORT"
    STRUCTURALLY_UNMEASURABLE = "STRUCTURALLY_UNMEASURABLE"
    ECONOMICALLY_UNJUSTIFIED = "ECONOMICALLY_UNJUSTIFIED"
    UNKNOWN_DUE_TO_MISSING_ASSUMPTIONS = "UNKNOWN_DUE_TO_MISSING_ASSUMPTIONS"


class EvidenceBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    maximum_representative_records: int = Field(ge=0)
    remaining_representative_records: int = Field(ge=0)
    maximum_positive_references: int | None = Field(default=None, ge=0)
    remaining_physical_requests: int = Field(ge=0)
    remaining_input_tokens: int | None = Field(default=None, ge=0)
    remaining_cost_usd: float = Field(ge=0)
    requests_per_representative_pair: int = Field(default=1, ge=1)
    expected_input_tokens_per_pair: float | None = Field(default=None, ge=0)
    expected_cost_per_pair_usd: float | None = Field(default=None, ge=0)
    physical_arms_available: bool = True


class ConstraintFeasibility(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    estimand: str
    threshold: float
    effective_alpha: float
    current_denominator: int
    current_errors: int
    minimum_qualifying_denominator: int | None
    minimum_additional_observations: int | None
    maximum_errors_at_budget: int | None
    feasible_within_budget: bool


class FeasibilityCertificate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "m5.2.1-feasibility-certificate-v1"
    candidate_plan_fingerprint: str
    qualification_policy_fingerprint: str
    requirements: QualificationRequirements
    current_evidence: EvidenceCounts
    remaining_budget: EvidenceBudget
    recall_constraint: ConstraintFeasibility
    agreement_constraint: ConstraintFeasibility
    binding_constraints: tuple[str, ...]
    mathematical_verdict: FeasibilityVerdict
    statistical_power: float | None = Field(default=None, ge=0, le=1)
    economic_verdict: FeasibilityVerdict | None = None
    recommended_next_action: str
    explanation: tuple[str, ...]
    certificate_fingerprint: str


def _logsumexp(values: list[float]) -> float:
    if not values:
        return float("-inf")
    largest = max(values)
    return largest + math.log(sum(math.exp(value - largest) for value in values))


def binomial_tail_probability(successes_at_least: int, trials: int, probability: float) -> float:
    """Stable P[X >= successes_at_least] for a binomial random variable."""
    if trials < 0 or successes_at_least < 0 or successes_at_least > trials + 1:
        raise ValueError("invalid binomial tail arguments")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    if successes_at_least <= 0:
        return 1.0
    if successes_at_least > trials:
        return 0.0
    if probability == 0:
        return 0.0
    if probability == 1:
        return 1.0
    log_p = math.log(probability)
    log_q = math.log1p(-probability)

    def probability_range(start: int, stop: int) -> float:
        terms = [
            math.lgamma(trials + 1)
            - math.lgamma(value + 1)
            - math.lgamma(trials - value + 1)
            + value * log_p
            + (trials - value) * log_q
            for value in range(start, stop)
        ]
        return min(1.0, math.exp(_logsumexp(terms)))

    if successes_at_least <= trials * probability:
        return max(0.0, 1 - probability_range(0, successes_at_least))
    return probability_range(successes_at_least, trials + 1)


def lower_bound_satisfies(successes: int, trials: int, threshold: float, alpha: float) -> bool:
    if successes < 0 or trials < 0 or successes > trials:
        raise ValueError("invalid binomial counts")
    if not 0 < threshold < 1 or not 0 < alpha < 1:
        raise ValueError("threshold and alpha must be in (0, 1)")
    return binomial_tail_probability(successes, trials, threshold) <= alpha * (1 + 1e-12)


def zero_error_minimum(threshold: float, alpha: float) -> int:
    if not 0 < threshold < 1 or not 0 < alpha < 1:
        raise ValueError("threshold and alpha must be in (0, 1)")
    return math.ceil(math.log(alpha) / math.log(threshold))


def minimum_additional_successes(
    *, successes: int, trials: int, threshold: float, alpha: float, maximum_additional: int
) -> int | None:
    if maximum_additional < 0:
        raise ValueError("maximum additional evidence cannot be negative")
    if lower_bound_satisfies(successes, trials, threshold, alpha):
        return 0
    low, high = 1, maximum_additional
    if high == 0 or not lower_bound_satisfies(
        successes + high, trials + high, threshold, alpha
    ):
        return None
    while low < high:
        middle = (low + high) // 2
        if lower_bound_satisfies(successes + middle, trials + middle, threshold, alpha):
            high = middle
        else:
            low = middle + 1
    return low


def minimum_additional_successes_unbounded(
    *, successes: int, trials: int, threshold: float, alpha: float
) -> int:
    """Return the global best-case evidence gap, independent of a local budget.

    Future observations are assumed to be successes.  For every valid threshold
    below one, a finite all-success continuation eventually crosses the exact
    lower bound.  Exponential bracketing avoids an arbitrary evidence ceiling.
    """
    if lower_bound_satisfies(successes, trials, threshold, alpha):
        return 0
    high = 1
    while not lower_bound_satisfies(
        successes + high, trials + high, threshold, alpha
    ):
        high *= 2
    low = high // 2 + 1
    while low < high:
        middle = (low + high) // 2
        if lower_bound_satisfies(
            successes + middle, trials + middle, threshold, alpha
        ):
            high = middle
        else:
            low = middle + 1
    return low


def maximum_tolerable_errors(trials: int, threshold: float, alpha: float) -> int:
    if trials <= 0:
        return -1
    for errors in range(trials + 1):
        if not lower_bound_satisfies(trials - errors, trials, threshold, alpha):
            return errors - 1
    return trials


def _constraint(
    name: str,
    *,
    successes: int,
    trials: int,
    threshold: float,
    alpha: float,
    maximum_denominator: int,
) -> ConstraintFeasibility:
    remaining = max(0, maximum_denominator - trials)
    additional = minimum_additional_successes_unbounded(
        successes=successes,
        trials=trials,
        threshold=threshold,
        alpha=alpha,
    )
    minimum = trials + additional
    return ConstraintFeasibility(
        estimand=name,
        threshold=threshold,
        effective_alpha=alpha,
        current_denominator=trials,
        current_errors=trials - successes,
        minimum_qualifying_denominator=minimum,
        minimum_additional_observations=additional,
        maximum_errors_at_budget=maximum_tolerable_errors(
            maximum_denominator, threshold, alpha
        ),
        feasible_within_budget=additional <= remaining,
    )


def analyze_feasibility(
    *,
    candidate_plan_fingerprint: str,
    requirements: QualificationRequirements,
    counts: EvidenceCounts,
    budget: EvidenceBudget,
) -> FeasibilityCertificate:
    maximum_records = counts.observations + budget.remaining_representative_records
    positive_cap = budget.maximum_positive_references
    maximum_positives = (
        maximum_records if positive_cap is None else max(counts.reference_positives, positive_cap)
    )
    recall = _constraint(
        "recall_fnr",
        successes=counts.true_positive,
        trials=counts.reference_positives,
        threshold=requirements.minimum_recall,
        alpha=requirements.recall_alpha,
        maximum_denominator=maximum_positives,
    )
    agreement = _constraint(
        "agreement_changed_decision",
        successes=counts.true_positive + counts.true_negative,
        trials=counts.observations,
        threshold=requirements.minimum_agreement,
        alpha=requirements.agreement_alpha,
        maximum_denominator=maximum_records,
    )
    explanations: list[str] = []
    if not budget.physical_arms_available:
        verdict = FeasibilityVerdict.STRUCTURALLY_UNMEASURABLE
        explanations.append("required paired physical measurement arm is unavailable")
    elif budget.remaining_physical_requests < (
        budget.requests_per_representative_pair
        * max(
            recall.minimum_additional_observations or 0,
            agreement.minimum_additional_observations or 0,
        )
    ) and recall.feasible_within_budget and agreement.feasible_within_budget:
        verdict = FeasibilityVerdict.IMPOSSIBLE_WITHIN_CALL_BUDGET
        explanations.append("record evidence could suffice but the physical-call budget cannot")
    elif (
        budget.expected_input_tokens_per_pair is not None
        and budget.remaining_input_tokens is not None
        and budget.expected_input_tokens_per_pair
        * max(
            recall.minimum_additional_observations or 0,
            agreement.minimum_additional_observations or 0,
        )
        > budget.remaining_input_tokens
    ):
        verdict = FeasibilityVerdict.IMPOSSIBLE_WITHIN_TOKEN_BUDGET
        explanations.append("forecast input tokens exceed the remaining token budget")
    elif not agreement.feasible_within_budget:
        verdict = FeasibilityVerdict.IMPOSSIBLE_WITHIN_RECORD_BUDGET
        explanations.append(
            "agreement/change-rate confidence bound cannot cross within record budget"
        )
    elif not recall.feasible_within_budget:
        verdict = (
            FeasibilityVerdict.INSUFFICIENT_POSITIVE_SUPPORT
            if maximum_positives < zero_error_minimum(
                requirements.minimum_recall, requirements.recall_alpha
            )
            else FeasibilityVerdict.IMPOSSIBLE_WITHIN_RECORD_BUDGET
        )
        explanations.append("recall/FNR confidence bound cannot cross within positive budget")
    elif (
        recall.minimum_additional_observations == 0
        and agreement.minimum_additional_observations == 0
    ):
        verdict = FeasibilityVerdict.FEASIBLE_NOW
        explanations.append("all distinct confidence constraints currently cross")
    else:
        verdict = FeasibilityVerdict.FEASIBLE_WITH_MORE_EVIDENCE
        explanations.append("a zero-further-error sequence can qualify within budget")
    gaps = {
        "recall_fnr": recall.minimum_additional_observations,
        "agreement_changed_decision": agreement.minimum_additional_observations,
    }
    finite = {name: value for name, value in gaps.items() if value is not None}
    binding = (
        tuple(name for name, value in finite.items() if value == max(finite.values()))
        if finite
        else tuple(name for name, value in gaps.items() if value is None)
    )
    action = (
        "continue_representative_profiling"
        if verdict == FeasibilityVerdict.FEASIBLE_WITH_MORE_EVIDENCE
        else "qualify_now"
        if verdict == FeasibilityVerdict.FEASIBLE_NOW
        else "fallback_without_additional_profiling"
    )
    payload = {
        "candidate": candidate_plan_fingerprint,
        "policy": requirements.policy_fingerprint,
        "counts": counts.model_dump(mode="json"),
        "budget": budget.model_dump(mode="json"),
        "recall": recall.model_dump(mode="json"),
        "agreement": agreement.model_dump(mode="json"),
        "verdict": verdict.value,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return FeasibilityCertificate(
        candidate_plan_fingerprint=candidate_plan_fingerprint,
        qualification_policy_fingerprint=requirements.policy_fingerprint,
        requirements=requirements,
        current_evidence=counts,
        remaining_budget=budget,
        recall_constraint=recall,
        agreement_constraint=agreement,
        binding_constraints=binding,
        mathematical_verdict=verdict,
        recommended_next_action=action,
        explanation=tuple(explanations),
        certificate_fingerprint=fingerprint,
    )
