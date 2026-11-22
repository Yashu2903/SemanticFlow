from __future__ import annotations

import math

import pytest

from semanticflow.optimizer.m52_evidence import EvidenceCounts
from semanticflow.optimizer.m521_economic_feasibility import (
    EconomicVerdict,
    evaluate_economic_feasibility,
)
from semanticflow.optimizer.m521_error_budget import (
    legacy_requirements,
    nonredundant_requirements,
    redundancy_audit,
    unequal_requirements,
)
from semanticflow.optimizer.m521_feasibility import (
    EvidenceBudget,
    FeasibilityVerdict,
    analyze_feasibility,
    binomial_tail_probability,
    lower_bound_satisfies,
    maximum_tolerable_errors,
    minimum_additional_successes,
    zero_error_minimum,
)
from semanticflow.optimizer.m521_risk_estimation import (
    RiskPolicyType,
    exploratory_beta_risk,
)
from semanticflow.optimizer.m521_statistical_power import (
    certification_power,
    estimate_prevalence,
    expected_records_for_positives,
    minimum_records_for_support_probability,
    positive_support_probability,
)


def empty_counts() -> EvidenceCounts:
    return EvidenceCounts(
        observations=0,
        reference_positives=0,
        reference_negatives=0,
        true_positive=0,
        true_negative=0,
        false_positive=0,
        false_negative=0,
    )


def budget(records: int, *, calls: int = 10_000) -> EvidenceBudget:
    return EvidenceBudget(
        maximum_representative_records=records,
        remaining_representative_records=records,
        maximum_positive_references=records,
        remaining_physical_requests=calls,
        remaining_input_tokens=10_000_000,
        remaining_cost_usd=10,
        requests_per_representative_pair=2,
        expected_input_tokens_per_pair=700,
        expected_cost_per_pair_usd=0.00003,
    )


def test_legacy_zero_error_minimums_are_exact() -> None:
    config = legacy_requirements()
    assert config.recall_alpha == pytest.approx(0.000390625)
    assert zero_error_minimum(0.95, config.recall_alpha) == 153
    assert zero_error_minimum(0.98, config.agreement_alpha) == 389
    assert lower_bound_satisfies(153, 153, 0.95, config.recall_alpha)
    assert not lower_bound_satisfies(152, 152, 0.95, config.recall_alpha)


def test_nonredundant_policy_is_versioned_and_less_conservative() -> None:
    legacy = legacy_requirements()
    alternative = nonredundant_requirements()
    assert legacy.policy_fingerprint != alternative.policy_fingerprint
    assert zero_error_minimum(0.95, alternative.recall_alpha) == 140
    assert zero_error_minimum(0.98, alternative.agreement_alpha) == 355
    assert redundancy_audit()["legacy_simultaneous_events_redundant"] is True


def test_unequal_allocation_preserves_family_budget() -> None:
    config = unequal_requirements(0.6)
    per_checkpoint_family = config.family_alpha / (
        config.candidate_count * config.checkpoint_count
    )
    assert config.recall_alpha + config.agreement_alpha == pytest.approx(
        per_checkpoint_family
    )


def test_sixty_record_pilot_is_mathematically_impossible() -> None:
    certificate = analyze_feasibility(
        candidate_plan_fingerprint="plan",
        requirements=legacy_requirements(),
        counts=empty_counts(),
        budget=budget(60),
    )
    assert certificate.mathematical_verdict == FeasibilityVerdict.IMPOSSIBLE_WITHIN_RECORD_BUDGET
    assert certificate.recommended_next_action == "fallback_without_additional_profiling"
    assert certificate.agreement_constraint.minimum_additional_observations == 389
    assert certificate.agreement_constraint.minimum_qualifying_denominator == 389
    assert certificate.recall_constraint.minimum_additional_observations == 153


def test_nonzero_errors_raise_required_evidence() -> None:
    config = legacy_requirements()
    no_error = minimum_additional_successes(
        successes=0,
        trials=0,
        threshold=0.95,
        alpha=config.recall_alpha,
        maximum_additional=1000,
    )
    one_error = minimum_additional_successes(
        successes=2,
        trials=3,
        threshold=0.95,
        alpha=config.recall_alpha,
        maximum_additional=1000,
    )
    assert no_error == 153
    assert one_error is not None and one_error > no_error
    assert maximum_tolerable_errors(153, 0.95, config.recall_alpha) == 0


def test_call_and_token_limits_are_distinct_from_record_impossibility() -> None:
    config = legacy_requirements()
    call_limited = analyze_feasibility(
        candidate_plan_fingerprint="plan",
        requirements=config,
        counts=empty_counts(),
        budget=budget(500, calls=10),
    )
    assert call_limited.mathematical_verdict == FeasibilityVerdict.IMPOSSIBLE_WITHIN_CALL_BUDGET

    token_budget = budget(500).model_copy(update={"remaining_input_tokens": 1})
    token_limited = analyze_feasibility(
        candidate_plan_fingerprint="plan",
        requirements=config,
        counts=empty_counts(),
        budget=token_budget,
    )
    assert token_limited.mathematical_verdict == FeasibilityVerdict.IMPOSSIBLE_WITHIN_TOKEN_BUDGET


def test_large_n_tiny_alpha_tail_is_numerically_stable() -> None:
    probability = binomial_tail_probability(990, 1000, 0.95)
    assert math.isfinite(probability)
    assert 0 < probability < 1
    assert zero_error_minimum(0.999, 1e-12) > 20_000


@pytest.mark.parametrize("threshold", [0, 1, -0.1, 1.1])
def test_invalid_thresholds_fail(threshold: float) -> None:
    with pytest.raises(ValueError):
        zero_error_minimum(threshold, 0.05)


def test_zero_positive_prevalence_has_nonzero_upper_bound() -> None:
    estimate = estimate_prevalence(0, 400, provenance="test")
    assert estimate.point_estimate == 0
    assert estimate.upper > 0
    assert expected_records_for_positives(153, 0) is None


def test_positive_support_and_power_are_monotone() -> None:
    low = positive_support_probability(1000, 0.05, 153)
    high = positive_support_probability(4000, 0.05, 153)
    assert high > low
    p95 = certification_power(
        trials=153, true_quality=0.95, threshold=0.95, alpha=0.000390625
    )
    p100 = certification_power(
        trials=153, true_quality=1.0, threshold=0.95, alpha=0.000390625
    )
    assert p100.certification_probability > p95.certification_probability
    assert p100.certification_probability == 1
    designed = minimum_records_for_support_probability(153, 0.05, 0.95)
    assert designed is not None
    assert positive_support_probability(designed, 0.05, 153) >= 0.95
    assert positive_support_probability(designed - 1, 0.05, 153) < 0.95


def test_economic_upper_bound_stops_before_spend() -> None:
    result = evaluate_economic_feasibility(
        reuse_horizon=10,
        execution_saving_per_run_usd=0.0005,
        remaining_profiling_cost_usd=0.01,
    )
    assert result.verdict == EconomicVerdict.UNJUSTIFIED
    assert result.maximum_gross_benefit_usd < result.remaining_profiling_cost_usd


def test_exploratory_risk_is_not_formal_certification() -> None:
    result = exploratory_beta_risk(0, 3)
    assert result.policy_type == RiskPolicyType.ESTIMATED_RISK_ONLY
    assert not result.formal_frequentist_guarantee
    assert result.posterior_mean_failure_rate == pytest.approx(0.125)
