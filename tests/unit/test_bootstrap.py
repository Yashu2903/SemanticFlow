import itertools
import random

import pytest

from semanticflow.optimizer.bootstrap import (
    collect_paired_bootstrap,
    collect_sparse_posterior_bootstrap,
    confidence_aware_choice,
    order_key,
    paired_bootstrap,
    quantile,
    select_minimum,
)
from semanticflow.optimizer.models import BootstrapSettings, CorrelationSettings
from tests.m3_support import hashes, matrix_from_patterns

EXACT = CorrelationSettings(prior_strength=0.0, conditional_usage=False)


def run(matrix, original, *, estimator="correlation_aware", seed=11, iterations=200, **extra):
    return paired_bootstrap(
        matrix,
        estimator_name=estimator,
        original_order=original,
        objective="evaluations",
        evaluation_record_count=100,
        question_hashes=hashes(matrix.predicate_ids),
        settings=BootstrapSettings(iterations=iterations, seed=seed),
        correlation=EXACT,
        input_price_per_million=None,
        output_price_per_million=None,
        **extra,
    )


def near_tie_matrix():
    generator = random.Random(5)
    patterns = [
        (generator.random() < 0.5, generator.random() < 0.5, True) for _ in range(60)
    ]
    return matrix_from_patterns(["A", "B", "C"], patterns)


def test_bootstrap_is_deterministic_for_a_fixed_seed() -> None:
    matrix = near_tie_matrix()
    first = run(matrix, ["A", "B", "C"])
    second = run(matrix, ["A", "B", "C"])
    assert first == second
    other = run(matrix, ["A", "B", "C"], seed=12)
    assert other["seed"] == 12
    assert other["reference_savings_vs_original"] != first["reference_savings_vs_original"]


def test_resampling_keeps_each_review_row_intact() -> None:
    matrix = matrix_from_patterns(["A", "B"], [(True, False), (False, True)])
    resample = matrix.take([0, 0, 1])
    assert resample.rows[0] is resample.rows[1] is matrix.rows[0]
    assert resample.resampled
    # A and B are identical in every row, so any paired resample prices A->B->C and
    # B->A->C identically; resampling columns independently would break the tie.
    generator = random.Random(9)
    patterns = []
    for _ in range(40):
        shared = generator.random() < 0.4
        patterns.append((shared, shared, generator.random() < 0.7))
    twins = matrix_from_patterns(["A", "B", "C"], patterns)
    report = run(twins, ["A", "B", "C"], reference_order=["B", "A", "C"])
    savings = report["reference_savings_vs_original"]
    assert savings["lower"] == savings["upper"] == 0.0
    assert report["probability_reference_cheaper_than_original"] == 0.0
    distribution = report["objective_distribution_by_order"]
    assert distribution[order_key(["A", "B", "C"])] == distribution[order_key(["B", "A", "C"])]


def test_near_tie_is_reported_as_unstable() -> None:
    report = run(near_tie_matrix(), ["A", "B", "C"])
    assert report["stable_selection"] is False
    assert report["unstable_selection_evidence"] is not None
    assert sum(report["selection_frequency"].values()) == pytest.approx(1.0)
    assert report["modal_frequency"] < 0.95
    assert report["interpretation"] == "sampling_uncertainty_diagnostic"


def test_dominant_selective_predicate_is_stable() -> None:
    generator = random.Random(3)
    patterns = [
        (generator.random() < 0.9, generator.random() < 0.1, True) for _ in range(100)
    ]
    matrix = matrix_from_patterns(["A", "B", "C"], patterns)
    report = run(matrix, ["A", "B", "C"])
    assert report["full_sample_selected_order"][0] == "B"
    assert report["stable_selection"] is True
    assert report["probability_reference_cheaper_than_original"] >= 0.95
    assert report["reference_regret"]["mean"] >= 0.0


def test_independent_estimator_is_supported_by_the_bootstrap() -> None:
    report = run(near_tie_matrix(), ["A", "B", "C"], estimator="independent", iterations=50)
    assert report["estimator"] == "independent"
    assert report["iterations"] == 50


def test_confidence_aware_choice_falls_back_to_original_when_uncertain() -> None:
    settings = BootstrapSettings(confidence_level=0.95, minimum_relative_saving=0.01)
    common = {
        "minimum_order": ["B", "A"],
        "original_order": ["A", "B"],
        "minimum_objective": 95.0,
        "original_objective": 100.0,
        "settings": settings,
    }
    uncertain = {"reference_order": ["B", "A"], "probability_reference_cheaper_than_original": 0.77}
    order, explanation = confidence_aware_choice(bootstrap=uncertain, **common)
    assert order == ["A", "B"]
    assert explanation["reasons"] == ["savings_probability_below_confidence_level"]
    confident = {**uncertain, "probability_reference_cheaper_than_original": 0.99}
    order, explanation = confidence_aware_choice(bootstrap=confident, **common)
    assert order == ["B", "A"]
    assert explanation["decision"] == "adopt_minimum"
    immaterial = {**common, "minimum_objective": 99.5}
    order, explanation = confidence_aware_choice(bootstrap=confident, **immaterial)
    assert order == ["A", "B"]
    assert explanation["reasons"] == ["relative_saving_below_practical_threshold"]
    with pytest.raises(ValueError, match="reference order"):
        confidence_aware_choice(
            bootstrap={**confident, "reference_order": ["A", "B"]}, **common
        )


def test_tie_rule_prefers_original_then_lexicographic() -> None:
    costs = {order: 1.0 for order in itertools.permutations(["A", "B", "C"])}
    assert select_minimum(costs, ("C", "B", "A"), 1e-9) == ("C", "B", "A")
    costs[("C", "B", "A")] = 2.0
    assert select_minimum(costs, ("C", "B", "A"), 1e-9) == ("A", "B", "C")


def test_quantile_interpolates_linearly() -> None:
    assert quantile([3.0, 1.0, 2.0, 4.0], 0.5) == pytest.approx(2.5)
    assert quantile([5.0], 0.975) == 5.0
    with pytest.raises(ValueError):
        quantile([], 0.5)


def test_perfectly_correlated_predicates_keep_equal_order_costs() -> None:
    patterns = [(True, True, True), (False, False, True)] * 20
    matrix = matrix_from_patterns(["A", "B", "C"], patterns)
    report = run(matrix, ["A", "B", "C"], iterations=40)
    left = report["objective_distribution_by_order"][order_key(["A", "B", "C"])]
    right = report["objective_distribution_by_order"][order_key(["B", "A", "C"])]
    assert left == right


def test_zero_pass_predicate_is_placed_first() -> None:
    patterns = [(False, True, True) for _ in range(30)]
    matrix = matrix_from_patterns(["A", "B", "C"], patterns)
    report = run(matrix, ["B", "C", "A"], iterations=40)
    assert report["full_sample_selected_order"][0] == "A"
    assert report["probability_reference_cheaper_than_original"] == 1.0


def test_unequal_predicate_costs_are_resampled_with_the_row() -> None:
    patterns = [(index % 2 == 0, index % 3 == 0, True) for index in range(40)]
    matrix = matrix_from_patterns(
        ["A", "B", "C"], patterns, input_tokens={"A": 10, "B": 10, "C": 500}
    )
    report = paired_bootstrap(
        matrix,
        estimator_name="correlation_aware",
        original_order=["C", "A", "B"],
        objective="total_tokens",
        evaluation_record_count=100,
        question_hashes=hashes(["A", "B", "C"]),
        settings=BootstrapSettings(iterations=40, seed=11),
        correlation=EXACT,
        input_price_per_million=None,
        output_price_per_million=None,
    )
    assert report["full_sample_selected_order"][-1] == "C"


def test_sparse_conditionals_and_independent_predicates_do_not_crash() -> None:
    patterns = [(index < 3, index % 2 == 0, True) for index in range(40)]
    sparse = matrix_from_patterns(["A", "B", "C"], patterns)
    sparse_report = run(sparse, ["A", "B", "C"], iterations=25)
    assert sparse_report["iterations"] == 25
    generator = random.Random(4)
    independent = [
        (generator.random() < 0.5, generator.random() < 0.5, True) for _ in range(40)
    ]
    report = run(matrix_from_patterns(["A", "B", "C"], independent), ["A", "B", "C"], iterations=25)
    assert 0.0 <= report["probability_reference_cheaper_than_original"] <= 1.0


@pytest.mark.parametrize("positive_count", [0, 1, 2])
def test_sparse_posterior_bootstrap_is_not_degenerate_for_rare_predicates(
    positive_count: int,
) -> None:
    patterns = [
        (index < positive_count, index % 2 == 0, index % 3 == 0) for index in range(25)
    ]
    matrix = matrix_from_patterns(["A", "B", "C"], patterns)
    common = dict(
        estimator_name="correlation_aware",
        original_order=["B", "A", "C"],
        objective="evaluations",
        evaluation_record_count=100,
        question_hashes=hashes(["A", "B", "C"]),
        settings=BootstrapSettings(iterations=60, seed=91),
        correlation=CorrelationSettings(prior_strength=1.0, conditional_usage=False),
        input_price_per_million=None,
        output_price_per_million=None,
    )
    empirical = collect_paired_bootstrap(matrix, **common)
    posterior = collect_sparse_posterior_bootstrap(matrix, prior_strength=1.0, **common)
    candidate = ("A", "B", "C")
    if positive_count == 0:
        empirical_range = max(empirical["draws"][candidate]) - min(empirical["draws"][candidate])
        posterior_range = max(posterior["draws"][candidate]) - min(
            posterior["draws"][candidate]
        )
        # The empirical resample can vary because B/C vary, but A remains zero in every
        # resample.  Posterior prediction is wider because an unseen A pass remains possible.
        assert posterior_range > empirical_range
    assert len(set(posterior["draws"][candidate])) > 1
    assert posterior["sparse_data_treatment"]["draws_are_actual_provider_responses"] is False
    assert posterior["sparse_data_treatment"]["possible_pattern_count"] == 8


def test_sparse_posterior_bootstrap_handles_zero_conjunction_support_deterministically() -> None:
    matrix = matrix_from_patterns(
        ["A", "B", "C"],
        [(True, False, True), (False, True, True)] * 13,
    )
    kwargs = dict(
        estimator_name="correlation_aware",
        original_order=["A", "B", "C"],
        objective="evaluations",
        evaluation_record_count=100,
        question_hashes=hashes(["A", "B", "C"]),
        settings=BootstrapSettings(iterations=40, seed=113),
        correlation=CorrelationSettings(prior_strength=1.0, conditional_usage=False),
        input_price_per_million=None,
        output_price_per_million=None,
        prior_strength=1.0,
    )
    first = collect_sparse_posterior_bootstrap(matrix, **kwargs)
    second = collect_sparse_posterior_bootstrap(matrix, **kwargs)
    assert first["draws"] == second["draws"]
    assert first["sparse_data_treatment"]["observed_pattern_count"] == 2
