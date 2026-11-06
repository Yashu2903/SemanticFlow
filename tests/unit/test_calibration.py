import itertools

import pytest

from semanticflow.optimizer.calibration import (
    calibrate_orders,
    chain_conditionals,
    coverage_summary,
    cross_validated_calibration,
    observed_matrix_counts,
    quantity_names,
    two_proportion_test,
)
from semanticflow.optimizer.cardinality import build_estimator
from semanticflow.optimizer.models import CorrelationSettings
from tests.m3_support import hashes, matrix_from_patterns

IDS = ["A", "B", "C"]
EXACT = CorrelationSettings(prior_strength=0.0, conditional_usage=False)


def twins(size: int, prefix: str = "p"):
    # A and B always agree, C always passes: independence badly underestimates stage 3.
    return matrix_from_patterns(
        IDS, [(index % 2 == 0, index % 2 == 0, True) for index in range(size)], prefix=prefix
    )


def estimator(name, matrix, correlation=EXACT):
    return build_estimator(
        name,
        matrix,
        question_hashes=hashes(IDS),
        correlation=correlation,
        input_price_per_million=None,
        output_price_per_million=None,
    )


def calibrate(name, matrix, observations, records, *, iterations=200, seed=11):
    return calibrate_orders(
        matrix,
        observations,
        estimator_name=name,
        question_hashes=hashes(IDS),
        correlation=EXACT,
        record_count=records,
        iterations=iterations,
        seed=seed,
    )


@pytest.mark.parametrize("name", ["independent", "correlation_aware"])
def test_chain_conditionals_reproduce_estimator_reach_probabilities(name) -> None:
    patterns = list(itertools.product((False, True), repeat=3)) * 3 + [(True, True, True)] * 4
    matrix = matrix_from_patterns(IDS, patterns)
    fitted = estimator(name, matrix, CorrelationSettings())
    for order in itertools.permutations(IDS):
        conditionals = chain_conditionals(fitted, order)
        reach = 1.0
        for stage, conditional in zip(fitted.stage_estimates(order), conditionals, strict=True):
            assert stage.reach_probability == pytest.approx(reach)
            reach *= conditional


def test_observed_matrix_counts_are_later_stage_inputs_then_matches() -> None:
    matrix = twins(10)
    assert quantity_names(IDS) == [
        "stage_2_input_records",
        "stage_3_input_records",
        "final_matches",
    ]
    assert observed_matrix_counts(matrix, IDS) == [5, 5, 5]
    assert observed_matrix_counts(matrix, ["C", "A", "B"]) == [10, 5, 5]


def test_correlated_twins_are_covered_only_by_the_correlation_aware_intervals() -> None:
    profile = twins(100)
    held_out = twins(400, prefix="e")
    observations = {tuple(IDS): observed_matrix_counts(held_out, IDS)}
    aware = calibrate("correlation_aware", profile, observations, 400)[0]
    independent = calibrate("independent", profile, observations, 400)[0]
    assert all(item["covered"]["90"] for item in aware["quantities"])
    assert [item["covered"]["90"] for item in independent["quantities"]] == [True, False, False]
    # Independence predicts 100 records reach stage 3 where 200 do: every draw is below.
    assert independent["quantities"][1]["point_prediction"] == pytest.approx(100.0)
    assert independent["quantities"][1]["pit"] == 1.0
    transition = independent["stage_conditionals"][1]
    assert transition["predicted_conditional_pass_rate"] == pytest.approx(0.5)
    assert transition["observed_conditional_pass_rate"] == 1.0
    assert transition["predicted_within_observed_wilson_95"] is False
    assert aware["stage_conditionals"][1]["predicted_within_observed_wilson_95"] is True
    summary = coverage_summary([independent])
    assert summary["all_quantities"]["coverage_90"] == pytest.approx(1 / 3)
    assert summary["by_quantity"]["final_matches"]["pit_above_0_95"] == 1.0


def test_calibration_is_deterministic_for_a_seed_and_validates_observations() -> None:
    profile = twins(60)
    observations = {tuple(IDS): [30, 30, 30]}
    first = calibrate("correlation_aware", profile, observations, 60, iterations=50)
    second = calibrate("correlation_aware", profile, observations, 60, iterations=50)
    assert first == second
    other = calibrate("correlation_aware", profile, observations, 60, iterations=50, seed=12)
    assert other != first
    with pytest.raises(ValueError, match="needs 3 observed counts"):
        calibrate("correlation_aware", profile, {tuple(IDS): [30, 30]}, 60)


def test_cross_validated_calibration_uses_only_the_given_matrix() -> None:
    patterns = list(itertools.product((False, True), repeat=3)) * 5
    matrix = matrix_from_patterns(IDS, patterns)
    result = cross_validated_calibration(
        matrix,
        question_hashes=hashes(IDS),
        correlation=CorrelationSettings(),
        folds=4,
        fold_seed=7,
        iterations=30,
        seed=3,
    )
    for name in ("independent", "correlation_aware"):
        summary = result["summary"][name]["all_quantities"]
        assert summary["checks"] == 4 * 6 * 3
        assert 0.0 <= summary["coverage_50"] <= summary["coverage_90"] <= 1.0
        assert 0.0 <= summary["mean_pit"] <= 1.0
    assert {record["fold"] for record in result["records"]["correlation_aware"]} == {0, 1, 2, 3}
    with pytest.raises(ValueError, match="at least two folds"):
        cross_validated_calibration(
            matrix,
            question_hashes=hashes(IDS),
            correlation=CorrelationSettings(),
            folds=1,
            fold_seed=7,
            iterations=30,
            seed=3,
        )


def test_two_proportion_test_matches_the_profiling_versus_held_out_marginal() -> None:
    result = two_proportion_test(28, 200, 91, 800)
    assert result["difference"] == pytest.approx(0.02625)
    assert result["z"] == pytest.approx(1.0255, abs=1e-3)
    assert result["two_sided_p_value"] == pytest.approx(0.3051, abs=1e-3)
    assert two_proportion_test(0, 10, 0, 10)["two_sided_p_value"] == 1.0
    with pytest.raises(ValueError):
        two_proportion_test(1, 0, 1, 10)
