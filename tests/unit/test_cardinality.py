import itertools
import random

import pytest

from semanticflow.optimizer.cardinality import (
    CorrelationAwareEstimator,
    IndependentEstimator,
    exact_matrix_cardinalities,
    predicate_statistics_from_matrix,
)
from semanticflow.optimizer.cost_model import estimate_plan
from semanticflow.optimizer.matrix import MatrixRow, ProfileMatrix, joint_statistics
from semanticflow.optimizer.models import CorrelationSettings, PlannerSettings
from tests.m3_support import hashes, matrix_from_patterns, trace

EXACT = CorrelationSettings(prior_strength=0.0, conditional_usage=False)


def reach(estimator, order: list[str], records: int = 100) -> list[float]:
    return [records * item.reach_probability for item in estimator.stage_estimates(order)]


def independent_for(matrix: ProfileMatrix) -> IndependentEstimator:
    return IndependentEstimator(
        predicate_statistics_from_matrix(
            matrix,
            question_hashes=hashes(matrix.predicate_ids),
            input_price_per_million=None,
            output_price_per_million=None,
        )
    )


def pairwise(statistics: dict, first: str, second: str) -> dict:
    return next(item for item in statistics["pairwise"] if item["predicates"] == [first, second])


def test_independent_predicates_have_zero_phi_and_both_estimators_agree() -> None:
    patterns = list(itertools.product((False, True), repeat=2)) * 25
    matrix = matrix_from_patterns(["A", "B"], patterns)
    stats = pairwise(joint_statistics(matrix), "A", "B")
    assert stats["phi"] == pytest.approx(0.0)
    assert stats["lift"] == pytest.approx(1.0)
    assert reach(CorrelationAwareEstimator(matrix, EXACT), ["A", "B"]) == pytest.approx(
        reach(independent_for(matrix), ["A", "B"])
    )


def test_strongly_correlated_predicates_break_independence() -> None:
    patterns = [(True, True)] * 18 + [(True, False)] * 2 + [(False, True)] * 2 + [
        (False, False)
    ] * 78
    matrix = matrix_from_patterns(["A", "B", "C"], [(*p, True) for p in patterns])
    stats = pairwise(joint_statistics(matrix), "A", "B")
    assert stats["phi"] > 0.8
    assert stats["p_B_given_A"] == pytest.approx(0.9)
    assert stats["p_B_given_A_support"] == 20
    correlation = reach(CorrelationAwareEstimator(matrix, EXACT), ["A", "B", "C"])
    independent = reach(independent_for(matrix), ["A", "B", "C"])
    assert correlation == pytest.approx([100, 20, 18])
    assert independent == pytest.approx([100, 20, 4])


def test_perfectly_correlated_predicates_are_estimated_exactly() -> None:
    patterns = [(True, True, True)] * 10 + [(False, False, True)] * 30
    matrix = matrix_from_patterns(["A", "B", "C"], patterns)
    assert pairwise(joint_statistics(matrix), "A", "B")["phi"] == pytest.approx(1.0)
    estimator = CorrelationAwareEstimator(matrix, EXACT)
    for order in itertools.permutations(["A", "B", "C"]):
        assert reach(estimator, list(order), records=40) == pytest.approx(
            exact_matrix_cardinalities(matrix, order)
        )


def test_negatively_correlated_predicates_and_smoothing() -> None:
    patterns = [(True, False)] * 50 + [(False, True)] * 50
    matrix = matrix_from_patterns(["A", "B"], patterns)
    stats = pairwise(joint_statistics(matrix), "A", "B")
    assert stats["phi"] == pytest.approx(-1.0)
    assert stats["both_pass"] == 0
    exact = CorrelationAwareEstimator(matrix, EXACT)
    assert reach(exact, ["A", "B"]) == pytest.approx([100, 50])
    final = exact.conditional(matrix.index("B"), [matrix.index("A")])
    assert final[:3] == (0.0, 0, 50)
    smoothed = CorrelationAwareEstimator(matrix, CorrelationSettings(prior_strength=1.0))
    estimate, passing, support, method = smoothed.conditional(1, [0])
    assert (passing, support, method) == (0, 50, "smoothed")
    assert estimate == pytest.approx((0 + 1.0 * (50 + 0.5) / 101) / 51)


def test_zero_selectivity_predicate_zeroes_later_stages() -> None:
    patterns = [(False, True)] * 30
    matrix = matrix_from_patterns(["A", "B"], patterns)
    assert reach(CorrelationAwareEstimator(matrix, EXACT), ["A", "B"]) == [100, 0]
    assert reach(independent_for(matrix), ["A", "B"]) == [100, 0]
    smoothed = CorrelationAwareEstimator(matrix, CorrelationSettings(prior_strength=1.0))
    assert reach(smoothed, ["A", "B"])[1] == pytest.approx(100 * 0.5 / 31)


def test_sparse_and_unseen_conditioning_sets_back_off_and_are_low_confidence() -> None:
    patterns = [(True, True, True)] * 2 + [(True, False, False)] * 3 + [(False, False, True)] * 45
    matrix = matrix_from_patterns(["A", "B", "C"], patterns)
    settings = CorrelationSettings(prior_strength=1.0, minimum_conditional_support=20)
    estimator = CorrelationAwareEstimator(matrix, settings)
    stages = estimator.stage_estimates(["A", "B", "C"])
    # Stage 3 reach uses P(B | A), which only five profiled rows support.
    assert [item.minimum_chain_support for item in stages] == [50, 50, 5]
    assert [item.confidence for item in stages] == ["high", "high", "low"]
    empty = matrix_from_patterns(["A", "B", "C"], [(False, True, True)] * 10)
    unseen = CorrelationAwareEstimator(empty, settings)
    estimate, passing, support, method = unseen.conditional(2, [0, 1])
    assert (passing, support, method) == (0, 0, "backoff_unseen_prefix")
    assert 0.0 < estimate < 1.0


def test_four_predicates_in_arbitrary_order_match_exact_matrix_replay() -> None:
    generator = random.Random(7)
    ids = ["A", "B", "C", "D"]
    patterns = []
    for _ in range(120):
        latent = generator.random() < 0.3
        patterns.append(
            tuple(
                (generator.random() < 0.8) if latent else (generator.random() < 0.2)
                for _ in ids
            )
        )
    matrix = matrix_from_patterns(ids, patterns)
    estimator = CorrelationAwareEstimator(matrix, EXACT)
    for order in itertools.permutations(ids):
        assert reach(estimator, list(order), records=matrix.size) == pytest.approx(
            exact_matrix_cardinalities(matrix, order)
        )


def test_conditional_usage_reflects_rows_that_reach_the_stage() -> None:
    rows = [
        MatrixRow("r1", (True, True), (10, 100), (0, 0), (0.1, 0.1)),
        MatrixRow("r2", (False, True), (10, 10), (0, 0), (0.1, 0.1)),
    ]
    matrix = ProfileMatrix(["A", "B"], rows)
    usage = CorrelationSettings(prior_strength=0.0, usage_prior_strength=0.0)
    stage = CorrelationAwareEstimator(matrix, usage).stage_estimates(["A", "B"])[1]
    assert stage.average_input_tokens == pytest.approx(100)
    plain = CorrelationAwareEstimator(matrix, EXACT).stage_estimates(["A", "B"])[1]
    assert plain.average_input_tokens == pytest.approx(55)


def test_independent_estimator_reproduces_default_cost_model_exactly() -> None:
    matrix = matrix_from_patterns(
        ["A", "B", "C"],
        [(True, False, True), (True, True, True), (False, True, False), (True, True, False)],
        input_tokens={"A": 40, "B": 7, "C": 13},
    )
    statistics = predicate_statistics_from_matrix(
        matrix,
        question_hashes=hashes(["A", "B", "C"]),
        input_price_per_million=0.042,
        output_price_per_million=0.0,
    )
    for order in itertools.permutations(["A", "B", "C"]):
        default = estimate_plan(
            list(order), statistics, input_record_count=800, objective="estimated_cost_usd",
            original_order=["A", "B", "C"],
        )
        explicit = estimate_plan(
            list(order), statistics, input_record_count=800, objective="estimated_cost_usd",
            original_order=["A", "B", "C"], estimator=IndependentEstimator(statistics),
        )
        assert default.objective_value == explicit.objective_value
        assert default.estimated_evaluations == explicit.estimated_evaluations


def test_incomplete_matrix_requires_explicit_complete_case_opt_in() -> None:
    rows = [
        MatrixRow("r1", (True, None), (10, None), (0, None), (0.1, None)),
        MatrixRow("r2", (True, True), (10, 10), (0, 0), (0.1, 0.1)),
    ]
    matrix = ProfileMatrix(["A", "B"], rows)
    with pytest.raises(ValueError, match="complete profiling matrix"):
        CorrelationAwareEstimator(matrix)
    estimator = CorrelationAwareEstimator(matrix, CorrelationSettings(allow_incomplete_matrix=True))
    assert estimator.sample_size == 1


def test_matrix_rejects_held_out_and_out_of_split_traces() -> None:
    profiling_traces = [trace("p1", "A", True), trace("p1", "B", False)]
    matrix = ProfileMatrix.from_traces(
        profiling_traces, ["A", "B"], review_ids=["p1"], forbidden_review_ids=["e1"]
    )
    assert matrix.complete
    with pytest.raises(ValueError, match="held-out"):
        ProfileMatrix.from_traces(
            [*profiling_traces, trace("e1", "A", True)],
            ["A", "B"],
            review_ids=["p1"],
            forbidden_review_ids=["e1"],
        )
    with pytest.raises(ValueError, match="outside the profiling split"):
        ProfileMatrix.from_traces(
            [*profiling_traces, trace("x9", "A", True)], ["A", "B"], review_ids=["p1"]
        )
    with pytest.raises(ValueError, match="duplicate"):
        ProfileMatrix.from_traces(
            [*profiling_traces, trace("p1", "A", False)], ["A", "B"], review_ids=["p1"]
        )
    with pytest.raises(ValueError, match="overlap"):
        ProfileMatrix.from_traces(
            profiling_traces, ["A", "B"], review_ids=["p1"], forbidden_review_ids=["p1"]
        )


def test_estimator_selection_requires_matching_cost_model_version() -> None:
    assert PlannerSettings().cardinality_estimator == "independent"
    with pytest.raises(ValueError):
        PlannerSettings(cardinality_estimator="correlation_aware")
    PlannerSettings(
        cardinality_estimator="correlation_aware", cost_model_version="correlation-aware-and-v1"
    )
