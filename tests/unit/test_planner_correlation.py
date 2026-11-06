import pytest

from semanticflow.optimizer.cardinality import exact_matrix_cardinalities
from semanticflow.optimizer.models import OptimizerSettings
from semanticflow.optimizer.planner import CostBasedPlanner
from tests.m3_support import matrix_from_patterns, profile_for, query, split_for

IDS = ["X", "Y", "Z"]
TOKENS = {"X": 10, "Y": 10, "Z": 100}


def correlated_matrix():
    # X and Y are the same predicate in disguise; Z is independent of both.
    patterns = [(index < 10, index < 10, index % 2 == 0) for index in range(20)]
    return matrix_from_patterns(IDS, patterns, input_tokens=TOKENS)


def settings(**planner) -> OptimizerSettings:
    return OptimizerSettings.model_validate(
        {
            "planner": {"objective": "total_tokens", **planner},
            "correlation": {"prior_strength": 0.0, "conditional_usage": False},
            "bootstrap": {"iterations": 200, "seed": 17},
        }
    )


CORRELATION = {
    "cardinality_estimator": "correlation_aware",
    "cost_model_version": "correlation-aware-and-v1",
}


def exact_tokens_per_record(matrix, order) -> float:
    stages = exact_matrix_cardinalities(matrix, order)
    tokens = sum(count * TOKENS[item] for count, item in zip(stages, order, strict=True))
    return tokens / matrix.size


def test_correlation_changes_the_plan_and_matches_exact_replay() -> None:
    matrix = correlated_matrix()
    profile, split = profile_for(matrix), split_for(matrix, evaluation_size=100)
    independent = CostBasedPlanner(settings()).plan(query(IDS), profile, split)
    correlated = CostBasedPlanner(settings(**CORRELATION)).plan(
        query(IDS), profile, split, matrix=matrix
    )
    assert independent.selected_predicate_order == ["X", "Y", "Z"]
    assert independent.selected_estimated_objective == pytest.approx(100 * 40)
    assert correlated.selected_predicate_order == ["X", "Z", "Y"]
    assert correlated.selected_estimated_objective == pytest.approx(100 * 62.5)
    assert correlated.original_estimated_objective == pytest.approx(100 * 65)
    assert exact_tokens_per_record(matrix, ["X", "Y", "Z"]) == pytest.approx(65)
    assert exact_tokens_per_record(matrix, ["X", "Z", "Y"]) == pytest.approx(62.5)
    assert correlated.cardinality_estimator == "correlation_aware"
    assert correlated.joint_matrix_fingerprint == matrix.fingerprint()
    assert correlated.candidates[0].stage_estimates[1]["method"] == "prefix_empirical"
    assert independent.cardinality_estimator == "independent"
    assert independent.joint_matrix_fingerprint is None


def test_correlation_aware_planning_requires_a_validated_matrix() -> None:
    matrix = correlated_matrix()
    profile, split = profile_for(matrix), split_for(matrix)
    planner = CostBasedPlanner(settings(**CORRELATION))
    with pytest.raises(ValueError, match="decision matrix"):
        planner.plan(query(IDS), profile, split)
    other = matrix_from_patterns(IDS, [(True, True, True)] * 20, input_tokens=TOKENS)
    with pytest.raises(ValueError, match="disagree with statistics"):
        planner.plan(query(IDS), profile, split, matrix=other)
    leaked = matrix_from_patterns(IDS, [(True, True, True)] * 20, prefix="e")
    with pytest.raises(ValueError, match="exactly the profiling split"):
        planner.plan(query(IDS), profile, split, matrix=leaked)


def test_confidence_aware_policy_adopts_credible_savings_and_documents_why() -> None:
    matrix = correlated_matrix()
    profile, split = profile_for(matrix), split_for(matrix)
    plan = CostBasedPlanner(
        settings(**CORRELATION, selection_policy="confidence_aware")
    ).plan(query(IDS), profile, split, matrix=matrix)
    explanation = plan.selection_explanation
    assert explanation["decision"] == "adopt_minimum"
    assert explanation["probability_minimum_cheaper_than_original"] >= 0.95
    assert plan.selected_predicate_order == ["X", "Z", "Y"]
    assert plan.minimum_objective_predicate_order == ["X", "Z", "Y"]


def test_confidence_aware_policy_keeps_original_when_saving_is_immaterial() -> None:
    matrix = correlated_matrix()
    profile, split = profile_for(matrix), split_for(matrix)
    optimizer = OptimizerSettings.model_validate(
        {
            "planner": {
                "objective": "total_tokens",
                **CORRELATION,
                "selection_policy": "confidence_aware",
            },
            "correlation": {"prior_strength": 0.0, "conditional_usage": False},
            "bootstrap": {"iterations": 200, "seed": 17, "minimum_relative_saving": 0.1},
        }
    )
    plan = CostBasedPlanner(optimizer).plan(query(IDS), profile, split, matrix=matrix)
    assert plan.selected_predicate_order == ["X", "Y", "Z"]
    assert plan.minimum_objective_predicate_order == ["X", "Z", "Y"]
    assert not plan.reordered
    assert plan.selection_explanation["reasons"] == ["relative_saving_below_practical_threshold"]
    assert plan.candidates[0].selected
    assert plan.candidates[0].predicate_order == ["X", "Y", "Z"]
