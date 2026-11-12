import itertools

import pytest

from semanticflow.optimizer.m45_calibration import (
    _exact_cost_usd,
    _plan_ranking,
    calibrate_bootstrap_fold,
    calibrate_cardinality_fold,
    evaluate_policy_on_fold,
    fingerprint_mapping,
    freeze_configuration,
)
from semanticflow.optimizer.models import (
    CorrelationSettings,
    OptimizerSettings,
)
from semanticflow.optimizer.profiling_splits import split_matrix
from tests.m3_support import hashes, matrix_from_patterns

IDS = ["A", "B", "C"]


def correlated_matrix(size: int = 40):
    patterns = [
        (index < size // 2, index < size // 2, index % 2 == 0) for index in range(size)
    ]
    return matrix_from_patterns(
        IDS, patterns, input_tokens={"A": 10, "B": 10, "C": 40}, prefix="c"
    )


def test_exact_cost_short_circuits_on_rejection() -> None:
    matrix = matrix_from_patterns(
        ["A", "B"],
        [(False, True), (True, True)],
        input_tokens={"A": 100, "B": 100},
        prefix="e",
    )
    cost = _exact_cost_usd(matrix, ["A", "B"], input_price=1.0, output_price=0.0)
    # First row evaluates only A; second evaluates A and B.
    assert cost == pytest.approx(300 / 1_000_000)


def test_cardinality_fold_has_no_train_validation_leak() -> None:
    matrix = correlated_matrix()
    train, validation, meta = split_matrix(matrix, folds=4, fold=0, seed=11)
    assert meta["held_out_live_reviews_used"] is False
    report = calibrate_cardinality_fold(
        train,
        validation,
        question_hashes=hashes(IDS),
        correlation=CorrelationSettings(prior_strength=1.0, conditional_usage=False),
        prior_strength=1.0,
        iterations=10,
        seed=3,
    )
    assert "estimators" in report
    assert "correlation_aware" in report["estimators"]
    ranking = report["estimators"]["correlation_aware"]["plan_ranking"]
    assert 0.0 <= ranking["pairwise_concordance"] <= 1.0
    assert ranking["selection_regret_tokens"] >= 0.0


def test_bootstrap_coverage_keys_and_labeling() -> None:
    matrix = correlated_matrix()
    train, validation, _ = split_matrix(matrix, folds=4, fold=1, seed=11)
    report = calibrate_bootstrap_fold(
        train,
        validation,
        question_hashes=hashes(IDS),
        correlation=CorrelationSettings(prior_strength=1.0, conditional_usage=False),
        iterations=30,
        seed=5,
        input_price=1.0,
        output_price=0.0,
    )
    assert report["not_a_calibrated_production_probability"] is True
    assert "0.95" in report["interval_coverage"]


def test_plan_ranking_reports_concordance_and_nonnegative_regret() -> None:
    patterns = [(True, True, index % 2 == 0) for index in range(24)]
    matrix = matrix_from_patterns(
        IDS, patterns, input_tokens={"A": 10, "B": 10, "C": 200}, prefix="r"
    )
    train, validation, _ = split_matrix(matrix, folds=3, fold=0, seed=2)
    ranking = _plan_ranking(
        train,
        validation,
        estimator_name="correlation_aware",
        question_hashes=hashes(IDS),
        correlation=CorrelationSettings(prior_strength=0.0, conditional_usage=False),
        input_price=1.0,
        output_price=0.0,
        objective="total_tokens",
    )
    assert 0.0 <= ranking["pairwise_concordance"] <= 1.0
    assert ranking["selection_regret_tokens"] >= 0.0
    assert len(ranking["predicted_selected_order"]) == 3


def test_policy_fold_fixed_and_risk_aware() -> None:
    matrix = correlated_matrix(36)
    train, validation, _ = split_matrix(matrix, folds=3, fold=0, seed=4)
    optimizer = OptimizerSettings.model_validate(
        {
            "planner": {
                "objective": "estimated_cost_usd",
                "cardinality_estimator": "correlation_aware",
                "cost_model_version": "correlation-aware-and-v1",
                "selection_policy": "risk_aware",
            },
            "correlation": {"prior_strength": 1.0, "conditional_usage": False},
            "bootstrap": {"iterations": 20, "seed": 8},
            "risk": {
                "minimum_relative_saving": 0.01,
                "minimum_saving_probability": 0.9,
            },
            "adaptive": {"checkpoints": [8, 16, 24]},
            "joint": {
                "bootstrap_iterations": 20,
                "voi_scenarios": 4,
                "seed": 9,
            },
        }
    )
    fixed = evaluate_policy_on_fold(
        train,
        validation,
        original=IDS,
        question_hashes=hashes(IDS),
        optimizer=optimizer,
        policy="fixed",
        horizon=10,
        input_price=1.0,
        output_price=0.0,
    )
    assert fixed["selected_order"] == IDS
    assert fixed["profiling_cost_usd"] == 0.0
    risk = evaluate_policy_on_fold(
        train,
        validation,
        original=IDS,
        question_hashes=hashes(IDS),
        optimizer=optimizer,
        policy="risk_aware",
        horizon=10,
        input_price=1.0,
        output_price=0.0,
    )
    assert risk["action"] in {"KEEP_ORIGINAL", "ADOPT_REORDER", "REUSE_EXISTING_PLAN"}
    assert risk["validation_regret_usd"] >= 0.0


def test_freeze_configuration_records_fingerprint_without_held_out_use() -> None:
    optimizer = OptimizerSettings()
    selection = {
        "recommended_default": "existing_adaptive",
        "adopt_tuned_as_default": False,
        "selection_objective": "maximize profiling-validation cold-start net",
        "best_joint_candidate": {
            "prior_strength": 1.0,
            "minimum_saving_probability": 0.95,
            "minimum_relative_saving": 0.01,
            "checkpoints": [25, 50, 100, 150, 200],
            "voi_decision_rule": "point_estimate",
            "uncertainty_contraction_exponent": 0.5,
        },
    }
    frozen, manifest = freeze_configuration(
        optimizer, selection, calibration_fingerprint="abc"
    )
    assert manifest["held_out_live_reviews_used_for_selection"] is False
    assert manifest["config_fingerprint"] == fingerprint_mapping(manifest["parameters"])
    assert frozen.risk.minimum_saving_probability == 0.95
    assert frozen.joint.bootstrap_iterations == 2000


def test_zero_and_perfect_correlation_matrices_do_not_crash_ranking() -> None:
    zero = matrix_from_patterns(
        IDS, [(False, False, False)] * 12, input_tokens={"A": 5, "B": 5, "C": 5}
    )
    perfect = matrix_from_patterns(
        IDS, [(True, True, True)] * 12, input_tokens={"A": 5, "B": 5, "C": 5}
    )
    for matrix in (zero, perfect):
        train, validation, _ = split_matrix(matrix, folds=3, fold=0, seed=1)
        ranking = _plan_ranking(
            train,
            validation,
            estimator_name="independent",
            question_hashes=hashes(IDS),
            correlation=CorrelationSettings(prior_strength=1.0, conditional_usage=False),
            input_price=1.0,
            output_price=0.0,
            objective="total_tokens",
        )
        assert ranking["selection_regret_tokens"] >= 0.0
        assert len(list(itertools.permutations(IDS))) == 6


def test_negative_correlation_and_sparse_support_ranking() -> None:
    patterns = []
    for index in range(30):
        a = index % 2 == 0
        b = not a
        c = index % 3 == 0
        patterns.append((a, b, c))
    matrix = matrix_from_patterns(
        IDS, patterns, input_tokens={"A": 20, "B": 20, "C": 5}, prefix="n"
    )
    train, validation, _ = split_matrix(matrix, folds=3, fold=1, seed=6)
    ranking = _plan_ranking(
        train,
        validation,
        estimator_name="correlation_aware",
        question_hashes=hashes(IDS),
        correlation=CorrelationSettings(prior_strength=0.5, conditional_usage=False),
        input_price=1.0,
        output_price=0.0,
        objective="total_tokens",
    )
    assert ranking["selection_regret_tokens"] >= 0.0
    assert ranking["predicted_selected_order"]


def test_fingerprint_mapping_is_order_insensitive() -> None:
    left = fingerprint_mapping({"b": 2, "a": {"z": 1, "y": 0}})
    right = fingerprint_mapping({"a": {"y": 0, "z": 1}, "b": 2})
    assert left == right


def test_joint_policy_never_sees_validation_rows_for_selection() -> None:
    matrix = correlated_matrix(36)
    train, validation, _ = split_matrix(matrix, folds=3, fold=0, seed=4)
    optimizer = OptimizerSettings.model_validate(
        {
            "planner": {
                "objective": "estimated_cost_usd",
                "cardinality_estimator": "correlation_aware",
                "cost_model_version": "correlation-aware-and-v1",
                "selection_policy": "risk_aware",
            },
            "correlation": {"prior_strength": 1.0, "conditional_usage": False},
            "bootstrap": {"iterations": 20, "seed": 8},
            "risk": {
                "minimum_relative_saving": 0.01,
                "minimum_saving_probability": 0.9,
            },
            "adaptive": {"checkpoints": [8, 16, 24]},
            "joint": {
                "bootstrap_iterations": 20,
                "voi_scenarios": 4,
                "seed": 9,
            },
        }
    )
    result = evaluate_policy_on_fold(
        train,
        validation,
        original=IDS,
        question_hashes=hashes(IDS),
        optimizer=optimizer,
        policy="joint",
        horizon=10,
        input_price=1.0,
        output_price=0.0,
    )
    assert result["profiling_decisions"] <= train.size * len(IDS)
    assert result["cold_start_net_usd"] == pytest.approx(
        result["horizon"] * result["validation_saving_usd"] - result["profiling_cost_usd"]
    )
