import pytest

from semanticflow.optimizer.adaptive import sampling_order
from semanticflow.optimizer.joint import choose_joint_action, simulate_joint_adaptive_risk
from semanticflow.optimizer.matrix import MatrixRow, ProfileMatrix
from semanticflow.optimizer.models import (
    BootstrapSettings,
    CorrelationSettings,
    EconomicsSettings,
    JointAdaptiveSettings,
    RiskSettings,
)
from tests.m3_support import hashes

IDS = ["A", "B", "C"]
SEED = 4049


def matrix_by_position(pattern) -> ProfileMatrix:
    review_ids = [f"j{index:03}" for index in range(40)]
    ranked = sampling_order(review_ids, SEED)
    position = {review_id: index for index, review_id in enumerate(ranked)}
    return ProfileMatrix(
        IDS,
        [
            MatrixRow(
                review_id=review_id,
                outcomes=pattern(position[review_id]),
                input_tokens=(10, 10, 10),
                output_tokens=(0, 0, 0),
                latency_seconds=(0.1, 0.1, 0.1),
            )
            for review_id in review_ids
        ],
    )


def run(matrix, *, horizon=10, statistics_available=False, reusable_order=None, **extra):
    return simulate_joint_adaptive_risk(
        matrix,
        estimator_name="correlation_aware",
        original_order=IDS,
        objective="estimated_cost_usd",
        evaluation_record_count=100,
        question_hashes=hashes(IDS),
        checkpoints=[10, 20, 40],
        sampling_seed=SEED,
        bootstrap=BootstrapSettings(iterations=30, seed=7),
        correlation=CorrelationSettings(prior_strength=1.0, conditional_usage=False),
        risk=RiskSettings(minimum_relative_saving=0.01, minimum_saving_probability=0.9),
        economics=EconomicsSettings(expected_executions=horizon),
        joint=JointAdaptiveSettings(
            bootstrap_iterations=30,
            seed=17,
            voi_scenarios=5,
        ),
        input_price_per_million=1.0,
        output_price_per_million=0.0,
        expected_executions=horizon,
        savings_upper_bound_per_execution=0.001,
        estimated_cost_per_decision=0.00001,
        statistics_available=statistics_available,
        reusable_order=reusable_order,
        **extra,
    )


def mixed(position: int) -> tuple[bool, bool, bool]:
    return (position % 2 == 0, position % 3 == 0, True)


def test_zero_expected_executions_skips_before_the_first_sample() -> None:
    result = run(matrix_by_position(mixed), horizon=0)
    assert result["action"] == "SKIP_PROFILING"
    assert result["profiling_decisions_consumed"] == 0
    assert result["trajectory"] == []


def test_reusable_statistics_reuse_the_existing_plan_without_profiling() -> None:
    result = run(
        matrix_by_position(mixed),
        statistics_available=True,
        reusable_order=["B", "A", "C"],
    )
    assert result["action"] == "REUSE_EXISTING_PLAN"
    assert result["selected_order"] == ["B", "A", "C"]
    assert result["profiling_reviews_consumed"] == 0


def test_no_lookahead_first_checkpoint_is_invariant_to_future_outcomes() -> None:
    left = matrix_by_position(
        lambda position: mixed(position) if position < 10 else (False, False, False)
    )
    right = matrix_by_position(
        lambda position: mixed(position) if position < 10 else (True, True, True)
    )
    first = run(left)
    second = run(right)
    assert first["trajectory"][0] == second["trajectory"][0]
    assert first["trajectory"][0]["future_rows_accessible_to_controller"] is False
    assert first["trajectory"][0]["value_of_additional_profiling"][
        "uses_stored_future_rows"
    ] is False


def test_joint_replay_is_deterministic() -> None:
    matrix = matrix_by_position(mixed)
    assert run(matrix) == run(matrix)


def test_voi_trace_has_structured_arithmetic_and_checkpoint_local_uncertainty() -> None:
    result = run(matrix_by_position(mixed))
    checkpoint = result["trajectory"][0]
    value = checkpoint["value_of_additional_profiling"]
    assert value["uses_only_current_checkpoint_evidence"] is True
    assert value["uses_stored_future_rows"] is False
    assert 0.0 <= value["probability_next_checkpoint_changes_selected_plan"] <= 1.0
    assert value["expected_reduction_in_decision_loss_per_execution"] == pytest.approx(
        value["estimated_benefit_from_plan_changes_per_execution"]
        + value["estimated_benefit_without_plan_change_per_execution"]
    )
    assert value["estimated_benefit_without_plan_change_per_execution"] == pytest.approx(0.0)
    assert value["net_value_of_information_usd"] == pytest.approx(
        value["gross_value_of_information_usd"]
        - value["incremental_profiling_cost_usd"]
    )
    uncertainty = checkpoint["statistical_uncertainty"]
    assert uncertainty["decision_input"] is False
    assert set(uncertainty["marginal_selectivity"]) == set(IDS)
    assert all(
        item["trials"] == checkpoint["profiled_reviews"]
        for item in uncertainty["marginal_selectivity"].values()
    )
    assert checkpoint["controller_explanation"]["action"] == checkpoint[
        "controller_action"
    ]


def test_budget_exhaustion_forces_conservative_fallback() -> None:
    result = run(matrix_by_position(mixed), maximum_profiling_decisions=30)
    assert result["profiling_reviews_consumed"] == 10
    if result["action"] != "ADOPT_REORDER":
        assert result["action"] == "KEEP_ORIGINAL"
        assert "profiling_decision_budget_exhausted" in result["reason"]


def test_positive_and_negative_value_of_information_actions() -> None:
    common = {
        "risk_decision": "KEEP_ORIGINAL",
        "risk_selected_order": IDS,
        "next_sample_size": 20,
    }
    positive = {
        **common,
        "value_of_additional_profiling": {
            "incremental_profiling_cost_usd": 0.1,
            "net_value_of_information_usd": 0.01,
        },
    }
    action, _, reason = choose_joint_action(
        positive,
        original_order=IDS,
        profiling_cost_incurred_usd=0.1,
        profiling_budget_usd=None,
        maximum_profiling_decisions=None,
        minimum_voi_usd=0.0,
    )
    assert action == "PROFILE_MORE"
    assert reason == ["positive_value_of_additional_profiling"]
    negative = {
        **positive,
        "value_of_additional_profiling": {
            "incremental_profiling_cost_usd": 0.1,
            "net_value_of_information_usd": -0.01,
        },
    }
    action, selected, _ = choose_joint_action(
        negative,
        original_order=IDS,
        profiling_cost_incurred_usd=0.1,
        profiling_budget_usd=None,
        maximum_profiling_decisions=None,
        minimum_voi_usd=0.0,
    )
    assert action == "KEEP_ORIGINAL"
    assert selected == IDS


def test_lower_confidence_rule_can_reject_a_positive_point_estimate() -> None:
    checkpoint = {
        "risk_decision": "KEEP_ORIGINAL",
        "risk_selected_order": IDS,
        "next_sample_size": 20,
        "value_of_additional_profiling": {
            "incremental_profiling_cost_usd": 0.1,
            "net_value_of_information_usd": 0.02,
            "continuation_value_basis": "lower_confidence_bound",
            "continuation_value_usd": -0.01,
        },
    }
    action, selected, reasons = choose_joint_action(
        checkpoint,
        original_order=IDS,
        profiling_cost_incurred_usd=0.1,
        profiling_budget_usd=None,
        maximum_profiling_decisions=None,
        minimum_voi_usd=0.0,
    )
    assert action == "KEEP_ORIGINAL"
    assert selected == IDS
    assert reasons == [
        "lower_confidence_value_of_additional_profiling_not_positive"
    ]


def test_early_adoption_and_final_checkpoint_fallback() -> None:
    adopt = {
        "risk_decision": "ADOPT_REORDER",
        "risk_selected_order": ["B", "A", "C"],
        "next_sample_size": 20,
    }
    action, selected, _ = choose_joint_action(
        adopt,
        original_order=IDS,
        profiling_cost_incurred_usd=0.1,
        profiling_budget_usd=None,
        maximum_profiling_decisions=None,
        minimum_voi_usd=0.0,
    )
    assert action == "ADOPT_REORDER"
    assert selected == ["B", "A", "C"]
    final = {
        "risk_decision": "KEEP_ORIGINAL",
        "risk_selected_order": IDS,
        "next_sample_size": None,
    }
    action, selected, reason = choose_joint_action(
        final,
        original_order=IDS,
        profiling_cost_incurred_usd=0.1,
        profiling_budget_usd=None,
        maximum_profiling_decisions=None,
        minimum_voi_usd=0.0,
    )
    assert action == "KEEP_ORIGINAL"
    assert selected == IDS
    assert reason == ["maximum_checkpoint_reached"]


def test_first_checkpoint_budget_can_skip_profiling() -> None:
    result = run(matrix_by_position(mixed), profiling_budget_usd=0.00001)
    assert result["action"] == "SKIP_PROFILING"
    assert result["profiling_decisions_consumed"] == 0


def test_joint_settings_reject_too_few_bootstrap_draws() -> None:
    with pytest.raises(ValueError):
        JointAdaptiveSettings(bootstrap_iterations=10)
