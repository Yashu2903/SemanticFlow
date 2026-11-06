import pytest

from semanticflow.optimizer.adaptive import sampling_order, simulate_adaptive_profiling
from semanticflow.optimizer.matrix import MatrixRow, ProfileMatrix
from semanticflow.optimizer.models import (
    AdaptiveProfilingSettings,
    BootstrapSettings,
    CorrelationSettings,
    EconomicsSettings,
)
from tests.m3_support import hashes

IDS = ["A", "B", "C"]
SEED = 4049
PRICE = 1.0
TOKENS = 10


def matrix_along_sampling_order(patterns_for_position) -> ProfileMatrix:
    review_ids = [f"p{index:03}" for index in range(40)]
    ordered = sampling_order(review_ids, SEED)
    position = {review_id: index for index, review_id in enumerate(ordered)}
    rows = [
        MatrixRow(
            review_id,
            patterns_for_position(position[review_id]),
            (TOKENS,) * 3,
            (0,) * 3,
            (0.1,) * 3,
        )
        for review_id in review_ids
    ]
    return ProfileMatrix(IDS, rows)


def alternating(position: int) -> tuple[bool, bool, bool]:
    # Every even prefix ties A and B exactly; resamples break the tie either way.
    return (True, False, True) if position % 2 else (False, True, True)


def simulate(matrix, *, horizon, iterations=60, records=800, gate=None, **adaptive):
    settings = AdaptiveProfilingSettings(
        checkpoints=adaptive.pop("checkpoints", [10, 20, 40]),
        seed=SEED,
        bootstrap_iterations=iterations,
        **adaptive,
    )
    return simulate_adaptive_profiling(
        matrix,
        estimator_name="correlation_aware",
        original_order=IDS,
        objective="estimated_cost_usd",
        evaluation_record_count=records,
        question_hashes=hashes(IDS),
        settings=settings,
        bootstrap=BootstrapSettings(),
        correlation=CorrelationSettings(prior_strength=0.0, conditional_usage=False),
        input_price_per_million=PRICE,
        output_price_per_million=0.0,
        expected_executions=horizon,
        **(gate or {}),
    )


def per_review_cost() -> float:
    return len(IDS) * TOKENS * PRICE / 1_000_000


def test_zero_benefit_profiling_stops_at_the_first_checkpoint() -> None:
    matrix = matrix_along_sampling_order(lambda position: (position % 3 == 0,) * 3)
    result = simulate(matrix, horizon=1_000_000)
    stop = result["stopping_checkpoint"]
    assert stop["profiled_reviews"] == 10
    assert "expected_value_of_information_below_increment_cost" in stop["reasons"]
    assert stop["expected_value_of_information_bound"] == 0.0
    assert result["profiling_decisions_saved"] == (40 - 10) * 3
    assert result["profiling_cost_saved_usd_offline_estimate"] == pytest.approx(
        30 * per_review_cost()
    )
    assert result["live_provider_calls"] == 0


def test_unstable_plan_with_valuable_information_keeps_profiling() -> None:
    result = simulate(matrix_along_sampling_order(alternating), horizon=10**9)
    decisions = [entry["controller_decision"] for entry in result["trajectory"]]
    assert decisions == ["continue", "continue", "stop"]
    assert result["stopping_checkpoint"]["reasons"] == ["maximum_checkpoint_reached"]
    assert all(
        entry["bootstrap_selection_frequency_of_selected"] < 0.95
        for entry in result["trajectory"]
    )
    assert result["profiling_decisions_saved"] == 0


def test_low_reuse_stops_when_information_is_worth_less_than_more_profiling() -> None:
    result = simulate(matrix_along_sampling_order(alternating), horizon=1, records=10)
    assert result["stopping_checkpoint"]["profiled_reviews"] == 10
    assert result["stopping_checkpoint"]["reasons"] == [
        "expected_value_of_information_below_increment_cost"
    ]
    assert result["trajectory"][1]["controller_decision"] == "after_stop_diagnostic"


def test_profiling_budget_exhaustion_stops_before_the_next_increment() -> None:
    result = simulate(
        matrix_along_sampling_order(alternating),
        horizon=10**9,
        profiling_budget_usd=15 * per_review_cost(),
    )
    stop = result["stopping_checkpoint"]
    assert stop["profiled_reviews"] == 10
    assert stop["reasons"] == ["profiling_budget_exhausted_before_next_checkpoint"]


def test_decision_budget_limits_the_profile_size() -> None:
    result = simulate(
        matrix_along_sampling_order(alternating), horizon=10**9, maximum_profiling_decisions=60
    )
    stop = result["stopping_checkpoint"]
    assert stop["profiled_reviews"] == 20
    assert stop["reasons"] == ["decision_budget_exhausted_before_next_checkpoint"]


def test_checkpoints_are_nested_samples_with_exact_stage_accounting() -> None:
    result = simulate(matrix_along_sampling_order(alternating), horizon=10**9)
    for entry in result["trajectory"]:
        reviews = entry["profiled_reviews"]
        assert entry["profiling_decisions"] == reviews * 3
        assert entry["profiling_cost_usd_offline_estimate"] == pytest.approx(
            reviews * per_review_cost()
        )
    first, second, _ = result["trajectory"]
    assert first["next_increment_cost_usd_estimate"] == pytest.approx(10 * per_review_cost())
    assert second["next_increment_cost_usd_estimate"] == pytest.approx(20 * per_review_cost())
    assert result["full_profile"]["profiling_decisions"] == 120
    assert result["sampling"]["design"] == "uniform_without_replacement_complete_rows"


def test_economics_can_skip_adaptive_profiling_entirely() -> None:
    matrix = matrix_along_sampling_order(alternating)
    result = simulate(
        matrix,
        horizon=1,
        gate={
            "economics": EconomicsSettings(),
            # The first checkpoint costs 10 reviews; one execution can save at most 5.
            "savings_upper_bound_per_execution": 5 * per_review_cost(),
            "savings_bound_basis": "transferred_prediction",
            "estimated_cost_per_decision": per_review_cost() / 3,
        },
    )
    assert result["profiling_started"] is False
    assert result["start_decision"]["action"] == "skip_profiling_execute_original"
    assert result["start_decision"]["reasons"] == ["profiling_cannot_break_even_within_horizon"]
    assert result["start_decision"]["first_checkpoint_cost_source"] == (
        "estimated_cost_per_decision_times_decisions"
    )
    assert result["trajectory"] == []
    assert result["stopping_checkpoint"] is None
    assert result["profiling_decisions_used"] == 0
    assert result["profiling_decisions_saved"] == 120
    assert result["profiling_cost_saved_usd_offline_estimate"] == pytest.approx(
        40 * per_review_cost()
    )
    assert result["selected_order"] == IDS
    assert result["selected_order_source"] == "original_order_without_profiling"
    assert result["live_provider_calls"] == 0


def test_first_checkpoint_over_budget_is_never_profiled() -> None:
    result = simulate(
        matrix_along_sampling_order(alternating),
        horizon=10**9,
        profiling_budget_usd=5 * per_review_cost(),
    )
    assert result["profiling_started"] is False
    assert result["start_decision"]["reasons"] == ["first_checkpoint_exceeds_profiling_budget"]
    assert result["start_decision"]["first_checkpoint_cost_source"] == (
        "recorded_usage_of_first_checkpoint_rows_offline"
    )


def test_a_passed_gate_leaves_the_controller_unchanged() -> None:
    matrix = matrix_along_sampling_order(alternating)
    ungated = simulate(matrix, horizon=10**9)
    gated = simulate(
        matrix,
        horizon=10**9,
        gate={
            "economics": EconomicsSettings(),
            "savings_upper_bound_per_execution": per_review_cost(),
            "savings_bound_basis": "structural_upper_bound",
        },
    )
    assert gated["profiling_started"] is True
    assert gated["start_decision"]["action"] == "start_adaptive_profiling"
    assert gated["trajectory"] == ungated["trajectory"]
    assert ungated["start_decision"]["reasons"] == ["economics_gate_not_configured"]
    assert gated["profiling_decisions_used"] == gated["stopping_checkpoint"]["profiling_decisions"]


def test_checkpoints_are_validated_and_must_fit_the_matrix() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        AdaptiveProfilingSettings(checkpoints=[50, 25])
    with pytest.raises(ValueError, match="no adaptive checkpoint"):
        simulate(matrix_along_sampling_order(alternating), horizon=1, checkpoints=[100, 200])
    result = simulate(
        matrix_along_sampling_order(alternating), horizon=10**9, checkpoints=[10, 40, 400]
    )
    assert result["checkpoints"] == [10, 40]
