import pytest

from semanticflow.optimizer.economics import (
    break_even_executions,
    break_even_range,
    decide_adaptive_profiling_start,
    decide_optimization,
    profiling_cost_accounting,
)
from semanticflow.optimizer.models import EconomicsSettings


def settings(**values) -> EconomicsSettings:
    return EconomicsSettings(**values)


def test_break_even_counts_whole_executions_and_never_for_nonpositive_savings() -> None:
    assert break_even_executions(0.0079, 0.00007) == 113
    assert break_even_executions(1.0, 0.5) == 2
    assert break_even_executions(0.008, 0.013 - 0.0129) == 80
    assert break_even_executions(1.0, 0.3) == 4
    assert break_even_executions(0.0, 0.0) == 0
    assert break_even_executions(1.0, 0.0) is None
    assert break_even_executions(1.0, -0.1) is None
    with pytest.raises(ValueError):
        break_even_executions(-1.0, 1.0)


def test_profiling_cost_exceeding_attainable_savings_executes_original() -> None:
    result = decide_optimization(
        settings=settings(expected_executions=10),
        original_cost_per_execution=0.013,
        statistics_state="none",
        new_profiling_cost=0.008,
        maximum_savings_per_execution=0.0001,
    )
    assert result["action"] == "execute_original_without_profiling"
    assert result["reasons"] == ["profiling_cannot_break_even_within_horizon"]
    assert result["best_case_break_even_executions"] == 80


def test_profitable_reuse_and_sunk_cost_does_not_change_forward_decision() -> None:
    common = {
        "settings": settings(expected_executions=50),
        "original_cost_per_execution": 0.013,
        "optimized_cost_per_execution": 0.0129,
        "statistics_state": "available",
        "existing_plan_available": True,
    }
    without_sunk = decide_optimization(**common, sunk_profiling_cost=0.0)
    with_sunk = decide_optimization(**common, sunk_profiling_cost=0.008)
    assert without_sunk["action"] == with_sunk["action"] == "reuse_existing_plan"
    assert with_sunk["forward_net_savings_usd"] == pytest.approx(50 * 0.0001)
    assert with_sunk["retrospective_net_savings_including_sunk_profiling_usd"] == pytest.approx(
        50 * 0.0001 - 0.008
    )
    assert with_sunk["break_even_executions_for_sunk_profiling"] == 80


def test_new_statistics_are_collected_only_when_savings_could_pay_for_them() -> None:
    result = decide_optimization(
        settings=settings(expected_executions=1000),
        original_cost_per_execution=0.013,
        statistics_state="none",
        new_profiling_cost=0.008,
        maximum_savings_per_execution=0.0001,
    )
    assert result["action"] == "collect_additional_statistics"
    assert result["best_case_net_savings_usd"] == pytest.approx(0.1 - 0.008)


def test_zero_and_one_execution() -> None:
    zero_new = decide_optimization(
        settings=settings(expected_executions=0),
        original_cost_per_execution=0.013,
        statistics_state="none",
        new_profiling_cost=0.008,
        maximum_savings_per_execution=1.0,
    )
    assert zero_new["action"] == "execute_original_without_profiling"
    assert zero_new["reasons"] == ["no_expected_executions"]
    zero_existing = decide_optimization(
        settings=settings(expected_executions=0),
        original_cost_per_execution=0.013,
        optimized_cost_per_execution=0.012,
        statistics_state="available",
    )
    assert zero_existing["action"] == "decline_optimization_execute_original"
    assert zero_existing["reasons"] == ["no_expected_executions"]
    one = decide_optimization(
        settings=settings(expected_executions=1),
        original_cost_per_execution=0.013,
        optimized_cost_per_execution=0.012,
        statistics_state="available",
    )
    assert one["action"] == "optimize_with_existing_statistics"
    assert one["forward_net_savings_usd"] == pytest.approx(0.001)


def test_unknown_reuse_defaults_to_one_execution() -> None:
    result = decide_optimization(
        settings=settings(),
        original_cost_per_execution=0.013,
        statistics_state="none",
        new_profiling_cost=0.008,
        maximum_savings_per_execution=0.0028,
    )
    assert result["expected_executions"] == 1
    assert result["expected_executions_known"] is False
    assert result["action"] == "execute_original_without_profiling"


def test_equal_cost_plans_are_not_worth_optimizing() -> None:
    result = decide_optimization(
        settings=settings(expected_executions=100),
        original_cost_per_execution=0.013,
        optimized_cost_per_execution=0.013,
        statistics_state="available",
    )
    assert result["action"] == "decline_optimization_execute_original"
    assert result["reasons"] == ["nonpositive_expected_savings"]
    assert result["break_even_executions_for_sunk_profiling"] == 0


def test_missing_pricing_is_insufficient_information() -> None:
    result = decide_optimization(
        settings=settings(expected_executions=100),
        original_cost_per_execution=None,
        statistics_state="none",
        new_profiling_cost=None,
    )
    assert result["action"] == "execute_original_without_profiling"
    assert result["status"] == "insufficient_information"
    assert result["reasons"] == ["pricing_or_usage_unavailable"]
    unknown_profile = decide_optimization(
        settings=settings(expected_executions=100),
        original_cost_per_execution=0.013,
        statistics_state="none",
        new_profiling_cost=None,
    )
    assert unknown_profile["reasons"] == ["profiling_cost_unknown"]


def test_profiling_budget_exhaustion_blocks_new_profiling() -> None:
    result = decide_optimization(
        settings=settings(expected_executions=10_000, profiling_budget_usd=0.005),
        original_cost_per_execution=0.013,
        statistics_state="none",
        new_profiling_cost=0.008,
        maximum_savings_per_execution=0.001,
    )
    assert result["action"] == "execute_original_without_profiling"
    assert result["reasons"] == ["profiling_budget_exceeded"]


def test_uncertainty_and_probability_gates_are_conservative() -> None:
    gated = decide_optimization(
        settings=settings(
            expected_executions=100,
            maximum_relative_uncertainty=1.0,
            minimum_savings_probability=0.95,
        ),
        original_cost_per_execution=0.0133,
        optimized_cost_per_execution=0.0132,
        statistics_state="available",
        savings_interval=(-0.0002, 0.0005),
        savings_probability=0.77,
    )
    assert gated["action"] == "decline_optimization_execute_original"
    assert gated["reasons"] == [
        "savings_uncertainty_too_high",
        "savings_probability_below_minimum",
    ]
    missing = decide_optimization(
        settings=settings(expected_executions=100, minimum_savings_probability=0.95),
        original_cost_per_execution=0.0133,
        optimized_cost_per_execution=0.0132,
        statistics_state="available",
    )
    assert missing["reasons"] == ["savings_probability_unavailable"]


def test_overhead_is_charged_unless_an_existing_plan_is_reused() -> None:
    common = {
        "settings": settings(expected_executions=2, optimization_overhead_usd=0.01),
        "original_cost_per_execution": 0.013,
        "optimized_cost_per_execution": 0.012,
        "statistics_state": "available",
    }
    fresh = decide_optimization(**common, existing_plan_available=False)
    reused = decide_optimization(**common, existing_plan_available=True)
    assert fresh["action"] == "decline_optimization_execute_original"
    assert fresh["reasons"] == ["net_savings_not_above_minimum"]
    assert reused["action"] == "reuse_existing_plan"


def test_break_even_range_maps_the_savings_interval_and_reports_never() -> None:
    # Upper savings give the fewest executions; a non-positive lower bound never pays back.
    assert break_even_range(0.0079, -0.0002, 0.0004) == {
        "fewest_executions": 20,
        "most_executions": None,
    }
    assert break_even_range(0.0079, 0.00005, 0.00007) == {
        "fewest_executions": 113,
        "most_executions": 158,
    }
    with pytest.raises(ValueError, match="lower bound exceeds"):
        break_even_range(1.0, 0.2, 0.1)


def test_break_even_charges_the_same_overhead_as_net_savings() -> None:
    common = {
        "settings": settings(expected_executions=50, optimization_overhead_usd=0.002),
        "original_cost_per_execution": 0.013,
        "optimized_cost_per_execution": 0.0129,
        "statistics_state": "available",
        "sunk_profiling_cost": 0.008,
        "new_profiling_cost": 0.008,
    }
    fresh = decide_optimization(**common, existing_plan_available=False)
    reused = decide_optimization(**common, existing_plan_available=True)
    assert fresh["break_even_executions_for_sunk_profiling"] == 100
    assert reused["break_even_executions_for_sunk_profiling"] == 80
    assert fresh["break_even_executions_for_new_comparable_profile"] == 100
    assert reused["break_even_executions_for_new_comparable_profile"] == 100
    assert fresh["retrospective_net_savings_including_sunk_profiling_usd"] == pytest.approx(
        50 * 0.0001 - 0.002 - 0.008
    )


def test_savings_basis_is_recorded_and_validated() -> None:
    result = decide_optimization(
        settings=settings(expected_executions=10),
        original_cost_per_execution=0.013,
        statistics_state="none",
        new_profiling_cost=0.008,
        maximum_savings_per_execution=0.0001,
        savings_basis="transferred_prediction",
    )
    assert result["savings_basis"] == "transferred_prediction"
    with pytest.raises(ValueError, match="unknown savings basis"):
        decide_optimization(
            settings=settings(),
            original_cost_per_execution=0.013,
            statistics_state="none",
            savings_basis="measured",
        )


def gate(**values):
    arguments = {
        "expected_executions": 1,
        "first_checkpoint_decisions": 75,
        "first_checkpoint_cost": 75 * 1.318289e-5,
        "settings": settings(),
        "savings_upper_bound_per_execution": 1.2662e-4,
        "savings_bound_basis": "transferred_prediction",
    }
    arguments.update(values)
    return decide_adaptive_profiling_start(**arguments)


def test_adaptive_profiling_is_skipped_when_even_the_bounded_saving_cannot_pay() -> None:
    result = gate()
    assert result["action"] == "skip_profiling_execute_original"
    assert result["reasons"] == ["profiling_cannot_break_even_within_horizon"]
    assert result["best_case_break_even_executions"] == 8
    assert result["best_case_net_savings_usd"] < 0
    started = gate(expected_executions=10)
    assert started["action"] == "start_adaptive_profiling"
    assert started["reasons"] == ["bounded_savings_may_exceed_first_checkpoint_cost"]


def test_adaptive_profiling_is_skipped_without_executions_or_with_reusable_statistics() -> None:
    assert gate(expected_executions=0)["reasons"] == ["no_expected_executions"]
    reuse = gate(expected_executions=1000, statistics_available=True)
    assert reuse["action"] == "skip_profiling_use_existing_statistics"
    assert reuse["reasons"] == ["reusable_statistics_available"]
    stale = gate(
        expected_executions=1000,
        statistics_available=True,
        settings=settings(statistics_reusable=False),
    )
    assert stale["action"] == "start_adaptive_profiling"


def test_adaptive_first_checkpoint_respects_caps_and_budgets() -> None:
    capped = gate(
        expected_executions=1000,
        maximum_profiling_decisions=60,
        adaptive_budget_usd=0.0005,
    )
    assert capped["action"] == "skip_profiling_execute_original"
    assert capped["reasons"] == [
        "first_checkpoint_exceeds_decision_cap",
        "first_checkpoint_exceeds_profiling_budget",
    ]
    economics_budget = gate(
        expected_executions=1000, settings=settings(profiling_budget_usd=0.0005)
    )
    assert economics_budget["reasons"] == ["first_checkpoint_exceeds_profiling_budget"]
    # Without economics settings only the caps apply, so zero reuse still starts.
    unconfigured = gate(expected_executions=0, settings=None)
    assert unconfigured["action"] == "start_adaptive_profiling"
    assert unconfigured["reasons"] == ["economics_gate_not_configured"]
    assert gate(settings=None, maximum_profiling_decisions=60)["reasons"] == [
        "first_checkpoint_exceeds_decision_cap"
    ]


def test_adaptive_gate_is_conservative_when_cost_or_bound_is_unknown() -> None:
    unknown_cost = gate(expected_executions=1000, first_checkpoint_cost=None)
    assert unknown_cost["status"] == "insufficient_information"
    assert unknown_cost["reasons"] == ["profiling_cost_unknown"]
    unknown_bound = gate(expected_executions=1000, savings_upper_bound_per_execution=None)
    assert unknown_bound["action"] == "skip_profiling_execute_original"
    assert unknown_bound["reasons"] == ["savings_upper_bound_unknown"]


def test_cached_decisions_reduce_new_profiling_spend() -> None:
    accounting = profiling_cost_accounting(
        required_decisions=600,
        reusable_decisions=249,
        average_cost_per_decision=0.00001,
        incurred_cost=0.0079,
    )
    assert accounting["new_decisions"] == 351
    assert accounting["new_profiling_cost_usd"] == pytest.approx(0.00351)
    assert accounting["sunk_profiling_cost_usd"] == 0.0079
    capped = profiling_cost_accounting(
        required_decisions=10, reusable_decisions=50, average_cost_per_decision=None
    )
    assert capped["new_decisions"] == 0
    assert capped["new_profiling_cost_usd"] is None
