import pytest

from semanticflow.optimizer.joint_benchmark import _rescale, _route_existing_adaptive
from semanticflow.optimizer.models import OptimizerSettings
from semanticflow.optimizer.voi_validation import _error, _metric_transition


def test_uncertainty_prediction_error_arithmetic() -> None:
    result = _metric_transition(current=0.4, predicted_next=0.3, realized_next=0.25)
    assert result["predicted_reduction"] == pytest.approx(0.1)
    assert result["realized_reduction"] == pytest.approx(0.15)
    assert result["next_value_error"]["absolute"] == pytest.approx(0.05)
    assert result["reduction_error"]["absolute"] == pytest.approx(-0.05)


def test_relative_error_is_explicitly_unavailable_at_zero() -> None:
    result = _error(0.1, 0.0)
    assert result["absolute"] == pytest.approx(0.1)
    assert result["relative_to_realized"] is None


def test_missing_uncertainty_values_remain_unavailable() -> None:
    result = _metric_transition(current=0.2, predicted_next=None, realized_next=0.1)
    assert result["predicted_reduction"] is None
    assert result["next_value_error"] == {
        "absolute": None,
        "relative_to_realized": None,
    }


def test_lower_bound_voi_rescales_with_reuse_horizon() -> None:
    checkpoint = {
        "value_of_additional_profiling": {
            "expected_reduction_in_decision_loss_per_execution": 0.2,
            "incremental_profiling_cost_usd": 1.0,
            "continuation_value_basis": "lower_confidence_bound",
            "continuation_value_usd": -0.5,
            "monte_carlo_confidence": {"lower_reduction_per_execution": 0.1},
        }
    }
    result = _rescale(checkpoint, 20, charge_cost=True)
    value = result["value_of_additional_profiling"]
    assert value["net_value_of_information_usd"] == pytest.approx(3.0)
    assert value["monte_carlo_confidence"]["lower_net_value_usd"] == pytest.approx(1.0)
    assert value["continuation_value_usd"] == pytest.approx(1.0)


def test_independent_adaptive_depth_can_change_at_high_reuse() -> None:
    ctx = {"optimizer": OptimizerSettings(), "original": ["A", "B"]}
    packs = [
        {"size": 25, "incurred_cost_usd": 0.1},
        {"size": 50, "incurred_cost_usd": 0.2},
    ]
    old = {
        "trajectory": [
            {
                "selected_order": ["A", "B"],
                "next_checkpoint_reviews": 50,
                "bootstrap_selection_frequency_of_selected": 0.5,
                "next_increment_cost_usd_estimate": 0.1,
                "bootstrap_mean_regret_per_execution": 0.001,
            },
            {
                "selected_order": ["B", "A"],
                "next_checkpoint_reviews": None,
                "bootstrap_selection_frequency_of_selected": 0.5,
                "next_increment_cost_usd_estimate": None,
                "bootstrap_mean_regret_per_execution": 0.0,
            },
        ]
    }
    low = _route_existing_adaptive(
        ctx, packs, old, horizon=10, apply_risk_gate=False
    )
    high = _route_existing_adaptive(
        ctx, packs, old, horizon=1000, apply_risk_gate=False
    )
    assert low["profiling_reviews_consumed"] == 25
    assert high["profiling_reviews_consumed"] == 50
    assert high["selected_order"] == ["B", "A"]
