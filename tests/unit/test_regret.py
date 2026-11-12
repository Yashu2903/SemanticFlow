import pytest

from semanticflow.optimizer.regret import cvar, plan_risk_metrics
from semanticflow.optimizer.uncertainty import (
    cost_distributions,
    response_variability_sensitivity,
)


def test_cvar_is_the_mean_of_the_upper_tail() -> None:
    losses = [float(value) for value in range(100)]
    report = cvar(losses, 0.95)
    assert report["var"] == pytest.approx(94.05)
    assert report["cvar"] == pytest.approx(97.0)
    assert report["tail_count"] == 5
    with pytest.raises(ValueError):
        cvar([], 0.95)
    with pytest.raises(ValueError):
        cvar(losses, 1.0)


def test_degenerate_and_single_value_distributions() -> None:
    constant = [4.0, 4.0, 4.0, 4.0]
    report = cvar(constant, 0.95)
    assert report["cvar"] == 4.0
    assert cvar([7.0], 0.95)["cvar"] == 7.0


def test_regret_is_relative_to_the_modeled_minimum_on_each_draw() -> None:
    original = ("A", "B")
    cheaper = ("B", "A")
    draws = {
        original: [10.0, 10.0, 12.0, 8.0],
        cheaper: [9.0, 11.0, 9.0, 8.0],
    }
    metrics = plan_risk_metrics(
        draws,
        original_order=original,
        candidate_order=cheaper,
        interval_level=0.5,
        cvar_level=0.75,
    )
    assert metrics["expected_regret_vs_modeled_minimum"]["mean"] == pytest.approx(0.25)
    assert metrics["probability_cheaper_than_original"] == pytest.approx(0.5)
    assert metrics["interpretation"] == "sampling_uncertainty_diagnostic"
    assert "unknown true optimum" in metrics["regret_scope"]


def test_confidence_interval_uses_the_configured_level() -> None:
    draws = {("A",): [1.0, 2.0, 3.0, 4.0, 5.0]}
    report = cost_distributions(
        draws, original_order=["A"], interval_level=0.6, cvar_level=0.8
    )
    summary = report["by_order"]["A"]
    assert summary["mean"] == 3.0
    assert summary["median"] == 3.0
    assert summary["min"] == 1.0
    assert summary["max"] == 5.0
    assert summary["lower"] < summary["upper"]
    assert report["not_a_calibrated_production_probability"] is True


def test_response_variability_stays_a_sensitivity_analysis() -> None:
    empty = response_variability_sensitivity(None)
    assert empty["status"] == "insufficient_repeated_trace_evidence"
    assert empty["not_a_probabilistic_response_model"] is True
    live = {
        "orders": {
            "A -> B": {
                "replicates": 2,
                "variability": {
                    "replicate_costs_usd": [1.0, 1.1],
                    "replicate_standard_deviation_usd": 0.07,
                    "pairs": [{"threshold_crossings": 3, "decision_difference": 3}],
                },
            }
        }
    }
    report = response_variability_sensitivity(live)
    assert report["status"] == "observed_replicate_sensitivity"
    assert report["total_recorded_threshold_crossings"] == 3
    assert "small_sample_warning" in report
