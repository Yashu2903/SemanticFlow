import json

import pytest

from semanticflow.cli import main
from semanticflow.optimizer.models import load_optimizer_settings
from semanticflow.optimizer.risk_benchmark import (
    load_observations,
    observe_selection,
    summarize_strategy,
)


def test_unavailable_live_trace_is_not_replaced_with_an_estimate() -> None:
    observations = {
        "fixed_order": "A -> B",
        "observed_cheapest_order": "B -> A",
        "mean_cost_usd": {"A -> B": 1.0, "B -> A": 0.8},
    }
    measured = observe_selection(
        observations, ["B", "A"], predicted_saving=0.25, profiling_cost_usd=0.1
    )
    assert measured["observed_result"] == "measured"
    assert measured["observed_saving_usd"] == pytest.approx(0.2)
    assert measured["predicted_minus_observed_saving_usd"] == pytest.approx(0.05)
    assert measured["observed_regret_usd"] == 0.0
    missing = observe_selection(
        observations, ["A", "C"], predicted_saving=0.4, profiling_cost_usd=0.1
    )
    assert missing["observed_result"] == "unavailable"
    assert missing["observed_cost_usd"] is None
    assert missing["predicted_saving_usd"] == 0.4


def test_held_out_costs_are_not_used_to_choose_a_plan() -> None:
    analysis = {
        "fixed_order": "A -> B",
        "observed_cheapest_order": "B -> A",
        "all_orders_executed": False,
        "orders": {
            "A -> B": {"mean_cost_usd": 2.0},
            "B -> A": {"mean_cost_usd": 1.0},
        },
    }
    observations = load_observations(analysis)
    assert observations["executed_orders"] == ["A -> B", "B -> A"]
    summary = summarize_strategy(
        [
            {
                "workload_id": "one",
                "original_order": ["A", "B"],
                "selections": {
                    "risk_aware": observe_selection(
                        observations,
                        ["A", "B"],
                        predicted_saving=0.0,
                        profiling_cost_usd=0.0,
                    )
                    | {"decision": "KEEP_ORIGINAL"}
                },
            },
            {
                "workload_id": "two",
                "original_order": ["A", "B"],
                "selections": {
                    "risk_aware": observe_selection(
                        observations,
                        ["B", "A"],
                        predicted_saving=1.0,
                        profiling_cost_usd=0.05,
                    )
                    | {"decision": "ADOPT_REORDER"}
                },
            },
        ],
        "risk_aware",
    )
    assert summary["workloads_improved"] == 1
    assert summary["workloads_retaining_original"] == 1
    assert summary["abstention_or_fallback_count"] == 1
    assert summary["mean_saving_percent"] == 25.0


def test_milestone_4_config_loads_risk_aware_defaults() -> None:
    settings = load_optimizer_settings("configs/optimizer.milestone_4.yaml")
    assert settings.planner.selection_policy == "risk_aware"
    assert settings.bootstrap.iterations == 2000
    assert settings.risk.minimum_relative_saving == 0.01
    assert settings.risk.minimum_saving_probability == 0.95
    assert settings.risk.apply_profiling_economics is False
    assert settings.joint.bootstrap_iterations == 2000
    assert settings.joint.sparse_posterior_predictive is True
    assert settings.joint.voi_scenarios == 32
    assert settings.joint.voi_decision_rule == "point_estimate"
    assert settings.joint.voi_confidence_level == 0.95
    lower_bound = load_optimizer_settings("configs/optimizer.milestone_4_voi_lcb.yaml")
    assert lower_bound.joint.voi_decision_rule == "lower_confidence_bound"
    assert settings.model_copy(
        update={
            "joint": settings.joint.model_copy(
                update={"voi_decision_rule": "lower_confidence_bound"}
            )
        }
    ) == lower_bound


def test_risk_plan_cli_skips_profiling_without_a_provider(tmp_path) -> None:
    output = tmp_path / "decision.json"
    code = main(
        [
            "risk-plan",
            "--query",
            "configs/queries/product_quality.yaml",
            "--statistics-state",
            "none",
            "--first-checkpoint-cost",
            "0.01",
            "--savings-upper-bound",
            "0.0001",
            "--output",
            str(output),
            "--experiment-dir",
            str(tmp_path / "unused"),
        ]
    )
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["explanation"]["decision"] == "SKIP_PROFILING"
    assert payload["selected_order"] == [
        "poor_quality",
        "product_defect",
        "dissatisfaction",
    ]
