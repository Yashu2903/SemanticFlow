import itertools

import pytest

from semanticflow.optimizer.bootstrap import collect_paired_bootstrap, order_key
from semanticflow.optimizer.disagreement import compare_estimators
from semanticflow.optimizer.models import (
    BootstrapSettings,
    CorrelationSettings,
    EconomicsSettings,
    OptimizerSettings,
    RiskSettings,
)
from semanticflow.optimizer.planner import CostBasedPlanner
from semanticflow.optimizer.risk import net_benefit_draws, select_risk_aware_plan
from tests.m3_support import hashes, matrix_from_patterns, profile_for, query, split_for

EXACT = CorrelationSettings(prior_strength=0.0, conditional_usage=False)
BOOT = BootstrapSettings(iterations=80, seed=11)
IDS = ["A", "B", "C"]


def collected(draws, original, selected):
    orders = tuple(sorted(draws))
    count = len(next(iter(draws.values())))
    return {
        "estimator": "correlation_aware",
        "iterations": count,
        "seed": 0,
        "orders": orders,
        "full_sample_selected_order": tuple(selected),
        "draws": draws,
        "original_order": tuple(original),
    }


def select(draws, original, selected, *, robust=False, independent=None, **risk):
    pack = collected(draws, original, selected)
    other = independent or pack
    return select_risk_aware_plan(
        None,
        original_order=original,
        objective="evaluations",
        evaluation_record_count=100,
        question_hashes={},
        bootstrap=BOOT,
        correlation=EXACT,
        risk=RiskSettings(**risk),
        economics=EconomicsSettings(expected_executions=1),
        input_price_per_million=None,
        output_price_per_million=None,
        tie_relative_tolerance=1e-9,
        primary_estimator="correlation_aware",
        robust=robust,
        primary_collected=pack,
        independent_collected=other,
        correlation_collected=pack,
    )


def winner_draws(n=40):
    original = ("A", "B")
    better = ("B", "A")
    return (
        original,
        better,
        {original: [100.0] * n, better: [80.0] * n},
    )


def test_clear_winner_is_adopted() -> None:
    original, better, draws = winner_draws()
    order, explanation = select(draws, original, better)
    assert order == list(better)
    assert explanation["decision"] == "ADOPT_REORDER"
    assert explanation["bootstrap_probability_is_not_calibrated"] is True
    assert "Adopt B -> A" in explanation["reason"]


def test_near_tie_keeps_the_original_order() -> None:
    original, better, draws = winner_draws()
    draws = {original: [100.0] * 40, better: [99.4] * 40}
    order, explanation = select(draws, original, better)
    assert order == list(original)
    assert explanation["decision"] == "KEEP_ORIGINAL"
    assert "Expected saving is smaller than the configured risk tolerance." in explanation["reason"]


def test_high_saving_with_high_downside_risk_can_be_rejected() -> None:
    original = ("A", "B")
    better = ("B", "A")
    draws = {
        original: [100.0] * 20,
        better: [70.0] * 19 + [160.0],
    }
    order, explanation = select(draws, original, better, maximum_cvar_loss=5.0)
    assert order == list(original)
    assert "cvar_loss_above_maximum" in explanation["reasons"]


def test_low_saving_with_high_confidence_still_respects_materiality() -> None:
    original = ("A", "B")
    better = ("B", "A")
    draws = {original: [100.0] * 40, better: [99.2] * 40}
    order, explanation = select(draws, original, better, minimum_saving_probability=0.5)
    assert order == list(original)
    assert "expected_saving_smaller_than_configured_risk_tolerance" in explanation["reasons"]


def test_estimators_that_disagree_keep_the_original() -> None:
    original = ("A", "B")
    left = ("B", "A")
    draws_ca = {original: [100.0] * 20, left: [70.0] * 20}
    draws_ind = {original: [100.0] * 20, left: [110.0] * 20}
    order, explanation = select(
        draws_ca,
        original,
        left,
        robust=True,
        independent=collected(draws_ind, original, original),
    )
    assert order == list(original)
    assert explanation["decision"] == "KEEP_ORIGINAL"
    assert explanation["disagreement"]["select_same_order"] is False
    assert "disagree" in explanation["reason"].lower()


def test_disagreement_still_adopts_when_both_estimators_say_original_is_worse() -> None:
    original = ("A", "B", "C")
    ca = ("C", "A", "B")
    independent = ("B", "A", "C")
    orders = list(itertools.permutations(original))
    draws_ca = {order: [90.0 if order == ca else 100.0] * 20 for order in orders}
    draws_ind = {order: [90.0 if order == independent else 100.0] * 20 for order in orders}
    order, explanation = select(
        draws_ca,
        original,
        ca,
        robust=True,
        independent=collected(draws_ind, original, independent),
    )
    assert order == list(ca)
    assert explanation["decision"] == "ADOPT_REORDER"
    assert explanation["disagreement"]["agree_original_is_worse"] is True


def test_all_reorders_more_expensive_keep_original() -> None:
    original = ("A", "B")
    other = ("B", "A")
    draws = {original: [80.0] * 20, other: [100.0] * 20}
    order, explanation = select(draws, original, original)
    assert order == list(original)
    assert explanation["decision"] == "KEEP_ORIGINAL"


def test_missing_estimates_raise() -> None:
    original = ("A", "B")
    with pytest.raises(ValueError, match="decision matrix"):
        select_risk_aware_plan(
            None,
            original_order=original,
            objective="evaluations",
            evaluation_record_count=10,
            question_hashes={},
            bootstrap=BOOT,
            correlation=EXACT,
            risk=RiskSettings(),
            economics=EconomicsSettings(),
            input_price_per_million=None,
            output_price_per_million=None,
            tie_relative_tolerance=1e-9,
            primary_estimator="correlation_aware",
        )


def test_configurable_probability_threshold() -> None:
    original, better, draws = winner_draws()
    mixed = {original: [100.0] * 10, better: [80.0] * 9 + [120.0]}
    keep, _ = select(mixed, original, better, minimum_saving_probability=0.95)
    adopt, _ = select(mixed, original, better, minimum_saving_probability=0.5)
    assert keep == list(original)
    assert adopt == list(better)


def test_skip_profiling_when_the_first_checkpoint_cannot_break_even() -> None:
    order, explanation = select_risk_aware_plan(
        None,
        original_order=["A", "B"],
        objective="estimated_cost_usd",
        evaluation_record_count=800,
        question_hashes={},
        bootstrap=BOOT,
        correlation=EXACT,
        risk=RiskSettings(),
        economics=EconomicsSettings(expected_executions=1),
        input_price_per_million=None,
        output_price_per_million=None,
        tie_relative_tolerance=1e-9,
        primary_estimator="correlation_aware",
        statistics_state="none",
        first_checkpoint_decisions=75,
        first_checkpoint_cost=0.01,
        savings_upper_bound_per_execution=0.001,
    )
    assert order == ["A", "B"]
    assert explanation["decision"] == "SKIP_PROFILING"
    assert "economics_skip_profiling" in explanation["reasons"]


def test_unknown_reuse_is_one_execution() -> None:
    savings = [0.01, 0.02]
    net = net_benefit_draws(
        savings, executions=1, new_profiling_cost=0.05, optimization_overhead=0.0
    )
    assert net == pytest.approx([-0.04, -0.03])
    reuse = net_benefit_draws(
        savings, executions=10, new_profiling_cost=0.05, optimization_overhead=0.0
    )
    assert reuse[0] == pytest.approx(0.05)


def test_sunk_profiling_is_reported_and_not_charged_by_default() -> None:
    original, better, draws = winner_draws()
    _, explanation = select(draws, original, better)
    assert explanation["net_benefit"]["new_profiling_cost_charged_usd"] == 0.0
    assert explanation["net_benefit"]["sunk_profiling_cost_usd"] == 0.0


def test_economics_gate_can_decline_a_reorder() -> None:
    original, better, draws = winner_draws()
    order, explanation = select(
        draws,
        original,
        better,
        apply_profiling_economics=True,
    )
    # Existing statistics, horizon 1, positive savings: economics should not reverse the adopt.
    assert order == list(better)
    declined, explanation = select_risk_aware_plan(
        None,
        original_order=original,
        objective="evaluations",
        evaluation_record_count=100,
        question_hashes={},
        bootstrap=BOOT,
        correlation=EXACT,
        risk=RiskSettings(apply_profiling_economics=True, minimum_relative_saving=0.0),
        economics=EconomicsSettings(expected_executions=1, minimum_net_savings_usd=50.0),
        input_price_per_million=None,
        output_price_per_million=None,
        tie_relative_tolerance=1e-9,
        primary_estimator="correlation_aware",
        primary_collected=collected(draws, original, better),
    )
    assert declined == list(original)
    assert "economics_decline_optimization" in explanation["reasons"]


def test_estimator_interval_overlap_and_saving_sign() -> None:
    original = ("A", "B")
    other = ("B", "A")
    independent = {original: [10.0] * 10, other: [9.0] * 10}
    correlated = {original: [10.0] * 10, other: [12.0] * 10}
    report = compare_estimators(
        independent=independent,
        correlation_aware=correlated,
        original_order=original,
        independent_selected=other,
        correlation_selected=original,
        interval_level=0.95,
    )
    assert report["select_same_order"] is False
    assert report["by_order"][order_key(other)]["one_predicts_loss_other_gain"] is True
    assert report["by_order"][order_key(original)]["cost_intervals_overlap"] is True


def test_planner_risk_aware_policy_documents_keep_original() -> None:
    patterns = [(True, True, False) if index < 18 else (False, False, True) for index in range(20)]
    matrix = matrix_from_patterns(IDS, patterns, input_tokens={"A": 10, "B": 10, "C": 10})
    optimizer = OptimizerSettings.model_validate(
        {
            "planner": {
                "objective": "total_tokens",
                "cardinality_estimator": "correlation_aware",
                "cost_model_version": "correlation-aware-and-v1",
                "selection_policy": "risk_aware",
            },
            "correlation": {"prior_strength": 0.0, "conditional_usage": False},
            "bootstrap": {"iterations": 40, "seed": 17},
            "risk": {"minimum_relative_saving": 0.5, "minimum_saving_probability": 0.99},
        }
    )
    plan = CostBasedPlanner(optimizer).plan(
        query(IDS), profile_for(matrix), split_for(matrix), matrix=matrix
    )
    assert plan.selection_policy == "risk_aware"
    assert plan.selection_explanation["decision"] in {"KEEP_ORIGINAL", "ADOPT_REORDER"}
    assert "reason" in plan.selection_explanation


def test_paired_bootstrap_collection_is_deterministic_and_keeps_rows() -> None:
    patterns = [(True, False), (False, True), (True, True), (False, False)] * 5
    matrix = matrix_from_patterns(["A", "B"], patterns)
    kwargs = dict(
        estimator_name="independent",
        original_order=["A", "B"],
        objective="evaluations",
        evaluation_record_count=10,
        question_hashes=hashes(["A", "B"]),
        settings=BootstrapSettings(iterations=30, seed=99),
        correlation=EXACT,
        input_price_per_million=None,
        output_price_per_million=None,
    )
    first = collect_paired_bootstrap(matrix, **kwargs)
    second = collect_paired_bootstrap(matrix, **kwargs)
    assert first["draws"] == second["draws"]
    assert first["selected_orders"] == second["selected_orders"]
    assert first["resampling_unit"] == "review_row_with_all_predicate_outcomes_and_usage"
