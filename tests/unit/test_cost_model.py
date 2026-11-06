import itertools

import pytest

from semanticflow.optimizer.cost_model import estimate_plan
from semanticflow.optimizer.models import PredicateStatistics


def statistic(
    predicate_id: str,
    *,
    selectivity: float,
    cost: float,
    input_tokens: float | None = None,
) -> PredicateStatistics:
    input_tokens = cost if input_tokens is None else input_tokens
    return PredicateStatistics(
        predicate_id=predicate_id,
        question_hash=f"hash-{predicate_id}",
        sample_size=10,
        completed_decisions=10,
        passing_decisions=round(selectivity * 10),
        rejected_decisions=10 - round(selectivity * 10),
        estimated_selectivity=selectivity,
        estimated_rejection_rate=1 - selectivity,
        wilson_lower=0,
        wilson_upper=1,
        average_input_tokens=input_tokens,
        average_output_tokens=0,
        average_total_tokens=input_tokens,
        average_inference_latency_seconds=cost,
        average_estimated_cost_usd=cost,
        provider_attempts=10,
        http_attempts=0,
        successful_api_calls=0,
        failed_http_attempts=0,
        retry_attempts=0,
        provider_errors=0,
        budget_blocked_evaluations=0,
        model_versions=["mock"],
    )


def test_expected_cost_formula_examples_and_global_ranking() -> None:
    statistics = {
        "A": statistic("A", selectivity=0.2, cost=2),
        "B": statistic("B", selectivity=0.9, cost=1),
        "C": statistic("C", selectivity=0.5, cost=3),
    }
    abc = estimate_plan(
        ["A", "B", "C"], statistics, input_record_count=100, objective="estimated_cost_usd"
    )
    acb = estimate_plan(
        ["A", "C", "B"], statistics, input_record_count=100, objective="estimated_cost_usd"
    )
    assert abc.objective_value == pytest.approx(274)
    assert acb.objective_value == pytest.approx(270)
    all_candidates = [
        estimate_plan(order, statistics, input_record_count=100, objective="estimated_cost_usd")
        for order in itertools.permutations(statistics)
    ]
    assert min(all_candidates, key=lambda item: item.objective_value).objective_value <= 270


@pytest.mark.parametrize("predicate_count,expected", [(2, 2), (3, 6), (6, 720)])
def test_candidate_permutation_counts(predicate_count: int, expected: int) -> None:
    assert len(list(itertools.permutations(range(predicate_count)))) == expected


def test_zero_and_full_selectivity_and_zero_cost() -> None:
    statistics = {
        "zero": statistic("zero", selectivity=0, cost=0),
        "full": statistic("full", selectivity=1, cost=2),
    }
    plan = estimate_plan(
        ["zero", "full"], statistics, input_record_count=100, objective="estimated_cost_usd"
    )
    assert plan.estimated_evaluations == 100
    assert plan.objective_value == 0


def test_missing_statistics_and_missing_token_cost_are_rejected() -> None:
    with pytest.raises(ValueError, match="missing statistics"):
        estimate_plan(["A", "B"], {}, input_record_count=10, objective="evaluations")


def test_six_predicate_cost_model_is_finite() -> None:
    statistics = {
        str(index): statistic(str(index), selectivity=0.5, cost=index)
        for index in range(6)
    }
    candidate = estimate_plan(
        list(statistics),
        statistics,
        input_record_count=800,
        objective="estimated_cost_usd",
    )
    assert candidate.estimated_evaluations == pytest.approx(1575)
    assert candidate.objective_value >= 0
    no_tokens = statistic("A", selectivity=0.5, cost=1).model_copy(
        update={"average_input_tokens": None, "average_output_tokens": None}
    )
    with pytest.raises(ValueError, match="token"):
        estimate_plan(
            ["A", "B"],
            {"A": no_tokens, "B": statistic("B", selectivity=0.5, cost=1)},
            input_record_count=10,
            objective="total_tokens",
        )
