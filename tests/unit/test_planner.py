import pytest

from semanticflow.optimizer.models import (
    OptimizerSettings,
    PredicateStatistics,
    ProfilingResult,
    SplitManifest,
)
from semanticflow.optimizer.planner import CostBasedPlanner, predicate_hash
from semanticflow.query.models import QuerySpec


def query() -> QuerySpec:
    return QuerySpec.model_validate(
        {
            "query_id": "q",
            "dataset_path": "unused.parquet",
            "limit": 10,
            "threshold": 0.5,
            "max_calls": 100,
            "predicates": [
                {"id": "A", "instructions": "A?"},
                {"id": "B", "instructions": "B?"},
                {"id": "C", "instructions": "C?"},
            ],
        }
    )


def stats(predicate_id: str, selectivity: float, tokens: float = 10) -> PredicateStatistics:
    passing = round(selectivity * 10)
    return PredicateStatistics(
        predicate_id=predicate_id,
        question_hash=predicate_hash(predicate_id, f"{predicate_id}?"),
        sample_size=10,
        completed_decisions=10,
        passing_decisions=passing,
        rejected_decisions=10 - passing,
        estimated_selectivity=selectivity,
        estimated_rejection_rate=1 - selectivity,
        wilson_lower=0,
        wilson_upper=1,
        average_input_tokens=tokens,
        average_output_tokens=0,
        average_total_tokens=tokens,
        average_inference_latency_seconds=0,
        average_estimated_cost_usd=None,
        provider_attempts=10,
        http_attempts=0,
        successful_api_calls=0,
        failed_http_attempts=0,
        retry_attempts=0,
        provider_errors=0,
        budget_blocked_evaluations=0,
        model_versions=["mock"],
    )


def profile(items: list[PredicateStatistics]) -> ProfilingResult:
    q = query()
    return ProfilingResult(
        complete=True,
        query_id="q",
        workload_id="q",
        dataset_fingerprint="dataset",
        profiling_fingerprint="profile",
        threshold=0.5,
        requested_model="mock",
        resolved_model_versions=["mock"],
        predicate_order=["A", "B", "C"],
        predicate_instructions={item.id: item.instructions for item in q.predicates},
        predicate_hashes={
            item.id: predicate_hash(item.id, item.instructions) for item in q.predicates
        },
        sample_size=10,
        requested_evaluations=30,
        completed_decisions=30,
        budget_blocked_evaluations=0,
        provider_attempts=30,
        http_attempts=0,
        successful_api_calls=0,
        failed_http_attempts=0,
        retry_attempts=0,
        provider_errors=0,
        input_tokens=300,
        output_tokens=0,
        total_tokens=300,
        summed_inference_latency_seconds=0,
        wall_time_seconds=0,
        estimated_cost_usd=None,
        pricing={},
        statistics_version="unconditional-profile-v1",
        statistics=items,
    )


def split() -> SplitManifest:
    return SplitManifest(
        dataset_path="unused.parquet",
        dataset_fingerprint="dataset",
        dataset_version="v",
        seed=1,
        source_record_count=20,
        profiling_size=10,
        evaluation_size=10,
        profiling_ids=[f"p{index}" for index in range(10)],
        evaluation_ids=[f"e{index}" for index in range(10)],
        profiling_fingerprint="profile",
        evaluation_fingerprint="evaluation",
    )


def test_planner_selects_selective_predicate_first_and_ranks_all_candidates() -> None:
    result = CostBasedPlanner(OptimizerSettings()).plan(
        query(),
        profile([stats("A", 0.9), stats("B", 0.1), stats("C", 0.5)]),
        split(),
        provider_model="mock",
    )
    assert result.selected_predicate_order[0] == "B"
    assert result.candidate_count == 6
    assert [item.rank for item in result.candidates] == list(range(1, 7))
    assert result.candidates[0].selected


def test_planner_preserves_original_order_when_costs_tie() -> None:
    result = CostBasedPlanner(OptimizerSettings()).plan(
        query(),
        profile([stats("A", 1), stats("B", 1), stats("C", 1)]),
        split(),
    )
    assert result.selected_predicate_order == ["A", "B", "C"]
    assert not result.reordered


def test_planner_rejects_incompatible_prompt_statistics() -> None:
    incompatible = profile([stats("A", 0.2), stats("B", 0.2), stats("C", 0.2)])
    incompatible.predicate_hashes["A"] = "wrong"
    with pytest.raises(ValueError, match="definitions"):
        CostBasedPlanner(OptimizerSettings()).plan(query(), incompatible, split())
