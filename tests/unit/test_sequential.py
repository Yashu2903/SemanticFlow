import pytest

from semanticflow.dataset import ReviewRecord
from semanticflow.executor import RequestCache, SequentialExecutor
from semanticflow.providers.mock import MockProvider
from semanticflow.query import QuerySpec


def query(predicates=2, threshold=0.5, max_calls=100):
    return QuerySpec.model_validate(
        {
            "query_id": "q",
            "dataset_path": "unused.parquet",
            "limit": 100,
            "threshold": threshold,
            "max_calls": max_calls,
            "predicates": [
                {"id": f"p{i}", "instructions": f"Predicate {i}?"} for i in range(predicates)
            ],
        }
    )


@pytest.mark.asyncio
async def test_and_short_circuit_threshold_and_order() -> None:
    records = [ReviewRecord("a", "A", {}), ReviewRecord("b", "B", {})]
    provider = MockProvider(
        {("A", "p0"): 0.5, ("B", "p0"): 0.49, ("A", "p1"): 0.9, ("B", "p1"): 1.0}
    )
    result = await SequentialExecutor(provider, concurrency=2).execute(records, query(), "run")
    assert [row.review_id for row in result.matching_records] == ["a"]
    assert [(trace.review_id, trace.predicate_id) for trace in result.traces] == [
        ("a", "p0"),
        ("b", "p0"),
        ("a", "p1"),
    ]
    assert result.logical_evaluations == 3
    assert result.actual_provider_calls == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("value,expected", [(1.0, 5), (0.0, 0)])
async def test_all_pass_or_fail(value: float, expected: int) -> None:
    records = [ReviewRecord(str(i), str(i), {}) for i in range(5)]
    probabilities = {(str(i), f"p{j}"): value for i in range(5) for j in range(2)}
    result = await SequentialExecutor(MockProvider(probabilities)).execute(records, query(), "r")
    assert len(result.matching_records) == expected
    if value == 0:
        assert len(result.traces) == 5
        assert result.predicates[0].completed_decisions == 5
        assert result.predicates[0].selectivity == 0.0
    else:
        assert all(item.selectivity == 1.0 for item in result.predicates)


@pytest.mark.asyncio
async def test_empty_input_and_deterministic_output() -> None:
    result = await SequentialExecutor(MockProvider()).execute([], query(), "r")
    assert result.complete and not result.matching_records and not result.traces
    assert len(result.predicates) == 2
    assert all(item.selectivity is None for item in result.predicates)
    assert all(not item.reached for item in result.predicates)


@pytest.mark.asyncio
async def test_concurrency_limit() -> None:
    records = [ReviewRecord(str(i), str(i), {}) for i in range(12)]
    provider = MockProvider({(str(i), "p0"): 1 for i in range(12)}, delay_seconds=0.005)
    result = await SequentialExecutor(provider, concurrency=3).execute(records, query(1), "r")
    assert result.complete
    assert provider.max_active_calls == 3
    assert [row.review_id for row in result.matching_records] == [str(i) for i in range(12)]


@pytest.mark.asyncio
async def test_budget_exhaustion_marks_incomplete_and_returns_no_partial_matches() -> None:
    records = [ReviewRecord(str(i), str(i), {}) for i in range(5)]
    result = await SequentialExecutor(MockProvider(), concurrency=2).execute(
        records, query(1, max_calls=2), "r"
    )
    assert not result.complete
    assert not result.matching_records
    assert not result.errors
    assert result.budget_diagnostics
    assert result.incomplete_reason == "request_budget_exhausted"
    assert result.requested_evaluations == 4
    assert result.completed_decisions == 2
    assert result.budget_blocked_evaluations == 2
    assert result.provider_errors == 0
    assert result.request_attempts == 2
    assert result.actual_provider_calls == 2


@pytest.mark.asyncio
async def test_cache_trace_and_counts() -> None:
    records = [ReviewRecord("a", "duplicate", {}), ReviewRecord("b", "duplicate", {})]
    provider = MockProvider({("duplicate", "p0"): 1}, delay_seconds=0.001)
    result = await SequentialExecutor(
        provider, concurrency=2, cache=RequestCache(enabled=True)
    ).execute(records, query(1), "r")
    assert result.actual_provider_calls == 1
    assert result.logical_evaluations == 2
    assert sorted(trace.cache_status for trace in result.traces) == ["hit", "miss"]
    assert sum(item.cache_hits for item in result.predicates) == 1
    assert sum(item.cache_misses for item in result.predicates) == 1


@pytest.mark.asyncio
async def test_partial_execution_selectivity_uses_only_completed_decisions() -> None:
    records = [ReviewRecord(str(i), str(i), {}) for i in range(5)]
    probabilities = {(str(i), "p0"): 1.0 if i == 0 else 0.0 for i in range(5)}
    result = await SequentialExecutor(MockProvider(probabilities), concurrency=5).execute(
        records, query(1, max_calls=3), "r"
    )
    stage = result.predicates[0]
    assert stage.requested_evaluations == 5
    assert stage.completed_decisions == 3
    assert stage.budget_blocked_evaluations == 2
    assert stage.passing_record_count == 1
    assert stage.rejected_record_count == 2
    assert stage.selectivity == pytest.approx(1 / 3)


@pytest.mark.asyncio
async def test_cache_disabled_does_not_record_lookups() -> None:
    records = [ReviewRecord("a", "A", {}), ReviewRecord("b", "B", {})]
    result = await SequentialExecutor(MockProvider()).execute(records, query(1), "r")
    stage = result.predicates[0]
    assert not result.caching_enabled
    assert not stage.caching_enabled
    assert stage.cache_hits == 0
    assert stage.cache_misses == 0
    assert all(trace.cache_status == "disabled" for trace in result.traces)
