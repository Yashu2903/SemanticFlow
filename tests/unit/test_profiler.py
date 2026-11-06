import pytest

from semanticflow.dataset.loader import ReviewRecord
from semanticflow.optimizer.models import OptimizerSettings
from semanticflow.optimizer.profiler import PredicateProfiler
from semanticflow.providers.base import Decision, InferenceProvider, RequestBudget
from semanticflow.providers.mock import MockProvider
from semanticflow.query.models import PredicateSpec, QuerySpec


class RetryProvider(InferenceProvider):
    name = "jev"

    @property
    def model_identifier(self) -> str:
        return "recorded-jev"

    async def decide(
        self, record_text: str, predicate: PredicateSpec, budget: RequestBudget
    ) -> Decision:
        self.stats.calls += 1
        await budget.reserve()
        self.stats.attempts += 1
        self.stats.failed_attempts += 1
        self.stats.retries += 1
        await budget.reserve()
        self.stats.attempts += 1
        self.stats.successful_requests += 1
        return Decision(
            probability=0.75,
            model_identifier="jev-1.14.0",
            input_tokens=10,
            output_tokens=1,
            latency_seconds=0.01,
            request_attempts=2,
            failed_attempts=1,
            retries=1,
        )


def query(max_calls: int = 100) -> QuerySpec:
    return QuerySpec.model_validate(
        {
            "query_id": "profile",
            "dataset_path": "unused.parquet",
            "limit": 3,
            "threshold": 0.5,
            "max_calls": max_calls,
            "predicates": [
                {"id": "first", "instructions": "First?"},
                {"id": "second", "instructions": "Second?"},
            ],
        }
    )


@pytest.mark.asyncio
async def test_profiler_evaluates_every_predicate_unconditionally() -> None:
    records = [ReviewRecord(str(index), f"text {index}", {}) for index in range(3)]
    probabilities = {
        (record.review_text, predicate): (0.9 if predicate == "second" else 0.1)
        for record in records
        for predicate in ("first", "second")
    }
    settings = OptimizerSettings.model_validate(
        {"profiling": {"concurrency": 2, "max_attempts": 6}}
    )
    result = await PredicateProfiler(
        MockProvider(
            probabilities,
            input_tokens_per_evaluation=10,
            output_tokens_per_evaluation=2,
        ),
        settings,
    ).profile(
        records,
        query(),
        run_id="profile-run",
        dataset_fingerprint="dataset",
        profiling_fingerprint="sample",
    )
    assert result.complete
    assert result.completed_decisions == 6
    assert [item.completed_decisions for item in result.statistics] == [3, 3]
    assert [item.estimated_selectivity for item in result.statistics] == [0.0, 1.0]
    assert result.total_tokens == 72
    assert len(result.traces) == 6


@pytest.mark.asyncio
async def test_incomplete_profiling_is_rejected_before_planning() -> None:
    records = [ReviewRecord(str(index), f"text {index}", {}) for index in range(3)]
    settings = OptimizerSettings.model_validate(
        {"profiling": {"concurrency": 1, "max_attempts": 5}}
    )
    result = await PredicateProfiler(MockProvider(), settings).profile(
        records,
        query(),
        run_id="profile-run",
        dataset_fingerprint="dataset",
        profiling_fingerprint="sample",
    )
    assert not result.complete
    assert result.completed_decisions == 5
    assert result.budget_blocked_evaluations == 1
    with pytest.raises(ValueError, match="incomplete"):
        result.require_complete()


@pytest.mark.asyncio
async def test_profiler_counts_retries_as_attempts_not_logical_decisions() -> None:
    records = [ReviewRecord(str(index), f"text {index}", {}) for index in range(3)]
    settings = OptimizerSettings.model_validate(
        {"profiling": {"concurrency": 2, "max_attempts": 12}}
    )
    result = await PredicateProfiler(RetryProvider(), settings).profile(
        records,
        query(),
        run_id="retry-run",
        dataset_fingerprint="dataset",
        profiling_fingerprint="sample",
    )
    assert result.complete
    assert result.completed_decisions == 6
    assert result.provider_attempts == 12
    assert result.http_attempts == 12
    assert result.successful_api_calls == 6
    assert result.failed_http_attempts == 6
    assert result.retry_attempts == 6
    assert all(trace.request_attempt_count == 2 for trace in result.traces)


@pytest.mark.asyncio
async def test_profiler_measures_predicate_specific_mock_usage() -> None:
    records = [ReviewRecord(str(index), f"text {index}", {}) for index in range(3)]
    provider = MockProvider(
        usage_by_predicate={"first": (100, 10), "second": (20, 2)}
    )
    result = await PredicateProfiler(provider, OptimizerSettings()).profile(
        records,
        query(),
        run_id="usage-run",
        dataset_fingerprint="dataset",
        profiling_fingerprint="sample",
    )
    by_id = {item.predicate_id: item for item in result.statistics}
    assert by_id["first"].average_total_tokens == 110
    assert by_id["second"].average_total_tokens == 22
