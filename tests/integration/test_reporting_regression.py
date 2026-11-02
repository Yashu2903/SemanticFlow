import json

import httpx
import pytest

from semanticflow.dataset import ReviewRecord
from semanticflow.executor import SequentialExecutor
from semanticflow.metrics.reporter import build_report
from semanticflow.providers.jev import JevProvider
from semanticflow.query import QuerySpec


def historical_query(max_calls: int = 10) -> QuerySpec:
    return QuerySpec.model_validate(
        {
            "query_id": "product_quality",
            "workload_id": "workload_a_product_quality",
            "dataset_path": "data/processed/reviews_1000.parquet",
            "dataset_version": "semceb_amazon_arts_crafts_sewing_sample_1000",
            "limit": 10,
            "threshold": 0.5,
            "max_calls": max_calls,
            "predicates": [
                {"id": "poor_quality", "instructions": "Poor quality?"},
                {"id": "product_defect", "instructions": "Defect?"},
                {"id": "dissatisfaction", "instructions": "Dissatisfied?"},
            ],
        }
    )


@pytest.mark.asyncio
async def test_historical_jev_budget_exhaustion_reporting_without_network() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        question_name = next(iter(json.loads(request.content)["questions"]))
        probability = 0.9 if calls <= 2 else 0.1
        return httpx.Response(
            200,
            json={
                "model": "jev-1.14.0",
                "answers": {question_name: {"type": "noul", "noul": probability}},
                "usage": {"input_tokens": 10, "output_tokens": 2},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = JevProvider(
        "fixture-key",
        "jev-latest",
        client=client,
        max_retries=2,
        backoff_seconds=0,
    )
    records = [ReviewRecord(str(index), f"review {index}", {}) for index in range(10)]
    execution = await SequentialExecutor(provider, concurrency=4).execute(
        records, historical_query(), "historical-regression"
    )
    report = build_report(execution, historical_query(), provider="jev", model="jev-latest")
    await client.aclose()

    assert calls == 10
    assert report["requested_evaluations"] == 12
    assert report["completed_decisions"] == 10
    assert report["budget_blocked_evaluations"] == 2
    assert report["http_attempts"] == 10
    assert report["successful_api_calls"] == 10
    assert report["failed_http_attempts"] == 0
    assert report["retry_attempts"] == 0
    assert report["provider_errors"] == 0
    assert report["completion_status"] == "incomplete"
    assert report["incomplete_reason"] == "request_budget_exhausted"
    assert report["errors"] == []
    assert len(report["budget_diagnostics"]) == 2
    assert report["stages_reached"] == 2
    assert report["stages_with_completed_decisions"] == 1

    first, second, third = report["predicates"]
    assert first["completed_decisions"] == 10
    assert first["passing_record_count"] == 2
    assert first["rejected_record_count"] == 8
    assert first["selectivity"] == 0.2
    assert second["input_record_count"] == 2
    assert second["requested_evaluations"] == 2
    assert second["completed_decisions"] == 0
    assert second["budget_blocked_evaluations"] == 2
    assert second["provider_errors"] == 0
    assert second["selectivity"] is None
    assert third["predicate_id"] == "dissatisfaction"
    assert third["reached"] is False
    assert third["requested_evaluations"] == 0
    assert third["completed_decisions"] == 0
    assert third["selectivity"] is None
    assert len(execution.traces) == 10


@pytest.mark.asyncio
async def test_retry_attempts_do_not_create_additional_logical_decisions() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503)
        return httpx.Response(
            200,
            json={
                "model": "jev-1.14.0",
                "answers": {"poor_quality": {"type": "noul", "noul": 0.9}},
                "usage": {},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = JevProvider(
        "fixture-key",
        "jev-latest",
        client=client,
        max_retries=1,
        backoff_seconds=0,
    )
    query = historical_query(max_calls=2)
    query.predicates = query.predicates[:1]
    execution = await SequentialExecutor(provider).execute(
        [ReviewRecord("1", "review", {})], query, "retry-regression"
    )
    await client.aclose()

    assert execution.requested_evaluations == 1
    assert execution.completed_decisions == 1
    assert execution.http_attempts == 2
    assert execution.failed_http_attempts == 1
    assert execution.retry_attempts == 1
    assert len(execution.traces) == 1


@pytest.mark.asyncio
async def test_invalid_provider_response_is_not_a_budget_or_http_failure() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={"answers": {}}))
    )
    provider = JevProvider(
        "fixture-key",
        "jev-latest",
        client=client,
        max_retries=0,
        backoff_seconds=0,
    )
    query = historical_query(max_calls=1)
    query.predicates = query.predicates[:1]
    execution = await SequentialExecutor(provider).execute(
        [ReviewRecord("1", "review", {})], query, "provider-error-regression"
    )
    await client.aclose()

    stage = execution.predicates[0]
    assert not execution.complete
    assert execution.incomplete_reason == "provider_error"
    assert stage.requested_evaluations == 1
    assert stage.completed_decisions == 0
    assert stage.budget_blocked_evaluations == 0
    assert stage.http_attempts == 1
    assert stage.failed_http_attempts == 0
    assert stage.provider_errors == 1
    assert stage.selectivity is None
