from semanticflow.dataset import ReviewRecord
from semanticflow.metrics.collector import ExecutionResult, PredicateMetrics
from semanticflow.metrics.reporter import build_report
from semanticflow.query import QuerySpec


def make_query():
    return QuerySpec.model_validate(
        {
            "query_id": "q",
            "dataset_path": "x",
            "limit": 2,
            "threshold": 0.5,
            "max_calls": 2,
            "predicates": [{"id": "p", "instructions": "P?"}],
        }
    )


def make_execution():
    result = ExecutionResult("r", 2, wall_time_seconds=1.0)
    result.matching_records.append(ReviewRecord("match", "text", {}))
    result.predicates.append(
        PredicateMetrics(
            predicate_id="p",
            stage=1,
            reached=True,
            caching_enabled=True,
            input_record_count=2,
            requested_evaluations=2,
            completed_decisions=2,
            budget_blocked_evaluations=0,
            passing_record_count=1,
            rejected_record_count=1,
            selectivity=0.5,
            actual_provider_calls=2,
            provider_attempts=3,
            successful_provider_calls=2,
            http_attempts=3,
            successful_api_calls=2,
            failed_http_attempts=1,
            provider_errors=0,
            retry_attempts=1,
            input_tokens=100,
            output_tokens=10,
            inference_latency_seconds=1.5,
            cache_hits=0,
            cache_misses=2,
        )
    )
    return result


def test_metric_totals_and_unknown_cost() -> None:
    report = build_report(make_execution(), make_query(), provider="jev", model="m")
    assert report["requested_evaluations"] == 2
    assert report["total_input_reviews"] == 2
    assert report["total_completed_reviews"] == 2
    assert report["final_matching_reviews"] == 1
    assert report["final_rejected_reviews"] == 1
    assert report["completed_decisions"] == 2
    assert report["budget_blocked_evaluations"] == 0
    assert report["logical_evaluations"] == 2
    assert report["http_attempts"] == 3
    assert report["successful_api_calls"] == 2
    assert report["failed_http_attempts"] == 1
    assert report["provider_errors"] == 0
    assert report["retry_attempts"] == 1
    assert report["total_tokens"] == 110
    assert report["summed_inference_latency_seconds"] == 1.5
    assert report["average_inference_latency_seconds"] == 0.75
    assert report["stages_reached"] == 1
    assert report["stages_with_completed_decisions"] == 1
    assert report["predicate_stages_evaluated"] == 1
    assert report["estimated_cost_usd"] is None
    assert report["predicates"][0]["total_tokens"] == 110
    assert report["predicates"][0]["average_inference_latency_seconds"] == 0.75
    assert report["predicates"][0]["estimated_cost_usd"] is None
    assert report["quality"]["status"] == "pending"


def test_configured_cost() -> None:
    report = build_report(
        make_execution(),
        make_query(),
        provider="jev",
        model="m",
        input_price_per_million=2.0,
        output_price_per_million=4.0,
        pricing_metadata={"id": "test-price"},
    )
    assert report["estimated_cost_usd"] == (100 * 2 + 10 * 4) / 1_000_000
    assert report["predicates"][0]["estimated_cost_usd"] == (
        100 * 2 + 10 * 4
    ) / 1_000_000
    assert report["pricing"] == {"id": "test-price"}
    assert report["retries"] == 1
