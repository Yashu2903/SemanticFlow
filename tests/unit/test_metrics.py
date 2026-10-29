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
    result.predicates.append(
        PredicateMetrics("p", 1, 2, 1, 1, 0.5, 2, 2, 3, 2, 1, 1, 100, 10, 1.5, 0, 0, 2)
    )
    return result


def test_metric_totals_and_unknown_cost() -> None:
    report = build_report(make_execution(), make_query(), provider="jev", model="m")
    assert report["logical_evaluations"] == 2
    assert report["http_attempts"] == 3
    assert report["successful_api_calls"] == 2
    assert report["failed_api_calls"] == 1
    assert report["retry_attempts"] == 1
    assert report["total_tokens"] == 110
    assert report["summed_inference_latency_seconds"] == 1.5
    assert report["average_inference_latency_seconds"] == 0.75
    assert report["predicate_stages_evaluated"] == 1
    assert report["estimated_cost_usd"] is None
    assert report["quality"]["status"] == "pending"


def test_configured_cost() -> None:
    report = build_report(
        make_execution(),
        make_query(),
        provider="jev",
        model="m",
        input_price_per_million=2.0,
        output_price_per_million=4.0,
    )
    assert report["estimated_cost_usd"] == (100 * 2 + 10 * 4) / 1_000_000
    assert report["retries"] == 1
