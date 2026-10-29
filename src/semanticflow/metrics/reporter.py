from __future__ import annotations

from typing import Any

from semanticflow.metrics.collector import ExecutionResult
from semanticflow.metrics.evaluation import QualityMetrics
from semanticflow.query.models import QuerySpec


def build_report(
    execution: ExecutionResult,
    query: QuerySpec,
    *,
    provider: str,
    model: str,
    input_price_per_million: float | None = None,
    output_price_per_million: float | None = None,
    quality: QualityMetrics | None = None,
) -> dict[str, Any]:
    successful_requests = sum(item.successful_requests for item in execution.predicates)
    failed_attempts = sum(item.failed_attempts for item in execution.predicates)
    retry_attempts = sum(item.retries for item in execution.predicates)
    reviews_with_decisions = len({trace.review_id for trace in execution.traces})
    predicate_decisions = len(execution.traces)
    predicate_stages_evaluated = sum(
        item.logical_evaluations > 0 for item in execution.predicates
    )
    average_inference_latency = (
        execution.inference_latency_seconds / successful_requests if successful_requests else 0.0
    )
    if input_price_per_million is None or output_price_per_million is None:
        estimated_cost = None
    else:
        estimated_cost = (
            execution.input_tokens * input_price_per_million
            + execution.output_tokens * output_price_per_million
        ) / 1_000_000
    return {
        "query_id": query.query_id,
        "workload_id": query.workload_id,
        "run_id": execution.run_id,
        "dataset_version": query.dataset_version,
        "provider": provider,
        "model_identifier": model,
        "predicate_order": [predicate.id for predicate in query.predicates],
        "threshold": query.threshold,
        "initial_record_count": execution.initial_record_count,
        "reviews_evaluated": reviews_with_decisions,
        "predicate_decisions_evaluated": predicate_decisions,
        "predicate_stages_evaluated": predicate_stages_evaluated,
        "final_record_count": len(execution.matching_records),
        "logical_evaluations": execution.logical_evaluations,
        "actual_provider_calls": execution.actual_provider_calls,
        "successful_api_calls": successful_requests,
        "failed_api_calls": failed_attempts,
        "http_attempts": execution.request_attempts if provider == "jev" else 0,
        "request_attempts": execution.request_attempts,
        "input_tokens": execution.input_tokens,
        "output_tokens": execution.output_tokens,
        "total_tokens": execution.input_tokens + execution.output_tokens,
        "wall_time_seconds": execution.wall_time_seconds,
        "summed_inference_latency_seconds": execution.inference_latency_seconds,
        "average_inference_latency_seconds": average_inference_latency,
        "estimated_cost_usd": estimated_cost,
        "cache_hits": sum(item.cache_hits for item in execution.predicates),
        "cache_misses": sum(item.cache_misses for item in execution.predicates),
        "retries": retry_attempts,
        "retry_attempts": retry_attempts,
        "failed_attempts": failed_attempts,
        "completion_status": "complete" if execution.complete else "incomplete",
        "errors": execution.errors,
        "predicates": [item.to_dict() for item in execution.predicates],
        "quality": quality.to_dict() if quality else {"status": "pending", "labeled_records": 0},
    }
