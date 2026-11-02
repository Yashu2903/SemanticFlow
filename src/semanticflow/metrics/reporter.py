from __future__ import annotations

from typing import Any

from semanticflow.metrics.collector import REPORT_SCHEMA_VERSION, ExecutionResult
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
    pricing_metadata: dict[str, Any] | None = None,
    quality: QualityMetrics | None = None,
) -> dict[str, Any]:
    reviews_with_decisions = len({trace.review_id for trace in execution.traces})
    stages_reached = sum(item.reached for item in execution.predicates)
    stages_with_completed_decisions = sum(
        item.completed_decisions > 0 for item in execution.predicates
    )
    average_inference_latency = (
        execution.inference_latency_seconds / execution.successful_provider_calls
        if execution.successful_provider_calls
        else 0.0
    )
    if input_price_per_million is None or output_price_per_million is None:
        estimated_cost = None
    else:
        estimated_cost = (
            execution.input_tokens * input_price_per_million
            + execution.output_tokens * output_price_per_million
        ) / 1_000_000
    predicate_reports: list[dict[str, Any]] = []
    for item in execution.predicates:
        predicate_report = item.to_dict()
        predicate_report["total_tokens"] = item.input_tokens + item.output_tokens
        predicate_report["summed_inference_latency_seconds"] = (
            item.inference_latency_seconds
        )
        predicate_report["average_inference_latency_seconds"] = (
            item.inference_latency_seconds / item.successful_provider_calls
            if item.successful_provider_calls
            else 0.0
        )
        if input_price_per_million is None or output_price_per_million is None:
            predicate_report["estimated_cost_usd"] = None
        else:
            predicate_report["estimated_cost_usd"] = (
                item.input_tokens * input_price_per_million
                + item.output_tokens * output_price_per_million
            ) / 1_000_000
        predicate_reports.append(predicate_report)
    final_rejected_reviews = sum(
        item.rejected_record_count for item in execution.predicates
    )
    total_completed_reviews = final_rejected_reviews + len(execution.matching_records)
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "query_id": query.query_id,
        "workload_id": query.workload_id,
        "run_id": execution.run_id,
        "dataset_version": query.dataset_version,
        "provider": provider,
        "model_identifier": model,
        "predicate_order": [predicate.id for predicate in query.predicates],
        "threshold": query.threshold,
        "initial_record_count": execution.initial_record_count,
        "total_input_reviews": execution.initial_record_count,
        "total_completed_reviews": total_completed_reviews,
        "reviews_evaluated": reviews_with_decisions,
        "requested_evaluations": execution.requested_evaluations,
        "completed_decisions": execution.completed_decisions,
        "budget_blocked_evaluations": execution.budget_blocked_evaluations,
        "stages_reached": stages_reached,
        "stages_with_completed_decisions": stages_with_completed_decisions,
        "final_record_count": len(execution.matching_records),
        "final_matching_reviews": len(execution.matching_records),
        "final_rejected_reviews": final_rejected_reviews,
        "actual_provider_calls": execution.actual_provider_calls,
        "provider_attempts": execution.provider_attempts,
        "http_attempts": execution.http_attempts,
        "successful_api_calls": execution.successful_api_calls,
        "failed_http_attempts": execution.failed_http_attempts,
        "provider_errors": execution.provider_errors,
        "retry_attempts": execution.retry_attempts,
        "input_tokens": execution.input_tokens,
        "output_tokens": execution.output_tokens,
        "total_tokens": execution.input_tokens + execution.output_tokens,
        "wall_time_seconds": execution.wall_time_seconds,
        "summed_inference_latency_seconds": execution.inference_latency_seconds,
        "average_inference_latency_seconds": average_inference_latency,
        "estimated_cost_usd": estimated_cost,
        "pricing": pricing_metadata or {"status": "unverified"},
        "caching_enabled": execution.caching_enabled,
        "cache_hits": sum(item.cache_hits for item in execution.predicates),
        "cache_misses": sum(item.cache_misses for item in execution.predicates),
        "completion_status": "complete" if execution.complete else "incomplete",
        "incomplete_reason": execution.incomplete_reason,
        "errors": execution.errors,
        "budget_diagnostics": execution.budget_diagnostics,
        "predicates": predicate_reports,
        "quality": quality.to_dict() if quality else {"status": "pending", "labeled_records": 0},
        # Deprecated aliases retained for consumers of the original Milestone 1 schema.
        "logical_evaluations": execution.requested_evaluations,
        "predicate_decisions_evaluated": execution.completed_decisions,
        "predicate_stages_evaluated": stages_with_completed_decisions,
        "request_attempts": execution.provider_attempts,
        "failed_api_calls": execution.failed_http_attempts,
        "failed_attempts": execution.failed_http_attempts,
        "retries": execution.retry_attempts,
        "deprecated_metric_aliases": {
            "logical_evaluations": "requested_evaluations",
            "predicate_decisions_evaluated": "completed_decisions",
            "predicate_stages_evaluated": "stages_with_completed_decisions",
            "request_attempts": "provider_attempts",
            "failed_api_calls": "failed_http_attempts",
            "failed_attempts": "failed_http_attempts",
            "retries": "retry_attempts",
        },
    }
