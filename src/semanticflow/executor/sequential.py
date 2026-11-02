from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from semanticflow.dataset.loader import ReviewRecord
from semanticflow.executor.cache import RequestCache, cache_key
from semanticflow.metrics.collector import DecisionTrace, ExecutionResult, PredicateMetrics
from semanticflow.providers.base import BudgetExceeded, InferenceProvider, RequestBudget
from semanticflow.query.models import PredicateSpec, QuerySpec


@dataclass(frozen=True, slots=True)
class _Evaluated:
    record: ReviewRecord
    trace: DecisionTrace


class SequentialExecutor:
    """Ordered AND evaluation with per-stage short circuiting; never reorders predicates."""

    def __init__(
        self,
        provider: InferenceProvider,
        *,
        concurrency: int = 4,
        cache: RequestCache | None = None,
    ) -> None:
        if concurrency <= 0:
            raise ValueError("concurrency must be positive")
        self.provider = provider
        self.concurrency = concurrency
        self.cache = cache or RequestCache(enabled=False)

    async def execute(
        self, records: list[ReviewRecord], query: QuerySpec, run_id: str
    ) -> ExecutionResult:
        started = time.perf_counter()
        result = ExecutionResult(
            run_id=run_id,
            initial_record_count=len(records),
            caching_enabled=self.cache.enabled,
        )
        active = list(records)
        budget = RequestBudget(query.max_calls)

        for stage, predicate in enumerate(query.predicates, start=1):
            if not active:
                result.predicates.append(self._empty_predicate_metrics(predicate, stage))
                continue

            provider_before = self.provider.stats.copy()
            cache_before = self.cache.stats.copy()
            input_count = len(active)
            stage_results: list[_Evaluated] = []
            stage_errors: list[str] = []
            stage_budget_diagnostics: list[str] = []
            requested_evaluations = 0

            # Batches cap both active tasks and total created tasks at any one time.
            for offset in range(0, len(active), self.concurrency):
                batch = active[offset : offset + self.concurrency]
                requested_evaluations += len(batch)
                evaluated = await asyncio.gather(
                    *(
                        self._evaluate(record, predicate, stage, query.threshold, budget, run_id)
                        for record in batch
                    ),
                    return_exceptions=True,
                )
                for record, item in zip(batch, evaluated, strict=True):
                    diagnostic = (
                        f"{predicate.id}/{record.review_id}: {type(item).__name__}: {item}"
                        if isinstance(item, BaseException)
                        else None
                    )
                    if isinstance(item, BudgetExceeded):
                        stage_budget_diagnostics.append(diagnostic or str(item))
                    elif isinstance(item, BaseException):
                        stage_errors.append(diagnostic or str(item))
                    else:
                        stage_results.append(item)
                if stage_errors or stage_budget_diagnostics:
                    break

            provider_delta = self.provider.stats.delta(provider_before)
            cache_delta = self.cache.stats.delta(cache_before)
            traces = [item.trace for item in stage_results]
            result.traces.extend(traces)
            passed = [item.record for item in stage_results if item.trace.passed]
            rejected = sum(not item.trace.passed for item in stage_results)
            completed_decisions = len(stage_results)
            is_http_provider = self.provider.name == "jev"
            result.predicates.append(
                PredicateMetrics(
                    predicate_id=predicate.id,
                    stage=stage,
                    reached=True,
                    caching_enabled=self.cache.enabled,
                    input_record_count=input_count,
                    requested_evaluations=requested_evaluations,
                    completed_decisions=completed_decisions,
                    budget_blocked_evaluations=len(stage_budget_diagnostics),
                    passing_record_count=len(passed),
                    rejected_record_count=rejected,
                    selectivity=(
                        len(passed) / completed_decisions if completed_decisions else None
                    ),
                    actual_provider_calls=provider_delta.calls,
                    provider_attempts=provider_delta.attempts,
                    successful_provider_calls=provider_delta.successful_requests,
                    http_attempts=provider_delta.attempts if is_http_provider else 0,
                    successful_api_calls=(
                        provider_delta.successful_requests if is_http_provider else 0
                    ),
                    failed_http_attempts=(
                        provider_delta.failed_attempts if is_http_provider else 0
                    ),
                    provider_errors=len(stage_errors),
                    retry_attempts=provider_delta.retries,
                    input_tokens=sum(trace.input_tokens or 0 for trace in traces),
                    output_tokens=sum(trace.output_tokens or 0 for trace in traces),
                    inference_latency_seconds=sum(
                        trace.request_latency_seconds
                        for trace in traces
                        if trace.cache_status != "hit"
                    ),
                    cache_hits=cache_delta.hits,
                    cache_misses=cache_delta.misses,
                )
            )
            if stage_errors or stage_budget_diagnostics:
                result.complete = False
                result.errors.extend(stage_errors)
                result.budget_diagnostics.extend(stage_budget_diagnostics)
                if stage_errors and stage_budget_diagnostics:
                    result.incomplete_reason = "multiple_execution_failures"
                elif stage_budget_diagnostics:
                    result.incomplete_reason = "request_budget_exhausted"
                else:
                    result.incomplete_reason = "provider_error"
                active = []
                continue
            active = passed

        result.matching_records = active if result.complete else []
        result.wall_time_seconds = time.perf_counter() - started
        return result

    def _empty_predicate_metrics(
        self, predicate: PredicateSpec, stage: int
    ) -> PredicateMetrics:
        return PredicateMetrics(
            predicate_id=predicate.id,
            stage=stage,
            reached=False,
            caching_enabled=self.cache.enabled,
            input_record_count=0,
            requested_evaluations=0,
            completed_decisions=0,
            budget_blocked_evaluations=0,
            passing_record_count=0,
            rejected_record_count=0,
            selectivity=None,
            actual_provider_calls=0,
            provider_attempts=0,
            successful_provider_calls=0,
            http_attempts=0,
            successful_api_calls=0,
            failed_http_attempts=0,
            provider_errors=0,
            retry_attempts=0,
            input_tokens=0,
            output_tokens=0,
            inference_latency_seconds=0.0,
            cache_hits=0,
            cache_misses=0,
        )

    async def _evaluate(
        self,
        record: ReviewRecord,
        predicate: PredicateSpec,
        stage: int,
        threshold: float,
        budget: RequestBudget,
        run_id: str,
    ) -> _Evaluated:
        key = cache_key(self.provider, record.review_text, predicate)

        async def compute():
            return await self.provider.decide(record.review_text, predicate, budget)

        cached = await self.cache.get_or_compute(key, compute)
        decision = cached.decision
        question_hash = hashlib.sha256(
            f"{self.provider.question_type}\0{predicate.id}\0{predicate.instructions}".encode()
        ).hexdigest()
        cache_status = (
            "disabled" if not self.cache.enabled else "hit" if cached.hit else "miss"
        )
        trace = DecisionTrace(
            run_id=run_id,
            review_id=record.review_id,
            predicate_id=predicate.id,
            predicate_stage=stage,
            probability=decision.probability,
            threshold=threshold,
            passed=decision.probability >= threshold,
            provider=self.provider.name,
            model_identifier=decision.model_identifier,
            input_tokens=decision.input_tokens,
            output_tokens=decision.output_tokens,
            request_latency_seconds=0.0 if cached.hit else decision.latency_seconds,
            cache_status=cache_status,
            request_attempt_count=0 if cached.hit else decision.request_attempts,
            timestamp=datetime.now(UTC).isoformat(),
            question_hash=question_hash,
            question_type=self.provider.question_type,
            provider_metadata=decision.metadata,
        )
        return _Evaluated(record=record, trace=trace)
