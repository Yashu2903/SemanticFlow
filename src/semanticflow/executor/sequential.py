from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from semanticflow.dataset.loader import ReviewRecord
from semanticflow.executor.cache import RequestCache, cache_key
from semanticflow.metrics.collector import DecisionTrace, ExecutionResult, PredicateMetrics
from semanticflow.providers.base import InferenceProvider, ProviderError, RequestBudget
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
        result = ExecutionResult(run_id=run_id, initial_record_count=len(records))
        active = list(records)
        budget = RequestBudget(query.max_calls)

        for stage, predicate in enumerate(query.predicates, start=1):
            if not active:
                result.predicates.append(
                    PredicateMetrics(
                        predicate_id=predicate.id,
                        stage=stage,
                        input_record_count=0,
                        passing_record_count=0,
                        rejected_record_count=0,
                        selectivity=0.0,
                        logical_evaluations=0,
                        actual_provider_calls=0,
                        request_attempts=0,
                        successful_requests=0,
                        failed_attempts=0,
                        retries=0,
                        input_tokens=0,
                        output_tokens=0,
                        inference_latency_seconds=0.0,
                        errors=0,
                        cache_hits=0,
                        cache_misses=0,
                    )
                )
                continue

            provider_before = self.provider.stats.copy()
            cache_before = self.cache.stats.copy()
            input_count = len(active)
            stage_results: list[_Evaluated] = []
            stage_errors: list[str] = []
            logical_evaluations = 0

            # Batches cap both active tasks and total created tasks at any one time.
            for offset in range(0, len(active), self.concurrency):
                batch = active[offset : offset + self.concurrency]
                logical_evaluations += len(batch)
                evaluated = await asyncio.gather(
                    *(
                        self._evaluate(record, predicate, stage, query.threshold, budget, run_id)
                        for record in batch
                    ),
                    return_exceptions=True,
                )
                for record, item in zip(batch, evaluated, strict=True):
                    if isinstance(item, BaseException):
                        stage_errors.append(
                            f"{predicate.id}/{record.review_id}: {type(item).__name__}: {item}"
                        )
                    else:
                        stage_results.append(item)
                if stage_errors:
                    break

            provider_delta = self.provider.stats.delta(provider_before)
            cache_delta = self.cache.stats.delta(cache_before)
            traces = [item.trace for item in stage_results]
            result.traces.extend(traces)
            passed = [item.record for item in stage_results if item.trace.passed]
            rejected = sum(not item.trace.passed for item in stage_results)
            result.predicates.append(
                PredicateMetrics(
                    predicate_id=predicate.id,
                    stage=stage,
                    input_record_count=input_count,
                    passing_record_count=len(passed),
                    rejected_record_count=rejected,
                    selectivity=(len(passed) / input_count) if input_count else 0.0,
                    logical_evaluations=logical_evaluations,
                    actual_provider_calls=provider_delta.calls,
                    request_attempts=provider_delta.attempts,
                    successful_requests=provider_delta.successful_requests,
                    failed_attempts=provider_delta.failed_attempts,
                    retries=provider_delta.retries,
                    input_tokens=sum(trace.input_tokens or 0 for trace in traces),
                    output_tokens=sum(trace.output_tokens or 0 for trace in traces),
                    inference_latency_seconds=sum(
                        trace.request_latency_seconds
                        for trace in traces
                        if trace.cache_status == "miss"
                    ),
                    errors=len(stage_errors),
                    cache_hits=cache_delta.hits,
                    cache_misses=cache_delta.misses,
                )
            )
            if stage_errors:
                result.complete = False
                result.errors.extend(stage_errors)
                active = []
                break
            active = passed

        result.matching_records = active if result.complete else []
        result.wall_time_seconds = time.perf_counter() - started
        return result

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

        try:
            cached = await self.cache.get_or_compute(key, compute)
        except ProviderError:
            raise
        decision = cached.decision
        question_hash = hashlib.sha256(
            f"{self.provider.question_type}\0{predicate.id}\0{predicate.instructions}".encode()
        ).hexdigest()
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
            cache_status="hit" if cached.hit else "miss",
            request_attempt_count=0 if cached.hit else decision.request_attempts,
            timestamp=datetime.now(UTC).isoformat(),
            question_hash=question_hash,
            question_type=self.provider.question_type,
            provider_metadata=decision.metadata,
        )
        return _Evaluated(record=record, trace=trace)
