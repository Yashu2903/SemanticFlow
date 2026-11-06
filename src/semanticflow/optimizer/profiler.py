from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from semanticflow.dataset.loader import ReviewRecord
from semanticflow.metrics.collector import DecisionTrace
from semanticflow.optimizer.models import (
    OptimizerSettings,
    PredicateStatistics,
    ProfilingResult,
)
from semanticflow.optimizer.planner import predicate_hash
from semanticflow.optimizer.selectivity import observed_selectivity, wilson_interval
from semanticflow.providers.base import BudgetExceeded, InferenceProvider, RequestBudget
from semanticflow.query.models import PredicateSpec, QuerySpec


@dataclass(slots=True)
class _ProfileStage:
    predicate: PredicateSpec
    traces: list[DecisionTrace]
    requested_evaluations: int
    budget_blocked_evaluations: int
    provider_attempts: int
    http_attempts: int
    successful_api_calls: int
    failed_http_attempts: int
    retry_attempts: int
    provider_errors: int


class PredicateProfiler:
    """Profiles every predicate unconditionally on every profiling record."""

    def __init__(self, provider: InferenceProvider, settings: OptimizerSettings) -> None:
        self.provider = provider
        self.settings = settings

    async def profile(
        self,
        records: list[ReviewRecord],
        query: QuerySpec,
        *,
        run_id: str,
        dataset_fingerprint: str,
        profiling_fingerprint: str,
        pricing: dict[str, Any] | None = None,
    ) -> ProfilingResult:
        if not records:
            raise ValueError("profiling requires at least one record")
        if not 2 <= len(query.predicates) <= 6:
            raise ValueError("profiling requires 2 to 6 predicates")
        started = time.perf_counter()
        budget = RequestBudget(self.settings.profiling.max_attempts)
        all_traces: list[DecisionTrace] = []
        stages: list[_ProfileStage] = []
        errors: list[str] = []
        budget_diagnostics: list[str] = []
        incomplete_reason: str | None = None

        for stage_number, predicate in enumerate(query.predicates, start=1):
            provider_before = self.provider.stats.copy()
            traces: list[DecisionTrace] = []
            requested = 0
            stage_errors: list[str] = []
            stage_budget: list[str] = []
            for offset in range(0, len(records), self.settings.profiling.concurrency):
                batch = records[offset : offset + self.settings.profiling.concurrency]
                requested += len(batch)
                outcomes = await asyncio.gather(
                    *(
                        self._evaluate(
                            record,
                            predicate,
                            stage_number,
                            query.threshold,
                            budget,
                            run_id,
                        )
                        for record in batch
                    ),
                    return_exceptions=True,
                )
                for record, outcome in zip(batch, outcomes, strict=True):
                    if isinstance(outcome, BaseException):
                        diagnostic = (
                            f"{predicate.id}/{record.review_id}: "
                            f"{type(outcome).__name__}: {outcome}"
                        )
                        if isinstance(outcome, BudgetExceeded):
                            stage_budget.append(diagnostic)
                        else:
                            stage_errors.append(diagnostic)
                    else:
                        traces.append(outcome)
                if stage_errors or stage_budget:
                    break

            provider_delta = self.provider.stats.delta(provider_before)
            is_http_provider = self.provider.name == "jev"
            stages.append(
                _ProfileStage(
                    predicate=predicate,
                    traces=traces,
                    requested_evaluations=requested,
                    budget_blocked_evaluations=len(stage_budget),
                    provider_attempts=provider_delta.attempts,
                    http_attempts=provider_delta.attempts if is_http_provider else 0,
                    successful_api_calls=(
                        provider_delta.successful_requests if is_http_provider else 0
                    ),
                    failed_http_attempts=(
                        provider_delta.failed_attempts if is_http_provider else 0
                    ),
                    retry_attempts=provider_delta.retries,
                    provider_errors=len(stage_errors),
                )
            )
            all_traces.extend(traces)
            errors.extend(stage_errors)
            budget_diagnostics.extend(stage_budget)
            if stage_errors or stage_budget:
                if stage_errors and stage_budget:
                    incomplete_reason = "multiple_profiling_failures"
                elif stage_budget:
                    incomplete_reason = "profiling_request_budget_exhausted"
                else:
                    incomplete_reason = "profiling_provider_error"
                break

        pricing = pricing or {}
        input_rate = pricing.get("input_per_million_usd")
        output_rate = pricing.get("output_per_million_usd")
        statistics = [
            self._statistics(
                stage,
                sample_size=len(records),
                input_price_per_million=(float(input_rate) if input_rate is not None else None),
                output_price_per_million=(float(output_rate) if output_rate is not None else None),
            )
            for stage in stages
        ]
        complete = (
            incomplete_reason is None
            and len(stages) == len(query.predicates)
            and all(item.completed_decisions == len(records) for item in statistics)
        )
        input_tokens = sum(trace.input_tokens or 0 for trace in all_traces)
        output_tokens = sum(trace.output_tokens or 0 for trace in all_traces)
        estimated_cost = None
        if input_rate is not None and output_rate is not None:
            estimated_cost = (
                input_tokens * float(input_rate) + output_tokens * float(output_rate)
            ) / 1_000_000
        return ProfilingResult(
            complete=complete,
            incomplete_reason=None if complete else incomplete_reason or "profiling_incomplete",
            query_id=query.query_id,
            workload_id=query.workload_id or query.query_id,
            dataset_fingerprint=dataset_fingerprint,
            profiling_fingerprint=profiling_fingerprint,
            threshold=query.threshold,
            requested_model=self.provider.model_identifier,
            resolved_model_versions=sorted({trace.model_identifier for trace in all_traces}),
            predicate_order=[predicate.id for predicate in query.predicates],
            predicate_instructions={
                predicate.id: predicate.instructions for predicate in query.predicates
            },
            predicate_hashes={
                predicate.id: predicate_hash(predicate.id, predicate.instructions)
                for predicate in query.predicates
            },
            sample_size=len(records),
            requested_evaluations=sum(item.requested_evaluations for item in stages),
            completed_decisions=len(all_traces),
            budget_blocked_evaluations=sum(item.budget_blocked_evaluations for item in stages),
            provider_attempts=sum(item.provider_attempts for item in stages),
            http_attempts=sum(item.http_attempts for item in stages),
            successful_api_calls=sum(item.successful_api_calls for item in stages),
            failed_http_attempts=sum(item.failed_http_attempts for item in stages),
            retry_attempts=sum(item.retry_attempts for item in stages),
            provider_errors=sum(item.provider_errors for item in stages),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            summed_inference_latency_seconds=sum(
                trace.request_latency_seconds for trace in all_traces
            ),
            wall_time_seconds=time.perf_counter() - started,
            estimated_cost_usd=estimated_cost,
            pricing=pricing,
            statistics_version=self.settings.planner.statistics_version,
            statistics=statistics,
            traces=all_traces,
            errors=errors,
            budget_diagnostics=budget_diagnostics,
        )

    async def _evaluate(
        self,
        record: ReviewRecord,
        predicate: PredicateSpec,
        stage: int,
        threshold: float,
        budget: RequestBudget,
        run_id: str,
    ) -> DecisionTrace:
        decision = await self.provider.decide(record.review_text, predicate, budget)
        return DecisionTrace(
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
            request_latency_seconds=decision.latency_seconds,
            cache_status="disabled",
            request_attempt_count=decision.request_attempts,
            timestamp=datetime.now(UTC).isoformat(),
            question_hash=predicate_hash(predicate.id, predicate.instructions),
            question_type=self.provider.question_type,
            provider_metadata=decision.metadata,
        )

    @staticmethod
    def _statistics(
        stage: _ProfileStage,
        *,
        sample_size: int,
        input_price_per_million: float | None,
        output_price_per_million: float | None,
    ) -> PredicateStatistics:
        completed = len(stage.traces)
        passing = sum(trace.passed for trace in stage.traces)
        selectivity = observed_selectivity(passing, completed)
        interval = wilson_interval(passing, completed)
        tokens_known = completed > 0 and all(
            trace.input_tokens is not None and trace.output_tokens is not None
            for trace in stage.traces
        )
        average_input = (
            sum(trace.input_tokens or 0 for trace in stage.traces) / completed
            if tokens_known
            else None
        )
        average_output = (
            sum(trace.output_tokens or 0 for trace in stage.traces) / completed
            if tokens_known
            else None
        )
        average_cost = None
        if (
            average_input is not None
            and average_output is not None
            and input_price_per_million is not None
            and output_price_per_million is not None
        ):
            average_cost = (
                average_input * input_price_per_million + average_output * output_price_per_million
            ) / 1_000_000
        return PredicateStatistics(
            predicate_id=stage.predicate.id,
            question_hash=predicate_hash(stage.predicate.id, stage.predicate.instructions),
            sample_size=sample_size,
            completed_decisions=completed,
            passing_decisions=passing,
            rejected_decisions=completed - passing,
            estimated_selectivity=selectivity,
            estimated_rejection_rate=(1 - selectivity if selectivity is not None else None),
            wilson_lower=interval[0] if interval else None,
            wilson_upper=interval[1] if interval else None,
            average_input_tokens=average_input,
            average_output_tokens=average_output,
            average_total_tokens=(
                average_input + average_output
                if average_input is not None and average_output is not None
                else None
            ),
            average_inference_latency_seconds=(
                sum(trace.request_latency_seconds for trace in stage.traces) / completed
                if completed
                else 0.0
            ),
            average_estimated_cost_usd=average_cost,
            provider_attempts=stage.provider_attempts,
            http_attempts=stage.http_attempts,
            successful_api_calls=stage.successful_api_calls,
            failed_http_attempts=stage.failed_http_attempts,
            retry_attempts=stage.retry_attempts,
            provider_errors=stage.provider_errors,
            budget_blocked_evaluations=stage.budget_blocked_evaluations,
            model_versions=sorted({trace.model_identifier for trace in stage.traces}),
        )
