from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from semanticflow.dataset.loader import ReviewRecord
from semanticflow.executor.cache import RequestCache
from semanticflow.optimizer.fusion import FusedDecision, FusedOperator
from semanticflow.optimizer.fusion_planner import FusionPhysicalPlan
from semanticflow.providers.base import BudgetExceeded, InferenceProvider, RequestBudget


@dataclass(frozen=True, slots=True)
class FusedDecisionTrace:
    run_id: str
    review_id: str
    group_id: str
    group_stage: int
    predicate_ids: tuple[str, ...]
    fusion_mode: str
    group_probability: float | None
    predicate_probabilities: dict[str, float]
    threshold: float
    passed: bool
    provider: str
    model_identifier: str
    input_tokens: int | None
    output_tokens: int | None
    request_latency_seconds: float
    request_attempt_count: int
    cache_status: str
    timestamp: str
    fusion_prompt_hash: str
    result_schema_version: str
    provider_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FusedGroupMetrics:
    group_id: str
    predicate_ids: tuple[str, ...]
    fusion_mode: str
    stage: int
    reached: bool
    input_record_count: int
    requested_group_evaluations: int
    completed_group_decisions: int
    passing_record_count: int
    rejected_record_count: int
    provider_attempts: int
    failed_attempts: int
    retries: int
    provider_errors: int
    budget_blocked_evaluations: int
    input_tokens: int
    output_tokens: int
    summed_inference_latency_seconds: float
    cache_hits: int
    cache_misses: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FusedExecutionResult:
    run_id: str
    plan_id: str
    initial_record_count: int
    matching_records: list[ReviewRecord] = field(default_factory=list)
    traces: list[FusedDecisionTrace] = field(default_factory=list)
    groups: list[FusedGroupMetrics] = field(default_factory=list)
    complete: bool = True
    incomplete_reason: str | None = None
    errors: list[str] = field(default_factory=list)
    budget_diagnostics: list[str] = field(default_factory=list)
    wall_time_seconds: float = 0.0

    @property
    def logical_group_evaluations(self) -> int:
        return sum(group.requested_group_evaluations for group in self.groups)

    @property
    def provider_attempts(self) -> int:
        return sum(group.provider_attempts for group in self.groups)

    @property
    def named_question_evaluations(self) -> int:
        return sum(
            group.completed_group_decisions
            * (len(group.predicate_ids) if group.fusion_mode == "multi_decision" else 1)
            for group in self.groups
        )

    @property
    def completed_physical_groups(self) -> int:
        return sum(group.completed_group_decisions for group in self.groups)

    @property
    def input_tokens(self) -> int:
        return sum(group.input_tokens for group in self.groups)

    @property
    def output_tokens(self) -> int:
        return sum(group.output_tokens for group in self.groups)


@dataclass(frozen=True, slots=True)
class _FusedEvaluated:
    record: ReviewRecord
    trace: FusedDecisionTrace


class FusedExecutor:
    """Execute ordered physical predicate groups with AND short-circuiting."""

    def __init__(
        self,
        provider: InferenceProvider,
        *,
        concurrency: int = 4,
        cache: RequestCache[FusedDecision] | None = None,
    ) -> None:
        if concurrency <= 0:
            raise ValueError("concurrency must be positive")
        if not hasattr(provider, "decide_fused"):
            raise TypeError("provider does not implement genuine fused requests")
        self.provider = provider
        self.concurrency = concurrency
        self.cache = cache or RequestCache(enabled=False)

    async def execute(
        self,
        records: list[ReviewRecord],
        plan: FusionPhysicalPlan,
        *,
        run_id: str,
        maximum_http_attempts: int,
    ) -> FusedExecutionResult:
        capabilities = getattr(self.provider, "fusion_capabilities", None)
        if capabilities is None:
            raise TypeError("provider has no declared fusion capabilities")
        for group in plan.groups:
            if not capabilities.supports(group.fusion_mode, len(group.predicate_ids)):
                raise ValueError(
                    f"provider does not support {group.fusion_mode} for {group.group_id}"
                )
        started = time.perf_counter()
        result = FusedExecutionResult(
            run_id=run_id,
            plan_id=plan.plan_id,
            initial_record_count=len(records),
        )
        active = list(records)
        budget = RequestBudget(maximum_http_attempts)
        for stage, group in enumerate(plan.groups, start=1):
            if not active:
                result.groups.append(self._empty_group(group, stage))
                continue
            provider_before = self.provider.stats.copy()
            cache_before = self.cache.stats.copy()
            input_count = len(active)
            stage_results: list[_FusedEvaluated] = []
            stage_errors: list[str] = []
            stage_budget: list[str] = []
            requested = 0
            for offset in range(0, len(active), self.concurrency):
                batch = active[offset : offset + self.concurrency]
                requested += len(batch)
                evaluated = await asyncio.gather(
                    *(
                        self._evaluate(record, group, stage, budget, run_id)
                        for record in batch
                    ),
                    return_exceptions=True,
                )
                for record, item in zip(batch, evaluated, strict=True):
                    if isinstance(item, BaseException):
                        diagnostic = (
                            f"{group.group_id}/{record.review_id}: "
                            f"{type(item).__name__}: {item}"
                        )
                        if isinstance(item, BudgetExceeded):
                            stage_budget.append(diagnostic)
                        else:
                            stage_errors.append(diagnostic)
                    else:
                        stage_results.append(item)
                if stage_errors or stage_budget:
                    break
            provider_delta = self.provider.stats.delta(provider_before)
            cache_delta = self.cache.stats.delta(cache_before)
            traces = [item.trace for item in stage_results]
            passed = [item.record for item in stage_results if item.trace.passed]
            result.traces.extend(traces)
            result.groups.append(
                FusedGroupMetrics(
                    group_id=group.group_id,
                    predicate_ids=group.predicate_ids,
                    fusion_mode=group.fusion_mode,
                    stage=stage,
                    reached=True,
                    input_record_count=input_count,
                    requested_group_evaluations=requested,
                    completed_group_decisions=len(stage_results),
                    passing_record_count=len(passed),
                    rejected_record_count=sum(not item.trace.passed for item in stage_results),
                    provider_attempts=provider_delta.attempts,
                    failed_attempts=provider_delta.failed_attempts,
                    retries=provider_delta.retries,
                    provider_errors=len(stage_errors),
                    budget_blocked_evaluations=len(stage_budget),
                    input_tokens=sum(trace.input_tokens or 0 for trace in traces),
                    output_tokens=sum(trace.output_tokens or 0 for trace in traces),
                    summed_inference_latency_seconds=sum(
                        trace.request_latency_seconds for trace in traces
                    ),
                    cache_hits=cache_delta.hits,
                    cache_misses=cache_delta.misses,
                )
            )
            if stage_errors or stage_budget or len(stage_results) != input_count:
                result.complete = False
                result.errors.extend(stage_errors)
                result.budget_diagnostics.extend(stage_budget)
                result.incomplete_reason = (
                    "request_budget_exhausted" if stage_budget else "provider_error"
                )
                active = []
                continue
            active = passed
        result.matching_records = active if result.complete else []
        result.wall_time_seconds = time.perf_counter() - started
        return result

    async def _evaluate(
        self,
        record: ReviewRecord,
        group: FusedOperator,
        stage: int,
        budget: RequestBudget,
        run_id: str,
    ) -> _FusedEvaluated:
        cache_payload = {
            **self.provider.cache_identity,
            "record_text": record.review_text,
            "group_id": group.group_id,
            "fusion_prompt_hash": group.fusion_prompt_hash,
            "threshold": group.threshold,
            "execution_parameters": dict(group.execution_parameters),
        }
        key = hashlib.sha256(
            json.dumps(cache_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

        async def compute() -> FusedDecision:
            return await self.provider.decide_fused(record.review_text, group, budget)  # type: ignore[attr-defined]

        cached = await self.cache.get_or_compute(key, compute)
        decision = cached.decision
        passed = decision.passes(group)
        return _FusedEvaluated(
            record=record,
            trace=FusedDecisionTrace(
                run_id=run_id,
                review_id=record.review_id,
                group_id=group.group_id,
                group_stage=stage,
                predicate_ids=group.predicate_ids,
                fusion_mode=group.fusion_mode,
                group_probability=decision.group_probability,
                predicate_probabilities=dict(decision.predicate_probabilities),
                threshold=group.threshold,
                passed=passed,
                provider=self.provider.name,
                model_identifier=decision.model_identifier,
                input_tokens=decision.input_tokens,
                output_tokens=decision.output_tokens,
                request_latency_seconds=0.0 if cached.hit else decision.latency_seconds,
                request_attempt_count=0 if cached.hit else decision.request_attempts,
                cache_status=(
                    "disabled"
                    if not self.cache.enabled
                    else "hit" if cached.hit else "miss"
                ),
                timestamp=datetime.now(UTC).isoformat(),
                fusion_prompt_hash=group.fusion_prompt_hash,
                result_schema_version=group.result_schema_version,
                provider_metadata=dict(decision.metadata),
            ),
        )

    def _empty_group(self, group: FusedOperator, stage: int) -> FusedGroupMetrics:
        return FusedGroupMetrics(
            group_id=group.group_id,
            predicate_ids=group.predicate_ids,
            fusion_mode=group.fusion_mode,
            stage=stage,
            reached=False,
            input_record_count=0,
            requested_group_evaluations=0,
            completed_group_decisions=0,
            passing_record_count=0,
            rejected_record_count=0,
            provider_attempts=0,
            failed_attempts=0,
            retries=0,
            provider_errors=0,
            budget_blocked_evaluations=0,
            input_tokens=0,
            output_tokens=0,
            summed_inference_latency_seconds=0.0,
            cache_hits=0,
            cache_misses=0,
        )
