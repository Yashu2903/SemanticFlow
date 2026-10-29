from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from semanticflow.dataset.loader import ReviewRecord


@dataclass(frozen=True, slots=True)
class DecisionTrace:
    run_id: str
    review_id: str
    predicate_id: str
    predicate_stage: int
    probability: float
    threshold: float
    passed: bool
    provider: str
    model_identifier: str
    input_tokens: int | None
    output_tokens: int | None
    request_latency_seconds: float
    cache_status: str
    request_attempt_count: int
    timestamp: str
    question_hash: str
    question_type: str
    provider_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PredicateMetrics:
    predicate_id: str
    stage: int
    input_record_count: int
    passing_record_count: int
    rejected_record_count: int
    selectivity: float
    logical_evaluations: int
    actual_provider_calls: int
    request_attempts: int
    successful_requests: int
    failed_attempts: int
    retries: int
    input_tokens: int
    output_tokens: int
    inference_latency_seconds: float
    errors: int
    cache_hits: int
    cache_misses: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExecutionResult:
    run_id: str
    initial_record_count: int
    matching_records: list[ReviewRecord] = field(default_factory=list)
    traces: list[DecisionTrace] = field(default_factory=list)
    predicates: list[PredicateMetrics] = field(default_factory=list)
    complete: bool = True
    errors: list[str] = field(default_factory=list)
    wall_time_seconds: float = 0.0

    @property
    def logical_evaluations(self) -> int:
        return sum(item.logical_evaluations for item in self.predicates)

    @property
    def actual_provider_calls(self) -> int:
        return sum(item.actual_provider_calls for item in self.predicates)

    @property
    def request_attempts(self) -> int:
        return sum(item.request_attempts for item in self.predicates)

    @property
    def input_tokens(self) -> int:
        return sum(item.input_tokens for item in self.predicates)

    @property
    def output_tokens(self) -> int:
        return sum(item.output_tokens for item in self.predicates)

    @property
    def inference_latency_seconds(self) -> float:
        return sum(item.inference_latency_seconds for item in self.predicates)

