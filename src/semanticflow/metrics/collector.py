from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from semanticflow.dataset.loader import ReviewRecord

REPORT_SCHEMA_VERSION = "1.1.0"


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
    reached: bool
    caching_enabled: bool
    input_record_count: int
    requested_evaluations: int
    completed_decisions: int
    budget_blocked_evaluations: int
    passing_record_count: int
    rejected_record_count: int
    selectivity: float | None
    actual_provider_calls: int
    provider_attempts: int
    successful_provider_calls: int
    http_attempts: int
    successful_api_calls: int
    failed_http_attempts: int
    provider_errors: int
    retry_attempts: int
    input_tokens: int
    output_tokens: int
    inference_latency_seconds: float
    cache_hits: int
    cache_misses: int

    @property
    def logical_evaluations(self) -> int:
        """Deprecated alias for requested_evaluations."""
        return self.requested_evaluations

    @property
    def request_attempts(self) -> int:
        """Deprecated provider-neutral attempt count."""
        return self.provider_attempts

    @property
    def successful_requests(self) -> int:
        """Deprecated alias for successful_provider_calls."""
        return self.successful_provider_calls

    @property
    def failed_attempts(self) -> int:
        """Deprecated alias for failed_http_attempts."""
        return self.failed_http_attempts

    @property
    def retries(self) -> int:
        """Deprecated alias for retry_attempts."""
        return self.retry_attempts

    @property
    def errors(self) -> int:
        """Deprecated alias for provider_errors; budget blocks are not errors."""
        return self.provider_errors

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "logical_evaluations": self.requested_evaluations,
                "request_attempts": self.provider_attempts,
                "successful_requests": self.successful_provider_calls,
                "failed_attempts": self.failed_http_attempts,
                "retries": self.retry_attempts,
                "errors": self.provider_errors,
            }
        )
        return payload


@dataclass(slots=True)
class ExecutionResult:
    run_id: str
    initial_record_count: int
    caching_enabled: bool = False
    matching_records: list[ReviewRecord] = field(default_factory=list)
    traces: list[DecisionTrace] = field(default_factory=list)
    predicates: list[PredicateMetrics] = field(default_factory=list)
    complete: bool = True
    incomplete_reason: str | None = None
    errors: list[str] = field(default_factory=list)
    budget_diagnostics: list[str] = field(default_factory=list)
    wall_time_seconds: float = 0.0

    @property
    def requested_evaluations(self) -> int:
        return sum(item.requested_evaluations for item in self.predicates)

    @property
    def completed_decisions(self) -> int:
        return sum(item.completed_decisions for item in self.predicates)

    @property
    def budget_blocked_evaluations(self) -> int:
        return sum(item.budget_blocked_evaluations for item in self.predicates)

    @property
    def provider_errors(self) -> int:
        return sum(item.provider_errors for item in self.predicates)

    @property
    def logical_evaluations(self) -> int:
        """Deprecated alias for requested_evaluations."""
        return self.requested_evaluations

    @property
    def actual_provider_calls(self) -> int:
        return sum(item.actual_provider_calls for item in self.predicates)

    @property
    def provider_attempts(self) -> int:
        return sum(item.provider_attempts for item in self.predicates)

    @property
    def request_attempts(self) -> int:
        """Deprecated provider-neutral attempt count."""
        return self.provider_attempts

    @property
    def http_attempts(self) -> int:
        return sum(item.http_attempts for item in self.predicates)

    @property
    def successful_api_calls(self) -> int:
        return sum(item.successful_api_calls for item in self.predicates)

    @property
    def successful_provider_calls(self) -> int:
        return sum(item.successful_provider_calls for item in self.predicates)

    @property
    def failed_http_attempts(self) -> int:
        return sum(item.failed_http_attempts for item in self.predicates)

    @property
    def retry_attempts(self) -> int:
        return sum(item.retry_attempts for item in self.predicates)

    @property
    def input_tokens(self) -> int:
        return sum(item.input_tokens for item in self.predicates)

    @property
    def output_tokens(self) -> int:
        return sum(item.output_tokens for item in self.predicates)

    @property
    def inference_latency_seconds(self) -> float:
        return sum(item.inference_latency_seconds for item in self.predicates)
