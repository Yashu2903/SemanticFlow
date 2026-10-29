from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from semanticflow.query.models import PredicateSpec


class ProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        attempts: int = 0,
        failed_attempts: int = 0,
        retries: int = 0,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.failed_attempts = failed_attempts
        self.retries = retries


class AuthenticationError(ProviderError):
    pass


class ResponseValidationError(ProviderError):
    pass


class BudgetExceeded(ProviderError):
    pass


class RequestBudget:
    """Concurrency-safe cap on all provider submission attempts, including retries."""

    def __init__(self, maximum: int) -> None:
        if maximum <= 0:
            raise ValueError("request budget must be positive")
        self.maximum = maximum
        self.used = 0
        self._lock = asyncio.Lock()

    async def reserve(self) -> None:
        async with self._lock:
            if self.used >= self.maximum:
                raise BudgetExceeded(
                    f"inference request budget exhausted ({self.used}/{self.maximum})"
                )
            self.used += 1


@dataclass(frozen=True, slots=True)
class Decision:
    probability: float
    model_identifier: str
    input_tokens: int | None
    output_tokens: int | None
    latency_seconds: float
    request_attempts: int = 1
    successful_requests: int = 1
    failed_attempts: int = 0
    retries: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        if self.input_tokens is not None and self.input_tokens < 0:
            raise ValueError("input_tokens cannot be negative")
        if self.output_tokens is not None and self.output_tokens < 0:
            raise ValueError("output_tokens cannot be negative")
        if self.latency_seconds < 0:
            raise ValueError("latency_seconds cannot be negative")


@dataclass(slots=True)
class ProviderStats:
    calls: int = 0
    attempts: int = 0
    successful_requests: int = 0
    failed_attempts: int = 0
    retries: int = 0

    def copy(self) -> ProviderStats:
        return ProviderStats(
            calls=self.calls,
            attempts=self.attempts,
            successful_requests=self.successful_requests,
            failed_attempts=self.failed_attempts,
            retries=self.retries,
        )

    def delta(self, earlier: ProviderStats) -> ProviderStats:
        return ProviderStats(
            calls=self.calls - earlier.calls,
            attempts=self.attempts - earlier.attempts,
            successful_requests=self.successful_requests - earlier.successful_requests,
            failed_attempts=self.failed_attempts - earlier.failed_attempts,
            retries=self.retries - earlier.retries,
        )


class InferenceProvider(ABC):
    name: str
    question_type = "noul"

    def __init__(self) -> None:
        self.stats = ProviderStats()

    @property
    @abstractmethod
    def model_identifier(self) -> str: ...

    @property
    def cache_identity(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model_identifier,
            "question_type": self.question_type,
        }

    @abstractmethod
    async def decide(
        self,
        record_text: str,
        predicate: PredicateSpec,
        budget: RequestBudget,
    ) -> Decision: ...

    async def aclose(self) -> None:
        return None

    async def __aenter__(self) -> InferenceProvider:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()
