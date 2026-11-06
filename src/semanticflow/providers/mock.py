from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Mapping

from semanticflow.providers.base import Decision, InferenceProvider, RequestBudget
from semanticflow.query.models import PredicateSpec


class MockProvider(InferenceProvider):
    """Credential-free deterministic test provider; never a research-quality model."""

    name = "mock"

    def __init__(
        self,
        probabilities: Mapping[tuple[str, str], float] | None = None,
        *,
        delay_seconds: float = 0.0,
        model: str = "mock-deterministic-v1",
        input_tokens_per_evaluation: int | None = None,
        output_tokens_per_evaluation: int | None = None,
        usage_by_predicate: Mapping[str, tuple[int, int]] | None = None,
    ) -> None:
        super().__init__()
        if input_tokens_per_evaluation is not None and input_tokens_per_evaluation < 0:
            raise ValueError("mock input token usage cannot be negative")
        if output_tokens_per_evaluation is not None and output_tokens_per_evaluation < 0:
            raise ValueError("mock output token usage cannot be negative")
        for predicate_id, usage in (usage_by_predicate or {}).items():
            if len(usage) != 2 or any(value < 0 for value in usage):
                raise ValueError(
                    f"mock token usage for {predicate_id} must contain two nonnegative values"
                )
        self.probabilities = dict(probabilities or {})
        self.delay_seconds = delay_seconds
        self._model = model
        self.input_tokens_per_evaluation = input_tokens_per_evaluation
        self.output_tokens_per_evaluation = output_tokens_per_evaluation
        self.usage_by_predicate = dict(usage_by_predicate or {})
        self.active_calls = 0
        self.max_active_calls = 0

    @property
    def model_identifier(self) -> str:
        return self._model

    def _probability(self, text: str, predicate: PredicateSpec) -> float:
        if (text, predicate.id) in self.probabilities:
            return float(self.probabilities[(text, predicate.id)])
        digest = hashlib.sha256(
            f"{self._model}\0{text}\0{predicate.id}\0{predicate.instructions}".encode()
        ).digest()
        return int.from_bytes(digest[:8], "big") / (2**64 - 1)

    async def decide(
        self, record_text: str, predicate: PredicateSpec, budget: RequestBudget
    ) -> Decision:
        await budget.reserve()
        self.stats.calls += 1
        self.stats.attempts += 1
        started = time.perf_counter()
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            if self.delay_seconds:
                await asyncio.sleep(self.delay_seconds)
            probability = self._probability(record_text, predicate)
        finally:
            self.active_calls -= 1
        self.stats.successful_requests += 1
        input_tokens, output_tokens = self.usage_by_predicate.get(
            predicate.id,
            (self.input_tokens_per_evaluation, self.output_tokens_per_evaluation),
        )
        return Decision(
            probability=probability,
            model_identifier=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=time.perf_counter() - started,
            metadata={
                "mock": True,
                "token_usage_available": input_tokens is not None
                and output_tokens is not None,
                "synthetic_token_usage": input_tokens is not None
                or output_tokens is not None,
            },
        )
