from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx
from dotenv import load_dotenv

from semanticflow.providers.base import (
    AuthenticationError,
    BudgetExceeded,
    Decision,
    InferenceProvider,
    ProviderError,
    RequestBudget,
    ResponseValidationError,
)
from semanticflow.query.models import PredicateSpec

DEFAULT_URL = "https://api.typesafe.ai/v1/systemone"


class JevProvider(InferenceProvider):
    name = "jev"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        api_url: str = DEFAULT_URL,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        backoff_seconds: float = 0.5,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__()
        if not api_key.strip():
            raise ValueError("TYPESAFE_API_KEY must not be empty")
        if not model.strip():
            raise ValueError("Jev model identifier must not be empty")
        if timeout_seconds <= 0 or max_retries < 0 or backoff_seconds < 0:
            raise ValueError("invalid timeout or retry configuration")
        self.api_key = api_key
        self._model = model
        self.api_url = api_url
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)
        self._owns_client = client is None

    @classmethod
    def from_env(cls, model: str, **kwargs: Any) -> JevProvider:
        # Existing process variables win; .env is a local-development fallback.
        load_dotenv(override=False)
        api_key = os.getenv("TYPESAFE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("TYPESAFE_API_KEY is required for the Jev provider")
        return cls(api_key=api_key, model=model, **kwargs)

    @property
    def model_identifier(self) -> str:
        return self._model

    @property
    def cache_identity(self) -> dict[str, Any]:
        return {
            **super().cache_identity,
            "endpoint": self.api_url,
            "request_schema": "typesafe-systemone-v0.2.0",
        }

    async def decide(
        self, record_text: str, predicate: PredicateSpec, budget: RequestBudget
    ) -> Decision:
        question_name = predicate.id
        payload = {
            "model": self._model,
            "state": record_text,
            "questions": {
                question_name: {"type": self.question_type, "instructions": predicate.instructions}
            },
        }
        attempts = 0
        started = time.perf_counter()
        last_error = "request failed"
        for retry_number in range(self.max_retries + 1):
            try:
                await budget.reserve()
            except BudgetExceeded as exc:
                raise BudgetExceeded(
                    str(exc),
                    attempts=attempts,
                    failed_attempts=attempts,
                    retries=max(0, attempts - 1),
                ) from exc
            if attempts == 0:
                self.stats.calls += 1
            attempts += 1
            self.stats.attempts += 1
            try:
                response = await self._client.post(
                    self.api_url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=self.timeout_seconds,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                self.stats.failed_attempts += 1
                last_error = f"Jev transport failure: {exc}"
                retryable = True
            else:
                if response.status_code in {401, 403}:
                    self.stats.failed_attempts += 1
                    raise AuthenticationError(
                        "Jev authentication failed",
                        attempts=attempts,
                        failed_attempts=attempts,
                        retries=max(0, attempts - 1),
                    )
                if response.status_code == 429 or response.status_code >= 500:
                    self.stats.failed_attempts += 1
                    last_error = f"Jev transient HTTP {response.status_code}"
                    retryable = True
                elif response.status_code >= 400:
                    self.stats.failed_attempts += 1
                    raise ProviderError(
                        f"Jev rejected the request with HTTP {response.status_code}",
                        attempts=attempts,
                        failed_attempts=attempts,
                        retries=max(0, attempts - 1),
                    )
                else:
                    try:
                        data = response.json()
                        answer = data["answers"][question_name]
                        if answer.get("type") != self.question_type:
                            raise KeyError("unexpected answer type")
                        probability = float(answer["noul"])
                        if not 0.0 <= probability <= 1.0:
                            raise ValueError("probability outside [0, 1]")
                        usage = data.get("usage") or {}
                        input_tokens = usage.get("input_tokens")
                        output_tokens = usage.get("output_tokens")
                        if input_tokens is not None:
                            input_tokens = int(input_tokens)
                        if output_tokens is not None:
                            output_tokens = int(output_tokens)
                        if input_tokens is not None and input_tokens < 0:
                            raise ValueError("negative input token usage")
                        if output_tokens is not None and output_tokens < 0:
                            raise ValueError("negative output token usage")
                        resolved_model = str(data.get("model") or self._model)
                    except (KeyError, TypeError, ValueError) as exc:
                        self.stats.failed_attempts += 1
                        raise ResponseValidationError(
                            f"Malformed Jev response: {exc}",
                            attempts=attempts,
                            failed_attempts=attempts,
                            retries=max(0, attempts - 1),
                        ) from exc
                    self.stats.successful_requests += 1
                    retries = attempts - 1
                    return Decision(
                        probability=probability,
                        model_identifier=resolved_model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        latency_seconds=time.perf_counter() - started,
                        request_attempts=attempts,
                        successful_requests=1,
                        failed_attempts=retries,
                        retries=retries,
                        metadata={
                            "question_name": question_name,
                            "question_type": self.question_type,
                            "http_status": response.status_code,
                        },
                    )
            if not retryable or retry_number >= self.max_retries:
                break
            self.stats.retries += 1
            await asyncio.sleep(self.backoff_seconds * (2**retry_number))

        raise ProviderError(
            last_error,
            attempts=attempts,
            failed_attempts=attempts,
            retries=max(0, attempts - 1),
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
