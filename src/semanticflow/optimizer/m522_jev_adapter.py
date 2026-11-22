from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx

from semanticflow.optimizer.m522_live_acquisition import AmbiguousDispatchError
from semanticflow.providers.base import (
    AuthenticationError,
    BudgetExceeded,
    ProviderError,
    RequestBudget,
    ResponseValidationError,
)
from semanticflow.providers.jev import JevProvider


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


class PilotAJevProvider(JevProvider):
    """Pilot-only strict receipt semantics using the established Jev client.

    Definitive 429/5xx responses may be retried. A transport failure after dispatch is
    ambiguous and is never automatically retried.
    """

    async def _request_questions(
        self,
        record_text: str,
        questions: dict[str, dict[str, str]],
        budget: RequestBudget,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        if not questions:
            raise ValueError("Jev requires at least one question")
        payload = {"model": self._model, "state": record_text, "questions": questions}
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
            else:
                self.stats.retries += 1
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
                raise AmbiguousDispatchError(
                    f"Jev transport failure after possible dispatch: {exc}",
                    attempts=attempts,
                ) from exc
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
                if retry_number < self.max_retries:
                    await asyncio.sleep(self.backoff_seconds * (2**retry_number))
                    continue
                break
            if response.status_code >= 400:
                self.stats.failed_attempts += 1
                raise ProviderError(
                    f"Jev rejected the request with HTTP {response.status_code}",
                    attempts=attempts,
                    failed_attempts=attempts,
                    retries=max(0, attempts - 1),
                )
            try:
                data = json.loads(response.text, object_pairs_hook=_unique_object)
                answers = data["answers"]
                if not isinstance(answers, dict):
                    raise TypeError("answers must be an object")
                if set(answers) != set(questions):
                    missing = sorted(set(questions) - set(answers))
                    extra = sorted(set(answers) - set(questions))
                    raise KeyError(f"answer names differ; missing={missing}, extra={extra}")
                probabilities: dict[str, float] = {}
                for question_name in questions:
                    answer = answers[question_name]
                    if answer.get("type") != self.question_type:
                        raise KeyError("unexpected answer type")
                    probability = float(answer["noul"])
                    if not 0 <= probability <= 1:
                        raise ValueError("probability outside [0, 1]")
                    probabilities[question_name] = probability
                usage = data.get("usage") or {}
                input_tokens = int(usage["input_tokens"])
                output_tokens = int(usage["output_tokens"])
                if input_tokens < 0 or output_tokens < 0:
                    raise ValueError("negative token usage")
                resolved_model = str(data.get("model") or self._model)
            except (KeyError, TypeError, ValueError) as exc:
                raise ResponseValidationError(
                    f"Malformed Jev response: {exc}",
                    attempts=attempts,
                    failed_attempts=attempts,
                    retries=max(0, attempts - 1),
                ) from exc
            self.stats.successful_requests += 1
            return probabilities, {
                "resolved_model": resolved_model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_seconds": time.perf_counter() - started,
                "attempts": attempts,
                "retries": attempts - 1,
                "http_status": response.status_code,
            }
        raise ProviderError(
            last_error,
            attempts=attempts,
            failed_attempts=attempts,
            retries=max(0, attempts - 1),
        )
