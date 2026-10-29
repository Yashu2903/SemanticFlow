from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from semanticflow.providers.base import Decision, InferenceProvider
from semanticflow.query.models import PredicateSpec


def cache_key(
    provider: InferenceProvider,
    record_text: str,
    predicate: PredicateSpec,
    inference_parameters: dict[str, Any] | None = None,
) -> str:
    payload = {
        **provider.cache_identity,
        "record_text": record_text,
        "predicate_id": predicate.id,
        "instructions": predicate.instructions,
        "inference_parameters": inference_parameters or {},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class CacheResult:
    decision: Decision
    hit: bool


@dataclass(slots=True)
class CacheStats:
    hits: int = 0
    misses: int = 0

    def copy(self) -> CacheStats:
        return CacheStats(hits=self.hits, misses=self.misses)

    def delta(self, previous: CacheStats) -> CacheStats:
        return CacheStats(hits=self.hits - previous.hits, misses=self.misses - previous.misses)


class RequestCache:
    """Per-run memory cache with single-flight suppression for concurrent duplicates."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self.stats = CacheStats()
        self._values: dict[str, Decision] = {}
        self._inflight: dict[str, asyncio.Future[Decision]] = {}
        self._lock = asyncio.Lock()

    async def get_or_compute(
        self, key: str, compute: Callable[[], Awaitable[Decision]]
    ) -> CacheResult:
        if not self.enabled:
            self.stats.misses += 1
            return CacheResult(await compute(), hit=False)

        leader = False
        async with self._lock:
            if key in self._values:
                self.stats.hits += 1
                return CacheResult(self._values[key], hit=True)
            future = self._inflight.get(key)
            if future is None:
                future = asyncio.get_running_loop().create_future()
                # Consume a future exception even when no follower is awaiting it.
                future.add_done_callback(
                    lambda item: item.exception() if not item.cancelled() else None
                )
                self._inflight[key] = future
                self.stats.misses += 1
                leader = True
            else:
                self.stats.hits += 1

        if not leader:
            return CacheResult(await future, hit=True)

        try:
            decision = await compute()
        except BaseException as exc:
            async with self._lock:
                self._inflight.pop(key, None)
                if not future.done():
                    future.set_exception(exc)
            raise
        async with self._lock:
            self._values[key] = decision
            self._inflight.pop(key, None)
            future.set_result(decision)
        return CacheResult(decision, hit=False)
