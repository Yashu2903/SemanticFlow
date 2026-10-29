import asyncio

import pytest

from semanticflow.executor.cache import RequestCache, cache_key
from semanticflow.providers.base import Decision
from semanticflow.providers.mock import MockProvider
from semanticflow.query import PredicateSpec


def decision(value=0.7):
    return Decision(value, "m", 1, 1, 0.1)


@pytest.mark.asyncio
async def test_hit_miss_disabled_and_failures_not_cached() -> None:
    cache = RequestCache(enabled=True)
    calls = 0

    async def compute():
        nonlocal calls
        calls += 1
        return decision()

    assert not (await cache.get_or_compute("x", compute)).hit
    assert (await cache.get_or_compute("x", compute)).hit
    assert calls == 1

    disabled = RequestCache(enabled=False)
    await disabled.get_or_compute("x", compute)
    await disabled.get_or_compute("x", compute)
    assert calls == 3

    failures = RequestCache(enabled=True)

    async def fail():
        raise RuntimeError("no")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await failures.get_or_compute("bad", fail)
    assert failures.stats.misses == 2


@pytest.mark.asyncio
async def test_concurrent_duplicate_single_flight() -> None:
    cache = RequestCache(enabled=True)
    calls = 0

    async def compute():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return decision()

    results = await asyncio.gather(*(cache.get_or_compute("same", compute) for _ in range(5)))
    assert calls == 1
    assert sum(item.hit for item in results) == 4


def test_key_varies_by_predicate_and_model() -> None:
    first = MockProvider(model="one")
    second = MockProvider(model="two")
    a = PredicateSpec(id="a", instructions="A")
    b = PredicateSpec(id="b", instructions="B")
    assert cache_key(first, "text", a) != cache_key(first, "text", b)
    assert cache_key(first, "text", a) != cache_key(second, "text", a)
