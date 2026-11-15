import json

import httpx
import pytest

from semanticflow.optimizer.fusion import build_fused_operator
from semanticflow.providers.base import (
    AuthenticationError,
    ProviderError,
    RequestBudget,
    ResponseValidationError,
)
from semanticflow.providers.jev import JevProvider
from semanticflow.query import PredicateSpec

PREDICATE = PredicateSpec(id="quality", instructions="Is quality poor?")


@pytest.mark.asyncio
async def test_conjunctive_fusion_is_one_real_http_request() -> None:
    operator = build_fused_operator(
        [PREDICATE, PredicateSpec(id="value", instructions="Is value poor?")],
        mode="conjunctive_binary",
        requested_model="jev-latest",
        threshold=0.5,
    )
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        payload = json.loads(request.content)
        assert list(payload["questions"]) == [operator.group_id]
        return httpx.Response(
            200,
            json={
                "model": "jev-1.14.0",
                "answers": {
                    operator.group_id: {"type": "noul", "noul": 0.8}
                },
                "usage": {"input_tokens": 20, "output_tokens": 2},
            },
        )

    selected = provider(handler)
    result = await selected.decide_fused("text", operator, RequestBudget(1))
    assert requests == 1 and result.group_probability == 0.8
    assert not result.predicate_probabilities
    await selected.aclose()


@pytest.mark.asyncio
async def test_multi_decision_fusion_uses_named_answers_in_one_request() -> None:
    operator = build_fused_operator(
        [PREDICATE, PredicateSpec(id="value", instructions="Is value poor?")],
        mode="multi_decision",
        requested_model="jev-latest",
        threshold=0.5,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert set(payload["questions"]) == {"quality", "value"}
        return httpx.Response(
            200,
            json={
                "answers": {
                    "quality": {"type": "noul", "noul": 0.8},
                    "value": {"type": "noul", "noul": 0.2},
                },
                "usage": {"input_tokens": 20, "output_tokens": 4},
            },
        )

    selected = provider(handler)
    result = await selected.decide_fused("text", operator, RequestBudget(1))
    assert result.group_probability is None
    assert result.predicate_probabilities == {"quality": 0.8, "value": 0.2}
    assert result.passes(operator) is False
    await selected.aclose()


def provider(handler, retries=2):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return JevProvider(
        "secret", "jev-latest", client=client, max_retries=retries, backoff_seconds=0
    )


@pytest.mark.asyncio
async def test_request_schema_auth_and_response_parsing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer secret"
        payload = json.loads(request.content)
        assert payload == {
            "model": "jev-latest",
            "state": "bad item",
            "questions": {
                "quality": {"type": "noul", "instructions": "Is quality poor?"}
            },
        }
        return httpx.Response(
            200,
            json={
                "model": "jev-2026-09-15",
                "answers": {"quality": {"type": "noul", "noul": 0.91}},
                "usage": {"input_tokens": 12, "output_tokens": 2},
            },
        )

    selected = provider(handler)
    result = await selected.decide("bad item", PREDICATE, RequestBudget(3))
    assert result.probability == 0.91
    assert (result.input_tokens, result.output_tokens) == (12, 2)
    assert result.model_identifier == "jev-2026-09-15"
    await selected.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 500, 503])
async def test_transient_http_retries(status: int) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(status)
        return httpx.Response(
            200,
            json={"answers": {"quality": {"type": "noul", "noul": 0.5}}, "usage": {}},
        )

    selected = provider(handler)
    result = await selected.decide("x", PREDICATE, RequestBudget(3))
    assert result.request_attempts == 3 and result.retries == 2
    assert selected.stats.failed_attempts == 2
    assert selected.stats.retries == 2
    await selected.aclose()


@pytest.mark.asyncio
async def test_auth_is_not_retried() -> None:
    selected = provider(lambda _request: httpx.Response(401))
    with pytest.raises(AuthenticationError) as error:
        await selected.decide("x", PREDICATE, RequestBudget(10))
    assert error.value.attempts == 1
    await selected.aclose()


@pytest.mark.asyncio
async def test_malformed_and_invalid_probability() -> None:
    for payload in ({"answers": {}}, {"answers": {"quality": {"type": "noul", "noul": 2}}}):
        selected = provider(lambda _request, payload=payload: httpx.Response(200, json=payload))
        with pytest.raises(ResponseValidationError):
            await selected.decide("x", PREDICATE, RequestBudget(2))
        await selected.aclose()


@pytest.mark.asyncio
async def test_timeout_retry_exhaustion_and_budget() -> None:
    def timeout(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    selected = provider(timeout, retries=1)
    with pytest.raises(ProviderError) as error:
        await selected.decide("x", PREDICATE, RequestBudget(2))
    assert error.value.attempts == 2
    assert selected.stats.attempts == 2
    await selected.aclose()


def test_from_env_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr("semanticflow.providers.jev.load_dotenv", lambda **_kwargs: False)
    with pytest.raises(RuntimeError):
        JevProvider.from_env("jev-latest")


def test_from_env_loads_dotenv_without_overriding_process_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "process-key")
    called = {}

    def fake_load_dotenv(**kwargs):
        called.update(kwargs)
        return True

    monkeypatch.setattr("semanticflow.providers.jev.load_dotenv", fake_load_dotenv)
    selected = JevProvider.from_env("jev-latest")
    assert selected.api_key == "process-key"
    assert called == {"override": False}
