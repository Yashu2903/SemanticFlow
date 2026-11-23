from __future__ import annotations

import httpx
import pytest

from semanticflow.optimizer.fusion import build_fused_operator
from semanticflow.optimizer.m522_jev_adapter import PilotAJevProvider
from semanticflow.optimizer.m522_live_acquisition import (
    AmbiguousDispatchError,
    LiveAcquisitionAdapter,
)
from semanticflow.optimizer.m522_protocol import build_action_universe, build_protocol
from semanticflow.providers.base import (
    ProviderError,
    RequestBudget,
    ResponseValidationError,
)
from semanticflow.query.models import PredicateSpec


def operator() -> object:
    return build_fused_operator(
        (
            PredicateSpec(id="p0", instructions="zero"),
            PredicateSpec(id="p1", instructions="one"),
        ),
        mode="multi_decision",
        requested_model="m",
        threshold=0.5,
    )


@pytest.mark.asyncio
async def test_established_single_probability_shape_and_raw_usage_are_accepted() -> None:
    protocol = build_protocol()
    action = next(
        action
        for action in build_action_universe(protocol).actions
        if action.physical_operator.value == "single"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        question_id = next(iter(payload["questions"]))
        return httpx.Response(
            200,
            json={
                "model": protocol.model,
                "answers": {question_id: {"type": "noul", "noul": 0.73}},
                "usage": {"input_tokens": 321, "output_tokens": 7},
            },
        )

    provider = PilotAJevProvider(
        "unused",
        protocol.model,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    adapter = LiveAcquisitionAdapter(provider)
    response = await adapter.dispatch(action, "record")
    assert response.group_probability == 0.73
    assert response.predicate_probabilities == {action.predicate_ids[0]: 0.73}
    assert response.input_tokens == 321
    assert response.output_tokens == 7
    assert response.raw_provider_response is not None
    receipt = adapter.validate_response(action, response)
    assert receipt.probabilities == {action.predicate_ids[0]: 0.73}
    assert receipt.input_tokens == 321
    await provider.aclose()


@pytest.mark.asyncio
async def test_duplicate_named_answer_key_is_rejected() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=(
                '{"answers":{"p0":{"type":"noul","noul":0.4},'
                '"p0":{"type":"noul","noul":0.6},'
                '"p1":{"type":"noul","noul":0.5}},'
                '"usage":{"input_tokens":1,"output_tokens":1}}'
            ),
            headers={"content-type": "application/json"},
        )

    provider = PilotAJevProvider(
        "unused", "m", client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    with pytest.raises(ResponseValidationError, match="duplicate"):
        await provider.decide_fused("x", operator(), RequestBudget(1))
    await provider.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response_text,error",
    (
        ("not-json", "Malformed"),
        (
            '{"answers":{"p0":{"type":"noul","noul":0.4}},'
            '"usage":{"input_tokens":1,"output_tokens":1}}',
            "answer names differ",
        ),
        (
            '{"answers":{"p0":{"type":"noul","noul":0.4},'
            '"p1":{"type":"noul","noul":1.5}},'
            '"usage":{"input_tokens":1,"output_tokens":1}}',
            "probability outside",
        ),
        (
            '{"answers":{"p0":{"type":"noul","noul":0.4},"p1":{"type":"noul","noul":0.5}}}',
            "input_tokens",
        ),
    ),
)
async def test_malformed_or_incomplete_response_is_rejected(response_text: str, error: str) -> None:
    provider = PilotAJevProvider(
        "unused",
        "m",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _request: httpx.Response(200, text=response_text))
        ),
    )
    with pytest.raises(ResponseValidationError, match=error):
        await provider.decide_fused("x", operator(), RequestBudget(1))
    await provider.aclose()


@pytest.mark.asyncio
async def test_transport_ambiguity_is_not_retried_in_pilot_mode() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("ambiguous", request=request)

    provider = PilotAJevProvider(
        "unused",
        "m",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=2,
    )
    with pytest.raises(AmbiguousDispatchError) as captured:
        await provider.decide_fused("x", operator(), RequestBudget(3))
    assert captured.value.attempts == 1
    assert calls == 1
    await provider.aclose()


@pytest.mark.asyncio
async def test_definitive_retry_exhaustion_is_bounded_and_accounted() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, text="unavailable")

    provider = PilotAJevProvider(
        "unused",
        "m",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=2,
        backoff_seconds=0,
    )
    with pytest.raises(ProviderError) as captured:
        await provider.decide_fused("x", operator(), RequestBudget(3))
    assert captured.value.attempts == 3
    assert captured.value.retries == 2
    assert calls == 3
    await provider.aclose()
