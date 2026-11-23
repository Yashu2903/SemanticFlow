from __future__ import annotations

import json
from pathlib import Path

import pytest

from semanticflow.optimizer.fusion import FusedDecision
from semanticflow.optimizer.m522_budget import BudgetLedger
from semanticflow.optimizer.m522_live_acquisition import (
    DeterministicMockProvider,
    LiveAcquisitionAdapter,
)
from semanticflow.optimizer.m522_live_controller import LiveProfilingController
from semanticflow.optimizer.m522_protocol import (
    build_action_universe,
    build_authorization_proposal,
    build_protocol,
    load_authorized_record_texts,
    policy_fingerprint,
)
from semanticflow.optimizer.m522_reconciliation import reconcile_session
from semanticflow.optimizer.m522_session import (
    AppendOnlyEventLog,
    SessionStatus,
    rebuild_session,
)


def make_controller(tmp_path: Path) -> LiveProfilingController:
    protocol = build_protocol()
    universe = build_action_universe(protocol)
    manifest = build_authorization_proposal(protocol, universe)
    return LiveProfilingController(
        protocol=protocol,
        universe=universe,
        event_log=AppendOnlyEventLog(tmp_path / "events.jsonl"),
        adapter=LiveAcquisitionAdapter(
            DeterministicMockProvider(model=protocol.model, scenario="informative")
        ),
        record_texts=load_authorized_record_texts(protocol, universe),
        authorization_scope_fingerprint=manifest.scope_fingerprint,
        session_id="fault-test",
        mock_only=True,
    )


def reconcile(item: LiveProfilingController, *, repair: bool) -> object:
    return reconcile_session(
        event_log=item.event_log,
        universe=item.universe,
        expected_protocol_fingerprint=item.protocol.protocol_fingerprint,
        expected_authorization_scope_fingerprint=item.authorization_scope_fingerprint,
        expected_policy_fingerprint=policy_fingerprint(item.protocol),
        physical_request_ceiling=item.protocol.physical_request_ceiling,
        http_attempt_ceiling=item.protocol.http_attempt_ceiling,
        maximum_retries=item.protocol.maximum_retries,
        repair=repair,
    )


@pytest.mark.asyncio
async def test_persisted_receipt_is_recovered_without_reissue(tmp_path: Path) -> None:
    item = make_controller(tmp_path)
    state = item.initialize()
    action, _ = item.choose_next(state)
    assert action is not None
    ledger = item._ledger(state)
    snapshot = ledger.reserve_action(
        action.action_id,
        predicted_input_tokens=action.predicted_input_tokens,
        predicted_cost_usd=action.predicted_cost_usd,
    )
    item.event_log.append(
        "action_reserved",
        {"action_id": action.action_id, "budget": snapshot.model_dump(mode="json")},
    )
    item.event_log.append("dispatch_started", {"action_id": action.action_id})
    receipt = await item.adapter.acquire(action, item.record_texts[action.record_id])
    item.event_log.append("receipt_persisted", {"receipt": receipt.model_dump(mode="json")})
    before = rebuild_session(item.event_log.read())
    assert before.pending_action_ids == (action.action_id,)
    result = reconcile(item, repair=True)
    assert result.valid  # type: ignore[attr-defined]
    assert result.recovered_receipts == 1  # type: ignore[attr-defined]
    after = rebuild_session(item.event_log.read())
    assert action.action_id in after.completed_action_ids
    assert after.budget.actual_http_attempts == 1


@pytest.mark.asyncio
async def test_persisted_raw_provider_response_is_validated_after_restart(
    tmp_path: Path,
) -> None:
    item = make_controller(tmp_path)
    state = item.initialize()
    action, _ = item.choose_next(state)
    assert action is not None
    ledger = item._ledger(state)
    snapshot = ledger.reserve_action(
        action.action_id,
        predicted_input_tokens=action.predicted_input_tokens,
        predicted_cost_usd=action.predicted_cost_usd,
    )
    item.event_log.append(
        "action_reserved",
        {"action_id": action.action_id, "budget": snapshot.model_dump(mode="json")},
    )
    item.event_log.append("dispatch_started", {"action_id": action.action_id})
    response = await item.adapter.dispatch(action, item.record_texts[action.record_id])
    item.event_log.append(
        "provider_response_persisted",
        {"response": response.model_dump(mode="json")},
    )

    before = rebuild_session(item.event_log.read())
    assert before.pending_action_ids == (action.action_id,)
    assert len(before.provider_responses) == 1
    result = reconcile(item, repair=True)

    assert result.valid  # type: ignore[attr-defined]
    assert result.recovered_provider_responses == 1  # type: ignore[attr-defined]
    assert result.next_action_may_execute  # type: ignore[attr-defined]
    after = rebuild_session(item.event_log.read())
    assert action.action_id in after.completed_action_ids
    assert after.budget.actual_http_attempts == response.http_attempts
    assert after.budget.input_tokens == response.input_tokens
    assert after.budget.output_tokens == response.output_tokens


@pytest.mark.asyncio
async def test_invalid_persisted_provider_response_recovers_failed_closed(
    tmp_path: Path,
) -> None:
    item = make_controller(tmp_path)
    state = item.initialize()
    action, _ = item.choose_next(state)
    assert action is not None
    ledger = item._ledger(state)
    snapshot = ledger.reserve_action(
        action.action_id,
        predicted_input_tokens=action.predicted_input_tokens,
        predicted_cost_usd=action.predicted_cost_usd,
    )
    item.event_log.append(
        "action_reserved",
        {"action_id": action.action_id, "budget": snapshot.model_dump(mode="json")},
    )
    item.event_log.append("dispatch_started", {"action_id": action.action_id})
    response = await item.adapter.dispatch(action, item.record_texts[action.record_id])
    response = response.model_copy(update={"resolved_model": "unexpected-model"})
    item.event_log.append(
        "provider_response_persisted",
        {"response": response.model_dump(mode="json")},
    )

    result = reconcile(item, repair=True)

    assert result.valid  # type: ignore[attr-defined]
    assert result.failed_provider_responses == 1  # type: ignore[attr-defined]
    assert not result.next_action_may_execute  # type: ignore[attr-defined]
    after = rebuild_session(item.event_log.read())
    assert after.status == SessionStatus.FAILED_CLOSED
    assert after.budget.actual_http_attempts == response.http_attempts
    assert after.budget.input_tokens == response.input_tokens
    assert after.budget.output_tokens == response.output_tokens


def test_dispatch_without_receipt_becomes_ambiguous(tmp_path: Path) -> None:
    item = make_controller(tmp_path)
    state = item.initialize()
    action, _ = item.choose_next(state)
    assert action is not None
    ledger = item._ledger(state)
    snapshot = ledger.reserve_action(
        action.action_id,
        predicted_input_tokens=action.predicted_input_tokens,
        predicted_cost_usd=action.predicted_cost_usd,
    )
    item.event_log.append(
        "action_reserved",
        {"action_id": action.action_id, "budget": snapshot.model_dump(mode="json")},
    )
    item.event_log.append("dispatch_started", {"action_id": action.action_id})
    result = reconcile(item, repair=True)
    assert not result.valid  # type: ignore[attr-defined]
    assert result.ambiguous_actions == 1  # type: ignore[attr-defined]
    assert not result.next_action_may_execute  # type: ignore[attr-defined]


def test_reconciliation_cannot_reactivate_failed_closed_session(tmp_path: Path) -> None:
    item = make_controller(tmp_path)
    state = item.initialize()
    action, _ = item.choose_next(state)
    assert action is not None
    ledger = item._ledger(state)
    snapshot = ledger.reserve_action(
        action.action_id,
        predicted_input_tokens=action.predicted_input_tokens,
        predicted_cost_usd=action.predicted_cost_usd,
    )
    item.event_log.append(
        "action_reserved",
        {"action_id": action.action_id, "budget": snapshot.model_dump(mode="json")},
    )
    item.event_log.append(
        "status_changed",
        {"status": SessionStatus.FAILED_CLOSED.value, "reason": "test"},
    )

    result = reconcile(item, repair=True)

    assert result.valid  # type: ignore[attr-defined]
    assert result.released_undispatched_reservations == 1  # type: ignore[attr-defined]
    assert not result.next_action_may_execute  # type: ignore[attr-defined]
    assert rebuild_session(item.event_log.read()).status == SessionStatus.FAILED_CLOSED


@pytest.mark.asyncio
async def test_mixed_recovery_cannot_reauthorize_ambiguous_session(tmp_path: Path) -> None:
    item = make_controller(tmp_path)
    state = item.initialize()
    first, _ = item.choose_next(state)
    assert first is not None
    remaining = tuple(
        action
        for action in item.universe.actions
        if action.action_id != first.action_id and not action.prerequisite_action_ids
    )
    second = remaining[0]
    ledger = item._ledger(state)
    for action in (first, second):
        snapshot = ledger.reserve_action(
            action.action_id,
            predicted_input_tokens=action.predicted_input_tokens,
            predicted_cost_usd=action.predicted_cost_usd,
        )
        item.event_log.append(
            "action_reserved",
            {"action_id": action.action_id, "budget": snapshot.model_dump(mode="json")},
        )
        item.event_log.append("dispatch_started", {"action_id": action.action_id})
    receipt = await item.adapter.acquire(first, item.record_texts[first.record_id])
    item.event_log.append("receipt_persisted", {"receipt": receipt.model_dump(mode="json")})

    result = reconcile(item, repair=True)
    assert not result.valid  # type: ignore[attr-defined]
    assert result.recovered_receipts == 1  # type: ignore[attr-defined]
    assert result.ambiguous_actions == 1  # type: ignore[attr-defined]
    state = rebuild_session(item.event_log.read())
    assert state.status == SessionStatus.RECONCILIATION_REQUIRED
    attempts = state.budget.actual_http_attempts
    after = await item.step()
    assert after.status == SessionStatus.RECONCILIATION_REQUIRED
    assert after.budget.actual_http_attempts == attempts


def test_duplicate_persisted_receipt_fails_reconciliation(tmp_path: Path) -> None:
    item = make_controller(tmp_path)
    state = item.initialize()
    action, _ = item.choose_next(state)
    assert action is not None
    fake = {
        "action_id": action.action_id,
        "http_attempts": 1,
        "expected_question_ids": list(action.predicate_ids),
        "input_tokens": 1,
        "output_tokens": 1,
        "token_priced_cost_usd": 0.0,
        "response_identity": "duplicate",
    }
    item.event_log.append("receipt_persisted", {"receipt": fake})
    item.event_log.append("receipt_persisted", {"receipt": fake})
    result = reconcile(item, repair=False)
    assert not result.valid  # type: ignore[attr-defined]
    assert result.duplicate_receipts == 1  # type: ignore[attr-defined]


def test_corrupted_event_hash_chain_is_rejected(tmp_path: Path) -> None:
    item = make_controller(tmp_path)
    item.initialize()
    lines = item.event_log.path.read_text("utf-8").splitlines()
    value = json.loads(lines[-1])
    value["payload"]["status"] = SessionStatus.ABORTED.value
    lines[-1] = json.dumps(value)
    item.event_log.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    reloaded = AppendOnlyEventLog(item.event_log.path)
    with pytest.raises(ValueError, match="hash"):
        reloaded.read()


def test_mismatched_restart_scope_is_rejected(tmp_path: Path) -> None:
    item = make_controller(tmp_path)
    item.initialize()
    altered = LiveProfilingController(
        protocol=item.protocol,
        universe=item.universe,
        event_log=AppendOnlyEventLog(item.event_log.path),
        adapter=item.adapter,
        record_texts=item.record_texts,
        authorization_scope_fingerprint="wrong",
        session_id=item.session_id,
        mock_only=True,
    )
    with pytest.raises(PermissionError, match="scope"):
        altered.initialize()


def test_mock_session_cannot_restart_as_live_execution(tmp_path: Path) -> None:
    item = make_controller(tmp_path)
    item.initialize()
    altered = LiveProfilingController(
        protocol=item.protocol,
        universe=item.universe,
        event_log=AppendOnlyEventLog(item.event_log.path),
        adapter=item.adapter,
        record_texts=item.record_texts,
        authorization_scope_fingerprint=item.authorization_scope_fingerprint,
        session_id=item.session_id,
        mock_only=False,
    )
    with pytest.raises(PermissionError, match="scope"):
        altered.initialize()


@pytest.mark.asyncio
async def test_safe_interruption_resumes_same_scope(tmp_path: Path) -> None:
    item = make_controller(tmp_path)
    interrupted = item.interrupt()
    assert interrupted.status == SessionStatus.INTERRUPTED
    resumed = await item.step()
    assert len(resumed.completed_action_ids) == 1
    assert resumed.status == SessionStatus.EXECUTING


def test_retry_and_attempt_metadata_are_charged_exactly_once(tmp_path: Path) -> None:
    protocol = build_protocol()
    universe = build_action_universe(protocol)
    manifest = build_authorization_proposal(protocol, universe)
    item = LiveProfilingController(
        protocol=protocol,
        universe=universe,
        event_log=AppendOnlyEventLog(tmp_path / "events.jsonl"),
        adapter=LiveAcquisitionAdapter(
            DeterministicMockProvider(model=protocol.model, scenario="informative", attempts=3)
        ),
        record_texts=load_authorized_record_texts(protocol, universe),
        authorization_scope_fingerprint=manifest.scope_fingerprint,
        session_id="retry-test",
        mock_only=True,
    )
    import asyncio

    state = asyncio.run(item.step())
    assert state.budget.actual_http_attempts == 3
    assert len(state.budget.completed_action_ids) == 1


@pytest.mark.asyncio
async def test_post_request_validation_failure_charges_all_attempts(tmp_path: Path) -> None:
    class WrongModelAfterRetriesProvider:
        model_identifier = "jev-1.14.0"

        async def decide_fused(
            self, record_text: str, operator: object, budget: object
        ) -> FusedDecision:
            del record_text, operator, budget
            return FusedDecision(
                group_probability=0.5,
                predicate_probabilities={},
                model_identifier="unexpected-model",
                input_tokens=10,
                output_tokens=1,
                latency_seconds=0.01,
                request_attempts=3,
                successful_requests=1,
                failed_attempts=2,
                retries=2,
                metadata={
                    "raw_provider_response": {
                        "answers": {"single": {"type": "noul", "noul": 0.5}},
                        "usage": {"input_tokens": 10, "output_tokens": 1},
                    }
                },
            )

    item = make_controller(tmp_path)
    item.adapter = LiveAcquisitionAdapter(WrongModelAfterRetriesProvider())
    state = await item.step()
    assert state.status == SessionStatus.FAILED_CLOSED
    assert state.budget.actual_http_attempts == 3
    assert state.budget.input_tokens == 10
    assert state.budget.output_tokens == 1
    assert len(state.budget.completed_action_ids) == 1
    assert len(state.provider_responses) == 1
    assert state.provider_responses[0]["raw_provider_response"]["usage"] == {
        "input_tokens": 10,
        "output_tokens": 1,
    }
    event_types = [event.event_type for event in item.event_log.read()]
    assert event_types.index("provider_response_persisted") < event_types.index("action_failed")
    result = reconcile(item, repair=False)
    assert result.valid  # type: ignore[attr-defined]
    assert not result.next_action_may_execute  # type: ignore[attr-defined]


def test_budget_rejects_more_than_two_retries() -> None:
    ledger = BudgetLedger(
        physical_request_ceiling=1,
        http_attempt_ceiling=3,
        maximum_retries=2,
    )
    ledger.reserve_action("a", predicted_input_tokens=1, predicted_cost_usd=0)
    with pytest.raises(ValueError, match="retry"):
        ledger.charge_receipt(
            "a",
            attempts=4,
            named_question_evaluations=1,
            input_tokens=1,
            output_tokens=1,
            cost_proxy_usd=0,
        )


def test_action_outside_universe_is_rejected(tmp_path: Path) -> None:
    item = make_controller(tmp_path)
    with pytest.raises(PermissionError, match="outside"):
        item.universe.action("not-authorized")
