from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from semanticflow.optimizer.m522_budget import BudgetSnapshot


class SessionStatus(StrEnum):
    CREATED = "CREATED"
    VALIDATED = "VALIDATED"
    AUTHORIZED = "AUTHORIZED"
    EXECUTING = "EXECUTING"
    INTERRUPTED = "INTERRUPTED"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"
    FAILED_CLOSED = "FAILED_CLOSED"
    INSTRUMENTATION_COMPLETE = "INSTRUMENTATION_COMPLETE"
    ABORTED = "ABORTED"


class SessionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(ge=0)
    timestamp: str
    event_type: str
    payload: dict[str, Any]
    previous_event_hash: str
    event_hash: str


class PersistentSessionState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "m5.2.2-persistent-session-v1"
    session_id: str
    protocol_fingerprint: str
    authorization_scope_fingerprint: str
    policy_fingerprint: str
    action_universe_fingerprint: str
    initial_seed: int
    objective: str
    mock_only: bool
    status: SessionStatus
    checkpoint: int = Field(default=0, ge=0)
    completed_action_ids: tuple[str, ...] = ()
    pending_action_ids: tuple[str, ...] = ()
    ambiguous_action_ids: tuple[str, ...] = ()
    representative_evidence_action_ids: tuple[str, ...] = ()
    discovery_evidence_action_ids: tuple[str, ...] = ()
    controller_decisions: tuple[dict[str, Any], ...] = ()
    provider_responses: tuple[dict[str, Any], ...] = ()
    receipts: tuple[dict[str, Any], ...] = ()
    budget: BudgetSnapshot
    termination_reason: str | None = None
    last_event_hash: str


def _event_hash(sequence: int, event_type: str, payload: dict[str, Any], previous: str) -> str:
    value = {
        "sequence": sequence,
        "event_type": event_type,
        "payload": payload,
        "previous_event_hash": previous,
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class AppendOnlyEventLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._events_cache: tuple[SessionEvent, ...] | None = None

    def read(self) -> tuple[SessionEvent, ...]:
        if self._events_cache is not None:
            return self._events_cache
        if not self.path.exists():
            self._events_cache = ()
            return ()
        events = tuple(
            SessionEvent.model_validate_json(line)
            for line in self.path.read_text("utf-8").splitlines()
            if line.strip()
        )
        previous = "GENESIS"
        for sequence, event in enumerate(events):
            if event.sequence != sequence or event.previous_event_hash != previous:
                raise ValueError("session event sequence or hash chain is corrupted")
            expected = _event_hash(
                event.sequence, event.event_type, event.payload, event.previous_event_hash
            )
            if event.event_hash != expected:
                raise ValueError("session event hash is corrupted")
            previous = event.event_hash
        self._events_cache = events
        return events

    def append(self, event_type: str, payload: dict[str, Any]) -> SessionEvent:
        with self._lock:
            events = self.read()
            previous = events[-1].event_hash if events else "GENESIS"
            sequence = len(events)
            event = SessionEvent(
                sequence=sequence,
                timestamp=datetime.now(UTC).isoformat(),
                event_type=event_type,
                payload=payload,
                previous_event_hash=previous,
                event_hash=_event_hash(sequence, event_type, payload, previous),
            )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(event.model_dump_json() + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._events_cache = (*events, event)
            return event


def rebuild_session(events: tuple[SessionEvent, ...]) -> PersistentSessionState:
    if not events or events[0].event_type != "session_created":
        raise ValueError("session log lacks creation event")
    creation = events[0].payload
    status = SessionStatus.CREATED
    pending: set[str] = set()
    completed: set[str] = set()
    ambiguous: set[str] = set()
    representative: set[str] = set()
    discovery: set[str] = set()
    decisions: list[dict[str, Any]] = []
    provider_responses: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    budget = BudgetSnapshot.model_validate(creation["budget"])
    termination: str | None = None
    checkpoint = 0
    for event in events[1:]:
        payload = event.payload
        if event.event_type == "status_changed":
            status = SessionStatus(payload["status"])
            termination = payload.get("reason")
        elif event.event_type == "action_selected":
            decisions.append(payload)
            checkpoint += 1
        elif event.event_type == "action_reserved":
            pending.add(payload["action_id"])
            budget = BudgetSnapshot.model_validate(payload["budget"])
        elif event.event_type == "provider_response_persisted":
            provider_responses.append(payload["response"])
        elif event.event_type == "receipt_persisted":
            receipts.append(payload["receipt"])
        elif event.event_type == "evidence_accepted":
            action_id = payload["action_id"]
            pending.discard(action_id)
            completed.add(action_id)
            (representative if payload["track"] == "certification" else discovery).add(action_id)
            budget = BudgetSnapshot.model_validate(payload["budget"])
        elif event.event_type == "action_failed":
            action_id = payload["action_id"]
            pending.discard(action_id)
            completed.add(action_id)
            budget = BudgetSnapshot.model_validate(payload["budget"])
        elif event.event_type == "reservation_released":
            pending.discard(payload["action_id"])
            budget = BudgetSnapshot.model_validate(payload["budget"])
        elif event.event_type == "ambiguous_receipt":
            action_id = payload["action_id"]
            pending.discard(action_id)
            ambiguous.add(action_id)
            budget = BudgetSnapshot.model_validate(payload["budget"])
            status = SessionStatus.RECONCILIATION_REQUIRED
        elif event.event_type == "interrupted":
            status = SessionStatus.INTERRUPTED
    return PersistentSessionState(
        session_id=creation["session_id"],
        protocol_fingerprint=creation["protocol_fingerprint"],
        authorization_scope_fingerprint=creation["authorization_scope_fingerprint"],
        policy_fingerprint=creation["policy_fingerprint"],
        action_universe_fingerprint=creation["action_universe_fingerprint"],
        initial_seed=creation["initial_seed"],
        objective=creation["objective"],
        mock_only=creation["mock_only"],
        status=status,
        checkpoint=checkpoint,
        completed_action_ids=tuple(sorted(completed)),
        pending_action_ids=tuple(sorted(pending)),
        ambiguous_action_ids=tuple(sorted(ambiguous)),
        representative_evidence_action_ids=tuple(sorted(representative)),
        discovery_evidence_action_ids=tuple(sorted(discovery)),
        controller_decisions=tuple(decisions),
        provider_responses=tuple(provider_responses),
        receipts=tuple(receipts),
        budget=budget,
        termination_reason=termination,
        last_event_hash=events[-1].event_hash,
    )
