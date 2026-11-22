from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from semanticflow.optimizer.m522_action_universe import AuthorizedActionUniverse
from semanticflow.optimizer.m522_budget import BudgetLedger
from semanticflow.optimizer.m522_session import (
    AppendOnlyEventLog,
    SessionStatus,
    rebuild_session,
)


class ReconciliationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "m5.2.2-reconciliation-result-v1"
    valid: bool
    completed_actions: int
    pending_actions: int
    ambiguous_actions: int
    recovered_receipts: int
    released_undispatched_reservations: int
    duplicate_receipts: int
    errors: tuple[str, ...]
    next_action_may_execute: bool


def reconcile_session(
    *,
    event_log: AppendOnlyEventLog,
    universe: AuthorizedActionUniverse,
    expected_protocol_fingerprint: str,
    expected_authorization_scope_fingerprint: str,
    expected_policy_fingerprint: str,
    physical_request_ceiling: int,
    http_attempt_ceiling: int,
    maximum_retries: int,
    repair: bool = False,
) -> ReconciliationResult:
    events = event_log.read()
    state = rebuild_session(events)
    errors: list[str] = []
    if state.protocol_fingerprint != expected_protocol_fingerprint:
        errors.append("protocol_fingerprint_mismatch")
    if state.authorization_scope_fingerprint != expected_authorization_scope_fingerprint:
        errors.append("authorization_scope_fingerprint_mismatch")
    if state.policy_fingerprint != expected_policy_fingerprint:
        errors.append("policy_fingerprint_mismatch")
    known = {action.action_id for action in universe.actions}
    if set(state.completed_action_ids) - known or set(state.pending_action_ids) - known:
        errors.append("session_action_outside_authorized_universe")
    receipts: dict[str, list[dict[str, object]]] = {}
    dispatched: set[str] = set()
    for event in events:
        if event.event_type == "receipt_persisted":
            receipt = event.payload["receipt"]
            receipts.setdefault(str(receipt["action_id"]), []).append(receipt)
        elif event.event_type == "dispatch_started":
            dispatched.add(str(event.payload["action_id"]))
    duplicate_count = sum(max(0, len(values) - 1) for values in receipts.values())
    if duplicate_count:
        errors.append("duplicate_persisted_receipt")
    ledger = BudgetLedger(
        physical_request_ceiling=physical_request_ceiling,
        http_attempt_ceiling=http_attempt_ceiling,
        maximum_retries=maximum_retries,
        snapshot=state.budget,
    )
    recovered = 0
    released = 0
    newly_ambiguous = 0
    if repair and not errors:
        for action_id in state.pending_action_ids:
            values = receipts.get(action_id, [])
            if len(values) == 1:
                receipt = values[0]
                snapshot = ledger.charge_receipt(
                    action_id,
                    attempts=int(receipt["http_attempts"]),
                    named_question_evaluations=len(receipt["expected_question_ids"]),
                    input_tokens=int(receipt["input_tokens"]),
                    output_tokens=int(receipt["output_tokens"]),
                    cost_proxy_usd=float(receipt["token_priced_cost_usd"]),
                )
                action = universe.action(action_id)
                event_log.append(
                    "evidence_accepted",
                    {
                        "action_id": action_id,
                        "track": action.track.value,
                        "response_identity": receipt["response_identity"],
                        "budget": snapshot.model_dump(mode="json"),
                        "recovered_after_restart": True,
                    },
                )
                recovered += 1
            elif action_id not in dispatched:
                action = universe.action(action_id)
                snapshot = ledger.release_unattempted_reservation(
                    action_id,
                    predicted_input_tokens=action.predicted_input_tokens,
                    predicted_cost_usd=action.predicted_cost_usd,
                )
                event_log.append(
                    "reservation_released",
                    {
                        "action_id": action_id,
                        "reason": "durable_log_proves_dispatch_never_started",
                        "budget": snapshot.model_dump(mode="json"),
                    },
                )
                released += 1
            else:
                snapshot = ledger.mark_ambiguous(action_id, attempts=1)
                event_log.append(
                    "ambiguous_receipt",
                    {
                        "action_id": action_id,
                        "attempts": 1,
                        "budget": snapshot.model_dump(mode="json"),
                        "reason": "dispatch recorded without durable receipt",
                    },
                )
                newly_ambiguous += 1
        if (recovered or released) and not newly_ambiguous:
            event_log.append(
                "status_changed",
                {"status": SessionStatus.AUTHORIZED.value, "reason": "reconciled"},
            )
        state = rebuild_session(event_log.read())
    ambiguous = len(state.ambiguous_action_ids)
    valid = not errors and ambiguous == 0 and not state.pending_action_ids
    return ReconciliationResult(
        valid=valid,
        completed_actions=len(state.completed_action_ids),
        pending_actions=len(state.pending_action_ids),
        ambiguous_actions=ambiguous,
        recovered_receipts=recovered,
        released_undispatched_reservations=released,
        duplicate_receipts=duplicate_count,
        errors=tuple(errors),
        next_action_may_execute=valid,
    )
