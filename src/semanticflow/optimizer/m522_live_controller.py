from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from semanticflow.optimizer.m522_action_universe import (
    AuthorizedActionUniverse,
    AuthorizedLiveAction,
    LiveMeasurementType,
)
from semanticflow.optimizer.m522_budget import BudgetLedger, BudgetSnapshot
from semanticflow.optimizer.m522_experiment_objective import ExperimentObjective
from semanticflow.optimizer.m522_live_acquisition import (
    AmbiguousDispatchError,
    LiveAcquisitionAdapter,
)
from semanticflow.optimizer.m522_protocol import PilotAProtocol, policy_fingerprint
from semanticflow.optimizer.m522_session import (
    AppendOnlyEventLog,
    PersistentSessionState,
    SessionStatus,
    rebuild_session,
)


class InstrumentationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "m5.2.2-instrumentation-result-v1"
    session_id: str
    status: SessionStatus
    formal_quality_certified: bool = False
    production_deployment_permitted: bool = False
    qualification_feasible: bool = False
    representative_actions: int = Field(ge=0)
    discovery_screens: int = Field(ge=0)
    discovery_followups: int = Field(ge=0)
    skipped_followups: int = Field(ge=0)
    response_influenced_actions: int = Field(ge=0)
    budget: BudgetSnapshot
    event_count: int = Field(ge=0)
    termination_reason: str


def _evidence_fingerprint(receipts: tuple[dict[str, Any], ...]) -> str:
    return hashlib.sha256(
        json.dumps(receipts, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _receipt_by_action(state: PersistentSessionState) -> dict[str, dict[str, Any]]:
    return {str(receipt["action_id"]): receipt for receipt in state.receipts}


class LiveProfilingController:
    def __init__(
        self,
        *,
        protocol: PilotAProtocol,
        universe: AuthorizedActionUniverse,
        event_log: AppendOnlyEventLog,
        adapter: LiveAcquisitionAdapter,
        record_texts: dict[str, str],
        authorization_scope_fingerprint: str,
        session_id: str,
        mock_only: bool,
    ) -> None:
        if protocol.objective.objective != ExperimentObjective.RESEARCH_INSTRUMENTATION:
            raise ValueError("live instrumentation controller requires research objective")
        self.protocol = protocol
        self.universe = universe
        self.event_log = event_log
        self.adapter = adapter
        self.record_texts = record_texts
        self.authorization_scope_fingerprint = authorization_scope_fingerprint
        self.session_id = session_id
        self.mock_only = mock_only

    def initialize(self) -> PersistentSessionState:
        events = self.event_log.read()
        if events:
            state = rebuild_session(events)
            self._validate_scope(state)
            return state
        ledger = BudgetLedger(
            physical_request_ceiling=self.protocol.physical_request_ceiling,
            http_attempt_ceiling=self.protocol.http_attempt_ceiling,
            maximum_retries=self.protocol.maximum_retries,
        )
        self.event_log.append(
            "session_created",
            {
                "session_id": self.session_id,
                "protocol_fingerprint": self.protocol.protocol_fingerprint,
                "authorization_scope_fingerprint": self.authorization_scope_fingerprint,
                "policy_fingerprint": policy_fingerprint(self.protocol),
                "action_universe_fingerprint": self.universe.universe_fingerprint,
                "initial_seed": self.protocol.seed,
                "objective": self.protocol.objective.objective.value,
                "mock_only": self.mock_only,
                "budget": ledger.snapshot().model_dump(mode="json"),
            },
        )
        self.event_log.append("status_changed", {"status": SessionStatus.VALIDATED.value})
        self.event_log.append("status_changed", {"status": SessionStatus.AUTHORIZED.value})
        return rebuild_session(self.event_log.read())

    def _validate_scope(self, state: PersistentSessionState) -> None:
        expected = {
            "protocol": self.protocol.protocol_fingerprint,
            "authorization": self.authorization_scope_fingerprint,
            "policy": policy_fingerprint(self.protocol),
            "universe": self.universe.universe_fingerprint,
            "objective": self.protocol.objective.objective.value,
            "mock_only": self.mock_only,
        }
        actual = {
            "protocol": state.protocol_fingerprint,
            "authorization": state.authorization_scope_fingerprint,
            "policy": state.policy_fingerprint,
            "universe": state.action_universe_fingerprint,
            "objective": state.objective,
            "mock_only": state.mock_only,
        }
        if actual != expected:
            raise PermissionError("session restart scope differs from frozen authorization")

    def _ledger(self, state: PersistentSessionState) -> BudgetLedger:
        return BudgetLedger(
            physical_request_ceiling=self.protocol.physical_request_ceiling,
            http_attempt_ceiling=self.protocol.http_attempt_ceiling,
            maximum_retries=self.protocol.maximum_retries,
            snapshot=state.budget,
        )

    def interrupt(self, reason: str = "operator_interruption") -> PersistentSessionState:
        state = self.initialize()
        if state.pending_action_ids:
            raise PermissionError("cannot mark a session safely interrupted with a pending action")
        self.event_log.append("interrupted", {"reason": reason})
        return rebuild_session(self.event_log.read())

    def abort(self, reason: str) -> PersistentSessionState:
        state = self.initialize()
        if state.pending_action_ids:
            raise PermissionError("pending action requires reconciliation before abort")
        self.event_log.append(
            "status_changed", {"status": SessionStatus.ABORTED.value, "reason": reason}
        )
        return rebuild_session(self.event_log.read())

    def _screen_is_informative(self, receipt: dict[str, Any]) -> bool:
        probability = next(iter(receipt["probabilities"].values()))
        return (
            abs(float(probability) - self.protocol.threshold) <= self.protocol.near_threshold_margin
        )

    def choose_next(self, state: PersistentSessionState) -> tuple[AuthorizedLiveAction | None, str]:
        completed = set(state.completed_action_ids)
        receipts = _receipt_by_action(state)
        eligible: list[tuple[float, AuthorizedLiveAction, str]] = []
        skipped_followups = 0
        for action in self.universe.actions:
            if action.action_id in completed or action.action_id in state.pending_action_ids:
                continue
            if action.action_id in state.ambiguous_action_ids:
                continue
            if not set(action.prerequisite_action_ids) <= completed:
                continue
            if action.measurement_type == LiveMeasurementType.DISCOVERY_FOLLOWUP:
                screen_receipt = receipts[action.prerequisite_action_ids[0]]
                if not self._screen_is_informative(screen_receipt):
                    skipped_followups += 1
                    continue
                score, reason = 95.0, "informative_paid_screen_triggered_followup"
            elif action.measurement_type == LiveMeasurementType.REPRESENTATIVE_REFERENCE:
                score, reason = 100.0, "required_representative_reference"
            elif action.measurement_type == LiveMeasurementType.REPRESENTATIVE_CANDIDATE:
                score, reason = 99.0, "reference_complete_candidate_pair"
            else:
                score, reason = 90.0, "next_paid_discovery_screen"
            score -= action.predicted_cost_usd * 1000
            eligible.append((score, action, reason))
        if eligible:
            eligible.sort(key=lambda row: (-row[0], row[1].action_id))
            _, action, reason = eligible[0]
            return action, reason
        screens = [
            action
            for action in self.universe.actions
            if action.measurement_type == LiveMeasurementType.DISCOVERY_SCREEN
        ]
        representatives = [
            action
            for action in self.universe.actions
            if action.measurement_type
            in {
                LiveMeasurementType.REPRESENTATIVE_REFERENCE,
                LiveMeasurementType.REPRESENTATIVE_CANDIDATE,
            }
        ]
        if all(action.action_id in completed for action in (*screens, *representatives)):
            return (
                None,
                f"instrumentation_objectives_complete_skipped_followups={skipped_followups}",
            )
        return None, "no_eligible_authorized_action"

    async def step(self) -> PersistentSessionState:
        state = self.initialize()
        if state.status in {
            SessionStatus.RECONCILIATION_REQUIRED,
            SessionStatus.FAILED_CLOSED,
            SessionStatus.INSTRUMENTATION_COMPLETE,
            SessionStatus.ABORTED,
        }:
            return state
        if state.pending_action_ids:
            self.event_log.append(
                "status_changed",
                {
                    "status": SessionStatus.RECONCILIATION_REQUIRED.value,
                    "reason": "pending_action_requires_durable_reconciliation",
                },
            )
            return rebuild_session(self.event_log.read())
        if state.status != SessionStatus.EXECUTING:
            self.event_log.append("status_changed", {"status": SessionStatus.EXECUTING.value})
            state = rebuild_session(self.event_log.read())
        action, rationale = self.choose_next(state)
        if action is None:
            status = (
                SessionStatus.INSTRUMENTATION_COMPLETE
                if rationale.startswith("instrumentation_objectives_complete")
                else SessionStatus.FAILED_CLOSED
            )
            self.event_log.append("status_changed", {"status": status.value, "reason": rationale})
            return rebuild_session(self.event_log.read())
        evidence_hash = _evidence_fingerprint(state.receipts)
        self.event_log.append(
            "action_selected",
            {
                "action_id": action.action_id,
                "workload_id": action.workload_id,
                "record_id": action.record_id,
                "measurement_type": action.measurement_type.value,
                "physical_operator": action.physical_operator.value,
                "required_prior_evidence": list(action.prerequisite_action_ids),
                "available_evidence_fingerprint": evidence_hash,
                "policy_version": self.protocol.policy_version,
                "selection_rationale": rationale,
                "predicted_acquisition_cost_usd": action.predicted_cost_usd,
                "budget_before": state.budget.model_dump(mode="json"),
            },
        )
        ledger = self._ledger(state)
        try:
            snapshot = ledger.reserve_action(
                action.action_id,
                predicted_input_tokens=action.predicted_input_tokens,
                predicted_cost_usd=action.predicted_cost_usd,
            )
        except PermissionError as exc:
            self.event_log.append(
                "status_changed",
                {
                    "status": SessionStatus.INSTRUMENTATION_COMPLETE.value,
                    "reason": f"budget_exhausted:{exc}",
                },
            )
            return rebuild_session(self.event_log.read())
        self.event_log.append(
            "action_reserved",
            {"action_id": action.action_id, "budget": snapshot.model_dump(mode="json")},
        )
        self.event_log.append("dispatch_started", {"action_id": action.action_id})
        try:
            receipt = await self.adapter.acquire(action, self.record_texts[action.record_id])
        except AmbiguousDispatchError as exc:
            snapshot = ledger.mark_ambiguous(action.action_id, attempts=exc.attempts)
            self.event_log.append(
                "ambiguous_receipt",
                {
                    "action_id": action.action_id,
                    "attempts": exc.attempts,
                    "budget": snapshot.model_dump(mode="json"),
                    "reason": str(exc),
                },
            )
            return rebuild_session(self.event_log.read())
        except Exception as exc:
            snapshot = ledger.charge_receipt(
                action.action_id,
                attempts=max(1, int(getattr(exc, "attempts", 1))),
                named_question_evaluations=0,
                input_tokens=0,
                output_tokens=0,
                cost_proxy_usd=0,
            )
            self.event_log.append(
                "action_failed",
                {
                    "action_id": action.action_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "budget": snapshot.model_dump(mode="json"),
                },
            )
            self.event_log.append(
                "status_changed",
                {"status": SessionStatus.FAILED_CLOSED.value, "reason": "provider_failure"},
            )
            return rebuild_session(self.event_log.read())
        self.event_log.append("receipt_persisted", {"receipt": receipt.model_dump(mode="json")})
        snapshot = ledger.charge_receipt(
            action.action_id,
            attempts=receipt.http_attempts,
            named_question_evaluations=len(receipt.expected_question_ids),
            input_tokens=receipt.input_tokens,
            output_tokens=receipt.output_tokens,
            cost_proxy_usd=receipt.token_priced_cost_usd,
        )
        self.event_log.append(
            "evidence_accepted",
            {
                "action_id": action.action_id,
                "track": action.track.value,
                "response_identity": receipt.response_identity,
                "budget": snapshot.model_dump(mode="json"),
            },
        )
        return rebuild_session(self.event_log.read())

    async def run(self, *, maximum_steps: int | None = None) -> InstrumentationResult:
        steps = 0
        while maximum_steps is None or steps < maximum_steps:
            state = await self.step()
            steps += 1
            if state.status in {
                SessionStatus.RECONCILIATION_REQUIRED,
                SessionStatus.FAILED_CLOSED,
                SessionStatus.INSTRUMENTATION_COMPLETE,
                SessionStatus.ABORTED,
            }:
                break
        state = rebuild_session(self.event_log.read())
        receipts = _receipt_by_action(state)
        screens = [
            receipt
            for receipt in receipts.values()
            if receipt["measurement_type"] == LiveMeasurementType.DISCOVERY_SCREEN.value
        ]
        followups = sum(
            receipt["measurement_type"] == LiveMeasurementType.DISCOVERY_FOLLOWUP.value
            for receipt in receipts.values()
        )
        informative = sum(self._screen_is_informative(receipt) for receipt in screens)
        representative = sum(receipt["track"] == "certification" for receipt in receipts.values())
        return InstrumentationResult(
            session_id=state.session_id,
            status=state.status,
            representative_actions=representative,
            discovery_screens=len(screens),
            discovery_followups=followups,
            skipped_followups=len(screens) - informative,
            response_influenced_actions=followups + len(screens) - informative,
            budget=state.budget,
            event_count=len(self.event_log.read()),
            termination_reason=state.termination_reason or state.status.value,
        )
