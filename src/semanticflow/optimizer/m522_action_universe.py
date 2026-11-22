from __future__ import annotations

import hashlib
import json
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from semanticflow.optimizer.m52_evidence import EvidenceTrack


class LiveMeasurementType(StrEnum):
    REPRESENTATIVE_REFERENCE = "representative_reference"
    REPRESENTATIVE_CANDIDATE = "representative_candidate"
    DISCOVERY_SCREEN = "discovery_screen"
    DISCOVERY_FOLLOWUP = "discovery_followup"


class PhysicalOperatorType(StrEnum):
    SINGLE = "single"
    MULTI_DECISION = "multi_decision"


class AuthorizedLiveAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action_id: str
    workload_id: str
    dataset: str
    source_partition: str
    record_id: str
    record_text_sha256: str
    measurement_type: LiveMeasurementType
    physical_operator: PhysicalOperatorType
    track: EvidenceTrack
    predicate_ids: tuple[str, ...]
    predicate_instructions: tuple[str, ...]
    threshold: float = Field(ge=0, le=1)
    requested_model: str
    prerequisite_action_ids: tuple[str, ...] = ()
    maximum_attempts: int = Field(default=3, ge=1)
    predicted_input_tokens: int = Field(ge=0)
    predicted_output_tokens: int = Field(ge=0)
    predicted_cost_usd: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_action(self) -> AuthorizedLiveAction:
        if len(self.predicate_ids) != len(self.predicate_instructions):
            raise ValueError("predicate IDs and instructions must align")
        if len(set(self.predicate_ids)) != len(self.predicate_ids):
            raise ValueError("predicate IDs must be unique")
        if self.physical_operator == PhysicalOperatorType.SINGLE and len(self.predicate_ids) != 1:
            raise ValueError("single operator requires exactly one predicate")
        if (
            self.physical_operator == PhysicalOperatorType.MULTI_DECISION
            and len(self.predicate_ids) < 2
        ):
            raise ValueError("multi-decision operator requires at least two predicates")
        return self

    @property
    def named_question_evaluations(self) -> int:
        return len(self.predicate_ids)

    @property
    def content_fingerprint(self) -> str:
        payload = self.model_dump(mode="json", exclude={"action_id"})
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class AuthorizedActionUniverse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "m5.2.2-authorized-action-universe-v1"
    experiment_id: str
    actions: tuple[AuthorizedLiveAction, ...]
    maximum_actions_per_record: int = Field(ge=1)
    universe_fingerprint: str

    @model_validator(mode="after")
    def validate_universe(self) -> AuthorizedActionUniverse:
        ids = [action.action_id for action in self.actions]
        if len(ids) != len(set(ids)):
            raise ValueError("action identities must be unique")
        known = set(ids)
        if any(set(action.prerequisite_action_ids) - known for action in self.actions):
            raise ValueError("action prerequisite lies outside authorized universe")
        counts: dict[tuple[str, str], int] = {}
        for action in self.actions:
            key = (action.workload_id, action.record_id)
            counts[key] = counts.get(key, 0) + 1
        if counts and max(counts.values()) > self.maximum_actions_per_record:
            raise ValueError("record exceeds maximum authorized actions")
        expected = universe_fingerprint(self.experiment_id, self.actions)
        if self.universe_fingerprint != expected:
            raise ValueError("action-universe fingerprint mismatch")
        return self

    def action(self, action_id: str) -> AuthorizedLiveAction:
        for action in self.actions:
            if action.action_id == action_id:
                return action
        raise PermissionError(f"action {action_id} is outside the authorized universe")


def action_identity(payload: dict[str, object]) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"m522action_{digest[:24]}"


def universe_fingerprint(
    experiment_id: str, actions: tuple[AuthorizedLiveAction, ...] | list[AuthorizedLiveAction]
) -> str:
    payload = {
        "experiment_id": experiment_id,
        "actions": [action.model_dump(mode="json") for action in actions],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
