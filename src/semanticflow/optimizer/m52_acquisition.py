from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from semanticflow.optimizer.m52_evidence import EvidenceTrack


class ActionKind(StrEnum):
    ACQUIRE_REPRESENTATIVE_PAIR = "acquire_representative_pair"
    SCREEN_DISCOVERY_REFERENCE = "screen_discovery_reference"
    ACQUIRE_DISCOVERY_CANDIDATE = "acquire_discovery_candidate"
    STOP = "stop"


class ProfilingBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    maximum_physical_requests: int = Field(default=400, ge=0)
    maximum_cost_usd: float = Field(default=0.01, ge=0)
    maximum_records: int = Field(default=200, ge=0)
    maximum_actions: int = Field(default=200, ge=0)
    maximum_certification_records: int = Field(default=160, ge=0)
    maximum_discovery_records: int = Field(default=40, ge=0)
    maximum_positive_enrichment_records: int = Field(default=20, ge=0)


class PredictedAcquisitionCost(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prediction_id: str
    predicted_physical_requests: float = Field(ge=0)
    predicted_input_tokens: float = Field(ge=0)
    predicted_output_tokens: float = Field(ge=0)
    predicted_cost_usd: float = Field(ge=0)
    estimator_version: str
    input_measurement_ids: tuple[str, ...]
    available_partition: str
    created_checkpoint: int = Field(ge=0)


class ProfilingAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action_id: str
    kind: ActionKind
    record_id: str | None = None
    plan_id: str | None = None
    track: EvidenceTrack | None = None
    acquisition_method: str
    prerequisite_measurement_ids: tuple[str, ...] = ()
    predicted_cost: PredictedAcquisitionCost | None = None
    score: float | None = None
    score_components: dict[str, float | str | bool | None] = {}

    @model_validator(mode="after")
    def validate_shape(self) -> ProfilingAction:
        if self.kind == ActionKind.STOP:
            if self.record_id is not None or self.plan_id is not None:
                raise ValueError("stop actions cannot target records or plans")
        elif self.record_id is None or self.plan_id is None or self.track is None:
            raise ValueError("acquisition actions require a record, plan, and track")
        return self


class AcquisitionReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action_id: str
    acquired_measurement_ids: tuple[str, ...]
    reused_measurement_ids: tuple[str, ...]
    physical_requests: int = Field(ge=0)
    named_question_evaluations: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    token_priced_cost_usd: float = Field(ge=0)
    provider_failure: bool = False
    unsupported_reason: str | None = None


class BudgetUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    physical_requests: int = Field(default=0, ge=0)
    named_question_evaluations: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    token_priced_cost_usd: float = Field(default=0, ge=0)
    records: int = Field(default=0, ge=0)
    actions: int = Field(default=0, ge=0)
    certification_records: int = Field(default=0, ge=0)
    discovery_records: int = Field(default=0, ge=0)
    positive_enrichment_records: int = Field(default=0, ge=0)

    def permits(self, budget: ProfilingBudget, predicted: PredictedAcquisitionCost) -> bool:
        return (
            self.physical_requests + math_ceil(predicted.predicted_physical_requests)
            <= budget.maximum_physical_requests
            and self.token_priced_cost_usd + predicted.predicted_cost_usd <= budget.maximum_cost_usd
            and self.actions + 1 <= budget.maximum_actions
        )


def math_ceil(value: float) -> int:
    integer = int(value)
    return integer if value == integer else integer + 1
