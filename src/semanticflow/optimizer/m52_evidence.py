from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidenceTrack(StrEnum):
    DISCOVERY = "discovery"
    CERTIFICATION = "certification"


class QualificationState(StrEnum):
    QUALIFIED = "QUALIFIED"
    REJECTED = "REJECTED"
    INSUFFICIENT_SUPPORT = "INSUFFICIENT_SUPPORT"
    UNCERTAIN = "UNCERTAIN"
    UNSUPPORTED = "UNSUPPORTED"
    PROVIDER_FAILURE = "PROVIDER_FAILURE"


class PairedQualityMeasurement(BaseModel):
    """One acquired complete-query comparison.

    Discovery measurements are retained for diagnostics but are never included in the
    unweighted representative qualification counts.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    measurement_id: str
    record_id: str
    dataset: str
    workload_id: str
    plan_id: str
    track: EvidenceTrack
    acquisition_method: str
    source_partition: str
    acquisition_checkpoint: int = Field(ge=0)
    reference_decision: bool | None
    candidate_decision: bool | None
    reference_probability_min_margin: float | None = Field(default=None, ge=0, le=0.5)
    inclusion_probability: float | None = Field(default=None, gt=0, le=1)
    request_ids: tuple[str, ...]
    physical_requests: int = Field(ge=0)
    named_question_evaluations: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    token_priced_cost_usd: float = Field(ge=0)
    provider_failure: bool = False
    complete_plan_evidence: bool = True

    @model_validator(mode="after")
    def decisions_match_status(self) -> PairedQualityMeasurement:
        missing = self.reference_decision is None or self.candidate_decision is None
        if self.provider_failure and not missing:
            raise ValueError("provider failures cannot contain a complete paired outcome")
        if not self.provider_failure and self.complete_plan_evidence and missing:
            raise ValueError("complete paired evidence requires both query decisions")
        return self


class EvidenceCounts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    observations: int = Field(ge=0)
    reference_positives: int = Field(ge=0)
    reference_negatives: int = Field(ge=0)
    true_positive: int = Field(ge=0)
    true_negative: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    false_negative: int = Field(ge=0)


class QualityEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "m5.2-quality-evidence-v1"
    plan_id: str
    plan_fingerprint: str
    dataset: str
    dataset_fingerprint: str
    workload_id: str
    workload_fingerprint: str
    physical_groups: tuple[tuple[str, ...], ...]
    reference_method: Literal["separate_jev", "human_labels"] = "separate_jev"
    measurements: tuple[PairedQualityMeasurement, ...] = ()
    evidence_provenance: tuple[str, ...] = ()
    complete_physical_plan_evidence: bool = True

    @model_validator(mode="after")
    def unique_and_isolated(self) -> QualityEvidence:
        ids = [row.measurement_id for row in self.measurements]
        if len(ids) != len(set(ids)):
            raise ValueError("quality measurement IDs must be unique")
        discovery = {
            row.record_id for row in self.measurements if row.track == EvidenceTrack.DISCOVERY
        }
        certification = {
            row.record_id for row in self.measurements if row.track == EvidenceTrack.CERTIFICATION
        }
        if discovery & certification:
            raise ValueError("discovery and certification record IDs must be disjoint")
        if any(row.plan_id != self.plan_id for row in self.measurements):
            raise ValueError("measurement plan IDs must match their evidence container")
        return self

    def for_track(self, track: EvidenceTrack) -> tuple[PairedQualityMeasurement, ...]:
        return tuple(
            row
            for row in self.measurements
            if row.track == track
            and not row.provider_failure
            and row.complete_plan_evidence
            and row.reference_decision is not None
            and row.candidate_decision is not None
        )

    def counts(self, track: EvidenceTrack = EvidenceTrack.CERTIFICATION) -> EvidenceCounts:
        rows = self.for_track(track)
        tp = sum(row.reference_decision is True and row.candidate_decision is True for row in rows)
        tn = sum(
            row.reference_decision is False and row.candidate_decision is False for row in rows
        )
        fp = sum(row.reference_decision is False and row.candidate_decision is True for row in rows)
        fn = sum(row.reference_decision is True and row.candidate_decision is False for row in rows)
        return EvidenceCounts(
            observations=len(rows),
            reference_positives=tp + fn,
            reference_negatives=tn + fp,
            true_positive=tp,
            true_negative=tn,
            false_positive=fp,
            false_negative=fn,
        )

    @property
    def measurement_cost_usd(self) -> float:
        return sum(row.token_priced_cost_usd for row in self.measurements)

    def append(self, measurement: PairedQualityMeasurement) -> QualityEvidence:
        return self.model_copy(update={"measurements": (*self.measurements, measurement)})


class ProportionBounds(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    estimate: float | None = Field(default=None, ge=0, le=1)
    wilson_lower: float | None = Field(default=None, ge=0, le=1)
    wilson_upper: float | None = Field(default=None, ge=0, le=1)
    sequential_lower: float | None = Field(default=None, ge=0, le=1)
    sequential_upper: float | None = Field(default=None, ge=0, le=1)
    successes: int = Field(ge=0)
    trials: int = Field(ge=0)


class QualityEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "m5.2-quality-estimate-v1"
    plan_id: str
    checkpoint: int = Field(ge=0)
    state: QualificationState
    reasons: tuple[str, ...]
    counts: EvidenceCounts
    agreement: ProportionBounds
    recall: ProportionBounds
    false_negative_rate: ProportionBounds
    changed_decision_rate: ProportionBounds
    family_alpha: float = Field(gt=0, lt=1)
    allocated_one_sided_alpha: float = Field(gt=0, lt=1)
    maximum_checkpoints: int = Field(ge=1)
    family_candidate_count: int = Field(ge=1)
    complete_physical_plan_evidence: bool
    representative_evidence_only: bool = True
    semantic_accuracy_established: bool = False
