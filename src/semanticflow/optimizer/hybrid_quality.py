from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from semanticflow.optimizer.fusion_quality import (
    BinaryQualityMetrics,
    FusionQualityGate,
    compare_binary_decisions,
    evaluate_quality_gate,
)

QualityEvidenceStatus = Literal[
    "sufficient_evidence_passed",
    "sufficient_evidence_failed",
    "insufficient_positive_examples",
    "insufficient_total_observations",
    "undefined_metric",
    "missing_group_evidence",
    "missing_complete_plan_evidence",
    "provider_execution_failure",
]


class ClassifiedQualityEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: QualityEvidenceStatus
    eligible: bool
    reasons: tuple[str, ...]
    metrics: BinaryQualityMetrics | None
    population_representative: bool
    enriched_sample: bool


def classify_quality_evidence(
    metrics: BinaryQualityMetrics | None,
    gate: FusionQualityGate,
    *,
    evidence_scope: Literal["group", "complete_plan", "missing"],
    real_provider_measurement: bool,
    provider_failed: bool = False,
    population_representative: bool = True,
    enriched_sample: bool = False,
) -> ClassifiedQualityEvidence:
    if provider_failed:
        return ClassifiedQualityEvidence(
            status="provider_execution_failure",
            eligible=False,
            reasons=("provider_execution_failure",),
            metrics=metrics,
            population_representative=population_representative,
            enriched_sample=enriched_sample,
        )
    if metrics is None:
        status: QualityEvidenceStatus = (
            "missing_group_evidence"
            if evidence_scope != "complete_plan"
            else "missing_complete_plan_evidence"
        )
        return ClassifiedQualityEvidence(
            status=status,
            eligible=False,
            reasons=(status,),
            metrics=None,
            population_representative=population_representative,
            enriched_sample=enriched_sample,
        )
    if evidence_scope != "complete_plan" and gate.require_complete_plan_measurement:
        status = "missing_complete_plan_evidence"
    elif metrics.sample_count < gate.minimum_records:
        status = "insufficient_total_observations"
    elif metrics.reference_positive_count < gate.minimum_reference_positives:
        status = "insufficient_positive_examples"
    elif metrics.recall is None or metrics.false_negative_rate is None:
        status = "undefined_metric"
    else:
        decision = evaluate_quality_gate(
            metrics,
            gate,
            evidence_scope=evidence_scope,
            real_fused_measurement=real_provider_measurement,
        )
        status = (
            "sufficient_evidence_passed"
            if decision.eligible
            else "sufficient_evidence_failed"
        )
    decision = evaluate_quality_gate(
        metrics,
        gate,
        evidence_scope=evidence_scope,
        real_fused_measurement=real_provider_measurement,
    )
    return ClassifiedQualityEvidence(
        status=status,
        eligible=decision.eligible and status == "sufficient_evidence_passed",
        reasons=decision.reasons,
        metrics=metrics,
        population_representative=population_representative,
        enriched_sample=enriched_sample,
    )


def compare_named_batch_decisions(
    reference: Mapping[str, Sequence[bool]],
    candidate: Mapping[str, Sequence[bool]],
) -> dict[str, BinaryQualityMetrics]:
    if set(reference) != set(candidate):
        raise ValueError("named batch evidence must cover the same predicates")
    return {
        predicate_id: compare_binary_decisions(reference[predicate_id], candidate[predicate_id])
        for predicate_id in reference
    }


class HumanPredicateLabel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str
    record_id: str
    predicate_id: str
    label: Literal["true", "false", "uncertain"]
    annotator_id: str
    guideline_version: str
    annotation_timestamp: str
    rationale: str | None = None
    adjudication_status: Literal["not_required", "pending", "resolved"] = "not_required"
