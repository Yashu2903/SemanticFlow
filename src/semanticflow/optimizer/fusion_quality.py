from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BinaryQualityMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_count: int = Field(ge=0)
    reference_positive_count: int = Field(ge=0)
    candidate_positive_count: int = Field(ge=0)
    true_positive: int = Field(ge=0)
    true_negative: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    false_negative: int = Field(ge=0)
    agreement: float | None = Field(default=None, ge=0.0, le=1.0)
    precision: float | None = Field(default=None, ge=0.0, le=1.0)
    recall: float | None = Field(default=None, ge=0.0, le=1.0)
    f1: float | None = Field(default=None, ge=0.0, le=1.0)
    matthews_correlation: float | None = Field(default=None, ge=-1.0, le=1.0)
    false_positive_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    false_negative_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    changed_decisions: int = Field(ge=0)
    jaccard: float | None = Field(default=None, ge=0.0, le=1.0)
    reference_kind: Literal["separate_llm", "human_labels"]
    semantic_accuracy_established: bool


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def compare_binary_decisions(
    reference: Sequence[bool],
    candidate: Sequence[bool],
    *,
    reference_kind: Literal["separate_llm", "human_labels"] = "separate_llm",
) -> BinaryQualityMetrics:
    if len(reference) != len(candidate):
        raise ValueError("reference and candidate decisions must align")
    tp = sum(ref and cand for ref, cand in zip(reference, candidate, strict=True))
    tn = sum(not ref and not cand for ref, cand in zip(reference, candidate, strict=True))
    fp = sum(not ref and cand for ref, cand in zip(reference, candidate, strict=True))
    fn = sum(ref and not cand for ref, cand in zip(reference, candidate, strict=True))
    count = len(reference)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denominator) if denominator else None
    union = tp + fp + fn
    jaccard = tp / union if union else (1.0 if count else None)
    return BinaryQualityMetrics(
        sample_count=count,
        reference_positive_count=tp + fn,
        candidate_positive_count=tp + fp,
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
        agreement=_ratio(tp + tn, count),
        precision=precision,
        recall=recall,
        f1=f1,
        matthews_correlation=mcc,
        false_positive_rate=_ratio(fp, fp + tn),
        false_negative_rate=_ratio(fn, fn + tp),
        changed_decisions=fp + fn,
        jaccard=jaccard,
        reference_kind=reference_kind,
        semantic_accuracy_established=reference_kind == "human_labels",
    )


class FusionQualityGate(BaseModel):
    """Preregistered eligibility gate; absent or sparse evidence fails closed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_records: int = Field(default=40, ge=1)
    minimum_reference_positives: int = Field(default=3, ge=0)
    minimum_agreement: float = Field(default=0.98, ge=0.0, le=1.0)
    minimum_recall: float = Field(default=0.95, ge=0.0, le=1.0)
    maximum_false_negative_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    maximum_changed_decision_rate: float = Field(default=0.02, ge=0.0, le=1.0)
    require_real_fused_measurement: bool = True
    require_complete_plan_measurement: bool = True


class QualityGateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    eligible: bool
    reasons: tuple[str, ...]
    evidence_scope: Literal["group", "complete_plan", "missing"]
    reference_kind: str | None
    semantic_accuracy_established: bool


def evaluate_quality_gate(
    metrics: BinaryQualityMetrics | None,
    gate: FusionQualityGate,
    *,
    evidence_scope: Literal["group", "complete_plan", "missing"],
    real_fused_measurement: bool,
) -> QualityGateDecision:
    reasons: list[str] = []
    if metrics is None:
        reasons.append("missing_quality_evidence")
    else:
        if gate.require_real_fused_measurement and not real_fused_measurement:
            reasons.append("quality_evidence_is_not_real_fused_inference")
        if gate.require_complete_plan_measurement and evidence_scope != "complete_plan":
            reasons.append("complete_plan_quality_not_measured")
        if metrics.sample_count < gate.minimum_records:
            reasons.append("insufficient_quality_records")
        if metrics.reference_positive_count < gate.minimum_reference_positives:
            reasons.append("insufficient_reference_positives")
        if metrics.agreement is None or metrics.agreement < gate.minimum_agreement:
            reasons.append("agreement_below_threshold_or_undefined")
        if metrics.recall is None or metrics.recall < gate.minimum_recall:
            reasons.append("recall_below_threshold_or_undefined")
        if (
            metrics.false_negative_rate is None
            or metrics.false_negative_rate > gate.maximum_false_negative_rate
        ):
            reasons.append("false_negative_rate_above_threshold_or_undefined")
        changed_rate = _ratio(metrics.changed_decisions, metrics.sample_count)
        if changed_rate is None or changed_rate > gate.maximum_changed_decision_rate:
            reasons.append("changed_decision_rate_above_threshold_or_undefined")
    return QualityGateDecision(
        eligible=not reasons,
        reasons=tuple(reasons),
        evidence_scope=evidence_scope,
        reference_kind=metrics.reference_kind if metrics else None,
        semantic_accuracy_established=(metrics.semantic_accuracy_established if metrics else False),
    )
