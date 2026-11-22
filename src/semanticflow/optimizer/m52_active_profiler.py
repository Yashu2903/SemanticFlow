from __future__ import annotations

import hashlib
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from semanticflow.optimizer.m52_acquisition import (
    ActionKind,
    BudgetUsage,
    PredictedAcquisitionCost,
    ProfilingAction,
    ProfilingBudget,
)
from semanticflow.optimizer.m52_cost_model import FrozenCostPrediction
from semanticflow.optimizer.m52_evidence import EvidenceTrack, QualityEvidence
from semanticflow.optimizer.m52_replay import HistoricalReplayMatrix


class SamplingStrategy(StrEnum):
    RANDOM_REPRESENTATIVE = "random_representative"
    POSITIVE_ENRICHMENT = "positive_enrichment"
    THRESHOLD_FOCUSED = "threshold_focused"
    DISAGREEMENT_FOCUSED = "disagreement_focused"
    COST_AWARE = "cost_aware"
    HYBRID_ACTIVE = "hybrid_active"


class AcquisitionContext(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, frozen=True)

    strategy: SamplingStrategy
    workload_id: str
    plan_id: str
    checkpoint: int
    certification_pool: tuple[str, ...]
    discovery_pool: tuple[str, ...]
    evidence: QualityEvidence
    usage: BudgetUsage
    budget: ProfilingBudget
    fallback_prediction: FrozenCostPrediction
    candidate_prediction: FrozenCostPrediction
    evaluation_record_count: int
    reuse_horizon: int


def deterministic_record_order(record_ids: tuple[str, ...], seed: int) -> list[str]:
    return sorted(
        record_ids,
        key=lambda record_id: hashlib.sha256(f"{seed}|{record_id}".encode()).hexdigest(),
    )


def _prediction(
    context: AcquisitionContext,
    *,
    include_reference: bool,
) -> PredictedAcquisitionCost:
    reference = context.fallback_prediction if include_reference else None
    candidate = context.candidate_prediction
    cost = candidate.predicted_cost_usd / context.evaluation_record_count + (
        reference.predicted_cost_usd / context.evaluation_record_count
        if reference is not None
        else 0
    )
    requests = candidate.predicted_requests / context.evaluation_record_count + (
        reference.predicted_requests / context.evaluation_record_count
        if reference is not None
        else 0
    )
    inputs = set(candidate.input_measurement_ids)
    if reference is not None:
        inputs.update(reference.input_measurement_ids)
    identifier = hashlib.sha256(
        f"{context.plan_id}|{context.checkpoint}|{include_reference}|{cost}".encode()
    ).hexdigest()[:24]
    return PredictedAcquisitionCost(
        prediction_id=f"m52acqpred_{identifier}",
        predicted_physical_requests=requests,
        predicted_input_tokens=cost / 0.042 * 1_000_000,
        predicted_output_tokens=0,
        predicted_cost_usd=cost,
        estimator_version=candidate.estimator_version,
        input_measurement_ids=tuple(sorted(inputs)),
        available_partition=candidate.available_data_partition,
        created_checkpoint=context.checkpoint,
    )


def _action(
    context: AcquisitionContext,
    *,
    kind: ActionKind,
    record_id: str,
    track: EvidenceTrack,
    method: str,
    include_reference: bool,
    prerequisites: tuple[str, ...] = (),
    score: float | None = None,
    components: dict[str, float | str | bool | None] | None = None,
) -> ProfilingAction:
    payload = f"{kind}|{record_id}|{context.plan_id}|{context.checkpoint}|{method}"
    predicted = _prediction(context, include_reference=include_reference)
    return ProfilingAction(
        action_id="m52action_" + hashlib.sha256(payload.encode()).hexdigest()[:24],
        kind=kind,
        record_id=record_id,
        plan_id=context.plan_id,
        track=track,
        acquisition_method=method,
        prerequisite_measurement_ids=prerequisites,
        predicted_cost=predicted,
        score=score,
        score_components={"checkpoint": context.checkpoint, **(components or {})},
    )


def _unused(pool: tuple[str, ...], evidence: QualityEvidence) -> list[str]:
    used = {row.record_id for row in evidence.measurements}
    return [record_id for record_id in pool if record_id not in used]


def _representative_action(context: AcquisitionContext, *, seed: int) -> ProfilingAction | None:
    unused = _unused(context.certification_pool, context.evidence)
    if (
        not unused
        or context.usage.certification_records >= context.budget.maximum_certification_records
    ):
        return None
    ordered = deterministic_record_order(tuple(unused), seed)
    return _action(
        context,
        kind=ActionKind.ACQUIRE_REPRESENTATIVE_PAIR,
        record_id=ordered[0],
        track=EvidenceTrack.CERTIFICATION,
        method=SamplingStrategy.RANDOM_REPRESENTATIVE,
        include_reference=True,
        components={"inclusion_probability": 1 / len(unused)},
    )


def _screen_or_discovery_action(
    context: AcquisitionContext,
    matrix: HistoricalReplayMatrix,
    *,
    seed: int,
    mode: SamplingStrategy,
) -> ProfilingAction | None:
    paired = {row.record_id for row in context.evidence.measurements}
    screened = []
    unscreened = []
    for record_id in context.discovery_pool:
        screen = matrix.visible_screen(record_id, context.workload_id)
        if screen is None:
            unscreened.append(record_id)
        elif record_id not in paired:
            screened.append((record_id, screen))
    if screened:
        discovery_rows = context.evidence.for_track(EvidenceTrack.DISCOVERY)
        disagreements = sum(
            row.reference_decision != row.candidate_decision for row in discovery_rows
        )
        smoothed_disagreement = (disagreements + 1) / (len(discovery_rows) + 2)

        def priority(item: tuple[str, object]) -> tuple[float, str]:
            record_id, raw = item
            screen = raw
            decision = bool(screen["decision"])
            margin = float(screen["margin"] if screen["margin"] is not None else 0.5)
            if mode == SamplingStrategy.POSITIVE_ENRICHMENT:
                value = 1.0 if decision else 0.0
            elif mode == SamplingStrategy.THRESHOLD_FOCUSED:
                value = 1 - 2 * margin
            elif mode == SamplingStrategy.DISAGREEMENT_FOCUSED:
                value = smoothed_disagreement + (1 - 2 * margin) + (0.25 if decision else 0)
            else:
                predicted = _prediction(context, include_reference=False)
                information = smoothed_disagreement + (1 - 2 * margin) + (0.5 if decision else 0)
                value = information / max(predicted.predicted_cost_usd, 1e-12)
            return (-value, record_id)

        record_id, screen = min(screened, key=priority)
        prerequisites = tuple(str(item) for item in screen["request_ids"])
        return _action(
            context,
            kind=ActionKind.ACQUIRE_DISCOVERY_CANDIDATE,
            record_id=record_id,
            track=EvidenceTrack.DISCOVERY,
            method=mode,
            include_reference=False,
            prerequisites=prerequisites,
            score=-priority((record_id, screen))[0],
            components={
                "screen_reference_positive": bool(screen["decision"]),
                "screen_minimum_margin": float(screen["margin"] or 0),
                "diagnostic_disagreement_rate_smoothed": smoothed_disagreement,
            },
        )
    if unscreened and context.usage.discovery_records < context.budget.maximum_discovery_records:
        record_id = deterministic_record_order(tuple(unscreened), seed)[0]
        return _action(
            context,
            kind=ActionKind.SCREEN_DISCOVERY_REFERENCE,
            record_id=record_id,
            track=EvidenceTrack.DISCOVERY,
            method=f"{mode}_screen",
            include_reference=True,
        )
    return None


class AcquisitionPolicy:
    def __init__(self, strategy: SamplingStrategy, *, seed: int = 5201) -> None:
        self.strategy = strategy
        self.seed = seed

    def choose(
        self, context: AcquisitionContext, matrix: HistoricalReplayMatrix
    ) -> ProfilingAction:
        if self.strategy == SamplingStrategy.RANDOM_REPRESENTATIVE:
            selected = _representative_action(context, seed=self.seed)
        elif self.strategy == SamplingStrategy.HYBRID_ACTIVE:
            representative_count = len(context.evidence.for_track(EvidenceTrack.CERTIFICATION))
            discovery_count = len(context.evidence.for_track(EvidenceTrack.DISCOVERY))
            if representative_count <= 3 * discovery_count:
                selected = _representative_action(context, seed=self.seed)
            else:
                selected = _screen_or_discovery_action(
                    context,
                    matrix,
                    seed=self.seed,
                    mode=SamplingStrategy.COST_AWARE,
                ) or _representative_action(context, seed=self.seed)
        else:
            selected = _screen_or_discovery_action(
                context, matrix, seed=self.seed, mode=self.strategy
            )
        if selected is None:
            return ProfilingAction(
                action_id=f"m52stop_{context.checkpoint}_{context.plan_id}",
                kind=ActionKind.STOP,
                acquisition_method=self.strategy,
                score_components={"reason": "no_feasible_unacquired_measurement"},
            )
        return selected
