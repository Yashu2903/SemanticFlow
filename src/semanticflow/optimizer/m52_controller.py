from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from semanticflow.optimizer.m52_acquisition import (
    ActionKind,
    BudgetUsage,
    ProfilingAction,
    ProfilingBudget,
)
from semanticflow.optimizer.m52_active_profiler import (
    AcquisitionContext,
    AcquisitionPolicy,
    SamplingStrategy,
)
from semanticflow.optimizer.m52_cost_model import FrozenCostPrediction
from semanticflow.optimizer.m52_evidence import (
    EvidenceTrack,
    QualificationState,
    QualityEstimate,
    QualityEvidence,
)
from semanticflow.optimizer.m52_planner import PlanDecision, PolicyMode, select_plan
from semanticflow.optimizer.m52_quality_uncertainty import (
    SequentialQualificationConfig,
    qualify_evidence,
)
from semanticflow.optimizer.m52_replay import HistoricalReplayMatrix


class ControllerPhase(StrEnum):
    INITIALIZE = "INITIALIZE"
    EVALUATE_PRIOR = "EVALUATE_PRIOR"
    SELECT_ACTION = "SELECT_ACTION"
    ACQUIRE_EVIDENCE = "ACQUIRE_EVIDENCE"
    UPDATE_QUALITY_ESTIMATES = "UPDATE_QUALITY_ESTIMATES"
    UPDATE_COST_ESTIMATES = "UPDATE_COST_ESTIMATES"
    CHECK_QUALIFICATION = "CHECK_QUALIFICATION"
    CHECK_ECONOMIC_VALUE = "CHECK_ECONOMIC_VALUE"
    CONTINUE = "CONTINUE"
    SELECT_PLAN = "SELECT_PLAN"
    FALLBACK = "FALLBACK"
    TERMINATE = "TERMINATE"


class M52ControllerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "m5.2-controller-v1"
    mode: PolicyMode = PolicyMode.STRICT_CERTIFICATION
    sampling_strategy: SamplingStrategy = SamplingStrategy.HYBRID_ACTIVE
    qualification: SequentialQualificationConfig = SequentialQualificationConfig()
    budget: ProfilingBudget = ProfilingBudget()
    representative_checkpoints: tuple[int, ...] = (40, 80, 120, 160)
    reuse_horizon: int = Field(default=10, ge=1)
    minimum_expected_value_of_information_usd: float = 0.0
    optimization_cost_usd: float | None = Field(default=None, ge=0)
    risk_budget_minimum_wilson_recall_lower: float = Field(default=0.80, ge=0, le=1)
    risk_budget_maximum_wilson_changed_upper: float = Field(default=0.05, ge=0, le=1)
    seed: int = 5201


class ControllerTransition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(ge=0)
    phase: ControllerPhase
    reason: str
    checkpoint: int = Field(default=0, ge=0)
    action_id: str | None = None
    accrued_cost_usd: float = Field(ge=0)
    physical_requests: int = Field(ge=0)
    certification_records: int = Field(ge=0)
    discovery_records: int = Field(ge=0)
    quality_state: QualificationState | None = None
    expected_value_of_information_usd: float | None = None


class ProfilingState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    workload_id: str
    candidate_plan_id: str
    fallback_plan_id: str
    checkpoint: int = Field(default=0, ge=0)
    evidence: QualityEvidence
    usage: BudgetUsage
    quality_estimate: QualityEstimate | None = None
    fallback_prediction: FrozenCostPrediction
    candidate_prediction: FrozenCostPrediction
    expected_saving_usd: float
    reuse_horizon: int = Field(ge=1)
    expected_value_of_information_usd: float | None = None
    selected_next_action: ProfilingAction | None = None
    stopping_reason: str | None = None
    selected_plan_id: str | None = None


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "m5.2-evaluation-result-v1"
    policy_fingerprint: str
    workload_id: str
    candidate_plan_id: str
    fallback_plan_id: str
    decision: PlanDecision
    final_state: ProfilingState
    transitions: tuple[ControllerTransition, ...]
    acquired_measurement_ids: tuple[str, ...]
    historical_replay_only: bool = True
    independent_validation: bool = False
    provider_initialized: bool = False
    new_provider_requests: int = 0


def _point_gate(evidence: QualityEvidence) -> bool:
    counts = evidence.counts(EvidenceTrack.CERTIFICATION)
    if counts.observations < 40 or counts.reference_positives < 3:
        return False
    agreement = (counts.true_positive + counts.true_negative) / counts.observations
    recall = counts.true_positive / counts.reference_positives
    fnr = counts.false_negative / counts.reference_positives
    changed = (counts.false_positive + counts.false_negative) / counts.observations
    return agreement >= 0.98 and recall >= 0.95 and fnr <= 0.05 and changed <= 0.02


class M52Optimizer:
    def __init__(self, config: M52ControllerConfig) -> None:
        self.config = config
        payload = config.model_dump(mode="json")
        self.policy_fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _transition(
        self,
        output: list[ControllerTransition],
        phase: ControllerPhase,
        reason: str,
        state: ProfilingState,
        *,
        action_id: str | None = None,
        voi: float | None = None,
    ) -> None:
        output.append(
            ControllerTransition(
                sequence=len(output),
                phase=phase,
                reason=reason,
                checkpoint=state.checkpoint,
                action_id=action_id,
                accrued_cost_usd=state.usage.token_priced_cost_usd,
                physical_requests=state.usage.physical_requests,
                certification_records=state.usage.certification_records,
                discovery_records=state.usage.discovery_records,
                quality_state=(
                    state.quality_estimate.state if state.quality_estimate is not None else None
                ),
                expected_value_of_information_usd=voi,
            )
        )

    def _voi(self, state: ProfilingState, action: ProfilingAction) -> float:
        if action.predicted_cost is None:
            return float("-inf")
        optimistic_saving = max(
            0.0,
            state.fallback_prediction.predicted_cost_upper_usd
            - state.candidate_prediction.predicted_cost_lower_usd,
        )
        remaining_value = max(
            0.0,
            state.reuse_horizon * optimistic_saving
            - state.usage.token_priced_cost_usd
            - (self.config.optimization_cost_usd or 0),
        )
        counts = state.evidence.counts(EvidenceTrack.CERTIFICATION)
        record_gap = max(0, self.config.qualification.minimum_records - counts.observations)
        positive_gap = max(
            0,
            self.config.qualification.minimum_reference_positives - counts.reference_positives,
        )
        probability_of_decision_change = min(
            1.0,
            1 / (1 + record_gap / 20 + positive_gap),
        )
        if counts.observations == 0:
            probability_of_decision_change = 1.0
        return (
            probability_of_decision_change * remaining_value
            - action.predicted_cost.predicted_cost_usd
        )

    def run(
        self,
        *,
        matrix: HistoricalReplayMatrix,
        evidence: QualityEvidence,
        fallback_prediction: FrozenCostPrediction,
        candidate_prediction: FrozenCostPrediction,
        certification_pool: tuple[str, ...],
        discovery_pool: tuple[str, ...],
        evaluation_record_count: int,
        continuation_guard: Callable[[QualityEvidence, BudgetUsage], str | None] | None = None,
    ) -> EvaluationResult:
        fallback_plan_id = next(
            plan_id
            for plan_id, plan in matrix.plans.items()
            if plan["workload_id"] == evidence.workload_id
            and plan["plan_fingerprint"] == fallback_prediction.candidate_plan_fingerprint
        )
        state = ProfilingState(
            workload_id=evidence.workload_id,
            candidate_plan_id=evidence.plan_id,
            fallback_plan_id=fallback_plan_id,
            evidence=evidence,
            usage=BudgetUsage(),
            fallback_prediction=fallback_prediction,
            candidate_prediction=candidate_prediction,
            expected_saving_usd=(
                fallback_prediction.predicted_cost_usd - candidate_prediction.predicted_cost_usd
            ),
            reuse_horizon=self.config.reuse_horizon,
        )
        transitions: list[ControllerTransition] = []
        self._transition(transitions, ControllerPhase.INITIALIZE, "controller_initialized", state)
        self._transition(
            transitions, ControllerPhase.EVALUATE_PRIOR, "frozen_cost_predictions_loaded", state
        )
        policy = AcquisitionPolicy(self.config.sampling_strategy, seed=self.config.seed)
        stopping_reason: str | None = None
        while stopping_reason is None:
            if continuation_guard is not None:
                guarded_reason = continuation_guard(state.evidence, state.usage)
                if guarded_reason is not None:
                    stopping_reason = guarded_reason
                    break
            cert_count = len(state.evidence.for_track(EvidenceTrack.CERTIFICATION))
            checkpoint = sum(
                cert_count >= value for value in self.config.representative_checkpoints
            )
            if cert_count in self.config.representative_checkpoints:
                estimate = qualify_evidence(
                    state.evidence,
                    self.config.qualification,
                    checkpoint=max(0, checkpoint - 1),
                )
                state = state.model_copy(
                    update={"checkpoint": checkpoint, "quality_estimate": estimate}
                )
                self._transition(
                    transitions,
                    ControllerPhase.CHECK_QUALIFICATION,
                    "scheduled_representative_checkpoint",
                    state,
                )
                if estimate.state == QualificationState.QUALIFIED:
                    stopping_reason = "candidate_statistically_qualified"
                    break
                if estimate.state in {
                    QualificationState.REJECTED,
                    QualificationState.PROVIDER_FAILURE,
                    QualificationState.UNSUPPORTED,
                }:
                    stopping_reason = f"quality_{estimate.state.value.lower()}"
                    break
            context = AcquisitionContext(
                strategy=self.config.sampling_strategy,
                workload_id=state.workload_id,
                plan_id=state.candidate_plan_id,
                checkpoint=state.checkpoint,
                certification_pool=certification_pool,
                discovery_pool=discovery_pool,
                evidence=state.evidence,
                usage=state.usage,
                budget=self.config.budget,
                fallback_prediction=state.fallback_prediction,
                candidate_prediction=state.candidate_prediction,
                evaluation_record_count=evaluation_record_count,
                reuse_horizon=state.reuse_horizon,
            )
            action = policy.choose(context, matrix)
            state = state.model_copy(update={"selected_next_action": action})
            self._transition(
                transitions,
                ControllerPhase.SELECT_ACTION,
                "acquisition_policy_ranked_visible_actions",
                state,
                action_id=action.action_id,
            )
            if action.kind == ActionKind.STOP:
                stopping_reason = str(action.score_components.get("reason", "no_feasible_action"))
                break
            assert action.predicted_cost is not None
            if not state.usage.permits(self.config.budget, action.predicted_cost):
                stopping_reason = "profiling_budget_exhausted"
                break
            voi = self._voi(state, action)
            state = state.model_copy(update={"expected_value_of_information_usd": voi})
            self._transition(
                transitions,
                ControllerPhase.CHECK_ECONOMIC_VALUE,
                "explicit_hard_quality_constrained_voi_approximation",
                state,
                action_id=action.action_id,
                voi=voi,
            )
            if voi <= self.config.minimum_expected_value_of_information_usd:
                stopping_reason = "expected_value_of_information_below_acquisition_cost"
                break
            acquisition = matrix.acquire(action)
            self._transition(
                transitions,
                ControllerPhase.ACQUIRE_EVIDENCE,
                "historical_measurement_revealed_and_charged",
                state,
                action_id=action.action_id,
            )
            if acquisition.receipt.unsupported_reason is not None:
                stopping_reason = "missing_historical_measurement_or_prerequisite"
                break
            measurements = state.evidence.measurements
            if acquisition.quality_measurement is not None:
                measurements = (*measurements, acquisition.quality_measurement)
            records = {row.record_id for row in measurements}
            cert_records = {
                row.record_id for row in measurements if row.track == EvidenceTrack.CERTIFICATION
            }
            discovery_records = set(matrix.visibility.screened_record_ids) | {
                row.record_id for row in measurements if row.track == EvidenceTrack.DISCOVERY
            }
            receipt = acquisition.receipt
            usage = BudgetUsage(
                physical_requests=state.usage.physical_requests + receipt.physical_requests,
                named_question_evaluations=(
                    state.usage.named_question_evaluations
                    + receipt.named_question_evaluations
                ),
                input_tokens=state.usage.input_tokens + receipt.input_tokens,
                output_tokens=state.usage.output_tokens + receipt.output_tokens,
                token_priced_cost_usd=(
                    state.usage.token_priced_cost_usd + receipt.token_priced_cost_usd
                ),
                records=len(records | discovery_records),
                actions=state.usage.actions + 1,
                certification_records=len(cert_records),
                discovery_records=len(discovery_records),
                positive_enrichment_records=sum(
                    row.reference_decision is True
                    for row in measurements
                    if row.track == EvidenceTrack.DISCOVERY
                ),
            )
            evidence = state.evidence.model_copy(update={"measurements": measurements})
            state = state.model_copy(update={"evidence": evidence, "usage": usage})
            self._transition(
                transitions,
                ControllerPhase.UPDATE_QUALITY_ESTIMATES,
                "discovery_and_certification_tracks_updated_separately",
                state,
            )
            self._transition(
                transitions,
                ControllerPhase.CONTINUE,
                "profiling_continues",
                state,
            )
        if state.quality_estimate is None:
            checkpoint = min(
                self.config.qualification.maximum_checkpoints - 1,
                max(0, state.checkpoint),
            )
            state = state.model_copy(
                update={
                    "quality_estimate": qualify_evidence(
                        state.evidence, self.config.qualification, checkpoint=checkpoint
                    )
                }
            )
        decision = select_plan(
            mode=self.config.mode,
            fallback_plan_id=state.fallback_plan_id,
            candidate_plan_id=state.candidate_plan_id,
            fallback_prediction=state.fallback_prediction,
            candidate_prediction=state.candidate_prediction,
            quality=state.quality_estimate,
            point_gate_passed=_point_gate(state.evidence),
            profiling_cost_usd=state.usage.token_priced_cost_usd,
            reuse_horizon=state.reuse_horizon,
            risk_budget_allows_uncertain=bool(
                self.config.mode == PolicyMode.RISK_BUDGET
                and state.quality_estimate.state == QualificationState.UNCERTAIN
                and state.quality_estimate.recall.wilson_lower is not None
                and state.quality_estimate.recall.wilson_lower
                >= self.config.risk_budget_minimum_wilson_recall_lower
                and state.quality_estimate.changed_decision_rate.wilson_upper is not None
                and state.quality_estimate.changed_decision_rate.wilson_upper
                <= self.config.risk_budget_maximum_wilson_changed_upper
            ),
        )
        phase = (
            ControllerPhase.SELECT_PLAN if not decision.used_fallback else ControllerPhase.FALLBACK
        )
        state = state.model_copy(
            update={
                "stopping_reason": stopping_reason,
                "selected_plan_id": decision.selected_plan_id,
            }
        )
        self._transition(transitions, phase, stopping_reason or decision.reason, state)
        self._transition(transitions, ControllerPhase.TERMINATE, "controller_terminated", state)
        return EvaluationResult(
            policy_fingerprint=self.policy_fingerprint,
            workload_id=state.workload_id,
            candidate_plan_id=state.candidate_plan_id,
            fallback_plan_id=state.fallback_plan_id,
            decision=decision,
            final_state=state,
            transitions=tuple(transitions),
            acquired_measurement_ids=tuple(sorted(matrix.visibility.acquired_measurement_ids)),
        )
