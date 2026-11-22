from __future__ import annotations

import hashlib
import json
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from semanticflow.optimizer.m52_acquisition import BudgetUsage
from semanticflow.optimizer.m52_controller import (
    EvaluationResult,
    M52ControllerConfig,
    M52Optimizer,
)
from semanticflow.optimizer.m52_cost_model import FrozenCostPrediction
from semanticflow.optimizer.m52_evidence import EvidenceTrack, QualityEvidence
from semanticflow.optimizer.m52_replay import HistoricalReplayMatrix
from semanticflow.optimizer.m521_economic_feasibility import (
    EconomicFeasibility,
    EconomicVerdict,
    evaluate_economic_feasibility,
)
from semanticflow.optimizer.m521_error_budget import QualificationRequirements
from semanticflow.optimizer.m521_feasibility import (
    EvidenceBudget,
    FeasibilityCertificate,
    FeasibilityVerdict,
    analyze_feasibility,
)


class ProfilingDecision(StrEnum):
    CONTINUE = "CONTINUE"
    QUALIFY = "QUALIFY"
    REJECT = "REJECT"
    FALLBACK = "FALLBACK"
    STOP = "STOP"


class M521ControllerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "m5.2.1-controller-v1"
    requirements: QualificationRequirements
    legacy_controller: M52ControllerConfig
    enforce_mathematical_feasibility: bool = True
    enforce_economic_feasibility: bool = True

    @property
    def policy_fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()


class M521Evaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_fingerprint: str
    initial_certificate: FeasibilityCertificate
    final_certificate: FeasibilityCertificate
    economic_feasibility: EconomicFeasibility
    decision: ProfilingDecision
    stop_reason: str
    legacy_result: EvaluationResult | None = None
    provider_initialized: bool = False
    new_provider_requests: int = 0


class M521Optimizer:
    def __init__(self, config: M521ControllerConfig) -> None:
        self.config = config
        self.certificates: list[FeasibilityCertificate] = []

    def _budget(
        self,
        evidence: QualityEvidence,
        usage: BudgetUsage,
        *,
        input_tokens_per_pair: float,
        cost_per_pair_usd: float,
        physical_arms_available: bool,
    ) -> EvidenceBudget:
        counts = evidence.counts(EvidenceTrack.CERTIFICATION)
        legacy = self.config.legacy_controller.budget
        remaining_records = max(0, legacy.maximum_certification_records - counts.observations)
        return EvidenceBudget(
            maximum_representative_records=legacy.maximum_certification_records,
            remaining_representative_records=remaining_records,
            maximum_positive_references=counts.reference_positives + remaining_records,
            remaining_physical_requests=max(
                0, legacy.maximum_physical_requests - usage.physical_requests
            ),
            remaining_input_tokens=None,
            remaining_cost_usd=max(
                0.0, legacy.maximum_cost_usd - usage.token_priced_cost_usd
            ),
            requests_per_representative_pair=2,
            expected_input_tokens_per_pair=input_tokens_per_pair,
            expected_cost_per_pair_usd=cost_per_pair_usd,
            physical_arms_available=physical_arms_available,
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
        input_tokens_per_pair: float,
        cost_per_pair_usd: float,
        physical_arms_available: bool = True,
    ) -> M521Evaluation:
        zero_usage = BudgetUsage()

        def certificate(current: QualityEvidence, usage: BudgetUsage) -> FeasibilityCertificate:
            result = analyze_feasibility(
                candidate_plan_fingerprint=current.plan_fingerprint,
                requirements=self.config.requirements,
                counts=current.counts(EvidenceTrack.CERTIFICATION),
                budget=self._budget(
                    current,
                    usage,
                    input_tokens_per_pair=input_tokens_per_pair,
                    cost_per_pair_usd=cost_per_pair_usd,
                    physical_arms_available=physical_arms_available,
                ),
            )
            self.certificates.append(result)
            return result

        initial = certificate(evidence, zero_usage)
        needed = max(
            initial.recall_constraint.minimum_additional_observations or 0,
            initial.agreement_constraint.minimum_additional_observations or 0,
        )
        saving = fallback_prediction.predicted_cost_usd - candidate_prediction.predicted_cost_usd
        economic = evaluate_economic_feasibility(
            reuse_horizon=self.config.legacy_controller.reuse_horizon,
            execution_saving_per_run_usd=saving,
            remaining_profiling_cost_usd=needed * cost_per_pair_usd,
        )
        impossible = initial.mathematical_verdict not in {
            FeasibilityVerdict.FEASIBLE_NOW,
            FeasibilityVerdict.FEASIBLE_WITH_MORE_EVIDENCE,
        }
        if self.config.enforce_mathematical_feasibility and impossible:
            return M521Evaluation(
                policy_fingerprint=self.config.policy_fingerprint,
                initial_certificate=initial,
                final_certificate=initial,
                economic_feasibility=economic,
                decision=ProfilingDecision.FALLBACK,
                stop_reason=f"mathematical_feasibility:{initial.mathematical_verdict.value}",
            )
        if (
            self.config.enforce_economic_feasibility
            and economic.verdict == EconomicVerdict.UNJUSTIFIED
        ):
            return M521Evaluation(
                policy_fingerprint=self.config.policy_fingerprint,
                initial_certificate=initial,
                final_certificate=initial,
                economic_feasibility=economic,
                decision=ProfilingDecision.FALLBACK,
                stop_reason="economic_feasibility:maximum_benefit_cannot_repay_profile",
            )

        def guard(current: QualityEvidence, usage: BudgetUsage) -> str | None:
            current_certificate = certificate(current, usage)
            if current_certificate.mathematical_verdict not in {
                FeasibilityVerdict.FEASIBLE_NOW,
                FeasibilityVerdict.FEASIBLE_WITH_MORE_EVIDENCE,
            }:
                return (
                    "mathematical_feasibility:"
                    f"{current_certificate.mathematical_verdict.value}"
                )
            return None

        result = M52Optimizer(self.config.legacy_controller).run(
            matrix=matrix,
            evidence=evidence,
            fallback_prediction=fallback_prediction,
            candidate_prediction=candidate_prediction,
            certification_pool=certification_pool,
            discovery_pool=discovery_pool,
            evaluation_record_count=evaluation_record_count,
            continuation_guard=guard,
        )
        final = self.certificates[-1]
        return M521Evaluation(
            policy_fingerprint=self.config.policy_fingerprint,
            initial_certificate=initial,
            final_certificate=final,
            economic_feasibility=economic,
            decision=(
                ProfilingDecision.FALLBACK
                if result.decision.used_fallback
                else ProfilingDecision.QUALIFY
            ),
            stop_reason=result.final_state.stopping_reason or result.decision.reason,
            legacy_result=result,
        )
