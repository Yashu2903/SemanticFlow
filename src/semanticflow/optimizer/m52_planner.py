from __future__ import annotations

import hashlib
import json
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from semanticflow.optimizer.m52_cost_model import FrozenCostPrediction
from semanticflow.optimizer.m52_evidence import QualificationState, QualityEstimate


class PolicyMode(StrEnum):
    STRICT_CERTIFICATION = "strict_certification"
    RISK_BUDGET = "risk_budget"
    COST_ONLY = "cost_only"
    M51_FIXED_GATE = "m5.1_fixed_gate"
    EXISTING_ADAPTIVE = "existing_adaptive"


class PlanDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_plan_id: str
    fallback_plan_id: str
    used_fallback: bool
    reason: str
    policy_mode: PolicyMode
    predicted_execution_saving_usd: float
    profiling_cost_usd: float = Field(ge=0)
    reuse_horizon: int = Field(ge=1)
    profiling_inclusive_net_benefit_usd: float
    quality_state: QualificationState | None
    deployment_quality_safe: bool


def policy_fingerprint(configuration: BaseModel | dict[str, object]) -> str:
    payload = (
        configuration.model_dump(mode="json")
        if isinstance(configuration, BaseModel)
        else configuration
    )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def select_plan(
    *,
    mode: PolicyMode,
    fallback_plan_id: str,
    candidate_plan_id: str,
    fallback_prediction: FrozenCostPrediction,
    candidate_prediction: FrozenCostPrediction,
    quality: QualityEstimate | None,
    point_gate_passed: bool,
    profiling_cost_usd: float,
    reuse_horizon: int,
    risk_budget_allows_uncertain: bool = False,
) -> PlanDecision:
    saving = fallback_prediction.predicted_cost_usd - candidate_prediction.predicted_cost_usd
    net = reuse_horizon * saving - profiling_cost_usd
    if mode == PolicyMode.EXISTING_ADAPTIVE:
        reason = "control_baseline"
        eligible = False
    elif saving <= 0:
        reason = "non_positive_expected_execution_saving"
        eligible = False
    elif net <= 0:
        reason = "profiling_inclusive_net_benefit_not_positive"
        eligible = False
    elif mode == PolicyMode.COST_ONLY:
        reason = "cost_only_ablation_not_quality_safe"
        eligible = True
    elif mode == PolicyMode.M51_FIXED_GATE:
        reason = "m5.1_point_gate_passed" if point_gate_passed else "m5.1_point_gate_failed"
        eligible = point_gate_passed
    elif mode == PolicyMode.STRICT_CERTIFICATION:
        eligible = quality is not None and quality.state == QualificationState.QUALIFIED
        reason = "strict_quality_certified" if eligible else "strict_quality_not_certified"
    else:
        eligible = quality is not None and (
            quality.state == QualificationState.QUALIFIED
            or (quality.state == QualificationState.UNCERTAIN and risk_budget_allows_uncertain)
        )
        reason = "risk_budget_admitted" if eligible else "risk_budget_not_satisfied"
    selected = candidate_plan_id if eligible else fallback_plan_id
    quality_safe = bool(
        eligible
        and mode not in {PolicyMode.COST_ONLY}
        and quality is not None
        and quality.state == QualificationState.QUALIFIED
    )
    return PlanDecision(
        selected_plan_id=selected,
        fallback_plan_id=fallback_plan_id,
        used_fallback=not eligible,
        reason=reason,
        policy_mode=mode,
        predicted_execution_saving_usd=saving if eligible else 0,
        profiling_cost_usd=profiling_cost_usd,
        reuse_horizon=reuse_horizon,
        profiling_inclusive_net_benefit_usd=net if eligible else -profiling_cost_usd,
        quality_state=quality.state if quality is not None else None,
        deployment_quality_safe=quality_safe,
    )
