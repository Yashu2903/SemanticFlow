from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from semanticflow.optimizer.fusion_planner import FusionPhysicalPlan
from semanticflow.optimizer.hybrid_quality import ClassifiedQualityEvidence

HybridStrategy = Literal[
    "fixed_separate",
    "existing_adaptive",
    "full_conjunctive_fusion",
    "full_multi_decision_batch",
    "fixed_hybrid",
    "cost_only_hybrid",
    "quality_constrained_hybrid",
    "profiling_aware_hybrid",
]


class HybridPlanSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: HybridStrategy
    selected_plan_id: str
    fallback_plan_id: str
    used_fallback: bool
    reason: str
    candidates_considered: int = Field(ge=0)
    quality_eligible_candidates: int = Field(ge=0)
    predicted_execution_cost_usd: float | None = Field(default=None, ge=0)
    profiling_cost_usd: float = Field(default=0, ge=0)
    projected_net_benefit_usd: float | None = None


def select_hybrid_plan(
    plans: Sequence[FusionPhysicalPlan],
    costs: Mapping[str, float | None],
    *,
    fallback: FusionPhysicalPlan,
    strategy: Literal[
        "cost_only_hybrid", "quality_constrained_hybrid", "profiling_aware_hybrid"
    ],
    quality: Mapping[str, ClassifiedQualityEvidence] | None = None,
    profiling_costs: Mapping[str, float] | None = None,
    reuse_horizon: int = 1,
) -> HybridPlanSelection:
    if reuse_horizon <= 0:
        raise ValueError("reuse_horizon must be positive")
    fallback_cost = costs.get(fallback.plan_id)
    if fallback_cost is None:
        raise ValueError("fallback requires a complete cost estimate")
    complete = [plan for plan in plans if costs.get(plan.plan_id) is not None]
    if strategy == "cost_only_hybrid":
        eligible = complete
    else:
        quality = quality or {}
        eligible = [
            plan
            for plan in complete
            if all(len(group.predicate_ids) == 1 for group in plan.groups)
            or quality.get(
                plan.plan_id,
                ClassifiedQualityEvidence(
                    status="missing_complete_plan_evidence",
                    eligible=False,
                    reasons=("missing_complete_plan_evidence",),
                    metrics=None,
                    population_representative=True,
                    enriched_sample=False,
                ),
            ).eligible
        ]
    physical_candidates = [
        plan for plan in eligible if any(len(group.predicate_ids) > 1 for group in plan.groups)
    ]
    if strategy != "cost_only_hybrid" and not physical_candidates:
        return HybridPlanSelection(
            strategy=strategy,
            selected_plan_id=fallback.plan_id,
            fallback_plan_id=fallback.plan_id,
            used_fallback=True,
            reason="no_supported_hybrid_has_sufficient_complete_plan_quality_evidence",
            candidates_considered=len(complete),
            quality_eligible_candidates=len(eligible),
            predicted_execution_cost_usd=fallback_cost,
        )
    if not eligible:
        return HybridPlanSelection(
            strategy=strategy,
            selected_plan_id=fallback.plan_id,
            fallback_plan_id=fallback.plan_id,
            used_fallback=True,
            reason="no_candidate_has_complete_cost_evidence",
            candidates_considered=0,
            quality_eligible_candidates=0,
            predicted_execution_cost_usd=fallback_cost,
        )
    profiling_costs = profiling_costs or {}
    if strategy == "profiling_aware_hybrid":
        score = lambda plan: (  # noqa: E731
            reuse_horizon * float(costs[plan.plan_id])
            + profiling_costs.get(plan.plan_id, 0),
            plan.plan_id,
        )
    else:
        score = lambda plan: (float(costs[plan.plan_id]), plan.plan_id)  # noqa: E731
    selected = min(eligible, key=score)
    selected_cost = float(costs[selected.plan_id])
    profiling_cost = profiling_costs.get(selected.plan_id, 0)
    net = reuse_horizon * (fallback_cost - selected_cost) - profiling_cost
    if strategy == "profiling_aware_hybrid" and net <= 0:
        selected = fallback
        selected_cost = fallback_cost
        profiling_cost = 0
        reason = "incremental_profiling_not_economically_justified"
    else:
        reason = "minimum_supported_token_priced_cost"
    return HybridPlanSelection(
        strategy=strategy,
        selected_plan_id=selected.plan_id,
        fallback_plan_id=fallback.plan_id,
        used_fallback=selected.plan_id == fallback.plan_id,
        reason=reason,
        candidates_considered=len(complete),
        quality_eligible_candidates=len(eligible),
        predicted_execution_cost_usd=selected_cost,
        profiling_cost_usd=profiling_cost,
        projected_net_benefit_usd=net,
    )
