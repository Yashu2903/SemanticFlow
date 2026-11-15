from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

from semanticflow.optimizer.fusion_cost import FusionPlanCost, GroupStatistics
from semanticflow.optimizer.fusion_planner import FusionPhysicalPlan


class HybridGroupCostEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    group_id: str
    execution_mode: str
    named_question_count: int = Field(gt=0)
    evidence_kind: str
    observed_requests: int = Field(ge=0)
    mean_input_tokens: float | None = Field(default=None, ge=0)
    mean_output_tokens: float | None = Field(default=None, ge=0)
    mean_cost_usd: float | None = Field(default=None, ge=0)
    prompt_overhead_tokens: float | None = None
    empirical_batch_cost_available: bool
    semantic_quality_verified: bool


def estimate_hybrid_plan_cost(
    plan: FusionPhysicalPlan,
    *,
    evaluation_records: int,
    statistics: Mapping[str, GroupStatistics],
    input_price_per_million: float,
    output_price_per_million: float,
    conditional_prefix_reach: Mapping[tuple[str, ...], float] | None = None,
) -> FusionPlanCost:
    """Mode-aware expected cost with fail-closed treatment of unmeasured groups."""
    from semanticflow.optimizer.fusion_cost import estimate_fusion_plan_cost

    result = estimate_fusion_plan_cost(
        plan,
        evaluation_records=evaluation_records,
        statistics=statistics,
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
        conditional_prefix_reach=conditional_prefix_reach,
    )
    # A later group remains required evidence even when its estimated reach is zero.
    missing = [group.group_id for group in plan.groups if group.group_id not in statistics]
    if missing and result.complete:
        raise AssertionError("an unmeasured hybrid group was marked complete")
    return result


class HybridEconomicsView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reuse_horizon: int = Field(gt=0)
    baseline_execution_cost_usd: float = Field(ge=0)
    candidate_execution_cost_usd: float = Field(ge=0)
    actual_profiling_expenditure_usd: float = Field(ge=0)
    standalone_profiling_cost_usd: float = Field(ge=0)
    shared_incremental_profiling_cost_usd: float = Field(ge=0)
    optimization_overhead_usd: float = Field(ge=0)
    cold_start_standalone_net_benefit_usd: float
    cold_start_shared_net_benefit_usd: float
    warm_start_net_benefit_usd: float


def hybrid_economics_views(
    *,
    reuse_horizon: int,
    baseline_execution_cost_usd: float,
    candidate_execution_cost_usd: float,
    actual_profiling_expenditure_usd: float,
    standalone_profiling_cost_usd: float,
    shared_incremental_profiling_cost_usd: float,
    optimization_overhead_usd: float = 0.0,
) -> HybridEconomicsView:
    values = (
        baseline_execution_cost_usd,
        candidate_execution_cost_usd,
        actual_profiling_expenditure_usd,
        standalone_profiling_cost_usd,
        shared_incremental_profiling_cost_usd,
        optimization_overhead_usd,
    )
    if reuse_horizon <= 0 or any(value < 0 for value in values):
        raise ValueError("reuse horizon must be positive and costs nonnegative")
    if shared_incremental_profiling_cost_usd > standalone_profiling_cost_usd:
        raise ValueError("shared incremental profiling cannot exceed standalone profiling")
    gross = reuse_horizon * (
        baseline_execution_cost_usd - candidate_execution_cost_usd
    )
    return HybridEconomicsView(
        reuse_horizon=reuse_horizon,
        baseline_execution_cost_usd=baseline_execution_cost_usd,
        candidate_execution_cost_usd=candidate_execution_cost_usd,
        actual_profiling_expenditure_usd=actual_profiling_expenditure_usd,
        standalone_profiling_cost_usd=standalone_profiling_cost_usd,
        shared_incremental_profiling_cost_usd=shared_incremental_profiling_cost_usd,
        optimization_overhead_usd=optimization_overhead_usd,
        cold_start_standalone_net_benefit_usd=(
            gross - standalone_profiling_cost_usd - optimization_overhead_usd
        ),
        cold_start_shared_net_benefit_usd=(
            gross - shared_incremental_profiling_cost_usd - optimization_overhead_usd
        ),
        warm_start_net_benefit_usd=gross - optimization_overhead_usd,
    )
