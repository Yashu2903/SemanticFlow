from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from semanticflow.optimizer.fusion_planner import FusionPhysicalPlan

EvidenceSource = Literal[
    "measured_fused",
    "measured_separate",
    "conservative_model",
    "mock_infrastructure",
]


class GroupStatistics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    group_id: str
    predicate_ids: tuple[str, ...]
    fusion_mode: str
    sample_count: int = Field(ge=0)
    passing_count: int = Field(ge=0)
    empirical_selectivity: float | None = Field(default=None, ge=0.0, le=1.0)
    smoothed_selectivity: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_input_tokens: float | None = Field(default=None, ge=0.0)
    mean_output_tokens: float | None = Field(default=None, ge=0.0)
    mean_cost_usd: float | None = Field(default=None, ge=0.0)
    mean_latency_seconds: float | None = Field(default=None, ge=0.0)
    provider_error_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_source: EvidenceSource
    evidence_artifact: str | None = None
    reference_agreement: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_statistics(self) -> GroupStatistics:
        if self.passing_count > self.sample_count:
            raise ValueError("passing count cannot exceed sample count")
        if self.sample_count == 0:
            if self.empirical_selectivity is not None:
                raise ValueError("unmeasured groups cannot have empirical selectivity")
        elif self.empirical_selectivity is None or not math.isclose(
            self.empirical_selectivity,
            self.passing_count / self.sample_count,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("empirical selectivity must match measured counts")
        if self.evidence_source == "measured_fused" and self.sample_count == 0:
            raise ValueError("measured fused evidence requires observations")
        return self

    @property
    def selectivity(self) -> float | None:
        return (
            self.smoothed_selectivity
            if self.smoothed_selectivity is not None
            else self.empirical_selectivity
        )

    @property
    def is_real_fused_measurement(self) -> bool:
        return self.evidence_source == "measured_fused"


class FusionPlanCost(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str
    evaluation_records: int = Field(gt=0)
    complete: bool
    expected_provider_requests: float | None = Field(default=None, ge=0.0)
    no_early_rejection_requests: int = Field(ge=0)
    expected_input_tokens: float | None = Field(default=None, ge=0.0)
    expected_output_tokens: float | None = Field(default=None, ge=0.0)
    expected_cost_usd: float | None = Field(default=None, ge=0.0)
    expected_latency_seconds: float | None = Field(default=None, ge=0.0)
    expected_stage_cardinalities: tuple[float, ...]
    stage_estimates: tuple[dict[str, object], ...]
    unknown_groups: tuple[str, ...]
    measured_fused_groups: tuple[str, ...]
    modeled_groups: tuple[str, ...]
    reach_model: str


def jeffreys_selectivity(passing: int, total: int) -> float:
    if total < 0 or passing < 0 or passing > total:
        raise ValueError("invalid Bernoulli counts")
    return (passing + 0.5) / (total + 1.0)


def estimate_fusion_plan_cost(
    plan: FusionPhysicalPlan,
    *,
    evaluation_records: int,
    statistics: Mapping[str, GroupStatistics],
    input_price_per_million: float,
    output_price_per_million: float,
    conditional_prefix_reach: Mapping[tuple[str, ...], float] | None = None,
) -> FusionPlanCost:
    if evaluation_records <= 0:
        raise ValueError("evaluation_records must be positive")
    if input_price_per_million < 0 or output_price_per_million < 0:
        raise ValueError("prices cannot be negative")
    reach = 1.0
    expected_requests = 0.0
    input_tokens = 0.0
    output_tokens = 0.0
    cost = 0.0
    latency = 0.0
    has_input = has_output = has_cost = has_latency = True
    unknown: list[str] = []
    measured: list[str] = []
    modeled: list[str] = []
    stages: list[dict[str, object]] = []
    cardinalities: list[float] = []
    prefix: list[str] = []

    for stage, group in enumerate(plan.groups, start=1):
        prefix_key = tuple(prefix)
        if conditional_prefix_reach is not None and prefix_key in conditional_prefix_reach:
            reach = conditional_prefix_reach[prefix_key]
            reach_source = "measured_or_modeled_prefix"
        else:
            reach_source = "recursive_group_selectivity"
        stage_cardinality = evaluation_records * reach
        cardinalities.append(stage_cardinality)
        expected_requests += stage_cardinality
        stat = statistics.get(group.group_id)
        if stat is None:
            unknown.append(group.group_id)
            has_input = has_output = has_cost = has_latency = False
            selectivity = None
            evidence_source = "unknown"
        else:
            selectivity = stat.selectivity
            evidence_source = stat.evidence_source
            if stat.is_real_fused_measurement and len(group.predicate_ids) > 1:
                measured.append(group.group_id)
            elif len(group.predicate_ids) > 1:
                modeled.append(group.group_id)
            if stat.mean_input_tokens is None:
                has_input = False
            else:
                input_tokens += stage_cardinality * stat.mean_input_tokens
            if stat.mean_output_tokens is None:
                has_output = False
            else:
                output_tokens += stage_cardinality * stat.mean_output_tokens
            if stat.mean_cost_usd is not None:
                cost += stage_cardinality * stat.mean_cost_usd
            elif stat.mean_input_tokens is not None and stat.mean_output_tokens is not None:
                cost += stage_cardinality * (
                    stat.mean_input_tokens * input_price_per_million
                    + stat.mean_output_tokens * output_price_per_million
                ) / 1_000_000
            else:
                has_cost = False
            if stat.mean_latency_seconds is None:
                has_latency = False
            else:
                latency += stage_cardinality * stat.mean_latency_seconds
        stages.append(
            {
                "stage": stage,
                "group_id": group.group_id,
                "predicate_ids": list(group.predicate_ids),
                "reach_probability": reach,
                "expected_invocations": stage_cardinality,
                "reach_source": reach_source,
                "selectivity": selectivity,
                "evidence_source": evidence_source,
            }
        )
        if selectivity is None:
            reach = 0.0
        else:
            reach *= selectivity
        prefix.append(group.group_id)

    complete = not unknown and has_cost
    return FusionPlanCost(
        plan_id=plan.plan_id,
        evaluation_records=evaluation_records,
        complete=complete,
        expected_provider_requests=expected_requests,
        no_early_rejection_requests=evaluation_records * len(plan.groups),
        expected_input_tokens=input_tokens if has_input else None,
        expected_output_tokens=output_tokens if has_output else None,
        expected_cost_usd=cost if has_cost and not unknown else None,
        expected_latency_seconds=latency if has_latency else None,
        expected_stage_cardinalities=tuple(cardinalities),
        stage_estimates=tuple(stages),
        unknown_groups=tuple(unknown),
        measured_fused_groups=tuple(sorted(set(measured))),
        modeled_groups=tuple(sorted(set(modeled))),
        reach_model=(
            "conditional_prefix_override_then_recursive"
            if conditional_prefix_reach
            else "recursive_group_selectivity"
        ),
    )


class FusionEconomics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reuse_horizon: int = Field(gt=0)
    baseline_execution_cost_usd: float = Field(ge=0.0)
    candidate_execution_cost_usd: float = Field(ge=0.0)
    incremental_profiling_cost_usd: float = Field(ge=0.0)
    optimization_overhead_usd: float = Field(ge=0.0)
    gross_saving_usd: float
    net_benefit_usd: float
    break_even_reuse_count: int | None
    repetitions_are_projected: bool = True


def fusion_economics(
    *,
    reuse_horizon: int,
    baseline_execution_cost_usd: float,
    candidate_execution_cost_usd: float,
    incremental_profiling_cost_usd: float,
    optimization_overhead_usd: float = 0.0,
) -> FusionEconomics:
    if reuse_horizon <= 0:
        raise ValueError("reuse horizon must be positive")
    values = (
        baseline_execution_cost_usd,
        candidate_execution_cost_usd,
        incremental_profiling_cost_usd,
        optimization_overhead_usd,
    )
    if any(value < 0 for value in values):
        raise ValueError("economic costs cannot be negative")
    saving_per_execution = baseline_execution_cost_usd - candidate_execution_cost_usd
    acquisition = incremental_profiling_cost_usd + optimization_overhead_usd
    gross = reuse_horizon * saving_per_execution
    break_even = (
        math.ceil(acquisition / saving_per_execution)
        if saving_per_execution > 0
        else None
    )
    return FusionEconomics(
        reuse_horizon=reuse_horizon,
        baseline_execution_cost_usd=baseline_execution_cost_usd,
        candidate_execution_cost_usd=candidate_execution_cost_usd,
        incremental_profiling_cost_usd=incremental_profiling_cost_usd,
        optimization_overhead_usd=optimization_overhead_usd,
        gross_saving_usd=gross,
        net_benefit_usd=gross - acquisition,
        break_even_reuse_count=break_even,
    )
