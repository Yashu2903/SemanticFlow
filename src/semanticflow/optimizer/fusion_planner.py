from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from semanticflow.optimizer.fusion import (
    FusedOperator,
    FusionMode,
    ProviderFusionCapabilities,
    build_fused_operator,
)
from semanticflow.query.models import PredicateSpec

FUSION_PLANNER_VERSION = "m5-ordered-set-partition-v1"

FusionObjective = Literal[
    "minimum_inference_dollars",
    "minimum_latency",
    "minimum_projected_total_cost",
    "minimum_cost_subject_to_quality",
]


class FusionPhysicalPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str
    workload_id: str
    logical_query_id: str
    logical_predicate_ids: tuple[str, ...]
    groups: tuple[FusedOperator, ...]
    estimated_execution_cost_usd: float | None = Field(default=None, ge=0.0)
    estimated_provider_requests: float | None = Field(default=None, ge=0.0)
    estimated_input_tokens: float | None = Field(default=None, ge=0.0)
    estimated_output_tokens: float | None = Field(default=None, ge=0.0)
    estimated_latency_seconds: float | None = Field(default=None, ge=0.0)
    expected_stage_cardinalities: tuple[float, ...] | None = None
    quality_support_metadata: Mapping[str, Any] = Field(default_factory=dict)
    profiling_requirements: tuple[str, ...] = ()
    profiling_cost_usd: float | None = Field(default=None, ge=0.0)
    fallback_strategy: str = "existing_adaptive"
    planner_version: str = FUSION_PLANNER_VERSION
    configuration_fingerprint: str

    @model_validator(mode="after")
    def validate_plan(self) -> FusionPhysicalPlan:
        if not self.groups:
            raise ValueError("a physical plan requires at least one group")
        flattened = [predicate for group in self.groups for predicate in group.predicate_ids]
        if len(flattened) != len(set(flattened)):
            raise ValueError("a logical predicate appears more than once")
        if set(flattened) != set(self.logical_predicate_ids):
            raise ValueError("physical plan must cover every logical predicate exactly once")
        if any(not group.predicate_ids for group in self.groups):
            raise ValueError("physical groups cannot be empty")
        if self.expected_stage_cardinalities is not None and len(
            self.expected_stage_cardinalities
        ) != len(self.groups):
            raise ValueError("stage cardinalities must align with groups")
        return self

    @property
    def contains_fusion(self) -> bool:
        return any(len(group.predicate_ids) > 1 for group in self.groups)

    @property
    def canonical_grouping(self) -> tuple[tuple[str, ...], ...]:
        return tuple(group.predicate_ids for group in self.groups)


def _unordered_set_partitions(items: tuple[str, ...]) -> list[tuple[tuple[str, ...], ...]]:
    if not items:
        return [()]
    first, rest = items[0], items[1:]
    output: list[tuple[tuple[str, ...], ...]] = []
    for partition in _unordered_set_partitions(rest):
        output.append(((first,), *partition))
        for index in range(len(partition)):
            expanded = list(partition)
            expanded[index] = (first, *expanded[index])
            output.append(tuple(expanded))
    deduplicated = {tuple(sorted(partition)): partition for partition in output}
    return list(deduplicated.values())


def ordered_set_partitions(
    predicate_ids: Sequence[str], *, maximum_group_size: int | None = None
) -> list[tuple[tuple[str, ...], ...]]:
    """Enumerate ordered partitions while keeping internal group order canonical."""
    ids = tuple(predicate_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("predicate IDs must be nonempty and unique")
    maximum = maximum_group_size or len(ids)
    if maximum < 1:
        raise ValueError("maximum_group_size must be positive")
    position = {predicate_id: index for index, predicate_id in enumerate(ids)}
    plans: set[tuple[tuple[str, ...], ...]] = set()
    for partition in _unordered_set_partitions(ids):
        canonical = tuple(
            tuple(sorted(group, key=position.__getitem__)) for group in partition
        )
        if any(len(group) > maximum for group in canonical):
            continue
        plans.update(itertools.permutations(canonical))
    return sorted(plans)


def _plan_identity_payload(
    workload_id: str,
    logical_query_id: str,
    logical_predicate_ids: Sequence[str],
    groups: Sequence[FusedOperator],
    configuration_fingerprint: str,
) -> dict[str, Any]:
    return {
        "workload_id": workload_id,
        "logical_query_id": logical_query_id,
        "logical_predicate_ids": list(logical_predicate_ids),
        "groups": [
            {
                "predicate_ids": list(group.predicate_ids),
                "fusion_mode": group.fusion_mode,
                "fusion_prompt_hash": group.fusion_prompt_hash,
            }
            for group in groups
        ],
        "planner_version": FUSION_PLANNER_VERSION,
        "configuration_fingerprint": configuration_fingerprint,
    }


def build_physical_plan(
    *,
    workload_id: str,
    logical_query_id: str,
    logical_predicate_ids: Sequence[str],
    groups: Sequence[FusedOperator],
    configuration_fingerprint: str,
    fallback_strategy: str = "existing_adaptive",
) -> FusionPhysicalPlan:
    payload = _plan_identity_payload(
        workload_id,
        logical_query_id,
        logical_predicate_ids,
        groups,
        configuration_fingerprint,
    )
    plan_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    profiling_requirements = tuple(
        group.group_id for group in groups if len(group.predicate_ids) > 1
    )
    return FusionPhysicalPlan(
        plan_id=f"m5plan_{plan_hash[:20]}",
        workload_id=workload_id,
        logical_query_id=logical_query_id,
        logical_predicate_ids=tuple(logical_predicate_ids),
        groups=tuple(groups),
        profiling_requirements=profiling_requirements,
        fallback_strategy=fallback_strategy,
        configuration_fingerprint=configuration_fingerprint,
    )


def enumerate_fusion_plans(
    predicates: Sequence[PredicateSpec],
    *,
    workload_id: str,
    logical_query_id: str,
    requested_model: str,
    threshold: float,
    fusion_mode: FusionMode,
    capabilities: ProviderFusionCapabilities,
    configuration_fingerprint: str,
    maximum_group_size: int | None = None,
) -> list[FusionPhysicalPlan]:
    if fusion_mode == "single":
        raise ValueError("candidate fusion mode must support groups larger than one")
    by_id = {predicate.id: predicate for predicate in predicates}
    partitions = ordered_set_partitions(
        list(by_id), maximum_group_size=maximum_group_size
    )
    plans: list[FusionPhysicalPlan] = []
    for partition in partitions:
        groups: list[FusedOperator] = []
        supported = True
        for ids in partition:
            mode: FusionMode = "single" if len(ids) == 1 else fusion_mode
            if not capabilities.supports(mode, len(ids)):
                supported = False
                break
            groups.append(
                build_fused_operator(
                    [by_id[predicate_id] for predicate_id in ids],
                    mode=mode,
                    requested_model=requested_model,
                    threshold=threshold,
                )
            )
        if supported:
            plans.append(
                build_physical_plan(
                    workload_id=workload_id,
                    logical_query_id=logical_query_id,
                    logical_predicate_ids=list(by_id),
                    groups=groups,
                    configuration_fingerprint=configuration_fingerprint,
                )
            )
    unique = {plan.plan_id: plan for plan in plans}
    return sorted(unique.values(), key=lambda plan: plan.canonical_grouping)


def full_fusion_plan(plans: Sequence[FusionPhysicalPlan]) -> FusionPhysicalPlan:
    candidates = [plan for plan in plans if len(plan.groups) == 1]
    if len(candidates) != 1:
        raise ValueError("candidate set must contain exactly one full-fusion plan")
    return candidates[0]


def naive_adjacent_partial_plan(
    plans: Sequence[FusionPhysicalPlan], original_order: Sequence[str]
) -> FusionPhysicalPlan:
    if len(original_order) < 2:
        raise ValueError("partial fusion requires at least two predicates")
    target = (tuple(original_order[:2]), *(tuple([item]) for item in original_order[2:]))
    matches = [plan for plan in plans if plan.canonical_grouping == target]
    if len(matches) != 1:
        raise ValueError("candidate set does not contain the preregistered adjacent plan")
    return matches[0]


def separate_plan(
    plans: Sequence[FusionPhysicalPlan], predicate_order: Sequence[str]
) -> FusionPhysicalPlan:
    target = tuple((predicate_id,) for predicate_id in predicate_order)
    matches = [plan for plan in plans if plan.canonical_grouping == target]
    if len(matches) != 1:
        raise ValueError("candidate set does not contain the requested separate plan")
    return matches[0]


class FusionPlanSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: Literal[
        "full_fusion",
        "naive_partial_fusion",
        "cost_only_joint",
        "quality_constrained_joint",
        "profiling_aware_joint",
    ]
    selected_plan_id: str
    fallback_plan_id: str
    used_fallback: bool
    reason: str
    objective: FusionObjective
    candidate_count: int = Field(gt=0)
    eligible_candidate_count: int = Field(ge=0)
    predicted_execution_cost_usd: float | None = Field(default=None, ge=0.0)
    incremental_profiling_cost_usd: float = Field(default=0.0, ge=0.0)
    projected_net_benefit_usd: float | None = None
    quality_evidence_kind: str


def select_cost_only_plan(
    plans: Sequence[FusionPhysicalPlan],
    costs_usd: Mapping[str, float | None],
    *,
    fallback: FusionPhysicalPlan,
) -> FusionPlanSelection:
    eligible = [plan for plan in plans if costs_usd.get(plan.plan_id) is not None]
    if not eligible:
        return FusionPlanSelection(
            strategy="cost_only_joint",
            selected_plan_id=fallback.plan_id,
            fallback_plan_id=fallback.plan_id,
            used_fallback=True,
            reason="no_candidate_has_a_complete_cost_estimate",
            objective="minimum_inference_dollars",
            candidate_count=len(plans),
            eligible_candidate_count=0,
            quality_evidence_kind="not_applied_cost_only_ablation",
        )
    selected = min(eligible, key=lambda plan: (costs_usd[plan.plan_id], plan.plan_id))
    return FusionPlanSelection(
        strategy="cost_only_joint",
        selected_plan_id=selected.plan_id,
        fallback_plan_id=fallback.plan_id,
        used_fallback=False,
        reason="minimum_complete_estimated_execution_cost",
        objective="minimum_inference_dollars",
        candidate_count=len(plans),
        eligible_candidate_count=len(eligible),
        predicted_execution_cost_usd=costs_usd[selected.plan_id],
        quality_evidence_kind="not_applied_cost_only_ablation",
    )


def select_quality_constrained_plan(
    plans: Sequence[FusionPhysicalPlan],
    costs_usd: Mapping[str, float | None],
    quality_eligible: Mapping[str, bool],
    *,
    fallback: FusionPhysicalPlan,
) -> FusionPlanSelection:
    eligible = [
        plan
        for plan in plans
        if costs_usd.get(plan.plan_id) is not None
        and (not plan.contains_fusion or quality_eligible.get(plan.plan_id, False))
    ]
    fused_eligible = [plan for plan in eligible if plan.contains_fusion]
    if not fused_eligible:
        return FusionPlanSelection(
            strategy="quality_constrained_joint",
            selected_plan_id=fallback.plan_id,
            fallback_plan_id=fallback.plan_id,
            used_fallback=True,
            reason="no_fused_plan_has_complete_plan_quality_support",
            objective="minimum_cost_subject_to_quality",
            candidate_count=len(plans),
            eligible_candidate_count=len(eligible),
            predicted_execution_cost_usd=costs_usd.get(fallback.plan_id),
            quality_evidence_kind="measured_complete_plan_required",
        )
    selected = min(eligible, key=lambda plan: (costs_usd[plan.plan_id], plan.plan_id))
    return FusionPlanSelection(
        strategy="quality_constrained_joint",
        selected_plan_id=selected.plan_id,
        fallback_plan_id=fallback.plan_id,
        used_fallback=selected.plan_id == fallback.plan_id,
        reason="minimum_cost_among_quality_eligible_plans",
        objective="minimum_cost_subject_to_quality",
        candidate_count=len(plans),
        eligible_candidate_count=len(eligible),
        predicted_execution_cost_usd=costs_usd[selected.plan_id],
        quality_evidence_kind="measured_complete_plan_required",
    )


def select_profiling_aware_plan(
    plans: Sequence[FusionPhysicalPlan],
    costs_usd: Mapping[str, float | None],
    profiling_costs_usd: Mapping[str, float],
    quality_eligible: Mapping[str, bool],
    *,
    fallback: FusionPhysicalPlan,
    reuse_horizon: int,
    optimization_overhead_usd: float = 0.0,
) -> FusionPlanSelection:
    if reuse_horizon <= 0:
        raise ValueError("reuse_horizon must be positive")
    baseline_cost = costs_usd.get(fallback.plan_id)
    if baseline_cost is None:
        raise ValueError("fallback plan requires a complete cost estimate")
    eligible = [
        plan
        for plan in plans
        if costs_usd.get(plan.plan_id) is not None
        and (not plan.contains_fusion or quality_eligible.get(plan.plan_id, False))
    ]
    if not any(plan.contains_fusion for plan in eligible):
        return FusionPlanSelection(
            strategy="profiling_aware_joint",
            selected_plan_id=fallback.plan_id,
            fallback_plan_id=fallback.plan_id,
            used_fallback=True,
            reason="no_quality_eligible_fused_plan",
            objective="minimum_projected_total_cost",
            candidate_count=len(plans),
            eligible_candidate_count=len(eligible),
            predicted_execution_cost_usd=baseline_cost,
            quality_evidence_kind="measured_complete_plan_required",
        )
    scored: list[tuple[float, FusionPhysicalPlan]] = []
    for plan in eligible:
        execution_cost = costs_usd[plan.plan_id]
        assert execution_cost is not None
        total = (
            reuse_horizon * execution_cost
            + profiling_costs_usd.get(plan.plan_id, 0.0)
            + optimization_overhead_usd
        )
        scored.append((total, plan))
    _, selected = min(scored, key=lambda item: (item[0], item[1].plan_id))
    selected_cost = costs_usd[selected.plan_id]
    assert selected_cost is not None
    profiling_cost = profiling_costs_usd.get(selected.plan_id, 0.0)
    net = reuse_horizon * (baseline_cost - selected_cost) - profiling_cost - (
        optimization_overhead_usd
    )
    if net <= 0:
        selected = fallback
        selected_cost = baseline_cost
        profiling_cost = 0.0
        reason = "projected_net_benefit_nonpositive"
        used_fallback = True
    else:
        reason = "minimum_projected_total_cost_with_positive_net_benefit"
        used_fallback = selected.plan_id == fallback.plan_id
    return FusionPlanSelection(
        strategy="profiling_aware_joint",
        selected_plan_id=selected.plan_id,
        fallback_plan_id=fallback.plan_id,
        used_fallback=used_fallback,
        reason=reason,
        objective="minimum_projected_total_cost",
        candidate_count=len(plans),
        eligible_candidate_count=len(eligible),
        predicted_execution_cost_usd=selected_cost,
        incremental_profiling_cost_usd=profiling_cost,
        projected_net_benefit_usd=net,
        quality_evidence_kind="measured_complete_plan_required",
    )
