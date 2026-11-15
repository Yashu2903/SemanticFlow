from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from semanticflow.optimizer.fusion import (
    FusedOperator,
    FusionMode,
    ProviderFusionCapabilities,
    build_fused_operator,
)
from semanticflow.optimizer.fusion_planner import (
    FusionPhysicalPlan,
    build_physical_plan,
    ordered_set_partitions,
)
from semanticflow.query.models import PredicateSpec

HYBRID_PLANNER_VERSION = "m5.1-mixed-mode-v1"


class HybridOperatorMetadata(BaseModel):
    """Frozen identity of the complete provider-visible physical request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operator_id: str
    execution_mode: FusionMode
    predicate_ids: tuple[str, ...]
    exact_instructions: tuple[str, ...]
    question_names: tuple[str, ...]
    prompt_hashes: Mapping[str, str]
    request_fingerprint: str
    dataset_fingerprint: str
    model_version: str
    threshold: float = Field(ge=0, le=1)
    provider_settings: Mapping[str, object]
    result_schema: str


def hybrid_operator_metadata(
    operator: FusedOperator,
    *,
    dataset_fingerprint: str,
    provider_settings: Mapping[str, object] | None = None,
) -> HybridOperatorMetadata:
    questions = operator.provider_questions()
    payload = {
        "dataset_fingerprint": dataset_fingerprint,
        "questions": questions,
        "question_order": list(questions),
        "model": operator.requested_model,
        "threshold": operator.threshold,
        "provider_settings": dict(provider_settings or operator.execution_parameters),
        "result_schema": operator.result_schema_version,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return HybridOperatorMetadata(
        operator_id=operator.group_id,
        execution_mode=operator.fusion_mode,
        predicate_ids=operator.predicate_ids,
        exact_instructions=operator.source_instructions,
        question_names=tuple(questions),
        prompt_hashes=dict(operator.predicate_hashes),
        request_fingerprint=fingerprint,
        dataset_fingerprint=dataset_fingerprint,
        model_version=operator.requested_model,
        threshold=operator.threshold,
        provider_settings=dict(provider_settings or operator.execution_parameters),
        result_schema=operator.result_schema_version,
    )


def enumerate_hybrid_plans(
    predicates: Sequence[PredicateSpec],
    *,
    workload_id: str,
    logical_query_id: str,
    requested_model: str,
    threshold: float,
    capabilities: ProviderFusionCapabilities,
    configuration_fingerprint: str,
    multi_predicate_modes: Sequence[Literal["conjunctive_binary", "multi_decision"]] = (
        "conjunctive_binary",
        "multi_decision",
    ),
    maximum_group_size: int | None = None,
) -> list[FusionPhysicalPlan]:
    """Enumerate ordered groupings and every supported physical mode assignment."""
    if not multi_predicate_modes or len(set(multi_predicate_modes)) != len(
        multi_predicate_modes
    ):
        raise ValueError("multi-predicate modes must be nonempty and unique")
    by_id = {predicate.id: predicate for predicate in predicates}
    if len(by_id) != len(predicates):
        raise ValueError("predicate IDs must be unique")
    candidates: dict[str, FusionPhysicalPlan] = {}
    for partition in ordered_set_partitions(
        list(by_id), maximum_group_size=maximum_group_size
    ):
        mode_options: list[tuple[FusionMode, ...]] = []
        supported = True
        for group in partition:
            choices: tuple[FusionMode, ...]
            if len(group) == 1:
                choices = ("single",)
            else:
                choices = tuple(
                    mode
                    for mode in multi_predicate_modes
                    if capabilities.supports(mode, len(group))
                )
            if not choices:
                supported = False
                break
            mode_options.append(choices)
        if not supported:
            continue
        for modes in itertools.product(*mode_options):
            groups = [
                build_fused_operator(
                    [by_id[predicate_id] for predicate_id in predicate_ids],
                    mode=mode,
                    requested_model=requested_model,
                    threshold=threshold,
                )
                for predicate_ids, mode in zip(partition, modes, strict=True)
            ]
            plan = build_physical_plan(
                workload_id=workload_id,
                logical_query_id=logical_query_id,
                logical_predicate_ids=list(by_id),
                groups=groups,
                configuration_fingerprint=configuration_fingerprint,
            ).model_copy(update={"planner_version": HYBRID_PLANNER_VERSION})
            candidates[plan.plan_id] = plan
    return sorted(
        candidates.values(),
        key=lambda plan: (
            tuple(group.predicate_ids for group in plan.groups),
            tuple(group.fusion_mode for group in plan.groups),
        ),
    )


def fixed_selective_first_hybrid(
    plans: Sequence[FusionPhysicalPlan], original_order: Sequence[str]
) -> FusionPhysicalPlan:
    target_groups = ((original_order[0],), tuple(original_order[1:]))
    matches = [
        plan
        for plan in plans
        if plan.canonical_grouping == target_groups
        and plan.groups[-1].fusion_mode == "multi_decision"
    ]
    if len(matches) != 1:
        raise ValueError("candidate set lacks one selective-first multi-decision plan")
    return matches[0]


def full_batch_plan(plans: Sequence[FusionPhysicalPlan]) -> FusionPhysicalPlan:
    matches = [
        plan
        for plan in plans
        if len(plan.groups) == 1 and plan.groups[0].fusion_mode == "multi_decision"
    ]
    if len(matches) != 1:
        raise ValueError("candidate set lacks exactly one full multi-decision plan")
    return matches[0]
