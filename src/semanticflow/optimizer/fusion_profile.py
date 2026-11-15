from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from semanticflow.executor.fused import FusedDecisionTrace
from semanticflow.optimizer.fusion import FusedOperator
from semanticflow.optimizer.fusion_cost import GroupStatistics, jeffreys_selectivity


class FusionProfilingBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    maximum_unique_groups: int = Field(gt=0)
    maximum_records_per_group: int = Field(gt=0)
    maximum_logical_requests: int = Field(gt=0)
    maximum_http_attempts: int = Field(gt=0)
    maximum_estimated_expenditure_usd: float = Field(gt=0.0)
    max_retries: int = Field(ge=0)


class FusionProfilingAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    group_id: str
    predicate_ids: tuple[str, ...]
    records: int = Field(gt=0)
    estimated_cost_per_request_usd: float = Field(gt=0.0)
    optimistic_execution_saving_per_reuse_usd: float = Field(ge=0.0)
    required_for_plan_ids: tuple[str, ...]

    @property
    def estimated_profiling_cost_usd(self) -> float:
        return self.records * self.estimated_cost_per_request_usd


class FusionProfilingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_actions: tuple[FusionProfilingAction, ...]
    skipped_actions: Mapping[str, str]
    logical_requests: int = Field(ge=0)
    retry_inclusive_http_attempts: int = Field(ge=0)
    estimated_expenditure_usd: float = Field(ge=0.0)
    stop_reason: str
    policy_version: str = "m5-deterministic-net-value-v1"
    future_records_inspected: bool = False


def profiling_decision_identity(
    *,
    dataset_fingerprint: str,
    record_id: str,
    group: FusedOperator,
    model_version: str,
) -> str:
    payload = {
        "dataset_fingerprint": dataset_fingerprint,
        "record_id": record_id,
        "predicate_ids": list(group.predicate_ids),
        "predicate_hashes": dict(group.predicate_hashes),
        "fusion_prompt_hash": group.fusion_prompt_hash,
        "fusion_mode": group.fusion_mode,
        "model_version": model_version,
        "threshold": group.threshold,
        "execution_parameters": dict(group.execution_parameters),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def choose_fusion_profiling_actions(
    actions: Sequence[FusionProfilingAction],
    *,
    budget: FusionProfilingBudget,
    reuse_horizon: int,
    already_measured_group_ids: Iterable[str] = (),
) -> FusionProfilingDecision:
    """Rank actions using only declared prospective economics; never future outcomes."""
    if reuse_horizon <= 0:
        raise ValueError("reuse_horizon must be positive")
    measured = set(already_measured_group_ids)
    unique = {action.group_id: action for action in actions}
    if len(unique) != len(actions):
        raise ValueError("profiling actions must have unique group IDs")
    ranked = sorted(
        actions,
        key=lambda action: (
            -(
                reuse_horizon * action.optimistic_execution_saving_per_reuse_usd
                - action.estimated_profiling_cost_usd
            ),
            action.group_id,
        ),
    )
    selected: list[FusionProfilingAction] = []
    skipped: dict[str, str] = {}
    requests = 0
    attempts = 0
    expenditure = 0.0
    for action in ranked:
        if action.group_id in measured:
            skipped[action.group_id] = "compatible_measurement_already_available"
            continue
        net_value = (
            reuse_horizon * action.optimistic_execution_saving_per_reuse_usd
            - action.estimated_profiling_cost_usd
        )
        if net_value <= 0:
            skipped[action.group_id] = "optimistic_net_value_nonpositive"
            continue
        if action.records > budget.maximum_records_per_group:
            skipped[action.group_id] = "records_per_group_limit"
            continue
        proposed_requests = requests + action.records
        proposed_attempts = attempts + action.records * (budget.max_retries + 1)
        proposed_cost = expenditure + action.estimated_profiling_cost_usd
        if len(selected) >= budget.maximum_unique_groups:
            skipped[action.group_id] = "unique_group_limit"
        elif proposed_requests > budget.maximum_logical_requests:
            skipped[action.group_id] = "logical_request_limit"
        elif proposed_attempts > budget.maximum_http_attempts:
            skipped[action.group_id] = "http_attempt_limit"
        elif proposed_cost > budget.maximum_estimated_expenditure_usd:
            skipped[action.group_id] = "estimated_expenditure_limit"
        else:
            selected.append(action)
            requests = proposed_requests
            attempts = proposed_attempts
            expenditure = proposed_cost
    if selected:
        stop_reason = "selected_positive_net_value_actions_within_all_budgets"
    elif actions:
        stop_reason = "no_eligible_positive_net_value_action"
    else:
        stop_reason = "no_candidate_fused_groups"
    return FusionProfilingDecision(
        selected_actions=tuple(selected),
        skipped_actions=skipped,
        logical_requests=requests,
        retry_inclusive_http_attempts=attempts,
        estimated_expenditure_usd=expenditure,
        stop_reason=stop_reason,
    )


def group_statistics_from_traces(
    group: FusedOperator,
    traces: Sequence[FusedDecisionTrace],
    *,
    input_price_per_million: float,
    output_price_per_million: float,
    evidence_artifact: str,
) -> GroupStatistics:
    relevant = [trace for trace in traces if trace.group_id == group.group_id]
    if not relevant:
        raise ValueError("no traces found for fused group")
    if any(trace.fusion_prompt_hash != group.fusion_prompt_hash for trace in relevant):
        raise ValueError("trace prompt hash differs from fused group")
    input_usage = [trace.input_tokens for trace in relevant]
    output_usage = [trace.output_tokens for trace in relevant]
    mean_input = (
        sum(value for value in input_usage if value is not None) / len(input_usage)
        if all(value is not None for value in input_usage)
        else None
    )
    mean_output = (
        sum(value for value in output_usage if value is not None) / len(output_usage)
        if all(value is not None for value in output_usage)
        else None
    )
    mean_cost = (
        (
            mean_input * input_price_per_million
            + mean_output * output_price_per_million
        )
        / 1_000_000
        if mean_input is not None and mean_output is not None
        else None
    )
    passing = sum(trace.passed for trace in relevant)
    return GroupStatistics(
        group_id=group.group_id,
        predicate_ids=group.predicate_ids,
        fusion_mode=group.fusion_mode,
        sample_count=len(relevant),
        passing_count=passing,
        empirical_selectivity=passing / len(relevant),
        smoothed_selectivity=jeffreys_selectivity(passing, len(relevant)),
        mean_input_tokens=mean_input,
        mean_output_tokens=mean_output,
        mean_cost_usd=mean_cost,
        mean_latency_seconds=sum(
            trace.request_latency_seconds for trace in relevant
        )
        / len(relevant),
        provider_error_rate=0.0,
        evidence_source="measured_fused",
        evidence_artifact=evidence_artifact,
    )
