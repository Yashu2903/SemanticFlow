from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field


class HybridProfilingAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    group_id: str
    execution_mode: str
    predicate_ids: tuple[str, ...]
    records: int = Field(gt=0)
    estimated_request_cost_usd: float = Field(gt=0)
    optimistic_saving_per_reuse_usd: float = Field(ge=0)
    impact_plan_ids: tuple[str, ...]


class HybridProfilingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    actions: tuple[HybridProfilingAction, ...]
    skipped: Mapping[str, str]
    logical_requests: int = Field(ge=0)
    maximum_http_attempts: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0)
    stop_reason: str
    future_records_inspected: bool = False


def choose_hybrid_profiling_actions(
    actions: Sequence[HybridProfilingAction],
    *,
    reuse_horizon: int,
    measured_group_ids: Sequence[str],
    maximum_unique_groups: int,
    maximum_records_per_group: int,
    maximum_logical_requests: int,
    maximum_http_attempts: int,
    maximum_estimated_cost_usd: float,
    max_retries: int,
) -> HybridProfilingDecision:
    if reuse_horizon <= 0 or maximum_unique_groups <= 0:
        raise ValueError("reuse horizon and unique-group limit must be positive")
    if len({action.group_id for action in actions}) != len(actions):
        raise ValueError("profiling groups must be unique")
    measured = set(measured_group_ids)
    ranked = sorted(
        actions,
        key=lambda action: (
            -(
                reuse_horizon * action.optimistic_saving_per_reuse_usd
                - action.records * action.estimated_request_cost_usd
            ),
            action.group_id,
        ),
    )
    chosen: list[HybridProfilingAction] = []
    skipped: dict[str, str] = {}
    logical = attempts = 0
    cost = 0.0
    for action in ranked:
        acquisition = action.records * action.estimated_request_cost_usd
        net = reuse_horizon * action.optimistic_saving_per_reuse_usd - acquisition
        proposed_logical = logical + action.records
        proposed_attempts = attempts + action.records * (max_retries + 1)
        if action.group_id in measured:
            skipped[action.group_id] = "compatible_measurement_already_paid"
        elif net <= 0:
            skipped[action.group_id] = "optimistic_net_value_nonpositive"
        elif action.records > maximum_records_per_group:
            skipped[action.group_id] = "records_per_group_limit"
        elif len(chosen) >= maximum_unique_groups:
            skipped[action.group_id] = "unique_group_limit"
        elif proposed_logical > maximum_logical_requests:
            skipped[action.group_id] = "logical_request_limit"
        elif proposed_attempts > maximum_http_attempts:
            skipped[action.group_id] = "http_attempt_limit"
        elif cost + acquisition > maximum_estimated_cost_usd:
            skipped[action.group_id] = "estimated_cost_limit"
        else:
            chosen.append(action)
            logical = proposed_logical
            attempts = proposed_attempts
            cost += acquisition
    return HybridProfilingDecision(
        actions=tuple(chosen),
        skipped=skipped,
        logical_requests=logical,
        maximum_http_attempts=attempts,
        estimated_cost_usd=cost,
        stop_reason=(
            "selected_positive_net_value_shared_group_measurements"
            if chosen
            else "no_economically_justified_measurement"
        ),
    )
