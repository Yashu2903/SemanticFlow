from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, ConfigDict, Field

from semanticflow.optimizer.m522_session import AppendOnlyEventLog, rebuild_session


class InstrumentationAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "m5.2.2-instrumentation-analysis-v1"
    session_status: str
    physical_requests: int = Field(ge=0)
    http_attempts: int = Field(ge=0)
    named_question_evaluations: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    experimental_cost_proxy_usd: float = Field(ge=0)
    representative_actions: int = Field(ge=0)
    discovery_screens: int = Field(ge=0)
    discovery_followups: int = Field(ge=0)
    near_threshold_screens: int = Field(ge=0)
    diagnostic_disagreements: int = Field(ge=0)
    invalid_responses: int = Field(ge=0)
    restart_events: int = Field(ge=0)
    formal_quality_certified: bool = False
    semantic_accuracy_established: bool = False


def analyze_session(
    event_log: AppendOnlyEventLog, *, threshold: float = 0.5
) -> InstrumentationAnalysis:
    events = event_log.read()
    state = rebuild_session(events)
    kinds = Counter(str(receipt["measurement_type"]) for receipt in state.receipts)
    near = 0
    disagreements = 0
    for receipt in state.receipts:
        values = [float(value) for value in receipt["probabilities"].values()]
        near += any(abs(value - threshold) <= 0.05 for value in values)
        decisions = list(receipt["decisions"].values())
        disagreements += bool(decisions) and any(decisions) and not all(decisions)
    return InstrumentationAnalysis(
        session_status=state.status.value,
        physical_requests=len(state.budget.completed_action_ids),
        http_attempts=state.budget.actual_http_attempts,
        named_question_evaluations=state.budget.named_question_evaluations,
        input_tokens=state.budget.input_tokens,
        output_tokens=state.budget.output_tokens,
        experimental_cost_proxy_usd=state.budget.actual_cost_proxy_usd,
        representative_actions=sum(
            receipt["track"] == "certification" for receipt in state.receipts
        ),
        discovery_screens=kinds["discovery_screen"],
        discovery_followups=kinds["discovery_followup"],
        near_threshold_screens=near,
        diagnostic_disagreements=disagreements,
        invalid_responses=sum(event.event_type == "action_failed" for event in events),
        restart_events=sum(
            event.event_type == "status_changed" and event.payload.get("reason") == "reconciled"
            for event in events
        ),
    )
