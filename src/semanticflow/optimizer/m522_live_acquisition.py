from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from semanticflow.optimizer.fusion import FusedDecision, build_fused_operator
from semanticflow.optimizer.m522_action_universe import (
    AuthorizedLiveAction,
    PhysicalOperatorType,
)
from semanticflow.providers.base import (
    InferenceProvider,
    RequestBudget,
    ResponseValidationError,
)
from semanticflow.query.models import PredicateSpec

INPUT_PRICE_PER_MILLION = 0.042


class AmbiguousDispatchError(RuntimeError):
    def __init__(self, message: str, *, attempts: int = 1) -> None:
        super().__init__(message)
        self.attempts = attempts


class LiveProvider(Protocol):
    @property
    def model_identifier(self) -> str: ...

    async def decide_fused(
        self, record_text: str, operator: object, budget: RequestBudget
    ) -> FusedDecision: ...


class AcquisitionReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "m5.2.2-acquisition-receipt-v1"
    action_id: str
    request_identity: str
    response_identity: str
    workload_id: str
    record_id: str
    measurement_type: str
    track: str
    expected_question_ids: tuple[str, ...]
    probabilities: dict[str, float]
    decisions: dict[str, bool]
    resolved_model: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    http_attempts: int = Field(ge=1)
    retries: int = Field(ge=0)
    latency_seconds: float = Field(ge=0)
    token_priced_cost_usd: float = Field(ge=0)


def validate_named_response(
    expected_question_ids: tuple[str, ...], probabilities: Mapping[str, float]
) -> dict[str, float]:
    if len(expected_question_ids) != len(set(expected_question_ids)):
        raise ValueError("expected named decisions contain duplicates")
    if set(probabilities) != set(expected_question_ids):
        missing = sorted(set(expected_question_ids) - set(probabilities))
        extra = sorted(set(probabilities) - set(expected_question_ids))
        raise ValueError(f"named decisions differ; missing={missing}, extra={extra}")
    result: dict[str, float] = {}
    for name in expected_question_ids:
        value = float(probabilities[name])
        if not 0 <= value <= 1:
            raise ValueError(f"probability outside [0,1] for {name}")
        result[name] = value
    return result


class LiveAcquisitionAdapter:
    """Bridge to the existing provider interface; it does not own authorization."""

    def __init__(self, provider: InferenceProvider | LiveProvider) -> None:
        self.provider = provider

    async def acquire(self, action: AuthorizedLiveAction, record_text: str) -> AcquisitionReceipt:
        if self.provider.model_identifier != action.requested_model:
            raise ValueError("provider model differs from authorized action")
        predicates = tuple(
            PredicateSpec(id=predicate_id, instructions=instructions)
            for predicate_id, instructions in zip(
                action.predicate_ids, action.predicate_instructions, strict=True
            )
        )
        mode = (
            "single"
            if action.physical_operator == PhysicalOperatorType.SINGLE
            else "multi_decision"
        )
        operator = build_fused_operator(
            predicates,
            mode=mode,
            requested_model=action.requested_model,
            threshold=action.threshold,
            execution_parameters={"cache": False, "m522_action_id": action.action_id},
        )
        decision = await self.provider.decide_fused(
            record_text,
            operator,
            RequestBudget(action.maximum_attempts),
        )

        def invalid_response(message: str) -> ResponseValidationError:
            return ResponseValidationError(
                message,
                attempts=decision.request_attempts,
                failed_attempts=decision.request_attempts,
                retries=decision.retries,
            )

        if decision.model_identifier != action.requested_model:
            raise invalid_response("resolved provider model differs from frozen model")
        if decision.input_tokens is None or decision.output_tokens is None:
            raise invalid_response("Pilot A requires complete provider usage metadata")
        if decision.request_attempts < 1 or decision.request_attempts > action.maximum_attempts:
            raise invalid_response("provider attempt count violates authorized action")
        if decision.retries != decision.request_attempts - 1:
            raise invalid_response("provider retry metadata is inconsistent")
        if mode == "single":
            if decision.group_probability is None or decision.predicate_probabilities:
                raise invalid_response("single response has invalid probability shape")
            raw = {action.predicate_ids[0]: decision.group_probability}
        else:
            if decision.group_probability is not None:
                raise invalid_response("multi-decision response cannot contain group probability")
            raw = dict(decision.predicate_probabilities)
        try:
            probabilities = validate_named_response(action.predicate_ids, raw)
        except ValueError as exc:
            raise invalid_response(str(exc)) from exc
        decisions = {
            name: probability >= action.threshold for name, probability in probabilities.items()
        }
        response_payload = {
            "action_id": action.action_id,
            "probabilities": probabilities,
            "model": decision.model_identifier,
            "input_tokens": decision.input_tokens,
            "output_tokens": decision.output_tokens,
            "attempts": decision.request_attempts,
        }
        response_identity = hashlib.sha256(
            json.dumps(response_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return AcquisitionReceipt(
            action_id=action.action_id,
            request_identity=action.action_id,
            response_identity=response_identity,
            workload_id=action.workload_id,
            record_id=action.record_id,
            measurement_type=action.measurement_type.value,
            track=action.track.value,
            expected_question_ids=action.predicate_ids,
            probabilities=probabilities,
            decisions=decisions,
            resolved_model=decision.model_identifier,
            input_tokens=decision.input_tokens,
            output_tokens=decision.output_tokens,
            http_attempts=decision.request_attempts,
            retries=decision.retries,
            latency_seconds=decision.latency_seconds,
            token_priced_cost_usd=(decision.input_tokens / 1_000_000 * INPUT_PRICE_PER_MILLION),
        )


class DeterministicMockProvider:
    """Test-only provider using record-text keyed probabilities; never valid live auth."""

    name = "m522-mock-jev"

    def __init__(
        self,
        *,
        model: str,
        scenario: str,
        attempts: int = 1,
        fault_by_text_hash: Mapping[str, str] | None = None,
    ) -> None:
        self._model = model
        self.scenario = scenario
        self.attempts = attempts
        self.fault_by_text_hash = dict(fault_by_text_hash or {})

    @property
    def model_identifier(self) -> str:
        return self._model

    async def decide_fused(
        self, record_text: str, operator: object, budget: RequestBudget
    ) -> FusedDecision:
        del budget
        from semanticflow.optimizer.fusion import FusedOperator

        assert isinstance(operator, FusedOperator)
        text_hash = hashlib.sha256(record_text.encode()).hexdigest()
        fault = self.fault_by_text_hash.get(text_hash)
        if fault == "ambiguous":
            raise AmbiguousDispatchError("mock timeout after possible dispatch")
        if fault == "before_dispatch":
            raise ConnectionError("mock failure before dispatch")
        base = int(text_hash[:8], 16) / 0xFFFFFFFF
        if self.scenario == "informative":
            probability = 0.5 + (base - 0.5) * 0.18
        elif self.scenario == "uninformative":
            probability = 0.05 + base * 0.15
        elif self.scenario == "budget_pressure":
            probability = 0.48 + base * 0.04
        else:
            probability = base
        values = {
            predicate_id: min(1.0, max(0.0, probability + index * 0.01))
            for index, predicate_id in enumerate(operator.predicate_ids)
        }
        tokens = 330 + 110 * len(operator.predicate_ids) + len(record_text) // 4
        if operator.fusion_mode == "single":
            group_probability = values[operator.predicate_ids[0]]
            values = {}
        else:
            group_probability = None
        return FusedDecision(
            group_probability=group_probability,
            predicate_probabilities=values,
            model_identifier=self._model,
            input_tokens=tokens,
            output_tokens=25 * len(operator.predicate_ids),
            latency_seconds=0.01,
            request_attempts=self.attempts,
            successful_requests=1,
            failed_attempts=self.attempts - 1,
            retries=self.attempts - 1,
            metadata={"mock_scenario": self.scenario},
        )
