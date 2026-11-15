from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from semanticflow.query.models import PredicateSpec

FusionMode = Literal["single", "conjunctive_binary", "multi_decision"]

FUSION_PROMPT_TEMPLATE_VERSION = "m5-conjunction-v1"
FUSION_RESULT_SCHEMA_VERSION = "m5-fused-decision-v1"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def predicate_hash(predicate: PredicateSpec) -> str:
    return sha256_text(f"noul\0{predicate.id}\0{predicate.instructions}")


def build_conjunctive_prompt(predicates: Sequence[PredicateSpec]) -> str:
    """Build a stable provider-visible conjunction without paraphrasing predicates."""
    if len(predicates) < 2:
        raise ValueError("a conjunctive fused prompt requires at least two predicates")
    conditions = "\n".join(
        f"Condition {index} [{predicate.id}]: {predicate.instructions}"
        for index, predicate in enumerate(predicates, start=1)
    )
    return (
        "Evaluate all conditions below against the supplied record. A positive answer requires "
        "every condition to be satisfied; if any condition is not satisfied, answer negatively.\n\n"
        f"{conditions}\n\nAre all conditions satisfied?"
    )


class ProviderFusionCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    conjunctive_binary: bool
    multi_decision: bool
    maximum_questions_per_request: int | None = Field(default=None, ge=1)
    evidence: str
    live_validated: bool = False

    def supports(self, mode: FusionMode, predicate_count: int) -> bool:
        if mode == "single":
            return predicate_count == 1
        if self.maximum_questions_per_request is not None and (
            mode == "multi_decision" and predicate_count > self.maximum_questions_per_request
        ):
            return False
        return self.conjunctive_binary if mode == "conjunctive_binary" else self.multi_decision


class FusedOperator(BaseModel):
    """One physical provider request evaluating one or more logical predicates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    group_id: str
    predicate_ids: tuple[str, ...]
    source_instructions: tuple[str, ...]
    fusion_mode: FusionMode
    generated_prompt: str
    prompt_template_version: str = FUSION_PROMPT_TEMPLATE_VERSION
    predicate_hashes: Mapping[str, str]
    fusion_prompt_hash: str
    requested_model: str
    resolved_model: str | None = None
    threshold: float = Field(ge=0.0, le=1.0)
    execution_parameters: Mapping[str, Any] = Field(default_factory=dict)
    result_schema_version: str = FUSION_RESULT_SCHEMA_VERSION

    @model_validator(mode="after")
    def validate_operator(self) -> FusedOperator:
        if not self.predicate_ids:
            raise ValueError("a fused operator cannot be empty")
        if len(set(self.predicate_ids)) != len(self.predicate_ids):
            raise ValueError("a fused operator cannot duplicate predicates")
        if len(self.predicate_ids) != len(self.source_instructions):
            raise ValueError("predicate IDs and instructions must have equal length")
        if set(self.predicate_hashes) != set(self.predicate_ids):
            raise ValueError("predicate hashes must cover the group exactly")
        if self.fusion_mode == "single" and len(self.predicate_ids) != 1:
            raise ValueError("single mode requires exactly one predicate")
        if self.fusion_mode != "single" and len(self.predicate_ids) < 2:
            raise ValueError("fused modes require at least two predicates")
        return self

    def provider_questions(self) -> dict[str, dict[str, str]]:
        if self.fusion_mode in {"single", "conjunctive_binary"}:
            return {
                self.group_id: {
                    "type": "noul",
                    "instructions": self.generated_prompt,
                }
            }
        return {
            predicate_id: {"type": "noul", "instructions": instructions}
            for predicate_id, instructions in zip(
                self.predicate_ids, self.source_instructions, strict=True
            )
        }

    def canonical_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"resolved_model"})


def build_fused_operator(
    predicates: Sequence[PredicateSpec],
    *,
    mode: FusionMode,
    requested_model: str,
    threshold: float,
    execution_parameters: Mapping[str, Any] | None = None,
) -> FusedOperator:
    if not predicates:
        raise ValueError("cannot build an empty fused operator")
    ids = tuple(predicate.id for predicate in predicates)
    instructions = tuple(predicate.instructions for predicate in predicates)
    if mode == "single":
        if len(predicates) != 1:
            raise ValueError("single mode requires one predicate")
        generated_prompt = predicates[0].instructions
    elif mode == "conjunctive_binary":
        generated_prompt = build_conjunctive_prompt(predicates)
    elif mode == "multi_decision":
        if len(predicates) < 2:
            raise ValueError("multi-decision mode requires at least two predicates")
        generated_prompt = json.dumps(
            {
                "semantics": "all predicate questions are submitted in one provider request",
                "questions": [
                    {"id": predicate.id, "instructions": predicate.instructions}
                    for predicate in predicates
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    else:  # pragma: no cover - protected by the type and validation boundary
        raise ValueError(f"unsupported fusion mode: {mode}")
    hash_payload = {
        "mode": mode,
        "prompt_template_version": FUSION_PROMPT_TEMPLATE_VERSION,
        "questions": (
            [{"id": "__group__", "instructions": generated_prompt}]
            if mode != "multi_decision"
            else [
                {"id": predicate.id, "instructions": predicate.instructions}
                for predicate in predicates
            ]
        ),
    }
    fusion_hash = sha256_text(
        json.dumps(hash_payload, sort_keys=True, separators=(",", ":"))
    )
    group_id = f"fusion_{fusion_hash[:16]}"
    return FusedOperator(
        group_id=group_id,
        predicate_ids=ids,
        source_instructions=instructions,
        fusion_mode=mode,
        generated_prompt=generated_prompt,
        predicate_hashes={predicate.id: predicate_hash(predicate) for predicate in predicates},
        fusion_prompt_hash=fusion_hash,
        requested_model=requested_model,
        threshold=threshold,
        execution_parameters=dict(execution_parameters or {}),
    )


@dataclass(frozen=True, slots=True)
class FusedDecision:
    group_probability: float | None
    predicate_probabilities: Mapping[str, float]
    model_identifier: str
    input_tokens: int | None
    output_tokens: int | None
    latency_seconds: float
    request_attempts: int = 1
    successful_requests: int = 1
    failed_attempts: int = 0
    retries: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        probabilities = list(self.predicate_probabilities.values())
        if self.group_probability is not None:
            probabilities.append(self.group_probability)
        if not probabilities or any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("fused probabilities must be present and between 0 and 1")
        if self.input_tokens is not None and self.input_tokens < 0:
            raise ValueError("input_tokens cannot be negative")
        if self.output_tokens is not None and self.output_tokens < 0:
            raise ValueError("output_tokens cannot be negative")
        if self.latency_seconds < 0:
            raise ValueError("latency_seconds cannot be negative")

    def passes(self, operator: FusedOperator) -> bool:
        if operator.fusion_mode in {"single", "conjunctive_binary"}:
            if self.group_probability is None:
                raise ValueError("group-level mode requires a group probability")
            return self.group_probability >= operator.threshold
        if set(self.predicate_probabilities) != set(operator.predicate_ids):
            raise ValueError("multi-decision result does not cover the group")
        return all(
            self.predicate_probabilities[predicate_id] >= operator.threshold
            for predicate_id in operator.predicate_ids
        )
