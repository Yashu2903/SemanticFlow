from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Literal

from semanticflow.optimizer.fusion import FusedOperator, predicate_hash, sha256_text
from semanticflow.query.models import PredicateSpec

ConjunctiveTemplate = Literal[
    "m5.1-original-conjunction-v1",
    "m5.1-explicit-conjunction-v1",
    "m5.1-structured-checklist-v1",
]


def build_conjunctive_template_prompt(
    predicates: Sequence[PredicateSpec], template: ConjunctiveTemplate
) -> str:
    if len(predicates) < 2:
        raise ValueError("conjunctive templates require at least two predicates")
    if template == "m5.1-original-conjunction-v1":
        joined = " AND ".join(f"({predicate.instructions})" for predicate in predicates)
        return f"Does the supplied record satisfy {joined}?"
    conditions = "\n".join(
        f"Condition {index} [{predicate.id}]: {predicate.instructions}"
        for index, predicate in enumerate(predicates, 1)
    )
    if template == "m5.1-explicit-conjunction-v1":
        return (
            "Evaluate each condition independently against the same supplied record. "
            "Answer positively only when every condition is true.\n\n"
            f"{conditions}\n\nAre all listed conditions true?"
        )
    if template == "m5.1-structured-checklist-v1":
        return (
            "Use the labeled conditions as a checklist. Do not merge or omit a condition. "
            "The single final answer is positive only if every checklist item holds.\n\n"
            f"{conditions}\n\nFinal conjunction: do all checklist items hold?"
        )
    raise ValueError(f"unsupported template: {template}")


def build_template_operator(
    predicates: Sequence[PredicateSpec],
    *,
    template: ConjunctiveTemplate,
    requested_model: str,
    threshold: float,
    execution_parameters: Mapping[str, object] | None = None,
) -> FusedOperator:
    prompt = build_conjunctive_template_prompt(predicates, template)
    payload = {
        "mode": "conjunctive_binary",
        "prompt_template_version": template,
        "questions": [{"id": "__group__", "instructions": prompt}],
    }
    prompt_hash = sha256_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
    )
    return FusedOperator(
        group_id=f"fusion_{prompt_hash[:16]}",
        predicate_ids=tuple(predicate.id for predicate in predicates),
        source_instructions=tuple(predicate.instructions for predicate in predicates),
        fusion_mode="conjunctive_binary",
        generated_prompt=prompt,
        prompt_template_version=template,
        predicate_hashes={predicate.id: predicate_hash(predicate) for predicate in predicates},
        fusion_prompt_hash=prompt_hash,
        requested_model=requested_model,
        threshold=threshold,
        execution_parameters=dict(execution_parameters or {}),
    )
