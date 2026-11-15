from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from semanticflow.optimizer.fusion_planner import FusionPhysicalPlan
from semanticflow.optimizer.fusion_quality import (
    BinaryQualityMetrics,
    FusionQualityGate,
    compare_binary_decisions,
    evaluate_quality_gate,
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def replay_measured_plan(
    plan: FusionPhysicalPlan,
    *,
    record_ids: Sequence[str],
    separate_traces: Mapping[tuple[str, str], Mapping[str, Any]],
    fused_traces: Mapping[tuple[str, str], Mapping[str, Any]],
    input_price_per_million: float,
    output_price_per_million: float,
) -> dict[str, Any]:
    """Replay only real measured group outcomes; missing cells make a plan ineligible."""
    reference: list[bool] = []
    candidate: list[bool] = []
    input_tokens = output_tokens = requests = 0
    missing: set[str] = set()
    stage_inputs = [0] * len(plan.groups)
    for group in plan.groups:
        if len(group.predicate_ids) > 1 and not any(
            (record_id, group.group_id) in fused_traces for record_id in record_ids
        ):
            missing.add(f"group_evidence:{group.group_id}")
    for record_id in record_ids:
        reference_cells = [
            separate_traces.get((record_id, predicate_id))
            for predicate_id in plan.logical_predicate_ids
        ]
        if any(cell is None for cell in reference_cells):
            missing.add(f"reference:{record_id}")
            continue
        reference.append(all(bool(cell["passed"]) for cell in reference_cells if cell))
        passed = True
        for stage, group in enumerate(plan.groups):
            if not passed:
                break
            stage_inputs[stage] += 1
            key = (record_id, group.group_id)
            if len(group.predicate_ids) == 1:
                trace = separate_traces.get((record_id, group.predicate_ids[0]))
            else:
                trace = fused_traces.get(key)
            if trace is None:
                missing.add(f"{group.group_id}:{record_id}")
                passed = False
                break
            requests += 1
            input_tokens += int(trace.get("input_tokens") or 0)
            output_tokens += int(trace.get("output_tokens") or 0)
            passed = bool(trace["passed"])
        candidate.append(passed)
    complete = not missing and len(candidate) == len(record_ids)
    metrics: BinaryQualityMetrics | None = (
        compare_binary_decisions(reference, candidate) if complete else None
    )
    cost = (
        input_tokens * input_price_per_million
        + output_tokens * output_price_per_million
    ) / 1_000_000
    return {
        "plan_id": plan.plan_id,
        "canonical_grouping": [list(group) for group in plan.canonical_grouping],
        "contains_fusion": plan.contains_fusion,
        "eligible_measured_replay": complete,
        "missing_measurements": sorted(missing),
        "provider_requests": requests,
        "stage_input_cardinalities": stage_inputs,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": cost if complete else None,
        "quality": metrics.model_dump(mode="json") if metrics else None,
        "quality_evidence": (
            "complete physical-plan replay from real fused and separate calls"
            if complete and plan.contains_fusion
            else "separate-reference replay" if complete else "incomplete"
        ),
        "semantic_accuracy_established": False,
    }


def apply_quality_gate_to_replay(
    replay: Mapping[str, Any], gate: FusionQualityGate
) -> dict[str, Any]:
    metrics = (
        BinaryQualityMetrics.model_validate(replay["quality"])
        if replay.get("quality") is not None
        else None
    )
    decision = evaluate_quality_gate(
        metrics,
        gate,
        evidence_scope="complete_plan" if metrics else "missing",
        real_fused_measurement=bool(replay.get("contains_fusion")) and metrics is not None,
    )
    return decision.model_dump(mode="json")


def summarize_measured_candidates(
    plans: Sequence[FusionPhysicalPlan],
    *,
    record_ids: Sequence[str],
    separate_traces: Mapping[tuple[str, str], Mapping[str, Any]],
    fused_traces: Mapping[tuple[str, str], Mapping[str, Any]],
    quality_gate: FusionQualityGate,
    input_price_per_million: float,
    output_price_per_million: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for plan in plans:
        replay = replay_measured_plan(
            plan,
            record_ids=record_ids,
            separate_traces=separate_traces,
            fused_traces=fused_traces,
            input_price_per_million=input_price_per_million,
            output_price_per_million=output_price_per_million,
        )
        replay["quality_gate"] = apply_quality_gate_to_replay(replay, quality_gate)
        output.append(replay)
    return output


def existing_adaptive_order_from_reference(
    *,
    traces: Sequence[Mapping[str, Any]],
    predicate_ids: Sequence[str],
    original_order: Sequence[str],
    review_ids: Sequence[str],
    evaluation_record_count: int,
    optimizer_config_path: str | Path = "configs/optimizer.milestone_4.yaml",
    expected_executions: int = 10,
) -> dict[str, Any]:
    """Run the unchanged existing adaptive controller over pilot reference evidence."""
    from semanticflow.optimizer.adaptive import simulate_adaptive_profiling
    from semanticflow.optimizer.matrix import ProfileMatrix
    from semanticflow.optimizer.models import load_optimizer_settings

    settings = load_optimizer_settings(optimizer_config_path)
    relevant = [trace for trace in traces if trace["predicate_id"] in predicate_ids]
    question_hashes = {
        predicate_id: next(
            str(trace["question_hash"])
            for trace in relevant
            if trace["predicate_id"] == predicate_id
        )
        for predicate_id in predicate_ids
    }
    matrix = ProfileMatrix.from_traces(
        relevant,
        predicate_ids,
        review_ids=review_ids,
        question_hashes=question_hashes,
    )
    costs = [
        (float(trace.get("input_tokens") or 0) * 0.042) / 1_000_000
        for trace in relevant
    ]
    result = simulate_adaptive_profiling(
        matrix,
        estimator_name="correlation_aware",
        original_order=original_order,
        objective=settings.planner.objective,
        evaluation_record_count=evaluation_record_count,
        question_hashes=question_hashes,
        settings=settings.adaptive,
        bootstrap=settings.bootstrap,
        correlation=settings.correlation,
        input_price_per_million=0.042,
        output_price_per_million=0.0,
        expected_executions=expected_executions,
        tie_relative_tolerance=settings.planner.tie_relative_tolerance,
        economics=settings.economics,
        estimated_cost_per_decision=sum(costs) / len(costs),
        statistics_available=False,
    )
    stop = result.get("stopping_checkpoint")
    return {
        "selected_order": result["selected_order"],
        "profiled_reviews": (stop or {}).get("profiling_reviews", 0),
        "profiling_decisions": result["profiling_decisions_used"],
        "profiling_cost_usd": (stop or {}).get(
            "profiling_cost_usd_offline_estimate", 0.0
        ),
        "stop_reason": (stop or {}).get(
            "reasons", [result["selected_order_source"]]
        ),
        "controller_schema_version": result["schema_version"],
        "source_config": str(optimizer_config_path),
    }
