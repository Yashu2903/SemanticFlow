from __future__ import annotations

import json
from pathlib import Path

import pytest

from semanticflow.dataset.loader import ReviewRecord
from semanticflow.executor.fused import FusedExecutor
from semanticflow.optimizer.fusion import (
    FusedDecision,
    ProviderFusionCapabilities,
    build_fused_operator,
)
from semanticflow.optimizer.fusion_analysis import replay_measured_plan
from semanticflow.optimizer.fusion_cost import (
    GroupStatistics,
    estimate_fusion_plan_cost,
    fusion_economics,
)
from semanticflow.optimizer.fusion_experiment import (
    STAGE_B_AUTHORIZATION,
    build_authorization_manifest,
    build_pilot_budget,
    load_fusion_protocol,
    load_fusion_suite,
    load_yaml,
    validate_stage_b_launch,
)
from semanticflow.optimizer.fusion_planner import (
    build_physical_plan,
    enumerate_fusion_plans,
    ordered_set_partitions,
    select_quality_constrained_plan,
)
from semanticflow.optimizer.fusion_profile import (
    FusionProfilingAction,
    FusionProfilingBudget,
    choose_fusion_profiling_actions,
)
from semanticflow.optimizer.fusion_quality import (
    FusionQualityGate,
    compare_binary_decisions,
    evaluate_quality_gate,
)
from semanticflow.providers.mock import MockProvider
from semanticflow.query.models import PredicateSpec


def predicates(count: int) -> list[PredicateSpec]:
    return [PredicateSpec(id=f"p{i}", instructions=f"Condition {i}?") for i in range(count)]


def capabilities() -> ProviderFusionCapabilities:
    return ProviderFusionCapabilities(
        provider="test",
        conjunctive_binary=True,
        multi_decision=True,
        evidence="test",
    )


def plans(count: int):
    return enumerate_fusion_plans(
        predicates(count),
        workload_id="w",
        logical_query_id="q",
        requested_model="m",
        threshold=0.5,
        fusion_mode="conjunctive_binary",
        capabilities=capabilities(),
        configuration_fingerprint="cfg",
        maximum_group_size=4,
    )


@pytest.mark.parametrize(("count", "expected"), [(2, 3), (3, 13), (4, 75)])
def test_ordered_set_partition_counts_and_coverage(count: int, expected: int) -> None:
    candidates = plans(count)
    assert len(candidates) == expected
    for candidate in candidates:
        flattened = [item for group in candidate.groups for item in group.predicate_ids]
        assert sorted(flattened) == [f"p{i}" for i in range(count)]
        assert len(flattened) == len(set(flattened))


def test_deterministic_prompts_and_plan_hashes() -> None:
    first = build_fused_operator(
        predicates(2), mode="conjunctive_binary", requested_model="m", threshold=0.5
    )
    second = build_fused_operator(
        predicates(2), mode="conjunctive_binary", requested_model="m", threshold=0.5
    )
    assert first == second
    assert "every condition" in first.generated_prompt
    assert first.provider_questions()[first.group_id]["instructions"] == first.generated_prompt
    assert plans(3) == plans(3)


def test_multi_decision_requires_complete_predicate_results() -> None:
    operator = build_fused_operator(
        predicates(2), mode="multi_decision", requested_model="m", threshold=0.5
    )
    decision = FusedDecision(
        group_probability=None,
        predicate_probabilities={"p0": 0.8},
        model_identifier="m",
        input_tokens=1,
        output_tokens=1,
        latency_seconds=0,
    )
    with pytest.raises(ValueError, match="does not cover"):
        decision.passes(operator)


@pytest.mark.asyncio
async def test_fused_executor_uses_one_request_per_group_and_early_rejects() -> None:
    p = predicates(3)
    first = build_fused_operator(
        p[:2], mode="conjunctive_binary", requested_model="m", threshold=0.5
    )
    last = build_fused_operator(p[2:], mode="single", requested_model="m", threshold=0.5)
    plan = build_physical_plan(
        workload_id="w",
        logical_query_id="q",
        logical_predicate_ids=[item.id for item in p],
        groups=[first, last],
        configuration_fingerprint="cfg",
    )
    provider = MockProvider(
        probabilities={("a", "p2"): 0.9, ("b", "p2"): 0.9},
        fused_probabilities={("a", first.group_id): 0.9, ("b", first.group_id): 0.1},
        input_tokens_per_evaluation=10,
        output_tokens_per_evaluation=2,
    )
    records = [ReviewRecord("a", "a", {}), ReviewRecord("b", "b", {})]
    result = await FusedExecutor(provider, concurrency=2).execute(
        records, plan, run_id="r", maximum_http_attempts=3
    )
    assert result.complete
    assert provider.stats.attempts == 3
    assert [group.input_record_count for group in result.groups] == [2, 1]
    assert [record.review_id for record in result.matching_records] == ["a"]


@pytest.mark.asyncio
async def test_fused_executor_fails_closed_on_budget() -> None:
    operator = build_fused_operator(
        predicates(2), mode="conjunctive_binary", requested_model="m", threshold=0.5
    )
    plan = build_physical_plan(
        workload_id="w",
        logical_query_id="q",
        logical_predicate_ids=operator.predicate_ids,
        groups=[operator],
        configuration_fingerprint="cfg",
    )
    records = [ReviewRecord(str(i), str(i), {}) for i in range(3)]
    result = await FusedExecutor(MockProvider(), concurrency=3).execute(
        records, plan, run_id="r", maximum_http_attempts=2
    )
    assert not result.complete
    assert result.incomplete_reason == "request_budget_exhausted"
    assert not result.matching_records


def test_fusion_cost_reachability_and_zero_smoothed_selectivity() -> None:
    candidate = next(plan for plan in plans(2) if len(plan.groups) == 2)
    first, second = candidate.groups
    statistics = {
        first.group_id: GroupStatistics(
            group_id=first.group_id,
            predicate_ids=first.predicate_ids,
            fusion_mode="single",
            sample_count=10,
            passing_count=0,
            empirical_selectivity=0.0,
            smoothed_selectivity=0.0,
            mean_input_tokens=10,
            mean_output_tokens=2,
            mean_cost_usd=0.001,
            evidence_source="measured_separate",
        ),
        second.group_id: GroupStatistics(
            group_id=second.group_id,
            predicate_ids=second.predicate_ids,
            fusion_mode="single",
            sample_count=10,
            passing_count=5,
            empirical_selectivity=0.5,
            mean_input_tokens=10,
            mean_output_tokens=2,
            mean_cost_usd=0.001,
            evidence_source="measured_separate",
        ),
    }
    result = estimate_fusion_plan_cost(
        candidate,
        evaluation_records=100,
        statistics=statistics,
        input_price_per_million=0.042,
        output_price_per_million=0,
    )
    assert result.expected_provider_requests == 100
    assert result.expected_stage_cardinalities == (100.0, 0.0)
    assert result.no_early_rejection_requests == 200


def test_quality_metrics_undefined_and_gate_fails_closed() -> None:
    metrics = compare_binary_decisions([False, False], [False, False])
    assert metrics.recall is None and metrics.matthews_correlation is None
    decision = evaluate_quality_gate(
        metrics,
        FusionQualityGate(minimum_records=2, minimum_reference_positives=1),
        evidence_scope="complete_plan",
        real_fused_measurement=True,
    )
    assert not decision.eligible
    assert "insufficient_reference_positives" in decision.reasons


def test_quality_constrained_selection_falls_back_without_evidence() -> None:
    candidates = plans(2)
    fallback = next(item for item in candidates if not item.contains_fusion)
    costs = {item.plan_id: index / 100 + 0.01 for index, item in enumerate(candidates)}
    result = select_quality_constrained_plan(
        candidates, costs, {}, fallback=fallback
    )
    assert result.used_fallback and result.selected_plan_id == fallback.plan_id


def test_profiling_controller_enforces_all_budgets_and_reuse() -> None:
    actions = [
        FusionProfilingAction(
            group_id="g1",
            predicate_ids=("p1", "p2"),
            records=10,
            estimated_cost_per_request_usd=0.001,
            optimistic_execution_saving_per_reuse_usd=0.01,
            required_for_plan_ids=("x",),
        ),
        FusionProfilingAction(
            group_id="g2",
            predicate_ids=("p2", "p3"),
            records=10,
            estimated_cost_per_request_usd=0.001,
            optimistic_execution_saving_per_reuse_usd=0.01,
            required_for_plan_ids=("y",),
        ),
    ]
    decision = choose_fusion_profiling_actions(
        actions,
        budget=FusionProfilingBudget(
            maximum_unique_groups=1,
            maximum_records_per_group=10,
            maximum_logical_requests=10,
            maximum_http_attempts=30,
            maximum_estimated_expenditure_usd=0.02,
            max_retries=2,
        ),
        reuse_horizon=10,
        already_measured_group_ids=("g1",),
    )
    assert [item.group_id for item in decision.selected_actions] == ["g2"]
    assert decision.retry_inclusive_http_attempts == 30
    assert decision.future_records_inspected is False


def test_economics_break_even() -> None:
    result = fusion_economics(
        reuse_horizon=5,
        baseline_execution_cost_usd=0.02,
        candidate_execution_cost_usd=0.01,
        incremental_profiling_cost_usd=0.025,
    )
    assert result.break_even_reuse_count == 3
    assert result.net_benefit_usd == pytest.approx(0.025)


def test_measured_replay_does_not_invent_missing_fused_outcomes() -> None:
    candidate = next(item for item in plans(2) if item.contains_fusion)
    separate = {
        ("r", "p0"): {"passed": True, "input_tokens": 10, "output_tokens": 1},
        ("r", "p1"): {"passed": True, "input_tokens": 10, "output_tokens": 1},
    }
    result = replay_measured_plan(
        candidate,
        record_ids=["r"],
        separate_traces=separate,
        fused_traces={},
        input_price_per_million=0.042,
        output_price_per_million=0,
    )
    assert not result["eligible_measured_replay"]
    assert result["quality"] is None


def test_unreached_unmeasured_fused_group_is_not_quality_evidence() -> None:
    candidate = next(
        item
        for item in plans(3)
        if len(item.groups) == 2
        and len(item.groups[0].predicate_ids) == 1
        and len(item.groups[1].predicate_ids) == 2
    )
    first = candidate.groups[0].predicate_ids[0]
    other = candidate.groups[1].predicate_ids
    separate = {
        ("r", first): {"passed": False, "input_tokens": 10, "output_tokens": 1},
        ("r", other[0]): {"passed": True, "input_tokens": 10, "output_tokens": 1},
        ("r", other[1]): {"passed": True, "input_tokens": 10, "output_tokens": 1},
    }
    result = replay_measured_plan(
        candidate,
        record_ids=["r"],
        separate_traces=separate,
        fused_traces={},
        input_price_per_million=0.042,
        output_price_per_million=0,
    )
    assert not result["eligible_measured_replay"]
    assert result["missing_measurements"] == [
        f"group_evidence:{candidate.groups[1].group_id}"
    ]


def test_stage_b_authorization_binds_budget_schedule_and_output(tmp_path: Path) -> None:
    suite_path = "configs/milestone_5/workloads.yaml"
    protocol_path = "configs/milestone_5/protocol.yaml"
    fusion_path = "configs/milestone_5/fusion.yaml"
    groups_path = "configs/milestone_5/pilot_groups.yaml"
    suite = load_fusion_suite(suite_path)
    budget = build_pilot_budget(
        suite,
        load_fusion_protocol(protocol_path),
        load_yaml(fusion_path),
        load_yaml(groups_path),
    )
    manifest = build_authorization_manifest(
        budget=budget,
        suite_path=suite_path,
        protocol_path=protocol_path,
        fusion_config_path=fusion_path,
        pilot_groups_path=groups_path,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "new"
    with pytest.raises(PermissionError):
        validate_stage_b_launch(
            manifest_path=manifest_path,
            authorization="wrong",
            output=output,
            suite_path=suite_path,
            protocol_path=protocol_path,
            fusion_config_path=fusion_path,
            pilot_groups_path=groups_path,
        )
    validated = validate_stage_b_launch(
        manifest_path=manifest_path,
        authorization=STAGE_B_AUTHORIZATION,
        output=output,
        suite_path=suite_path,
        protocol_path=protocol_path,
        fusion_config_path=fusion_path,
        pilot_groups_path=groups_path,
    )
    assert validated["http_attempt_ceiling"] == 3690
    output.mkdir()
    with pytest.raises(FileExistsError):
        validate_stage_b_launch(
            manifest_path=manifest_path,
            authorization=STAGE_B_AUTHORIZATION,
            output=output,
            suite_path=suite_path,
            protocol_path=protocol_path,
            fusion_config_path=fusion_path,
            pilot_groups_path=groups_path,
        )


def test_maximum_group_size_filters_without_arbitrary_grouping() -> None:
    candidates = ordered_set_partitions(["a", "b", "c"], maximum_group_size=2)
    assert len(candidates) == 12
    assert all(len(group) <= 2 for plan in candidates for group in plan)
