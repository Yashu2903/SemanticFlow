from __future__ import annotations

import httpx
import pytest

from semanticflow.dataset.loader import ReviewRecord
from semanticflow.executor.hybrid import HybridExecutor
from semanticflow.optimizer.fusion import ProviderFusionCapabilities, build_fused_operator
from semanticflow.optimizer.fusion_cost import GroupStatistics
from semanticflow.optimizer.fusion_planner import build_physical_plan
from semanticflow.optimizer.fusion_quality import FusionQualityGate, compare_binary_decisions
from semanticflow.optimizer.hybrid import (
    enumerate_hybrid_plans,
    fixed_selective_first_hybrid,
    full_batch_plan,
    hybrid_operator_metadata,
)
from semanticflow.optimizer.hybrid_cost import (
    estimate_hybrid_plan_cost,
    hybrid_economics_views,
)
from semanticflow.optimizer.hybrid_profile import (
    HybridProfilingAction,
    choose_hybrid_profiling_actions,
)
from semanticflow.optimizer.hybrid_prompts import build_template_operator
from semanticflow.optimizer.hybrid_quality import (
    classify_quality_evidence,
    compare_named_batch_decisions,
)
from semanticflow.providers.base import RequestBudget, ResponseValidationError
from semanticflow.providers.jev import JevProvider
from semanticflow.providers.mock import MockProvider
from semanticflow.query.models import PredicateSpec


def _predicates(count: int) -> list[PredicateSpec]:
    return [PredicateSpec(id=f"p{i}", instructions=f"Condition {i}?") for i in range(count)]


def _plans(count: int):
    return enumerate_hybrid_plans(
        _predicates(count),
        workload_id="w",
        logical_query_id="q",
        requested_model="jev-1.14.0",
        threshold=0.5,
        capabilities=ProviderFusionCapabilities(
            provider="test",
            conjunctive_binary=True,
            multi_decision=True,
            maximum_questions_per_request=4,
            evidence="test",
        ),
        configuration_fingerprint="cfg",
    )


@pytest.mark.parametrize(("count", "expected"), [(2, 4), (3, 20), (4, 138)])
def test_mixed_mode_plan_counts(count: int, expected: int) -> None:
    plans = _plans(count)
    assert len(plans) == expected
    assert len({plan.plan_id for plan in plans}) == expected
    for plan in plans:
        flattened = [item for group in plan.groups for item in group.predicate_ids]
        assert len(flattened) == count and len(set(flattened)) == count


def test_full_batch_and_selective_first_are_deterministic() -> None:
    plans = _plans(4)
    assert full_batch_plan(plans).groups[0].fusion_mode == "multi_decision"
    hybrid = fixed_selective_first_hybrid(plans, ["p0", "p1", "p2", "p3"])
    assert [group.fusion_mode for group in hybrid.groups] == ["single", "multi_decision"]


def test_complete_request_metadata_is_fingerprinted() -> None:
    operator = build_fused_operator(
        _predicates(2), mode="multi_decision", requested_model="jev-1.14.0", threshold=0.5
    )
    first = hybrid_operator_metadata(operator, dataset_fingerprint="data")
    second = hybrid_operator_metadata(operator, dataset_fingerprint="changed")
    assert first.question_names == ("p0", "p1")
    assert first.request_fingerprint != second.request_fingerprint


@pytest.mark.asyncio
async def test_multi_decision_is_one_request_and_preserves_named_decisions() -> None:
    p = _predicates(3)
    operator = build_fused_operator(
        p, mode="multi_decision", requested_model="m", threshold=0.5
    )
    plan = build_physical_plan(
        workload_id="w",
        logical_query_id="q",
        logical_predicate_ids=[item.id for item in p],
        groups=[operator],
        configuration_fingerprint="cfg",
    )
    provider = MockProvider(
        probabilities={("x", "p0"): 0.9, ("x", "p1"): 0.1, ("x", "p2"): 0.9},
        input_tokens_per_evaluation=20,
        output_tokens_per_evaluation=6,
    )
    result = await HybridExecutor(provider).execute(
        [ReviewRecord("x", "x", {})], plan, run_id="r", maximum_http_attempts=1
    )
    assert provider.stats.attempts == 1
    assert result.named_question_evaluations == 3
    assert result.traces[0].predicate_probabilities == {"p0": 0.9, "p1": 0.1, "p2": 0.9}
    assert not result.matching_records


@pytest.mark.asyncio
async def test_hybrid_rejects_before_later_batch() -> None:
    p = _predicates(4)
    first = build_fused_operator(p[:1], mode="single", requested_model="m", threshold=0.5)
    batch = build_fused_operator(
        p[1:], mode="multi_decision", requested_model="m", threshold=0.5
    )
    plan = build_physical_plan(
        workload_id="w",
        logical_query_id="q",
        logical_predicate_ids=[item.id for item in p],
        groups=[first, batch],
        configuration_fingerprint="cfg",
    )
    provider = MockProvider(probabilities={("x", "p0"): 0.1})
    result = await HybridExecutor(provider).execute(
        [ReviewRecord("x", "x", {})], plan, run_id="r", maximum_http_attempts=1
    )
    assert result.complete and provider.stats.attempts == 1
    assert not result.groups[1].reached


@pytest.mark.asyncio
async def test_fallback_is_explicit_and_billed() -> None:
    p = _predicates(2)
    primary = build_physical_plan(
        workload_id="w",
        logical_query_id="q",
        logical_predicate_ids=[item.id for item in p],
        groups=[build_fused_operator(p, mode="multi_decision", requested_model="m", threshold=0.5)],
        configuration_fingerprint="cfg",
    )
    separate = build_physical_plan(
        workload_id="w",
        logical_query_id="q",
        logical_predicate_ids=[item.id for item in p],
        groups=[
            build_fused_operator([item], mode="single", requested_model="m", threshold=0.5)
            for item in p
        ],
        configuration_fingerprint="cfg",
    )
    records = [ReviewRecord("x", "x", {}), ReviewRecord("y", "y", {})]
    result = await HybridExecutor(MockProvider()).execute_with_fallback(
        records,
        primary,
        run_id="r",
        maximum_http_attempts=1,
        fallback_plan=separate,
        fallback_authorized=False,
    )
    assert not result.complete and result.fallback is None
    result = await HybridExecutor(MockProvider()).execute_with_fallback(
        records,
        primary,
        run_id="r2",
        maximum_http_attempts=1,
        fallback_plan=separate,
        fallback_authorized=True,
        fallback_maximum_http_attempts=4,
    )
    assert result.complete and result.fallback is not None
    assert result.total_provider_attempts > result.fallback.provider_attempts


def test_unmeasured_zero_reach_group_never_completes_cost() -> None:
    plan = fixed_selective_first_hybrid(_plans(3), ["p0", "p1", "p2"])
    first = plan.groups[0]
    stats = {
        first.group_id: GroupStatistics(
            group_id=first.group_id,
            predicate_ids=first.predicate_ids,
            fusion_mode="single",
            sample_count=10,
            passing_count=0,
            empirical_selectivity=0,
            smoothed_selectivity=0,
            mean_input_tokens=10,
            mean_output_tokens=2,
            evidence_source="measured_separate",
        )
    }
    result = estimate_hybrid_plan_cost(
        plan,
        evaluation_records=100,
        statistics=stats,
        input_price_per_million=0.042,
        output_price_per_million=0,
    )
    assert not result.complete
    assert plan.groups[1].group_id in result.unknown_groups


def test_quality_evidence_states_and_named_metrics() -> None:
    gate = FusionQualityGate()
    sparse = compare_binary_decisions([False] * 40, [False] * 40)
    result = classify_quality_evidence(
        sparse,
        gate,
        evidence_scope="complete_plan",
        real_provider_measurement=True,
    )
    assert result.status == "insufficient_positive_examples"
    metrics = compare_named_batch_decisions(
        {"p0": [True, False], "p1": [False, False]},
        {"p0": [False, False], "p1": [False, False]},
    )
    assert metrics["p0"].false_negative == 1
    assert metrics["p1"].recall is None


def test_hybrid_economics_separates_accounting_views() -> None:
    result = hybrid_economics_views(
        reuse_horizon=10,
        baseline_execution_cost_usd=1,
        candidate_execution_cost_usd=0.9,
        actual_profiling_expenditure_usd=0.2,
        standalone_profiling_cost_usd=0.2,
        shared_incremental_profiling_cost_usd=0.1,
    )
    assert result.warm_start_net_benefit_usd == pytest.approx(1)
    assert result.cold_start_shared_net_benefit_usd == pytest.approx(0.9)


def test_profiling_controller_shares_and_stops() -> None:
    actions = [
        HybridProfilingAction(
            group_id="g1",
            execution_mode="multi_decision",
            predicate_ids=("p0", "p1"),
            records=10,
            estimated_request_cost_usd=0.001,
            optimistic_saving_per_reuse_usd=0.01,
            impact_plan_ids=("a", "b"),
        )
    ]
    result = choose_hybrid_profiling_actions(
        actions,
        reuse_horizon=5,
        measured_group_ids=(),
        maximum_unique_groups=1,
        maximum_records_per_group=10,
        maximum_logical_requests=10,
        maximum_http_attempts=30,
        maximum_estimated_cost_usd=0.02,
        max_retries=2,
    )
    assert result.logical_requests == 10 and result.maximum_http_attempts == 30


def test_conjunctive_templates_are_versioned_and_distinct() -> None:
    operators = [
        build_template_operator(
            _predicates(2), template=template, requested_model="m", threshold=0.5
        )
        for template in (
            "m5.1-original-conjunction-v1",
            "m5.1-explicit-conjunction-v1",
            "m5.1-structured-checklist-v1",
        )
    ]
    assert len({operator.fusion_prompt_hash for operator in operators}) == 3


@pytest.mark.asyncio
async def test_jev_rejects_missing_and_extra_batch_answers() -> None:
    operator = build_fused_operator(
        _predicates(2), mode="multi_decision", requested_model="m", threshold=0.5
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "answers": {
                    "p0": {"type": "noul", "noul": 0.5},
                    "p1": {"type": "noul", "noul": 0.5},
                    "extra": {"type": "noul", "noul": 0.5},
                }
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = JevProvider("unused", "m", client=client, max_retries=0)
    with pytest.raises(ResponseValidationError):
        await provider.decide_fused("x", operator, RequestBudget(1))
    await provider.aclose()
