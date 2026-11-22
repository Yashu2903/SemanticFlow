from __future__ import annotations

import json
from pathlib import Path

import pytest

from semanticflow.optimizer.m51_selection_analysis import load_evidence
from semanticflow.optimizer.m52_acquisition import (
    ActionKind,
    ProfilingAction,
    ProfilingBudget,
)
from semanticflow.optimizer.m52_active_profiler import (
    AcquisitionContext,
    AcquisitionPolicy,
    SamplingStrategy,
)
from semanticflow.optimizer.m52_controller import M52ControllerConfig, M52Optimizer
from semanticflow.optimizer.m52_cost_model import M52CostEstimator
from semanticflow.optimizer.m52_evidence import EvidenceTrack, QualityEvidence
from semanticflow.optimizer.m52_planner import PolicyMode, policy_fingerprint, select_plan
from semanticflow.optimizer.m52_replay import HistoricalReplayMatrix

pytestmark = pytest.mark.research_artifacts(
    "results/milestone_5_1/selection_stage_results/frozen_strategy_selections.json",
)


@pytest.fixture(scope="module")
def selection_data() -> tuple[HistoricalReplayMatrix, dict[str, object], list[dict[str, object]]]:
    matrix = HistoricalReplayMatrix.selection()
    evidence = load_evidence()
    selections = json.loads(
        Path(
            "results/milestone_5_1/selection_stage_results/frozen_strategy_selections.json"
        ).read_text("utf-8")
    )["strategies"]
    return matrix, evidence, selections


def plans(
    matrix: HistoricalReplayMatrix,
    selections: list[dict[str, object]],
    workload_id: str,
) -> tuple[dict[str, object], dict[str, object]]:
    fallback_id = next(
        row["plan_id"]
        for row in selections
        if row["strategy"] == "existing_adaptive" and row["workload_id"] == workload_id
    )
    batch_id = next(
        row["plan_id"]
        for row in selections
        if row["strategy"] == "full_multi_decision_batch" and row["workload_id"] == workload_id
    )
    return matrix.plans[str(fallback_id)], matrix.plans[str(batch_id)]


def empty_evidence(plan: dict[str, object]) -> QualityEvidence:
    return QualityEvidence(
        plan_id=str(plan["plan_id"]),
        plan_fingerprint=str(plan["plan_fingerprint"]),
        dataset=str(plan["dataset"]),
        dataset_fingerprint="dataset-fingerprint",
        workload_id=str(plan["workload_id"]),
        workload_fingerprint="workload-fingerprint",
        physical_groups=tuple(tuple(group["predicate_ids"]) for group in plan["groups"]),
    )


def test_unacquired_trace_is_invisible(selection_data: tuple[object, object, object]) -> None:
    matrix, _, _ = selection_data
    with pytest.raises(PermissionError, match="has not been acquired"):
        matrix.trace(str(matrix.rows[0]["request_id"]))


def test_representative_pair_is_charged_and_reused(
    selection_data: tuple[HistoricalReplayMatrix, dict[str, object], list[dict[str, object]]],
) -> None:
    template, _, selections = selection_data
    matrix = HistoricalReplayMatrix(
        rows=template.rows,
        plans=tuple(template.plans.values()),
        selections=selections,
        config=template.config,
        source_partition="selection",
    )
    _, batch = plans(matrix, selections, "m51_banking_card_unauthorized_refund")
    record_id = matrix.records(str(batch["dataset"]))[0]
    action = ProfilingAction(
        action_id="pair",
        kind=ActionKind.ACQUIRE_REPRESENTATIVE_PAIR,
        record_id=record_id,
        plan_id=str(batch["plan_id"]),
        track=EvidenceTrack.CERTIFICATION,
        acquisition_method="test",
    )
    first = matrix.acquire(action)
    second = matrix.acquire(action.model_copy(update={"action_id": "pair2"}))
    assert first.receipt.physical_requests > 0
    assert first.receipt.token_priced_cost_usd > 0
    assert second.receipt.physical_requests == 0
    assert second.receipt.token_priced_cost_usd == 0
    assert second.receipt.reused_measurement_ids


def test_discovery_candidate_requires_paid_screen(
    selection_data: tuple[HistoricalReplayMatrix, dict[str, object], list[dict[str, object]]],
) -> None:
    template, _, selections = selection_data
    matrix = HistoricalReplayMatrix(
        rows=template.rows,
        plans=tuple(template.plans.values()),
        selections=selections,
        config=template.config,
        source_partition="selection",
    )
    _, batch = plans(matrix, selections, "m51_beauty_failure_usability")
    record_id = matrix.records(str(batch["dataset"]))[0]
    candidate = ProfilingAction(
        action_id="candidate",
        kind=ActionKind.ACQUIRE_DISCOVERY_CANDIDATE,
        record_id=record_id,
        plan_id=str(batch["plan_id"]),
        track=EvidenceTrack.DISCOVERY,
        acquisition_method="positive_enrichment",
    )
    denied = matrix.acquire(candidate)
    assert denied.receipt.unsupported_reason == "record_not_screened"
    assert denied.receipt.physical_requests == 0
    screen = candidate.model_copy(
        update={"action_id": "screen", "kind": ActionKind.SCREEN_DISCOVERY_REFERENCE}
    )
    screened = matrix.acquire(screen)
    assert screened.receipt.physical_requests > 0
    acquired = matrix.acquire(candidate)
    assert acquired.quality_measurement is not None
    assert acquired.quality_measurement.track == EvidenceTrack.DISCOVERY


def test_estimator_rejects_future_evaluation_partition(
    selection_data: tuple[HistoricalReplayMatrix, dict[str, object], list[dict[str, object]]],
) -> None:
    _, evidence, _ = selection_data
    rows = [dict(evidence["normalized"][0]), dict(evidence["normalized"][1])]
    rows[1]["partition"] = "evaluation"
    with pytest.raises(ValueError, match="unauthorized partitions"):
        M52CostEstimator(rows, allowed_partition="selection")


def test_cost_prediction_persists_selection_measurement_provenance(
    selection_data: tuple[HistoricalReplayMatrix, dict[str, object], list[dict[str, object]]],
) -> None:
    matrix, evidence, selections = selection_data
    fallback, batch = plans(matrix, selections, "m51_beauty_failure_usability")
    estimator = M52CostEstimator(evidence["normalized"], allowed_partition="selection")
    prediction = estimator.predict(
        batch,
        record_count=100,
        checkpoint=0,
        dataset_fingerprint="dataset",
        workload_fingerprint="workload",
    )
    assert prediction.available_data_partition == "selection"
    assert prediction.input_measurement_ids
    assert prediction.uses_future_evaluation_measurements is False
    assert all(
        identifier.startswith("m51selreq_") for identifier in prediction.input_measurement_ids
    )
    assert fallback["plan_id"] != batch["plan_id"]


def test_random_policy_deterministic_and_does_not_read_outcomes(
    selection_data: tuple[HistoricalReplayMatrix, dict[str, object], list[dict[str, object]]],
) -> None:
    matrix, evidence, selections = selection_data
    fallback, batch = plans(matrix, selections, "m51_sms_prize_contact")
    estimator = M52CostEstimator(evidence["normalized"], allowed_partition="selection")
    args = dict(
        record_count=100,
        checkpoint=0,
        dataset_fingerprint="dataset",
        workload_fingerprint="workload",
    )
    context = AcquisitionContext(
        strategy=SamplingStrategy.RANDOM_REPRESENTATIVE,
        workload_id=str(batch["workload_id"]),
        plan_id=str(batch["plan_id"]),
        checkpoint=0,
        certification_pool=tuple(matrix.records(str(batch["dataset"]))[:20]),
        discovery_pool=(),
        evidence=empty_evidence(batch),
        usage={
            "physical_requests": 0,
            "token_priced_cost_usd": 0,
            "records": 0,
            "actions": 0,
            "certification_records": 0,
            "discovery_records": 0,
            "positive_enrichment_records": 0,
        },
        budget=ProfilingBudget(),
        fallback_prediction=estimator.predict(fallback, **args),
        candidate_prediction=estimator.predict(batch, **args),
        evaluation_record_count=100,
        reuse_horizon=10,
    )
    policy = AcquisitionPolicy(SamplingStrategy.RANDOM_REPRESENTATIVE, seed=7)
    assert policy.choose(context, matrix) == policy.choose(context, matrix)
    assert not matrix.visibility.acquired_measurement_ids


def test_controller_falls_back_when_profiling_is_uneconomic(
    selection_data: tuple[HistoricalReplayMatrix, dict[str, object], list[dict[str, object]]],
) -> None:
    template, evidence, selections = selection_data
    matrix = HistoricalReplayMatrix(
        rows=template.rows,
        plans=tuple(template.plans.values()),
        selections=selections,
        config=template.config,
        source_partition="selection",
    )
    fallback, batch = plans(matrix, selections, "m51_beauty_failure_usability")
    estimator = M52CostEstimator(evidence["normalized"], allowed_partition="selection")
    args = dict(
        record_count=100,
        checkpoint=0,
        dataset_fingerprint="dataset",
        workload_fingerprint="workload",
    )
    result = M52Optimizer(
        M52ControllerConfig(
            mode=PolicyMode.STRICT_CERTIFICATION,
            sampling_strategy=SamplingStrategy.RANDOM_REPRESENTATIVE,
            reuse_horizon=1,
            minimum_expected_value_of_information_usd=1.0,
            budget=ProfilingBudget(maximum_actions=10),
        )
    ).run(
        matrix=matrix,
        evidence=empty_evidence(batch),
        fallback_prediction=estimator.predict(fallback, **args),
        candidate_prediction=estimator.predict(batch, **args),
        certification_pool=tuple(matrix.records(str(batch["dataset"]))[:40]),
        discovery_pool=(),
        evaluation_record_count=100,
    )
    assert result.decision.used_fallback is True
    assert (
        result.final_state.stopping_reason == "expected_value_of_information_below_acquisition_cost"
    )
    assert result.final_state.usage.physical_requests == 0


def test_budget_exhaustion_is_distinct_stop_reason(
    selection_data: tuple[HistoricalReplayMatrix, dict[str, object], list[dict[str, object]]],
) -> None:
    template, evidence, selections = selection_data
    matrix = HistoricalReplayMatrix(
        rows=template.rows,
        plans=tuple(template.plans.values()),
        selections=selections,
        config=template.config,
        source_partition="selection",
    )
    fallback, batch = plans(matrix, selections, "m51_banking_card_unauthorized_refund")
    estimator = M52CostEstimator(evidence["normalized"], allowed_partition="selection")
    args = dict(
        record_count=100,
        checkpoint=0,
        dataset_fingerprint="dataset",
        workload_fingerprint="workload",
    )
    result = M52Optimizer(
        M52ControllerConfig(
            mode=PolicyMode.STRICT_CERTIFICATION,
            sampling_strategy=SamplingStrategy.RANDOM_REPRESENTATIVE,
            reuse_horizon=1000,
            minimum_expected_value_of_information_usd=-1,
            budget=ProfilingBudget(
                maximum_physical_requests=1,
                maximum_cost_usd=1,
                maximum_records=1,
                maximum_actions=1,
                maximum_certification_records=1,
                maximum_discovery_records=0,
            ),
        )
    ).run(
        matrix=matrix,
        evidence=empty_evidence(batch),
        fallback_prediction=estimator.predict(fallback, **args),
        candidate_prediction=estimator.predict(batch, **args),
        certification_pool=tuple(matrix.records(str(batch["dataset"]))[:1]),
        discovery_pool=(),
        evaluation_record_count=100,
    )
    assert result.final_state.stopping_reason == "profiling_budget_exhausted"
    assert result.decision.used_fallback is True


def test_planner_modes_and_policy_fingerprints(
    selection_data: tuple[HistoricalReplayMatrix, dict[str, object], list[dict[str, object]]],
) -> None:
    matrix, evidence, selections = selection_data
    fallback, batch = plans(matrix, selections, "m51_banking_card_unauthorized_refund")
    estimator = M52CostEstimator(evidence["normalized"], allowed_partition="selection")
    args = dict(
        record_count=100,
        checkpoint=0,
        dataset_fingerprint="dataset",
        workload_fingerprint="workload",
    )
    fallback_prediction = estimator.predict(fallback, **args)
    batch_prediction = estimator.predict(batch, **args)
    cost_only = select_plan(
        mode=PolicyMode.COST_ONLY,
        fallback_plan_id=str(fallback["plan_id"]),
        candidate_plan_id=str(batch["plan_id"]),
        fallback_prediction=fallback_prediction,
        candidate_prediction=batch_prediction,
        quality=None,
        point_gate_passed=False,
        profiling_cost_usd=0,
        reuse_horizon=10,
    )
    strict = select_plan(
        mode=PolicyMode.STRICT_CERTIFICATION,
        fallback_plan_id=str(fallback["plan_id"]),
        candidate_plan_id=str(batch["plan_id"]),
        fallback_prediction=fallback_prediction,
        candidate_prediction=batch_prediction,
        quality=None,
        point_gate_passed=True,
        profiling_cost_usd=0,
        reuse_horizon=10,
    )
    assert cost_only.used_fallback is False
    assert cost_only.deployment_quality_safe is False
    assert strict.used_fallback is True
    assert policy_fingerprint({"threshold": 0.95}) != policy_fingerprint({"threshold": 0.96})
