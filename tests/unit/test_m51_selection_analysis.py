from __future__ import annotations

import json

import pytest

from semanticflow.optimizer.m51_selection_analysis import (
    analyze,
    binary_metrics,
    load_evidence,
    quality_status,
    replay_plan,
)

pytestmark = pytest.mark.research_artifacts(
    "results/milestone_5_1/selection_stage_results/live_session_manifest.json",
)


def test_live_evidence_reconciles_exactly() -> None:
    evidence = load_evidence()
    rows = evidence["normalized"]
    assert len(rows) == 3000
    assert len({row["request_id"] for row in rows}) == 3000
    assert sum(row["named_question_evaluations"] for row in rows) == 5300
    assert sum(row["input_tokens"] for row in rows) == 1_064_824
    assert sum(row["output_tokens"] for row in rows) == 142_700


def test_binary_metrics_preserve_undefined_positive_metrics() -> None:
    metrics = binary_metrics([False, False], [False, False])
    assert metrics["agreement"] == 1
    assert metrics["recall"] is None
    assert metrics["false_negative_rate"] is None


def test_quality_gate_does_not_pass_sparse_support() -> None:
    metrics = binary_metrics([True, True, *([False] * 98)], [True, True, *([False] * 98)])
    gate = {
        "minimum_records": 40,
        "minimum_reference_positives": 3,
        "minimum_agreement": 0.98,
        "minimum_recall": 0.95,
        "maximum_false_negative_rate": 0.05,
        "maximum_changed_decision_rate": 0.02,
    }
    result = quality_status(metrics, gate)
    assert result["status"] == "insufficient_positive_support"
    assert not result["eligible"]


def test_quality_gate_passes_supported_perfect_candidate() -> None:
    metrics = binary_metrics(
        [True, True, True, *([False] * 97)], [True, True, True, *([False] * 97)]
    )
    gate = {
        "minimum_records": 40,
        "minimum_reference_positives": 3,
        "minimum_agreement": 0.98,
        "minimum_recall": 0.95,
        "maximum_false_negative_rate": 0.05,
        "maximum_changed_decision_rate": 0.02,
    }
    assert quality_status(metrics, gate)["eligible"]


def test_every_catalog_plan_is_replayable() -> None:
    evidence = load_evidence()
    records = sorted(
        {
            row["record_id"]
            for row in evidence["normalized"]
            if row["dataset"] == "amazon_all_beauty"
        }
    )
    plans = [plan for plan in evidence["catalog"] if plan["dataset"] == "amazon_all_beauty"]
    assert len(plans) == 37
    assert all(replay_plan(plan, records, evidence["lookup"])["requests"] > 0 for plan in plans)


def test_complete_analysis_is_nonoverwriting_and_stage_d_offline(tmp_path) -> None:
    output = tmp_path / "selection"
    result = analyze(output)
    assert result["summary"]["physical_requests"] == 3000
    assert result["summary"]["stage_d_executed"] is False
    alias = json.loads((output / "shared_alias_mapping.json").read_text())
    assert alias["count"] == 300
    assert alias["all_resolve"]
    schedule = json.loads((output / "stage_d_execution_schedule.json").read_text())
    assert schedule["held_out_outcomes_accessed"] is False
    assert schedule["records_per_dataset"] == 400
    replay = json.loads((output / "profiling_aware_selection_replay.json").read_text())
    assert replay["leakage_free"]
    assert all(
        all(not checkpoint["future_measurements_visible"] for checkpoint in workload["trajectory"])
        for workload in replay["workloads"]
    )
    with pytest.raises(FileExistsError):
        analyze(output)
