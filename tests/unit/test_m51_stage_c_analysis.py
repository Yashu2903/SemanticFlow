from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from semanticflow.optimizer.m51_stage_c_analysis import (
    build_batch_comparisons,
    build_batch_size_economics,
    build_cost_reconciliation,
    build_hybrid_plan_analysis,
    build_integrity_audit,
    build_profiling_economics,
    build_quality_gate_results,
    build_stage_d_readiness,
    generate_analysis,
    load_stage_b_requests,
    quality_status,
    read_json,
    read_yaml,
    token_cost,
)

ROOT = Path("results/milestone_5_1/live_batching_pilot_stage_b")

pytestmark = pytest.mark.research_artifacts(
    "results/milestone_5_1/stage_a_preflight/stage_b_authorization_proposal_final.json",
    "results/milestone_5_1/live_batching_pilot_stage_b/session.json",
)


@pytest.fixture(scope="module")
def evidence() -> dict[str, object]:
    workloads = read_yaml("configs/milestone_5_1/workloads.yaml")
    protocol = read_yaml("configs/milestone_5_1/protocol.yaml")
    schedule = read_yaml("configs/milestone_5_1/pilot_schedule.yaml")
    authorization = read_json(
        "results/milestone_5_1/stage_a_preflight/stage_b_authorization_proposal_final.json"
    )
    preflight = read_json("results/milestone_5_1/stage_a_preflight/preflight_final.json")
    rows, _ = load_stage_b_requests(ROOT, workloads)
    comparisons = build_batch_comparisons(rows)
    plans = build_hybrid_plan_analysis(rows, workloads)
    return {
        "workloads": workloads,
        "protocol": protocol,
        "schedule": schedule,
        "authorization": authorization,
        "preflight": preflight,
        "rows": rows,
        "comparisons": comparisons,
        "plans": plans,
    }


def test_complete_request_and_named_question_reconciliation(evidence: dict[str, object]) -> None:
    rows = evidence["rows"]
    assert isinstance(rows, list)
    assert len(rows) == 810
    assert sum(row["http_attempt_count"] for row in rows) == 810
    assert sum(len(row["predicate_question_names"]) for row in rows) == 1380
    assert len({row["request_id"] for row in rows}) == 810


def test_token_cost_arithmetic_and_categories(evidence: dict[str, object]) -> None:
    rows = evidence["rows"]
    assert isinstance(rows, list)
    reconciliation = build_cost_reconciliation(rows)
    assert reconciliation["total"]["input_tokens"] == 299_406
    assert reconciliation["total"]["output_tokens"] == 37_690
    assert token_cost(299_406, 37_690) == pytest.approx(0.012575052)
    assert reconciliation["total"]["cost_reconciles"]
    counts = {
        "representative_single": sum(
            row["sample_kind"] == "representative" and row["execution_mode"] == "single"
            for row in rows
        ),
        "representative_batch": sum(
            row["sample_kind"] == "representative"
            and row["execution_mode"] == "multi_decision"
            for row in rows
        ),
        "representative_conjunction": sum(
            row["sample_kind"] == "representative"
            and row["execution_mode"] == "conjunctive_binary"
            for row in rows
        ),
        "enriched": sum(row["sample_kind"] == "enriched" for row in rows),
    }
    assert counts == {
        "representative_single": 360,
        "representative_batch": 270,
        "representative_conjunction": 90,
        "enriched": 90,
    }


def test_paired_alignment_and_sample_distinction(evidence: dict[str, object]) -> None:
    comparisons, _, _, _ = evidence["comparisons"]
    assert len(comparisons) == 15
    assert all(row["evidence_type"] == "fully_measured_paired" for row in comparisons)
    assert all(not row["missing_reference_cells"] for row in comparisons)
    assert sum(row["sample_kind"] == "representative" for row in comparisons) == 12
    assert sum(row["sample_kind"] == "enriched" for row in comparisons) == 3
    assert all(
        row["partition"] == "development"
        for row in comparisons
        if row["sample_kind"] == "enriched"
    )


def test_early_reject_replay_and_batch_size_grouping(evidence: dict[str, object]) -> None:
    comparisons, _, _, _ = evidence["comparisons"]
    multi = [row for row in comparisons if row["execution_mode"] == "multi_decision"]
    assert all(
        row["early_reject_separate"]["requests"]
        <= row["all_questions_separate"]["requests"]
        for row in multi
    )
    sizes = build_batch_size_economics(comparisons)
    representative = [row for row in sizes if row["sample_kind"] == "representative"]
    assert [row["batch_size"] for row in representative] == [2, 3, 4]
    assert [row["records"] for row in representative] == [90, 90, 90]
    assert all(row["batch_requests"] == 90 for row in representative)


def test_missing_reference_is_not_zero_disagreement(evidence: dict[str, object]) -> None:
    rows = copy.deepcopy(evidence["rows"])
    assert isinstance(rows, list)
    removed = next(
        row
        for row in rows
        if row["sample_kind"] == "representative" and row["execution_mode"] == "single"
    )
    rows.remove(removed)
    comparisons, _, _, _ = build_batch_comparisons(rows)
    affected = [
        row
        for row in comparisons
        if row["dataset"] == removed["dataset"]
        and removed["predicate_ids"][0] in row["predicate_ids"]
        and row["sample_kind"] == "representative"
    ]
    assert affected
    assert any(row["evidence_type"] == "not_evaluable" for row in affected)
    assert any(row["missing_reference_cells"] for row in affected)


def test_zero_positive_metrics_and_false_error_counts() -> None:
    gate = {
        "minimum_records": 40,
        "minimum_reference_positives": 3,
        "minimum_agreement": 0.98,
        "minimum_recall": 0.95,
        "maximum_false_negative_rate": 0.05,
        "maximum_changed_decision_rate": 0.02,
    }
    metrics = {
        "sample_count": 40,
        "reference_positive_count": 0,
        "agreement": 1.0,
        "recall": None,
        "false_negative_rate": None,
        "changed_decision_rate": 0.0,
    }
    decision = quality_status(metrics, gate, representative=True)
    assert not decision["eligible"]
    assert decision["status"] == "insufficient_positive_examples"
    assert "undefined_recall" in decision["reasons"]


def test_query_metrics_and_quality_gate_fail_closed(evidence: dict[str, object]) -> None:
    _, _, _, query = evidence["comparisons"]
    assert query
    assert all(row["sample_count"] in {10, 30} for row in query)
    assert all(
        row["false_positive"] + row["false_negative"] == row["changed_decisions"]
        for row in query
    )
    plans, _, _ = evidence["plans"]
    quality = build_quality_gate_results(
        protocol=evidence["protocol"],
        query_agreement=query,
        plan_rows=plans,
    )
    assert quality["summary"]["eligible"] == 0
    assert not quality["summary"]["batching_or_hybrid_candidate_qualified"]


def test_hybrid_replay_missing_plans_and_prediction_regret(evidence: dict[str, object]) -> None:
    plans, summary, predictions = evidence["plans"]
    assert plans
    unavailable = [row for row in summary.values() if row["status"] == "not_evaluable"]
    assert len(unavailable) == 3
    assert len(predictions) == 6
    assert all(row["cost_mae_usd"] >= 0 for row in predictions)
    assert all(row["selection_regret_usd"] >= -1e-15 for row in predictions)
    assert all(
        len(row["stage_input_cardinalities"]) == len(row["plan"])
        for row in plans
    )


def test_profiling_accounting_break_even_and_no_double_charge(evidence: dict[str, object]) -> None:
    _, summary, _ = evidence["plans"]
    rows, accounting = build_profiling_economics(
        rows=evidence["rows"],
        workloads=evidence["workloads"],
        workload_summary=summary,
        horizons=evidence["protocol"]["reuse_horizons"],
    )
    assert rows
    assert accounting["shared_costs_double_counted"] is False
    for workload in accounting["workload_assumptions"].values():
        if workload["status"] == "exploratory":
            assert (
                workload["shared_incremental_cost_usd"]
                <= workload["standalone_profiling_cost_usd"] + 1e-15
            )
    positive = [row for row in rows if row["execution_saving_per_run_usd"] > 0]
    assert all(row["break_even_reuse_count"] is not None for row in positive)


def test_integrity_fingerprints_and_held_out_guard(evidence: dict[str, object]) -> None:
    audit = build_integrity_audit(
        root=ROOT,
        rows=evidence["rows"],
        workloads=evidence["workloads"],
        protocol=evidence["protocol"],
        schedule=evidence["schedule"],
        authorization_manifest=evidence["authorization"],
        preflight=evidence["preflight"],
    )
    assert audit["execution_reconciliation"]["all_requests_accounted"]
    assert audit["source_integrity"]["all_unchanged"]
    assert audit["historical_integrity"]["all_unchanged"]
    assert audit["partition_integrity"]["held_out_record_ids_observed"] == 0
    assert audit["partition_integrity"]["held_out_contents_read_by_analysis"] is False
    assert audit["physical_plan_integrity"]["status"] == "not_available"


def test_readiness_requires_selection_not_stage_d(evidence: dict[str, object]) -> None:
    _, _, _, query = evidence["comparisons"]
    plans, _, _ = evidence["plans"]
    quality = build_quality_gate_results(
        protocol=evidence["protocol"], query_agreement=query, plan_rows=plans
    )
    audit = build_integrity_audit(
        root=ROOT,
        rows=evidence["rows"],
        workloads=evidence["workloads"],
        protocol=evidence["protocol"],
        schedule=evidence["schedule"],
        authorization_manifest=evidence["authorization"],
        preflight=evidence["preflight"],
    )
    readiness = build_stage_d_readiness(integrity=audit, quality=quality)
    assert readiness["conclusion"] == "READY_FOR_SELECTION_STAGE_AUTHORIZATION"
    assert not readiness["stage_d_ready"]
    assert readiness["selection_stage_request_proposal"]["logical_request_ceiling"] == 3300
    assert not readiness["experimental_validity"]["leakage_free_adaptive_trajectory_available"]


def test_analysis_never_imports_provider(evidence: dict[str, object]) -> None:
    analysis_source = Path(
        "src/semanticflow/optimizer/m51_stage_c_analysis.py"
    ).read_text("utf-8")
    runner_source = Path("scripts/analyze_m51_stage_c.py").read_text("utf-8")
    assert "semanticflow.providers.jev" not in analysis_source
    assert "semanticflow.providers.jev" not in runner_source
    assert math.isclose(token_cost(1_000_000), 0.042)


def test_non_overwriting_artifact_generation(tmp_path: Path) -> None:
    output = tmp_path / "analysis"
    report = tmp_path / "report.md"
    result = generate_analysis(output_directory=output, report_path=report)
    assert result["status"] == "complete"
    assert result["live_jev_calls"] == 0
    assert (output / "analysis_manifest.json").is_file()
    with pytest.raises(FileExistsError):
        generate_analysis(output_directory=output, report_path=tmp_path / "report2.md")
