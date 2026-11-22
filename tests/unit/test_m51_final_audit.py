from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from semanticflow.optimizer.m51_final_audit import (
    BEAUTY_WORKLOAD,
    PRIMARY_PLAN,
    PRIMARY_WORKLOAD,
    build_audit_data,
    generate,
    tree_hash,
)

AUDIT = Path("results/milestone_5_1/final_research_audit")

pytestmark = pytest.mark.research_artifacts(
    "results/milestone_5_1/selection_stage_results/frozen_strategy_selections.json",
    "results/milestone_5_1/stage_d_live/session.json",
)


@pytest.fixture(scope="module")
def audit_data() -> dict[str, object]:
    data = build_audit_data(bootstrap_iterations=199)
    return data


def test_stage_d_request_reconciliation(audit_data: dict[str, object]) -> None:
    row = audit_data["reconciliation"]
    assert row["physical_requests"] == 10_800
    assert row["http_attempts"] == 10_800
    assert row["retries"] == 0
    assert row["named_question_evaluations"] == 20_000
    assert row["input_tokens"] == 3_847_687
    assert row["output_tokens"] == 528_800
    assert row["request_composition"] == {
        "separate_reference": 6_000,
        "full_batch": 3_600,
        "hybrid_remainder_batch": 1_200,
    }
    assert row["unique_request_ids"] == row["physical_requests"]
    assert row["duplicate_request_ids"] == []
    assert row["missing_request_keys"] == []
    assert row["unexpected_request_keys"] == []
    assert row["token_priced_cost_usd"] == pytest.approx(0.161602854)


def test_quality_duplicate_is_reporting_only(audit_data: dict[str, object]) -> None:
    row = audit_data["dedup"]
    assert row["original_rows"] == 18
    assert row["original_unique_plan_ids"] == 17
    assert row["canonical_rows"] == 17
    assert len(row["duplicate_groups"]) == 1
    duplicate = row["duplicate_groups"][0]
    assert duplicate["plan_id"] == PRIMARY_PLAN
    assert duplicate["rows_identical"] is True
    assert duplicate["physical_inference_duplicated"] is False
    assert duplicate["statistical_weight_duplicated"] is False


def test_estimator_provenance_and_corrected_projection(
    audit_data: dict[str, object],
) -> None:
    provenance = audit_data["provenance"]
    validation = audit_data["ex_ante"]
    assert provenance["valid_out_of_sample_claim"] is False
    assert validation["all_nine_historical_full_batch_predictions_equal_observed"] is True
    assert validation["all_selection_predictions_reproduced"] is True
    assert validation["all_plan_metrics"]["mae_usd"] > 0
    assert validation["full_batch_metrics"]["mae_usd"] > 0
    assert all(row["uses_stage_d_inputs"] is False for row in validation["rows"])
    assert all(
        row["originally_persisted_as_stage_d_forecast"] is False for row in validation["rows"]
    )


def test_complete_query_replay_and_quality_confusion_matrices(
    audit_data: dict[str, object],
) -> None:
    quality = {row["plan_id"]: row for row in audit_data["quality_rows"]}
    primary = quality[PRIMARY_PLAN]
    assert (primary["true_positive"], primary["false_negative"], primary["false_positive"]) == (
        2,
        1,
        0,
    )
    assert primary["reference_positive_count"] == 3
    assert primary["recall"] == pytest.approx(2 / 3)
    assert primary["false_negative_rate"] == pytest.approx(1 / 3)
    assert primary["eligible"] is False

    beauty_row = next(
        row for row in audit_data["workload_rows"] if row["workload_id"] == BEAUTY_WORKLOAD
    )
    beauty = quality[beauty_row["full_batch_plan_id"]]
    assert (beauty["true_positive"], beauty["false_negative"], beauty["false_positive"]) == (
        21,
        1,
        1,
    )
    assert beauty["recall"] == pytest.approx(21 / 22)
    assert beauty["eligible"] is True

    zero_positive_full_batches = [
        row
        for row in quality.values()
        if row["plan_kind"] == "full_batch" and row["reference_positive_count"] == 0
    ]
    assert len(zero_positive_full_batches) == 4
    assert all(row["recall"] is None for row in zero_positive_full_batches)
    assert all(row["eligible"] is False for row in zero_positive_full_batches)


def test_workload_economics_and_frozen_policy(audit_data: dict[str, object]) -> None:
    aggregate = audit_data["aggregate"]
    assert aggregate["existing_adaptive_cost_usd"] == pytest.approx(0.054139134)
    assert aggregate["full_batch_cost_usd"] == pytest.approx(0.059563938)
    assert aggregate["frozen_selected_execution_cost_usd"] == pytest.approx(0.053575452)
    assert aggregate["matrix_acquisition_cost_usd"] == pytest.approx(0.161602854)
    assert audit_data["findings"]["full_batch_cheaper_workloads"] == 2
    assert audit_data["findings"]["full_batch_costlier_workloads"] == 7
    assert audit_data["findings"]["quality_admissible_improvement_over_existing_adaptive"] is False


def test_primary_bootstrap_and_sms_degenerate_interval() -> None:
    uncertainty = json.loads(
        (AUDIT / "statistical_uncertainty_verified.json").read_text(encoding="utf-8")
    )
    primary = next(
        row for row in uncertainty["workloads"] if row["workload_id"] == PRIMARY_WORKLOAD
    )
    assert primary["estimate_usd"] == pytest.approx(0.000563682)
    assert primary["lower_95_usd"] == pytest.approx(0.000294084)
    assert primary["upper_95_usd"] == pytest.approx(0.000845754)

    sms = uncertainty["sms_four_predicate_degenerate_interval_diagnosis"]
    assert sms["unique_per_record_differences"] == 1
    assert sms["constant_per_record_difference_usd"] == pytest.approx(-0.0014952 / 400)
    assert sms["bootstrap_bug_detected"] is False


def test_fallback_regret_profiling_and_spend(audit_data: dict[str, object]) -> None:
    counts = audit_data["classifications"]["counts"]
    assert counts == {
        "correct_full_batch_fallback": 7,
        "excessively_conservative_fallback": 1,
        "selected_batch_not_confirmed": 1,
    }
    assert audit_data["regret"]["quality_constrained_measured_set_total_usd"] == pytest.approx(
        0.001391082
    )

    primary_profiles = [
        row for row in audit_data["profiling"] if row["workload_id"] == PRIMARY_WORKLOAD
    ]
    costs = {
        row["scenario"]: row["incremental_profiling_cost_usd"]
        for row in primary_profiles
        if row["reuse_horizon"] == 1
    }
    assert costs == {
        "standalone_cold_start": pytest.approx(0.007067130),
        "shared_statistics_cold_start": pytest.approx(0.002967972),
        "warm_start": 0.0,
    }
    assert all(
        row["economic_claim_status"] == "QUALITY_INADMISSIBLE_ON_HELD_OUT"
        for row in primary_profiles
    )
    assert audit_data["expenditure"]["four_experiment_token_priced_total_usd"] == pytest.approx(
        0.237840078
    )


def test_historical_fingerprints_and_zero_provider_calls(audit_data: dict[str, object]) -> None:
    integrity = audit_data["integrity"]
    assert integrity["valid"] is True
    assert integrity["manifest_covered_artifacts_all_match"] is True
    assert integrity["historical_artifacts_modified"] is False
    assert integrity["banking_batch_frozen_before_stage_d"] is True
    assert integrity["new_jev_calls"] == 0
    assert integrity["provider_initialized"] is False


def test_audit_never_initializes_real_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    from semanticflow.providers.jev import JevProvider

    def fail_initialization(*args: object, **kwargs: object) -> None:
        raise AssertionError("offline audit attempted to initialize Jev")

    monkeypatch.setattr(JevProvider, "__init__", fail_initialization)
    data = build_audit_data(bootstrap_iterations=1)
    assert data["integrity"]["new_jev_calls"] == 0


def test_generator_is_idempotent(tmp_path: Path) -> None:
    output = tmp_path / "audit"
    report = tmp_path / "report.md"
    figures = tmp_path / "figures"
    generate(output, report_path=report, figures_path=figures, bootstrap_iterations=19)
    first = (tree_hash(output), tree_hash(figures), report.read_bytes())
    generate(output, report_path=report, figures_path=figures, bootstrap_iterations=19)
    second = (tree_hash(output), tree_hash(figures), report.read_bytes())
    assert first == second
    assert not any(math.isnan(path.stat().st_size) for path in output.iterdir())
