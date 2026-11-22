from __future__ import annotations

import json
from pathlib import Path

import pytest

from semanticflow.dataset.fusion_benchmark import validate_fusion_dataset
from semanticflow.optimizer.m51_analysis import diagnose_stage_b
from semanticflow.optimizer.m51_experiment import (
    STAGE_B_AUTHORIZATION,
    baseline_audit,
    build_authorization_manifest,
    build_pilot_budget,
    load_yaml,
    pilot_request_counts,
    validate_stage_b_launch,
)

WORKLOADS = "configs/milestone_5_1/workloads.yaml"
PROTOCOL = "configs/milestone_5_1/protocol.yaml"
SCHEDULE = "configs/milestone_5_1/pilot_schedule.yaml"

pytestmark = pytest.mark.research_artifacts(
    "results/milestone_5/live_fusion_pilot_20261115_stage_b_approved/session.json",
    "results/milestone_5_1/live_batching_pilot_stage_b/session.json",
)


def test_frozen_stage_b_is_reconciled_without_calls() -> None:
    audit = baseline_audit()
    assert audit["stage_b_decisions"] == 1230
    assert audit["stage_b_http_attempts"] == 1230
    assert audit["live_jev_calls"] == 0


def test_fresh_datasets_validate_and_have_frozen_partitions() -> None:
    suite = load_yaml(WORKLOADS)
    for definition in suite["datasets"].values():
        check = validate_fusion_dataset(definition["path"])
        assert check["rows"] == 590
        assert check["partitions"] == {
            "development": 50,
            "evaluation": 400,
            "pilot": 40,
            "selection": 100,
        }


def test_pilot_counts_and_budget_are_exact() -> None:
    counts = pilot_request_counts(load_yaml(SCHEDULE))
    assert counts == {
        "representative_separate_requests": 360,
        "representative_batch_requests": 270,
        "representative_conjunctive_requests": 90,
        "enriched_diagnostic_requests": 90,
        "logical_requests": 810,
        "retry_inclusive_http_attempts": 2430,
    }
    budget = build_pilot_budget(
        workloads_path=WORKLOADS, protocol_path=PROTOCOL, schedule_path=SCHEDULE
    )
    assert budget["expected_input_tokens"] > 0
    assert budget["retry_inclusive_cost_proxy_usd"] == pytest.approx(
        budget["conservative_no_retry_cost_proxy_usd"] * 3
    )


def test_authorization_fails_closed_before_provider_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = build_authorization_manifest(
        workloads_path=WORKLOADS, protocol_path=PROTOCOL, schedule_path=SCHEDULE
    )
    output = tmp_path / "new-output"
    for phrase in (None, "AUTHORIZE SOMETHING ELSE"):
        with pytest.raises(PermissionError):
            validate_stage_b_launch(
                manifest,
                authorization=phrase,
                output_directory=output,
                workloads_path=WORKLOADS,
                protocol_path=PROTOCOL,
                schedule_path=SCHEDULE,
            )
    changed = json.loads(json.dumps(manifest))
    changed["budget"]["logical_requests"] -= 1
    with pytest.raises(ValueError, match="manifest differs"):
        validate_stage_b_launch(
            changed,
            authorization=STAGE_B_AUTHORIZATION,
            output_directory=output,
            workloads_path=WORKLOADS,
            protocol_path=PROTOCOL,
            schedule_path=SCHEDULE,
        )
    monkeypatch.setattr(Path, "exists", lambda self: True)
    with pytest.raises(FileExistsError):
        validate_stage_b_launch(
            manifest,
            authorization=STAGE_B_AUTHORIZATION,
            output_directory=manifest["output_directory"],
            workloads_path=WORKLOADS,
            protocol_path=PROTOCOL,
            schedule_path=SCHEDULE,
        )


def test_stage_b_diagnostics_reproduce_negative_result() -> None:
    report = diagnose_stage_b(
        "results/milestone_5/live_fusion_pilot_20261115_stage_b_approved",
        "data/milestone_5",
    )
    assert len(report["stage_b_groups"]) == 12
    assert len(report["measured_disagreements"]) == 8
    assert len(report["multi_decision_context_disagreements"]) == 1
    assert sum(row["fused_cost_usd"] for row in report["stage_b_groups"]) > sum(
        row["separate_early_reject_cost_usd"] for row in report["stage_b_groups"]
    )
    assert all(
        not row["linguistic_cause_established"]
        for row in report["measured_disagreements"]
    )
