from __future__ import annotations

import copy
import json
import sys

import pytest

from experiments.milestone_5_1.run_hybrid import _predicates
from experiments.milestone_5_1.run_stage_d import _smoke_request_ids, _validate_operator
from semanticflow.optimizer.fusion import build_fused_operator
from semanticflow.optimizer.m51_experiment import load_yaml
from semanticflow.optimizer.m51_stage_d_analysis import dry_run_contract
from semanticflow.optimizer.m51_stage_d_preflight import (
    AUTHORIZATION_PATH,
    CATALOG_PATH,
    PRIMARY_PLAN_FINGERPRINT,
    PRIMARY_PLAN_ID,
    SCHEDULE_PATH,
    SELECTIONS_PATH,
    WORKLOADS_PATH,
    _authorization_scope,
    build_budget,
    build_reduced_option,
    canonical_hash,
    reconcile_schedule,
    validate_frozen_plans,
    validate_partition_integrity,
    validate_stage_d_launch,
)

pytestmark = pytest.mark.research_artifacts(
    "results/milestone_5_1/selection_stage_results/frozen_strategy_selections.json",
)


@pytest.fixture(scope="module")
def evidence() -> dict:
    return {
        "schedule": json.loads(SCHEDULE_PATH.read_text("utf-8")),
        "catalog": json.loads(CATALOG_PATH.read_text("utf-8"))["plans"],
        "selections": json.loads(SELECTIONS_PATH.read_text("utf-8"))["strategies"],
        "config": load_yaml(WORKLOADS_PATH),
        "authorization": json.loads(AUTHORIZATION_PATH.read_text("utf-8")),
    }


def _authorized_manifest(
    evidence: dict, output: str = "results/milestone_5_1/stage_d_live"
) -> dict:
    scope = _authorization_scope(
        evidence["schedule"], evidence["authorization"]["authorization_phrase"]
    )
    scope["output_directory"] = output.replace("\\", "/")
    return {**scope, "scope_fingerprint": canonical_hash(scope), "paid_calls_authorized": True}


def test_all_nine_frozen_selections_and_primary_batch(evidence) -> None:
    validation, mapping = validate_frozen_plans(
        evidence["catalog"], evidence["selections"], evidence["config"]
    )
    assert validation["valid"], validation["errors"]
    assert len(mapping) == 9
    assert validation["quality_constrained_batches"] == 1
    assert validation["existing_adaptive_fallbacks"] == 8
    primary = next(row for row in mapping if row["selected_plan_id"] == PRIMARY_PLAN_ID)
    assert primary["selected_plan_fingerprint"] == PRIMARY_PLAN_FINGERPRINT
    assert primary["group_execution_modes"] == ["multi_decision"]
    assert len(primary["question_names"][0]) == 3


def test_plan_hash_changes_with_relevant_execution_configuration(evidence) -> None:
    plan = next(row for row in evidence["catalog"] if row["plan_id"] == PRIMARY_PLAN_ID)
    broken = copy.deepcopy(plan)
    broken["threshold"] = 0.51
    from semanticflow.optimizer.m51_stage_d_preflight import _plan_execution_payload

    assert canonical_hash(_plan_execution_payload(plan)) == PRIMARY_PLAN_FINGERPRINT
    assert canonical_hash(_plan_execution_payload(broken)) != PRIMARY_PLAN_FINGERPRINT


def test_actual_schedule_is_6000_singles_and_4800_batches(evidence) -> None:
    reconciliation, purpose, coverage, primary = reconcile_schedule(
        evidence["schedule"],
        evidence["catalog"],
        evidence["selections"],
        evidence["config"],
    )
    assert reconciliation["valid"], reconciliation["errors"]
    assert reconciliation["physical_requests"] == 10800
    assert reconciliation["single_reference_requests"] == 6000
    assert reconciliation["multi_decision_requests"] == 4800
    assert reconciliation["full_batch_requests"] == 3600
    assert reconciliation["additional_unique_remainder_batch_requests"] == 1200
    assert reconciliation["unique_operator_arms"] == 27
    assert reconciliation["deliberate_repetitions"] == 0
    assert sum(row["unique_physical_requests"] for row in purpose["purposes"]) == 10800
    assert all(row["fully_measurable"] for row in coverage)
    assert len(primary) == 4


def test_all_scheduled_operator_arms_rebuild_against_frozen_contract(evidence) -> None:
    operators = {}
    for row in evidence["schedule"]["physical_requests"]:
        operators.setdefault(row["operator_id"], row)
    assert len(operators) == 27
    for row in operators.values():
        operator = build_fused_operator(
            _predicates(evidence["config"], row["dataset"], list(row["predicate_ids"])),
            mode=row["execution_mode"],
            requested_model=row["model"],
            threshold=float(row["threshold"]),
            execution_parameters=row["inference_settings"],
        )
        _validate_operator(row, operator)


def test_strategy_aliases_are_not_duplicate_paid_requests(evidence) -> None:
    reconciliation, _, _, _ = reconcile_schedule(
        evidence["schedule"],
        evidence["catalog"],
        evidence["selections"],
        evidence["config"],
    )
    assert reconciliation["unique_plan_group_record_roles"] == 26400
    assert reconciliation["unique_plan_aliases"] == 15600
    assert reconciliation["strategy_group_record_roles_maximum"] == 60400
    assert reconciliation["strategy_aliases_maximum"] == 49600


def test_reduced_primary_option_is_separate_and_complete(evidence) -> None:
    _, _, _, primary = reconcile_schedule(
        evidence["schedule"],
        evidence["catalog"],
        evidence["selections"],
        evidence["config"],
    )
    reduced_schedule, reduced = build_reduced_option(evidence["schedule"], primary)
    assert reduced["unique_physical_requests"] == 1600
    assert reduced["maximum_http_attempts"] == 4800
    assert reduced["composition"] == {
        "individual_reference_requests": 1200,
        "three_question_batch_requests": 400,
    }
    assert reduced_schedule["authorization_phrase"] is None
    assert reduced_schedule["schedule_fingerprint"] != evidence["schedule"]["schedule_fingerprint"]


def test_evaluation_partition_metadata_is_disjoint_without_outcomes(evidence) -> None:
    report = validate_partition_integrity(evidence["config"], evidence["schedule"])
    assert report["valid"], report["errors"]
    assert all(row["evaluation_ids"] == 400 for row in report["datasets"])
    assert all(not any(row["partition_overlap_counts"].values()) for row in report["datasets"])
    assert report["held_out_record_contents_accessed"] is False
    assert report["held_out_inference_outcomes_accessed"] is False


def test_full_budget_reconciles_exactly(evidence) -> None:
    budget = build_budget(evidence["schedule"], evidence["config"])
    assert budget["unique_physical_requests"] == 10800
    assert budget["maximum_http_attempts"] == 32400
    assert budget["expected_input_tokens"] == 3887444
    assert budget["expected_output_tokens"] == 528800
    assert budget["expected_token_priced_cost_usd"] == pytest.approx(0.163272648)
    assert all(
        value is True
        for key, value in budget["reconciliation"].items()
        if key.endswith("_match")
    )
    assert all(
        value == pytest.approx(0.0)
        for key, value in budget["reconciliation"].items()
        if key.endswith("_usd")
    )


def test_launch_gate_accepts_only_countersigned_exact_scope(evidence, tmp_path) -> None:
    output = tmp_path / "stage_d"
    manifest = _authorized_manifest(evidence, str(output))
    provider_module_before = sys.modules.get("semanticflow.providers.jev")
    validate_stage_d_launch(
        manifest,
        authorization=manifest["authorization_phrase"],
        schedule=evidence["schedule"],
        catalog=evidence["catalog"],
        selections=evidence["selections"],
        output_directory=output,
    )
    assert sys.modules.get("semanticflow.providers.jev") is provider_module_before


@pytest.mark.parametrize("authorization", [None, "wrong authorization"])
def test_launch_gate_rejects_missing_or_wrong_authorization(
    evidence, tmp_path, authorization
) -> None:
    output = tmp_path / "stage_d"
    manifest = _authorized_manifest(evidence, str(output))
    with pytest.raises(PermissionError):
        validate_stage_d_launch(
            manifest,
            authorization=authorization,
            schedule=evidence["schedule"],
            catalog=evidence["catalog"],
            selections=evidence["selections"],
            output_directory=output,
        )


def test_launch_gate_rejects_unsigned_manifest(evidence, tmp_path) -> None:
    output = tmp_path / "stage_d"
    manifest = _authorized_manifest(evidence, str(output))
    manifest["paid_calls_authorized"] = False
    with pytest.raises(PermissionError):
        validate_stage_d_launch(
            manifest,
            authorization=manifest["authorization_phrase"],
            schedule=evidence["schedule"],
            catalog=evidence["catalog"],
            selections=evidence["selections"],
            output_directory=output,
        )


def test_launch_gate_rejects_changed_schedule_prompt_and_attempt_budget(evidence, tmp_path) -> None:
    output = tmp_path / "stage_d"
    manifest = _authorized_manifest(evidence, str(output))
    broken_schedule = copy.deepcopy(evidence["schedule"])
    broken_schedule["physical_requests"][0]["prompt_hash"] = "changed"
    with pytest.raises(ValueError):
        validate_stage_d_launch(
            manifest,
            authorization=manifest["authorization_phrase"],
            schedule=broken_schedule,
            catalog=evidence["catalog"],
            selections=evidence["selections"],
            output_directory=output,
        )
    broken_manifest = copy.deepcopy(manifest)
    broken_manifest["http_attempt_ceiling"] += 1
    with pytest.raises(ValueError):
        validate_stage_d_launch(
            broken_manifest,
            authorization=broken_manifest["authorization_phrase"],
            schedule=evidence["schedule"],
            catalog=evidence["catalog"],
            selections=evidence["selections"],
            output_directory=tmp_path / "stage_d",
        )


def test_launch_gate_rejects_changed_plan_and_dataset_fingerprint(evidence, tmp_path) -> None:
    output = tmp_path / "stage_d"
    manifest = _authorized_manifest(evidence, str(output))
    broken_catalog = copy.deepcopy(evidence["catalog"])
    plan = next(row for row in broken_catalog if row["plan_id"] == PRIMARY_PLAN_ID)
    plan["groups"][0]["execution_mode"] = "single"
    with pytest.raises(ValueError):
        validate_stage_d_launch(
            manifest,
            authorization=manifest["authorization_phrase"],
            schedule=evidence["schedule"],
            catalog=broken_catalog,
            selections=evidence["selections"],
            output_directory=output,
        )
    broken_manifest = copy.deepcopy(manifest)
    broken_manifest["scope_fingerprints"]["datasets"]["banking77_queries"][
        "evaluation_partition_fingerprint"
    ] = "changed"
    scope = {
        key: value
        for key, value in broken_manifest.items()
        if key not in {"paid_calls_authorized", "scope_fingerprint"}
    }
    broken_manifest["scope_fingerprint"] = canonical_hash(scope)
    with pytest.raises(ValueError):
        validate_stage_d_launch(
            broken_manifest,
            authorization=broken_manifest["authorization_phrase"],
            schedule=evidence["schedule"],
            catalog=evidence["catalog"],
            selections=evidence["selections"],
            output_directory=output,
        )


def test_launch_gate_rejects_alias_conflict_and_existing_output(evidence, tmp_path) -> None:
    output = tmp_path / "stage_d"
    manifest = _authorized_manifest(evidence, str(output))
    broken = copy.deepcopy(evidence["schedule"])
    broken["physical_requests"][0]["selected_plan_references"] = [PRIMARY_PLAN_ID]
    broken["schedule_fingerprint"] = canonical_hash(
        {key: value for key, value in broken.items() if key != "schedule_fingerprint"}
    )
    with pytest.raises(ValueError):
        validate_stage_d_launch(
            manifest,
            authorization=manifest["authorization_phrase"],
            schedule=broken,
            catalog=evidence["catalog"],
            selections=evidence["selections"],
            output_directory=output,
        )
    output = tmp_path / "existing"
    output.mkdir()
    manifest = _authorized_manifest(evidence, str(output))
    with pytest.raises(FileExistsError):
        validate_stage_d_launch(
            manifest,
            authorization=manifest["authorization_phrase"],
            schedule=evidence["schedule"],
            catalog=evidence["catalog"],
            selections=evidence["selections"],
            output_directory=output,
        )


def test_smoke_requests_are_frozen_members_for_all_batch_sizes(evidence) -> None:
    smoke = _smoke_request_ids(evidence["schedule"])
    scheduled = {row["request_id"] for row in evidence["schedule"]["physical_requests"]}
    assert len(smoke) == 4
    assert set(smoke) <= scheduled


def test_post_live_analyzer_contract_dry_run_uses_only_mock_data() -> None:
    result = dry_run_contract()
    assert result["mock_only"] is True
    assert result["binary_metrics_valid"] is True
    assert result["paired_bootstrap_has_bounds"] is True
    assert result["held_out_outcomes_accessed"] is False
