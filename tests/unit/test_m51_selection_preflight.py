from __future__ import annotations

import copy
import json

import pytest

from semanticflow.optimizer.m51_selection_preflight import (
    AUTHORIZATION_PHRASE,
    WORKLOADS,
    _beta_binomial_probability_less_than,
    _fingerprints,
    _operator,
    _plan_envelope,
    build_inventory,
    build_plan_catalog,
    build_plan_requirements,
    build_schedule,
    canonical_hash,
    generate,
    load_yaml,
    quality_feasibility,
    validate_schedule,
    validate_selection_launch,
)


@pytest.fixture(scope="module")
def config():
    return load_yaml(WORKLOADS)


@pytest.fixture(scope="module")
def matrix(config):
    schedule = build_schedule(config)
    catalog = build_plan_catalog(config)
    requirements = build_plan_requirements(catalog, schedule)
    return schedule, catalog, requirements


def test_all_nine_workloads_and_eighteen_predicates(config) -> None:
    inventory = build_inventory(config)
    assert len(inventory) == 9
    assert sum(len(row["predicates"]) for row in config["datasets"].values()) == 18


def test_exact_selection_partition_and_schedule_counts(matrix) -> None:
    schedule, _, _ = matrix
    assert schedule["counts"] == {
        "individual_reference_roles": 1800,
        "full_batch_roles": 900,
        "remainder_batch_roles": 600,
        "logical_measurement_roles": 3300,
        "unique_physical_requests": 3000,
        "shared_alias_roles": 300,
        "proposal_retry_inclusive_attempt_ceiling": 9900,
        "deduplicated_retry_inclusive_attempt_ceiling": 9000,
    }
    assert {row["partition"] for row in schedule["physical_requests"]} == {"selection"}


def test_complete_coverage_and_zero_reach_rule(config, matrix) -> None:
    schedule, catalog, requirements = matrix
    result = validate_schedule(config, schedule, catalog, requirements)
    assert result["valid"]
    assert result["fully_scheduled_catalog_plans"] == 111
    assert all(row["zero_reach_does_not_waive_measurement"] for row in requirements)


@pytest.mark.parametrize("field", ["dataset", "record_id", "prompt_hash", "predicate_ids"])
def test_incompatible_role_is_detected(config, matrix, field) -> None:
    schedule, catalog, requirements = matrix
    broken = copy.deepcopy(schedule)
    if field == "predicate_ids":
        broken["measurement_roles"][0][field] = ["wrong"]
    else:
        broken["measurement_roles"][0][field] = "wrong"
    result = validate_schedule(config, broken, catalog, requirements)
    assert not result["valid"]


def test_duplicate_request_identity_is_detected(config, matrix) -> None:
    schedule, catalog, requirements = matrix
    broken = copy.deepcopy(schedule)
    broken["physical_requests"].append(copy.deepcopy(broken["physical_requests"][0]))
    assert not validate_schedule(config, broken, catalog, requirements)["valid"]


def test_plan_objects_cover_every_predicate_once(matrix) -> None:
    _, catalog, _ = matrix
    for plan in catalog:
        flattened = [p for group in plan["groups"] for p in group["predicate_ids"]]
        assert len(flattened) == len(set(flattened))
        assert set(flattened) == set(plan["logical_predicate_ids"])


def test_plan_hash_is_deterministic_and_sensitive(config) -> None:
    workload = config["workloads"][0]
    group = _operator(config, workload["dataset"], workload["predicate_ids"], "multi_decision")
    first = _plan_envelope(config, workload, [group], ["x"], "config")
    second = _plan_envelope(config, workload, [group], ["y"], "config")
    assert first["plan_fingerprint"] == second["plan_fingerprint"]
    changed = group.model_copy(update={"threshold": 0.6})
    third = _plan_envelope(config, workload, [changed], ["x"], "config")
    assert first["plan_fingerprint"] != third["plan_fingerprint"]


def test_every_plan_maps_to_measurements(matrix) -> None:
    _, catalog, requirements = matrix
    assert len(catalog) == len(requirements) == 111
    assert all(row["evidence_status"] == "fully_scheduled" for row in requirements)


def test_missing_group_is_unsupported(matrix) -> None:
    schedule, catalog, _ = matrix
    broken = copy.deepcopy(schedule)
    prompt = catalog[0]["groups"][0]["prompt_hash"]
    broken["physical_requests"] = [
        row for row in broken["physical_requests"] if row["prompt_hash"] != prompt
    ]
    requirements = build_plan_requirements(catalog, broken)
    assert any(row["evidence_status"] == "unsupported" for row in requirements)


def test_quality_feasibility_preserves_sparse_and_missing_support(config) -> None:
    result = quality_feasibility(config)
    assert result["quality_gate"]["minimum_records"] == 40
    assert result["quality_gate"]["minimum_reference_positives"] == 3
    assert sum(row["expected_selection_positives"] is None for row in result["workloads"]) == 3
    assert any(
        row.get("probability_fewer_than_three", 0) > 0.7
        for row in result["workloads"]
        if row["expected_selection_positives"] is not None
    )


def test_beta_binomial_sparse_support() -> None:
    probability = _beta_binomial_probability_less_than(100, 3, 0.5, 30.5)
    assert 0 < probability < 1


def test_canonical_hash_ignores_mapping_order() -> None:
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})


def test_launch_gate_rejects_no_authorization(config, matrix, tmp_path) -> None:
    schedule, _, _ = matrix
    manifest = {
        "paid_calls_authorized": False,
        "schedule_fingerprint": schedule["schedule_fingerprint"],
        "requested_model": "jev-1.14.0",
        "fingerprints": _fingerprints(config, schedule),
    }
    with pytest.raises(PermissionError):
        validate_selection_launch(
            manifest,
            authorization=AUTHORIZATION_PHRASE,
            schedule=schedule,
            output_directory=tmp_path / "live",
        )


def test_launch_gate_rejects_fingerprint_mismatch(config, matrix, tmp_path) -> None:
    schedule, _, _ = matrix
    manifest = {
        "paid_calls_authorized": True,
        "schedule_fingerprint": "wrong",
        "requested_model": "jev-1.14.0",
        "fingerprints": _fingerprints(config, schedule),
    }
    with pytest.raises(ValueError):
        validate_selection_launch(
            manifest,
            authorization=AUTHORIZATION_PHRASE,
            schedule=schedule,
            output_directory=tmp_path / "live",
        )


def test_launch_gate_rejects_output_conflict(config, matrix, tmp_path) -> None:
    schedule, _, _ = matrix
    output = tmp_path / "live"
    output.mkdir()
    manifest = {
        "paid_calls_authorized": True,
        "schedule_fingerprint": schedule["schedule_fingerprint"],
        "requested_model": "jev-1.14.0",
        "fingerprints": _fingerprints(config, schedule),
    }
    with pytest.raises(FileExistsError):
        validate_selection_launch(
            manifest,
            authorization=AUTHORIZATION_PHRASE,
            schedule=schedule,
            output_directory=output,
        )


def test_generate_is_non_overwriting_and_offline(tmp_path) -> None:
    output = tmp_path / "preflight"
    summary = generate(output)
    assert summary["live_jev_calls"] == 0
    assert (
        json.loads((output / "authorization_manifest.json").read_text())["paid_calls_authorized"]
        is False
    )
    with pytest.raises(FileExistsError):
        generate(output)
