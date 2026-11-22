# ruff: noqa: E501
from __future__ import annotations

import csv
import hashlib
import html
import itertools
import json
import math
import random
import statistics
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from semanticflow.optimizer.m51_experiment import load_yaml
from semanticflow.optimizer.m51_selection_analysis import (
    INPUT_PRICE,
    _gate_for_plan,
    _plan_kind,
    canonical_hash,
    predict_plan,
    replay_plan,
)
from semanticflow.optimizer.m51_selection_analysis import (
    load_evidence as load_selection_evidence,
)
from semanticflow.optimizer.m51_stage_d_analysis import normalize_stage_d_session
from semanticflow.optimizer.m51_stage_d_preflight import (
    _schedule_payload,
    validate_frozen_plans,
    validate_partition_integrity,
)

ROOT = Path("results/milestone_5_1/final_research_audit")
REPORT = Path("docs/milestone_5_1_final_research_report.md")
FIGURES = Path("docs/figures/milestone_5_1_final")
STAGE_D_SESSION = Path("results/milestone_5_1/stage_d_live_v3")
STAGE_D_RESULTS = Path("results/milestone_5_1/stage_d_results")
STAGE_D_PREFLIGHT = Path("results/milestone_5_1/stage_d_final_preflight")
SELECTION_RESULTS = Path("results/milestone_5_1/selection_stage_results")
SELECTION_PREFLIGHT = Path("results/milestone_5_1/selection_stage_preflight")
STAGE_C_RESULTS = Path("results/milestone_5_1/stage_c_detailed_analysis")
STAGE_B_SESSION = Path("results/milestone_5_1/live_batching_pilot_stage_b")
M5_SESSION = Path("results/milestone_5/live_fusion_pilot_20261115_stage_b_approved")
SCHEDULE_PATH = SELECTION_RESULTS / "stage_d_execution_schedule.json"
CATALOG_PATH = SELECTION_RESULTS / "frozen_physical_plan_catalog.json"
SELECTIONS_PATH = SELECTION_RESULTS / "frozen_strategy_selections.json"
WORKLOADS_PATH = Path("configs/milestone_5_1/workloads.yaml")
PROTOCOL_PATH = Path("configs/milestone_5_1/protocol.yaml")
HORIZONS = (1, 5, 10, 25, 50, 100, 250, 1000)
PRIMARY_WORKLOAD = "m51_banking_card_unauthorized_refund"
PRIMARY_PLAN = "m51selplan_4e05511d49f2861f7dfb3361"
BEAUTY_WORKLOAD = "m51_beauty_failure_usability"
BEAUTY_BATCH = "m51selplan_380e432a3fa37511d27e0a1e"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
        digest.update(b"\0")
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text("utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def _strategy_map(
    selections: Sequence[Mapping[str, Any]], strategy: str
) -> dict[str, Mapping[str, Any]]:
    return {
        str(row["workload_id"]): row
        for row in selections
        if row.get("strategy") == strategy and row.get("status") == "selected"
    }


def _records_by_dataset(
    normalized: Sequence[Mapping[str, Any]], datasets: Sequence[str]
) -> dict[str, list[str]]:
    return {
        dataset: sorted({str(row["record_id"]) for row in normalized if row["dataset"] == dataset})
        for dataset in datasets
    }


def _paired_bootstrap(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    seed: int = 51103,
    iterations: int = 10000,
) -> dict[str, Any]:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("paired bootstrap requires aligned non-empty record vectors")
    differences = [left - right for left, right in zip(baseline, candidate, strict=True)]
    rng = random.Random(seed)
    n = len(differences)
    draws = [sum(differences[rng.randrange(n)] for _ in range(n)) for _ in range(iterations)]
    draws.sort()
    return {
        "estimate_usd": sum(differences),
        "lower_95_usd": draws[int(0.025 * iterations)],
        "upper_95_usd": draws[min(iterations - 1, int(0.975 * iterations))],
        "method": f"paired record bootstrap, {iterations} draws",
        "seed": seed,
        "sampling_unit": "held-out record with all plan costs kept together",
        "nominal_interval": True,
        "multiple_comparison_adjusted": False,
    }


def _ranking_concordance(predicted: Mapping[str, float], observed: Mapping[str, float]) -> float:
    score = comparisons = 0.0
    for left, right in itertools.combinations(sorted(predicted), 2):
        p_delta = predicted[left] - predicted[right]
        o_delta = observed[left] - observed[right]
        if p_delta == 0 or o_delta == 0:
            score += 0.5
        elif (p_delta < 0) == (o_delta < 0):
            score += 1.0
        comparisons += 1.0
    return score / comparisons if comparisons else 1.0


def _mean(values: Sequence[float]) -> float | None:
    return statistics.mean(values) if values else None


def _metrics(errors: Sequence[float], observed: Sequence[float]) -> dict[str, Any]:
    return {
        "count": len(errors),
        "mae_usd": _mean([abs(value) for value in errors]),
        "rmse_usd": math.sqrt(statistics.mean(value * value for value in errors))
        if errors
        else None,
        "mape": _mean(
            [abs(error) / actual for error, actual in zip(errors, observed, strict=True) if actual]
        ),
        "signed_bias_usd": _mean(list(errors)),
    }


def _manifest_audit(manifest_path: Path, artifact_root: Path) -> dict[str, Any]:
    manifest = _json(manifest_path)
    expected = manifest.get("artifact_sha256", {})
    rows = []
    for name, digest in sorted(expected.items()):
        path = artifact_root / name
        current = sha256_file(path) if path.exists() else None
        rows.append(
            {
                "path": path.as_posix(),
                "expected_sha256": digest,
                "current_sha256": current,
                "match": current == digest,
            }
        )
    return {
        "manifest": manifest_path.as_posix(),
        "artifacts_checked": len(rows),
        "all_match": all(row["match"] for row in rows),
        "artifacts": rows,
    }


def _historical_integrity(
    schedule: Mapping[str, Any],
    catalog: Sequence[Mapping[str, Any]],
    selections: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    selection_manifest = _manifest_audit(
        SELECTION_RESULTS / "artifact_integrity_manifest.json", SELECTION_RESULTS
    )
    stage_c_manifest = _json(STAGE_C_RESULTS / "analysis_manifest.json")
    stage_c_rows = []
    for name, info in sorted(stage_c_manifest["artifacts"].items()):
        path = STAGE_C_RESULTS / name
        current = sha256_file(path) if path.exists() else None
        stage_c_rows.append(
            {
                "path": path.as_posix(),
                "expected_sha256": info["sha256"],
                "current_sha256": current,
                "match": current == info["sha256"],
            }
        )
    preflight_manifest = _json(SELECTION_PREFLIGHT / "analysis_manifest.json")
    scope = preflight_manifest["fingerprints"]
    datasets = []
    for dataset, definition in config["datasets"].items():
        data_path = Path(definition["path"])
        manifest_path = Path(str(data_path) + ".manifest.json")
        expected = scope["datasets"][dataset]
        manifest = _json(manifest_path)
        datasets.append(
            {
                "dataset": dataset,
                "dataset_file_sha256": sha256_file(data_path),
                "expected_dataset_file_sha256": expected["dataset_file_sha256"],
                "manifest_sha256": sha256_file(manifest_path),
                "expected_manifest_sha256": expected["manifest_sha256"],
                "dataset_fingerprint": manifest["dataset_fingerprint_sha256"],
                "expected_dataset_fingerprint": expected["dataset_fingerprint"],
                "partition_counts": {
                    name: len(value["ids"]) for name, value in manifest["partitions"].items()
                },
                "partitions_disjoint": all(
                    not (set(left["ids"]) & set(right["ids"]))
                    for left, right in itertools.combinations(manifest["partitions"].values(), 2)
                ),
            }
        )
    plan_validation, _ = validate_frozen_plans(catalog, selections, config)
    partition_validation = validate_partition_integrity(config, schedule)
    authorized = _json(
        STAGE_D_PREFLIGHT / "stage_d_authorization_manifest_amendment_v3_authorized.json"
    )
    stage_b_session_hash = sha256_file(STAGE_B_SESSION / "session.json")
    stage_c_manifest_hash = sha256_file(STAGE_C_RESULTS / "analysis_manifest.json")
    session = _json(STAGE_D_SESSION / "session.json")
    all_checks = [
        selection_manifest["all_match"],
        all(row["match"] for row in stage_c_rows),
        all(
            row["dataset_file_sha256"] == row["expected_dataset_file_sha256"]
            and row["manifest_sha256"] == row["expected_manifest_sha256"]
            and row["dataset_fingerprint"] == row["expected_dataset_fingerprint"]
            and row["partitions_disjoint"]
            for row in datasets
        ),
        canonical_hash(_schedule_payload(schedule)) == schedule["schedule_fingerprint"],
        plan_validation["valid"],
        partition_validation["valid"],
        authorized["schedule_fingerprint"] == schedule["schedule_fingerprint"],
        authorized["scope_fingerprint"] == session["authorization_scope_fingerprint"],
        stage_b_session_hash == scope["stage_b_session_sha256"],
        stage_c_manifest_hash == scope["stage_c_manifest_sha256"],
    ]
    git_status = [
        row
        for row in _git("status", "--short").splitlines()
        if not any(
            marker in row.replace("\\", "/")
            for marker in (
                "results/milestone_5_1/final_research_audit",
                "docs/milestone_5_1_final_research_report.md",
                "docs/figures/milestone_5_1_final",
            )
        )
    ]
    return {
        "schema_version": "m5.1-final-historical-integrity-audit-v1",
        "git_branch": _git("branch", "--show-current"),
        "git_head": _git("rev-parse", "HEAD"),
        "git_status_context_excluding_generated_audit_outputs": git_status,
        "selection_result_manifest": selection_manifest,
        "stage_c_manifest": {
            "manifest": (STAGE_C_RESULTS / "analysis_manifest.json").as_posix(),
            "artifacts_checked": len(stage_c_rows),
            "all_match": all(row["match"] for row in stage_c_rows),
            "artifacts": stage_c_rows,
        },
        "datasets": datasets,
        "schedule": {
            "stored_fingerprint": schedule["schedule_fingerprint"],
            "recomputed_fingerprint": canonical_hash(_schedule_payload(schedule)),
            "physical_requests": len(schedule["physical_requests"]),
        },
        "plan_validation": plan_validation,
        "partition_validation": partition_validation,
        "stage_d_authorized_scope_matches_session": (
            authorized["scope_fingerprint"] == session["authorization_scope_fingerprint"]
        ),
        "banking_batch_frozen_before_stage_d": any(
            row.get("strategy") == "quality_constrained_hybrid"
            and row.get("workload_id") == PRIMARY_WORKLOAD
            and row.get("plan_id") == PRIMARY_PLAN
            for row in selections
        ),
        "quality_gate_hash_at_selection": scope["protocol_sha256"],
        "current_protocol_sha256": sha256_file(PROTOCOL_PATH),
        "stage_b_session_sha256": stage_b_session_hash,
        "stage_c_manifest_sha256": stage_c_manifest_hash,
        "historical_artifacts_modified": False,
        "manifest_covered_artifacts_all_match": all(all_checks),
        "new_jev_calls": 0,
        "provider_initialized": False,
        "valid": all(all_checks),
    }


def _request_reconciliation(
    normalized: Sequence[Mapping[str, Any]],
    schedule: Mapping[str, Any],
    catalog: Sequence[Mapping[str, Any]],
    selections: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    schedule_by_id = {row["request_id"]: row for row in schedule["physical_requests"]}
    observed_ids = [row["request_id"] for row in normalized]
    full_strategy = _strategy_map(selections, "full_multi_decision_batch")
    plan_by_id = {plan["plan_id"]: plan for plan in catalog}
    full_operators = {
        plan_by_id[row["plan_id"]]["groups"][0]["operator_id"] for row in full_strategy.values()
    }
    per_request = []
    for row in normalized:
        scheduled = schedule_by_id[row["request_id"]]
        if row["execution_mode"] == "single":
            request_type = "separate_reference"
        elif row["operator_id"] in full_operators:
            request_type = "full_batch"
        else:
            request_type = "hybrid_remainder_batch"
        per_request.append(
            {
                "request_id": row["request_id"],
                "dataset": row["dataset"],
                "record_id": row["record_id"],
                "operator_id": row["operator_id"],
                "request_type": request_type,
                "execution_mode": row["execution_mode"],
                "named_question_evaluations": row["named_question_evaluations"],
                "input_tokens": row["input_tokens"],
                "output_tokens": row["output_tokens"],
                "cost_usd": row["cost_usd"],
                "http_attempts": row["http_attempts"],
                "resolved_model": row["resolved_model"],
                "response_status": row["response_status"],
                "selected_plan_reference_count": len(scheduled["selected_plan_references"]),
            }
        )
    per_operator = []
    by_operator: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in per_request:
        by_operator[(str(row["dataset"]), str(row["operator_id"]))].append(row)
    for (dataset, operator_id), rows in sorted(by_operator.items()):
        per_operator.append(
            {
                "dataset": dataset,
                "operator_id": operator_id,
                "request_type": rows[0]["request_type"],
                "execution_mode": rows[0]["execution_mode"],
                "requests": len(rows),
                "named_question_evaluations": sum(
                    int(row["named_question_evaluations"]) for row in rows
                ),
                "input_tokens": sum(int(row["input_tokens"]) for row in rows),
                "output_tokens": sum(int(row["output_tokens"]) for row in rows),
                "cost_usd": sum(float(row["cost_usd"]) for row in rows),
            }
        )
    composition = Counter(str(row["request_type"]) for row in per_request)
    expected_keys = {
        (row["request_id"], row["dataset"], row["record_id"], row["operator_id"])
        for row in schedule["physical_requests"]
    }
    observed_keys = {
        (row["request_id"], row["dataset"], row["record_id"], row["operator_id"])
        for row in per_request
    }
    report = {
        "schema_version": "m5.1-final-request-reconciliation-v1",
        "source_session": STAGE_D_SESSION.as_posix(),
        "scheduled_requests": len(schedule["physical_requests"]),
        "physical_requests": len(per_request),
        "unique_request_ids": len(set(observed_ids)),
        "duplicate_request_ids": sorted(
            request_id for request_id, count in Counter(observed_ids).items() if count > 1
        ),
        "missing_request_keys": [list(value) for value in sorted(expected_keys - observed_keys)],
        "unexpected_request_keys": [list(value) for value in sorted(observed_keys - expected_keys)],
        "http_attempts": sum(int(row["http_attempts"]) for row in per_request),
        "retries": sum(int(row["http_attempts"]) - 1 for row in per_request),
        "named_question_evaluations": sum(
            int(row["named_question_evaluations"]) for row in per_request
        ),
        "input_tokens": sum(int(row["input_tokens"]) for row in per_request),
        "output_tokens": sum(int(row["output_tokens"]) for row in per_request),
        "token_priced_cost_usd": sum(float(row["cost_usd"]) for row in per_request),
        "pricing": {"input_per_million_usd": INPUT_PRICE, "output_per_million_usd": 0.0},
        "provider_invoice_established": False,
        "request_composition": dict(sorted(composition.items())),
        "logical_plan_group_roles": sum(
            int(row["selected_plan_reference_count"]) for row in per_request
        ),
        "resolved_models": sorted({str(row["resolved_model"]) for row in per_request}),
        "http_statuses": sorted({str(row["response_status"]) for row in per_request}),
        "all_complete_and_valid": (
            len(per_request) == len(schedule["physical_requests"])
            and len(set(observed_ids)) == len(per_request)
            and expected_keys == observed_keys
            and all(row["resolved_model"] == "jev-1.14.0" for row in per_request)
            and all(int(row["http_attempts"]) == 1 for row in per_request)
        ),
        "prior_failed_transport_attempts": 3,
        "cumulative_attempt_ledger": sum(int(row["http_attempts"]) for row in per_request) + 3,
        "prior_failed_attempt_billing_status": "unknown",
    }
    return report, per_request, per_operator


def _deduplicate_original_quality(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    original = _json(STAGE_D_RESULTS / "quality_gate_results.json")
    dataset_by_workload = {row["id"]: row["dataset"] for row in config["workloads"]}
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(original["plans"]):
        key = (
            dataset_by_workload[row["workload_id"]],
            row["workload_id"],
            row["plan_id"],
            "evaluation",
        )
        grouped[key].append({"index": index, "row": row})
    canonical = []
    duplicates = []
    for key, entries in sorted(grouped.items()):
        roles = []
        for entry in entries:
            row = entry["row"]
            roles.append(
                "full_batch_candidate" if entry["index"] % 2 == 0 else "frozen_selected_plan"
            )
        base = dict(entries[0]["row"])
        base.update(
            {
                "dataset": key[0],
                "partition": key[3],
                "evaluation_roles": sorted(set(roles)),
                "source_row_indices": [entry["index"] for entry in entries],
            }
        )
        canonical.append(base)
        if len(entries) > 1:
            comparison_fields = sorted(entries[0]["row"])
            identical = all(entry["row"] == entries[0]["row"] for entry in entries[1:])
            duplicates.append(
                {
                    "dataset": key[0],
                    "workload_id": key[1],
                    "plan_id": key[2],
                    "partition": key[3],
                    "source_row_indices": [entry["index"] for entry in entries],
                    "fields_compared": comparison_fields,
                    "rows_identical": identical,
                    "physical_inference_duplicated": False,
                    "statistical_weight_duplicated": False,
                    "cause": (
                        "Stage D analysis appended full_gate and selected_gate even when the "
                        "selected plan was the same full-batch plan"
                    ),
                }
            )
    audit = {
        "schema_version": "m5.1-quality-deduplication-audit-v1",
        "source_artifact": (STAGE_D_RESULTS / "quality_gate_results.json").as_posix(),
        "original_rows": len(original["plans"]),
        "original_unique_plan_ids": len({row["plan_id"] for row in original["plans"]}),
        "canonical_rows": len(canonical),
        "canonical_unique_plan_ids": len({row["plan_id"] for row in canonical}),
        "duplicate_groups": duplicates,
        "quality_classifications_changed": False,
        "policy_conclusions_changed": False,
        "correction_scope": "reporting rows only; historical artifact preserved",
    }
    return {"gate": original["gate"], "plans": canonical}, audit


def _quality_for_all_measured_plans(
    workloads: Sequence[Mapping[str, Any]],
    schedule: Mapping[str, Any],
    catalog: Sequence[Mapping[str, Any]],
    selections: Sequence[Mapping[str, Any]],
    records_by_dataset: Mapping[str, Sequence[str]],
    lookup: Mapping[tuple[str, str], Mapping[str, Any]],
    gate: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]], dict[str, Any]]:
    plan_by_id = {str(plan["plan_id"]): plan for plan in catalog}
    adaptive = _strategy_map(selections, "existing_adaptive")
    results = []
    replayed: dict[str, Mapping[str, Any]] = {}
    references = {}
    for workload in workloads:
        workload_id = str(workload["id"])
        record_ids = list(records_by_dataset[workload["dataset"]])
        adaptive_plan = plan_by_id[str(adaptive[workload_id]["plan_id"])]
        adaptive_result = replay_plan(adaptive_plan, record_ids, lookup)
        reference = [bool(row["passed"]) for row in adaptive_result["per_record"]]
        references[workload_id] = reference
        for plan_id in schedule["selected_plan_ids"]:
            plan = plan_by_id[str(plan_id)]
            if plan["workload_id"] != workload_id:
                continue
            replayed_plan = replay_plan(plan, record_ids, lookup)
            replayed[str(plan_id)] = replayed_plan
            quality = _gate_for_plan(plan, replayed_plan, reference, gate)
            quality.update(
                {
                    "dataset": workload["dataset"],
                    "partition": "evaluation",
                    "evaluation_role": "measured_frozen_plan_union",
                }
            )
            results.append(quality)
    return results, replayed, references


def _semantic_disagreements(
    workloads: Sequence[Mapping[str, Any]],
    catalog: Sequence[Mapping[str, Any]],
    selections: Sequence[Mapping[str, Any]],
    records_by_dataset: Mapping[str, Sequence[str]],
    lookup: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    plan_by_id = {str(plan["plan_id"]): plan for plan in catalog}
    adaptive = _strategy_map(selections, "existing_adaptive")
    full = _strategy_map(selections, "full_multi_decision_batch")
    output = []
    for workload in workloads:
        workload_id = str(workload["id"])
        adaptive_plan = plan_by_id[str(adaptive[workload_id]["plan_id"])]
        full_plan = plan_by_id[str(full[workload_id]["plan_id"])]
        record_ids = list(records_by_dataset[workload["dataset"]])
        adaptive_result = replay_plan(adaptive_plan, record_ids, lookup)
        full_result = replay_plan(full_plan, record_ids, lookup)
        adaptive_by_record = {row["record_id"]: row for row in adaptive_result["per_record"]}
        full_by_record = {row["record_id"]: row for row in full_result["per_record"]}
        single_operators = {
            group["predicate_ids"][0]: group["operator_id"] for group in adaptive_plan["groups"]
        }
        batch_group = full_plan["groups"][0]
        for record_id in record_ids:
            reference = bool(adaptive_by_record[record_id]["passed"])
            candidate = bool(full_by_record[record_id]["passed"])
            if reference == candidate:
                continue
            batch = lookup[(record_id, batch_group["operator_id"])]
            predicate_details = []
            for predicate in workload["predicate_ids"]:
                separate = lookup[(record_id, single_operators[predicate])]
                separate_probability = float(separate["predicate_probabilities"][predicate])
                batch_probability = float(batch["predicate_probabilities"][predicate])
                predicate_details.append(
                    {
                        "predicate_id": predicate,
                        "separate_probability": separate_probability,
                        "batch_probability": batch_probability,
                        "separate_decision": separate_probability >= 0.5,
                        "batch_decision": batch_probability >= 0.5,
                        "threshold": 0.5,
                    }
                )
            output.append(
                {
                    "dataset": workload["dataset"],
                    "workload_id": workload_id,
                    "record_id": record_id,
                    "full_batch_plan_id": full_plan["plan_id"],
                    "reference_query_decision": reference,
                    "batch_query_decision": candidate,
                    "error_kind": "false_negative" if reference else "false_positive",
                    "predicate_details": predicate_details,
                    "reference_is_human_ground_truth": False,
                }
            )
    return output


def _ex_ante_estimator(
    schedule: Mapping[str, Any],
    catalog: Sequence[Mapping[str, Any]],
    stage_d_replays: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    selection = load_selection_evidence()
    selection_ids = _records_by_dataset(
        selection["normalized"], list(selection["config"]["datasets"])
    )
    persisted = {
        row["plan_id"]: row for row in _read_csv(SELECTION_RESULTS / "cost_estimation_errors.csv")
    }
    plan_by_id = {str(plan["plan_id"]): plan for plan in catalog}
    rows = []
    heldout_rows = {
        row["plan_id"]: row for row in _read_csv(STAGE_D_RESULTS / "cost_estimator_errors.csv")
    }
    for plan_id in schedule["selected_plan_ids"]:
        plan = plan_by_id[str(plan_id)]
        ids = selection_ids[str(plan["dataset"])]
        recomputed_selection = predict_plan(plan, ids, selection["lookup"])["predicted_cost_usd"]
        persisted_selection = float(persisted[str(plan_id)]["predicted_cost_usd"])
        predicted = persisted_selection * (400 / len(ids))
        observed = float(stage_d_replays[str(plan_id)]["cost_usd"])
        retrospective = float(heldout_rows[str(plan_id)]["predicted_cost_usd"])
        rows.append(
            {
                "workload_id": plan["workload_id"],
                "plan_id": plan_id,
                "plan_kind": _plan_kind(plan),
                "selection_records": len(ids),
                "heldout_records": 400,
                "persisted_selection_prediction_usd": persisted_selection,
                "independently_recomputed_selection_prediction_usd": recomputed_selection,
                "selection_prediction_reproduced": math.isclose(
                    persisted_selection, recomputed_selection, abs_tol=1e-15
                ),
                "ex_ante_linear_projection_usd": predicted,
                "observed_stage_d_cost_usd": observed,
                "signed_error_usd": predicted - observed,
                "absolute_error_usd": abs(predicted - observed),
                "absolute_percentage_error": abs(predicted - observed) / observed,
                "historical_stage_d_reported_prediction_usd": retrospective,
                "historical_prediction_equals_observed": math.isclose(
                    retrospective, observed, abs_tol=1e-15
                ),
                "provenance": (
                    "pre-Stage-D persisted selection prediction, linearly projected from 100 "
                    "to 400 records using the frozen estimator's n-linear form"
                ),
                "originally_persisted_as_stage_d_forecast": False,
                "uses_stage_d_inputs": False,
            }
        )
    errors = [float(row["signed_error_usd"]) for row in rows]
    observed = [float(row["observed_stage_d_cost_usd"]) for row in rows]
    full_rows = [row for row in rows if row["plan_kind"] == "full_batch"]
    by_workload: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_workload[str(row["workload_id"])].append(row)
    concordances = []
    harmful = correct = 0
    ranking_rows = []
    for workload_id, candidates in sorted(by_workload.items()):
        predicted = {
            str(row["plan_id"]): float(row["ex_ante_linear_projection_usd"]) for row in candidates
        }
        actual = {
            str(row["plan_id"]): float(row["observed_stage_d_cost_usd"]) for row in candidates
        }
        concordance = _ranking_concordance(predicted, actual)
        predicted_best = min(predicted, key=lambda key: (predicted[key], key))
        actual_best = min(actual, key=lambda key: (actual[key], key))
        is_correct = predicted_best == actual_best
        correct += int(is_correct)
        harmful += int(not is_correct)
        concordances.append(concordance)
        ranking_rows.append(
            {
                "workload_id": workload_id,
                "candidate_count": len(candidates),
                "pairwise_ranking_concordance": concordance,
                "predicted_best_plan_id": predicted_best,
                "observed_best_plan_id": actual_best,
                "correct_cheapest_selection": is_correct,
                "cost_only_regret_usd": actual[predicted_best] - actual[actual_best],
            }
        )
    summary = {
        "schema_version": "m5.1-ex-ante-estimator-validation-v1",
        "status": "RECONSTRUCTED_EX_ANTE_PROJECTION_NOT_ORIGINALLY_PERSISTED_AS_STAGE_D_FORECAST",
        "all_selection_predictions_reproduced": all(
            row["selection_prediction_reproduced"] for row in rows
        ),
        "all_nine_historical_full_batch_predictions_equal_observed": all(
            row["historical_prediction_equals_observed"] for row in full_rows
        ),
        "all_plan_metrics": _metrics(errors, observed),
        "full_batch_metrics": _metrics(
            [float(row["signed_error_usd"]) for row in full_rows],
            [float(row["observed_stage_d_cost_usd"]) for row in full_rows],
        ),
        "mean_pairwise_ranking_concordance": statistics.mean(concordances),
        "correct_cheapest_selections": correct,
        "harmful_selections": harmful,
        "per_workload_ranking": ranking_rows,
        "rows": rows,
        "limitations": [
            "The 100-to-400 scaling was reconstructed after Stage D, although every input value was frozen before Stage D.",
            "The frozen estimator uses unconditional reach products and selection-sample mean group costs.",
            "This evaluates only the 29-plan measured Stage D union, not all theoretical plans.",
        ],
    }
    provenance = {
        "schema_version": "m5.1-estimator-prediction-provenance-v1",
        "historical_stage_d_artifact": (STAGE_D_RESULTS / "cost_estimator_errors.csv").as_posix(),
        "historical_mae_usd": _json(STAGE_D_RESULTS / "cost_estimator_summary.json")["mae_usd"],
        "historical_mae_classification": "held-out-informed retrospective diagnostic",
        "cause": (
            "Stage D analysis calls predict_plan(plan, heldout_record_ids, heldout_lookup). "
            "predict_plan derives mean group cost and unconditional selectivity from that same "
            "lookup. For one-group full batches, n times the held-out mean equals the observed total."
        ),
        "valid_out_of_sample_claim": False,
        "required_conclusion": (
            "The reported approximately $0.000022485 MAE is not valid out-of-sample accuracy; "
            "it is a retrospective held-out-informed model-fit diagnostic."
        ),
        "prediction_categories": {
            "historical_stage_d_predicted_cost_usd": "held-out-informed post-hoc reconstruction",
            "historical_observed_cost_usd": "direct observed held-out cost",
            "persisted_selection_prediction_usd": "frozen pre-Stage-D selection-statistics prediction",
            "ex_ante_linear_projection_usd": (
                "post-hoc reconstruction using only frozen pre-Stage-D inputs; not originally "
                "persisted as a Stage D forecast"
            ),
        },
        "full_batch_identity_proof": "predicted = N * mean(held-out request costs) = sum(held-out request costs)",
        "source_locations": [
            "src/semanticflow/optimizer/m51_stage_d_analysis.py:374",
            "src/semanticflow/optimizer/m51_selection_analysis.py:314",
        ],
        "corrected_artifact": "ex_ante_estimator_validation.json",
    }
    return provenance, summary


def _workload_economics(
    workloads: Sequence[Mapping[str, Any]],
    catalog: Sequence[Mapping[str, Any]],
    selections: Sequence[Mapping[str, Any]],
    records_by_dataset: Mapping[str, Sequence[str]],
    lookup: Mapping[tuple[str, str], Mapping[str, Any]],
    quality_rows: Sequence[Mapping[str, Any]],
    bootstrap_iterations: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    plan_by_id = {str(plan["plan_id"]): plan for plan in catalog}
    adaptive = _strategy_map(selections, "existing_adaptive")
    full = _strategy_map(selections, "full_multi_decision_batch")
    selected = _strategy_map(selections, "quality_constrained_hybrid")
    quality_by_id = {str(row["plan_id"]): row for row in quality_rows}
    candidate_by_workload: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for plan in catalog:
        if str(plan["plan_id"]) in quality_by_id:
            candidate_by_workload[str(plan["workload_id"])].append(plan)
    output = []
    profiling = []
    for workload in workloads:
        workload_id = str(workload["id"])
        ids = list(records_by_dataset[workload["dataset"]])
        adaptive_plan = plan_by_id[str(adaptive[workload_id]["plan_id"])]
        full_plan = plan_by_id[str(full[workload_id]["plan_id"])]
        selected_plan = plan_by_id[str(selected[workload_id]["plan_id"])]
        adaptive_result = replay_plan(adaptive_plan, ids, lookup)
        full_result = replay_plan(full_plan, ids, lookup)
        selected_result = replay_plan(selected_plan, ids, lookup)
        candidate_results = {
            str(plan["plan_id"]): replay_plan(plan, ids, lookup)
            for plan in candidate_by_workload[workload_id]
        }
        eligible = {
            plan_id: result
            for plan_id, result in candidate_results.items()
            if bool(quality_by_id[plan_id]["eligible"])
        }
        best_id = min(eligible, key=lambda plan_id: (eligible[plan_id]["cost_usd"], plan_id))
        best = eligible[best_id]
        interval = _paired_bootstrap(
            [float(row["cost_usd"]) for row in adaptive_result["per_record"]],
            [float(row["cost_usd"]) for row in full_result["per_record"]],
            iterations=bootstrap_iterations,
        )
        selected_saving = adaptive_result["cost_usd"] - selected_result["cost_usd"]
        selected_quality = quality_by_id[str(selected_plan["plan_id"])]
        full_quality = quality_by_id[str(full_plan["plan_id"])]
        frozen_fallback = bool(selected[workload_id]["used_existing_adaptive_fallback"])
        if (
            frozen_fallback
            and full_result["cost_usd"] < adaptive_result["cost_usd"]
            and full_quality["eligible"]
        ):
            classification = "excessively_conservative_fallback"
        elif frozen_fallback:
            classification = "correct_full_batch_fallback"
        elif full_result["cost_usd"] < adaptive_result["cost_usd"] and full_quality["eligible"]:
            classification = "selected_batch_confirmed"
        else:
            classification = "selected_batch_not_confirmed"
        output.append(
            {
                "dataset": workload["dataset"],
                "workload_id": workload_id,
                "predicate_count": len(workload["predicate_ids"]),
                "adaptive_plan_id": adaptive_plan["plan_id"],
                "adaptive_cost_usd": adaptive_result["cost_usd"],
                "adaptive_input_tokens": adaptive_result["input_tokens"],
                "adaptive_output_tokens": adaptive_result["output_tokens"],
                "adaptive_requests": adaptive_result["requests"],
                "full_batch_plan_id": full_plan["plan_id"],
                "full_batch_cost_usd": full_result["cost_usd"],
                "full_batch_input_tokens": full_result["input_tokens"],
                "full_batch_output_tokens": full_result["output_tokens"],
                "full_batch_requests": full_result["requests"],
                "full_batch_saving_usd": adaptive_result["cost_usd"] - full_result["cost_usd"],
                "full_batch_saving_pct": (adaptive_result["cost_usd"] - full_result["cost_usd"])
                / adaptive_result["cost_usd"],
                "full_batch_interval_95": interval,
                "full_batch_quality_status": full_quality["status"],
                "full_batch_quality_eligible": full_quality["eligible"],
                "selected_plan_id": selected_plan["plan_id"],
                "selected_plan_kind": _plan_kind(selected_plan),
                "selected_cost_usd": selected_result["cost_usd"],
                "selected_input_tokens": selected_result["input_tokens"],
                "selected_output_tokens": selected_result["output_tokens"],
                "selected_requests": selected_result["requests"],
                "selected_saving_usd": selected_saving,
                "selected_saving_pct": selected_saving / adaptive_result["cost_usd"],
                "selected_quality_status": selected_quality["status"],
                "selected_quality_eligible": selected_quality["eligible"],
                "best_measured_quality_eligible_plan_id": best_id,
                "best_measured_quality_eligible_plan_kind": quality_by_id[best_id]["plan_kind"],
                "best_measured_quality_eligible_cost_usd": best["cost_usd"],
                "quality_constrained_oracle_regret_usd": selected_result["cost_usd"]
                - best["cost_usd"],
                "classification": classification,
                "oracle_is_post_hoc": True,
                "global_optimality_claim": False,
            }
        )
        if workload_id == PRIMARY_WORKLOAD:
            scenarios = (
                ("standalone_cold_start", 0.007067130),
                ("shared_statistics_cold_start", 0.002967972),
                ("warm_start", 0.0),
            )
        else:
            scenarios = (("fallback_same_as_baseline", 0.0),)
        for scenario, profile_cost in scenarios:
            break_even = math.ceil(profile_cost / selected_saving) if selected_saving > 0 else None
            for horizon in HORIZONS:
                profiling.append(
                    {
                        "workload_id": workload_id,
                        "scenario": scenario,
                        "reuse_horizon": horizon,
                        "baseline_cost_usd": adaptive_result["cost_usd"],
                        "candidate_cost_usd": selected_result["cost_usd"],
                        "execution_saving_per_run_usd": selected_saving,
                        "incremental_profiling_cost_usd": profile_cost,
                        "optimization_cost_usd": None,
                        "net_benefit_excluding_unmetered_optimization_usd": horizon
                        * selected_saving
                        - profile_cost,
                        "break_even_executions_excluding_unmetered_optimization": break_even,
                        "quality_eligible": bool(selected_quality["eligible"]),
                        "economic_claim_status": (
                            "QUALITY_INADMISSIBLE_ON_HELD_OUT"
                            if workload_id == PRIMARY_WORKLOAD
                            else "frozen_fallback_same_as_baseline"
                        ),
                    }
                )
    aggregate = {
        "existing_adaptive_cost_usd": sum(float(row["adaptive_cost_usd"]) for row in output),
        "full_batch_cost_usd": sum(float(row["full_batch_cost_usd"]) for row in output),
        "frozen_selected_execution_cost_usd": sum(
            float(row["selected_cost_usd"]) for row in output
        ),
        "frozen_execution_only_saving_usd": sum(
            float(row["selected_saving_usd"]) for row in output
        ),
        "frozen_selected_policy_quality_admissible": all(
            bool(row["selected_quality_eligible"]) for row in output
        ),
        "prospective_deployment_recommendation": (
            "retain existing_adaptive; the only changed frozen plan failed held-out quality"
        ),
        "matrix_acquisition_cost_usd": 0.161602854,
        "matrix_cost_is_not_policy_execution_cost": True,
    }
    return output, aggregate, profiling


def _dataset_and_batch_rows(
    per_request: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_dataset = []
    for dataset in sorted({str(row["dataset"]) for row in per_request}):
        rows = [row for row in per_request if row["dataset"] == dataset]
        per_dataset.append(
            {
                "dataset": dataset,
                "requests": len(rows),
                "named_question_evaluations": sum(
                    int(row["named_question_evaluations"]) for row in rows
                ),
                "input_tokens": sum(int(row["input_tokens"]) for row in rows),
                "output_tokens": sum(int(row["output_tokens"]) for row in rows),
                "cost_usd": sum(float(row["cost_usd"]) for row in rows),
            }
        )
    batch_rows = []
    for size in (1, 2, 3, 4):
        rows = [row for row in per_request if int(row["named_question_evaluations"]) == size]
        batch_rows.append(
            {
                "named_questions_per_request": size,
                "requests": len(rows),
                "input_tokens": sum(int(row["input_tokens"]) for row in rows),
                "output_tokens": sum(int(row["output_tokens"]) for row in rows),
                "mean_input_tokens_per_request": statistics.mean(
                    int(row["input_tokens"]) for row in rows
                ),
                "cost_usd": sum(float(row["cost_usd"]) for row in rows),
            }
        )
    return per_dataset, batch_rows


def _statistical_audit(
    workload_rows: Sequence[Mapping[str, Any]],
    catalog: Sequence[Mapping[str, Any]],
    selections: Sequence[Mapping[str, Any]],
    records_by_dataset: Mapping[str, Sequence[str]],
    lookup: Mapping[tuple[str, str], Mapping[str, Any]],
    bootstrap_iterations: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan_by_id = {str(plan["plan_id"]): plan for plan in catalog}
    adaptive = _strategy_map(selections, "existing_adaptive")
    full = _strategy_map(selections, "full_multi_decision_batch")
    historical_intervals = {
        row["workload_id"]: row["full_batch_saving_interval"]
        for row in _json(STAGE_D_RESULTS / "statistical_uncertainty.json")["workloads"]
    }
    intervals = []
    degenerate = None
    for row in workload_rows:
        workload_id = str(row["workload_id"])
        ids = list(records_by_dataset[str(row["dataset"])])
        adaptive_result = replay_plan(
            plan_by_id[str(adaptive[workload_id]["plan_id"])], ids, lookup
        )
        full_result = replay_plan(plan_by_id[str(full[workload_id]["plan_id"])], ids, lookup)
        baseline = [float(value["cost_usd"]) for value in adaptive_result["per_record"]]
        candidate = [float(value["cost_usd"]) for value in full_result["per_record"]]
        differences = [left - right for left, right in zip(baseline, candidate, strict=True)]
        normalized_differences = {round(value, 15) for value in differences}
        interval = _paired_bootstrap(baseline, candidate, iterations=bootstrap_iterations)
        stored = historical_intervals[workload_id]
        stored_reproduced = bootstrap_iterations == 10000 and all(
            math.isclose(float(interval[key]), float(stored[key]), abs_tol=1e-15)
            for key in ("estimate_usd", "lower_95_usd", "upper_95_usd")
        )
        intervals.append(
            {
                "workload_id": workload_id,
                **interval,
                "unique_per_record_differences": len(normalized_differences),
                "raw_floating_point_unique_differences": len(set(differences)),
                "per_record_difference_min_usd": min(differences),
                "per_record_difference_max_usd": max(differences),
                "historical_interval": stored,
                "historical_interval_reproduced": stored_reproduced,
            }
        )
        if workload_id == "m51_sms_security_prize_urgency_contact":
            effectively_constant = len(normalized_differences) == 1
            degenerate = {
                "workload_id": workload_id,
                "unique_per_record_differences": len(normalized_differences),
                "raw_floating_point_unique_differences": len(set(differences)),
                "floating_point_range_usd": max(differences) - min(differences),
                "constant_per_record_difference_usd": statistics.mean(differences)
                if effectively_constant
                else None,
                "adaptive_stage_entries": adaptive_result["stage_entries"],
                "adaptive_stage_passes": adaptive_result["stage_passes"],
                "diagnosis": (
                    "genuinely degenerate after removing sub-token floating-point noise: every "
                    "record has the same integer-token cost difference and adaptive execution "
                    "never enters a later stage"
                    if effectively_constant
                    else "not degenerate; investigate analyzer"
                ),
                "bootstrap_bug_detected": not effectively_constant,
            }
    methodology = {
        "schema_version": "m5.1-bootstrap-methodology-audit-v1",
        "paired_alignment": "exact shared record IDs, sorted within each dataset",
        "sampling_unit": "record cluster",
        "draws": bootstrap_iterations,
        "seed": 51103,
        "statistic": "sum over resampled per-record baseline-minus-candidate cost differences",
        "cost_conversion": "$0.042 per million input tokens; output price zero",
        "interval_indexing": "sorted draws at integer 2.5% and 97.5% indices",
        "all_historical_intervals_reproduced": all(
            row["historical_interval_reproduced"] for row in intervals
        ),
        "dependence_warning": (
            "workloads and plans share records, predicates, and provider decisions; intervals are "
            "per-comparison nominal intervals and are not independent replications"
        ),
        "multiple_comparison_adjustment": "none; primary Banking77 comparison preregistered, others secondary",
        "sms_four_predicate_degenerate_interval": degenerate,
    }
    return {
        "schema_version": "m5.1-statistical-uncertainty-verified-v1",
        "workloads": intervals,
        "sms_four_predicate_degenerate_interval_diagnosis": degenerate,
    }, methodology


def _historical_expenditure(per_request: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def raw_usage(root: Path) -> tuple[int, int, int]:
        rows = []
        for path in root.rglob("decisions.jsonl"):
            rows.extend(json.loads(line) for line in path.read_text("utf-8").splitlines() if line)
        return (
            len(rows),
            sum(int(row["input_tokens"]) for row in rows),
            sum(int(row["output_tokens"]) for row in rows),
        )

    m5_requests, m5_input, m5_output = raw_usage(M5_SESSION)
    b_requests, b_input, b_output = raw_usage(STAGE_B_SESSION)
    selection = _json(SELECTION_RESULTS / "token_cost_reconciliation.json")
    stages = [
        {
            "stage": "M5 conjunctive-fusion live pilot",
            "requests": m5_requests,
            "input_tokens": m5_input,
            "output_tokens": m5_output,
            "cost_usd": m5_input * INPUT_PRICE / 1_000_000,
            "failed_pre_response_attempts": 0,
        },
        {
            "stage": "M5.1 Stage B named-question batching pilot",
            "requests": b_requests,
            "input_tokens": b_input,
            "output_tokens": b_output,
            "cost_usd": b_input * INPUT_PRICE / 1_000_000,
            "failed_pre_response_attempts": 0,
        },
        {
            "stage": "M5.1 selection stage",
            "requests": selection["successful_physical_requests"],
            "input_tokens": selection["input_tokens"],
            "output_tokens": selection["output_tokens"],
            "cost_usd": selection["token_priced_cost_usd"],
            "failed_pre_response_attempts": selection["failed_pre_response_attempts"],
        },
        {
            "stage": "M5.1 Stage D held-out matrix",
            "requests": len(per_request),
            "input_tokens": sum(int(row["input_tokens"]) for row in per_request),
            "output_tokens": sum(int(row["output_tokens"]) for row in per_request),
            "cost_usd": sum(float(row["cost_usd"]) for row in per_request),
            "failed_pre_response_attempts": 3,
        },
    ]
    return {
        "schema_version": "m5.1-experimental-expenditure-v1",
        "pricing": {"input_per_million_usd": INPUT_PRICE, "output_per_million_usd": 0.0},
        "completed_live_experiments": stages,
        "four_experiment_token_priced_total_usd": sum(row["cost_usd"] for row in stages),
        "m5_1_completed_live_subtotal_usd": sum(row["cost_usd"] for row in stages[1:]),
        "failed_pre_response_attempts_total": sum(
            row["failed_pre_response_attempts"] for row in stages
        ),
        "failed_attempt_provider_billing_status": "unknown",
        "provider_invoice_established": False,
        "complete_project_spend_claim": False,
        "exclusions": [
            "other milestones and provider work not explicitly listed",
            "unknown charges for failed pre-response attempts",
            "local CPU and engineering time",
        ],
    }


def _svg_bar_chart(
    path: Path,
    title: str,
    labels: Sequence[str],
    series: Sequence[tuple[str, Sequence[float], str]],
    *,
    y_label: str,
    width: int = 1100,
    height: int = 620,
) -> None:
    margin_left, margin_right, margin_top, margin_bottom = 100, 40, 70, 170
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    values = [value for _, items, _ in series for value in items]
    low = min(0.0, min(values))
    high = max(0.0, max(values))
    span = high - low or 1.0
    low -= span * 0.08
    high += span * 0.08
    span = high - low

    def y(value: float) -> float:
        return margin_top + (high - value) / span * plot_h

    group_w = plot_w / len(labels)
    bar_w = group_w * 0.72 / len(series)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="bold">{html.escape(title)}</text>',
        f'<text x="22" y="{margin_top + plot_h / 2}" transform="rotate(-90 22 {margin_top + plot_h / 2})" text-anchor="middle" font-family="sans-serif" font-size="15">{html.escape(y_label)}</text>',
    ]
    for tick in range(6):
        value = low + span * tick / 5
        yy = y(value)
        parts.append(
            f'<line x1="{margin_left}" y1="{yy}" x2="{width - margin_right}" y2="{yy}" stroke="#ddd"/>'
        )
        parts.append(
            f'<text x="{margin_left - 8}" y="{yy + 5}" text-anchor="end" font-family="monospace" font-size="12">{value:.4g}</text>'
        )
    zero_y = y(0)
    parts.append(
        f'<line x1="{margin_left}" y1="{zero_y}" x2="{width - margin_right}" y2="{zero_y}" stroke="#333" stroke-width="1.5"/>'
    )
    for index, label in enumerate(labels):
        center = margin_left + group_w * (index + 0.5)
        for series_index, (_, items, color) in enumerate(series):
            value = float(items[index])
            x = center - group_w * 0.36 + series_index * bar_w
            yy = min(y(value), zero_y)
            hh = abs(zero_y - y(value))
            parts.append(
                f'<rect x="{x}" y="{yy}" width="{bar_w * 0.9}" height="{max(hh, 0.8)}" fill="{color}"/>'
            )
        parts.append(
            f'<text x="{center}" y="{margin_top + plot_h + 14}" transform="rotate(42 {center} {margin_top + plot_h + 14})" text-anchor="start" font-family="sans-serif" font-size="12">{html.escape(label)}</text>'
        )
    legend_x = margin_left
    for name, _, color in series:
        parts.append(
            f'<rect x="{legend_x}" y="{height - 28}" width="14" height="14" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{legend_x + 20}" y="{height - 16}" font-family="sans-serif" font-size="13">{html.escape(name)}</text>'
        )
        legend_x += 185
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _svg_line_chart(
    path: Path,
    title: str,
    x_values: Sequence[float],
    series: Sequence[tuple[str, Sequence[float], str]],
    *,
    x_label: str,
    y_label: str,
    width: int = 1000,
    height: int = 560,
) -> None:
    left, right, top, bottom = 95, 40, 65, 85
    plot_w, plot_h = width - left - right, height - top - bottom
    all_y = [value for _, values, _ in series for value in values]
    y_low, y_high = min(0.0, min(all_y)), max(0.0, max(all_y))
    y_span = y_high - y_low or 1.0
    y_low -= y_span * 0.08
    y_high += y_span * 0.08
    y_span = y_high - y_low
    x_logs = [math.log10(value) for value in x_values]
    x_low, x_high = min(x_logs), max(x_logs)

    def x(value: float) -> float:
        return left + (math.log10(value) - x_low) / (x_high - x_low) * plot_w

    def y(value: float) -> float:
        return top + (y_high - value) / y_span * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="bold">{html.escape(title)}</text>',
        f'<text x="{width / 2}" y="{height - 16}" text-anchor="middle" font-family="sans-serif" font-size="15">{html.escape(x_label)}</text>',
        f'<text x="22" y="{top + plot_h / 2}" transform="rotate(-90 22 {top + plot_h / 2})" text-anchor="middle" font-family="sans-serif" font-size="15">{html.escape(y_label)}</text>',
        f'<line x1="{left}" y1="{y(0)}" x2="{width - right}" y2="{y(0)}" stroke="#333"/>',
    ]
    for value in x_values:
        xx = x(value)
        parts.append(f'<line x1="{xx}" y1="{top}" x2="{xx}" y2="{top + plot_h}" stroke="#eee"/>')
        parts.append(
            f'<text x="{xx}" y="{top + plot_h + 22}" text-anchor="middle" font-family="monospace" font-size="12">{int(value)}</text>'
        )
    for tick in range(6):
        value = y_low + y_span * tick / 5
        yy = y(value)
        parts.append(f'<line x1="{left}" y1="{yy}" x2="{width - right}" y2="{yy}" stroke="#eee"/>')
        parts.append(
            f'<text x="{left - 8}" y="{yy + 5}" text-anchor="end" font-family="monospace" font-size="12">{value:.4g}</text>'
        )
    legend_x = left
    for name, values, color in series:
        points = " ".join(f"{x(xv)},{y(yv)}" for xv, yv in zip(x_values, values, strict=True))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for xv, yv in zip(x_values, values, strict=True):
            parts.append(f'<circle cx="{x(xv)}" cy="{y(yv)}" r="4" fill="{color}"/>')
        parts.append(
            f'<line x1="{legend_x}" y1="{height - 48}" x2="{legend_x + 24}" y2="{height - 48}" stroke="{color}" stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{legend_x + 30}" y="{height - 43}" font-family="sans-serif" font-size="13">{html.escape(name)}</text>'
        )
        legend_x += 250
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _short_workload(workload_id: str) -> str:
    return workload_id.removeprefix("m51_").replace("_", " ")


def _figures(
    root: Path,
    workload_rows: Sequence[Mapping[str, Any]],
    batch_rows: Sequence[Mapping[str, Any]],
    quality_rows: Sequence[Mapping[str, Any]],
    profiling_rows: Sequence[Mapping[str, Any]],
    expenditure: Mapping[str, Any],
) -> list[str]:
    root.mkdir(parents=True, exist_ok=True)
    labels = [_short_workload(str(row["workload_id"])) for row in workload_rows]
    paths = []

    def bar(
        name: str, title: str, series: Sequence[tuple[str, Sequence[float], str]], y_label: str
    ) -> None:
        path = root / name
        _svg_bar_chart(path, title, labels, series, y_label=y_label)
        paths.append(path.as_posix())

    bar(
        "per_workload_batching_savings.svg",
        "Full-batch savings versus existing adaptive (secondary except Banking77 primary)",
        [
            (
                "Savings",
                [100 * float(row["full_batch_saving_pct"]) for row in workload_rows],
                "#2563eb",
            )
        ],
        "Savings (%)",
    )
    bar(
        "adaptive_vs_full_batch.svg",
        "Modeled 400-record execution cost by workload",
        [
            (
                "Existing adaptive",
                [float(row["adaptive_cost_usd"]) for row in workload_rows],
                "#475569",
            ),
            ("Full batch", [float(row["full_batch_cost_usd"]) for row in workload_rows], "#f97316"),
        ],
        "Token-priced cost (USD)",
    )
    bar(
        "selected_policy_vs_baseline.svg",
        "Frozen selected-policy execution cost versus baseline",
        [
            (
                "Existing adaptive",
                [float(row["adaptive_cost_usd"]) for row in workload_rows],
                "#475569",
            ),
            (
                "Frozen selected",
                [float(row["selected_cost_usd"]) for row in workload_rows],
                "#16a34a",
            ),
        ],
        "Token-priced cost (USD)",
    )
    bar(
        "early_rejection_request_economics.svg",
        "Early rejection changes request counts",
        [
            (
                "Adaptive requests",
                [float(row["adaptive_requests"]) for row in workload_rows],
                "#475569",
            ),
            (
                "Full-batch requests",
                [float(row["full_batch_requests"]) for row in workload_rows],
                "#f97316",
            ),
        ],
        "Physical requests per 400 records",
    )
    batch_labels = [str(row["named_questions_per_request"]) for row in batch_rows]
    path = root / "batch_size_token_cost.svg"
    _svg_bar_chart(
        path,
        "Measured request size and token cost",
        batch_labels,
        [
            (
                "Mean input tokens/request",
                [float(row["mean_input_tokens_per_request"]) for row in batch_rows],
                "#7c3aed",
            )
        ],
        y_label="Mean input tokens per physical request",
        width=820,
    )
    paths.append(path.as_posix())
    full_quality = [row for row in quality_rows if row["plan_kind"] == "full_batch"]
    quality_by_workload = {str(row["workload_id"]): row for row in full_quality}
    bar(
        "quality_agreement_support.svg",
        "Full-batch query agreement and positive support",
        [
            (
                "Agreement (%)",
                [
                    100 * float(quality_by_workload[str(row["workload_id"])]["agreement"])
                    for row in workload_rows
                ],
                "#0f766e",
            ),
            (
                "Reference positives",
                [
                    float(quality_by_workload[str(row["workload_id"])]["reference_positive_count"])
                    for row in workload_rows
                ],
                "#dc2626",
            ),
        ],
        "Percent / count (shared display scale)",
    )
    primary = next(row for row in workload_rows if row["workload_id"] == PRIMARY_WORKLOAD)
    beauty = next(row for row in workload_rows if row["workload_id"] == BEAUTY_WORKLOAD)
    for filename, title, row in (
        ("banking77_primary_case.svg", "Banking77 primary: economic win, quality failure", primary),
        (
            "beauty_missed_opportunity.svg",
            "Beauty counterexample: held-out admissible batching",
            beauty,
        ),
    ):
        path = root / filename
        _svg_bar_chart(
            path,
            title,
            ["existing adaptive", "full batch"],
            [
                (
                    "Cost",
                    [float(row["adaptive_cost_usd"]), float(row["full_batch_cost_usd"])],
                    "#2563eb",
                )
            ],
            y_label="Token-priced cost (USD)",
            width=760,
            height=520,
        )
        paths.append(path.as_posix())
    bar(
        "quality_constrained_oracle_regret.svg",
        "Post-hoc measured-set quality-constrained oracle regret",
        [
            (
                "Regret",
                [float(row["quality_constrained_oracle_regret_usd"]) for row in workload_rows],
                "#b91c1c",
            )
        ],
        "Cost difference (USD)",
    )
    primary_profiles = [row for row in profiling_rows if row["workload_id"] == PRIMARY_WORKLOAD]
    path = root / "profiling_break_even.svg"
    profile_series = []
    for scenario, color in (
        ("standalone_cold_start", "#dc2626"),
        ("shared_statistics_cold_start", "#2563eb"),
        ("warm_start", "#16a34a"),
    ):
        rows = sorted(
            (row for row in primary_profiles if row["scenario"] == scenario),
            key=lambda row: int(row["reuse_horizon"]),
        )
        profile_series.append(
            (
                scenario.replace("_", " "),
                [float(row["net_benefit_excluding_unmetered_optimization_usd"]) for row in rows],
                color,
            )
        )
    _svg_line_chart(
        path,
        "Banking77 profiling economics (quality-inadmissible on held-out)",
        list(HORIZONS),
        profile_series,
        x_label="Reuse executions (log scale)",
        y_label="Net benefit excluding unmetered optimization (USD)",
    )
    paths.append(path.as_posix())
    exp_labels = [str(row["stage"]) for row in expenditure["completed_live_experiments"]]
    path = root / "experimental_expenditure_by_stage.svg"
    _svg_bar_chart(
        path,
        "Completed live-experiment token-priced expenditure",
        exp_labels,
        [
            (
                "Cost",
                [float(row["cost_usd"]) for row in expenditure["completed_live_experiments"]],
                "#7c3aed",
            )
        ],
        y_label="Token-priced cost (USD)",
        width=980,
        height=600,
    )
    paths.append(path.as_posix())
    return paths


def _fmt_money(value: float) -> str:
    return f"${value:.9f}"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _report(
    integrity: Mapping[str, Any],
    reconciliation: Mapping[str, Any],
    workload_rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    canonical_quality: Sequence[Mapping[str, Any]],
    disagreements: Sequence[Mapping[str, Any]],
    estimator: Mapping[str, Any],
    provenance: Mapping[str, Any],
    dedup: Mapping[str, Any],
    statistics_audit: Mapping[str, Any],
    expenditure: Mapping[str, Any],
    profiling: Sequence[Mapping[str, Any]],
    figure_root: Path,
) -> str:
    primary = next(row for row in workload_rows if row["workload_id"] == PRIMARY_WORKLOAD)
    beauty = next(row for row in workload_rows if row["workload_id"] == BEAUTY_WORKLOAD)
    quality_by_id = {str(row["plan_id"]): row for row in canonical_quality}
    beauty_quality = quality_by_id[BEAUTY_BATCH]
    primary_disagreement = next(
        row for row in disagreements if row["workload_id"] == PRIMARY_WORKLOAD
    )
    workload_table = _markdown_table(
        [
            "Workload",
            "Adaptive",
            "Full batch",
            "Batch saving",
            "Saving %",
            "Batch quality",
            "Frozen selected",
        ],
        [
            (
                _short_workload(str(row["workload_id"])),
                _fmt_money(float(row["adaptive_cost_usd"])),
                _fmt_money(float(row["full_batch_cost_usd"])),
                _fmt_money(float(row["full_batch_saving_usd"])),
                f"{100 * float(row['full_batch_saving_pct']):.2f}%",
                str(row["full_batch_quality_status"]),
                str(row["selected_plan_kind"]),
            )
            for row in workload_rows
        ],
    )
    quality_table = _markdown_table(
        ["Workload", "Positives", "TP", "FP", "FN", "Agreement", "Recall", "FNR", "Status"],
        [
            (
                _short_workload(str(row["workload_id"])),
                row["reference_positive_count"],
                row["true_positive"],
                row["false_positive"],
                row["false_negative"],
                f"{100 * float(row['agreement']):.2f}%",
                "undefined" if row["recall"] is None else f"{100 * float(row['recall']):.2f}%",
                "undefined"
                if row["false_negative_rate"] is None
                else f"{100 * float(row['false_negative_rate']):.2f}%",
                row["status"],
            )
            for row in canonical_quality
            if row["plan_kind"] == "full_batch"
        ],
    )
    spending_table = _markdown_table(
        ["Experiment", "Requests", "Input tokens", "Output tokens", "Cost proxy"],
        [
            (
                row["stage"],
                row["requests"],
                row["input_tokens"],
                row["output_tokens"],
                _fmt_money(float(row["cost_usd"])),
            )
            for row in expenditure["completed_live_experiments"]
        ],
    )
    adaptive_cost_by_workload = {
        str(row["workload_id"]): float(row["adaptive_cost_usd"]) for row in workload_rows
    }
    hybrid_table = _markdown_table(
        ["Workload", "Hybrid plan", "Observed cost", "vs adaptive", "Quality"],
        [
            (
                _short_workload(str(row["workload_id"])),
                str(row["plan_id"]),
                _fmt_money(float(row["observed_stage_d_cost_usd"])),
                _fmt_money(
                    adaptive_cost_by_workload[str(row["workload_id"])]
                    - float(row["observed_stage_d_cost_usd"])
                ),
                str(quality_by_id[str(row["plan_id"])]["status"]),
            )
            for row in estimator["rows"]
            if row["plan_kind"] == "selective_first_hybrid"
        ],
    )
    directions = _markdown_table(
        [
            "Rank",
            "Direction",
            "Failure connection",
            "Novelty potential",
            "Experimental value",
            "Implementation",
            "Data need",
            "Provider cost",
            "Repeat-work risk",
        ],
        [
            (
                1,
                "Uncertainty-aware quality qualification",
                "Point recall was unstable under rare positives",
                "High if tied to plan admission rather than generic interval reporting",
                "Highest",
                "Medium",
                "Positive-enriched selection/evaluation partitions",
                "Medium",
                "Medium",
            ),
            (
                2,
                "Active semantic profiling",
                "Uniform samples spent heavily while finding few complete-query positives",
                "High for plan-specific information acquisition",
                "Highest",
                "Medium-high",
                "Unlabeled pool plus safeguarded evaluation split",
                "Medium; targeted",
                "Medium",
            ),
            (
                3,
                "Human-ground-truth validation",
                "Current reference is Jev consistency, not truth",
                "Moderate; established method, new role in this system",
                "High",
                "High",
                "Independently annotated stratified sample",
                "Low provider / high annotation",
                "Low",
            ),
            (
                4,
                "Quality-aware plan selection",
                "Cost-only gains can be semantically inadmissible",
                "Medium-high when uncertainty enters the physical objective",
                "High",
                "Medium-high",
                "Calibrated disagreement-risk estimates",
                "Medium",
                "Medium",
            ),
            (
                5,
                "Distribution-shift robustness",
                "Selection-to-held-out quality did not generalize",
                "High in combination with semantic-plan qualification",
                "High",
                "High",
                "Multiple temporal/domain partitions",
                "High",
                "Medium",
            ),
            (
                6,
                "More efficient amortized profiling",
                "Break-even is 6 or 13 runs before quality",
                "Medium; overlaps active sampling",
                "Medium",
                "Medium",
                "Reusable operator statistics",
                "Low-medium",
                "High unless distinguished from active profiling",
            ),
            (
                7,
                "Correlation-aware hybrid cost estimation",
                "Unconditional reach products limit hybrid forecasts",
                "Medium",
                "Medium",
                "Medium",
                "Conditional reach labels",
                "Medium",
                "High because correlation-aware estimation is already partly studied",
            ),
        ],
    )
    primary_detail = json.dumps(primary_disagreement["predicate_details"], sort_keys=True)
    standalone = next(
        row
        for row in profiling
        if row["workload_id"] == PRIMARY_WORKLOAD
        and row["scenario"] == "standalone_cold_start"
        and row["reuse_horizon"] == 1
    )
    shared = next(
        row
        for row in profiling
        if row["workload_id"] == PRIMARY_WORKLOAD
        and row["scenario"] == "shared_statistics_cold_start"
        and row["reuse_horizon"] == 1
    )
    sections = rf"""# SemanticFlow Milestone 5.1 Final Research Audit and Consolidated Evaluation Report

## 1. Abstract

Milestone 5.1 evaluated named-question multi-decision batching as a physical operator for LLM-powered semantic AND queries. The full lifecycle comprised an 810-request pilot, a 3,000-request selection experiment, and a preregistered 10,800-request held-out evaluation over three datasets and nine workloads. The held-out matrix reconciles exactly to {reconciliation["input_tokens"]:,} input tokens, {reconciliation["output_tokens"]:,} output tokens, {reconciliation["named_question_evaluations"]:,} named decisions, and {_fmt_money(float(reconciliation["token_priced_cost_usd"]))} under the frozen experimental price. Full batching was cheaper than `existing_adaptive` in two workloads and costlier in seven. The selected Banking77 batch saved {100 * float(primary["full_batch_saving_pct"]):.2f}% with a positive paired interval, but failed the unchanged quality gate after one false negative among only three separate-reference positives. Consequently the frozen policy did not establish a quality-admissible improvement. A Beauty workload exposed the converse error: batching was cheaper and passed the point-estimate gate on held-out data, but selection evidence had correctly recorded a quality failure. The audit also identifies a material analysis defect: the historical Stage D cost “predictions” were computed from held-out costs and selectivities, so the reported MAE is retrospective, not out-of-sample. A pre-Stage-D estimator projection is reconstructed separately and labeled as such. One duplicated quality row is a reporting-only defect and is corrected in versioned artifacts.

## 2. Research problem and motivation

SemanticFlow seeks to reduce the monetary and request cost of semantic queries without silently changing their results. Separate early-rejecting execution saves work when early predicates are selective; batching shares record and protocol tokens but evaluates every named question. The research problem is therefore conditional: batching can win only when shared-token savings exceed the loss of early rejection, and any economic gain must survive a semantic-consistency constraint.

## 3. Related system ideas and research positioning

The implementation combines database-style predicate ordering, selectivity estimation, physical-plan enumeration, adaptive profiling, confidence/risk controls, and LLM request batching. This report makes no novelty claim beyond the measured SemanticFlow system. Separate Jev calls are treated as a consistency reference, not human truth. Existing project literature notes are retained where documented; no unverified external citation is introduced here.

## 4. SemanticFlow optimizer architecture

The system separates logical predicates from physical groups. A `FusedOperator` binds predicate IDs, question names, prompt hash, response schema, model, threshold, and execution parameters. A physical plan is an ordered list of groups. The executor advances a record only when every named decision in the current group passes. The selection layer compares fixed separate, adaptive separate, full batch, and selective-first hybrid candidates under cost and quality rules.

## 5. Physical execution strategies

- **Existing adaptive:** frozen separate-predicate order with early rejection.
- **Full multi-decision batch:** one request per record containing every named predicate question.
- **Selective-first hybrid:** one selective group first, followed by a remainder batch only for surviving records.
- **Conjunctive binary (Milestone 5):** one binary conjunction question; retained as a historical negative control.

## 6. Cost model and early-rejection equations

For ordered groups $G_1,\ldots,G_k$ over $N$ records, the estimator used

$$\widehat C = N\sum_{{i=1}}^k \left(\prod_{{j<i}}\widehat s_j\right)\bar c_i,$$

where $\widehat s_j$ is an unconditional group pass rate and $\bar c_i$ is mean group request cost. Actual replay instead sums measured per-record costs only along the reached path:

$$C_\text{{replay}}=\sum_r\sum_{{i:\,r\text{{ reaches }}i}} c_{{ri}}.$$

Costs use $0.042 per million input tokens and $0 per output token. They are experimental proxies, not invoices.

## 7. Adaptive profiling and physical-plan selection

The baseline controller used frozen selection-stage measurements and a ten-execution horizon. Selection compared 111 catalog plans but Stage D measured a frozen 29-plan union supporting the declared strategies. The selected quality-constrained policy changed only {PRIMARY_WORKLOAD}; all other workloads retained `existing_adaptive`.

## 8. Quality-gate design

The unchanged gate required at least 40 records, at least three separate-reference positives, agreement at least 0.98, recall at least 0.95, FNR at most 0.05, changed-decision rate at most 0.02, and complete real evidence. It is a point-estimate consistency gate. It does not require the Wilson lower bound for recall to exceed 0.95 and does not establish human-label accuracy.

## 9. Experimental datasets and workloads

Amazon All Beauty, Banking77 queries, and SMS Spam each contributed 400 evaluation records. Nine two-, three-, and four-predicate AND workloads were evaluated. Manifest validation confirmed development, pilot, selection, and evaluation identity sets were disjoint.

## 10. Jev model and execution configuration

All successful Stage D traces resolved to `jev-1.14.0`, threshold 0.5, cache disabled, concurrency four, and no retry. The frozen schedule contained 6,000 single-reference calls, 3,600 full-batch calls, and 1,200 remainder-batch calls. Four smoke requests were schedule members, not additions.

## 11. Experimental timeline

1. Milestone 5 tested conjunctive-binary fusion.
2. M5.1 Stage B tested named-question batching.
3. Stage C analyzed batching and selective-first hybrids offline.
4. The selection stage acquired 3,000 physical measurements and froze nine decisions.
5. Stage D independently acquired 10,800 evaluation-partition measurements.
6. This audit replays only persisted evidence and makes zero provider calls.

## 12. Milestone 5 baseline findings

The conjunctive pilot made 1,230 requests and cost {_fmt_money(float(expenditure["completed_live_experiments"][0]["cost_usd"]))}. Full conjunction fusion was 21.77% more expensive than fixed separate replay; 472/480 decisions agreed with the separate conjunction, but no tested candidate passed the frozen quality gate. This remains a negative result.

## 13. Milestone 5.1 pilot results

Stage B completed 810 requests, 1,380 named evaluations, 299,406 input tokens, 37,690 output tokens, and $0.012575052. Stage C found workload-dependent two-question benefits, approximate three-question aggregate parity, frequent four-question losses, and exploratory hybrid gains, but incomplete coverage and no quality-qualified candidate.

## 14. Selection-stage results

The selection stage reconciles 3,000 successful physical requests, 3,300 logical roles, 300 aliases, 5,300 named evaluations, 1,064,824 input tokens, and $0.044722608. Three pre-response transport attempts remain in the ledger with unknown billing. Banking77 Card/Unauthorized/Refund was selected after a 7.34% measured selection saving and 100/100 query agreement with three positives. Beauty failure/usability was rejected because its selection batch had recall 7/8 (87.5%) and FNR 1/8 (12.5%), failing the unchanged gate.

## 15. Independent held-out results

{workload_table}

![Per-workload batching savings](figures/milestone_5_1_final/per_workload_batching_savings.svg)

The aggregate 400-record-per-workload modeled costs are {_fmt_money(float(aggregate["existing_adaptive_cost_usd"]))} for existing adaptive, {_fmt_money(float(aggregate["full_batch_cost_usd"]))} for full batching, and {_fmt_money(float(aggregate["frozen_selected_execution_cost_usd"]))} for the realized frozen selected strategy. These modeled costs are not the {_fmt_money(float(aggregate["matrix_acquisition_cost_usd"]))} matrix acquisition expense.

![Selected policy versus baseline](figures/milestone_5_1_final/selected_policy_vs_baseline.svg)

## 16. Full batching versus existing adaptive

Full batching was cheaper only for Beauty failure/usability and the Banking77 primary. It was costlier for the other seven workloads. This pattern supports the central mechanism: batching benefits shared token overhead, while early rejection dominates when the first predicate filters enough records or the batch prompt grows substantially.

![Adaptive versus full batch](figures/milestone_5_1_final/adaptive_vs_full_batch.svg)

The physical-request matrix also exposes the batch-size/token relationship. The plot is descriptive across the frozen operator mix; it does not isolate batch size causally because datasets, prompts, and predicates differ.

![Batch size versus token cost](figures/milestone_5_1_final/batch_size_token_cost.svg)

## 17. Hybrid execution performance

Selective-first hybrids were measured where a unique remainder group existed. Some reduced cost relative to a full batch, but the bounded measured union does not establish global plan optimality. A “correct fallback” against full batching can still have post-hoc regret relative to another separate order or hybrid. All such comparisons are secondary and share records and operators.

{hybrid_table}

![Batch size versus early-rejection request economics](figures/milestone_5_1_final/early_rejection_request_economics.svg)

## 18. Banking77 primary case study

The primary batch cost {_fmt_money(float(primary["full_batch_cost_usd"]))} versus {_fmt_money(float(primary["adaptive_cost_usd"]))}, saving {_fmt_money(float(primary["full_batch_saving_usd"]))} ({100 * float(primary["full_batch_saving_pct"]):.2f}%). The paired 10,000-draw record bootstrap interval was [{_fmt_money(float(primary["full_batch_interval_95"]["lower_95_usd"]))}, {_fmt_money(float(primary["full_batch_interval_95"]["upper_95_usd"]))}]. Economically, batching won on the held-out sample.

Quality did not. Query decisions matched 399/400. There were three separate-reference positives, two true positives, one false negative, and no false positives: recall 66.67%, FNR 33.33%, agreement 99.75%. The disagreement record was `{primary_disagreement["record_id"]}`. Its predicate probabilities were `{primary_detail}`. This is a threshold-crossing inconsistency relative to separate Jev inference, not proof of human-label error. The final primary classification is **economic success, quality failure**.

![Banking77 primary case](figures/milestone_5_1_final/banking77_primary_case.svg)

## 19. Beauty missed-opportunity case study

Beauty full batching cost {_fmt_money(float(beauty["full_batch_cost_usd"]))} versus {_fmt_money(float(beauty["adaptive_cost_usd"]))}, saving {_fmt_money(float(beauty["full_batch_saving_usd"]))} ({100 * float(beauty["full_batch_saving_pct"]):.2f}%). Its paired interval [{_fmt_money(float(beauty["full_batch_interval_95"]["lower_95_usd"]))}, {_fmt_money(float(beauty["full_batch_interval_95"]["upper_95_usd"]))}] was wholly positive. It matched 398/400 with 22 positives, 21 TP, one FP, one FN, recall 95.45%, and FNR 4.55%; it passed the frozen point-estimate gate.

The original fallback was nevertheless justified by the permitted selection evidence: the selection batch failed recall and FNR at 7/8. Held-out results make it excessively conservative in hindsight and expose quality-estimation uncertainty, not evidence that the optimizer could safely have selected it prospectively. The post-hoc quality-constrained oracle regret is {_fmt_money(float(beauty["quality_constrained_oracle_regret_usd"]))} per 400-record execution.

![Beauty missed opportunity](figures/milestone_5_1_final/beauty_missed_opportunity.svg)

## 20. Semantic-quality analysis

{quality_table}

![Quality agreement and positive-reference support](figures/milestone_5_1_final/quality_agreement_support.svg)

The canonical original reporting view has {dedup["canonical_rows"]} rows and {dedup["canonical_unique_plan_ids"]} unique plan IDs, versus {dedup["original_rows"]} rows in the historical artifact. The duplicate Banking77 row was byte-for-byte identical evidence appended under both full-batch and selected-plan roles. It did not cause duplicate inference or weighting.

## 21. Sparse-positive limitations

Four full-batch workloads had zero positive complete-query references and therefore undefined recall; they remain `insufficient_positive_support`. Banking77 primary had exactly three, the minimum, so one disagreement drove recall to 2/3. Perfect or near-perfect overall agreement in a rare-positive AND query cannot substitute for supported positive-class recall.

## 22. Cost-estimator provenance audit

The historical Stage D analyzer calls `predict_plan` with held-out IDs and the held-out lookup. Thus every group mean cost and selectivity comes from evaluation data. For a one-group batch, predicted cost is exactly $N\times\bar c=\sum c_r$, explaining all nine exact matches. The historical MAE of approximately $0.000022485 is therefore a **held-out-informed retrospective diagnostic**, not an independent forecast.

A corrected validation projects the pre-Stage-D persisted 100-record selection predictions to 400 records using the estimator's linear $N$ form. Those inputs reproduce exactly without Stage D information, but the 400-record projection itself was reconstructed after the fact and was not persisted as a Stage D forecast. Its all-plan MAE is {_fmt_money(float(estimator["all_plan_metrics"]["mae_usd"]))}; its full-batch MAE is {_fmt_money(float(estimator["full_batch_metrics"]["mae_usd"]))}; mean within-workload pairwise concordance is {float(estimator["mean_pairwise_ranking_concordance"]):.3f}. This is the defensible out-of-sample diagnostic, with the stated provenance limitation.

## 23. Plan-selection regret

Seven frozen fallback decisions correctly rejected full batching, one Beauty fallback was excessively conservative in hindsight, and the selected Banking77 batch was not confirmed because quality failed. Quality-constrained measured-set oracle regret sums to {_fmt_money(sum(float(row["quality_constrained_oracle_regret_usd"]) for row in workload_rows))}. This is post-hoc oracle regret within 29 measured plans; it is neither prospective optimizer performance nor global regret.

![Measured-set regret](figures/milestone_5_1_final/quality_constrained_oracle_regret.svg)

## 24. Profiling economics

Banking77 execution saving was {_fmt_money(float(primary["selected_saving_usd"]))} per 400-record run. Standalone profiling cost $0.007067130 and breaks even after 13 executions excluding unmetered optimization; shared statistics cost $0.002967972 and break even after 6; warm start has immediate positive API-token economics. At one execution, standalone net benefit is {_fmt_money(float(standalone["net_benefit_excluding_unmetered_optimization_usd"]))} and shared net benefit is {_fmt_money(float(shared["net_benefit_excluding_unmetered_optimization_usd"]))}. Every Banking77 row is marked **QUALITY_INADMISSIBLE_ON_HELD_OUT**; these savings cannot support deployment.

![Profiling break-even](figures/milestone_5_1_final/profiling_break_even.svg)

## 25. Statistical uncertainty

The audit reproduced every stored interval with exact record alignment, seed 51103, and 10,000 paired record-cluster draws. The SMS four-predicate interval is genuinely degenerate: every record has the same paired cost difference, so every bootstrap resample sums to the same statistic. It is not an aggregate-resampling bug. Intervals are nominal and unadjusted; only Banking77 was preregistered as primary. Shared records, predicates, and physical requests make secondary comparisons dependent.

## 26. Full expenditure accounting

{spending_table}

The four completed live experiments sum to {_fmt_money(float(expenditure["four_experiment_token_priced_total_usd"]))}. This is not a complete project spend or provider invoice. Six failed pre-response attempts across selection and Stage D have unknown provider billing, and local optimization/engineering costs are unmetered.

![Experimental expenditure](figures/milestone_5_1_final/experimental_expenditure_by_stage.svg)

## 27. Threats to validity

- Separate Jev inference is a consistency reference, not human ground truth.
- Positive AND outcomes are sparse; recall intervals remain wide (Beauty 21/22 Wilson lower bound {100 * float(beauty_quality["recall_interval_95"]["lower"]):.2f}%).
- There are only three datasets and nine dependent workloads.
- The experiment does not measure identical-request repeatability.
- The plan union is bounded and excludes unmeasured theoretical plans.
- Monetary values are token-price proxies, not invoices.
- Secondary intervals are not multiplicity-adjusted.
- Ex-ante Stage D forecasts were not explicitly persisted; corrected projections are reconstructed from frozen pre-Stage-D inputs.

## 28. Negative findings

Full batching usually lost to early rejection. The sole selected batch failed held-out quality. Four workloads could not certify positive-class behavior. The optimizer missed one admissible opportunity in hindsight. Profiling expense can dominate small per-run savings. The historical estimator artifact leaked held-out measurements into “prediction,” and provider consistency does not establish semantic truth.

## 29. Lessons from implementation

First-class named-question batching and strict response validation worked at scale: 10,800 requests completed without retry. Fingerprinted schedules, immutable plan catalogs, alias deduplication, and fail-closed authorization made the experiment auditable. Derived analyses require equally strict provenance: predictions need explicit information cutoffs, and result rows need evidence identities rather than strategy-role append semantics.

## 30. Research conclusions

Milestone 5.1 established that named-question batching is operationally feasible and economically workload dependent. It did not establish that the current quality-constrained selector generalizes, nor that it delivers a deployable improvement over `existing_adaptive`. The strongest empirical result is the paired combination of a Banking77 economic win with a semantic-consistency failure and a Beauty economic/quality win that selection rejected. Together they show that rare-positive quality qualification—not merely cost modeling—is the dominant unresolved problem.

## 31. Future research directions

{directions}

The recommended Milestone 5.2 hypothesis is: **an uncertainty-aware, actively sampled semantic qualification procedure can reduce paid profiling while lowering the probability of selecting a batch whose held-out recall violates the frozen threshold.** The experiment should preregister a risk criterion (for example, a lower confidence or posterior bound), acquire informative positive/near-threshold records without reading final evaluation outcomes, compare against uniform profiling at equal cost, and include independent human annotations for a subset. This is a proposal only; no M5.2 implementation or inference occurred.

## 32. Reproducibility appendix

- Branch: `{integrity["git_branch"]}`; HEAD: `{integrity["git_head"]}`.
- Stage D schedule fingerprint: `{integrity["schedule"]["stored_fingerprint"]}`.
- Source session: `{STAGE_D_SESSION.as_posix()}`.
- Corrected artifacts: `{ROOT.as_posix()}`.
- Audit generator: `src/semanticflow/optimizer/m51_final_audit.py`.
- Command: `.\\.venv\\Scripts\\python.exe scripts\\run_m51_final_audit.py`.
- No provider was initialized and no Jev call was made.
- Historical result artifacts were read-only and all manifest-covered fingerprints matched.

## 33. Explicit scientific verdict

### Research Question 1 — Does multi-decision batching reduce cost?

- **Observed evidence:** Two of nine full batches were cheaper; seven were costlier.
- **Quantitative result:** Banking77 saved 7.90%; Beauty saved 4.20%.
- **Interpretation:** Benefits are workload dependent and trade shared overhead against early rejection.
- **Limitation:** Nine dependent workloads do not establish a general population rate.
- **Conclusion:** Yes in limited conditions; no as a general replacement.

### Research Question 2 — Do selective-first hybrids add meaningful gains?

- **Observed evidence:** Some measured hybrids improved on full batching or a frozen selection, but not uniformly.
- **Quantitative result:** Effects are reported per measured plan in `per_workload_verified.csv` and regret artifacts.
- **Interpretation:** Hybrids can restore early rejection while sharing downstream tokens.
- **Limitation:** Only frozen remainder groups were measured.
- **Conclusion:** Promising secondarily, not established as a general gain.

### Research Question 3 — Can the optimizer predict batching costs reliably?

- **Observed evidence:** Historical Stage D “predictions” used Stage D measurements; corrected pre-Stage-D projections have nonzero error.
- **Quantitative result:** Corrected full-batch MAE {_fmt_money(float(estimator["full_batch_metrics"]["mae_usd"]))}.
- **Interpretation:** The historical MAE cannot support out-of-sample accuracy.
- **Limitation:** Corrected scaling was reconstructed rather than preregistered as a Stage D forecast.
- **Conclusion:** Reliability is not yet established strongly enough for a general claim.

### Research Question 4 — Does quality-constrained selection generalize?

- **Observed evidence:** The only selected batch passed selection quality and failed held-out quality; one rejected batch passed held-out.
- **Quantitative result:** Banking recall fell from 3/3 to 2/3; Beauty rose from 7/8 to 21/22.
- **Interpretation:** Sparse-positive qualification was unstable across partitions.
- **Limitation:** Counts are small and separate Jev is not truth.
- **Conclusion:** No, generalization was not demonstrated.

### Research Question 5 — Did the frozen policy achieve a quality-admissible net improvement?

- **Observed evidence:** Execution cost fell, but the changed plan failed recall/FNR.
- **Quantitative result:** Aggregate execution-only saving {_fmt_money(float(aggregate["frozen_execution_only_saving_usd"]))}; primary recall 66.67%.
- **Interpretation:** Economic and quality outcomes disagree.
- **Limitation:** Quality is measured as Jev consistency.
- **Conclusion:** No.

### Research Question 6 — How much opportunity did conservative fallback miss?

- **Observed evidence:** One of eight fallbacks was excessively conservative in hindsight.
- **Quantitative result:** Beauty missed {_fmt_money(float(beauty["full_batch_saving_usd"]))} per 400 records; total measured-set quality-constrained oracle regret {_fmt_money(sum(float(row["quality_constrained_oracle_regret_usd"]) for row in workload_rows))}.
- **Interpretation:** Conservatism avoided many losses but missed one admissible held-out opportunity.
- **Limitation:** This is post-hoc oracle hindsight.
- **Conclusion:** Small absolute measured-set regret, scientifically important selection uncertainty.

### Research Question 7 — Are profiling expenses justified?

- **Observed evidence:** Banking savings break even after 13 standalone or 6 shared executions.
- **Quantitative result:** Warm-start API economics are immediately positive.
- **Interpretation:** Amortization can work at reuse, but only for an admissible plan.
- **Limitation:** Optimization cost is unmetered and Banking quality failed.
- **Conclusion:** Not justified for this selected plan; potentially justified for future qualified plans.

### Research Question 8 — Strongest limitations

- **Observed evidence:** Sparse positives, three datasets, dependent comparisons, no human labels, bounded plan union.
- **Quantitative result:** Four full batches had zero positive references; primary had three.
- **Interpretation:** Quality uncertainty dominates.
- **Limitation:** The experiment was designed for consistency and economics, not truth labels or distribution shift.
- **Conclusion:** Claims must remain narrow and conditional.

### Research Question 9 — Most defensible contribution

- **Observed evidence:** A complete fingerprinted multi-decision execution/selection/evaluation pipeline and real held-out measurements.
- **Quantitative result:** 10,800 Stage D requests and 20,000 named evaluations reconciled exactly.
- **Interpretation:** The system exposes the coupled economics/quality trade-off with reproducible evidence.
- **Limitation:** It does not yet solve reliable qualification.
- **Conclusion:** The contribution is empirical infrastructure and the finding that batching suitability is workload- and quality-dependent.

### Research Question 10 — What should Milestone 5.2 investigate?

- **Observed evidence:** Both the harmful selection and missed opportunity arise from unstable rare-positive quality estimates.
- **Quantitative result:** Selection/held-out recall changed 100%→66.7% for Banking and 87.5%→95.45% for Beauty.
- **Interpretation:** Active, uncertainty-aware quality profiling directly targets the observed failure.
- **Limitation:** It will require new preregistration, annotations, and paid inference.
- **Conclusion:** Test whether active uncertainty-aware qualification improves held-out recall safety per profiling dollar.
"""
    return sections


def build_audit_data(*, bootstrap_iterations: int = 10000) -> dict[str, Any]:
    schedule = _json(SCHEDULE_PATH)
    catalog = _json(CATALOG_PATH)["plans"]
    selections = _json(SELECTIONS_PATH)["strategies"]
    config = load_yaml(WORKLOADS_PATH)
    gate = load_yaml(PROTOCOL_PATH)["quality_gate"]
    normalized, lookup, session = normalize_stage_d_session(STAGE_D_SESSION, schedule)
    records_by_dataset = _records_by_dataset(normalized, list(config["datasets"]))
    integrity = _historical_integrity(schedule, catalog, selections, config)
    reconciliation, per_request, per_operator = _request_reconciliation(
        normalized, schedule, catalog, selections
    )
    canonical_original, dedup = _deduplicate_original_quality(config)
    quality_rows, replays, _ = _quality_for_all_measured_plans(
        config["workloads"], schedule, catalog, selections, records_by_dataset, lookup, gate
    )
    disagreements = _semantic_disagreements(
        config["workloads"], catalog, selections, records_by_dataset, lookup
    )
    provenance, ex_ante = _ex_ante_estimator(schedule, catalog, replays)
    workload_rows, aggregate, profiling = _workload_economics(
        config["workloads"],
        catalog,
        selections,
        records_by_dataset,
        lookup,
        quality_rows,
        bootstrap_iterations,
    )
    per_dataset, batch_rows = _dataset_and_batch_rows(per_request)
    uncertainty, bootstrap = _statistical_audit(
        workload_rows,
        catalog,
        selections,
        records_by_dataset,
        lookup,
        bootstrap_iterations,
    )
    expenditure = _historical_expenditure(per_request)
    classifications = {
        "schema_version": "m5.1-fallback-classification-verified-v1",
        "workloads": [
            {
                "workload_id": row["workload_id"],
                "classification": row["classification"],
                "full_batch_cheaper": row["full_batch_saving_usd"] > 0,
                "full_batch_quality_eligible": row["full_batch_quality_eligible"],
                "frozen_selected_plan_id": row["selected_plan_id"],
                "post_hoc_oracle_plan_id": row["best_measured_quality_eligible_plan_id"],
            }
            for row in workload_rows
        ],
        "counts": dict(Counter(str(row["classification"]) for row in workload_rows)),
    }
    regret = {
        "schema_version": "m5.1-frozen-plan-regret-verified-v1",
        "quality_constrained_measured_set_total_usd": sum(
            float(row["quality_constrained_oracle_regret_usd"]) for row in workload_rows
        ),
        "historical_reported_total_usd": _json(STAGE_D_RESULTS / "frozen_plan_regret.json")[
            "total_selected_plan_regret_usd"
        ],
        "historical_metric_classification": (
            "cost-only measured-set regret despite the artifact's measured_eligible_set label"
        ),
        "workloads": [
            {
                "workload_id": row["workload_id"],
                "frozen_selected_plan_id": row["selected_plan_id"],
                "best_measured_quality_eligible_plan_id": row[
                    "best_measured_quality_eligible_plan_id"
                ],
                "regret_usd": row["quality_constrained_oracle_regret_usd"],
                "oracle_available_in_frozen_schedule": True,
                "quality_eligibility_uses_heldout_evidence": True,
                "post_hoc_oracle": True,
                "global_optimality_claim": False,
            }
            for row in workload_rows
        ],
    }
    findings = {
        "schema_version": "m5.1-research-findings-summary-v1",
        "primary_verdict": "economic_success_quality_failure",
        "quality_admissible_improvement_over_existing_adaptive": False,
        "full_batch_cheaper_workloads": sum(
            row["full_batch_saving_usd"] > 0 for row in workload_rows
        ),
        "full_batch_costlier_workloads": sum(
            row["full_batch_saving_usd"] < 0 for row in workload_rows
        ),
        "correct_full_batch_fallbacks": sum(
            row["classification"] == "correct_full_batch_fallback" for row in workload_rows
        ),
        "excessively_conservative_fallbacks": sum(
            row["classification"] == "excessively_conservative_fallback" for row in workload_rows
        ),
        "selected_batches_not_confirmed": sum(
            row["classification"] == "selected_batch_not_confirmed" for row in workload_rows
        ),
        "estimator_historical_out_of_sample_claim_valid": False,
        "quality_duplicate_reporting_only": True,
        "recommended_m52_hypothesis": (
            "uncertainty-aware active semantic profiling reduces qualification cost and harmful "
            "batch selections under rare positives"
        ),
    }
    return {
        "schedule": schedule,
        "catalog": catalog,
        "selections": selections,
        "config": config,
        "session": session,
        "integrity": integrity,
        "reconciliation": reconciliation,
        "per_request": per_request,
        "per_operator": per_operator,
        "canonical_original": canonical_original,
        "dedup": dedup,
        "quality_rows": quality_rows,
        "disagreements": disagreements,
        "provenance": provenance,
        "ex_ante": ex_ante,
        "workload_rows": workload_rows,
        "aggregate": aggregate,
        "profiling": profiling,
        "per_dataset": per_dataset,
        "batch_rows": batch_rows,
        "uncertainty": uncertainty,
        "bootstrap": bootstrap,
        "expenditure": expenditure,
        "classifications": classifications,
        "regret": regret,
        "findings": findings,
    }


def generate(
    output: Path = ROOT,
    *,
    report_path: Path = REPORT,
    figures_path: Path = FIGURES,
    bootstrap_iterations: int = 10000,
) -> dict[str, Any]:
    data = build_audit_data(bootstrap_iterations=bootstrap_iterations)
    output.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Any] = {
        "historical_integrity_audit.json": data["integrity"],
        "request_reconciliation.json": data["reconciliation"],
        "experimental_expenditure.json": data["expenditure"],
        "canonical_plan_quality_results.json": {
            "schema_version": "m5.1-canonical-plan-quality-results-v1",
            "original_reporting_view": data["canonical_original"],
            "all_measured_plan_quality": data["quality_rows"],
        },
        "quality_deduplication_audit.json": data["dedup"],
        "estimator_prediction_provenance.json": data["provenance"],
        "ex_ante_estimator_validation.json": data["ex_ante"],
        "statistical_uncertainty_verified.json": data["uncertainty"],
        "bootstrap_methodology_audit.json": data["bootstrap"],
        "fallback_classification_verified.json": data["classifications"],
        "frozen_plan_regret_verified.json": data["regret"],
        "research_findings_summary.json": data["findings"],
    }
    for name, value in artifacts.items():
        _write_json(output / name, value)
    _write_csv(output / "per_request_expenditure.csv", data["per_request"])
    _write_csv(output / "per_operator_expenditure.csv", data["per_operator"])
    _write_csv(output / "per_workload_verified.csv", data["workload_rows"])
    _write_csv(output / "per_dataset_verified.csv", data["per_dataset"])
    _write_csv(output / "batch_size_verified.csv", data["batch_rows"])
    _write_csv(output / "query_quality_verified.csv", data["quality_rows"])
    _write_csv(output / "semantic_disagreement_cases.csv", data["disagreements"])
    _write_csv(output / "profiling_economics_verified.csv", data["profiling"])
    _write_csv(
        output / "historical_milestone_comparison.csv",
        data["expenditure"]["completed_live_experiments"],
    )
    limitations = """# Analysis limitations

- Separate Jev decisions are a consistency reference, not human ground truth.
- Four full-batch workloads have zero positive complete-query references.
- The primary Banking77 workload has only three positives.
- Workloads share records, predicates, and physical measurements.
- Secondary confidence intervals are nominal and not multiplicity-adjusted.
- The measured 29-plan union does not establish global physical-plan optimality.
- The corrected ex-ante projection was reconstructed after Stage D from frozen pre-Stage-D inputs; it was not originally persisted as a Stage D forecast.
- Token-priced costs are experimental proxies, not provider invoices.
- Failed pre-response attempt billing is unknown.
- Local optimization and engineering costs are unmetered.
"""
    (output / "analysis_limitations.md").write_text(limitations, encoding="utf-8")
    figure_paths = _figures(
        figures_path,
        data["workload_rows"],
        data["batch_rows"],
        data["quality_rows"],
        data["profiling"],
        data["expenditure"],
    )
    report = _report(
        data["integrity"],
        data["reconciliation"],
        data["workload_rows"],
        data["aggregate"],
        data["quality_rows"],
        data["disagreements"],
        data["ex_ante"],
        data["provenance"],
        data["dedup"],
        data["bootstrap"],
        data["expenditure"],
        data["profiling"],
        figures_path,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    source_files = {
        "stage_d_schedule": SCHEDULE_PATH,
        "frozen_catalog": CATALOG_PATH,
        "frozen_selections": SELECTIONS_PATH,
        "stage_d_session": STAGE_D_SESSION / "session.json",
        "selection_integrity_manifest": SELECTION_RESULTS / "artifact_integrity_manifest.json",
        "stage_c_integrity_manifest": STAGE_C_RESULTS / "analysis_manifest.json",
    }
    integrity_manifest = {
        "schema_version": "m5.1-final-audit-artifact-integrity-v1",
        "source_sha256": {name: sha256_file(path) for name, path in source_files.items()},
        "artifact_sha256": {
            path.name: sha256_file(path)
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "artifact_integrity_manifest.json"
        },
        "report": {"path": report_path.as_posix(), "sha256": sha256_file(report_path)},
        "figures": {Path(path).name: sha256_file(Path(path)) for path in sorted(figure_paths)},
        "historical_artifacts_modified": False,
        "new_jev_calls": 0,
        "paid_experimentation": False,
        "frozen_policy_changes": 0,
        "provider_initialized": False,
    }
    _write_json(output / "artifact_integrity_manifest.json", integrity_manifest)
    return {
        "status": "complete",
        "output": output.as_posix(),
        "report": report_path.as_posix(),
        "figures": len(figure_paths),
        "requests_reconciled": data["reconciliation"]["physical_requests"],
        "historical_integrity_valid": data["integrity"]["valid"],
        "new_jev_calls": 0,
    }
