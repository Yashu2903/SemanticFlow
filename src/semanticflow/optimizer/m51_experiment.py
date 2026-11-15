from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from semanticflow.dataset.fusion_benchmark import validate_fusion_dataset
from semanticflow.dataset.loader import load_dataset
from semanticflow.optimizer.fusion_experiment import sha256_file
from semanticflow.storage.results import git_commit, git_worktree_dirty

STAGE_B_AUTHORIZATION = "AUTHORIZE M5.1 STAGE B SMALL LIVE BATCHING PILOT"


def baseline_audit() -> dict[str, Any]:
    stage_b = Path("results/milestone_5/live_fusion_pilot_20261115_stage_b_approved")
    session = json.loads((stage_b / "session.json").read_text("utf-8"))
    decision_files = list(stage_b.glob("*/*/decisions.jsonl"))
    decisions = sum(
        sum(1 for line in path.read_text("utf-8").splitlines() if line)
        for path in decision_files
    )
    historical_paths = [
        stage_b / "session.json",
        Path("results/milestone_5/frozen_plans_20261115_stage_c_corrected_v2/pilot_frozen_plans.json"),
        Path("results/milestone_5/frozen_plans_20261115_stage_c_corrected_v2/stage_d_authorization_proposal.json"),
        Path("configs/milestone_5/fusion.yaml"),
        Path("configs/milestone_5/protocol.yaml"),
    ]
    current_source_paths = [
        Path("src/semanticflow/optimizer/hybrid.py"),
        Path("src/semanticflow/optimizer/hybrid_cost.py"),
        Path("src/semanticflow/optimizer/hybrid_planner.py"),
        Path("src/semanticflow/optimizer/hybrid_profile.py"),
        Path("src/semanticflow/optimizer/hybrid_prompts.py"),
        Path("src/semanticflow/optimizer/hybrid_quality.py"),
        Path("src/semanticflow/optimizer/m51_analysis.py"),
        Path("src/semanticflow/optimizer/m51_experiment.py"),
        Path("src/semanticflow/executor/hybrid.py"),
        Path("src/semanticflow/executor/fused.py"),
        Path("src/semanticflow/providers/jev.py"),
    ]
    if session.get("status") != "complete" or session.get("actual_http_attempts") != 1230:
        raise ValueError("historical Stage B session does not match the published completion")
    if decisions != 1230:
        raise ValueError("historical Stage B decision count is not 1,230")
    return {
        "schema_version": "m5.1-baseline-audit-v1",
        "git_commit": git_commit(),
        "worktree_dirty": git_worktree_dirty(),
        "stage_b_status": session["status"],
        "stage_b_http_attempts": session["actual_http_attempts"],
        "stage_b_decisions": decisions,
        "historical_artifact_sha256": {
            str(path).replace("\\", "/"): sha256_file(path) for path in historical_paths
        },
        "current_source_sha256": {
            str(path).replace("\\", "/"): sha256_file(path)
            for path in current_source_paths
        },
        "held_out_stage_d_inspected": False,
        "provider_initialized": False,
        "live_jev_calls": 0,
    }


def load_yaml(path: str | Path) -> dict[str, Any]:
    result = yaml.safe_load(Path(path).read_text("utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"expected mapping: {path}")
    return result


def pilot_request_counts(schedule: dict[str, Any]) -> dict[str, int]:
    datasets = len(schedule["datasets"])
    representative = int(schedule["representative_records_per_dataset"])
    enriched = int(schedule["enriched_records_per_dataset"])
    core_size = 4
    separate = datasets * representative * core_size
    batches = datasets * representative * 3
    conjunctive = datasets * representative
    enriched_requests = datasets * enriched * 3
    logical = separate + batches + conjunctive + enriched_requests
    if logical != int(schedule["logical_request_ceiling"]):
        raise ValueError("pilot request ceiling differs from frozen schedule")
    return {
        "representative_separate_requests": separate,
        "representative_batch_requests": batches,
        "representative_conjunctive_requests": conjunctive,
        "enriched_diagnostic_requests": enriched_requests,
        "logical_requests": logical,
        "retry_inclusive_http_attempts": logical * 3,
    }


def build_pilot_budget(
    *,
    workloads_path: str | Path,
    protocol_path: str | Path,
    schedule_path: str | Path,
) -> dict[str, Any]:
    workloads = load_yaml(workloads_path)
    protocol = load_yaml(protocol_path)
    schedule = load_yaml(schedule_path)
    counts = pilot_request_counts(schedule)
    # Stage B measured two-question batch and conjunction costs provide the only live anchors.
    historical = Path("results/milestone_5/live_fusion_pilot_20261115_stage_b_approved")
    means: dict[str, dict[str, float]] = {}
    expected_input = expected_output = conservative_input = 0.0
    per_dataset: dict[str, Any] = {}
    for dataset_id, dataset in workloads["datasets"].items():
        data_path = Path(dataset["path"])
        records = [
            record
            for record in load_dataset(data_path)
            if record.metadata.get("partition") == "pilot"
        ][: int(schedule["representative_records_per_dataset"])]
        old_root = historical / dataset_id
        reference_rows = [
            json.loads(line)
            for line in (old_root / "separate_reference" / "decisions.jsonl")
            .read_text("utf-8")
            .splitlines()
            if line
        ]
        multi_path = next(old_root.glob("multi_decision_*/decisions.jsonl"))
        multi_rows = [json.loads(line) for line in multi_path.read_text("utf-8").splitlines()]
        conjunct_rows = [
            json.loads(line)
            for path in old_root.glob("conjunctive_binary_*/decisions.jsonl")
            for line in path.read_text("utf-8").splitlines()
        ]
        sep_in = sum(row["input_tokens"] for row in reference_rows) / len(reference_rows)
        sep_out = sum(row["output_tokens"] for row in reference_rows) / len(reference_rows)
        batch2_in = sum(row["input_tokens"] for row in multi_rows) / len(multi_rows)
        batch2_out = sum(row["output_tokens"] for row in multi_rows) / len(multi_rows)
        conj_in = sum(row["input_tokens"] for row in conjunct_rows) / len(conjunct_rows)
        conj_out = sum(row["output_tokens"] for row in conjunct_rows) / len(conjunct_rows)
        old_text = {
            record.review_id: record.review_text
            for record in load_dataset(Path("data/milestone_5") / f"{dataset_id}.parquet")
        }
        old_mean_chars = sum(
            len(old_text[str(row["review_id"])]) for row in reference_rows
        ) / len(reference_rows)
        new_mean_chars = sum(len(record.review_text) for record in records) / len(records)
        # Text-length adjustment is intentionally cautious; the ID proxy is never called a bound.
        length_adjustment = max(0.75, min(1.5, 1 + (new_mean_chars - old_mean_chars) / 2000))
        n = int(schedule["representative_records_per_dataset"])
        e = int(schedule["enriched_records_per_dataset"])
        input_estimate = (
            n * 4 * sep_in * length_adjustment
            + n * sum(batch2_in * (1 + 0.28 * (size - 2)) for size in (2, 3, 4))
            + n * conj_in
            + e * (2 * sep_in * length_adjustment + batch2_in)
        )
        output_estimate = (
            n * 4 * sep_out
            + n * sum(batch2_out * size / 2 for size in (2, 3, 4))
            + n * conj_out
            + e * (2 * sep_out + batch2_out)
        )
        conservative = input_estimate * 1.5
        expected_input += input_estimate
        expected_output += output_estimate
        conservative_input += conservative
        per_dataset[dataset_id] = {
            "records": n,
            "historical_anchor": {
                "separate_mean_input_tokens": sep_in,
                "two_question_batch_mean_input_tokens": batch2_in,
                "conjunctive_mean_input_tokens": conj_in,
            },
            "expected_input_tokens": round(input_estimate),
            "expected_output_tokens": round(output_estimate),
            "expected_cost_usd": input_estimate * 0.042 / 1_000_000,
            "conservative_input_token_proxy": round(conservative),
        }
        means[dataset_id] = {"new_mean_text_chars": new_mean_chars}
    expected_cost = expected_input * 0.042 / 1_000_000
    conservative_cost = conservative_input * 0.042 / 1_000_000
    payload = {
        "schema_version": "m5.1-stage-b-pilot-budget-v1",
        "model": protocol["model"],
        "cache": protocol["cache"],
        "concurrency": protocol["concurrency"],
        "max_retries": protocol["max_retries"],
        **counts,
        "expected_input_tokens": round(expected_input),
        "expected_output_tokens": round(expected_output),
        "expected_token_priced_cost_usd": expected_cost,
        "conservative_input_token_proxy": round(conservative_input),
        "conservative_no_retry_cost_proxy_usd": conservative_cost,
        "retry_inclusive_cost_proxy_usd": conservative_cost * 3,
        "per_dataset": per_dataset,
        "monetary_limit_kind": "empirical proxy; hard request ceilings only",
        "output_directory": schedule["output_directory"],
        "authorization_phrase": schedule["authorization_phrase"],
        "live_jev_calls": 0,
    }
    payload["budget_fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def build_authorization_manifest(
    *,
    workloads_path: str | Path,
    protocol_path: str | Path,
    schedule_path: str | Path,
) -> dict[str, Any]:
    budget = build_pilot_budget(
        workloads_path=workloads_path,
        protocol_path=protocol_path,
        schedule_path=schedule_path,
    )
    workloads = load_yaml(workloads_path)
    datasets = {
        dataset_id: {
            "path": definition["path"],
            "validation": validate_fusion_dataset(definition["path"]),
            "manifest_sha256": sha256_file(
                str(definition["path"]) + ".manifest.json"
            ),
        }
        for dataset_id, definition in workloads["datasets"].items()
    }
    return {
        "schema_version": "m5.1-stage-b-authorization-manifest-v1",
        "authorization_phrase": STAGE_B_AUTHORIZATION,
        "configuration_sha256": {
            "workloads": sha256_file(workloads_path),
            "protocol": sha256_file(protocol_path),
            "schedule": sha256_file(schedule_path),
        },
        "datasets": datasets,
        "budget": budget,
        "output_directory": budget["output_directory"],
        "provider_must_not_initialize_before_validation": True,
    }


def validate_stage_b_launch(
    manifest: dict[str, Any],
    *,
    authorization: str | None,
    output_directory: str | Path,
    workloads_path: str | Path,
    protocol_path: str | Path,
    schedule_path: str | Path,
) -> None:
    if authorization != STAGE_B_AUTHORIZATION:
        raise PermissionError("exact M5.1 Stage B authorization phrase required")
    expected = build_authorization_manifest(
        workloads_path=workloads_path,
        protocol_path=protocol_path,
        schedule_path=schedule_path,
    )
    if manifest != expected:
        raise ValueError("authorization manifest differs from frozen experiment")
    if str(output_directory) != expected["output_directory"]:
        raise ValueError("output directory differs from authorized manifest")
    if Path(output_directory).exists():
        raise FileExistsError("refusing to overwrite or resume a paid experiment")
