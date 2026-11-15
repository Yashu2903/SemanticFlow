# ruff: noqa: E501

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from semanticflow.optimizer.fusion import build_fused_operator
from semanticflow.query.models import PredicateSpec

INPUT_PRICE_PER_MILLION = 0.042
OUTPUT_PRICE_PER_MILLION = 0.0
ANALYSIS_SCHEMA = "m5.1-stage-c-detailed-analysis-v1"
ANALYSIS_SEED = 51031
BOOTSTRAP_ITERATIONS = 2000


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text("utf-8").splitlines()
        if line.strip()
    ]


def read_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def token_cost(input_tokens: int | float, output_tokens: int | float = 0) -> float:
    return (
        float(input_tokens) * INPUT_PRICE_PER_MILLION
        + float(output_tokens) * OUTPUT_PRICE_PER_MILLION
    ) / 1_000_000


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, Any]:
    if total == 0:
        return {"estimate": None, "lower": None, "upper": None, "numerator": 0, "denominator": 0}
    estimate = successes / total
    denominator = 1 + z * z / total
    center = (estimate + z * z / (2 * total)) / denominator
    half = (
        z
        * math.sqrt(estimate * (1 - estimate) / total + z * z / (4 * total * total))
        / denominator
    )
    return {
        "estimate": estimate,
        "lower": max(0.0, center - half),
        "upper": min(1.0, center + half),
        "numerator": successes,
        "denominator": total,
        "method": "Wilson score 95%",
    }


def binary_metrics(reference: Sequence[bool], candidate: Sequence[bool]) -> dict[str, Any]:
    if len(reference) != len(candidate):
        raise ValueError("binary comparisons must align")
    tp = sum(left and right for left, right in zip(reference, candidate, strict=True))
    tn = sum(not left and not right for left, right in zip(reference, candidate, strict=True))
    fp = sum(not left and right for left, right in zip(reference, candidate, strict=True))
    fn = sum(left and not right for left, right in zip(reference, candidate, strict=True))
    count = len(reference)
    union = tp + fp + fn
    return {
        "sample_count": count,
        "reference_positive_count": tp + fn,
        "reference_negative_count": tn + fp,
        "candidate_positive_count": tp + fp,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "agreements": tp + tn,
        "agreement": ratio(tp + tn, count),
        "changed_decisions": fp + fn,
        "changed_decision_rate": ratio(fp + fn, count),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "false_positive_rate": ratio(fp, fp + tn),
        "false_negative_rate": ratio(fn, fn + tp),
        "jaccard": tp / union if union else (1.0 if count else None),
        "agreement_interval_95": wilson_interval(tp + tn, count),
        "changed_interval_95": wilson_interval(fp + fn, count),
        "false_positive_interval_95": wilson_interval(fp, fp + tn),
        "false_negative_interval_95": wilson_interval(fn, fn + tp),
        "recall_interval_95": wilson_interval(tp, tp + fn),
        "reference_kind": "separate_jev",
        "semantic_accuracy_established": False,
    }


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def paired_bootstrap_interval(
    differences: Sequence[float], *, seed: int, iterations: int = BOOTSTRAP_ITERATIONS
) -> dict[str, Any]:
    if not differences:
        return {"estimate": None, "lower": None, "upper": None, "records": 0}
    generator = random.Random(seed)
    draws = []
    count = len(differences)
    for _ in range(iterations):
        draws.append(statistics.mean(differences[generator.randrange(count)] for _ in range(count)))
    return {
        "estimate": statistics.mean(differences),
        "lower": percentile(draws, 0.025),
        "upper": percentile(draws, 0.975),
        "records": count,
        "iterations": iterations,
        "method": "paired record bootstrap 95%",
    }


def _question_names(operator: Mapping[str, Any]) -> list[str]:
    if operator["fusion_mode"] == "multi_decision":
        return list(operator["predicate_ids"])
    return [str(operator["group_id"])]


def _named_decisions(trace: Mapping[str, Any], operator: Mapping[str, Any]) -> dict[str, bool]:
    threshold = float(trace["threshold"])
    if operator["fusion_mode"] == "multi_decision":
        return {
            str(name): float(probability) >= threshold
            for name, probability in trace["predicate_probabilities"].items()
        }
    predicate_id = str(operator["predicate_ids"][0]) if operator["fusion_mode"] == "single" else str(operator["group_id"])
    return {predicate_id: bool(trace["passed"])}


def _compatible_workloads(
    workloads: Mapping[str, Any], dataset: str, predicate_ids: Sequence[str]
) -> list[str]:
    group = set(predicate_ids)
    return [
        str(item["id"])
        for item in workloads["workloads"]
        if item["dataset"] == dataset and group.issubset(set(item["predicate_ids"]))
    ]


def load_stage_b_requests(
    root: str | Path,
    workloads: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    root = Path(root)
    rows: list[dict[str, Any]] = []
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for decisions_path in sorted(root.glob("*/*/*/decisions.jsonl")):
        relative = decisions_path.relative_to(root)
        dataset, sample_kind = relative.parts[:2]
        operator_path = decisions_path.with_name("operator.json")
        report_path = decisions_path.with_name("report.json")
        operator = read_json(operator_path)
        report = read_json(report_path)
        partition = "pilot" if sample_kind == "representative" else "development"
        predicates = list(operator["predicate_ids"])
        compatible = _compatible_workloads(workloads, dataset, predicates)
        exact = [
            item
            for item in workloads["workloads"]
            if item["dataset"] == dataset and list(item["predicate_ids"]) == predicates
        ]
        logical_workload = str(exact[0]["id"]) if len(exact) == 1 else None
        for trace in read_jsonl(decisions_path):
            named = _named_decisions(trace, operator)
            probabilities = dict(trace["predicate_probabilities"])
            if operator["fusion_mode"] in {"single", "conjunctive_binary"}:
                probabilities = {
                    next(iter(named)): float(trace["group_probability"])
                }
            attempt_count = int(trace["request_attempt_count"])
            row = {
                "request_id": sha256_json(
                    {
                        "trace": relative.as_posix(),
                        "record": trace["review_id"],
                    }
                )[:24],
                "dataset": dataset,
                "sample_kind": sample_kind,
                "partition": partition,
                "workload_id": logical_workload,
                "compatible_workload_ids": compatible,
                "record_id": str(trace["review_id"]),
                "operator_id": str(operator["group_id"]),
                "physical_plan_id": None,
                "physical_plan_id_status": "not_persisted_by_stage_b_runner",
                "execution_mode": str(operator["fusion_mode"]),
                "group_id": str(operator["group_id"]),
                "batch_size": len(predicates),
                "predicate_ids": predicates,
                "predicate_question_names": _question_names(operator),
                "prompt_hash": str(operator["fusion_prompt_hash"]),
                "predicate_hashes": dict(operator["predicate_hashes"]),
                "prompt_template_version": str(operator["prompt_template_version"]),
                "requested_model": str(operator["requested_model"]),
                "resolved_model": str(trace["model_identifier"]),
                "input_tokens": int(trace["input_tokens"]),
                "output_tokens": int(trace["output_tokens"]),
                "estimated_cost_usd": token_cost(trace["input_tokens"], trace["output_tokens"]),
                "provider_latency_seconds": float(trace["request_latency_seconds"]),
                "end_to_end_latency_seconds": None,
                "http_attempt_count": attempt_count,
                "retry_count": attempt_count - 1,
                "response_status": int(trace["provider_metadata"]["http_status"]),
                "individual_named_probabilities": probabilities,
                "individual_named_decisions": named,
                "group_conjunction_decision": bool(trace["passed"]),
                "timestamp": str(trace["timestamp"]),
                "cache_status": str(trace["cache_status"]),
                "run_id": str(trace["run_id"]),
                "trace_path": decisions_path.as_posix(),
                "operator_path": operator_path.as_posix(),
                "report_path": report_path.as_posix(),
                "operator_complete": bool(report["complete"]),
                "provider": str(trace["provider"]),
            }
            key = (dataset, str(trace["review_id"]), str(operator["group_id"]))
            if key in index:
                raise ValueError(f"duplicate physical request: {key}")
            index[key] = row
            rows.append(row)
    return rows, index


def validate_operator_hash(
    operator: Mapping[str, Any], dataset: str, workloads: Mapping[str, Any]
) -> bool:
    definitions = workloads["datasets"][dataset]["predicates"]
    specs = [
        PredicateSpec(id=predicate_id, instructions=definitions[predicate_id])
        for predicate_id in operator["predicate_ids"]
    ]
    rebuilt = build_fused_operator(
        specs,
        mode=operator["fusion_mode"],
        requested_model=operator["requested_model"],
        threshold=float(operator["threshold"]),
        execution_parameters=operator["execution_parameters"],
    )
    return rebuilt.model_dump(mode="json") == dict(operator)


def _manifest_partition_ids(manifest: Mapping[str, Any]) -> dict[str, set[str]]:
    return {
        name: set(value["ids"])
        for name, value in manifest["partitions"].items()
    }


def build_integrity_audit(
    *,
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    workloads: Mapping[str, Any],
    protocol: Mapping[str, Any],
    schedule: Mapping[str, Any],
    authorization_manifest: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    raw_files = sorted(path for path in root.rglob("*") if path.is_file())
    reports = [read_json(path) for path in root.glob("*/*/*/report.json")]
    operators = [
        (path, read_json(path)) for path in sorted(root.glob("*/*/*/operator.json"))
    ]
    partition_checks: dict[str, Any] = {}
    all_partition_valid = True
    held_out_overlap = 0
    dataset_hash_checks: dict[str, Any] = {}
    for dataset, definition in workloads["datasets"].items():
        manifest_path = Path(str(definition["path"]) + ".manifest.json")
        dataset_manifest = read_json(manifest_path)
        partitions = _manifest_partition_ids(dataset_manifest)
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        invalid = [
            row["record_id"]
            for row in dataset_rows
            if row["record_id"] not in partitions[row["partition"]]
        ]
        evaluation_overlap = sorted(
            {row["record_id"] for row in dataset_rows} & partitions["evaluation"]
        )
        held_out_overlap += len(evaluation_overlap)
        all_partition_valid &= not invalid and not evaluation_overlap
        authorized = authorization_manifest["datasets"][dataset]
        dataset_hash_checks[dataset] = {
            "dataset_file_sha256": sha256_file(definition["path"]),
            "authorized_output_sha256": authorized["validation"]["output_sha256"],
            "dataset_file_matches_authorization": (
                sha256_file(definition["path"])
                == authorized["validation"]["output_sha256"]
            ),
            "manifest_sha256": sha256_file(manifest_path),
            "authorized_manifest_sha256": authorized["manifest_sha256"],
            "manifest_matches_authorization": (
                sha256_file(manifest_path) == authorized["manifest_sha256"]
            ),
            "dataset_fingerprint_sha256": dataset_manifest["dataset_fingerprint_sha256"],
            "authorized_dataset_fingerprint_sha256": authorized["validation"][
                "dataset_fingerprint_sha256"
            ],
        }
        partition_checks[dataset] = {
            "recorded_requests": len(dataset_rows),
            "invalid_partition_membership": invalid,
            "evaluation_partition_overlap": evaluation_overlap,
            "partitions_used": sorted({str(row["partition"]) for row in dataset_rows}),
        }

    category_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if row["sample_kind"] == "enriched":
            category_counts["enriched_diagnostic_requests"] += 1
        elif row["execution_mode"] == "single":
            category_counts["representative_separate_requests"] += 1
        elif row["execution_mode"] == "multi_decision":
            category_counts["representative_batch_requests"] += 1
        elif row["execution_mode"] == "conjunctive_binary":
            category_counts["representative_conjunctive_requests"] += 1

    config_paths = {
        "workloads": Path("configs/milestone_5_1/workloads.yaml"),
        "protocol": Path("configs/milestone_5_1/protocol.yaml"),
        "schedule": Path("configs/milestone_5_1/pilot_schedule.yaml"),
    }
    config_hashes = {name: sha256_file(path) for name, path in config_paths.items()}
    config_match = {
        name: digest == authorization_manifest["configuration_sha256"][name]
        for name, digest in config_hashes.items()
    }
    source_before = preflight["baseline_audit"]["current_source_sha256"]
    source_checks = {
        path: {
            "authorized_preflight_sha256": expected,
            "current_sha256": sha256_file(path),
            "unchanged_since_preflight": sha256_file(path) == expected,
        }
        for path, expected in source_before.items()
    }
    historical_checks = {
        path: {
            "preflight_sha256": expected,
            "current_sha256": sha256_file(path),
            "unchanged": sha256_file(path) == expected,
        }
        for path, expected in preflight["baseline_audit"]["historical_artifact_sha256"].items()
    }
    prompt_checks = []
    for path, operator in operators:
        dataset = path.relative_to(root).parts[0]
        prompt_checks.append(
            {
                "operator_path": path.as_posix(),
                "group_id": operator["group_id"],
                "rebuild_matches_persisted": validate_operator_hash(operator, dataset, workloads),
                "trace_hashes_match": all(
                    row["prompt_hash"] == operator["fusion_prompt_hash"]
                    for row in rows
                    if row["operator_path"] == path.as_posix()
                ),
            }
        )
    named = sum(len(row["predicate_question_names"]) for row in rows)
    session = read_json(root / "session.json")
    attempts = sum(int(row["http_attempt_count"]) for row in rows)
    unique_keys = {
        (row["dataset"], row["record_id"], row["group_id"], row["sample_kind"])
        for row in rows
    }
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "audit_time_utc": datetime.now(UTC).isoformat(),
        "stage_b_root": root.as_posix(),
        "session": session,
        "authorization_scope": {
            "exact_phrase": schedule["authorization_phrase"],
            "manifest_phrase": authorization_manifest["authorization_phrase"],
            "output_directory": authorization_manifest["output_directory"],
            "stage_d_authorized": False,
        },
        "configuration": {
            "hashes": config_hashes,
            "match_authorized_manifest": config_match,
            "protocol_schema": protocol["schema_version"],
            "cache": protocol["cache"],
            "concurrency": protocol["concurrency"],
            "max_retries": protocol["max_retries"],
            "threshold": protocol["threshold"],
        },
        "source_integrity": {
            "git_commit_at_preflight": preflight["baseline_audit"]["git_commit"],
            "files": source_checks,
            "all_unchanged": all(item["unchanged_since_preflight"] for item in source_checks.values()),
        },
        "historical_integrity": {
            "files": historical_checks,
            "all_unchanged": all(item["unchanged"] for item in historical_checks.values()),
            "m4_6_note": "M4.6 files were not modified or reinterpreted; only published hashes were checked where available.",
        },
        "dataset_integrity": dataset_hash_checks,
        "partition_integrity": {
            "datasets": partition_checks,
            "all_record_ids_in_authorized_partitions": all_partition_valid,
            "held_out_record_ids_observed": held_out_overlap,
            "held_out_contents_read_by_analysis": False,
            "partition_membership_source": "frozen manifest IDs only",
        },
        "prompt_integrity": {
            "operators": prompt_checks,
            "all_rebuilt": all(item["rebuild_matches_persisted"] for item in prompt_checks),
            "all_trace_hashes_match": all(item["trace_hashes_match"] for item in prompt_checks),
        },
        "physical_plan_integrity": {
            "persisted_plan_files": 0,
            "status": "not_available",
            "reason": "Stage B persisted operator metadata and run IDs, but not the build_physical_plan result or plan hash.",
        },
        "execution_reconciliation": {
            "physical_requests": len(rows),
            "unique_physical_request_keys": len(unique_keys),
            "http_attempts": attempts,
            "named_question_evaluations": named,
            "operators": len(operators),
            "reports": len(reports),
            "complete_reports": sum(bool(report["complete"]) for report in reports),
            "provider_error_entries": sum(len(report["errors"]) for report in reports),
            "category_counts": dict(category_counts),
            "category_counts_match_schedule": all(
                category_counts[name] == int(authorization_manifest["budget"][name])
                for name in category_counts
            ),
            "all_requests_accounted": (
                len(rows) == attempts == int(session["actual_http_attempts"]) == 810
                and named == 1380
            ),
        },
        "model_and_provider": {
            "requested_models": sorted({str(row["requested_model"]) for row in rows}),
            "resolved_models": sorted({str(row["resolved_model"]) for row in rows}),
            "providers": sorted({str(row["provider"]) for row in rows}),
            "cache_statuses": sorted({str(row["cache_status"]) for row in rows}),
            "http_statuses": sorted({int(row["response_status"]) for row in rows}),
            "retry_count": sum(int(row["retry_count"]) for row in rows),
        },
        "raw_artifacts": {
            "file_count": len(raw_files),
            "total_bytes": sum(path.stat().st_size for path in raw_files),
            "files": {
                path.relative_to(root).as_posix(): sha256_file(path) for path in raw_files
            },
        },
        "live_jev_calls_during_analysis": 0,
    }


def _sum_fields(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    inputs = sum(int(row["input_tokens"]) for row in materialized)
    outputs = sum(int(row["output_tokens"]) for row in materialized)
    return {
        "requests": len(materialized),
        "http_attempts": sum(int(row["http_attempt_count"]) for row in materialized),
        "input_tokens": inputs,
        "output_tokens": outputs,
        "cost_usd": token_cost(inputs, outputs),
        "summed_provider_latency_seconds": sum(
            float(row["provider_latency_seconds"]) for row in materialized
        ),
    }


def build_cost_reconciliation(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def grouped(keys: Sequence[str]) -> list[dict[str, Any]]:
        buckets: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[tuple(row[key] for key in keys)].append(row)
        return [
            {**dict(zip(keys, key, strict=True)), **_sum_fields(values)}
            for key, values in sorted(buckets.items(), key=lambda item: str(item[0]))
        ]

    total = _sum_fields(rows)
    total["named_question_evaluations"] = sum(
        len(row["predicate_question_names"]) for row in rows
    )
    total["recorded_input_token_priced_cost_usd"] = 0.012575052
    total["cost_reconciles"] = math.isclose(
        total["cost_usd"], total["recorded_input_token_priced_cost_usd"], abs_tol=1e-15
    )
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "pricing": {
            "input_per_million_usd": INPUT_PRICE_PER_MILLION,
            "output_per_million_usd": OUTPUT_PRICE_PER_MILLION,
            "basis": "frozen SemanticFlow Jev experimental pricing assumption",
            "complete_provider_bill_claimed": False,
            "limitation": "The artifacts do not include a provider invoice or billing contract; this is token-priced experimental cost, not a verified bill.",
        },
        "total": total,
        "per_request": [
            {
                "request_id": row["request_id"],
                "dataset": row["dataset"],
                "sample_kind": row["sample_kind"],
                "group_id": row["group_id"],
                "record_id": row["record_id"],
                "input_tokens": row["input_tokens"],
                "output_tokens": row["output_tokens"],
                "cost_usd": row["estimated_cost_usd"],
            }
            for row in rows
        ],
        "per_operator": grouped(["dataset", "sample_kind", "execution_mode", "group_id"]),
        "per_dataset": grouped(["dataset"]),
        "per_dataset_sample": grouped(["dataset", "sample_kind"]),
        "per_mode": grouped(["sample_kind", "execution_mode", "batch_size"]),
        "per_workload": grouped(["workload_id"]),
    }


def _measurement_indexes(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[tuple[str, str, str, str], Mapping[str, Any]],
    dict[tuple[str, str, tuple[str, ...], str], list[Mapping[str, Any]]],
]:
    singles: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    groups: dict[tuple[str, str, tuple[str, ...], str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["execution_mode"] == "single":
            predicate_id = str(row["predicate_ids"][0])
            singles[(row["dataset"], row["sample_kind"], row["record_id"], predicate_id)] = row
        groups[
            (
                row["dataset"],
                row["sample_kind"],
                tuple(row["predicate_ids"]),
                row["execution_mode"],
            )
        ].append(row)
    return singles, groups


def _single_decision(row: Mapping[str, Any], predicate_id: str) -> bool:
    return bool(row["individual_named_decisions"][predicate_id])


def _single_probability(row: Mapping[str, Any], predicate_id: str) -> float:
    return float(row["individual_named_probabilities"][predicate_id])


def _cost_vector(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    return [float(row["estimated_cost_usd"]) for row in rows]


def build_batch_comparisons(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    singles, groups = _measurement_indexes(rows)
    comparisons: list[dict[str, Any]] = []
    predicate_agreement: list[dict[str, Any]] = []
    group_agreement: list[dict[str, Any]] = []
    query_agreement: list[dict[str, Any]] = []
    for key, batch_rows in sorted(groups.items()):
        dataset, sample_kind, predicate_tuple, mode = key
        if mode not in {"multi_decision", "conjunctive_binary"}:
            continue
        batch_rows = sorted(batch_rows, key=lambda row: row["record_id"])
        missing: list[str] = []
        separate_by_record: dict[str, list[Mapping[str, Any]]] = {}
        for batch in batch_rows:
            cells = []
            for predicate_id in predicate_tuple:
                cell = singles.get((dataset, sample_kind, batch["record_id"], predicate_id))
                if cell is None:
                    missing.append(f"{batch['record_id']}:{predicate_id}")
                else:
                    cells.append(cell)
            if len(cells) == len(predicate_tuple):
                separate_by_record[str(batch["record_id"])] = cells
        comparable = not missing and len(separate_by_record) == len(batch_rows)
        all_separate: list[Mapping[str, Any]] = []
        early_separate: list[Mapping[str, Any]] = []
        per_record_all_cost: list[float] = []
        per_record_early_cost: list[float] = []
        per_record_batch_cost: list[float] = []
        reference_query: list[bool] = []
        candidate_query: list[bool] = []
        if comparable:
            for batch in batch_rows:
                separate = separate_by_record[str(batch["record_id"])]
                all_separate.extend(separate)
                early_cells = []
                for cell, predicate_id in zip(separate, predicate_tuple, strict=True):
                    early_cells.append(cell)
                    if not _single_decision(cell, predicate_id):
                        break
                early_separate.extend(early_cells)
                per_record_all_cost.append(sum(_cost_vector(separate)))
                per_record_early_cost.append(sum(_cost_vector(early_cells)))
                per_record_batch_cost.append(float(batch["estimated_cost_usd"]))
                reference_query.append(
                    all(
                        _single_decision(cell, predicate_id)
                        for cell, predicate_id in zip(separate, predicate_tuple, strict=True)
                    )
                )
                candidate_query.append(bool(batch["group_conjunction_decision"]))
        batch_total = _sum_fields(batch_rows)
        all_total = _sum_fields(all_separate) if comparable else None
        early_total = _sum_fields(early_separate) if comparable else None
        comparison_id = f"{dataset}:{sample_kind}:{mode}:{batch_rows[0]['group_id']}"
        comparison = {
            "comparison_id": comparison_id,
            "dataset": dataset,
            "sample_kind": sample_kind,
            "partition": batch_rows[0]["partition"],
            "execution_mode": mode,
            "group_id": batch_rows[0]["group_id"],
            "workload_id": batch_rows[0]["workload_id"],
            "predicate_ids": list(predicate_tuple),
            "batch_size": len(predicate_tuple),
            "records": len(batch_rows),
            "evidence_type": "fully_measured_paired" if comparable else "not_evaluable",
            "missing_reference_cells": missing,
            "batch": batch_total,
            "all_questions_separate": all_total,
            "early_reject_separate": early_total,
            "batch_minus_all_separate_cost_usd": (
                batch_total["cost_usd"] - all_total["cost_usd"] if all_total else None
            ),
            "batch_saving_vs_all_separate_usd": (
                all_total["cost_usd"] - batch_total["cost_usd"] if all_total else None
            ),
            "batch_saving_vs_early_separate_usd": (
                early_total["cost_usd"] - batch_total["cost_usd"] if early_total else None
            ),
            "request_reduction_vs_all_separate": (
                all_total["requests"] - batch_total["requests"] if all_total else None
            ),
            "request_reduction_vs_early_separate": (
                early_total["requests"] - batch_total["requests"] if early_total else None
            ),
            "input_token_reduction_vs_all_separate": (
                all_total["input_tokens"] - batch_total["input_tokens"] if all_total else None
            ),
            "input_token_reduction_vs_early_separate": (
                early_total["input_tokens"] - batch_total["input_tokens"]
                if early_total
                else None
            ),
            "cost_per_record_usd": batch_total["cost_usd"] / len(batch_rows),
            "input_tokens_per_request": batch_total["input_tokens"] / len(batch_rows),
            "output_tokens_per_request": batch_total["output_tokens"] / len(batch_rows),
            "input_tokens_per_named_question": (
                batch_total["input_tokens"] / (len(batch_rows) * len(predicate_tuple))
                if mode == "multi_decision"
                else None
            ),
            "cost_per_named_question_usd": (
                batch_total["cost_usd"] / (len(batch_rows) * len(predicate_tuple))
                if mode == "multi_decision"
                else None
            ),
            "paired_batch_minus_all_cost_mean_interval_95": (
                paired_bootstrap_interval(
                    [batch - separate for batch, separate in zip(per_record_batch_cost, per_record_all_cost, strict=True)],
                    seed=ANALYSIS_SEED + len(comparisons),
                )
                if comparable
                else None
            ),
            "paired_batch_minus_early_cost_mean_interval_95": (
                paired_bootstrap_interval(
                    [batch - separate for batch, separate in zip(per_record_batch_cost, per_record_early_cost, strict=True)],
                    seed=ANALYSIS_SEED + 100 + len(comparisons),
                )
                if comparable
                else None
            ),
        }
        comparisons.append(comparison)
        if not comparable:
            continue
        query_metrics = binary_metrics(reference_query, candidate_query)
        query_agreement.append(
            {
                "comparison_id": comparison_id,
                "dataset": dataset,
                "sample_kind": sample_kind,
                "workload_id": batch_rows[0]["workload_id"],
                "execution_mode": mode,
                "group_id": batch_rows[0]["group_id"],
                "batch_size": len(predicate_tuple),
                "predicate_ids": list(predicate_tuple),
                **query_metrics,
                "match_set_equal": {
                    row["record_id"]
                    for row, passed in zip(batch_rows, candidate_query, strict=True)
                    if passed
                }
                == {
                    row["record_id"]
                    for row, passed in zip(batch_rows, reference_query, strict=True)
                    if passed
                },
                "complete_plan_evidence": mode in {"multi_decision", "conjunctive_binary"},
            }
        )
        group_agreement.append(
            {
                "comparison_id": comparison_id,
                "dataset": dataset,
                "sample_kind": sample_kind,
                "workload_id": batch_rows[0]["workload_id"],
                "execution_mode": mode,
                "group_id": batch_rows[0]["group_id"],
                "batch_size": len(predicate_tuple),
                "predicate_ids": list(predicate_tuple),
                **query_metrics,
            }
        )
        if mode != "multi_decision":
            continue
        for predicate_id in predicate_tuple:
            reference_decisions = []
            batch_decisions = []
            absolute_differences = []
            crossings = []
            near_threshold = 0
            high_confidence = 0
            for batch in batch_rows:
                separate = singles[(dataset, sample_kind, batch["record_id"], predicate_id)]
                reference_probability = _single_probability(separate, predicate_id)
                candidate_probability = float(
                    batch["individual_named_probabilities"][predicate_id]
                )
                reference_pass = _single_decision(separate, predicate_id)
                candidate_pass = bool(batch["individual_named_decisions"][predicate_id])
                reference_decisions.append(reference_pass)
                batch_decisions.append(candidate_pass)
                absolute_differences.append(abs(candidate_probability - reference_probability))
                if reference_pass != candidate_pass:
                    crossings.append(
                        {
                            "record_id": batch["record_id"],
                            "separate_probability": reference_probability,
                            "batch_probability": candidate_probability,
                        }
                    )
                    near_threshold += int(
                        min(abs(reference_probability - 0.5), abs(candidate_probability - 0.5))
                        <= 0.1
                    )
                    high_confidence += int(
                        abs(reference_probability - 0.5) >= 0.25
                        and abs(candidate_probability - 0.5) >= 0.25
                    )
            metrics = binary_metrics(reference_decisions, batch_decisions)
            predicate_agreement.append(
                {
                    "comparison_id": comparison_id,
                    "dataset": dataset,
                    "sample_kind": sample_kind,
                    "workload_id": batch_rows[0]["workload_id"],
                    "group_id": batch_rows[0]["group_id"],
                    "batch_size": len(predicate_tuple),
                    "predicate_id": predicate_id,
                    **metrics,
                    "mean_absolute_probability_difference": statistics.mean(absolute_differences),
                    "median_absolute_probability_difference": statistics.median(absolute_differences),
                    "p95_absolute_probability_difference": percentile(absolute_differences, 0.95),
                    "threshold_crossings": crossings,
                    "near_threshold_disagreements": near_threshold,
                    "high_confidence_disagreements": high_confidence,
                }
            )
    return comparisons, predicate_agreement, group_agreement, query_agreement


def build_batch_size_economics(
    comparisons: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    multi = [row for row in comparisons if row["execution_mode"] == "multi_decision"]
    for sample_kind in ("representative", "enriched"):
        sample = [row for row in multi if row["sample_kind"] == sample_kind]
        for size in sorted({int(row["batch_size"]) for row in sample}):
            selected = [row for row in sample if row["batch_size"] == size]
            if not selected:
                continue
            batch_requests = sum(row["batch"]["requests"] for row in selected)
            batch_input = sum(row["batch"]["input_tokens"] for row in selected)
            batch_output = sum(row["batch"]["output_tokens"] for row in selected)
            all_requests = sum(row["all_questions_separate"]["requests"] for row in selected)
            all_input = sum(row["all_questions_separate"]["input_tokens"] for row in selected)
            all_output = sum(row["all_questions_separate"]["output_tokens"] for row in selected)
            early_requests = sum(row["early_reject_separate"]["requests"] for row in selected)
            early_input = sum(row["early_reject_separate"]["input_tokens"] for row in selected)
            early_output = sum(row["early_reject_separate"]["output_tokens"] for row in selected)
            batch_latency = sum(
                row["batch"]["summed_provider_latency_seconds"] for row in selected
            )
            all_latency = sum(
                row["all_questions_separate"]["summed_provider_latency_seconds"]
                for row in selected
            )
            early_latency = sum(
                row["early_reject_separate"]["summed_provider_latency_seconds"]
                for row in selected
            )
            output.append(
                {
                    "sample_kind": sample_kind,
                    "batch_size": size,
                    "datasets": len(selected),
                    "records": sum(row["records"] for row in selected),
                    "batch_requests": batch_requests,
                    "batch_input_tokens": batch_input,
                    "batch_output_tokens": batch_output,
                    "batch_cost_usd": token_cost(batch_input, batch_output),
                    "all_separate_requests": all_requests,
                    "all_separate_input_tokens": all_input,
                    "all_separate_output_tokens": all_output,
                    "all_separate_cost_usd": token_cost(all_input, all_output),
                    "early_separate_requests": early_requests,
                    "early_separate_input_tokens": early_input,
                    "early_separate_output_tokens": early_output,
                    "early_separate_cost_usd": token_cost(early_input, early_output),
                    "batch_summed_provider_latency_seconds": batch_latency,
                    "all_separate_summed_provider_latency_seconds": all_latency,
                    "early_separate_summed_provider_latency_seconds": early_latency,
                    "request_reduction_vs_all": all_requests - batch_requests,
                    "input_token_reduction_vs_all": all_input - batch_input,
                    "saving_vs_all_usd": token_cost(all_input - batch_input, all_output - batch_output),
                    "saving_vs_early_usd": token_cost(early_input - batch_input, early_output - batch_output),
                    "batch_mean_input_tokens_per_request": batch_input / batch_requests,
                    "batch_mean_output_tokens_per_request": batch_output / batch_requests,
                    "batch_input_tokens_per_named_question": batch_input / (batch_requests * size),
                    "batch_cost_per_named_question_usd": token_cost(batch_input, batch_output) / (batch_requests * size),
                    "population_interpretation_permitted": sample_kind == "representative",
                }
            )
    representative = [row for row in output if row["sample_kind"] == "representative"]
    for row in representative:
        previous = next(
            (candidate for candidate in representative if candidate["batch_size"] == row["batch_size"] - 1),
            None,
        )
        row["marginal_mean_input_tokens_from_previous_batch_size"] = (
            row["batch_mean_input_tokens_per_request"]
            - previous["batch_mean_input_tokens_per_request"]
            if previous
            else None
        )
    return output


def _available_operator_groups(
    rows: Sequence[Mapping[str, Any]], dataset: str, workload_predicates: Sequence[str]
) -> dict[tuple[str, tuple[str, ...]], list[Mapping[str, Any]]]:
    allowed = set(workload_predicates)
    groups: dict[tuple[str, tuple[str, ...]], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        predicates = tuple(row["predicate_ids"])
        if (
            row["dataset"] == dataset
            and row["sample_kind"] == "representative"
            and set(predicates).issubset(allowed)
            and (row["execution_mode"] != "conjunctive_binary" or set(predicates) == allowed)
        ):
            groups[(row["execution_mode"], predicates)].append(row)
    return groups


def enumerate_measured_plans(
    rows: Sequence[Mapping[str, Any]], dataset: str, workload_predicates: Sequence[str]
) -> list[tuple[tuple[str, tuple[str, ...]], ...]]:
    available = _available_operator_groups(rows, dataset, workload_predicates)
    target = set(workload_predicates)
    group_keys = [
        key
        for key in available
        if key[0] in {"single", "multi_decision", "conjunctive_binary"}
    ]
    covers: set[tuple[tuple[str, tuple[str, ...]], ...]] = set()
    for size in range(1, len(workload_predicates) + 1):
        for chosen in itertools.combinations(group_keys, size):
            flattened = [predicate for _, predicates in chosen for predicate in predicates]
            if len(flattened) != len(set(flattened)) or set(flattened) != target:
                continue
            for ordered in itertools.permutations(chosen):
                covers.add(tuple(ordered))
    return sorted(covers, key=str)


def _group_trace_lookup(
    rows: Sequence[Mapping[str, Any]], dataset: str
) -> dict[tuple[str, tuple[str, ...], str], Mapping[str, Any]]:
    return {
        (row["execution_mode"], tuple(row["predicate_ids"]), row["record_id"]): row
        for row in rows
        if row["dataset"] == dataset and row["sample_kind"] == "representative"
    }


def replay_plan(
    plan: Sequence[tuple[str, tuple[str, ...]]],
    *,
    dataset: str,
    workload_id: str,
    workload_predicates: Sequence[str],
    original_order: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    lookup = _group_trace_lookup(rows, dataset)
    single_lookup = {
        (row["record_id"], row["predicate_ids"][0]): row
        for row in rows
        if row["dataset"] == dataset
        and row["sample_kind"] == "representative"
        and row["execution_mode"] == "single"
    }
    record_sets = [
        {
            record_id
            for mode, predicates, record_id in lookup
            if mode == group_mode and predicates == group_predicates
        }
        for group_mode, group_predicates in plan
    ]
    records = sorted(set.intersection(*record_sets)) if record_sets else []
    missing_reference = [
        (record_id, predicate_id)
        for record_id in records
        for predicate_id in workload_predicates
        if (record_id, predicate_id) not in single_lookup
    ]
    stage_inputs = [0] * len(plan)
    stage_passes = [0] * len(plan)
    requests = input_tokens = output_tokens = 0
    latency = 0.0
    reference: list[bool] = []
    candidate: list[bool] = []
    per_record_cost: list[float] = []
    if missing_reference:
        records = []
    for record_id in records:
        reference.append(
            all(
                _single_decision(single_lookup[(record_id, predicate_id)], predicate_id)
                for predicate_id in workload_predicates
            )
        )
        passed = True
        record_cost = 0.0
        for stage, (mode, predicates) in enumerate(plan):
            if not passed:
                break
            trace = lookup.get((mode, predicates, record_id))
            if trace is None:
                raise AssertionError("enumerated plan lacks a trace")
            stage_inputs[stage] += 1
            passed = bool(trace["group_conjunction_decision"])
            stage_passes[stage] += int(passed)
            requests += 1
            input_tokens += int(trace["input_tokens"])
            output_tokens += int(trace["output_tokens"])
            latency += float(trace["provider_latency_seconds"])
            record_cost += float(trace["estimated_cost_usd"])
        candidate.append(passed)
        per_record_cost.append(record_cost)
    evidence = "fully_measured" if len(records) == 30 and not missing_reference else "not_evaluable"
    metrics = binary_metrics(reference, candidate) if evidence == "fully_measured" else None

    predicted_reach = 1.0
    predicted_requests = 0.0
    predicted_cost = 0.0
    predicted_stage_inputs = []
    group_statistics = []
    for mode, predicates in plan:
        group_rows = [
            row
            for row in rows
            if row["dataset"] == dataset
            and row["sample_kind"] == "representative"
            and row["execution_mode"] == mode
            and tuple(row["predicate_ids"]) == predicates
        ]
        mean_cost = statistics.mean(float(row["estimated_cost_usd"]) for row in group_rows)
        selectivity = statistics.mean(bool(row["group_conjunction_decision"]) for row in group_rows)
        predicted_input = len(records) * predicted_reach
        predicted_stage_inputs.append(predicted_input)
        predicted_requests += predicted_input
        predicted_cost += predicted_input * mean_cost
        group_statistics.append(
            {
                "mode": mode,
                "predicate_ids": list(predicates),
                "sample_count": len(group_rows),
                "mean_cost_usd": mean_cost,
                "empirical_selectivity": selectivity,
            }
        )
        predicted_reach *= selectivity
    plan_payload = [
        {"mode": mode, "predicate_ids": list(predicates)} for mode, predicates in plan
    ]
    return {
        "plan_id": f"analysis_plan_{sha256_json({'workload': workload_id, 'plan': plan_payload})[:20]}",
        "dataset": dataset,
        "workload_id": workload_id,
        "predicate_ids": list(workload_predicates),
        "plan": plan_payload,
        "plan_label": " -> ".join(
            f"{mode}[{','.join(predicates)}]" for mode, predicates in plan
        ),
        "evidence_type": evidence,
        "records": len(records),
        "contains_batch": any(mode == "multi_decision" for mode, _ in plan),
        "contains_conjunction": any(mode == "conjunctive_binary" for mode, _ in plan),
        "is_full_batch": len(plan) == 1 and plan[0][0] == "multi_decision",
        "is_full_conjunction": len(plan) == 1 and plan[0][0] == "conjunctive_binary",
        "is_pure_separate": all(mode == "single" for mode, _ in plan),
        "is_original_separate": all(mode == "single" for mode, _ in plan)
        and [predicates[0] for _, predicates in plan] == list(original_order),
        "requests": requests,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": token_cost(input_tokens, output_tokens) if records else None,
        "summed_provider_latency_seconds": latency,
        "stage_input_cardinalities": stage_inputs,
        "stage_passing_counts": stage_passes,
        "conditional_stage_selectivities": [
            ratio(passing, entering)
            for passing, entering in zip(stage_passes, stage_inputs, strict=True)
        ],
        "predicted_requests": predicted_requests,
        "predicted_cost_usd": predicted_cost,
        "predicted_stage_input_cardinalities": predicted_stage_inputs,
        "prediction_basis": "in-sample unconditional group means and recursive selectivity",
        "group_statistics": group_statistics,
        "quality": metrics,
        "per_record_cost_usd": per_record_cost,
        "missing_reference_cells": [list(item) for item in missing_reference],
    }


def build_hybrid_plan_analysis(
    rows: Sequence[Mapping[str, Any]], workloads: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    plan_rows: list[dict[str, Any]] = []
    workload_summary: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    for workload in workloads["workloads"]:
        workload_id = str(workload["id"])
        dataset = str(workload["dataset"])
        predicates = list(workload["predicate_ids"])
        original_order = list(workload["original_order"])
        plans = enumerate_measured_plans(rows, dataset, predicates)
        replays = [
            replay_plan(
                plan,
                dataset=dataset,
                workload_id=workload_id,
                workload_predicates=predicates,
                original_order=original_order,
                rows=rows,
            )
            for plan in plans
        ]
        replays = [row for row in replays if row["evidence_type"] == "fully_measured"]
        plan_rows.extend(replays)
        original = next((row for row in replays if row["is_original_separate"]), None)
        full_batch = next((row for row in replays if row["is_full_batch"]), None)
        full_conjunction = next((row for row in replays if row["is_full_conjunction"]), None)
        hybrids = [row for row in replays if row["contains_batch"] and len(row["plan"]) > 1]
        best_hybrid = min(hybrids, key=lambda row: row["cost_usd"]) if hybrids else None
        best_observed = min(replays, key=lambda row: row["cost_usd"]) if replays else None
        predicted_best = min(replays, key=lambda row: row["predicted_cost_usd"]) if replays else None
        if replays:
            errors = [row["predicted_cost_usd"] - row["cost_usd"] for row in replays]
            request_errors = [row["predicted_requests"] - row["requests"] for row in replays]
            stage_errors = [
                predicted - observed
                for row in replays
                for predicted, observed in zip(
                    row["predicted_stage_input_cardinalities"],
                    row["stage_input_cardinalities"],
                    strict=True,
                )
            ]
            comparable = concordant = 0
            for left, right in itertools.combinations(replays, 2):
                predicted_delta = left["predicted_cost_usd"] - right["predicted_cost_usd"]
                observed_delta = left["cost_usd"] - right["cost_usd"]
                if math.isclose(predicted_delta, 0.0) or math.isclose(observed_delta, 0.0):
                    continue
                comparable += 1
                concordant += int((predicted_delta > 0) == (observed_delta > 0))
            prediction = {
                "workload_id": workload_id,
                "dataset": dataset,
                "plans": len(replays),
                "cost_mae_usd": statistics.mean(abs(error) for error in errors),
                "cost_rmse_usd": math.sqrt(statistics.mean(error * error for error in errors)),
                "cost_mape": statistics.mean(
                    abs(error) / row["cost_usd"]
                    for error, row in zip(errors, replays, strict=True)
                    if row["cost_usd"]
                ),
                "cost_signed_bias_usd": statistics.mean(errors),
                "request_count_mae": statistics.mean(abs(error) for error in request_errors),
                "request_count_signed_bias": statistics.mean(request_errors),
                "stage_cardinality_mae": statistics.mean(abs(error) for error in stage_errors),
                "stage_cardinality_signed_bias": statistics.mean(stage_errors),
                "pairwise_ranking_concordance": ratio(concordant, comparable),
                "pairwise_comparisons": comparable,
                "predicted_best_plan_id": predicted_best["plan_id"],
                "observed_best_plan_id": best_observed["plan_id"],
                "selection_regret_usd": predicted_best["cost_usd"] - best_observed["cost_usd"],
                "selected_observed_best": math.isclose(
                    predicted_best["cost_usd"], best_observed["cost_usd"], abs_tol=1e-15
                ),
                "evaluation_scope": "in-sample diagnostic among fully measured compatible plans",
            }
            prediction_rows.append(prediction)
        else:
            prediction = {
                "workload_id": workload_id,
                "dataset": dataset,
                "status": "not_evaluable",
                "reason": "Stage B omitted one or more workload predicates or its full batch group.",
            }
        workload_summary[workload_id] = {
            "dataset": dataset,
            "predicate_ids": predicates,
            "measured_plan_count": len(replays),
            "status": "fully_measured_plan_matrix" if replays else "not_evaluable",
            "original_separate": original,
            "full_batch": full_batch,
            "full_conjunction": full_conjunction,
            "best_supported_hybrid": best_hybrid,
            "best_observed_measured_plan": best_observed,
            "cost_only_in_sample_selection": predicted_best,
            "prediction_evaluation": prediction,
        }
    return plan_rows, workload_summary, prediction_rows


def quality_status(metrics: Mapping[str, Any] | None, gate: Mapping[str, Any], *, representative: bool) -> dict[str, Any]:
    if metrics is None:
        return {"status": "missing_complete_plan_evidence", "eligible": False, "reasons": ["missing_complete_plan_evidence"]}
    reasons = []
    if not representative:
        reasons.append("nonrepresentative_or_enriched_sample")
    if metrics["sample_count"] < int(gate["minimum_records"]):
        reasons.append("insufficient_total_observations")
    if metrics["reference_positive_count"] < int(gate["minimum_reference_positives"]):
        reasons.append("insufficient_positive_examples")
    if metrics["agreement"] is None or metrics["agreement"] < float(gate["minimum_agreement"]):
        reasons.append("agreement_below_threshold_or_undefined")
    if metrics["recall"] is None:
        reasons.append("undefined_recall")
    elif metrics["recall"] < float(gate["minimum_recall"]):
        reasons.append("recall_below_threshold")
    if metrics["false_negative_rate"] is None:
        reasons.append("undefined_false_negative_rate")
    elif metrics["false_negative_rate"] > float(gate["maximum_false_negative_rate"]):
        reasons.append("false_negative_rate_above_threshold")
    if metrics["changed_decision_rate"] is None or metrics["changed_decision_rate"] > float(gate["maximum_changed_decision_rate"]):
        reasons.append("changed_decision_rate_above_threshold_or_undefined")
    if "insufficient_total_observations" in reasons:
        status = "insufficient_total_observations"
    elif "insufficient_positive_examples" in reasons:
        status = "insufficient_positive_examples"
    elif "undefined_recall" in reasons or "undefined_false_negative_rate" in reasons:
        status = "undefined_metric"
    elif reasons:
        status = "sufficient_evidence_failed"
    else:
        status = "sufficient_evidence_passed"
    return {"status": status, "eligible": not reasons, "reasons": reasons}


def build_quality_gate_results(
    *,
    protocol: Mapping[str, Any],
    query_agreement: Sequence[Mapping[str, Any]],
    plan_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    gate = protocol["quality_gate"]
    candidates = []
    for row in query_agreement:
        metrics = {key: value for key, value in row.items() if key in {
            "sample_count", "reference_positive_count", "reference_negative_count",
            "candidate_positive_count", "true_positive", "true_negative", "false_positive",
            "false_negative", "agreement", "changed_decision_rate", "precision", "recall",
            "false_positive_rate", "false_negative_rate", "jaccard"
        }}
        decision = quality_status(metrics, gate, representative=row["sample_kind"] == "representative")
        candidates.append(
            {
                "candidate_id": row["comparison_id"],
                "scope": "direct_group_complete_query",
                "dataset": row["dataset"],
                "workload_id": row["workload_id"],
                "sample_kind": row["sample_kind"],
                "execution_mode": row["execution_mode"],
                "metrics": metrics,
                **decision,
            }
        )
    for row in plan_rows:
        if not (row["contains_batch"] or row["contains_conjunction"]):
            continue
        decision = quality_status(row["quality"], gate, representative=True)
        candidates.append(
            {
                "candidate_id": row["plan_id"],
                "scope": "fully_measured_plan_replay",
                "dataset": row["dataset"],
                "workload_id": row["workload_id"],
                "sample_kind": "representative",
                "execution_mode": "hybrid" if len(row["plan"]) > 1 else row["plan"][0]["mode"],
                "metrics": row["quality"],
                **decision,
            }
        )
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "preregistered_gate": gate,
        "gate_relaxed": False,
        "candidates": candidates,
        "summary": {
            "candidate_evaluations": len(candidates),
            "eligible": sum(bool(row["eligible"]) for row in candidates),
            "status_counts": {
                status: sum(row["status"] == status for row in candidates)
                for status in sorted({str(row["status"]) for row in candidates})
            },
            "batching_or_hybrid_candidate_qualified": any(
                bool(row["eligible"]) for row in candidates
            ),
            "human_ground_truth_available": False,
        },
        "enriched_combination_policy": {
            "combined_with_representative_for_gate": False,
            "reason": "Enriched development examples are not population representative and no sampling weights exist.",
        },
    }


def build_statistical_uncertainty(
    *,
    comparisons: Sequence[Mapping[str, Any]],
    predicate_agreement: Sequence[Mapping[str, Any]],
    group_agreement: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    representative_predicates = [
        row for row in predicate_agreement if row["sample_kind"] == "representative"
    ]
    by_size = []
    for size in sorted({int(row["batch_size"]) for row in representative_predicates}):
        selected = [row for row in representative_predicates if row["batch_size"] == size]
        total = sum(int(row["sample_count"]) for row in selected)
        agreements = sum(int(row["agreements"]) for row in selected)
        fp = sum(int(row["false_positive"]) for row in selected)
        fn = sum(int(row["false_negative"]) for row in selected)
        negatives = sum(int(row["reference_negative_count"]) for row in selected)
        positives = sum(int(row["reference_positive_count"]) for row in selected)
        by_size.append(
            {
                "batch_size": size,
                "named_decisions": total,
                "agreements": agreements,
                "false_positive": fp,
                "false_negative": fn,
                "agreement_interval_95": wilson_interval(agreements, total),
                "false_positive_interval_95": wilson_interval(fp, negatives),
                "false_negative_interval_95": wilson_interval(fn, positives),
                "dependence_warning": "Named decisions share physical requests and records; this pooled Wilson interval is descriptive, not a cluster-adjusted causal interval.",
            }
        )
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "cost_intervals": [
            {
                "comparison_id": row["comparison_id"],
                "batch_minus_all": row["paired_batch_minus_all_cost_mean_interval_95"],
                "batch_minus_early": row["paired_batch_minus_early_cost_mean_interval_95"],
            }
            for row in comparisons
            if row["evidence_type"] == "fully_measured_paired"
        ],
        "agreement_by_batch_size": by_size,
        "group_intervals": [
            {
                "comparison_id": row["comparison_id"],
                "agreement": row["agreement_interval_95"],
                "changed": row["changed_interval_95"],
                "false_positive": row["false_positive_interval_95"],
                "false_negative": row["false_negative_interval_95"],
                "recall": row["recall_interval_95"],
            }
            for row in group_agreement
        ],
        "statistical_units": {
            "cost": "record within one paired operator comparison",
            "agreement": "record-predicate within group, with dependence disclosed",
            "workload_heterogeneity": "reported separately by workload and dataset",
            "requests_treated_as_independent_experiments": False,
        },
        "provider_variability": {
            "identical_request_replicates": 0,
            "stability_established": False,
            "reason": "Each provider-visible request configuration and record was executed once.",
        },
    }


def _group_acquisition_costs(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, tuple[str, ...]], float]:
    grouped: dict[tuple[str, str, tuple[str, ...]], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["sample_kind"] == "representative":
            grouped[(row["dataset"], row["execution_mode"], tuple(row["predicate_ids"]))].append(row)
    return {key: _sum_fields(values)["cost_usd"] for key, values in grouped.items()}


def build_profiling_economics(
    *,
    rows: Sequence[Mapping[str, Any]],
    workloads: Mapping[str, Any],
    workload_summary: Mapping[str, Any],
    horizons: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    acquisition = _group_acquisition_costs(rows)
    shared_seen: dict[str, set[tuple[str, str, tuple[str, ...]]]] = defaultdict(set)
    output: list[dict[str, Any]] = []
    assumptions: dict[str, Any] = {}
    for workload in workloads["workloads"]:
        workload_id = str(workload["id"])
        dataset = str(workload["dataset"])
        summary = workload_summary[workload_id]
        baseline = summary["original_separate"]
        measured = [
            candidate
            for key in ("full_batch", "best_supported_hybrid", "full_conjunction")
            if (candidate := summary.get(key)) is not None
        ]
        if baseline is None or not measured:
            assumptions[workload_id] = {
                "status": "not_evaluable",
                "reason": "No compatible full batch or hybrid matrix was measured for this workload.",
            }
            continue
        all_candidate_plans = [
            candidate
            for candidate in (
                summary.get("full_batch"),
                summary.get("best_supported_hybrid"),
                summary.get("full_conjunction"),
                summary.get("cost_only_in_sample_selection"),
            )
            if candidate is not None
        ]
        strategies = {
            "fixed_separate": baseline,
            "full_multi_decision_batch": summary.get("full_batch"),
            "selective_first_hybrid": summary.get("best_supported_hybrid"),
            "full_conjunctive_fusion": summary.get("full_conjunction"),
            "cost_only_hybrid": summary.get("cost_only_in_sample_selection"),
            "quality_constrained_hybrid": baseline,
            "profiling_aware_hybrid": baseline,
        }
        required_groups = {
            key
            for key in acquisition
            if key[0] == dataset
            and set(key[2]).issubset(set(workload["predicate_ids"]))
            and key[1] in {"single", "multi_decision"}
        }
        standalone_cost = sum(acquisition[key] for key in required_groups)
        new_shared = required_groups - shared_seen[dataset]
        shared_cost = sum(acquisition[key] for key in new_shared)
        shared_seen[dataset].update(required_groups)
        assumptions[workload_id] = {
            "status": "exploratory",
            "standalone_profiled_groups": [str(key) for key in sorted(required_groups, key=str)],
            "standalone_profiling_cost_usd": standalone_cost,
            "shared_incremental_groups": [str(key) for key in sorted(new_shared, key=str)],
            "shared_incremental_cost_usd": shared_cost,
            "actual_experiment_expenditure_is_separate": True,
            "candidate_count": len(all_candidate_plans),
        }
        for strategy, candidate in strategies.items():
            if candidate is None:
                continue
            saving = float(baseline["cost_usd"]) - float(candidate["cost_usd"])
            admissible = strategy in {"fixed_separate", "quality_constrained_hybrid", "profiling_aware_hybrid"}
            profile_cost = 0.0 if strategy == "fixed_separate" else standalone_cost
            shared_profile = 0.0 if strategy == "fixed_separate" else shared_cost
            break_even_standalone = math.ceil(profile_cost / saving) if saving > 0 else None
            break_even_shared = math.ceil(shared_profile / saving) if saving > 0 else None
            for horizon in horizons:
                gross = horizon * saving
                for scenario, acquisition_cost in (
                    ("standalone_workload", profile_cost),
                    ("shared_statistics", shared_profile),
                    ("warm_start", 0.0),
                ):
                    output.append(
                        {
                            "dataset": dataset,
                            "workload_id": workload_id,
                            "strategy": strategy,
                            "quality_admissible": admissible,
                            "economic_result_kind": "exploratory" if not admissible else "admissible_fallback_or_baseline",
                            "scenario": scenario,
                            "reuse_horizon": horizon,
                            "baseline_cost_usd": baseline["cost_usd"],
                            "candidate_cost_usd": candidate["cost_usd"],
                            "execution_saving_per_run_usd": saving,
                            "incremental_profiling_cost_usd": acquisition_cost,
                            "optimization_overhead_usd": 0.0,
                            "net_benefit_usd": gross - acquisition_cost,
                            "break_even_reuse_count": (
                                break_even_standalone
                                if scenario == "standalone_workload"
                                else break_even_shared if scenario == "shared_statistics" else (1 if saving > 0 else None)
                            ),
                        }
                    )
    actual = _sum_fields(rows)
    return output, {
        "actual_experiment_expenditure": {
            **actual,
            "charged_once": True,
            "included_in_counterfactual_scenarios": False,
        },
        "workload_assumptions": assumptions,
        "shared_allocation_order": [str(item["id"]) for item in workloads["workloads"]],
        "shared_costs_double_counted": False,
    }


def build_strategy_comparison(
    workload_summary: Mapping[str, Any], profiling_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    strategies = [
        "fixed_separate",
        "existing_adaptive",
        "full_conjunctive_fusion",
        "full_multi_decision_batch",
        "selective_first_hybrid",
        "cost_only_hybrid",
        "quality_constrained_hybrid",
        "profiling_aware_hybrid",
    ]
    recommendations = {
        "fixed_separate": "Retain as measured reference and fallback.",
        "existing_adaptive": "Retain as production baseline; Stage B did not persist a new adaptive selection.",
        "full_conjunctive_fusion": "Retain only as a negative-control experimental candidate.",
        "full_multi_decision_batch": "Retain as an experimental candidate pending representative selection evidence.",
        "selective_first_hybrid": "Retain as an experimental candidate for measured compatible workloads.",
        "cost_only_hybrid": "Retain as an ablation; in-sample selection is not quality admissible.",
        "quality_constrained_hybrid": "Retain; currently falls back because no candidate passes the gate.",
        "profiling_aware_hybrid": "Retain; currently falls back and adaptive trajectory is not evaluable from saved traces.",
    }
    entries = []
    for strategy in strategies:
        costs = []
        supported = 0
        for summary in workload_summary.values():
            key = {
                "fixed_separate": "original_separate",
                "full_conjunctive_fusion": "full_conjunction",
                "full_multi_decision_batch": "full_batch",
                "selective_first_hybrid": "best_supported_hybrid",
                "cost_only_hybrid": "cost_only_in_sample_selection",
                "quality_constrained_hybrid": "original_separate",
                "profiling_aware_hybrid": "original_separate",
            }.get(strategy)
            candidate = summary.get(key) if key else None
            if candidate is not None:
                costs.append(float(candidate["cost_usd"]))
                supported += 1
        profile_costs = [
            float(row["incremental_profiling_cost_usd"])
            for row in profiling_rows
            if row["strategy"] == strategy
            and row["scenario"] == "standalone_workload"
            and row["reuse_horizon"] == 1
        ]
        entries.append(
            {
                "strategy": strategy,
                "measured_or_replayed_workloads": supported,
                "aggregate_execution_cost_usd": sum(costs) if costs else None,
                "mean_execution_cost_usd": statistics.mean(costs) if costs else None,
                "mean_standalone_profiling_cost_usd": statistics.mean(profile_costs) if profile_costs else None,
                "quality_admissible_batching_candidates": 0,
                "recommendation": recommendations[strategy],
                "production_default_changed": False,
            }
        )
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "strategies": entries,
        "ranking_scope": "Measured/replayed compatible Stage B workloads only; missing strategies are not assigned zero cost.",
        "primary_baseline": "existing_adaptive",
        "production_default_changed": False,
    }


def build_stage_d_readiness(
    *,
    integrity: Mapping[str, Any],
    quality: Mapping[str, Any],
) -> dict[str, Any]:
    selection_proposal = {
        "status": "proposal_only_no_calls_authorized",
        "partition": "selection",
        "records_per_dataset": 100,
        "separate_reference": {
            "unique_predicates_per_dataset": 6,
            "datasets": 3,
            "physical_requests": 1800,
        },
        "full_batch": {"workloads": 9, "physical_requests": 900},
        "one_selective_first_hybrid_remainder_group_for_each_3_or_4_predicate_workload": {
            "workloads": 6,
            "physical_requests": 600,
        },
        "logical_request_ceiling": 3300,
        "retry_inclusive_http_attempt_ceiling": 9900,
        "concurrency": 4,
        "max_retries": 2,
        "model": "jev-1.14.0",
        "cache": "disabled",
        "requires_new_exact_authorization": True,
        "selection_schedule_must_be_frozen_before_authorization": True,
        "conjunctive_negative_control_requests": 0,
        "rationale": "Acquire population-representative selection evidence for all nine workloads, including the three omitted by Stage B, while measuring one bounded selective-first candidate for each workload with at least three predicates.",
    }
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "conclusion": "READY_FOR_SELECTION_STAGE_AUTHORIZATION",
        "stage_d_ready": False,
        "experimental_validity": {
            "held_out_ids_untouched": integrity["partition_integrity"]["held_out_record_ids_observed"] == 0,
            "configuration_fingerprints_valid": all(integrity["configuration"]["match_authorized_manifest"].values()),
            "candidate_selection_used_only_stage_b": True,
            "leakage_free_adaptive_trajectory_available": False,
            "paired_selection_design_possible": True,
        },
        "statistical_readiness": {
            "quality_eligible_batching_or_hybrid_candidates": quality["summary"]["eligible"],
            "enough_stage_b_records_for_gate": False,
            "all_nine_workloads_measured": False,
            "reason": "Representative Stage B groups have 30 records versus the preregistered minimum of 40, and three declared workloads lack compatible full batch measurements.",
        },
        "economic_readiness": {
            "exploratory_rationale_available": True,
            "quality_admissible_savings_available": False,
            "selection_request_ceiling_defined": True,
        },
        "selection_stage_integrity": {
            "selection_partition_inference_present": False,
            "pilot_promoted_to_confirmatory_plan_freeze": False,
            "additional_selection_required_by_protocol": True,
        },
        "selection_stage_request_proposal": selection_proposal,
        "stage_d_authorization_requested": False,
        "live_jev_calls": 0,
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if value is None:
        return ""
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    if fields is None:
        fields = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", "utf-8")


def _flatten_normalized(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in row.items() if not key.endswith("_path") or key in {"trace_path", "operator_path"}} for row in rows]


def _comparison_csv_rows(comparisons: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in comparisons:
        output.append(
            {
                "comparison_id": row["comparison_id"],
                "dataset": row["dataset"],
                "sample_kind": row["sample_kind"],
                "workload_id": row["workload_id"],
                "execution_mode": row["execution_mode"],
                "group_id": row["group_id"],
                "batch_size": row["batch_size"],
                "predicate_ids": row["predicate_ids"],
                "records": row["records"],
                "evidence_type": row["evidence_type"],
                "batch_requests": row["batch"]["requests"],
                "batch_input_tokens": row["batch"]["input_tokens"],
                "batch_output_tokens": row["batch"]["output_tokens"],
                "batch_cost_usd": row["batch"]["cost_usd"],
                "all_separate_requests": row["all_questions_separate"]["requests"] if row["all_questions_separate"] else None,
                "all_separate_input_tokens": row["all_questions_separate"]["input_tokens"] if row["all_questions_separate"] else None,
                "all_separate_cost_usd": row["all_questions_separate"]["cost_usd"] if row["all_questions_separate"] else None,
                "early_separate_requests": row["early_reject_separate"]["requests"] if row["early_reject_separate"] else None,
                "early_separate_input_tokens": row["early_reject_separate"]["input_tokens"] if row["early_reject_separate"] else None,
                "early_separate_cost_usd": row["early_reject_separate"]["cost_usd"] if row["early_reject_separate"] else None,
                "saving_vs_all_usd": row["batch_saving_vs_all_separate_usd"],
                "saving_vs_early_usd": row["batch_saving_vs_early_separate_usd"],
                "summed_batch_latency_seconds": row["batch"]["summed_provider_latency_seconds"],
                "summed_all_separate_latency_seconds": row["all_questions_separate"]["summed_provider_latency_seconds"] if row["all_questions_separate"] else None,
                "summed_early_separate_latency_seconds": row["early_reject_separate"]["summed_provider_latency_seconds"] if row["early_reject_separate"] else None,
            }
        )
    return output


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "not evaluable"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _usd(value: Any) -> str:
    return "not evaluable" if value is None else f"${float(value):.9f}"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    output = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    output.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(output)


def _workload_csv_rows(workload_summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    for workload_id, summary in workload_summary.items():
        baseline = summary["original_separate"]
        batch = summary["full_batch"]
        hybrid = summary["best_supported_hybrid"]
        observed = summary["best_observed_measured_plan"]
        selected = summary["cost_only_in_sample_selection"]
        output.append(
            {
                "workload_id": workload_id,
                "dataset": summary["dataset"],
                "predicate_ids": summary["predicate_ids"],
                "status": summary["status"],
                "measured_plan_count": summary["measured_plan_count"],
                "original_separate_cost_usd": baseline["cost_usd"] if baseline else None,
                "original_separate_requests": baseline["requests"] if baseline else None,
                "full_batch_cost_usd": batch["cost_usd"] if batch else None,
                "full_batch_requests": batch["requests"] if batch else None,
                "full_batch_saving_vs_original_usd": (
                    baseline["cost_usd"] - batch["cost_usd"] if baseline and batch else None
                ),
                "best_hybrid_plan": hybrid["plan_label"] if hybrid else None,
                "best_hybrid_cost_usd": hybrid["cost_usd"] if hybrid else None,
                "best_hybrid_saving_vs_original_usd": (
                    baseline["cost_usd"] - hybrid["cost_usd"] if baseline and hybrid else None
                ),
                "best_observed_plan": observed["plan_label"] if observed else None,
                "best_observed_cost_usd": observed["cost_usd"] if observed else None,
                "cost_only_selected_plan": selected["plan_label"] if selected else None,
                "cost_only_selected_cost_usd": selected["cost_usd"] if selected else None,
                "cost_only_selection_regret_usd": (
                    selected["cost_usd"] - observed["cost_usd"] if selected and observed else None
                ),
                "quality_admissible_batch_or_hybrid": False,
            }
        )
    return output


def _dataset_csv_rows(comparisons: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in comparisons:
        if row["sample_kind"] != "representative":
            continue
        output.extend(_comparison_csv_rows([row]))
    return output


def _plan_csv_rows(plan_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "plan_id", "dataset", "workload_id", "plan_label", "evidence_type", "records",
        "contains_batch", "contains_conjunction", "is_full_batch", "is_full_conjunction",
        "is_pure_separate", "is_original_separate", "requests", "input_tokens",
        "output_tokens", "cost_usd", "summed_provider_latency_seconds", "stage_input_cardinalities",
        "stage_passing_counts", "conditional_stage_selectivities", "predicted_requests",
        "predicted_cost_usd", "predicted_stage_input_cardinalities", "prediction_basis", "quality",
        "plan",
    )
    return [{field: row.get(field) for field in fields} for row in plan_rows]


def build_report(
    *,
    integrity: Mapping[str, Any],
    reconciliation: Mapping[str, Any],
    comparisons: Sequence[Mapping[str, Any]],
    batch_size: Sequence[Mapping[str, Any]],
    predicate_agreement: Sequence[Mapping[str, Any]],
    group_agreement: Sequence[Mapping[str, Any]],
    query_agreement: Sequence[Mapping[str, Any]],
    workload_summary: Mapping[str, Any],
    prediction_rows: Sequence[Mapping[str, Any]],
    quality: Mapping[str, Any],
    uncertainty: Mapping[str, Any],
    profiling_rows: Sequence[Mapping[str, Any]],
    profiling_accounting: Mapping[str, Any],
    strategy: Mapping[str, Any],
    readiness: Mapping[str, Any],
) -> str:
    representative_sizes = [row for row in batch_size if row["sample_kind"] == "representative"]
    agreement_sizes = uncertainty["agreement_by_batch_size"]
    workload_rows = _workload_csv_rows(workload_summary)
    supported = [row for row in workload_rows if row["status"] != "not_evaluable"]
    full_batch_wins = sum(
        row["full_batch_saving_vs_original_usd"] is not None
        and row["full_batch_saving_vs_original_usd"] > 0
        for row in supported
    )
    best_hybrid_wins = sum(
        row["best_hybrid_saving_vs_original_usd"] is not None
        and row["best_hybrid_saving_vs_original_usd"] > 0
        for row in supported
    )
    total_predicate = [row for row in predicate_agreement if row["sample_kind"] == "representative"]
    total_named = sum(row["sample_count"] for row in total_predicate)
    total_agree = sum(row["agreements"] for row in total_predicate)
    total_fp = sum(row["false_positive"] for row in total_predicate)
    total_fn = sum(row["false_negative"] for row in total_predicate)
    total_crossings = [
        crossing
        for row in total_predicate
        for crossing in row["threshold_crossings"]
    ]
    near_threshold = sum(row["near_threshold_disagreements"] for row in total_predicate)
    high_confidence = sum(row["high_confidence_disagreements"] for row in total_predicate)
    representative_groups = [row for row in group_agreement if row["sample_kind"] == "representative"]
    exact_query_groups = sum(row["changed_decisions"] == 0 for row in representative_groups)

    batch_economics_table = markdown_table(
        ["Batch size", "Records", "Batch requests", "All-separate requests", "Batch cost", "All-separate cost", "Saving vs all", "Early-separate cost", "Saving vs early", "Batch latency (sum s)", "Early latency (sum s)"],
        [
            [
                row["batch_size"], row["records"], row["batch_requests"], row["all_separate_requests"],
                _usd(row["batch_cost_usd"]), _usd(row["all_separate_cost_usd"]),
                _usd(row["saving_vs_all_usd"]), _usd(row["early_separate_cost_usd"]),
                _usd(row["saving_vs_early_usd"]),
                _fmt(row["batch_summed_provider_latency_seconds"], 3),
                _fmt(row["early_separate_summed_provider_latency_seconds"], 3),
            ]
            for row in representative_sizes
        ],
    )
    dataset_economics_table = markdown_table(
        ["Dataset", "Size", "Batch cost", "All-separate cost", "Early cost", "Batch saving vs early"],
        [
            [
                row["dataset"], row["batch_size"], _usd(row["batch"]["cost_usd"]),
                _usd(row["all_questions_separate"]["cost_usd"]),
                _usd(row["early_reject_separate"]["cost_usd"]),
                _usd(row["batch_saving_vs_early_separate_usd"]),
            ]
            for row in comparisons
            if row["sample_kind"] == "representative"
            and row["execution_mode"] == "multi_decision"
        ],
    )
    agreement_table = markdown_table(
        ["Batch size", "Named decisions", "Agreement", "95% CI", "FP", "FN", "FNR"],
        [
            [
                row["batch_size"], row["named_decisions"],
                f"{row['agreement_interval_95']['estimate']:.2%}",
                f"[{row['agreement_interval_95']['lower']:.2%}, {row['agreement_interval_95']['upper']:.2%}]",
                row["false_positive"], row["false_negative"],
                _fmt(row["false_negative_interval_95"]["estimate"], 4),
            ]
            for row in agreement_sizes
        ],
    )
    group_semantic_table = markdown_table(
        ["Dataset", "Size", "Reference +", "Agreement", "FP", "FN", "Query Jaccard", "Exact set"],
        [
            [
                row["dataset"], row["batch_size"], row["reference_positive_count"],
                f"{row['agreement']:.2%}", row["false_positive"], row["false_negative"],
                _fmt(row["jaccard"], 3), row["match_set_equal"],
            ]
            for row in query_agreement
            if row["sample_kind"] == "representative"
            and row["execution_mode"] == "multi_decision"
        ],
    )
    disagreement_table = markdown_table(
        ["Dataset", "Size", "Predicate", "Separate p", "Batch p", "Record"],
        [
            [
                row["dataset"], row["batch_size"], row["predicate_id"],
                _fmt(crossing["separate_probability"], 2),
                _fmt(crossing["batch_probability"], 2), crossing["record_id"],
            ]
            for row in total_predicate
            for crossing in row["threshold_crossings"]
        ],
    )
    enriched_table = markdown_table(
        ["Dataset", "Records", "Reference +", "Agreement", "FP", "FN", "Population estimate"],
        [
            [
                row["dataset"], row["sample_count"], row["reference_positive_count"],
                f"{row['agreement']:.2%}", row["false_positive"], row["false_negative"], "no",
            ]
            for row in query_agreement
            if row["sample_kind"] == "enriched"
            and row["execution_mode"] == "multi_decision"
        ],
    )
    hybrid_table = markdown_table(
        ["Workload", "Status", "Original separate", "Full batch", "Full-batch delta", "Best measured hybrid", "Hybrid delta"],
        [
            [
                row["workload_id"], row["status"], _usd(row["original_separate_cost_usd"]),
                _usd(row["full_batch_cost_usd"]), _usd(row["full_batch_saving_vs_original_usd"]),
                _usd(row["best_hybrid_cost_usd"]), _usd(row["best_hybrid_saving_vs_original_usd"]),
            ]
            for row in workload_rows
        ],
    )
    prediction_table = markdown_table(
        ["Workload", "Plans", "Cost MAE", "MAPE", "Rank concordance", "Regret", "Selected best"],
        [
            [
                row["workload_id"], row["plans"], _usd(row["cost_mae_usd"]),
                f"{row['cost_mape']:.2%}", _fmt(row["pairwise_ranking_concordance"], 3),
                _usd(row["selection_regret_usd"]), row["selected_observed_best"],
            ]
            for row in prediction_rows
        ],
    )
    profiling_best = []
    for workload_id in workload_summary:
        candidates = [
            row for row in profiling_rows
            if row["workload_id"] == workload_id
            and row["scenario"] == "standalone_workload"
            and row["reuse_horizon"] == 1
            and row["strategy"] in {"full_multi_decision_batch", "selective_first_hybrid"}
        ]
        if candidates:
            candidate = max(candidates, key=lambda row: row["execution_saving_per_run_usd"])
            profiling_best.append(candidate)
    profiling_table = markdown_table(
        ["Workload", "Exploratory strategy", "Saving/run", "Standalone profile", "Break-even", "Quality admissible"],
        [
            [
                row["workload_id"], row["strategy"], _usd(row["execution_saving_per_run_usd"]),
                _usd(row["incremental_profiling_cost_usd"]), row["break_even_reuse_count"],
                row["quality_admissible"],
            ]
            for row in profiling_best
        ],
    )
    strategy_table = markdown_table(
        ["Strategy", "Supported workloads", "Aggregate cost", "Quality-admissible batching", "Recommendation"],
        [
            [
                row["strategy"], row["measured_or_replayed_workloads"],
                _usd(row["aggregate_execution_cost_usd"]), row["quality_admissible_batching_candidates"],
                row["recommendation"],
            ]
            for row in strategy["strategies"]
        ],
    )

    return f"""# Milestone 5.1 Stage C detailed analysis

## 1. Executive summary

The authorized Stage B run is internally complete: **810 physical requests**, **810 HTTP attempts**, **1,380 named-question evaluations**, 299,406 input tokens, 37,690 output tokens, and **$0.012575052** under the frozen $0.042/M-input, $0-output experimental pricing assumption. All 33 operators completed, all traces returned HTTP 200, and no retries or provider errors were recorded.

Multi-decision batching clearly reduced requests and input tokens relative to evaluating every question separately. On the representative sample it was cheaper than all-questions separate execution at every batch size. The realistic early-reject comparison is mixed: full batching won on {full_batch_wins}/{len(supported)} fully measurable declared workloads, while a measured selective-first hybrid beat the original-order separate replay on {best_hybrid_wins}/{sum(row['best_hybrid_cost_usd'] is not None for row in supported)} workloads with a compatible hybrid.

Semantic consistency is less conclusive. Across the representative multi-decision comparisons there were {total_agree}/{total_named} matching named decisions ({total_agree / total_named:.2%}), {total_fp} false positives and {total_fn} false negatives relative to separate Jev calls. These are consistency errors, not errors against human truth. Only {exact_query_groups}/{len(representative_groups)} representative batch/conjunction groups preserved the complete AND-query result set exactly. No batching or hybrid candidate passes the preregistered quality gate because every representative group has 30 records, below the minimum 40, and three of nine declared workloads lack compatible complete batch measurements.

The appropriate conclusion is **{readiness['conclusion']}**. Stage D is not ready. The protocol reserves a 100-record selection partition that has not received inference. A bounded 3,300-request selection proposal is documented, but no request is authorized or executed here.

## 2. Experimental setup

The run used `jev-1.14.0`, threshold 0.5, cache disabled, concurrency four, and at most two retries. The frozen schedule selected 30 pilot records per dataset for four single predicates, nested two/three/four-question batches, and a four-condition conjunctive comparator. Ten development records per dataset were selected by deterministic lexical enrichment for two singles plus one two-question batch. The datasets were Amazon All Beauty, Banking77, and UCI SMS Spam.

The executed composition reconciles exactly: 360 representative single requests, 270 representative multi-decision requests, 90 representative conjunctive requests, and 90 enriched diagnostic requests. No selection or evaluation ID appears in a trace. Partition checks used frozen manifest IDs; the analysis did not read evaluation text or outcomes.

An important coverage limitation is structural: the nested schedule does not equal all nine declared workloads. The Beauty two- and three-predicate workloads and the Banking three-predicate workload include predicates that were not measured as Stage B singles/batches. Those workloads are reported as not evaluable rather than assigned modeled zeroes.

## 3. Provider usage and cost reconciliation

The 810 JSONL rows reconcile to 810 unique `(dataset, record, operator, sample)` keys and to `session.json`. Their input tokens sum to 299,406; multiplying by $0.042/1,000,000 gives exactly $0.012575052. Output tokens sum to 37,690 and are reported separately. The frozen experimental rate assigns them $0, but the artifacts contain no invoice or billing contract; therefore this report calls $0.012575052 a token-priced experimental cost, not a verified provider bill.

The persisted operator metadata requests `jev-1.14.0`; every trace resolves to `jev-1.14.0`. Prompt hashes rebuild exactly from the frozen questions, and each trace hash equals its operator hash. Stage B did not persist the `FusionPhysicalPlan` object or its plan hash, so physical plan hashes cannot be independently verified. This is a provenance defect, not a missing request result.

Summed request latency is available; end-to-end wall time per operator is not. Concurrency makes timestamp-span reconstruction unsuitable as a substitute for persisted wall time.

## 4. Multi-decision batching performance

{batch_economics_table}

Per-dataset paired results show the heterogeneity hidden by the size aggregate:

{dataset_economics_table}

Scenario A is fully paired: the same predicates, records, and model are compared against all corresponding single calls. Scenario B replays the same single traces in the batch predicate order and stops after the first rejection. It is a saved-trace replay, not an additional call. Batching's request savings are large by construction, but dollar savings follow only when the shared record/prompt overhead outweighs lost early rejection.

The stored data support association, not provider-internal causality. Input savings are consistent with sharing record context and request framing, but provider-side tokenization components are not separately observable.

## 5. Effects of batch size

{agreement_table}

Complete-query consistency for the representative multi-decision groups was:

{group_semantic_table}

Average input tokens per request rise with batch size, while input tokens per named question fall. The marginal input-token field in `batch_size_economics.csv` compares adjacent aggregate sizes; it is descriptive because the added predicates differ by dataset. Larger-batch disagreement rates also differ, but batch size is confounded with predicate set and workload. The pilot cannot establish that size itself caused the differences.

## 6. Hybrid execution performance

{hybrid_table}

Every listed cost is exact replay from saved traces on the same 30 pilot records. Plans are enumerated only when every stage group was actually measured. A record enters the next stage only when all decisions in the current group pass. `hybrid_plan_costs.csv` records stage entrants, conditional selectivity, avoided downstream calls, tokens, and cost.

The best measured hybrid is not automatically an admissible optimizer choice: its search and evaluation use the same 30 records, its complete-plan evidence remains below the quality minimum, and unmeasured groupings are excluded. Zero reach never certifies an unmeasured group.

## 7. Semantic disagreement analysis

The representative multi-decision matrix contains {total_named} comparable named decisions and {len(total_crossings)} threshold crossings. All {near_threshold} disagreements had at least one probability within 0.10 of the 0.5 threshold, and {high_confidence} had both probabilities at least 0.25 away. The observed changes are therefore consistent with threshold proximity, although three cases cannot establish a general mechanism.

{disagreement_table}

Per-predicate probability deltas, crossing records, false positives, and false negatives are in `predicate_agreement.csv`.

Complete-query comparisons use the AND of separate decisions as reference. They report match-set equality and Jaccard overlap in `query_level_agreement.csv`. Conjunctive-binary traces expose only one group probability and are never represented as constituent predicate decisions.

Separate Jev calls are a reproducible LLM reference, not human ground truth. This analysis establishes batched/separate consistency only. With no identical-request repetitions it cannot isolate inference nondeterminism, and with no human labels it cannot establish semantic accuracy.

## 8. Representative versus enriched samples

Representative results use the first 30 sorted pilot IDs per dataset from the frozen 40-record pilot partition. Enriched development records were chosen by deterministic lexical counts (`fail/broke/return/refund/leak`, banking refund/unauthorized terms, or SMS prize/contact terms) with record ID as tie-breaker. They are diagnostic, not a probability sample.

The report never pools enriched and representative records for population agreement. Enriched rows are separate in every CSV. Sparse reference positives and undefined recall denominators are preserved in the quality artifacts; no zero-positive group is interpreted as perfect recall.

{enriched_table}

## 9. Statistical uncertainty

Cost intervals use a paired record bootstrap with 2,000 draws. Proportion intervals use 95% Wilson intervals and always retain their numerator and denominator. Pooled batch-size intervals are descriptive because decisions share records and physical requests. There is no repeated identical-request evidence for provider stability.

Heterogeneity is reported per group, dataset, and workload. Aggregate savings do not override harmful workload-level cases. Unpaired or missing groups are marked not evaluable.

## 10. Cost-estimation accuracy

{prediction_table}

These are in-sample diagnostics of the current reach-and-group-mean cost model, not saved preregistered predictions. The model uses unconditional group selectivity recursively, whereas observed replay uses correlated record-level outcomes. Stage-cardinality and request-count errors expose the resulting correlation bias. The only true ex-ante Stage A forecast was aggregate: 308,763 input tokens and $0.012968030 versus 299,406 and $0.012575052 observed (an input/cost overprediction of {(308763 - 299406) / 299406:.2%}).

## 11. Plan-selection quality

Within the fully measured candidate subset, `plan_ranking_analysis.json` reports pairwise rank concordance, predicted-best versus observed-best, and regret. It does not claim global optimality: most of the 4/20/138 enumerated theoretical plans lack measured groups. Three workloads have no compatible complete matrix.

No actual post-pilot plan selection was persisted before this analysis. “Cost-only selection” here means an explicitly labeled in-sample diagnostic. It must not be read as a frozen Stage D plan.

## 12. Quality-constrained planning

The gate remains unchanged: at least 40 records, at least three separate-reference positives, agreement at least 0.98, recall at least 0.95, FNR at most 0.05, changed rate at most 0.02, and real complete-plan evidence. Of {quality['summary']['candidate_evaluations']} direct or replayed candidate evaluations, {quality['summary']['eligible']} qualify. Quality-constrained and profiling-aware strategies therefore fall back to the existing adaptive baseline. Enriched development evidence cannot repair representative coverage without a valid weighting design.

## 13. Profiling-aware economics

{profiling_table}

The actual Stage B expenditure is charged once: {_usd(profiling_accounting['actual_experiment_expenditure']['cost_usd'])}. Standalone economics charge every compatible single and batch profile needed by one workload. Shared economics allocate an already-paid group once per dataset in declared workload order. Warm-start economics charge zero incremental profiling. Horizons 1, 5, 10, 25, 50, 100, 250, and 1,000 are in `profiling_economics.csv`.

All positive batching/hybrid economics remain exploratory because quality eligibility is zero. Local optimization overhead is recorded as $0 API cost; CPU time was not monetized. A leakage-free adaptive profiling trajectory is not reconstructable because all 30 traces are saved as a completed matrix without checkpointed controller state.

## 14. Negative and surprising results

* Request reduction did not guarantee a win against early rejection. Workloads with selective early predicates can avoid enough downstream single calls to erase batching's all-questions advantage.
* Three declared workloads are absent from the compatible matrix even though the aggregate schedule completed. Execution completeness is therefore not workload completeness.
* The quality gate rejects every candidate before Stage D. Strong aggregate agreement cannot substitute for the 40-record minimum, sufficient positives, or complete representative evidence.
* All three named-decision disagreements were near the threshold, but the sample is too small to infer that larger batches are generally stable away from the threshold.
* The cost predictor is in-sample and still shows ranking/cardinality error where stage outcomes are correlated. Independent confirmation is necessary.
* Physical operator hashes are strong, but physical plan IDs were not persisted. Future runners should persist the complete plan object and configuration fingerprint.

## 15. Research implications

Multi-decision batching is a credible economic mechanism for all-questions evaluation: it shares context and reduces tokens as well as requests. SemanticFlow's realistic problem is conditional execution, so selective-first hybrids are the more relevant candidate when an inexpensive, rejecting predicate exists. The pilot supports continued research on both, but not production adoption or a claim of general improvement.

The previous M5 conjunction result remains negative. Nothing here reinterprets its 21.77% cost increase or quality-gate failures. Multi-decision batching is a different operator because it preserves named outputs, but it has its own observed consistency changes.

## 16. Remaining limitations

There are no human labels, no identical-request replicates, no saved wall times, no persisted Stage B physical plan IDs, only 30 representative records per group, sparse positives, three unmeasured declared workloads, and no selection-partition inference. Plan search is limited to measured compatible groups. Profiling break-even assumes repeated workloads resemble the pilot and does not monetize local compute.

## 17. Next-stage recommendation

{strategy_table}

Aggregate strategy costs above are shown only with their support counts. They are directly comparable only when support sets match: fixed separate, full batch, cost-only, and the two fallback strategies cover the same six workloads; conjunction covers three and measured hybrids cover four. The table is not a cross-support global ranking.

Do **not** authorize Stage D from this evidence. The exact next stage is a separately authorized, versioned selection-stage measurement on the untouched 100-record selection partition. The proposed ceiling is 3,300 logical requests and 9,900 retry-inclusive attempts: 1,800 single references for all 18 dataset predicates, 900 full batches for all nine workloads, and 600 targeted remainder batches for one selective-first candidate on each three- or four-predicate workload. The schedule, prompts, hashes, and budget must be frozen before authorization. No paid call was made by this analysis.
"""


def generate_analysis(
    *,
    stage_b_root: str | Path = "results/milestone_5_1/live_batching_pilot_stage_b",
    output_directory: str | Path = "results/milestone_5_1/stage_c_detailed_analysis",
    report_path: str | Path = "docs/milestone_5_1_stage_c_detailed_analysis.md",
) -> dict[str, Any]:
    root = Path(stage_b_root)
    output = Path(output_directory)
    report_target = Path(report_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if report_target.exists():
        raise FileExistsError(f"refusing to overwrite {report_target}")
    output.mkdir(parents=True, exist_ok=False)

    workloads = read_yaml("configs/milestone_5_1/workloads.yaml")
    protocol = read_yaml("configs/milestone_5_1/protocol.yaml")
    schedule = read_yaml("configs/milestone_5_1/pilot_schedule.yaml")
    authorization = read_json(
        "results/milestone_5_1/stage_a_preflight/stage_b_authorization_proposal_final.json"
    )
    preflight = read_json("results/milestone_5_1/stage_a_preflight/preflight_final.json")
    rows, _ = load_stage_b_requests(root, workloads)
    integrity = build_integrity_audit(
        root=root,
        rows=rows,
        workloads=workloads,
        protocol=protocol,
        schedule=schedule,
        authorization_manifest=authorization,
        preflight=preflight,
    )
    reconciliation = build_cost_reconciliation(rows)
    comparisons, predicate_agreement, group_agreement, query_agreement = build_batch_comparisons(rows)
    batch_size = build_batch_size_economics(comparisons)
    plan_rows, workload_summary, prediction_rows = build_hybrid_plan_analysis(rows, workloads)
    quality = build_quality_gate_results(
        protocol=protocol,
        query_agreement=query_agreement,
        plan_rows=plan_rows,
    )
    uncertainty = build_statistical_uncertainty(
        comparisons=comparisons,
        predicate_agreement=predicate_agreement,
        group_agreement=group_agreement,
    )
    profiling_rows, profiling_accounting = build_profiling_economics(
        rows=rows,
        workloads=workloads,
        workload_summary=workload_summary,
        horizons=protocol["reuse_horizons"],
    )
    strategy = build_strategy_comparison(workload_summary, profiling_rows)
    readiness = build_stage_d_readiness(integrity=integrity, quality=quality)

    write_json(output / "artifact_integrity_audit.json", integrity)
    write_json(output / "request_cost_reconciliation.json", reconciliation)
    write_csv(output / "normalized_measurements.csv", _flatten_normalized(rows))
    write_csv(output / "per_dataset_comparison.csv", _dataset_csv_rows(comparisons))
    write_csv(output / "per_workload_comparison.csv", _workload_csv_rows(workload_summary))
    write_csv(output / "batch_size_economics.csv", batch_size)
    write_csv(output / "hybrid_plan_costs.csv", _plan_csv_rows(plan_rows))
    write_csv(output / "predicate_agreement.csv", predicate_agreement)
    write_csv(output / "group_agreement.csv", group_agreement)
    write_csv(output / "query_level_agreement.csv", query_agreement)
    write_json(output / "quality_gate_results.json", quality)
    write_json(output / "statistical_uncertainty.json", uncertainty)
    write_csv(output / "cost_prediction_errors.csv", prediction_rows)
    write_json(
        output / "plan_ranking_analysis.json",
        {
            "schema_version": ANALYSIS_SCHEMA,
            "workloads": workload_summary,
            "prediction_metrics": prediction_rows,
            "actual_post_pilot_optimizer_selection_persisted": False,
            "diagnostic_is_in_sample": True,
            "globally_optimal_claim_permitted": False,
        },
    )
    write_csv(output / "profiling_economics.csv", profiling_rows)
    write_json(output / "strategy_comparison.json", strategy)
    limitations = """# Experiment limitations

- No human labels exist; separate Jev output is a consistency reference, not ground truth.
- Each identical provider-visible request was run once, so repeatability is not estimable.
- Representative groups contain 30 records, below the preregistered 40-record quality minimum.
- Three declared workloads lack a compatible complete Stage B batch matrix.
- Enriched development records are not population representative and are never pooled for gating.
- Stage B persisted operator hashes but not physical plan objects or plan IDs.
- Per-request inference latency is available; end-to-end wall time is not.
- Hybrid predictions are in-sample diagnostics using unconditional group selectivity.
- Missing groups remain missing even after zero observed reach.
- No selection- or evaluation-partition inference outcome was read.
- Profiling break-even projections assume pilot economics recur and do not monetize local CPU time.
"""
    (output / "experiment_limitations.md").write_text(limitations, "utf-8")
    write_json(output / "stage_d_readiness.json", readiness)

    report = build_report(
        integrity=integrity,
        reconciliation=reconciliation,
        comparisons=comparisons,
        batch_size=batch_size,
        predicate_agreement=predicate_agreement,
        group_agreement=group_agreement,
        query_agreement=query_agreement,
        workload_summary=workload_summary,
        prediction_rows=prediction_rows,
        quality=quality,
        uncertainty=uncertainty,
        profiling_rows=profiling_rows,
        profiling_accounting=profiling_accounting,
        strategy=strategy,
        readiness=readiness,
    )
    report_target.write_text(report, "utf-8")

    artifacts = sorted(path for path in output.iterdir() if path.is_file())
    manifest = {
        "schema_version": ANALYSIS_SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "analysis_code": {
            "path": Path(__file__).as_posix(),
            "sha256": sha256_file(__file__),
        },
        "stage_b_session_sha256": sha256_file(root / "session.json"),
        "stage_b_artifact_tree_sha256": sha256_json(integrity["raw_artifacts"]["files"]),
        "configuration_sha256": integrity["configuration"]["hashes"],
        "dataset_fingerprints": {
            dataset: value["dataset_fingerprint_sha256"]
            for dataset, value in integrity["dataset_integrity"].items()
        },
        "artifacts": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in artifacts
        },
        "report": {
            "path": report_target.as_posix(),
            "sha256": sha256_file(report_target),
            "bytes": report_target.stat().st_size,
        },
        "live_jev_calls": 0,
        "provider_initialized": False,
        "held_out_outcomes_inspected": False,
        "historical_artifacts_modified": False,
    }
    write_json(output / "analysis_manifest.json", manifest)
    return {
        "status": "complete",
        "output_directory": output.as_posix(),
        "report": report_target.as_posix(),
        "physical_requests": len(rows),
        "named_question_evaluations": sum(len(row["predicate_question_names"]) for row in rows),
        "input_tokens": reconciliation["total"]["input_tokens"],
        "output_tokens": reconciliation["total"]["output_tokens"],
        "cost_usd": reconciliation["total"]["cost_usd"],
        "readiness": readiness["conclusion"],
        "live_jev_calls": 0,
    }
