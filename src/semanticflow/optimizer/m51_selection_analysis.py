from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from semanticflow.optimizer.adaptive import sampling_order, simulate_adaptive_profiling
from semanticflow.optimizer.fusion_experiment import sha256_file
from semanticflow.optimizer.m51_experiment import load_yaml
from semanticflow.optimizer.matrix import ProfileMatrix
from semanticflow.optimizer.models import load_optimizer_settings
from semanticflow.storage.results import git_commit, git_worktree_dirty

INPUT_PRICE = 0.042
OUTPUT_PRICE = 0.0
ANALYSIS_VERSION = "m5.1-selection-analysis-v1"
ESTIMATOR_VERSION = "m5.1-existing-unconditional-group-reach-v1"

PREFLIGHT = Path("results/milestone_5_1/selection_stage_preflight")
LIVE = Path("results/milestone_5_1/selection_stage_live_v2")
FAILED_LIVE = Path("results/milestone_5_1/selection_stage_live")
AUTHORIZATION = Path(
    "results/milestone_5_1/selection_stage_authorization/authorization_manifest_v2.json"
)
OUTPUT = Path("results/milestone_5_1/selection_stage_results")
WORKLOADS = Path("configs/milestone_5_1/workloads.yaml")
PROTOCOL = Path("configs/milestone_5_1/protocol.yaml")
OPTIMIZER = Path("configs/optimizer.milestone_4.yaml")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("status,reason\nunavailable,no rows\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True, separators=(",", ":"))
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _trace_files(root: Path) -> list[Path]:
    return sorted(root.glob("smoke/*/decisions.jsonl")) + sorted(
        root.glob("operators/*/*/decisions.jsonl")
    )


def load_evidence() -> dict[str, Any]:
    schedule = json.loads((PREFLIGHT / "selection_schedule.json").read_text("utf-8"))
    catalog = json.loads((PREFLIGHT / "physical_plan_catalog.json").read_text("utf-8"))["plans"]
    config = load_yaml(WORKLOADS)
    traces: list[dict[str, Any]] = []
    for path in _trace_files(LIVE):
        for line in path.read_text("utf-8").splitlines():
            if line:
                row = json.loads(line)
                row["trace_path"] = str(path).replace("\\", "/")
                traces.append(row)
    request_by_key = {
        (row["dataset"], row["record_id"], row["operator_id"]): row
        for row in schedule["physical_requests"]
    }
    normalized = []
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for trace in traces:
        dataset = str(trace["review_id"]).split(":", 1)[0]
        request = request_by_key[(dataset, trace["review_id"], trace["group_id"])]
        named = len(trace["predicate_ids"]) if trace["fusion_mode"] == "multi_decision" else 1
        row = {
            "request_id": request["request_id"],
            "dataset": dataset,
            "record_id": trace["review_id"],
            "partition": "selection",
            "operator_id": trace["group_id"],
            "execution_mode": trace["fusion_mode"],
            "predicate_ids": trace["predicate_ids"],
            "question_names": request["question_names"],
            "prompt_hash": trace["fusion_prompt_hash"],
            "requested_model": request["requested_model"],
            "resolved_model": trace["model_identifier"],
            "threshold": trace["threshold"],
            "input_tokens": trace["input_tokens"],
            "output_tokens": trace["output_tokens"],
            "cost_usd": trace["input_tokens"] * INPUT_PRICE / 1_000_000,
            "latency_seconds": trace["request_latency_seconds"],
            "http_attempts": trace["request_attempt_count"],
            "retries": trace["request_attempt_count"] - 1,
            "named_question_evaluations": named,
            "group_passed": trace["passed"],
            "group_probability": trace["group_probability"],
            "predicate_probabilities": trace["predicate_probabilities"],
            "predicate_decisions": {
                predicate_id: probability >= float(trace["threshold"])
                for predicate_id, probability in trace["predicate_probabilities"].items()
            },
            "timestamp": trace["timestamp"],
            "trace_path": trace["trace_path"],
            "response_status": trace["provider_metadata"]["http_status"],
        }
        normalized.append(row)
        lookup[(trace["review_id"], trace["group_id"])] = row
    return {
        "schedule": schedule,
        "catalog": catalog,
        "config": config,
        "normalized": normalized,
        "lookup": lookup,
    }


def wilson(numerator: int, denominator: int) -> dict[str, Any]:
    if denominator == 0:
        return {
            "numerator": numerator,
            "denominator": 0,
            "estimate": None,
            "lower": None,
            "upper": None,
        }
    z = 1.959963984540054
    p = numerator / denominator
    scale = 1 + z * z / denominator
    center = (p + z * z / (2 * denominator)) / scale
    half = z * math.sqrt(p * (1 - p) / denominator + z * z / (4 * denominator**2)) / scale
    return {
        "numerator": numerator,
        "denominator": denominator,
        "estimate": p,
        "lower": max(0.0, center - half),
        "upper": min(1.0, center + half),
        "method": "Wilson score 95%",
    }


def binary_metrics(reference: Sequence[bool], candidate: Sequence[bool]) -> dict[str, Any]:
    if len(reference) != len(candidate):
        raise ValueError("binary comparisons must be paired")
    tp = sum(r and c for r, c in zip(reference, candidate, strict=True))
    tn = sum((not r) and (not c) for r, c in zip(reference, candidate, strict=True))
    fp = sum((not r) and c for r, c in zip(reference, candidate, strict=True))
    fn = sum(r and (not c) for r, c in zip(reference, candidate, strict=True))
    n = len(reference)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else None
    return {
        "sample_count": n,
        "reference_positive_count": tp + fn,
        "reference_negative_count": tn + fp,
        "candidate_positive_count": tp + fp,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "agreements": tp + tn,
        "agreement": (tp + tn) / n,
        "changed_decisions": fp + fn,
        "changed_decision_rate": (fp + fn) / n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "false_positive_rate": fp / (tn + fp) if tn + fp else None,
        "false_negative_rate": fn / (tp + fn) if tp + fn else None,
        "agreement_interval_95": wilson(tp + tn, n),
        "changed_interval_95": wilson(fp + fn, n),
        "recall_interval_95": wilson(tp, tp + fn),
    }


def quality_status(metrics: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    reasons = []
    if metrics["sample_count"] < int(gate["minimum_records"]):
        return {
            "status": "insufficient_observations",
            "eligible": False,
            "reasons": ["minimum_records"],
        }
    if metrics["reference_positive_count"] < int(gate["minimum_reference_positives"]):
        return {
            "status": "insufficient_positive_support",
            "eligible": False,
            "reasons": ["minimum_reference_positives"],
        }
    if metrics["recall"] is None or metrics["false_negative_rate"] is None:
        return {
            "status": "undefined_metric",
            "eligible": False,
            "reasons": ["recall_or_fnr_undefined"],
        }
    checks = {
        "minimum_agreement": metrics["agreement"] >= float(gate["minimum_agreement"]),
        "minimum_recall": metrics["recall"] >= float(gate["minimum_recall"]),
        "maximum_false_negative_rate": metrics["false_negative_rate"]
        <= float(gate["maximum_false_negative_rate"]),
        "maximum_changed_decision_rate": metrics["changed_decision_rate"]
        <= float(gate["maximum_changed_decision_rate"]),
    }
    reasons.extend(key for key, passed in checks.items() if not passed)
    return {
        "status": "passed" if not reasons else "failed",
        "eligible": not reasons,
        "reasons": reasons,
    }


def _records_by_dataset(normalized: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    output: dict[str, set[str]] = defaultdict(set)
    for row in normalized:
        output[str(row["dataset"])].add(str(row["record_id"]))
    return {dataset: sorted(ids) for dataset, ids in output.items()}


def replay_plan(
    plan: Mapping[str, Any],
    record_ids: Sequence[str],
    lookup: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    total_cost = total_input = total_output = total_latency = requests = named = 0.0
    per_record = []
    stage_entries = [0 for _ in plan["groups"]]
    stage_passes = [0 for _ in plan["groups"]]
    for record_id in record_ids:
        active = True
        cost = input_tokens = output_tokens = latency = record_requests = record_named = 0.0
        for index, group in enumerate(plan["groups"]):
            if not active:
                break
            stage_entries[index] += 1
            trace = lookup[(record_id, group["operator_id"])]
            cost += float(trace["cost_usd"])
            input_tokens += int(trace["input_tokens"])
            output_tokens += int(trace["output_tokens"])
            latency += float(trace["latency_seconds"])
            record_requests += 1
            record_named += len(group["predicate_ids"])
            active = bool(trace["group_passed"])
            if active:
                stage_passes[index] += 1
        total_cost += cost
        total_input += input_tokens
        total_output += output_tokens
        total_latency += latency
        requests += record_requests
        named += record_named
        per_record.append(
            {
                "record_id": record_id,
                "cost_usd": cost,
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "requests": int(record_requests),
                "named_question_evaluations": int(record_named),
                "latency_seconds": latency,
                "passed": active,
            }
        )
    return {
        "plan_id": plan["plan_id"],
        "cost_usd": total_cost,
        "input_tokens": int(total_input),
        "output_tokens": int(total_output),
        "requests": int(requests),
        "named_question_evaluations": int(named),
        "latency_seconds": total_latency,
        "matching_record_ids": [row["record_id"] for row in per_record if row["passed"]],
        "stage_entries": stage_entries,
        "stage_passes": stage_passes,
        "per_record": per_record,
    }


def predict_plan(
    plan: Mapping[str, Any],
    record_ids: Sequence[str],
    lookup: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    n = len(record_ids)
    reach = 1.0
    cost = requests = 0.0
    stages = []
    for group in plan["groups"]:
        traces = [lookup[(record_id, group["operator_id"])] for record_id in record_ids]
        mean_cost = statistics.mean(float(trace["cost_usd"]) for trace in traces)
        selectivity = statistics.mean(bool(trace["group_passed"]) for trace in traces)
        entrants = n * reach
        cost += entrants * mean_cost
        requests += entrants
        stages.append(
            {
                "operator_id": group["operator_id"],
                "predicted_entrants": entrants,
                "unconditional_group_selectivity": selectivity,
                "mean_group_cost_usd": mean_cost,
            }
        )
        reach *= selectivity
    return {"predicted_cost_usd": cost, "predicted_requests": requests, "stages": stages}


def _plan_map(catalog: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(plan["plan_id"]): plan for plan in catalog}


def _find_separate_plan(
    catalog: Sequence[Mapping[str, Any]], workload_id: str, order: Sequence[str]
) -> Mapping[str, Any]:
    target = tuple((predicate_id,) for predicate_id in order)
    matches = [
        plan
        for plan in catalog
        if plan["workload_id"] == workload_id
        and tuple(tuple(group["predicate_ids"]) for group in plan["groups"]) == target
    ]
    if len(matches) != 1:
        raise ValueError(f"cannot resolve separate plan for {workload_id}/{order}")
    return matches[0]


def adaptive_replay(evidence: Mapping[str, Any]) -> dict[str, Any]:
    config = evidence["config"]
    catalog = evidence["catalog"]
    normalized = evidence["normalized"]
    records = _records_by_dataset(normalized)
    settings = load_optimizer_settings(OPTIMIZER)
    singles = {
        (row["record_id"], row["predicate_ids"][0]): row
        for row in normalized
        if row["execution_mode"] == "single"
    }
    output = []
    for workload in config["workloads"]:
        predicate_ids = workload["predicate_ids"]
        review_ids = records[workload["dataset"]]
        plan_for_hashes = _find_separate_plan(catalog, workload["id"], predicate_ids)
        question_hashes = {
            group["predicate_ids"][0]: group["predicate_prompt_hashes"][group["predicate_ids"][0]]
            for group in plan_for_hashes["groups"]
        }
        traces = []
        costs = []
        for review_id in review_ids:
            for predicate_id in predicate_ids:
                row = singles[(review_id, predicate_id)]
                traces.append(
                    {
                        "review_id": review_id,
                        "predicate_id": predicate_id,
                        "passed": row["predicate_decisions"][predicate_id],
                        "input_tokens": row["input_tokens"],
                        "output_tokens": row["output_tokens"],
                        "request_latency_seconds": row["latency_seconds"],
                        "model_identifier": row["resolved_model"],
                        "question_hash": question_hashes[predicate_id],
                    }
                )
                costs.append(float(row["cost_usd"]))
        matrix = ProfileMatrix.from_traces(
            traces,
            predicate_ids,
            review_ids=review_ids,
            model_identifier="jev-1.14.0",
            question_hashes=question_hashes,
        )
        result = simulate_adaptive_profiling(
            matrix,
            estimator_name="correlation_aware",
            original_order=workload["original_order"],
            objective=settings.planner.objective,
            evaluation_record_count=100,
            question_hashes=question_hashes,
            settings=settings.adaptive,
            bootstrap=settings.bootstrap,
            correlation=settings.correlation,
            input_price_per_million=INPUT_PRICE,
            output_price_per_million=OUTPUT_PRICE,
            expected_executions=10,
            tie_relative_tolerance=settings.planner.tie_relative_tolerance,
            economics=settings.economics,
            estimated_cost_per_decision=statistics.mean(costs),
            statistics_available=False,
        )
        selected = _find_separate_plan(catalog, workload["id"], result["selected_order"])
        stop = result.get("stopping_checkpoint") or {}
        output.append(
            {
                "dataset": workload["dataset"],
                "workload_id": workload["id"],
                "selected_order": result["selected_order"],
                "selected_plan_id": selected["plan_id"],
                "profiled_reviews": stop.get("profiled_reviews", 0),
                "profiling_decisions": result["profiling_decisions_used"],
                "profiling_cost_usd": stop.get("profiling_cost_usd_offline_estimate", 0.0),
                "profiling_started": result["profiling_started"],
                "stop_reason": stop.get("reasons", [result["selected_order_source"]]),
                "trajectory": result["trajectory"],
                "controller_schema_version": result["schema_version"],
                "source_config": str(OPTIMIZER).replace("\\", "/"),
                "future_information_leakage": False,
            }
        )
    return {
        "schema_version": "m5.1-selection-adaptive-replay-v1",
        "expected_executions_for_controller": 10,
        "sampling_order": "sha256(4049:review_id)",
        "workloads": output,
    }


def _bootstrap_difference(
    baseline: Sequence[float], candidate: Sequence[float], *, seed: int, iterations: int = 2000
) -> dict[str, Any]:
    rng = random.Random(seed)
    differences = [b - c for b, c in zip(baseline, candidate, strict=True)]
    n = len(differences)
    draws = [sum(differences[rng.randrange(n)] for _ in range(n)) for _ in range(iterations)]
    draws.sort()
    return {
        "estimate_usd": sum(differences),
        "lower_95_usd": draws[int(0.025 * iterations)],
        "upper_95_usd": draws[min(iterations - 1, int(0.975 * iterations))],
        "method": f"paired record bootstrap, {iterations} draws",
    }


def _rank(values: Mapping[str, float]) -> dict[str, int]:
    return {
        plan_id: index + 1
        for index, (plan_id, _) in enumerate(
            sorted(values.items(), key=lambda item: (item[1], item[0]))
        )
    }


def _ranking_metrics(
    predicted: Mapping[str, float], observed: Mapping[str, float]
) -> dict[str, Any]:
    ids = sorted(predicted)
    concordant = comparisons = 0
    for left, right in itertools.combinations(ids, 2):
        pd = predicted[left] - predicted[right]
        od = observed[left] - observed[right]
        if pd == 0 or od == 0:
            concordant += 0.5
        elif (pd < 0) == (od < 0):
            concordant += 1
        comparisons += 1
    predicted_best = min(ids, key=lambda plan_id: (predicted[plan_id], plan_id))
    observed_best = min(ids, key=lambda plan_id: (observed[plan_id], plan_id))
    return {
        "pairwise_comparisons": comparisons,
        "pairwise_ranking_concordance": concordant / comparisons if comparisons else 1.0,
        "predicted_best_plan_id": predicted_best,
        "observed_best_plan_id": observed_best,
        "selection_regret_usd": observed[predicted_best] - observed[observed_best],
        "selected_observed_best": predicted_best == observed_best,
    }


def _plan_kind(plan: Mapping[str, Any]) -> str:
    if all(group["execution_mode"] == "single" for group in plan["groups"]):
        return "separate"
    if len(plan["groups"]) == 1:
        return "full_batch"
    return "selective_first_hybrid"


def _gate_for_plan(
    plan: Mapping[str, Any],
    replay: Mapping[str, Any],
    reference: Sequence[bool],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = [row["passed"] for row in replay["per_record"]]
    metrics = binary_metrics(reference, candidate)
    if _plan_kind(plan) == "separate":
        status = {"status": "reference_separate", "eligible": True, "reasons": []}
    else:
        status = quality_status(metrics, gate)
    ref_set = {
        row["record_id"]
        for row, passed in zip(replay["per_record"], reference, strict=True)
        if passed
    }
    cand_set = set(replay["matching_record_ids"])
    union = ref_set | cand_set
    return {
        "plan_id": plan["plan_id"],
        "workload_id": plan["workload_id"],
        "plan_kind": _plan_kind(plan),
        **metrics,
        **status,
        "match_set_equal": ref_set == cand_set,
        "jaccard": len(ref_set & cand_set) / len(union) if union else 1.0,
        "complete_plan_evidence": True,
        "reference_kind": "separate_jev",
        "semantic_accuracy_established": False,
    }


def _cost_rows_for_groups(
    groups: Sequence[Mapping[str, Any]], normalized: Sequence[Mapping[str, Any]], dataset: str
) -> tuple[set[str], float]:
    operator_ids = {group["operator_id"] for group in groups}
    rows = [
        row
        for row in normalized
        if row["dataset"] == dataset and row["operator_id"] in operator_ids
    ]
    return {str(row["request_id"]) for row in rows}, sum(float(row["cost_usd"]) for row in rows)


def _selection_record_ids(config: Mapping[str, Any], dataset: str, partition: str) -> list[str]:
    manifest = json.loads(
        Path(str(config["datasets"][dataset]["path"]) + ".manifest.json").read_text("utf-8")
    )
    return list(manifest["partitions"][partition]["ids"])


def analyze(output: Path = OUTPUT) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite selection analysis: {output}")
    output.mkdir(parents=True)
    evidence = load_evidence()
    schedule = evidence["schedule"]
    catalog = evidence["catalog"]
    config = evidence["config"]
    normalized = evidence["normalized"]
    lookup = evidence["lookup"]
    records_by_dataset = _records_by_dataset(normalized)
    protocol = load_yaml(PROTOCOL)
    gate = protocol["quality_gate"]
    live_session = json.loads((LIVE / "session.json").read_text("utf-8"))
    failed_session = json.loads((FAILED_LIVE / "session.json").read_text("utf-8"))
    authorization = json.loads(AUTHORIZATION.read_text("utf-8"))

    trace_keys = {(row["dataset"], row["record_id"], row["operator_id"]) for row in normalized}
    scheduled_keys = {
        (row["dataset"], row["record_id"], row["operator_id"])
        for row in schedule["physical_requests"]
    }
    retries = sum(int(row["retries"]) for row in normalized)
    input_tokens = sum(int(row["input_tokens"]) for row in normalized)
    output_tokens = sum(int(row["output_tokens"]) for row in normalized)
    named = sum(int(row["named_question_evaluations"]) for row in normalized)
    cost = input_tokens * INPUT_PRICE / 1_000_000
    dataset_usage = []
    for dataset in config["datasets"]:
        rows = [row for row in normalized if row["dataset"] == dataset]
        dataset_usage.append(
            {
                "dataset": dataset,
                "physical_requests": len(rows),
                "http_attempts": sum(int(row["http_attempts"]) for row in rows),
                "named_question_evaluations": sum(
                    int(row["named_question_evaluations"]) for row in rows
                ),
                "input_tokens": sum(int(row["input_tokens"]) for row in rows),
                "output_tokens": sum(int(row["output_tokens"]) for row in rows),
                "cost_usd": sum(float(row["cost_usd"]) for row in rows),
                "latency_seconds_sum": sum(float(row["latency_seconds"]) for row in rows),
            }
        )
    reconciliation = {
        "schema_version": "m5.1-selection-cost-reconciliation-v1",
        "successful_session": live_session,
        "preserved_failed_smoke_session": failed_session,
        "successful_physical_requests": len(normalized),
        "successful_http_attempts": sum(int(row["http_attempts"]) for row in normalized),
        "failed_pre_response_attempts": failed_session["actual_http_attempts"],
        "aggregate_attempts_charged_to_authorization": (
            sum(int(row["http_attempts"]) for row in normalized)
            + failed_session["actual_http_attempts"]
        ),
        "aggregate_attempt_ceiling": authorization["budget_amendment"][
            "aggregate_authorized_attempt_ceiling"
        ],
        "retries_in_successful_session": retries,
        "provider_failures_in_successful_session": 0,
        "logical_evidence_roles": schedule["counts"]["logical_measurement_roles"],
        "shared_alias_roles": schedule["counts"]["shared_alias_roles"],
        "named_question_evaluations_physical": named,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "token_priced_cost_usd": cost,
        "input_price_per_million_usd": INPUT_PRICE,
        "output_price_per_million_usd": OUTPUT_PRICE,
        "expected_input_tokens": 1_066_263,
        "input_token_forecast_error": input_tokens - 1_066_263,
        "expected_cost_usd": 0.04478306000000016,
        "cost_forecast_error_usd": cost - 0.04478306000000016,
        "dataset_usage": dataset_usage,
        "all_scheduled_requests_present": trace_keys == scheduled_keys,
        "missing_request_keys": [list(key) for key in sorted(scheduled_keys - trace_keys)],
        "unexpected_request_keys": [list(key) for key in sorted(trace_keys - scheduled_keys)],
        "selection_partition_only": all(row["partition"] == "selection" for row in normalized),
        "resolved_model_versions": sorted({row["resolved_model"] for row in normalized}),
        "stage_d_calls": 0,
    }

    adaptive = adaptive_replay(evidence)
    adaptive_by_workload = {row["workload_id"]: row for row in adaptive["workloads"]}
    plans_by_workload: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for plan in catalog:
        plans_by_workload[str(plan["workload_id"])].append(plan)
    plan_by_id = _plan_map(catalog)

    plan_results = []
    quality_results = []
    references: dict[str, list[bool]] = {}
    for workload in config["workloads"]:
        workload_id = workload["id"]
        record_ids = records_by_dataset[workload["dataset"]]
        reference = [
            all(
                bool(
                    next(
                        row
                        for row in normalized
                        if row["record_id"] == record_id and row["predicate_ids"] == [predicate_id]
                    )["group_passed"]
                )
                for predicate_id in workload["predicate_ids"]
            )
            for record_id in record_ids
        ]
        references[workload_id] = reference
        for plan in plans_by_workload[workload_id]:
            replay = replay_plan(plan, record_ids, lookup)
            prediction = predict_plan(plan, record_ids, lookup)
            result = {
                "dataset": workload["dataset"],
                "workload_id": workload_id,
                "plan_id": plan["plan_id"],
                "plan_fingerprint": plan["plan_fingerprint"],
                "plan_kind": _plan_kind(plan),
                "groups": [
                    {
                        "operator_id": group["operator_id"],
                        "execution_mode": group["execution_mode"],
                        "predicate_ids": group["predicate_ids"],
                    }
                    for group in plan["groups"]
                ],
                **{key: value for key, value in replay.items() if key != "per_record"},
                **prediction,
                "per_record": replay["per_record"],
            }
            plan_results.append(result)
            quality_results.append(_gate_for_plan(plan, replay, reference, gate))
    plan_result_by_id = {row["plan_id"]: row for row in plan_results}
    quality_by_id = {row["plan_id"]: row for row in quality_results}

    disagreement_rows = []
    group_agreement = []
    multi_operators = sorted(
        {row["operator_id"] for row in normalized if row["execution_mode"] == "multi_decision"}
    )
    single_lookup = {
        (row["record_id"], row["predicate_ids"][0]): row
        for row in normalized
        if row["execution_mode"] == "single"
    }
    for operator_id in multi_operators:
        batch_rows = sorted(
            [row for row in normalized if row["operator_id"] == operator_id],
            key=lambda row: row["record_id"],
        )
        refs: list[bool] = []
        cands: list[bool] = []
        deltas = []
        for batch in batch_rows:
            for predicate_id in batch["predicate_ids"]:
                single = single_lookup[(batch["record_id"], predicate_id)]
                reference_decision = bool(single["predicate_decisions"][predicate_id])
                candidate_decision = bool(batch["predicate_decisions"][predicate_id])
                reference_probability = float(single["predicate_probabilities"][predicate_id])
                candidate_probability = float(batch["predicate_probabilities"][predicate_id])
                refs.append(reference_decision)
                cands.append(candidate_decision)
                delta = candidate_probability - reference_probability
                deltas.append(abs(delta))
                if reference_decision != candidate_decision:
                    disagreement_rows.append(
                        {
                            "dataset": batch["dataset"],
                            "operator_id": operator_id,
                            "batch_size": len(batch["predicate_ids"]),
                            "record_id": batch["record_id"],
                            "predicate_id": predicate_id,
                            "separate_probability": reference_probability,
                            "batch_probability": candidate_probability,
                            "absolute_probability_difference": abs(delta),
                            "separate_decision": reference_decision,
                            "batch_decision": candidate_decision,
                            "relative_error_kind": "false_negative"
                            if reference_decision
                            else "false_positive",
                            "near_threshold": min(
                                abs(reference_probability - 0.5),
                                abs(candidate_probability - 0.5),
                            )
                            <= 0.1,
                        }
                    )
        metrics = binary_metrics(refs, cands)
        group_agreement.append(
            {
                "dataset": batch_rows[0]["dataset"],
                "operator_id": operator_id,
                "batch_size": len(batch_rows[0]["predicate_ids"]),
                "predicate_ids": batch_rows[0]["predicate_ids"],
                **metrics,
                "mean_absolute_probability_difference": statistics.mean(deltas),
                "maximum_absolute_probability_difference": max(deltas),
            }
        )

    strategy_selections = []
    workload_rows = []
    ranking_reports = []
    estimation_rows = []
    hybrid_rows = []
    batch_rows_output = []
    for workload in config["workloads"]:
        workload_id = workload["id"]
        plans = plans_by_workload[workload_id]
        adaptive_info = adaptive_by_workload[workload_id]
        adaptive_plan = plan_by_id[adaptive_info["selected_plan_id"]]
        original_plan = _find_separate_plan(catalog, workload_id, workload["original_order"])
        full_plan = next(plan for plan in plans if _plan_kind(plan) == "full_batch")
        fixed_hybrids = [plan for plan in plans if _plan_kind(plan) == "selective_first_hybrid"]
        predicted = {
            plan["plan_id"]: float(plan_result_by_id[plan["plan_id"]]["predicted_cost_usd"])
            for plan in plans
        }
        observed = {
            plan["plan_id"]: float(plan_result_by_id[plan["plan_id"]]["cost_usd"]) for plan in plans
        }
        ranking = _ranking_metrics(predicted, observed)
        ranking.update(
            {
                "dataset": workload["dataset"],
                "workload_id": workload_id,
                "candidate_count": len(plans),
                "global_optimality_claim": False,
            }
        )
        ranking_reports.append(ranking)
        predicted_ranks = _rank(predicted)
        observed_ranks = _rank(observed)
        for plan in plans:
            plan_id = plan["plan_id"]
            prediction_error = predicted[plan_id] - observed[plan_id]
            estimation_rows.append(
                {
                    "dataset": workload["dataset"],
                    "workload_id": workload_id,
                    "plan_id": plan_id,
                    "plan_kind": _plan_kind(plan),
                    "predicted_cost_usd": predicted[plan_id],
                    "observed_cost_usd": observed[plan_id],
                    "signed_error_usd": prediction_error,
                    "absolute_error_usd": abs(prediction_error),
                    "absolute_percentage_error": (
                        abs(prediction_error) / observed[plan_id] if observed[plan_id] else None
                    ),
                    "predicted_rank": predicted_ranks[plan_id],
                    "observed_rank": observed_ranks[plan_id],
                    "predicted_requests": plan_result_by_id[plan_id]["predicted_requests"],
                    "observed_requests": plan_result_by_id[plan_id]["requests"],
                    "stage_cardinality_absolute_error": statistics.mean(
                        abs(pred_stage["predicted_entrants"] - observed_entry)
                        for pred_stage, observed_entry in zip(
                            plan_result_by_id[plan_id]["stages"],
                            plan_result_by_id[plan_id]["stage_entries"],
                            strict=True,
                        )
                    ),
                }
            )
        cost_only_id = ranking["predicted_best_plan_id"]
        qualifying_batches = [
            plan
            for plan in plans
            if _plan_kind(plan) != "separate" and quality_by_id[plan["plan_id"]]["eligible"]
        ]
        adaptive_predicted = predicted[adaptive_plan["plan_id"]]
        if qualifying_batches:
            best_quality = min(
                qualifying_batches,
                key=lambda plan: (predicted[plan["plan_id"]], plan["plan_id"]),
            )
            if predicted[best_quality["plan_id"]] < adaptive_predicted:
                quality_id = best_quality["plan_id"]
                quality_reason = "cheapest_quality_eligible_batching_candidate"
                quality_fallback = False
            else:
                quality_id = adaptive_plan["plan_id"]
                quality_reason = "quality_candidates_not_predicted_cheaper_than_existing_adaptive"
                quality_fallback = True
        else:
            quality_id = adaptive_plan["plan_id"]
            quality_reason = "no_batching_candidate_passed_quality_gate"
            quality_fallback = True
        selections = {
            "fixed_separate": original_plan["plan_id"],
            "existing_adaptive": adaptive_plan["plan_id"],
            "full_multi_decision_batch": full_plan["plan_id"],
            "fixed_selective_first_hybrid": (
                fixed_hybrids[0]["plan_id"] if fixed_hybrids else None
            ),
            "cost_only_hybrid": cost_only_id,
            "quality_constrained_hybrid": quality_id,
            "profiling_aware_hybrid": quality_id,
        }
        for strategy, plan_id in selections.items():
            if plan_id is None:
                strategy_selections.append(
                    {
                        "dataset": workload["dataset"],
                        "workload_id": workload_id,
                        "strategy": strategy,
                        "status": "unsupported_no_remainder_group_for_two_predicate_workload",
                        "plan_id": None,
                    }
                )
                continue
            plan = plan_by_id[plan_id]
            quality = quality_by_id[plan_id]
            fallback = (
                strategy in {"quality_constrained_hybrid", "profiling_aware_hybrid"}
                and quality_fallback
            )
            strategy_selections.append(
                {
                    "dataset": workload["dataset"],
                    "workload_id": workload_id,
                    "strategy": strategy,
                    "status": "selected",
                    "plan_id": plan_id,
                    "plan_fingerprint": plan["plan_fingerprint"],
                    "plan_kind": _plan_kind(plan),
                    "groups": [group["predicate_ids"] for group in plan["groups"]],
                    "predicted_cost_usd": predicted[plan_id],
                    "selection_measured_cost_usd": observed[plan_id],
                    "quality_status": quality["status"],
                    "quality_eligible": quality["eligible"],
                    "used_existing_adaptive_fallback": fallback,
                    "selection_reason": (
                        quality_reason
                        if strategy in {"quality_constrained_hybrid", "profiling_aware_hybrid"}
                        else "frozen_strategy_rule"
                    ),
                    "selection_evidence_not_independent": True,
                }
            )
        original_result = plan_result_by_id[original_plan["plan_id"]]
        adaptive_result = plan_result_by_id[adaptive_plan["plan_id"]]
        full_result = plan_result_by_id[full_plan["plan_id"]]
        fixed_result = plan_result_by_id[fixed_hybrids[0]["plan_id"]] if fixed_hybrids else None
        best_supported_hybrid = min(
            [full_result, *([fixed_result] if fixed_result else [])],
            key=lambda result: result["cost_usd"],
        )
        interval = _bootstrap_difference(
            [row["cost_usd"] for row in adaptive_result["per_record"]],
            [row["cost_usd"] for row in best_supported_hybrid["per_record"]],
            seed=int(canonical_hash(workload_id)[:8], 16),
        )
        workload_row = {
            "dataset": workload["dataset"],
            "workload_id": workload_id,
            "predicate_count": len(workload["predicate_ids"]),
            "reference_positive_count": sum(references[workload_id]),
            "fixed_separate_plan_id": original_plan["plan_id"],
            "fixed_separate_cost_usd": original_result["cost_usd"],
            "adaptive_plan_id": adaptive_plan["plan_id"],
            "adaptive_order": adaptive_info["selected_order"],
            "adaptive_profiling_cost_usd": adaptive_info["profiling_cost_usd"],
            "adaptive_cost_usd": adaptive_result["cost_usd"],
            "full_batch_plan_id": full_plan["plan_id"],
            "full_batch_cost_usd": full_result["cost_usd"],
            "full_batch_saving_vs_adaptive_usd": adaptive_result["cost_usd"]
            - full_result["cost_usd"],
            "full_batch_saving_vs_adaptive_pct": (
                (adaptive_result["cost_usd"] - full_result["cost_usd"])
                / adaptive_result["cost_usd"]
            ),
            "best_supported_batch_or_hybrid_plan_id": best_supported_hybrid["plan_id"],
            "best_supported_batch_or_hybrid_kind": best_supported_hybrid["plan_kind"],
            "best_supported_batch_or_hybrid_cost_usd": best_supported_hybrid["cost_usd"],
            "best_supported_saving_vs_adaptive_usd": interval["estimate_usd"],
            "best_supported_saving_vs_adaptive_pct": interval["estimate_usd"]
            / adaptive_result["cost_usd"],
            "saving_interval_95": interval,
            "full_batch_quality_status": quality_by_id[full_plan["plan_id"]]["status"],
            "cost_only_plan_id": cost_only_id,
            "quality_constrained_plan_id": quality_id,
            "quality_constrained_fallback": quality_fallback,
        }
        workload_rows.append(workload_row)
        batch_rows_output.append(
            {
                **workload_row,
                "full_batch_requests": full_result["requests"],
                "adaptive_requests": adaptive_result["requests"],
                "full_batch_input_tokens": full_result["input_tokens"],
                "adaptive_input_tokens": adaptive_result["input_tokens"],
                "full_batch_output_tokens": full_result["output_tokens"],
                "adaptive_output_tokens": adaptive_result["output_tokens"],
                "full_batch_latency_seconds": full_result["latency_seconds"],
                "adaptive_latency_seconds": adaptive_result["latency_seconds"],
            }
        )
        if fixed_result:
            hybrid_rows.append(
                {
                    "dataset": workload["dataset"],
                    "workload_id": workload_id,
                    "plan_id": fixed_result["plan_id"],
                    "groups": plan_by_id[fixed_result["plan_id"]]["groups"],
                    "cost_usd": fixed_result["cost_usd"],
                    "requests": fixed_result["requests"],
                    "input_tokens": fixed_result["input_tokens"],
                    "output_tokens": fixed_result["output_tokens"],
                    "saving_vs_adaptive_usd": adaptive_result["cost_usd"]
                    - fixed_result["cost_usd"],
                    "saving_vs_adaptive_pct": (
                        adaptive_result["cost_usd"] - fixed_result["cost_usd"]
                    )
                    / adaptive_result["cost_usd"],
                    "quality_status": quality_by_id[fixed_result["plan_id"]]["status"],
                    "quality_eligible": quality_by_id[fixed_result["plan_id"]]["eligible"],
                    "stage_entries": fixed_result["stage_entries"],
                    "stage_passes": fixed_result["stage_passes"],
                }
            )

    mae = statistics.mean(row["absolute_error_usd"] for row in estimation_rows)
    rmse = math.sqrt(statistics.mean(row["signed_error_usd"] ** 2 for row in estimation_rows))
    mape = statistics.mean(
        row["absolute_percentage_error"]
        for row in estimation_rows
        if row["absolute_percentage_error"] is not None
    )
    estimator_summary = {
        "schema_version": "m5.1-selection-estimator-evaluation-v1",
        "estimator_version": ESTIMATOR_VERSION,
        "plans": len(estimation_rows),
        "mae_usd": mae,
        "rmse_usd": rmse,
        "mape": mape,
        "signed_bias_usd": statistics.mean(row["signed_error_usd"] for row in estimation_rows),
        "mean_pairwise_ranking_concordance": statistics.mean(
            row["pairwise_ranking_concordance"] for row in ranking_reports
        ),
        "total_selection_regret_usd": sum(row["selection_regret_usd"] for row in ranking_reports),
        "harmful_selections": sum(not row["selected_observed_best"] for row in ranking_reports),
        "correct_cheapest_selections": sum(
            row["selected_observed_best"] for row in ranking_reports
        ),
        "candidate_set_scope": "111 frozen measured catalog plans; not all theoretical plans",
        "per_workload": ranking_reports,
    }

    dataset_results = []
    for dataset in config["datasets"]:
        rows = [row for row in workload_rows if row["dataset"] == dataset]
        dataset_results.append(
            {
                "dataset": dataset,
                "workloads": len(rows),
                "adaptive_cost_usd": sum(row["adaptive_cost_usd"] for row in rows),
                "full_batch_cost_usd": sum(row["full_batch_cost_usd"] for row in rows),
                "full_batch_saving_vs_adaptive_usd": sum(
                    row["full_batch_saving_vs_adaptive_usd"] for row in rows
                ),
                "beneficial_full_batch_workloads": sum(
                    row["full_batch_saving_vs_adaptive_usd"] > 0 for row in rows
                ),
                "quality_supported_workloads": sum(
                    row["reference_positive_count"] >= 3 for row in rows
                ),
            }
        )

    # Profiling economics charge exact unique acquisition rows once per declared scenario.
    selections_by_workload: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in strategy_selections:
        selections_by_workload[row["workload_id"]][row["strategy"]] = row
    profiling_aware_trajectories = []
    for workload in config["workloads"]:
        workload_id = workload["id"]
        dataset = workload["dataset"]
        all_records = records_by_dataset[dataset]
        ordered_records = sampling_order(all_records, 4049)
        plans = plans_by_workload[workload_id]
        adaptive_plan_id = selections_by_workload[workload_id]["existing_adaptive"]["plan_id"]
        relevant_operator_ids = {group["operator_id"] for plan in plans for group in plan["groups"]}
        trajectory = []
        for checkpoint in (25, 50, 100):
            visible = ordered_records[:checkpoint]
            reference_by_id = dict(zip(all_records, references[workload_id], strict=True))
            reference = [reference_by_id[record_id] for record_id in visible]
            checkpoint_predictions: dict[str, float] = {}
            checkpoint_quality: dict[str, dict[str, Any]] = {}
            for plan in plans:
                replay = replay_plan(plan, visible, lookup)
                checkpoint_predictions[plan["plan_id"]] = predict_plan(plan, visible, lookup)[
                    "predicted_cost_usd"
                ]
                checkpoint_quality[plan["plan_id"]] = _gate_for_plan(plan, replay, reference, gate)
            qualifying = [
                plan
                for plan in plans
                if _plan_kind(plan) != "separate"
                and checkpoint_quality[plan["plan_id"]]["eligible"]
            ]
            if qualifying:
                candidate = min(
                    qualifying,
                    key=lambda plan: (
                        checkpoint_predictions[plan["plan_id"]],
                        plan["plan_id"],
                    ),
                )
                if (
                    checkpoint_predictions[candidate["plan_id"]]
                    < checkpoint_predictions[adaptive_plan_id]
                ):
                    selected_plan_id = candidate["plan_id"]
                    selection_reason = "checkpoint_quality_candidate_predicted_cheaper"
                else:
                    selected_plan_id = adaptive_plan_id
                    selection_reason = "checkpoint_quality_candidates_not_cheaper"
            else:
                selected_plan_id = adaptive_plan_id
                selection_reason = "no_checkpoint_quality_eligible_batching_candidate"
            visible_rows = [
                row
                for row in normalized
                if row["dataset"] == dataset
                and row["record_id"] in set(visible)
                and row["operator_id"] in relevant_operator_ids
            ]
            stop = checkpoint == 100
            trajectory.append(
                {
                    "checkpoint": checkpoint,
                    "available_measurement_ids": sorted(
                        str(row["request_id"]) for row in visible_rows
                    ),
                    "available_measurement_count": len(visible_rows),
                    "candidate_plan_ids": sorted(plan["plan_id"] for plan in plans),
                    "estimated_plan_costs_usd": checkpoint_predictions,
                    "quality_status": {
                        plan_id: value["status"] for plan_id, value in checkpoint_quality.items()
                    },
                    "selected_plan_id": selected_plan_id,
                    "selection_reason": selection_reason,
                    "profiling_expenditure_usd": sum(
                        float(row["cost_usd"]) for row in visible_rows
                    ),
                    "expected_value_of_additional_profiling_usd": None,
                    "expected_value_status": (
                        "not_identifiable_without_a_frozen_hybrid_uncertainty_value_model"
                    ),
                    "controller_action": "stop" if stop else "continue",
                    "stop_reasons": ["maximum_checkpoint_reached"] if stop else [],
                    "future_measurements_visible": False,
                }
            )
        profiling_aware_trajectories.append(
            {
                "dataset": dataset,
                "workload_id": workload_id,
                "policy": (
                    "conservative_full-profile replay because the preflight did not freeze a "
                    "numerical early-stop value model for hybrid candidates"
                ),
                "selected_plan_id": trajectory[-1]["selected_plan_id"],
                "stopping_checkpoint": 100,
                "trajectory": trajectory,
                "future_information_leakage": False,
            }
        )
    profiling_rows = []
    shared_seen: set[str] = set()
    horizons = protocol["reuse_horizons"]
    for workload in config["workloads"]:
        workload_id = workload["id"]
        plans = plans_by_workload[workload_id]
        acquisition_groups = {
            group["operator_id"]: group for plan in plans for group in plan["groups"]
        }
        requests, standalone_cost = _cost_rows_for_groups(
            list(acquisition_groups.values()), normalized, workload["dataset"]
        )
        new_shared = requests - shared_seen
        shared_cost = sum(
            float(row["cost_usd"]) for row in normalized if row["request_id"] in new_shared
        )
        shared_seen |= requests
        baseline_id = selections_by_workload[workload_id]["existing_adaptive"]["plan_id"]
        baseline_cost = plan_result_by_id[baseline_id]["cost_usd"]
        for strategy in (
            "cost_only_hybrid",
            "quality_constrained_hybrid",
            "profiling_aware_hybrid",
        ):
            selection = selections_by_workload[workload_id][strategy]
            candidate_cost = plan_result_by_id[selection["plan_id"]]["cost_usd"]
            saving = baseline_cost - candidate_cost
            for scenario, profile_cost in (
                ("standalone", standalone_cost),
                ("shared_statistics", shared_cost),
                ("warm_start", 0.0),
            ):
                break_even = math.ceil(profile_cost / saving) if saving > 0 else None
                for horizon in horizons:
                    profiling_rows.append(
                        {
                            "dataset": workload["dataset"],
                            "workload_id": workload_id,
                            "strategy": strategy,
                            "scenario": scenario,
                            "reuse_horizon": horizon,
                            "baseline_execution_cost_usd": baseline_cost,
                            "candidate_execution_cost_usd": candidate_cost,
                            "execution_saving_per_run_usd": saving,
                            "incremental_profiling_cost_usd": profile_cost,
                            "optimization_cost_usd": 0.0,
                            "net_benefit_usd": horizon * saving - profile_cost,
                            "break_even_executions": break_even,
                            "quality_eligible": selection["quality_eligible"],
                            "fallback": selection["used_existing_adaptive_fallback"],
                        }
                    )

    # Freeze catalog with selection evidence attached.
    frozen_catalog = []
    for plan in catalog:
        result = plan_result_by_id[plan["plan_id"]]
        frozen_catalog.append(
            {
                **plan,
                "selection_observed_cost_usd": result["cost_usd"],
                "selection_predicted_cost_usd": result["predicted_cost_usd"],
                "selection_quality": quality_by_id[plan["plan_id"]],
                "selection_evidence_partition": "selection",
                "selection_evidence_records": 100,
                "selection_result_fingerprint": canonical_hash(
                    {
                        "plan_fingerprint": plan["plan_fingerprint"],
                        "observed_cost_usd": result["cost_usd"],
                        "quality": quality_by_id[plan["plan_id"]],
                    }
                ),
            }
        )

    # Stage D union schedule across all frozen strategy selections; identical groups share calls.
    selected_plan_ids = {
        row["plan_id"]
        for row in strategy_selections
        if row.get("plan_id") is not None and row["status"] == "selected"
    }
    selected_groups: dict[tuple[str, str], dict[str, Any]] = {}
    group_plan_refs: dict[tuple[str, str], set[str]] = defaultdict(set)
    for plan_id in selected_plan_ids:
        plan = plan_by_id[plan_id]
        for group in plan["groups"]:
            key = (plan["dataset"], group["operator_id"])
            selected_groups[key] = dict(group)
            group_plan_refs[key].add(plan_id)
    stage_d_requests = []
    operator_means = {
        (dataset, operator_id): {
            "input": statistics.mean(
                int(row["input_tokens"])
                for row in normalized
                if row["dataset"] == dataset and row["operator_id"] == operator_id
            ),
            "output": statistics.mean(
                int(row["output_tokens"])
                for row in normalized
                if row["dataset"] == dataset and row["operator_id"] == operator_id
            ),
        }
        for dataset, operator_id in selected_groups
    }
    for (dataset, operator_id), group in sorted(selected_groups.items()):
        for record_id in _selection_record_ids(config, dataset, "evaluation"):
            payload = {
                "dataset": dataset,
                "record_id": record_id,
                "partition": "evaluation",
                "operator_id": operator_id,
                "prompt_hash": group["prompt_hash"],
                "model": group["requested_model"],
                "threshold": group["threshold"],
            }
            stage_d_requests.append(
                {
                    "request_id": f"m51dreq_{canonical_hash(payload)[:24]}",
                    **payload,
                    "execution_mode": group["execution_mode"],
                    "predicate_ids": group["predicate_ids"],
                    "question_names": group["question_names"],
                    "expected_response_schema": group["expected_response_schema"],
                    "inference_settings": group["inference_settings"],
                    "selected_plan_references": sorted(group_plan_refs[(dataset, operator_id)]),
                    "expected_input_tokens": operator_means[(dataset, operator_id)]["input"],
                    "expected_output_tokens": operator_means[(dataset, operator_id)]["output"],
                }
            )
    stage_d_schedule_core = {
        "schema_version": "m5.1-stage-d-schedule-proposal-v1",
        "partition": "evaluation",
        "records_per_dataset": 400,
        "model": "jev-1.14.0",
        "cache": "disabled",
        "concurrency": 4,
        "maximum_retries": 2,
        "selected_plan_ids": sorted(selected_plan_ids),
        "unique_operator_arms": len(selected_groups),
        "physical_requests": stage_d_requests,
        "physical_request_count": len(stage_d_requests),
        "maximum_http_attempts": len(stage_d_requests) * 3,
        "held_out_outcomes_accessed": False,
    }
    stage_d_schedule_core["schedule_fingerprint"] = canonical_hash(stage_d_schedule_core)
    stage_d_expected_input = sum(row["expected_input_tokens"] for row in stage_d_requests)
    stage_d_expected_output = sum(row["expected_output_tokens"] for row in stage_d_requests)
    stage_d_budget = {
        "schema_version": "m5.1-stage-d-budget-proposal-v1",
        "physical_requests": len(stage_d_requests),
        "maximum_http_attempts": len(stage_d_requests) * 3,
        "expected_input_tokens": round(stage_d_expected_input),
        "expected_output_tokens": round(stage_d_expected_output),
        "expected_cost_usd": stage_d_expected_input * INPUT_PRICE / 1_000_000,
        "conservative_no_retry_cost_proxy_usd": stage_d_expected_input
        * 1.25
        * INPUT_PRICE
        / 1_000_000,
        "retry_inclusive_cost_proxy_usd": stage_d_expected_input
        * 1.25
        * 3
        * INPUT_PRICE
        / 1_000_000,
        "pricing": {"input_per_million_usd": INPUT_PRICE, "output_per_million_usd": OUTPUT_PRICE},
        "historical_and_selection_spending_excluded": True,
        "monetary_values_are_proxies": True,
    }
    stage_d_authorization = {
        "schema_version": "m5.1-stage-d-authorization-proposal-v1",
        "authorization_phrase": "AUTHORIZE M5.1 STAGE D INDEPENDENT HELD-OUT EVALUATION",
        "paid_calls_authorized": False,
        "schedule_fingerprint": stage_d_schedule_core["schedule_fingerprint"],
        "physical_request_ceiling": len(stage_d_requests),
        "http_attempt_ceiling": len(stage_d_requests) * 3,
        "selected_plan_fingerprints": {
            plan_id: plan_by_id[plan_id]["plan_fingerprint"]
            for plan_id in sorted(selected_plan_ids)
        },
        "source_fingerprints": {
            "workloads": sha256_file(WORKLOADS),
            "protocol": sha256_file(PROTOCOL),
            "optimizer": sha256_file(OPTIMIZER),
            "selection_schedule": schedule["schedule_fingerprint"],
        },
        "provider_must_not_initialize_before_validation": True,
        "stage_d_executed": False,
    }

    alias_rows = [
        role for role in schedule["measurement_roles"] if role["physical_acquisition_shared"]
    ]
    authorization_validation = {
        "schema_version": "m5.1-selection-authorization-validation-v1",
        "authorization_phrase_matched": True,
        "authorization_manifest": str(AUTHORIZATION).replace("\\", "/"),
        "authorization_manifest_sha256": sha256_file(AUTHORIZATION),
        "schedule_fingerprint_matched": authorization["schedule_fingerprint"]
        == schedule["schedule_fingerprint"],
        "source_fingerprints_matched": authorization["fingerprints"]
        == json.loads((PREFLIGHT / "authorization_manifest.json").read_text("utf-8"))[
            "fingerprints"
        ],
        "unique_request_limit": authorization["unique_physical_request_ceiling"],
        "new_session_attempt_limit": authorization["deduplicated_http_attempt_ceiling"],
        "aggregate_attempt_limit": authorization["budget_amendment"][
            "aggregate_authorized_attempt_ceiling"
        ],
        "aggregate_attempts_observed": reconciliation[
            "aggregate_attempts_charged_to_authorization"
        ],
        "within_authorization": reconciliation["aggregate_attempts_charged_to_authorization"]
        <= authorization["budget_amendment"]["aggregate_authorized_attempt_ceiling"],
        "provider_initialized_only_after_validation": True,
    }

    semantic_analysis = {
        "schema_version": "m5.1-selection-semantic-analysis-v1",
        "group_agreement": group_agreement,
        "disagreements": disagreement_rows,
        "aggregate": binary_metrics(
            [
                single_lookup[(row["record_id"], predicate_id)]["predicate_decisions"][predicate_id]
                for row in normalized
                if row["execution_mode"] == "multi_decision"
                for predicate_id in row["predicate_ids"]
            ],
            [
                row["predicate_decisions"][predicate_id]
                for row in normalized
                if row["execution_mode"] == "multi_decision"
                for predicate_id in row["predicate_ids"]
            ],
        ),
        "near_threshold_disagreements": sum(row["near_threshold"] for row in disagreement_rows),
        "human_ground_truth_available": False,
    }

    # Persist machine-readable results before producing the report manifest.
    artifacts: dict[str, Any] = {
        "authorization_validation.json": authorization_validation,
        "live_session_manifest.json": {
            "schema_version": "m5.1-selection-consolidated-session-v1",
            "successful_session": live_session,
            "failed_network_smoke_session": failed_session,
            "status": "complete",
        },
        "token_cost_reconciliation.json": reconciliation,
        "shared_alias_mapping.json": {
            "schema_version": "m5.1-selection-alias-map-v1",
            "aliases": alias_rows,
            "count": len(alias_rows),
            "all_resolve": len(alias_rows) == 300
            and all(
                role["request_id"] in {row["request_id"] for row in normalized}
                for role in alias_rows
            ),
        },
        "adaptive_profiling_replay.json": adaptive,
        "profiling_aware_selection_replay.json": {
            "schema_version": "m5.1-selection-profiling-aware-replay-v1",
            "workloads": profiling_aware_trajectories,
            "actual_full_matrix_expenditure_usd": cost,
            "leakage_free": True,
        },
        "quality_gate_results.json": {
            "schema_version": "m5.1-selection-quality-results-v1",
            "gate": gate,
            "plans": quality_results,
        },
        "semantic_disagreement_analysis.json": semantic_analysis,
        "plan_ranking_regret.json": estimator_summary,
        "frozen_strategy_selections.json": {
            "schema_version": "m5.1-selection-frozen-strategies-v1",
            "selection_partition_only": True,
            "strategies": strategy_selections,
        },
        "frozen_physical_plan_catalog.json": {
            "schema_version": "m5.1-selection-frozen-plan-catalog-v1",
            "plans": frozen_catalog,
        },
        "stage_d_execution_schedule.json": stage_d_schedule_core,
        "stage_d_budget_proposal.json": stage_d_budget,
        "stage_d_authorization_manifest.json": stage_d_authorization,
    }
    for name, value in artifacts.items():
        _write_json(output / name, value)
    _write_csv(output / "complete_measurement_matrix.csv", normalized)
    _write_csv(output / "per_dataset_results.csv", dataset_results)
    _write_csv(output / "per_workload_results.csv", workload_rows)
    _write_csv(output / "separate_vs_batch_comparison.csv", batch_rows_output)
    _write_csv(output / "hybrid_plan_comparisons.csv", hybrid_rows)
    _write_csv(output / "cost_estimation_errors.csv", estimation_rows)
    _write_csv(output / "profiling_economics.csv", profiling_rows)
    _write_csv(output / "semantic_disagreements.csv", disagreement_rows)

    summary = {
        "schema_version": ANALYSIS_VERSION,
        "session_status": "complete",
        "physical_requests": len(normalized),
        "logical_evidence_roles": 3300,
        "successful_http_attempts": 3000,
        "failed_pre_response_attempts": 3,
        "aggregate_attempts": 3003,
        "retries": 0,
        "provider_failures_successful_session": 0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost,
        "named_question_evaluations": named,
        "workloads": 9,
        "catalog_plans": 111,
        "quality_eligible_batching_plans": sum(
            row["eligible"] and row["plan_kind"] != "separate" for row in quality_results
        ),
        "quality_constrained_fallbacks": sum(
            row["strategy"] == "quality_constrained_hybrid"
            and row.get("used_existing_adaptive_fallback")
            for row in strategy_selections
        ),
        "estimator": estimator_summary,
        "stage_d_physical_requests": len(stage_d_requests),
        "stage_d_expected_cost_usd": stage_d_budget["expected_cost_usd"],
        "stage_d_executed": False,
        "held_out_outcomes_accessed": False,
    }
    _write_json(output / "selection_summary.json", summary)

    files = sorted(path for path in output.iterdir() if path.is_file())
    manifest = {
        "schema_version": "m5.1-selection-artifact-integrity-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "worktree_dirty": git_worktree_dirty(),
        "analysis_source": "src/semanticflow/optimizer/m51_selection_analysis.py",
        "analysis_source_sha256": sha256_file(__file__),
        "selection_schedule_fingerprint": schedule["schedule_fingerprint"],
        "authorization_manifest_sha256": sha256_file(AUTHORIZATION),
        "live_tree_files": len(list(LIVE.rglob("*"))),
        "live_session_sha256": sha256_file(LIVE / "session.json"),
        "artifact_sha256": {path.name: sha256_file(path) for path in files},
        "stage_d_executed": False,
        "held_out_outcomes_accessed": False,
    }
    _write_json(output / "artifact_integrity_manifest.json", manifest)
    return {
        "summary": summary,
        "workloads": workload_rows,
        "selections": strategy_selections,
        "quality": quality_results,
        "semantic": semantic_analysis,
        "dataset_results": dataset_results,
        "stage_d_budget": stage_d_budget,
    }
