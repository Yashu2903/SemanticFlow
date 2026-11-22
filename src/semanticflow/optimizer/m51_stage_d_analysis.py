from __future__ import annotations

import csv
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from semanticflow.optimizer.m51_experiment import load_yaml
from semanticflow.optimizer.m51_selection_analysis import (
    INPUT_PRICE,
    _bootstrap_difference,
    _gate_for_plan,
    binary_metrics,
    predict_plan,
    quality_status,
    replay_plan,
)

SCHEDULE = Path("results/milestone_5_1/selection_stage_results/stage_d_execution_schedule.json")
CATALOG = Path("results/milestone_5_1/selection_stage_results/frozen_physical_plan_catalog.json")
SELECTIONS = Path("results/milestone_5_1/selection_stage_results/frozen_strategy_selections.json")
WORKLOADS = Path("configs/milestone_5_1/workloads.yaml")
PROTOCOL = Path("configs/milestone_5_1/protocol.yaml")
HORIZONS = (1, 5, 10, 25, 50, 100, 250, 1000)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text("utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("status,reason\nunavailable,no compatible measurements\n", encoding="utf-8")
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
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _trace_files(root: Path) -> list[Path]:
    return sorted(root.glob("smoke/*/decisions.jsonl")) + sorted(
        root.glob("operators/*/*/decisions.jsonl")
    )


def normalize_stage_d_session(
    session_root: Path, schedule: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    session = _read_json(session_root / "session.json")
    if session.get("status") != "complete":
        raise ValueError("Stage D session is not complete")
    if session.get("schedule_fingerprint") != schedule.get("schedule_fingerprint"):
        raise ValueError("Stage D session/schedule fingerprint mismatch")
    request_by_key = {
        (row["dataset"], row["record_id"], row["operator_id"]): row
        for row in schedule["physical_requests"]
    }
    normalized: list[dict[str, Any]] = []
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for path in _trace_files(session_root):
        for line in path.read_text("utf-8").splitlines():
            if not line:
                continue
            trace = json.loads(line)
            dataset = str(trace["review_id"]).split(":", 1)[0]
            key = (dataset, trace["review_id"], trace["group_id"])
            request = request_by_key.get(key)
            if request is None:
                raise ValueError(f"unscheduled Stage D trace: {key}")
            if set(trace["predicate_probabilities"]) != set(request["predicate_ids"]):
                raise ValueError(f"partial named response: {request['request_id']}")
            row = {
                "request_id": request["request_id"],
                "dataset": dataset,
                "record_id": trace["review_id"],
                "partition": "evaluation",
                "operator_id": trace["group_id"],
                "execution_mode": trace["fusion_mode"],
                "predicate_ids": trace["predicate_ids"],
                "question_names": request["question_names"],
                "prompt_hash": trace["fusion_prompt_hash"],
                "requested_model": request["model"],
                "resolved_model": trace["model_identifier"],
                "threshold": trace["threshold"],
                "input_tokens": int(trace["input_tokens"]),
                "output_tokens": int(trace["output_tokens"]),
                "cost_usd": int(trace["input_tokens"]) * INPUT_PRICE / 1_000_000,
                "latency_seconds": float(trace["request_latency_seconds"]),
                "http_attempts": int(trace["request_attempt_count"]),
                "retries": int(trace["request_attempt_count"]) - 1,
                "named_question_evaluations": len(trace["predicate_ids"]),
                "group_passed": bool(trace["passed"]),
                "group_probability": trace["group_probability"],
                "predicate_probabilities": trace["predicate_probabilities"],
                "predicate_decisions": {
                    predicate: probability >= float(trace["threshold"])
                    for predicate, probability in trace["predicate_probabilities"].items()
                },
                "response_status": trace["provider_metadata"].get("http_status"),
                "trace_path": str(path).replace("\\", "/"),
            }
            if (row["record_id"], row["operator_id"]) in lookup:
                raise ValueError(f"duplicate Stage D trace: {key}")
            normalized.append(row)
            lookup[(row["record_id"], row["operator_id"])] = row
    scheduled = {
        (row["dataset"], row["record_id"], row["operator_id"])
        for row in schedule["physical_requests"]
    }
    observed = {(row["dataset"], row["record_id"], row["operator_id"]) for row in normalized}
    if observed != scheduled:
        raise ValueError(
            f"Stage D request reconciliation failed: missing={len(scheduled-observed)}, "
            f"unexpected={len(observed-scheduled)}"
        )
    return normalized, lookup, session


def _selections_by_strategy(
    selections: Sequence[Mapping[str, Any]], strategy: str
) -> dict[str, Mapping[str, Any]]:
    return {
        str(row["workload_id"]): row
        for row in selections
        if row["strategy"] == strategy and row["status"] == "selected"
    }


def _plan_kind(plan: Mapping[str, Any]) -> str:
    if all(group["execution_mode"] == "single" for group in plan["groups"]):
        return "separate"
    return "full_batch" if len(plan["groups"]) == 1 else "selective_first_hybrid"


def _individual_agreement(
    full_plan: Mapping[str, Any],
    adaptive_plan: Mapping[str, Any],
    record_ids: Sequence[str],
    lookup: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    singles = {group["predicate_ids"][0]: group["operator_id"] for group in adaptive_plan["groups"]}
    batch_group = full_plan["groups"][0]
    reference: list[bool] = []
    candidate: list[bool] = []
    differences: list[float] = []
    for record_id in record_ids:
        batch = lookup[(record_id, batch_group["operator_id"])]
        for predicate in batch_group["predicate_ids"]:
            separate = lookup[(record_id, singles[predicate])]
            reference.append(bool(separate["predicate_decisions"][predicate]))
            candidate.append(bool(batch["predicate_decisions"][predicate]))
            differences.append(
                abs(
                    float(separate["predicate_probabilities"][predicate])
                    - float(batch["predicate_probabilities"][predicate])
                )
            )
    return {
        **binary_metrics(reference, candidate),
        "mean_absolute_probability_difference": statistics.mean(differences),
        "maximum_absolute_probability_difference": max(differences),
        "semantic_reference": "separate_jev_not_human_ground_truth",
    }


def _profiling_rows(
    workload_id: str,
    baseline_cost: float,
    candidate_cost: float,
    quality_eligible: bool,
) -> list[dict[str, Any]]:
    saving = baseline_cost - candidate_cost
    if workload_id == "m51_banking_card_unauthorized_refund":
        scenarios = (
            ("standalone_cold_start", 0.007067130),
            ("shared_statistics_cold_start", 0.002967972),
            ("warm_start", 0.0),
        )
    else:
        scenarios = (("fallback_same_as_baseline", 0.0),)
    rows = []
    for scenario, profiling_cost in scenarios:
        break_even = math.ceil(profiling_cost / saving) if saving > 0 else None
        for horizon in HORIZONS:
            rows.append(
                {
                    "workload_id": workload_id,
                    "scenario": scenario,
                    "reuse_horizon": horizon,
                    "baseline_cost_usd": baseline_cost,
                    "candidate_cost_usd": candidate_cost,
                    "execution_saving_per_run_usd": saving,
                    "incremental_profiling_cost_usd": profiling_cost,
                    "optimization_cost_usd": None,
                    "net_benefit_excluding_unmetered_optimization_usd": (
                        horizon * saving - profiling_cost
                    ),
                    "break_even_executions_excluding_unmetered_optimization": break_even,
                    "quality_eligible": quality_eligible,
                }
            )
    return rows


def dry_run_contract() -> dict[str, Any]:
    metrics = binary_metrics([False, True, True, False], [False, True, False, False])
    gate = {
        "minimum_records": 4,
        "minimum_reference_positives": 2,
        "minimum_agreement": 0.75,
        "minimum_recall": 0.5,
        "maximum_false_negative_rate": 0.5,
        "maximum_changed_decision_rate": 0.25,
    }
    interval = _bootstrap_difference(
        [1.0, 2.0, 3.0, 4.0],
        [0.5, 2.5, 2.0, 3.5],
        seed=7,
        iterations=99,
    )
    return {
        "mock_only": True,
        "binary_metrics_valid": metrics["false_negative"] == 1 and metrics["false_positive"] == 0,
        "quality_status": quality_status(metrics, gate)["status"],
        "paired_bootstrap_has_bounds": interval["lower_95_usd"] is not None,
        "missing_measurements_are_unavailable_not_zero": True,
        "held_out_outcomes_accessed": False,
    }


def analyze(
    session_root: Path,
    output: Path,
    *,
    schedule_path: Path = SCHEDULE,
    catalog_path: Path = CATALOG,
    selections_path: Path = SELECTIONS,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Stage D analysis: {output}")
    output.mkdir(parents=True)
    schedule = _read_json(schedule_path)
    catalog = _read_json(catalog_path)["plans"]
    selections = _read_json(selections_path)["strategies"]
    config = load_yaml(WORKLOADS)
    gate = load_yaml(PROTOCOL)["quality_gate"]
    normalized, lookup, session = normalize_stage_d_session(session_root, schedule)
    plan_by_id = {str(plan["plan_id"]): plan for plan in catalog}
    adaptive = _selections_by_strategy(selections, "existing_adaptive")
    full = _selections_by_strategy(selections, "full_multi_decision_batch")
    quality_selected = _selections_by_strategy(selections, "quality_constrained_hybrid")
    records_by_dataset = {
        dataset: sorted(
            {row["record_id"] for row in normalized if row["dataset"] == dataset}
        )
        for dataset in config["datasets"]
    }

    workload_rows = []
    quality_rows = []
    query_rows = []
    semantic_rows = []
    estimation_rows = []
    fallback_rows = []
    profiling_rows = []
    uncertainty = []
    for workload in config["workloads"]:
        workload_id = workload["id"]
        record_ids = records_by_dataset[workload["dataset"]]
        adaptive_plan = plan_by_id[adaptive[workload_id]["plan_id"]]
        full_plan = plan_by_id[full[workload_id]["plan_id"]]
        selected_plan = plan_by_id[quality_selected[workload_id]["plan_id"]]
        adaptive_result = replay_plan(adaptive_plan, record_ids, lookup)
        full_result = replay_plan(full_plan, record_ids, lookup)
        selected_result = replay_plan(selected_plan, record_ids, lookup)
        reference = [row["passed"] for row in adaptive_result["per_record"]]
        full_gate = _gate_for_plan(full_plan, full_result, reference, gate)
        selected_gate = _gate_for_plan(selected_plan, selected_result, reference, gate)
        full_interval = _bootstrap_difference(
            [row["cost_usd"] for row in adaptive_result["per_record"]],
            [row["cost_usd"] for row in full_result["per_record"]],
            seed=51103,
            iterations=10000,
        )
        selected_interval = _bootstrap_difference(
            [row["cost_usd"] for row in adaptive_result["per_record"]],
            [row["cost_usd"] for row in selected_result["per_record"]],
            seed=51103,
            iterations=10000,
        )
        individual = _individual_agreement(full_plan, adaptive_plan, record_ids, lookup)
        saving = adaptive_result["cost_usd"] - selected_result["cost_usd"]
        workload_rows.append(
            {
                "dataset": workload["dataset"],
                "workload_id": workload_id,
                "adaptive_plan_id": adaptive_plan["plan_id"],
                "adaptive_cost_usd": adaptive_result["cost_usd"],
                "adaptive_requests": adaptive_result["requests"],
                "full_batch_plan_id": full_plan["plan_id"],
                "full_batch_cost_usd": full_result["cost_usd"],
                "full_batch_requests": full_result["requests"],
                "full_batch_saving_usd": adaptive_result["cost_usd"]
                - full_result["cost_usd"],
                "selected_plan_id": selected_plan["plan_id"],
                "selected_plan_kind": _plan_kind(selected_plan),
                "selected_cost_usd": selected_result["cost_usd"],
                "selected_saving_usd": saving,
                "selected_saving_pct": saving / adaptive_result["cost_usd"],
                "selected_cost_interval_95": selected_interval,
                "selected_quality_status": selected_gate["status"],
                "selected_quality_eligible": selected_gate["eligible"],
            }
        )
        quality_rows.extend((full_gate, selected_gate))
        query_rows.append(
            {
                "workload_id": workload_id,
                "candidate": "full_batch",
                **{key: full_gate[key] for key in (
                    "sample_count",
                    "reference_positive_count",
                    "agreement",
                    "false_positive",
                    "false_negative",
                    "precision",
                    "recall",
                    "changed_decision_rate",
                    "jaccard",
                    "match_set_equal",
                    "status",
                )},
            }
        )
        semantic_rows.append({"workload_id": workload_id, **individual})
        uncertainty.append(
            {
                "workload_id": workload_id,
                "full_batch_saving_interval": full_interval,
                "selected_plan_saving_interval": selected_interval,
            }
        )
        candidate_plans = {
            plan_id: plan_by_id[plan_id]
            for plan_id in schedule["selected_plan_ids"]
            if plan_by_id[plan_id]["workload_id"] == workload_id
        }
        plan_results = {}
        for plan_id, plan in candidate_plans.items():
            observed = replay_plan(plan, record_ids, lookup)
            predicted = predict_plan(plan, record_ids, lookup)
            plan_results[plan_id] = observed["cost_usd"]
            estimation_rows.append(
                {
                    "workload_id": workload_id,
                    "plan_id": plan_id,
                    "plan_kind": _plan_kind(plan),
                    "observed_cost_usd": observed["cost_usd"],
                    "predicted_cost_usd": predicted["predicted_cost_usd"],
                    "signed_error_usd": predicted["predicted_cost_usd"] - observed["cost_usd"],
                    "absolute_error_usd": abs(
                        predicted["predicted_cost_usd"] - observed["cost_usd"]
                    ),
                }
            )
        best = min(plan_results, key=lambda plan_id: (plan_results[plan_id], plan_id))
        regret = selected_result["cost_usd"] - plan_results[best]
        fallback = bool(quality_selected[workload_id]["used_existing_adaptive_fallback"])
        cheaper_full = full_result["cost_usd"] < adaptive_result["cost_usd"]
        fallback_rows.append(
            {
                "workload_id": workload_id,
                "frozen_fallback": fallback,
                "full_batch_cheaper": cheaper_full,
                "full_batch_quality_eligible": full_gate["eligible"],
                "classification": (
                    "excessively_conservative"
                    if fallback and cheaper_full and full_gate["eligible"]
                    else "correct_fallback"
                    if fallback
                    else "selected_batch_confirmed"
                    if cheaper_full and full_gate["eligible"]
                    else "selected_batch_not_confirmed"
                ),
                "measured_eligible_set_best_plan_id": best,
                "selected_plan_regret_usd": regret,
            }
        )
        profiling_rows.extend(
            _profiling_rows(
                workload_id,
                adaptive_result["cost_usd"],
                selected_result["cost_usd"],
                selected_gate["eligible"],
            )
        )

    input_tokens = sum(row["input_tokens"] for row in normalized)
    output_tokens = sum(row["output_tokens"] for row in normalized)
    attempts = sum(row["http_attempts"] for row in normalized)
    reconciliation = {
        "session_status": session["status"],
        "physical_requests": len(normalized),
        "scheduled_requests": len(schedule["physical_requests"]),
        "http_attempts": attempts,
        "retries": attempts - len(normalized),
        "named_question_evaluations": sum(
            row["named_question_evaluations"] for row in normalized
        ),
        "all_requests_reconciled": len(normalized) == len(schedule["physical_requests"]),
    }
    expenditure = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_token_priced_cost_usd": input_tokens * INPUT_PRICE / 1_000_000,
        "output_token_price_usd": 0.0,
        "provider_invoice_established": False,
    }
    per_dataset = []
    for dataset in config["datasets"]:
        rows = [row for row in normalized if row["dataset"] == dataset]
        per_dataset.append(
            {
                "dataset": dataset,
                "requests": len(rows),
                "input_tokens": sum(row["input_tokens"] for row in rows),
                "output_tokens": sum(row["output_tokens"] for row in rows),
                "cost_usd": sum(row["cost_usd"] for row in rows),
            }
        )
    batch_size = []
    for size in (2, 3, 4):
        rows = [
            row
            for row in normalized
            if row["execution_mode"] == "multi_decision" and len(row["predicate_ids"]) == size
        ]
        batch_size.append(
            {
                "batch_size": size,
                "requests": len(rows),
                "input_tokens": sum(row["input_tokens"] for row in rows),
                "output_tokens": sum(row["output_tokens"] for row in rows),
                "cost_usd": sum(row["cost_usd"] for row in rows),
            }
        )
    errors = [row["signed_error_usd"] for row in estimation_rows]
    regret_summary = {
        "workloads": fallback_rows,
        "total_selected_plan_regret_usd": sum(
            row["selected_plan_regret_usd"] for row in fallback_rows
        ),
        "correct_fallbacks": sum(
            row["classification"] == "correct_fallback" for row in fallback_rows
        ),
        "excessively_conservative_fallbacks": sum(
            row["classification"] == "excessively_conservative" for row in fallback_rows
        ),
    }
    negative = {
        "costly_full_batches": [
            row["workload_id"] for row in workload_rows if row["full_batch_saving_usd"] < 0
        ],
        "quality_failures": [
            row["workload_id"] for row in query_rows if row["status"] == "failed"
        ],
        "insufficient_positive_support": [
            row["workload_id"]
            for row in query_rows
            if row["status"] == "insufficient_positive_support"
        ],
        "post_hoc_policy_changes_permitted": False,
    }
    artifacts: dict[str, Any] = {
        "request_reconciliation.json": reconciliation,
        "provider_expenditure.json": expenditure,
        "semantic_agreement.json": {"full_batch_individual": semantic_rows},
        "quality_gate_results.json": {"gate": gate, "plans": quality_rows},
        "frozen_plan_regret.json": regret_summary,
        "fallback_correctness.json": {"workloads": fallback_rows},
        "statistical_uncertainty.json": {"workloads": uncertainty},
        "negative_result_analysis.json": negative,
        "cost_estimator_summary.json": {
            "mae_usd": statistics.mean(abs(value) for value in errors),
            "rmse_usd": math.sqrt(statistics.mean(value * value for value in errors)),
            "signed_bias_usd": statistics.mean(errors),
            "candidate_scope": "frozen measured Stage D plan union only",
        },
    }
    for name, value in artifacts.items():
        _write_json(output / name, value)
    _write_csv(output / "per_dataset_results.csv", per_dataset)
    _write_csv(output / "per_workload_results.csv", workload_rows)
    _write_csv(output / "adaptive_vs_batch.csv", workload_rows)
    _write_csv(output / "batch_size_economics.csv", batch_size)
    _write_csv(output / "query_level_agreement.csv", query_rows)
    _write_csv(output / "cost_estimator_errors.csv", estimation_rows)
    _write_csv(output / "profiling_economics.csv", profiling_rows)
    return {
        "status": "complete",
        "requests": len(normalized),
        "input_tokens": input_tokens,
        "cost_usd": expenditure["input_token_priced_cost_usd"],
        "workloads": len(workload_rows),
    }
