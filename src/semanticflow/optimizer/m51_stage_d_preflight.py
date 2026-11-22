from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from semanticflow.optimizer.fusion_experiment import sha256_file
from semanticflow.optimizer.m51_experiment import load_yaml

ROOT = Path("results/milestone_5_1/stage_d_final_preflight")
SELECTION_RESULTS = Path("results/milestone_5_1/selection_stage_results")
SELECTION_PREFLIGHT = Path("results/milestone_5_1/selection_stage_preflight")
SCHEDULE_PATH = SELECTION_RESULTS / "stage_d_execution_schedule.json"
CATALOG_PATH = SELECTION_RESULTS / "frozen_physical_plan_catalog.json"
SELECTIONS_PATH = SELECTION_RESULTS / "frozen_strategy_selections.json"
AUTHORIZATION_PATH = SELECTION_RESULTS / "stage_d_authorization_manifest.json"
WORKLOADS_PATH = Path("configs/milestone_5_1/workloads.yaml")
PROTOCOL_PATH = Path("configs/milestone_5_1/protocol.yaml")
OPTIMIZER_PATH = Path("configs/optimizer.milestone_4.yaml")
RUNNER_PATH = Path("experiments/milestone_5_1/run_stage_d.py")
ANALYZER_PATH = Path("src/semanticflow/optimizer/m51_stage_d_analysis.py")

INPUT_PRICE = 0.042
OUTPUT_PRICE = 0.0
PRIMARY_WORKLOAD = "m51_banking_card_unauthorized_refund"
PRIMARY_PLAN_ID = "m51selplan_4e05511d49f2861f7dfb3361"
PRIMARY_PLAN_FINGERPRINT = (
    "4e05511d49f2861f7dfb336183e349ed2c5b03c93e05a01cca2943431d91e1da"
)
PREFLIGHT_VERSION = "m5.1-stage-d-final-preflight-v1"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text("utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _plan_execution_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset": plan["dataset"],
        "workload_id": plan["workload_id"],
        "logical_predicate_ids": plan["logical_predicate_ids"],
        "groups": plan["groups"],
        "model": plan["model"],
        "threshold": plan["threshold"],
        "execution_settings": plan["execution_settings"],
        "planner_version": plan["planner_version"],
        "configuration_fingerprint": plan["configuration_fingerprint"],
    }


def _schedule_payload(schedule: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in schedule.items() if key != "schedule_fingerprint"}


def _request_payload(request: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset": request["dataset"],
        "record_id": request["record_id"],
        "partition": request["partition"],
        "operator_id": request["operator_id"],
        "prompt_hash": request["prompt_hash"],
        "model": request["model"],
        "threshold": request["threshold"],
    }


def _manifest(config: Mapping[str, Any], dataset: str) -> tuple[Path, dict[str, Any]]:
    path = Path(str(config["datasets"][dataset]["path"]) + ".manifest.json")
    return path, _read_json(path)


def _group_operator_ids(plan: Mapping[str, Any]) -> set[str]:
    return {str(group["operator_id"]) for group in plan["groups"]}


def _budget_rows(requests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    input_tokens = sum(float(row["expected_input_tokens"]) for row in requests)
    output_tokens = sum(float(row["expected_output_tokens"]) for row in requests)
    expected_cost = input_tokens * INPUT_PRICE / 1_000_000
    conservative_cost = input_tokens * 1.25 * INPUT_PRICE / 1_000_000
    return {
        "unique_physical_requests": len(requests),
        "expected_input_tokens": round(input_tokens),
        "expected_output_tokens": round(output_tokens),
        "expected_token_priced_cost_usd": expected_cost,
        "conservative_no_retry_cost_proxy_usd": conservative_cost,
        "retry_inclusive_cost_proxy_usd": conservative_cost * 3,
        "maximum_http_attempts": len(requests) * 3,
    }


def _selected_by_strategy(
    selections: Sequence[Mapping[str, Any]], strategy: str
) -> dict[str, Mapping[str, Any]]:
    return {
        str(row["workload_id"]): row
        for row in selections
        if row["strategy"] == strategy and row["status"] == "selected"
    }


def validate_frozen_plans(
    catalog: Sequence[Mapping[str, Any]],
    selections: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_id = {str(plan["plan_id"]): plan for plan in catalog}
    quality = _selected_by_strategy(selections, "quality_constrained_hybrid")
    profiling = _selected_by_strategy(selections, "profiling_aware_hybrid")
    adaptive = _selected_by_strategy(selections, "existing_adaptive")
    workload_config = {str(row["id"]): row for row in config["workloads"]}
    errors: list[str] = []
    mapping: list[dict[str, Any]] = []

    if set(quality) != set(workload_config) or set(profiling) != set(workload_config):
        errors.append("quality/profiling strategy mappings do not cover all nine workloads")

    for workload_id, definition in workload_config.items():
        selected = quality.get(workload_id)
        profile_selected = profiling.get(workload_id)
        adaptive_selected = adaptive.get(workload_id)
        if selected is None or profile_selected is None or adaptive_selected is None:
            continue
        plan = by_id.get(str(selected["plan_id"]))
        adaptive_plan = by_id.get(str(adaptive_selected["plan_id"]))
        if plan is None or adaptive_plan is None:
            errors.append(f"{workload_id}: selected plan missing from frozen catalog")
            continue
        observed_hash = canonical_hash(_plan_execution_payload(plan))
        if observed_hash != plan["plan_fingerprint"]:
            errors.append(f"{workload_id}: canonical plan fingerprint mismatch")
        if plan["plan_fingerprint"] != selected["plan_fingerprint"]:
            errors.append(f"{workload_id}: selection/catalog fingerprint mismatch")
        if profile_selected["plan_id"] != selected["plan_id"]:
            errors.append(f"{workload_id}: quality and profiling-aware selections differ")
        if plan["dataset"] != definition["dataset"]:
            errors.append(f"{workload_id}: selected plan dataset mismatch")
        if plan["logical_predicate_ids"] != definition["predicate_ids"]:
            errors.append(f"{workload_id}: logical predicate mismatch")
        flattened = [predicate for group in plan["groups"] for predicate in group["predicate_ids"]]
        if sorted(flattened) != sorted(definition["predicate_ids"]):
            errors.append(f"{workload_id}: plan does not contain every predicate exactly once")
        if plan["model"] != "jev-1.14.0" or float(plan["threshold"]) != 0.5:
            errors.append(f"{workload_id}: model or threshold changed")
        if plan["execution_settings"] != {
            "cache": "disabled",
            "concurrency": 4,
            "max_retries": 2,
        }:
            errors.append(f"{workload_id}: execution settings changed")

        fallback = bool(selected["used_existing_adaptive_fallback"])
        if workload_id == PRIMARY_WORKLOAD:
            if selected["plan_id"] != PRIMARY_PLAN_ID:
                errors.append("primary Banking77 plan ID changed")
            if plan["plan_fingerprint"] != PRIMARY_PLAN_FINGERPRINT:
                errors.append("primary Banking77 plan fingerprint changed")
            if fallback or len(plan["groups"]) != 1:
                errors.append("primary Banking77 plan is not one physical batch")
            group = plan["groups"][0]
            if group["execution_mode"] != "multi_decision":
                errors.append("primary Banking77 plan is not multi-decision")
            if len(group["question_names"]) != 3 or len(group["provider_questions"]) != 3:
                errors.append("primary Banking77 plan does not expose three named questions")
            if sorted(group["predicate_ids"]) != sorted(definition["predicate_ids"]):
                errors.append("primary Banking77 batch predicate set changed")
        else:
            if not fallback:
                errors.append(f"{workload_id}: expected adaptive fallback was not preserved")
            if selected["plan_id"] != adaptive_selected["plan_id"]:
                errors.append(f"{workload_id}: fallback differs from existing_adaptive")
            if any(group["execution_mode"] != "single" for group in plan["groups"]):
                errors.append(f"{workload_id}: fallback silently introduced batching")

        mapping.append(
            {
                "dataset": definition["dataset"],
                "workload_id": workload_id,
                "logical_predicate_ids": definition["predicate_ids"],
                "predicate_count": len(definition["predicate_ids"]),
                "original_predicate_order": definition["original_order"],
                "selected_strategy": "quality_constrained_hybrid",
                "selected_plan_id": plan["plan_id"],
                "selected_plan_fingerprint": plan["plan_fingerprint"],
                "selected_plan_kind": selected["plan_kind"],
                "group_order": [group["predicate_ids"] for group in plan["groups"]],
                "group_execution_modes": [group["execution_mode"] for group in plan["groups"]],
                "operator_ids": [group["operator_id"] for group in plan["groups"]],
                "prompt_hashes": [group["prompt_hash"] for group in plan["groups"]],
                "question_names": [group["question_names"] for group in plan["groups"]],
                "model": plan["model"],
                "threshold": plan["threshold"],
                "quality_status": selected["quality_status"],
                "quality_eligible": selected["quality_eligible"],
                "selection_reason": selected["selection_reason"],
                "used_existing_adaptive_fallback": fallback,
                "selection_measured_cost_usd": selected["selection_measured_cost_usd"],
                "selection_predicted_cost_usd": selected["predicted_cost_usd"],
                "adaptive_plan_id": adaptive_plan["plan_id"],
                "adaptive_plan_fingerprint": adaptive_plan["plan_fingerprint"],
                "canonical_fingerprint_recomputed": observed_hash,
                "canonical_fingerprint_valid": observed_hash == plan["plan_fingerprint"],
            }
        )

    return (
        {
            "schema_version": "m5.1-stage-d-frozen-plan-validation-v1",
            "valid": not errors,
            "errors": errors,
            "catalog_plan_count": len(catalog),
            "workloads": len(mapping),
            "quality_constrained_batches": sum(
                row["selected_plan_kind"] != "separate" for row in mapping
            ),
            "existing_adaptive_fallbacks": sum(
                row["used_existing_adaptive_fallback"] for row in mapping
            ),
            "primary_plan_id": PRIMARY_PLAN_ID,
            "primary_plan_fingerprint": PRIMARY_PLAN_FINGERPRINT,
            "provider_visible_named_questions": 3,
            "composite_binary_fusion": False,
            "simulated_from_separate_decisions": False,
        },
        sorted(mapping, key=lambda row: row["workload_id"]),
    )


def validate_partition_integrity(
    config: Mapping[str, Any], schedule: Mapping[str, Any]
) -> dict[str, Any]:
    errors: list[str] = []
    datasets: list[dict[str, Any]] = []
    for dataset, definition in config["datasets"].items():
        manifest_path, manifest = _manifest(config, dataset)
        partitions = manifest["partitions"]
        evaluation = set(partitions["evaluation"]["ids"])
        observed = {
            row["record_id"] for row in schedule["physical_requests"] if row["dataset"] == dataset
        }
        overlaps: dict[str, int] = {}
        for name, partition in partitions.items():
            if name == "evaluation":
                continue
            overlap = evaluation & set(partition["ids"])
            overlaps[f"evaluation_vs_{name}"] = len(overlap)
            if overlap:
                errors.append(f"{dataset}: evaluation overlaps {name}")
        all_ids = [record_id for value in partitions.values() for record_id in value["ids"]]
        if len(all_ids) != len(set(all_ids)):
            errors.append(f"{dataset}: duplicate identity appears across frozen partitions")
        if len(evaluation) != 400:
            errors.append(f"{dataset}: evaluation partition does not contain exactly 400 IDs")
        if observed != evaluation:
            errors.append(f"{dataset}: schedule IDs do not equal frozen evaluation IDs")
        datasets.append(
            {
                "dataset": dataset,
                "dataset_path": definition["path"],
                "dataset_file_sha256": sha256_file(definition["path"]),
                "dataset_fingerprint": manifest["dataset_fingerprint_sha256"],
                "manifest_path": str(manifest_path).replace("\\", "/"),
                "manifest_sha256": sha256_file(manifest_path),
                "evaluation_partition_fingerprint": partitions["evaluation"][
                    "fingerprint_sha256"
                ],
                "evaluation_ids": len(evaluation),
                "scheduled_ids": len(observed),
                "partition_overlap_counts": overlaps,
                "historical_records_excluded": manifest.get("historical_records_excluded"),
                "split_algorithm": manifest.get("split_algorithm"),
                "record_contents_accessed": False,
                "inference_outcomes_accessed": False,
            }
        )
    return {
        "valid": not errors,
        "errors": errors,
        "datasets": datasets,
        "held_out_record_contents_accessed": False,
        "held_out_inference_outcomes_accessed": False,
        "identity_and_fingerprint_metadata_only": True,
    }


def reconcile_schedule(
    schedule: Mapping[str, Any],
    catalog: Sequence[Mapping[str, Any]],
    selections: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], set[str]]:
    errors: list[str] = []
    requests = list(schedule["physical_requests"])
    by_plan = {str(plan["plan_id"]): plan for plan in catalog}
    request_ids = [str(row["request_id"]) for row in requests]
    keys = [
        (row["dataset"], row["record_id"], row["operator_id"], row["prompt_hash"])
        for row in requests
    ]
    if len(request_ids) != len(set(request_ids)):
        errors.append("duplicate physical request ID")
    if len(keys) != len(set(keys)):
        errors.append("duplicate physical dataset/record/operator acquisition")
    if any(row["partition"] != "evaluation" for row in requests):
        errors.append("non-evaluation row in Stage D schedule")
    if schedule.get("physical_request_count") != len(requests):
        errors.append("physical request count field mismatch")
    if schedule.get("maximum_http_attempts") != len(requests) * 3:
        errors.append("HTTP attempt ceiling is not three attempts per request")
    if schedule.get("maximum_retries") != 2:
        errors.append("maximum retry policy changed")
    if canonical_hash(_schedule_payload(schedule)) != schedule["schedule_fingerprint"]:
        errors.append("schedule fingerprint mismatch")
    for row in requests:
        expected = "m51dreq_" + canonical_hash(_request_payload(row))[:24]
        if row["request_id"] != expected:
            errors.append(f"request identity mismatch: {row['request_id']}")
            break
        if row["model"] != "jev-1.14.0" or float(row["threshold"]) != 0.5:
            errors.append(f"model/threshold mismatch: {row['request_id']}")
            break

    selected_plan_ids = {str(value) for value in schedule["selected_plan_ids"]}
    missing_plans = selected_plan_ids - set(by_plan)
    if missing_plans:
        errors.append(f"selected schedule plans absent from catalog: {sorted(missing_plans)}")
    scheduled_operators = {(row["dataset"], row["operator_id"]) for row in requests}
    expected_operators = {
        (by_plan[plan_id]["dataset"], group["operator_id"])
        for plan_id in selected_plan_ids - missing_plans
        for group in by_plan[plan_id]["groups"]
    }
    if scheduled_operators != expected_operators:
        errors.append("schedule operator union differs from selected plan operator union")
    for row in requests:
        for plan_id in row["selected_plan_references"]:
            plan = by_plan.get(str(plan_id))
            if plan is None or row["operator_id"] not in _group_operator_ids(plan):
                errors.append(f"inconsistent plan alias on {row['request_id']}")
                break

    strategy_rows = [row for row in selections if row["status"] == "selected"]
    strategy_group_record_roles = sum(len(row["groups"]) * 400 for row in strategy_rows)
    plan_group_record_roles = sum(len(row["selected_plan_references"]) for row in requests)

    adaptive = _selected_by_strategy(selections, "existing_adaptive")
    quality = _selected_by_strategy(selections, "quality_constrained_hybrid")
    full = _selected_by_strategy(selections, "full_multi_decision_batch")
    fixed_hybrid = _selected_by_strategy(selections, "fixed_selective_first_hybrid")
    primary_operator_ids = _group_operator_ids(by_plan[adaptive[PRIMARY_WORKLOAD]["plan_id"]])
    primary_operator_ids |= _group_operator_ids(by_plan[quality[PRIMARY_WORKLOAD]["plan_id"]])
    full_operator_ids = {
        next(iter(_group_operator_ids(by_plan[row["plan_id"]]))) for row in full.values()
    }
    all_multi = {
        row["operator_id"] for row in requests if row["execution_mode"] == "multi_decision"
    }
    all_single = {row["operator_id"] for row in requests if row["execution_mode"] == "single"}
    purpose_by_operator: dict[str, str] = {}
    for operator_id in all_single:
        purpose_by_operator[operator_id] = (
            "primary_confirmatory" if operator_id in primary_operator_ids else "secondary_reference"
        )
    for operator_id in full_operator_ids:
        purpose_by_operator[operator_id] = (
            "primary_confirmatory"
            if operator_id in primary_operator_ids
            else "negative_control_full_batch"
        )
    for operator_id in all_multi - full_operator_ids:
        purpose_by_operator[operator_id] = "secondary_hybrid_remainder"

    purpose_rows = []
    for purpose in (
        "primary_confirmatory",
        "secondary_reference",
        "negative_control_full_batch",
        "secondary_hybrid_remainder",
    ):
        selected = [row for row in requests if purpose_by_operator[row["operator_id"]] == purpose]
        purpose_rows.append(
            {
                "purpose": purpose,
                "operator_arms": len({row["operator_id"] for row in selected}),
                **_budget_rows(selected),
                "overlapping_scientific_uses": {
                    "primary_confirmatory": [
                        "Banking77 adaptive cost replay",
                        "three-question batch cost",
                        "individual and query-level consistency",
                    ],
                    "secondary_reference": [
                        "separate reference matrix",
                        "fallback verification",
                        "full-batch semantic comparison",
                    ],
                    "negative_control_full_batch": [
                        "eight fallback counterfactuals",
                        "batch-size and dataset heterogeneity",
                    ],
                    "secondary_hybrid_remainder": [
                        "three fixed selective-first hybrid counterfactuals not aliased "
                        "to full batches"
                    ],
                }[purpose],
            }
        )

    coverage: list[dict[str, Any]] = []
    schedule_by_operator = defaultdict(set)
    for request in requests:
        schedule_by_operator[(request["dataset"], request["operator_id"])].add(request["record_id"])
    evaluation_ids = {
        dataset: set(_manifest(config, dataset)[1]["partitions"]["evaluation"]["ids"])
        for dataset in config["datasets"]
    }
    for workload in config["workloads"]:
        workload_id = workload["id"]
        required: dict[str, list[str]] = {}
        comparisons = [
            ("adaptive_separate", adaptive[workload_id]),
            ("full_batch", full[workload_id]),
            ("quality_constrained", quality[workload_id]),
        ]
        if workload_id in fixed_hybrid:
            comparisons.append(("fixed_hybrid", fixed_hybrid[workload_id]))
        else:
            required["fixed_hybrid"] = []
        for label, selected in comparisons:
            required[label] = [
                group["operator_id"] for group in by_plan[selected["plan_id"]]["groups"]
            ]
        operators = sorted({operator for values in required.values() for operator in values})
        missing = [
            operator
            for operator in operators
            if schedule_by_operator[(workload["dataset"], operator)]
            != evaluation_ids[workload["dataset"]]
        ]
        coverage.append(
            {
                "dataset": workload["dataset"],
                "workload_id": workload_id,
                "records": len(evaluation_ids[workload["dataset"]]),
                "required_operator_ids_by_comparison": required,
                "unique_required_operator_arms": operators,
                "unique_physical_requests_for_coverage": len(operators) * 400,
                "fully_measurable": not missing,
                "missing_operator_arms": missing,
                "supports_arbitrary_catalog_plans": False,
                "scope_note": "Only frozen selected and declared control plans are covered.",
            }
        )
        if missing:
            errors.append(f"{workload_id}: incomplete record/operator coverage")

    per_dataset = []
    for dataset in config["datasets"]:
        selected = [row for row in requests if row["dataset"] == dataset]
        per_dataset.append(
            {
                "dataset": dataset,
                "records": len({row["record_id"] for row in selected}),
                "single_reference_requests": sum(
                    row["execution_mode"] == "single" for row in selected
                ),
                "batch_requests": sum(
                    row["execution_mode"] == "multi_decision" for row in selected
                ),
                "operator_arms": len({row["operator_id"] for row in selected}),
                "plan_group_record_roles": sum(
                    len(row["selected_plan_references"]) for row in selected
                ),
                **_budget_rows(selected),
            }
        )

    reconciliation = {
        "schema_version": "m5.1-stage-d-schedule-reconciliation-v1",
        "valid": not errors,
        "errors": errors,
        "schedule_fingerprint": schedule["schedule_fingerprint"],
        "physical_requests": len(requests),
        "single_reference_requests": sum(
            row["execution_mode"] == "single" for row in requests
        ),
        "multi_decision_requests": sum(
            row["execution_mode"] == "multi_decision" for row in requests
        ),
        "full_batch_requests": len(full_operator_ids) * 400,
        "additional_unique_remainder_batch_requests": len(all_multi - full_operator_ids) * 400,
        "unique_operator_arms": len(scheduled_operators),
        "unique_single_predicates": len(all_single),
        "full_batch_operator_arms": len(full_operator_ids),
        "remainder_operator_arms_not_aliased_to_full_batches": len(all_multi - full_operator_ids),
        "selected_unique_plan_ids": len(selected_plan_ids),
        "strategy_selections": len(strategy_rows),
        "unique_plan_group_record_roles": plan_group_record_roles,
        "unique_plan_aliases": plan_group_record_roles - len(requests),
        "strategy_group_record_roles_maximum": strategy_group_record_roles,
        "strategy_aliases_maximum": strategy_group_record_roles - len(requests),
        "deliberate_repetitions": 0,
        "maximum_http_attempts": schedule["maximum_http_attempts"],
        "decomposition_correction": (
            "The schedule is 6,000 singles plus 4,800 batches, not 7,200 singles plus "
            "3,600 batches. Only 15 of 18 configured predicates occur in the nine workloads; "
            "the union also includes three unique remainder-batch arms beyond the nine full "
            "batches."
        ),
        "per_dataset": per_dataset,
    }
    purpose = {
        "schema_version": "m5.1-stage-d-request-purpose-v1",
        "classification_is_exclusive_for_budget_reconciliation": True,
        "total": _budget_rows(requests),
        "purposes": purpose_rows,
        "primary_request_fraction": sum(
            row["unique_physical_requests"]
            for row in purpose_rows
            if row["purpose"] == "primary_confirmatory"
        )
        / len(requests),
        "all_measurements_have_declared_purpose": len(purpose_by_operator)
        == len({row["operator_id"] for row in requests}),
    }
    return reconciliation, purpose, coverage, primary_operator_ids


def build_reduced_option(
    schedule: Mapping[str, Any], primary_operator_ids: set[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    requests = [
        row for row in schedule["physical_requests"] if row["operator_id"] in primary_operator_ids
    ]
    reduced = {
        "schema_version": "m5.1-stage-d-reduced-confirmatory-proposal-v1",
        "parent_full_schedule_fingerprint": schedule["schedule_fingerprint"],
        "partition": "evaluation",
        "records_per_dataset": 400,
        "dataset_scope": ["banking77_queries"],
        "workload_scope": [PRIMARY_WORKLOAD],
        "model": schedule["model"],
        "cache": schedule["cache"],
        "concurrency": schedule["concurrency"],
        "maximum_retries": schedule["maximum_retries"],
        "physical_requests": requests,
        "physical_request_count": len(requests),
        "maximum_http_attempts": len(requests) * 3,
        "held_out_outcomes_accessed": False,
        "authorization_status": "separate_versioned_authorization_required",
        "authorization_phrase": None,
    }
    reduced["schedule_fingerprint"] = canonical_hash(reduced)
    budget = _budget_rows(requests)
    summary = {
        "schema_version": "m5.1-stage-d-reduced-scope-option-v1",
        "status": "alternative_not_substituted_for_frozen_full_schedule",
        "schedule_fingerprint": reduced["schedule_fingerprint"],
        **budget,
        "operator_arms": len(primary_operator_ids),
        "composition": {
            "individual_reference_requests": 1200,
            "three_question_batch_requests": 400,
        },
        "evidence_retained": [
            "paired Banking77 three-predicate adaptive-versus-batch execution cost",
            "all three individual named-decision comparisons on 400 records",
            "complete-query quality gate and positive-reference count",
            "primary profiling-inclusive economics",
        ],
        "evidence_lost": [
            "full-batch negative controls for the other eight workloads",
            "cross-dataset and two/three/four-question batch-size generalization",
            "held-out validation of the eight fallback choices",
            "three additional fixed-hybrid remainder controls",
            "held-out frozen-estimator ranking and regret across all workloads",
        ],
        "why_complete_singles_are_retained": (
            "Early rejection could avoid later calls during deployment, but the identities of "
            "later-stage records depend on unseen outcomes. Measuring all three singles supplies "
            "paired individual quality evidence and supports exact counterfactual replay without "
            "outcome-dependent schedule construction."
        ),
        "request_reduction_vs_full": schedule["physical_request_count"] - len(requests),
        "request_reduction_fraction": 1 - len(requests) / schedule["physical_request_count"],
        "recommended_automatically": False,
    }
    return reduced, summary


def build_budget(schedule: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    requests = list(schedule["physical_requests"])
    per_dataset = []
    for dataset in config["datasets"]:
        rows = [row for row in requests if row["dataset"] == dataset]
        per_dataset.append({"dataset": dataset, **_budget_rows(rows)})
    actual = _budget_rows(requests)
    stored = _read_json(SELECTION_RESULTS / "stage_d_budget_proposal.json")
    return {
        "schema_version": "m5.1-stage-d-full-scope-budget-v1",
        "calibration": (
            "Per-operator means from the completed 100-record selection matrix; no held-out "
            "text or outcome was inspected."
        ),
        "pricing": {
            "input_per_million_usd": INPUT_PRICE,
            "output_per_million_usd": OUTPUT_PRICE,
        },
        **actual,
        "per_dataset": per_dataset,
        "stored_proposal": stored,
        "reconciliation": {
            "physical_requests_match": actual["unique_physical_requests"]
            == stored["physical_requests"],
            "maximum_attempts_match": actual["maximum_http_attempts"]
            == stored["maximum_http_attempts"],
            "input_tokens_match": actual["expected_input_tokens"]
            == stored["expected_input_tokens"],
            "output_tokens_match": actual["expected_output_tokens"]
            == stored["expected_output_tokens"],
            "expected_cost_difference_usd": actual["expected_token_priced_cost_usd"]
            - stored["expected_cost_usd"],
            "conservative_cost_difference_usd": actual["conservative_no_retry_cost_proxy_usd"]
            - stored["conservative_no_retry_cost_proxy_usd"],
            "retry_proxy_difference_usd": actual["retry_inclusive_cost_proxy_usd"]
            - stored["retry_inclusive_cost_proxy_usd"],
        },
        "hard_limits": {
            "unique_physical_requests": schedule["physical_request_count"],
            "maximum_http_attempts": schedule["maximum_http_attempts"],
            "maximum_retries_per_request": schedule["maximum_retries"],
        },
        "monetary_values_are_predictions_not_runtime_caps": True,
        "historical_research_spending_excluded": True,
    }


def _historical_integrity() -> dict[str, Any]:
    selection_manifest = _read_json(SELECTION_RESULTS / "artifact_integrity_manifest.json")
    mismatches = []
    for name, expected in selection_manifest["artifact_sha256"].items():
        path = SELECTION_RESULTS / name
        if not path.exists() or sha256_file(path) != expected:
            mismatches.append(str(path).replace("\\", "/"))
    baseline = _read_json(SELECTION_PREFLIGHT / "baseline_integrity_audit.json")
    historical_mismatches = []
    for name, evidence in baseline["historical_integrity"]["files"].items():
        path = Path(name)
        if not path.exists() or sha256_file(path) != evidence["current_sha256"]:
            historical_mismatches.append(name)
    return {
        "schema_version": "m5.1-stage-d-historical-integrity-v1",
        "valid": not mismatches and not historical_mismatches,
        "selection_result_artifacts_checked": len(selection_manifest["artifact_sha256"]),
        "selection_result_artifact_mismatches": mismatches,
        "historical_sentinels_checked": len(baseline["historical_integrity"]["files"]),
        "historical_sentinel_mismatches": historical_mismatches,
        "stage_b_session_sha256": sha256_file(
            Path("results/milestone_5_1/live_batching_pilot_stage_b/session.json")
        ),
        "stage_c_manifest_sha256": sha256_file(
            Path("results/milestone_5_1/stage_c_detailed_analysis/analysis_manifest.json")
        ),
        "selection_result_manifest_sha256": sha256_file(
            SELECTION_RESULTS / "artifact_integrity_manifest.json"
        ),
        "selection_live_session_sha256": sha256_file(
            Path("results/milestone_5_1/selection_stage_live_v2/session.json")
        ),
        "historical_artifacts_modified": False,
        "new_jev_calls": 0,
        "held_out_outcomes_accessed": False,
    }


def current_stage_d_scope_fingerprints() -> dict[str, Any]:
    config = load_yaml(WORKLOADS_PATH)
    datasets = {}
    for dataset in config["datasets"]:
        manifest_path, manifest = _manifest(config, dataset)
        datasets[dataset] = {
            "dataset_fingerprint": manifest["dataset_fingerprint_sha256"],
            "evaluation_partition_fingerprint": manifest["partitions"]["evaluation"][
                "fingerprint_sha256"
            ],
            "manifest_sha256": sha256_file(manifest_path),
            "dataset_file_sha256": sha256_file(config["datasets"][dataset]["path"]),
        }
    fingerprints = {
        "workloads_sha256": sha256_file(WORKLOADS_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "optimizer_sha256": sha256_file(OPTIMIZER_PATH),
        "stage_d_schedule_sha256": sha256_file(SCHEDULE_PATH),
        "frozen_catalog_sha256": sha256_file(CATALOG_PATH),
        "frozen_selections_sha256": sha256_file(SELECTIONS_PATH),
        "runner_sha256": sha256_file(RUNNER_PATH) if RUNNER_PATH.exists() else None,
        "analyzer_sha256": sha256_file(ANALYZER_PATH) if ANALYZER_PATH.exists() else None,
        "datasets": datasets,
    }
    return fingerprints


def _authorization_scope(
    schedule: Mapping[str, Any], authorization_phrase: str
) -> dict[str, Any]:
    return {
        "schema_version": "m5.1-stage-d-final-authorization-proposal-v1",
        "authorization_phrase": authorization_phrase,
        "schedule_fingerprint": schedule["schedule_fingerprint"],
        "physical_request_ceiling": schedule["physical_request_count"],
        "http_attempt_ceiling": schedule["maximum_http_attempts"],
        "maximum_retries": schedule["maximum_retries"],
        "model": schedule["model"],
        "cache": schedule["cache"],
        "concurrency": schedule["concurrency"],
        "partition": schedule["partition"],
        "selected_plan_fingerprints": _read_json(AUTHORIZATION_PATH)[
            "selected_plan_fingerprints"
        ],
        "scope_fingerprints": current_stage_d_scope_fingerprints(),
        "provider_must_not_initialize_before_validation": True,
        "output_directory": "results/milestone_5_1/stage_d_live",
        "resume_policy": (
            "no_implicit_resume; interrupted sessions require reconciliation and versioned "
            "reauthorization"
        ),
        "stage_d_executed": False,
    }


def validate_stage_d_launch(
    manifest: Mapping[str, Any],
    *,
    authorization: str | None,
    schedule: Mapping[str, Any],
    catalog: Sequence[Mapping[str, Any]],
    selections: Sequence[Mapping[str, Any]],
    output_directory: str | Path,
) -> None:
    if authorization != manifest.get("authorization_phrase"):
        raise PermissionError("exact Stage D authorization phrase required")
    if manifest.get("paid_calls_authorized") is not True:
        raise PermissionError("Stage D manifest remains an unauthorised proposal")
    scope = {
        key: value
        for key, value in manifest.items()
        if key not in {"paid_calls_authorized", "scope_fingerprint"}
    }
    if manifest.get("scope_fingerprint") != canonical_hash(scope):
        raise ValueError("authorization scope fingerprint mismatch")
    if manifest.get("scope_fingerprints") != current_stage_d_scope_fingerprints():
        raise ValueError("source, runner, dataset, plan, or analysis fingerprint mismatch")
    if manifest.get("schedule_fingerprint") != schedule.get("schedule_fingerprint"):
        raise ValueError("schedule differs from authorization scope")
    if canonical_hash(_schedule_payload(schedule)) != schedule.get("schedule_fingerprint"):
        raise ValueError("schedule fingerprint is internally invalid")
    if manifest.get("physical_request_ceiling") != len(schedule["physical_requests"]):
        raise ValueError("physical request ceiling mismatch")
    if manifest.get("http_attempt_ceiling") != len(schedule["physical_requests"]) * 3:
        raise ValueError("HTTP attempt ceiling mismatch")
    if (manifest.get("model"), manifest.get("cache"), manifest.get("concurrency")) != (
        "jev-1.14.0",
        "disabled",
        4,
    ):
        raise ValueError("frozen provider settings mismatch")
    if manifest.get("maximum_retries") != 2 or schedule.get("maximum_retries") != 2:
        raise ValueError("frozen retry policy mismatch")
    if str(Path(output_directory)).replace("\\", "/") != manifest.get("output_directory"):
        raise ValueError("output directory differs from authorization scope")
    plan_by_id = {str(plan["plan_id"]): plan for plan in catalog}
    expected_plan_fingerprints = {
        plan_id: plan_by_id[plan_id]["plan_fingerprint"]
        for plan_id in schedule["selected_plan_ids"]
        if plan_id in plan_by_id
    }
    if len(expected_plan_fingerprints) != len(schedule["selected_plan_ids"]):
        raise ValueError("scheduled physical plan is absent from frozen catalog")
    if any(
        canonical_hash(_plan_execution_payload(plan_by_id[plan_id])) != fingerprint
        for plan_id, fingerprint in expected_plan_fingerprints.items()
    ):
        raise ValueError("canonical fingerprint failure in scheduled physical-plan union")
    if manifest.get("selected_plan_fingerprints") != expected_plan_fingerprints:
        raise ValueError("authorized physical-plan fingerprints changed")
    config = load_yaml(WORKLOADS_PATH)
    plan_validation, _ = validate_frozen_plans(catalog, selections, config)
    if not plan_validation["valid"]:
        raise ValueError("frozen plan validation failed")
    partition = validate_partition_integrity(config, schedule)
    if not partition["valid"]:
        raise ValueError("held-out partition identity validation failed")
    reconciliation, _, coverage, _ = reconcile_schedule(schedule, catalog, selections, config)
    if not reconciliation["valid"] or not all(row["fully_measurable"] for row in coverage):
        raise ValueError("Stage D schedule or alias coverage validation failed")
    if Path(output_directory).exists():
        raise FileExistsError("refusing overwrite or implicit Stage D resume")


def _protocol_artifacts() -> dict[str, Any]:
    gate = load_yaml(PROTOCOL_PATH)["quality_gate"]
    primary = {
        "schema_version": "m5.1-stage-d-primary-comparison-protocol-v1",
        "workload_id": PRIMARY_WORKLOAD,
        "baseline": "frozen existing_adaptive separate execution",
        "candidate_plan_id": PRIMARY_PLAN_ID,
        "candidate_plan_fingerprint": PRIMARY_PLAN_FINGERPRINT,
        "records": 400,
        "paired_unit": "held-out record",
        "primary_effect": "adaptive input-token-priced execution cost minus batch cost",
        "positive_value_means": "batching saved execution cost",
        "pricing": {"input_per_million_usd": INPUT_PRICE, "output_per_million_usd": OUTPUT_PRICE},
        "statistics": {
            "paired_bootstrap_resamples": 10000,
            "paired_bootstrap_seed": 51103,
            "confidence_level": 0.95,
            "resampling_unit": "record",
        },
        "outcome_classification": {
            "confirmed_economic_and_quality_success": (
                "paired 95% cost-saving interval lower bound is above zero and unchanged "
                "quality gate passes"
            ),
            "economic_failure": "paired 95% cost-saving interval upper bound is at or below zero",
            "quality_failure": "any frozen quality-gate requirement fails",
            "inconclusive_cost": "cost interval includes zero",
            "insufficient_quality_support": "fewer than three positive separate-reference queries",
        },
        "semantic_reference": "separate Jev execution, not human ground truth",
        "no_post_hoc_substitution": True,
    }
    quality = {
        "schema_version": "m5.1-stage-d-quality-analysis-protocol-v1",
        "gate": gate,
        "metrics": [
            "agreement numerator and denominator",
            "query false positives and false negatives",
            "precision, recall, F1, MCC",
            "changed-decision rate",
            "match-set Jaccard",
            "positive-reference count",
            "individual probability differences and threshold crossings",
        ],
        "status_values": [
            "passed",
            "failed",
            "insufficient_positive_support",
            "insufficient_observations",
            "undefined_metric",
            "missing_physical_group_evidence",
            "missing_complete_plan_evidence",
            "provider_failure",
        ],
        "zero_positive_references_do_not_validate_recall": True,
        "human_ground_truth_available": False,
    }
    fallback = {
        "schema_version": "m5.1-stage-d-fallback-evaluation-protocol-v1",
        "workloads": 8,
        "comparisons": [
            "full batch versus adaptive separate execution",
            "quality gate versus full-batch held-out consistency",
            "selection fallback correctness",
            "harmful batching avoided",
            "beneficial admissible batching forgone",
        ],
        "classification": {
            "correct_fallback": "batch is costlier, fails quality, or lacks support",
            "excessively_conservative": "batch is cheaper and independently passes quality",
            "inconclusive": "paired cost interval crosses zero or quality support is insufficient",
        },
        "unmeasured_candidates_are_not_classified": True,
    }
    profiling = {
        "schema_version": "m5.1-stage-d-profiling-economics-protocol-v1",
        "reuse_horizons": [1, 5, 10, 25, 50, 100, 250, 1000],
        "formula": "R*(C_existing_adaptive-C_candidate)-C_incremental_profiling-C_optimization",
        "baseline_profiling_cost_usd": 0.0,
        "baseline_reason": "existing_adaptive declined profiling at frozen ten-execution horizon",
        "selection_forecast": {
            "execution_saving_per_100_records_usd": 0.000130494,
            "standalone_profile_cost_usd": 0.007067130,
            "standalone_break_even_executions": 55,
            "shared_profile_cost_usd": 0.002967972,
            "shared_break_even_executions": 23,
            "warm_start_profile_cost_usd": 0.0,
            "warm_start_break_even": "immediate",
        },
        "scenarios": [
            "standalone_cold_start",
            "shared_statistics_cold_start",
            "warm_start",
            "actual_research_expenditure",
        ],
        "held_out_savings_replace_forecast_only_after_authorized_execution": True,
        "unknown_optimization_cost_reported_unavailable": True,
        "historical_failed_transport_provider_billing_status": "unknown",
    }
    statistics_protocol = {
        "schema_version": "m5.1-stage-d-statistical-analysis-protocol-v1",
        "primary": primary["statistics"],
        "quality_intervals": "Wilson score 95% with numerator and denominator",
        "secondary_analyses": [
            "full batching by workload and dataset",
            "two-, three-, and four-question batch economics",
            "early-rejection rates and stage reach",
            "cost-prediction error and measured-set regret",
            "fallback correctness and profiling economics",
        ],
        "dependence": (
            "Workloads share records and predicates; predicate outcomes within requests and "
            "workloads are not treated as independent trials."
        ),
        "dataset_count_limitation": 3,
        "workload_count_limitation": 9,
        "global_optimality_claim": False,
        "negative_result_rules": {
            "costly_batching": "report point estimate and interval without substitution",
            "semantic_inconsistency": "report gate violation and preserve frozen plan result",
            "inconclusive_saving": "do not call positive point estimate confirmed",
            "sparse_positives": "classify insufficient support",
            "net_negative_profiling": "report by scenario and reuse horizon",
            "conservative_fallback": "report independently admissible forgone opportunity",
        },
    }
    return {
        "primary_comparison_protocol.json": primary,
        "quality_analysis_protocol.json": quality,
        "fallback_evaluation_protocol.json": fallback,
        "profiling_economics_protocol.json": profiling,
        "statistical_analysis_protocol.json": statistics_protocol,
    }


def generate(output: Path = ROOT) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Stage D final preflight: {output}")
    output.mkdir(parents=True)
    schedule = _read_json(SCHEDULE_PATH)
    catalog = _read_json(CATALOG_PATH)["plans"]
    selections = _read_json(SELECTIONS_PATH)["strategies"]
    config = load_yaml(WORKLOADS_PATH)
    frozen_validation, mapping = validate_frozen_plans(catalog, selections, config)
    partitions = validate_partition_integrity(config, schedule)
    reconciliation, purpose, coverage, primary_operators = reconcile_schedule(
        schedule, catalog, selections, config
    )
    reduced_schedule, reduced = build_reduced_option(schedule, primary_operators)
    budget = build_budget(schedule, config)
    historical = _historical_integrity()
    original_authorization = _read_json(AUTHORIZATION_PATH)

    scope = _authorization_scope(schedule, original_authorization["authorization_phrase"])
    authorization_proposal = {
        **scope,
        "scope_fingerprint": canonical_hash(scope),
        "paid_calls_authorized": False,
    }
    safety = {
        "schema_version": "m5.1-stage-d-execution-safety-audit-v1",
        "valid": all(
            (
                frozen_validation["valid"],
                partitions["valid"],
                reconciliation["valid"],
                historical["valid"],
            )
        ),
        "provider_import_after_launch_validation": True,
        "provider_initialized_during_preflight": False,
        "authorization_exact_match_required": True,
        "paid_manifest_required": True,
        "schedule_fingerprint_required": True,
        "source_and_dataset_fingerprints_required": True,
        "prompt_and_plan_hash_validation_required": True,
        "model": "jev-1.14.0",
        "threshold": 0.5,
        "cache": "disabled",
        "concurrency": 4,
        "maximum_retries": 2,
        "physical_request_ceiling": 10800,
        "http_attempt_ceiling": 32400,
        "atomic_worst_case_attempt_reservation": True,
        "smoke_requests_are_members_of_frozen_schedule": True,
        "strict_named_probability_validation": True,
        "duplicate_request_identity_rejected": True,
        "output_conflict_rejected": True,
        "resume_permitted": False,
        "restart_rule": (
            "An incomplete session is fail-closed. Reconcile all attempted request identities and "
            "obtain a versioned authorization amendment before any restart."
        ),
        "ambiguous_provider_receipt_rule": "charge attempt and do not silently reissue",
        "selection_failed_transport_attempts_preserved": 3,
        "real_provider_calls_in_preflight": 0,
    }
    analysis_validation = {
        "schema_version": "m5.1-stage-d-post-live-analysis-validation-v1",
        "analyzer_path": str(ANALYZER_PATH).replace("\\", "/"),
        "analyzer_sha256": sha256_file(ANALYZER_PATH),
        "required_outputs": [
            "request_reconciliation.json",
            "provider_expenditure.json",
            "per_dataset_results.csv",
            "per_workload_results.csv",
            "adaptive_vs_batch.csv",
            "batch_size_economics.csv",
            "semantic_agreement.json",
            "query_level_agreement.csv",
            "quality_gate_results.json",
            "frozen_plan_regret.json",
            "cost_estimator_errors.csv",
            "fallback_correctness.json",
            "profiling_economics.csv",
            "statistical_uncertainty.json",
            "negative_result_analysis.json",
        ],
        "missing_measurement_policy": "unavailable with reason; never zero-filled",
        "dry_run_uses_mock_evidence_only": True,
        "held_out_outcomes_loaded_during_preflight": False,
    }
    readiness = {
        "schema_version": "m5.1-stage-d-authorization-readiness-v1",
        "ready": safety["valid"],
        "recommended_scope": "original_full_frozen_stage_d",
        "schedule_path": str(SCHEDULE_PATH).replace("\\", "/"),
        "schedule_fingerprint": schedule["schedule_fingerprint"],
        "authorization_manifest_path": str(output / "stage_d_authorization_manifest.json").replace(
            "\\", "/"
        ),
        "authorization_manifest_status": "proposal_not_authorized",
        "future_countersigned_manifest_path": (
            "results/milestone_5_1/stage_d_final_preflight/"
            "stage_d_authorization_manifest_authorized.json"
        ),
        "required_authorization_text": original_authorization["authorization_phrase"],
        "validation_command": (
            ".\\.venv\\Scripts\\python.exe -m experiments.milestone_5_1.run_stage_d validate "
            "--manifest results/milestone_5_1/stage_d_final_preflight/"
            "stage_d_authorization_manifest_authorized.json --schedule results/milestone_5_1/"
            "selection_stage_results/stage_d_execution_schedule.json --output "
            "results/milestone_5_1/stage_d_live --authorization \""
            f"{original_authorization['authorization_phrase']}\""
        ),
        "execution_command": (
            ".\\.venv\\Scripts\\python.exe -m experiments.milestone_5_1.run_stage_d execute "
            "--manifest results/milestone_5_1/stage_d_final_preflight/"
            "stage_d_authorization_manifest_authorized.json --schedule results/milestone_5_1/"
            "selection_stage_results/stage_d_execution_schedule.json --output "
            "results/milestone_5_1/stage_d_live --authorization \""
            f"{original_authorization['authorization_phrase']}\""
        ),
        "reduced_scope_requires_new_protocol_and_separate_authorization": True,
        "stage_d_executed": False,
        "held_out_outcomes_accessed": False,
    }

    artifacts: dict[str, Any] = {
        "historical_integrity_audit.json": historical,
        "frozen_plan_validation.json": frozen_validation,
        "workload_strategy_mapping.json": {
            "schema_version": "m5.1-stage-d-workload-strategy-mapping-v1",
            "workloads": mapping,
        },
        "schedule_reconciliation.json": reconciliation,
        "request_purpose_breakdown.json": purpose,
        "measurement_coverage_matrix.json": {
            "schema_version": "m5.1-stage-d-measurement-coverage-v1",
            "workloads": coverage,
            "all_declared_comparisons_measurable": all(
                row["fully_measurable"] for row in coverage
            ),
            "unsupported_scope": [
                "arbitrary unselected catalog plans",
                "conjunctive-binary fusion without a separately scheduled operator",
                "provider repeatability because there are no deliberate repeated calls",
            ],
        },
        "full_scope_budget.json": budget,
        "reduced_scope_option.json": reduced,
        "reduced_scope_schedule.json": reduced_schedule,
        "execution_safety_audit.json": safety,
        "post_live_analysis_validation.json": analysis_validation,
        "authorization_readiness.json": readiness,
        "stage_d_authorization_manifest.json": authorization_proposal,
        **_protocol_artifacts(),
    }
    for name, value in artifacts.items():
        _write_json(output / name, value)

    integrity = {
        "schema_version": "m5.1-stage-d-final-preflight-integrity-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "preflight_version": PREFLIGHT_VERSION,
        "preflight_source": str(Path(__file__)).replace("\\", "/"),
        "preflight_source_sha256": sha256_file(__file__),
        "runner_sha256": sha256_file(RUNNER_PATH),
        "analyzer_sha256": sha256_file(ANALYZER_PATH),
        "input_fingerprints": current_stage_d_scope_fingerprints(),
        "artifact_sha256": {
            name: sha256_file(output / name) for name in sorted(artifacts)
        },
        "new_jev_calls": 0,
        "held_out_outcomes_accessed": False,
        "stage_d_executed": False,
    }
    _write_json(output / "artifact_integrity_manifest.json", integrity)
    return {
        "valid": safety["valid"],
        "physical_requests": reconciliation["physical_requests"],
        "expected_cost_usd": budget["expected_token_priced_cost_usd"],
        "reduced_requests": reduced["unique_physical_requests"],
        "schedule_fingerprint": schedule["schedule_fingerprint"],
        "new_jev_calls": 0,
        "held_out_outcomes_accessed": False,
    }
