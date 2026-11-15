from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from semanticflow.optimizer.fusion import FusedOperator, build_fused_operator
from semanticflow.optimizer.fusion_experiment import sha256_file
from semanticflow.optimizer.fusion_planner import build_physical_plan
from semanticflow.optimizer.hybrid import HYBRID_PLANNER_VERSION
from semanticflow.optimizer.m51_experiment import load_yaml
from semanticflow.query.models import PredicateSpec
from semanticflow.storage.results import git_commit, git_worktree_dirty

PREFLIGHT_SCHEMA = "m5.1-selection-preflight-v1"
PLANNER_VERSION = "m5.1-selection-measured-candidates-v1"
ESTIMATOR_VERSION = "m5.1-correlation-aware-prefix-v1"
SCHEDULE_VERSION = "m5.1-selection-schedule-v1"
AUTHORIZATION_PHRASE = "AUTHORIZE M5.1 SELECTION STAGE FROZEN MEASUREMENT MATRIX"
INPUT_PRICE = 0.042
OUTPUT_PRICE = 0.0

ROOT = Path("results/milestone_5_1/selection_stage_preflight")
WORKLOADS = Path("configs/milestone_5_1/workloads.yaml")
PROTOCOL = Path("configs/milestone_5_1/protocol.yaml")
OPTIMIZER = Path("configs/optimizer.milestone_4.yaml")
SELECTION_CONFIG = Path("configs/milestone_5_1/selection_stage.yaml")
SELECTION_RUNNER = Path("experiments/milestone_5_1/run_selection.py")
STAGE_B = Path("results/milestone_5_1/live_batching_pilot_stage_b")
STAGE_C = Path("results/milestone_5_1/stage_c_detailed_analysis")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _predicates(config: Mapping[str, Any], dataset: str) -> dict[str, PredicateSpec]:
    return {
        predicate_id: PredicateSpec(id=predicate_id, instructions=instructions)
        for predicate_id, instructions in config["datasets"][dataset]["predicates"].items()
    }


def _manifest(config: Mapping[str, Any], dataset: str) -> dict[str, Any]:
    path = Path(str(config["datasets"][dataset]["path"]) + ".manifest.json")
    return json.loads(path.read_text("utf-8"))


def _operator(
    config: Mapping[str, Any], dataset: str, predicate_ids: Sequence[str], mode: str
) -> FusedOperator:
    predicates = _predicates(config, dataset)
    return build_fused_operator(
        [predicates[predicate_id] for predicate_id in predicate_ids],
        mode=mode,  # type: ignore[arg-type]
        requested_model=str(config["requested_model"]),
        threshold=float(config["threshold"]),
        execution_parameters={"cache": "disabled", "temperature": "provider_default"},
    )


def _operator_payload(operator: FusedOperator) -> dict[str, Any]:
    questions = operator.provider_questions()
    return {
        "operator_id": operator.group_id,
        "execution_mode": operator.fusion_mode,
        "predicate_ids": list(operator.predicate_ids),
        "question_names": list(questions),
        "provider_questions": questions,
        "prompt_version": operator.prompt_template_version,
        "prompt_hash": operator.fusion_prompt_hash,
        "predicate_prompt_hashes": dict(operator.predicate_hashes),
        "requested_model": operator.requested_model,
        "expected_response_schema": operator.result_schema_version,
        "threshold": operator.threshold,
        "inference_settings": dict(operator.execution_parameters),
    }


def selected_remainder_groups() -> dict[str, dict[str, Any]]:
    """Frozen using Stage B evidence only; never selection outcomes."""
    return {
        "m51_beauty_reaction_packaging_refund": {
            "first": "beauty_adverse_reaction",
            "remainder": ["beauty_packaging_failure", "beauty_return_or_refund"],
            "basis": "original-first deterministic fallback; no compatible Stage B matrix",
        },
        "m51_beauty_reaction_failure_value_return": {
            "first": "beauty_adverse_reaction",
            "remainder": [
                "beauty_malfunction",
                "beauty_poor_value",
                "beauty_return_or_refund",
            ],
            "basis": "Stage B measured candidate; tied original separate in replay",
        },
        "m51_banking_card_unauthorized_refund": {
            "first": "banking_card_related",
            "remainder": [
                "banking_unauthorized_activity",
                "banking_refund_or_reversal",
            ],
            "basis": "original-first deterministic fallback; no compatible Stage B matrix",
        },
        "m51_banking_withdrawal_unauthorized_refund_access": {
            "first": "banking_access_or_verification",
            "remainder": [
                "banking_unauthorized_activity",
                "banking_refund_or_reversal",
                "banking_cash_withdrawal",
            ],
            "basis": "lowest-cost fully measured Stage B selective-first candidate",
        },
        "m51_sms_prize_urgency_contact": {
            "first": "sms_urgency_pressure",
            "remainder": ["sms_prize_or_reward", "sms_contact_action"],
            "basis": "lowest-cost fully measured Stage B selective-first candidate",
        },
        "m51_sms_security_prize_urgency_contact": {
            "first": "sms_account_security",
            "remainder": [
                "sms_prize_or_reward",
                "sms_urgency_pressure",
                "sms_contact_action",
            ],
            "basis": "Stage B measured candidate; tied original separate in replay",
        },
    }


def build_inventory(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    stage_c = json.loads((STAGE_C / "plan_ranking_analysis.json").read_text("utf-8"))
    supported = {row["workload_id"] for row in stage_c["prediction_metrics"]}
    remainder = selected_remainder_groups()
    rows = []
    for workload in config["workloads"]:
        dataset = workload["dataset"]
        operators = {
            predicate_id: _operator(config, dataset, [predicate_id], "single")
            for predicate_id in workload["predicate_ids"]
        }
        rows.append(
            {
                "dataset": dataset,
                "workload_id": workload["id"],
                "logical_predicate_ids": workload["predicate_ids"],
                "predicate_count": len(workload["predicate_ids"]),
                "original_predicate_order": workload["original_order"],
                "predicate_prompt_hashes": {
                    predicate_id: operators[predicate_id].predicate_hashes[predicate_id]
                    for predicate_id in workload["predicate_ids"]
                },
                "stage_b_compatible_complete_matrix": workload["id"] in supported,
                "stage_b_separate_measurements": (
                    30 if workload["id"] in supported else "incomplete"
                ),
                "stage_b_batch_measurements": (
                    30 if workload["id"] in supported else "missing compatible group"
                ),
                "selected_remainder_group": remainder.get(workload["id"]),
                "selection_stage_records": 100,
                "selection_stage_coverage": (
                    "all singles + full batch + selected remainder"
                    if workload["id"] in remainder
                    else "all singles + full batch"
                ),
            }
        )
    return rows


def _acquisition_key(dataset: str, record_id: str, operator: FusedOperator) -> str:
    return canonical_hash(
        {
            "dataset": dataset,
            "record_id": record_id,
            "partition": "selection",
            "operator": operator.canonical_dict(),
        }
    )


def build_schedule(config: Mapping[str, Any]) -> dict[str, Any]:
    roles: list[dict[str, Any]] = []
    physical: dict[str, dict[str, Any]] = {}
    group_roles: dict[tuple[str, str, str], set[str]] = defaultdict(set)

    def add_role(
        *,
        dataset: str,
        record_id: str,
        arm: str,
        workload_id: str | None,
        operator: FusedOperator,
        role_suffix: str,
    ) -> None:
        key = _acquisition_key(dataset, record_id, operator)
        request_id = f"m51selreq_{key[:24]}"
        role_id = (
            "m51selrole_"
            + canonical_hash([arm, workload_id, record_id, operator.group_id, role_suffix])[:24]
        )
        payload = _operator_payload(operator)
        role = {
            "role_id": role_id,
            "request_id": request_id,
            "dataset": dataset,
            "record_id": record_id,
            "partition": "selection",
            "measurement_arm": arm,
            "workload_id": workload_id,
            "predicate_ids": payload["predicate_ids"],
            "execution_mode": payload["execution_mode"],
            "prompt_hash": payload["prompt_hash"],
            "physical_acquisition_shared": request_id in physical,
        }
        roles.append(role)
        group_roles[(dataset, operator.group_id, arm)].add(workload_id or "shared")
        if request_id not in physical:
            physical[request_id] = {
                "request_id": request_id,
                "dataset": dataset,
                "record_id": record_id,
                "partition": "selection",
                **payload,
                "cost_estimation_metadata": {
                    "calibration_source": "M5.1 Stage B jev-1.14.0 traces",
                    "price_input_per_million_usd": INPUT_PRICE,
                    "price_output_per_million_usd": OUTPUT_PRICE,
                },
                "logical_roles": [],
            }
        physical[request_id]["logical_roles"].append(role_id)

    for dataset, definition in config["datasets"].items():
        ids = _manifest(config, dataset)["partitions"]["selection"]["ids"]
        if len(ids) != 100:
            raise ValueError(f"{dataset} selection partition must contain exactly 100 IDs")
        for predicate_id in definition["predicates"]:
            operator = _operator(config, dataset, [predicate_id], "single")
            consumers = [
                workload["id"]
                for workload in config["workloads"]
                if workload["dataset"] == dataset and predicate_id in workload["predicate_ids"]
            ]
            for record_id in ids:
                add_role(
                    dataset=dataset,
                    record_id=record_id,
                    arm="individual_reference",
                    workload_id=None,
                    operator=operator,
                    role_suffix="|".join(sorted(consumers)),
                )
                physical[next(reversed(physical))]["compatible_workload_ids"] = consumers
        for workload in [w for w in config["workloads"] if w["dataset"] == dataset]:
            operator = _operator(config, dataset, workload["predicate_ids"], "multi_decision")
            for record_id in ids:
                add_role(
                    dataset=dataset,
                    record_id=record_id,
                    arm="full_batch",
                    workload_id=workload["id"],
                    operator=operator,
                    role_suffix="full",
                )
        for workload_id, choice in selected_remainder_groups().items():
            workload = next(w for w in config["workloads"] if w["id"] == workload_id)
            if workload["dataset"] != dataset:
                continue
            operator = _operator(config, dataset, choice["remainder"], "multi_decision")
            for record_id in ids:
                add_role(
                    dataset=dataset,
                    record_id=record_id,
                    arm="remainder_batch",
                    workload_id=workload_id,
                    operator=operator,
                    role_suffix="remainder",
                )

    counts = {
        "individual_reference_roles": sum(
            r["measurement_arm"] == "individual_reference" for r in roles
        ),
        "full_batch_roles": sum(r["measurement_arm"] == "full_batch" for r in roles),
        "remainder_batch_roles": sum(r["measurement_arm"] == "remainder_batch" for r in roles),
        "logical_measurement_roles": len(roles),
        "unique_physical_requests": len(physical),
        "shared_alias_roles": len(roles) - len(physical),
        "proposal_retry_inclusive_attempt_ceiling": len(roles) * 3,
        "deduplicated_retry_inclusive_attempt_ceiling": len(physical) * 3,
    }
    schedule = {
        "schema_version": SCHEDULE_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "model": config["requested_model"],
        "cache": "disabled",
        "maximum_retries": 2,
        "concurrency": 4,
        "partition": "selection",
        "counts": counts,
        "measurement_roles": roles,
        "physical_requests": sorted(physical.values(), key=lambda x: x["request_id"]),
        "duplicate_policy": (
            "exactly compatible full/remainder roles share one provider request; "
            "logical roles remain explicit"
        ),
        "smoke_checkpoint": [],
        "held_out_record_contents_accessed": False,
    }
    by_size: dict[int, dict[str, Any]] = {}
    for row in schedule["physical_requests"]:
        size = len(row["predicate_ids"])
        if size not in by_size:
            by_size[size] = row
    schedule["smoke_checkpoint"] = [by_size[size]["request_id"] for size in sorted(by_size)]
    schedule["schedule_fingerprint"] = canonical_hash(
        {k: v for k, v in schedule.items() if k not in {"created_at", "schedule_fingerprint"}}
    )
    return schedule


def _plan_envelope(
    config: Mapping[str, Any],
    workload: Mapping[str, Any],
    groups: Sequence[FusedOperator],
    strategy_eligibility: Sequence[str],
    configuration_fingerprint: str,
) -> dict[str, Any]:
    plan = build_physical_plan(
        workload_id=workload["id"],
        logical_query_id=workload["id"],
        logical_predicate_ids=workload["predicate_ids"],
        groups=groups,
        configuration_fingerprint=configuration_fingerprint,
    ).model_copy(update={"planner_version": HYBRID_PLANNER_VERSION})
    execution = {
        "dataset": workload["dataset"],
        "workload_id": workload["id"],
        "logical_predicate_ids": workload["predicate_ids"],
        "groups": [_operator_payload(group) for group in groups],
        "model": config["requested_model"],
        "threshold": config["threshold"],
        "execution_settings": {"cache": "disabled", "concurrency": 4, "max_retries": 2},
        "planner_version": PLANNER_VERSION,
        "configuration_fingerprint": configuration_fingerprint,
    }
    fingerprint = canonical_hash(execution)
    return {
        "plan_id": f"m51selplan_{fingerprint[:24]}",
        "plan_fingerprint": fingerprint,
        "base_plan_id": plan.plan_id,
        **execution,
        "optimizer_strategy_eligibility": list(strategy_eligibility),
        "cost_prediction": {"status": "pending selection evidence", "value_usd": None},
        "quality_evidence_status": "pending selection evidence",
        "profiling_evidence_references": [],
        "fallback_strategy": "existing_adaptive",
    }


def build_plan_catalog(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    config_fingerprint = canonical_hash(
        {
            "workloads": sha256_file(WORKLOADS),
            "protocol": sha256_file(PROTOCOL),
            "optimizer": sha256_file(OPTIMIZER),
            "planner_version": PLANNER_VERSION,
        }
    )
    remainder = selected_remainder_groups()
    catalog: dict[str, dict[str, Any]] = {}
    for workload in config["workloads"]:
        dataset = workload["dataset"]
        for order in itertools.permutations(workload["predicate_ids"]):
            groups = [
                _operator(config, dataset, [predicate_id], "single") for predicate_id in order
            ]
            strategies = [
                "existing_adaptive",
                "cost_only_hybrid",
                "quality_constrained_hybrid",
                "profiling_aware_hybrid",
            ]
            if list(order) == workload["original_order"]:
                strategies.append("fixed_separate")
            plan = _plan_envelope(
                config,
                workload,
                groups,
                strategies,
                config_fingerprint,
            )
            catalog[plan["plan_id"]] = plan
        full = _plan_envelope(
            config,
            workload,
            [_operator(config, dataset, workload["predicate_ids"], "multi_decision")],
            [
                "full_multi_decision_batch",
                "cost_only_hybrid",
                "quality_constrained_hybrid",
                "profiling_aware_hybrid",
            ],
            config_fingerprint,
        )
        catalog[full["plan_id"]] = full
        if workload["id"] in remainder:
            choice = remainder[workload["id"]]
            hybrid = _plan_envelope(
                config,
                workload,
                [
                    _operator(config, dataset, [choice["first"]], "single"),
                    _operator(config, dataset, choice["remainder"], "multi_decision"),
                ],
                [
                    "fixed_hybrid",
                    "cost_only_hybrid",
                    "quality_constrained_hybrid",
                    "profiling_aware_hybrid",
                ],
                config_fingerprint,
            )
            hybrid["remainder_selection_basis"] = choice["basis"]
            catalog[hybrid["plan_id"]] = hybrid
    return sorted(catalog.values(), key=lambda row: (row["workload_id"], row["plan_id"]))


def build_plan_requirements(
    catalog: Sequence[Mapping[str, Any]], schedule: Mapping[str, Any]
) -> list[dict[str, Any]]:
    available = {
        (row["dataset"], row["record_id"], row["prompt_hash"], row["execution_mode"])
        for row in schedule["physical_requests"]
    }
    records = defaultdict(set)
    for row in schedule["physical_requests"]:
        records[row["dataset"]].add(row["record_id"])
    output = []
    for plan in catalog:
        required = []
        missing = []
        for group in plan["groups"]:
            required.append(group["operator_id"])
            for record_id in records[plan["dataset"]]:
                key = (plan["dataset"], record_id, group["prompt_hash"], group["execution_mode"])
                if key not in available:
                    missing.append({"record_id": record_id, "operator_id": group["operator_id"]})
        output.append(
            {
                "plan_id": plan["plan_id"],
                "workload_id": plan["workload_id"],
                "required_operator_ids": required,
                "record_count": len(records[plan["dataset"]]),
                "evidence_status": "fully_scheduled" if not missing else "unsupported",
                "missing_measurements": missing,
                "shared_measurements_permitted": True,
                "zero_reach_does_not_waive_measurement": True,
            }
        )
    return output


def validate_schedule(
    config: Mapping[str, Any],
    schedule: Mapping[str, Any],
    catalog: Sequence[Mapping[str, Any]],
    requirements: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    counts = schedule["counts"]
    expected = {
        "individual_reference_roles": 1800,
        "full_batch_roles": 900,
        "remainder_batch_roles": 600,
        "logical_measurement_roles": 3300,
    }
    for key, value in expected.items():
        if counts.get(key) != value:
            errors.append(f"{key}: expected {value}, observed {counts.get(key)}")
    if counts.get("unique_physical_requests") != 3000:
        errors.append("expected 3,000 unique physical acquisitions after exact reuse")
    role_ids = [row["role_id"] for row in schedule["measurement_roles"]]
    request_ids = [row["request_id"] for row in schedule["physical_requests"]]
    if len(role_ids) != len(set(role_ids)):
        errors.append("duplicate logical role identity")
    if len(request_ids) != len(set(request_ids)):
        errors.append("duplicate physical request identity")
    physical = {row["request_id"]: row for row in schedule["physical_requests"]}
    for role in schedule["measurement_roles"]:
        request = physical.get(role["request_id"])
        if request is None:
            errors.append(f"missing acquisition for {role['role_id']}")
            continue
        for key in ("dataset", "record_id", "partition", "execution_mode", "prompt_hash"):
            if role[key] != request[key]:
                errors.append(f"incompatible {key} for {role['role_id']}")
        if role["predicate_ids"] != request["predicate_ids"]:
            errors.append(f"incompatible predicate set for {role['role_id']}")
    for dataset in config["datasets"]:
        allowed = set(_manifest(config, dataset)["partitions"]["selection"]["ids"])
        observed = {
            row["record_id"] for row in schedule["physical_requests"] if row["dataset"] == dataset
        }
        if observed != allowed:
            errors.append(f"{dataset} record IDs do not exactly match selection partition")
    if len(config["workloads"]) != 9:
        errors.append("workload count is not nine")
    if sum(len(v["predicates"]) for v in config["datasets"].values()) != 18:
        errors.append("unique dataset-predicate count is not 18")
    unsupported = [
        row["plan_id"] for row in requirements if row["evidence_status"] != "fully_scheduled"
    ]
    if unsupported:
        errors.append(f"unsupported catalog plans: {len(unsupported)}")
    expected_fingerprint = canonical_hash(
        {k: v for k, v in schedule.items() if k not in {"created_at", "schedule_fingerprint"}}
    )
    if schedule["schedule_fingerprint"] != expected_fingerprint:
        errors.append("schedule fingerprint mismatch")
    return {
        "schema_version": "m5.1-selection-schedule-validation-v1",
        "valid": not errors,
        "errors": errors,
        "workloads": len(config["workloads"]),
        "unique_predicates": 18,
        "selection_records_per_dataset": 100,
        "catalog_plans": len(catalog),
        "fully_scheduled_catalog_plans": len(requirements) - len(unsupported),
        "unsupported_catalog_plans": unsupported,
        "logical_role_counts": dict(counts),
        "important_reconciliation": (
            "3,300 evidence roles map to 3,000 unique calls because three 100-record "
            "remainder arms exactly reuse scheduled full batches"
        ),
        "held_out_outcomes_accessed": False,
    }


def _beta_binomial_probability_less_than(n: int, cutoff: int, alpha: float, beta: float) -> float:
    def log_beta(a: float, b: float) -> float:
        return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

    total = 0.0
    for k in range(cutoff):
        value = (
            math.lgamma(n + 1)
            - math.lgamma(k + 1)
            - math.lgamma(n - k + 1)
            + log_beta(k + alpha, n - k + beta)
            - log_beta(alpha, beta)
        )
        total += math.exp(value)
    return total


def quality_feasibility(config: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(csv.DictReader((STAGE_C / "query_level_agreement.csv").open(encoding="utf-8")))
    by_workload: dict[str, dict[str, Any]] = {}
    # Only representative M5.1 groups with exact workload predicate sets are permitted.
    for workload in config["workloads"]:
        matches = [
            row
            for row in rows
            if row["workload_id"] == workload["id"]
            and row["sample_kind"] == "representative"
            and row["execution_mode"] == "multi_decision"
        ]
        if not matches:
            by_workload[workload["id"]] = {
                "workload_id": workload["id"],
                "dataset": workload["dataset"],
                "historical_records": 0,
                "historical_reference_positives": None,
                "status": "not_estimable_missing_compatible_historical_group",
                "expected_selection_positives": None,
                "probability_fewer_than_three": None,
                "recall_feasibility": "unknown",
            }
            continue
        row = matches[0]
        n = int(row["sample_count"])
        positives = int(row["reference_positive_count"])
        alpha = positives + 0.5
        beta = n - positives + 0.5
        expected = 100 * alpha / (alpha + beta)
        by_workload[workload["id"]] = {
            "workload_id": workload["id"],
            "dataset": workload["dataset"],
            "historical_records": n,
            "historical_reference_positives": positives,
            "observed_prevalence": positives / n,
            "sampling_model": "Jeffreys Beta(0.5,0.5) posterior predictive beta-binomial",
            "expected_selection_positives": expected,
            "probability_fewer_than_three": _beta_binomial_probability_less_than(
                100, 3, alpha, beta
            ),
            "recall_feasibility": "high sparse-support risk"
            if expected < 3
            else "potentially evaluable",
            "zero_observed_does_not_imply_zero_prevalence": positives == 0,
        }
    return {
        "schema_version": "m5.1-selection-quality-feasibility-v1",
        "quality_gate": load_yaml(PROTOCOL)["quality_gate"],
        "workloads": list(by_workload.values()),
        "discrete_thresholds": {
            "three_to_nineteen_positive_references": "zero false negatives required",
            "twenty_to_thirty_nine_positive_references": "at most one false negative",
            "one_hundred_total_records_changed_decision_limit": (
                "at most two changed final decisions"
            ),
        },
        "reference_limitation": (
            "separate Jev consistency reference; not human semantic ground truth"
        ),
        "selection_outcomes_inspected": False,
    }


def _historical_cost_validation() -> dict[str, Any]:
    ranking = json.loads((STAGE_C / "plan_ranking_analysis.json").read_text("utf-8"))
    existing = ranking["prediction_metrics"]
    # The prefix correction is specified and frozen, but cannot be leakage-free validated from
    # the persisted aggregate plan-ranking artifact; retaining the existing model is safer.
    return {
        "schema_version": "m5.1-selection-cost-estimator-validation-v1",
        "development_data": "M5.1 Stage B only",
        "existing_unconditional_group_model": {
            "workloads": len(existing),
            "mean_mae_usd": statistics.mean(row["cost_mae_usd"] for row in existing),
            "rmse_of_workload_predictions_usd": math.sqrt(
                statistics.mean(row["cost_rmse_usd"] ** 2 for row in existing)
            ),
            "mean_pairwise_ranking_concordance": statistics.mean(
                row["pairwise_ranking_concordance"] for row in existing
            ),
            "total_selection_regret_usd": sum(row["selection_regret_usd"] for row in existing),
            "signed_bias_usd": statistics.mean(row["cost_signed_bias_usd"] for row in existing),
            "harmful_selections": sum(not row["selected_observed_best"] for row in existing),
            "per_workload": existing,
        },
        "correlation_aware_prefix_model": {
            "formula": "N * sum_i P(prefix pass) * E[cost(group_i) | prefix pass]",
            "implementation_version": ESTIMATOR_VERSION,
            "validation_status": "specified_but_not_adopted",
            "reason": (
                "saved Stage B plan diagnostics do not contain checkpointed out-of-sample "
                "conditional cost estimates; fitting and scoring exact prefixes on the same "
                "30 records would be tautological"
            ),
        },
        "independence_fallback": {
            "status": "frozen documented fallback",
            "rule": (
                "multiply marginal predicate pass rates only when compatible joint prefix "
                "evidence is unavailable"
            ),
            "missing_measurements_are_never_zero": True,
        },
        "decision": "retain_existing_reach_model_for_selection_with_uncertainty_labels",
        "selection_outcomes_used": False,
        "development_diagnostic_not_independent_validation": True,
    }


def budget(schedule: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(csv.DictReader((STAGE_C / "normalized_measurements.csv").open(encoding="utf-8")))
    rep = [row for row in rows if row["sample_kind"] == "representative"]
    anchors: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rep:
        mode = "single" if row["execution_mode"] == "single" else "multi_decision"
        if mode in {"single", "multi_decision"}:
            anchors[(row["dataset"], mode, int(row["batch_size"]))].append(row)
    per_dataset: dict[str, dict[str, Any]] = {}
    totals = {
        "expected_input_tokens": 0.0,
        "expected_output_tokens": 0.0,
        "conservative_input_tokens": 0.0,
        "conservative_output_tokens": 0.0,
    }
    for dataset in sorted({row["dataset"] for row in schedule["physical_requests"]}):
        requests = [row for row in schedule["physical_requests"] if row["dataset"] == dataset]
        expected_in = expected_out = conservative_in = conservative_out = latency = 0.0
        arm_counts = defaultdict(int)
        for request in requests:
            mode = "single" if request["execution_mode"] == "single" else "multi_decision"
            size = len(request["predicate_ids"])
            samples = anchors[(dataset, mode, size)]
            if not samples:
                raise ValueError(f"missing Stage B calibration anchor for {dataset}/{mode}/{size}")
            inputs = [int(row["input_tokens"]) for row in samples]
            outputs = [int(row["output_tokens"]) for row in samples]
            expected_in += statistics.mean(inputs)
            expected_out += statistics.mean(outputs)
            conservative_in += max(max(inputs), statistics.mean(inputs) * 1.25)
            conservative_out += max(max(outputs), statistics.mean(outputs) * 1.25)
            latency += statistics.mean(float(row["provider_latency_seconds"]) for row in samples)
            arm = "singles" if mode == "single" else ("full_or_remainder_batches")
            arm_counts[arm] += 1
        data = {
            "individual_reference_physical_requests": arm_counts["singles"],
            "full_and_remainder_unique_batch_requests": arm_counts["full_or_remainder_batches"],
            "unique_physical_requests": len(requests),
            "expected_input_tokens": round(expected_in),
            "expected_output_tokens": round(expected_out),
            "expected_latency_sum_seconds": latency,
            "expected_token_priced_cost_usd": expected_in * INPUT_PRICE / 1_000_000,
            "conservative_input_tokens": round(conservative_in),
            "conservative_output_tokens": round(conservative_out),
            "conservative_no_retry_cost_proxy_usd": conservative_in * INPUT_PRICE / 1_000_000,
        }
        per_dataset[dataset] = data
        totals["expected_input_tokens"] += expected_in
        totals["expected_output_tokens"] += expected_out
        totals["conservative_input_tokens"] += conservative_in
        totals["conservative_output_tokens"] += conservative_out
    expected_cost = totals["expected_input_tokens"] * INPUT_PRICE / 1_000_000
    conservative_cost = totals["conservative_input_tokens"] * INPUT_PRICE / 1_000_000
    return {
        "schema_version": "m5.1-selection-budget-v1",
        "calibration": (
            "dataset/mode/question-count means and maxima from actual M5.1 Stage B "
            "representative calls"
        ),
        "model": "jev-1.14.0",
        "pricing": {"input_per_million_usd": INPUT_PRICE, "output_per_million_usd": OUTPUT_PRICE},
        "counts": schedule["counts"],
        "expected_input_tokens": round(totals["expected_input_tokens"]),
        "expected_output_tokens": round(totals["expected_output_tokens"]),
        "expected_token_priced_cost_usd": expected_cost,
        "conservative_input_tokens": round(totals["conservative_input_tokens"]),
        "conservative_output_tokens": round(totals["conservative_output_tokens"]),
        "conservative_no_retry_cost_proxy_usd": conservative_cost,
        "deduplicated_retry_inclusive_cost_proxy_usd": conservative_cost * 3,
        "proposal_9900_attempt_outer_proxy_usd": conservative_cost * (9900 / 3000),
        "per_dataset": per_dataset,
        "historical_stage_b_cost_excluded": 0.012575052,
        "monetary_values_are_proxies_not_runtime_limits": True,
    }


def _fingerprints(config: Mapping[str, Any], schedule: Mapping[str, Any]) -> dict[str, Any]:
    datasets = {}
    for dataset, definition in config["datasets"].items():
        manifest_path = Path(str(definition["path"]) + ".manifest.json")
        manifest = json.loads(manifest_path.read_text("utf-8"))
        datasets[dataset] = {
            "dataset_fingerprint": manifest["dataset_fingerprint_sha256"],
            "selection_partition_fingerprint": manifest["partitions"]["selection"][
                "fingerprint_sha256"
            ],
            "manifest_sha256": sha256_file(manifest_path),
            "dataset_file_sha256": sha256_file(definition["path"]),
        }
    return {
        "workloads_sha256": sha256_file(WORKLOADS),
        "protocol_sha256": sha256_file(PROTOCOL),
        "optimizer_sha256": sha256_file(OPTIMIZER),
        "selection_config_sha256": sha256_file(SELECTION_CONFIG),
        "analysis_source_sha256": sha256_file(__file__),
        "execution_runner_sha256": sha256_file(SELECTION_RUNNER),
        "stage_b_session_sha256": sha256_file(STAGE_B / "session.json"),
        "stage_c_manifest_sha256": sha256_file(STAGE_C / "analysis_manifest.json"),
        "schedule_fingerprint": schedule["schedule_fingerprint"],
        "datasets": datasets,
    }


def validate_selection_launch(
    manifest: Mapping[str, Any],
    *,
    authorization: str | None,
    schedule: Mapping[str, Any],
    output_directory: str | Path,
) -> None:
    if authorization != AUTHORIZATION_PHRASE:
        raise PermissionError("exact M5.1 selection authorization phrase required")
    if manifest.get("paid_calls_authorized") is not True:
        raise PermissionError("manifest is a proposal and has not been countersigned")
    if manifest.get("schedule_fingerprint") != schedule.get("schedule_fingerprint"):
        raise ValueError("schedule fingerprint differs from authorization manifest")
    current_fingerprints = _fingerprints(load_yaml(WORKLOADS), schedule)
    if manifest.get("fingerprints") != current_fingerprints:
        raise ValueError("source, configuration, dataset, or historical fingerprint mismatch")
    if manifest.get("requested_model") != "jev-1.14.0":
        raise ValueError("frozen model mismatch")
    if schedule["counts"]["logical_measurement_roles"] != 3300:
        raise ValueError("logical measurement-role ceiling mismatch")
    if schedule["counts"]["unique_physical_requests"] != 3000:
        raise ValueError("physical acquisition ceiling mismatch")
    if Path(output_directory).exists():
        raise FileExistsError("refusing to overwrite or implicitly resume selection results")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def generate(output: Path = ROOT) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"non-overwriting preflight directory already exists: {output}")
    output.mkdir(parents=True)
    config = load_yaml(WORKLOADS)
    protocol = load_yaml(PROTOCOL)
    inventory = build_inventory(config)
    schedule = build_schedule(config)
    catalog = build_plan_catalog(config)
    requirements = build_plan_requirements(catalog, schedule)
    validation = validate_schedule(config, schedule, catalog, requirements)
    if not validation["valid"]:
        raise ValueError("invalid frozen selection schedule: " + "; ".join(validation["errors"]))
    feasibility = quality_feasibility(config)
    estimator = _historical_cost_validation()
    budget_report = budget(schedule)
    fingerprints = _fingerprints(config, schedule)
    stage_c_audit = json.loads((STAGE_C / "artifact_integrity_audit.json").read_text("utf-8"))
    historical_files = stage_c_audit["historical_integrity"]["files"]
    historical_verification = {
        path: {
            "stage_c_sha256": metadata["current_sha256"],
            "current_sha256": sha256_file(path),
            "unchanged": sha256_file(path) == metadata["current_sha256"],
        }
        for path, metadata in historical_files.items()
    }
    baseline = {
        "schema_version": "m5.1-selection-baseline-integrity-v1",
        "audit_time": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "git_branch": "milestone-4.5-calibration",
        "worktree_dirty_before_preflight": git_worktree_dirty(),
        "baseline_tests": "325 passed before implementation",
        "fingerprints": fingerprints,
        "stage_b_session": json.loads((STAGE_B / "session.json").read_text("utf-8")),
        "historical_integrity": {
            "all_unchanged": all(row["unchanged"] for row in historical_verification.values()),
            "files": historical_verification,
        },
        "production_default": protocol["production_default"],
        "production_default_modified": False,
        "historical_artifacts_modified": False,
        "selection_outcomes_accessed": False,
        "held_out_outcomes_accessed": False,
        "provider_initialized": False,
        "live_jev_calls": 0,
    }
    coverage = {
        "schema_version": "m5.1-selection-coverage-v1",
        "workloads": [
            {
                **row,
                "scheduled_plans": sum(p["workload_id"] == row["workload_id"] for p in catalog),
                "all_catalog_plans_fully_measurable": all(
                    r["evidence_status"] == "fully_scheduled"
                    for r in requirements
                    if next(p for p in catalog if p["plan_id"] == r["plan_id"])["workload_id"]
                    == row["workload_id"]
                ),
            }
            for row in inventory
        ],
        "unsupported_theoretical_plans": (
            "all physical plans outside the frozen catalog, including unselected remainder "
            "groups, arbitrary multi-group partitions, and conjunctive-binary plans"
        ),
    }
    adaptive = {
        "schema_version": "m5.1-existing-adaptive-selection-protocol-v1",
        "implementation": "semanticflow.optimizer.adaptive.simulate_adaptive_profiling",
        "wrapper": "semanticflow.optimizer.fusion_analysis.existing_adaptive_order_from_reference",
        "source_config": str(OPTIMIZER).replace("\\", "/"),
        "source_config_sha256": sha256_file(OPTIMIZER),
        "controller_schema": "m3-adaptive-profiling-v2",
        "checkpoints": [25, 50, 100],
        "sampling_seed": 4049,
        "visibility_rule": (
            "checkpoint k receives only the first k IDs in frozen sha256(seed:review_id) order"
        ),
        "profiling_measurements": "shared individual references only",
        "profiling_cost": "sum of exact token-priced costs for revealed single requests",
        "evaluation_rule": (
            "replay selected order on the same complete selection matrix; label as selection "
            "evidence, never independent evaluation"
        ),
        "future_information_available_to_controller": False,
        "default_policy_modified": False,
    }
    profiling = {
        "schema_version": "m5.1-selection-adaptive-profiling-protocol-v1",
        "state_fields": [
            "checkpoint",
            "available_measurement_ids",
            "candidate_plan_ids",
            "estimated_plan_costs",
            "uncertainty",
            "selected_action",
            "expected_value_of_additional_profiling",
            "profiling_expenditure_usd",
            "stop_decision",
            "selected_plan_id",
        ],
        "reveal_order": "sha256(seed:record_id), fixed before outcomes",
        "temporal_invariant": (
            "available measurements at checkpoint k are a subset of acquisitions through k"
        ),
        "actual_matrix_cost": "charge every unique physical request once",
        "standalone_cost": "charge only compatible requests required by that workload",
        "shared_cost": "charge each physical request once across compatible workloads",
        "warm_start_cost": "charge only newly acquired compatible statistics",
        "cross_strategy_sharing_default": False,
        "stop_conditions": [
            "selected plan stable",
            "expected value of information below next increment",
            "profiling budget exhausted",
            "maximum checkpoint reached",
        ],
    }
    profiling_cost = {
        "schema_version": "m5.1-selection-profiling-cost-accounting-v1",
        "logical_roles": schedule["counts"]["logical_measurement_roles"],
        "actual_unique_acquisitions": schedule["counts"]["unique_physical_requests"],
        "shared_aliases_not_double_charged": schedule["counts"]["shared_alias_roles"],
        "scenarios": [
            "actual_full_matrix",
            "standalone_workload",
            "shared_statistics",
            "warm_start",
        ],
        "reuse_horizons": protocol["reuse_horizons"],
        "net_benefit_formula": (
            "R*(baseline_execution_cost-candidate_execution_cost)"
            "-incremental_profiling_cost-optimization_cost"
        ),
        "optimization_api_cost_usd": 0.0,
        "break_even_rule": "ceil((profiling+optimization)/positive per-execution saving)",
    }
    authorization = {
        "schema_version": "m5.1-selection-authorization-proposal-v1",
        "authorization_phrase": AUTHORIZATION_PHRASE,
        "paid_calls_authorized": False,
        "requested_model": "jev-1.14.0",
        "schedule_fingerprint": schedule["schedule_fingerprint"],
        "fingerprints": fingerprints,
        "logical_measurement_role_ceiling": 3300,
        "unique_physical_request_ceiling": 3000,
        "outer_http_attempt_ceiling": 9900,
        "deduplicated_http_attempt_ceiling": 9000,
        "maximum_retries": 2,
        "atomic_budget_reservation_required": True,
        "output_directory_must_not_exist": True,
        "resume_permitted": False,
        "smoke_checkpoint_request_ids": schedule["smoke_checkpoint"],
        "smoke_requests_included_in_total": True,
        "provider_must_not_initialize_before_validation": True,
    }
    stage_d = {
        "schema_version": "m5.1-stage-d-protocol-template-v1",
        "status": "template_only_not_authorized",
        "primary_comparison": (
            "existing_adaptive vs selection-frozen quality-constrained or profiling-aware hybrid"
        ),
        "selected_plan_ids": None,
        "freeze_requirements": [
            "complete selection matrix validated",
            "quality classifications persisted",
            "selected plan objects and fingerprints persisted",
            "cost forecast generated",
            "held-out outcomes remain untouched",
            "separate Stage D authorization",
        ],
        "held_out_records_per_dataset": 400,
        "identical_physical_plans_share_compatible_measurements": True,
        "future_metrics": [
            "inference cost",
            "requests",
            "tokens",
            "latency",
            "stage reach",
            "individual agreement",
            "query agreement",
            "quality gate",
            "selection regret",
            "cold/warm net savings",
            "break-even",
            "confidence intervals",
        ],
        "future_results": None,
    }
    summary = {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "READY_WITH_DOCUMENTED_LIMITATIONS",
        "workloads": 9,
        "unique_predicates": 18,
        "selection_records": 300,
        "logical_measurement_roles": 3300,
        "unique_physical_requests": 3000,
        "shared_alias_roles": 300,
        "outer_attempt_ceiling": 9900,
        "deduplicated_attempt_ceiling": 9000,
        "all_frozen_catalog_plans_measurable": True,
        "catalog_plans": len(catalog),
        "quality_gate_preserved": True,
        "cost_estimator_decision": estimator["decision"],
        "expected_input_tokens": budget_report["expected_input_tokens"],
        "expected_output_tokens": budget_report["expected_output_tokens"],
        "expected_cost_usd": budget_report["expected_token_priced_cost_usd"],
        "limitations": [
            "three workloads have no compatible Stage B prevalence estimate",
            "seven of nine observed Stage B groups had zero conjunction positives",
            "selection evidence will not be an unbiased estimate of selected-plan generalization",
            "3,300 logical roles deduplicate to 3,000 paid calls",
            (
                "correlation-aware prefix correction was specified but not adopted without "
                "non-tautological validation"
            ),
        ],
        "live_jev_calls": 0,
        "selection_executed": False,
        "held_out_outcomes_accessed": False,
    }
    artifacts = {
        "baseline_integrity_audit.json": baseline,
        "workload_inventory.json": {
            "schema_version": "m5.1-workload-inventory-v1",
            "workloads": inventory,
        },
        "measurement_coverage_matrix.json": coverage,
        "physical_plan_catalog.json": {"schema_version": "m5.1-plan-catalog-v1", "plans": catalog},
        "plan_measurement_requirements.json": {
            "schema_version": "m5.1-plan-requirements-v1",
            "plans": requirements,
        },
        "selection_schedule.json": schedule,
        "selection_schedule_validation.json": validation,
        "adaptive_baseline_protocol.json": adaptive,
        "quality_feasibility_analysis.json": feasibility,
        "cost_estimator_validation.json": estimator,
        "adaptive_profiling_protocol.json": profiling,
        "profiling_cost_accounting.json": profiling_cost,
        "selection_stage_budget.json": budget_report,
        "authorization_manifest.json": authorization,
        "stage_d_protocol_template.json": stage_d,
        "preflight_summary.json": summary,
    }
    for name, value in artifacts.items():
        _write_json(output / name, value)
    manifest = {
        "schema_version": "m5.1-selection-preflight-manifest-v1",
        "analysis_source": "src/semanticflow/optimizer/m51_selection_preflight.py",
        "analysis_source_sha256": sha256_file(__file__),
        "planner_version": PLANNER_VERSION,
        "estimator_version": ESTIMATOR_VERSION,
        "fingerprints": fingerprints,
        "artifact_sha256": {name: sha256_file(output / name) for name in artifacts},
        "provider_initialized": False,
        "live_jev_calls": 0,
    }
    _write_json(output / "analysis_manifest.json", manifest)
    return summary
