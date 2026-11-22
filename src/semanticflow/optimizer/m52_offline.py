# ruff: noqa: E501
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from semanticflow.optimizer.fusion_quality import (
    FusionQualityGate,
    compare_binary_decisions,
    evaluate_quality_gate,
)
from semanticflow.optimizer.m51_final_audit import sha256_file
from semanticflow.optimizer.m51_selection_analysis import load_evidence, replay_plan
from semanticflow.optimizer.m52_acquisition import (
    ActionKind,
    ProfilingAction,
    ProfilingBudget,
)
from semanticflow.optimizer.m52_active_profiler import SamplingStrategy, deterministic_record_order
from semanticflow.optimizer.m52_controller import M52ControllerConfig, M52Optimizer
from semanticflow.optimizer.m52_cost_model import M52CostEstimator
from semanticflow.optimizer.m52_evidence import (
    EvidenceTrack,
    QualityEvidence,
)
from semanticflow.optimizer.m52_planner import PolicyMode
from semanticflow.optimizer.m52_quality_uncertainty import (
    SequentialQualificationConfig,
    exact_one_sided_bounds,
    qualify_evidence,
    wilson_interval,
)
from semanticflow.optimizer.m52_replay import HistoricalReplayMatrix

OUTPUT = Path("results/milestone_5_2/offline_development")
DOCS = Path("docs")
M51 = Path("results/milestone_5_1/selection_stage_results")
M51_AUDIT = Path("results/milestone_5_1/final_research_audit")
WORKLOADS = Path("configs/milestone_5_1/workloads.yaml")
PROTOCOL = Path("configs/milestone_5_1/protocol.yaml")
REUSE_HORIZONS = (1, 5, 10, 25, 50, 100, 250, 1000)


def _json(path: Path) -> Any:
    return json.loads(path.read_text("utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = list(rows)
    fields = list(dict.fromkeys(key for row in values for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in values:
            writer.writerow(row)


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _manifest_fingerprints() -> dict[str, str]:
    return {
        "workloads_sha256": sha256_file(WORKLOADS),
        "protocol_sha256": sha256_file(PROTOCOL),
        "frozen_catalog_sha256": sha256_file(M51 / "frozen_physical_plan_catalog.json"),
        "frozen_selections_sha256": sha256_file(M51 / "frozen_strategy_selections.json"),
        "stage_d_schedule_sha256": sha256_file(M51 / "stage_d_execution_schedule.json"),
        "m51_final_audit_manifest_sha256": sha256_file(
            M51_AUDIT / "artifact_integrity_manifest.json"
        ),
    }


def _strategy_map(
    selections: Sequence[Mapping[str, Any]], strategy: str
) -> dict[str, Mapping[str, Any]]:
    return {str(row["workload_id"]): row for row in selections if row["strategy"] == strategy}


def _dataset_fingerprints() -> dict[str, str]:
    audit = _json(M51_AUDIT / "historical_integrity_audit.json")
    return {row["dataset"]: row["dataset_fingerprint"] for row in audit["datasets"]}


def _quality_evidence(
    plan: Mapping[str, Any],
    *,
    dataset_fingerprint: str,
) -> QualityEvidence:
    return QualityEvidence(
        plan_id=str(plan["plan_id"]),
        plan_fingerprint=str(plan["plan_fingerprint"]),
        dataset=str(plan["dataset"]),
        dataset_fingerprint=dataset_fingerprint,
        workload_id=str(plan["workload_id"]),
        workload_fingerprint=_canonical_hash(
            {
                "workload_id": plan["workload_id"],
                "predicates": plan["logical_predicate_ids"],
            }
        ),
        physical_groups=tuple(tuple(group["predicate_ids"]) for group in plan["groups"]),
        evidence_provenance=("M5.1 stored real Jev measurement matrix",),
    )


def _new_matrix(template: HistoricalReplayMatrix) -> HistoricalReplayMatrix:
    selections = _json(M51 / "frozen_strategy_selections.json")["strategies"]
    return HistoricalReplayMatrix(
        rows=template.rows,
        plans=tuple(template.plans.values()),
        selections=selections,
        config=template.config,
        source_partition=template.source_partition,
    )


def _predictions(
    estimator: M52CostEstimator,
    matrix: HistoricalReplayMatrix,
    plan: Mapping[str, Any],
    fallback: Mapping[str, Any],
    *,
    record_count: int,
    dataset_fingerprint: str,
) -> tuple[Any, Any]:
    workload_fingerprint = _canonical_hash(
        {"id": plan["workload_id"], "predicates": plan["logical_predicate_ids"]}
    )
    args = {
        "record_count": record_count,
        "checkpoint": 0,
        "dataset_fingerprint": dataset_fingerprint,
        "workload_fingerprint": workload_fingerprint,
    }
    return estimator.predict(fallback, **args), estimator.predict(plan, **args)


def _score_candidate(
    stage_d: HistoricalReplayMatrix,
    workload_id: str,
    plan_id: str,
    fallback_plan_id: str,
) -> dict[str, Any]:
    plan = stage_d.plans[plan_id]
    fallback = stage_d.plans[fallback_plan_id]
    record_ids = stage_d.records(str(plan["dataset"]))
    candidate = replay_plan(plan, record_ids, stage_d.lookup)
    reference = replay_plan(fallback, record_ids, stage_d.lookup)
    reference_by_id = {row["record_id"]: bool(row["passed"]) for row in reference["per_record"]}
    candidate_by_id = {row["record_id"]: bool(row["passed"]) for row in candidate["per_record"]}
    metrics = compare_binary_decisions(
        [reference_by_id[item] for item in record_ids],
        [candidate_by_id[item] for item in record_ids],
    )
    gate = evaluate_quality_gate(
        metrics,
        FusionQualityGate(),
        evidence_scope="complete_plan",
        real_fused_measurement=True,
    )
    return {
        "workload_id": workload_id,
        "plan_id": plan_id,
        "fallback_plan_id": fallback_plan_id,
        "candidate_cost_usd": candidate["cost_usd"],
        "fallback_cost_usd": reference["cost_usd"],
        "execution_saving_usd": reference["cost_usd"] - candidate["cost_usd"],
        "heldout_point_gate_eligible": gate.eligible,
        "heldout_gate_reasons": list(gate.reasons),
        **metrics.model_dump(mode="json"),
        "retrospective_only": True,
    }


def _run_policy(
    *,
    template: HistoricalReplayMatrix,
    estimator: M52CostEstimator,
    stage_d: HistoricalReplayMatrix,
    workload: Mapping[str, Any],
    plan: Mapping[str, Any],
    fallback: Mapping[str, Any],
    mode: PolicyMode,
    strategy: SamplingStrategy,
    budget_records: int,
    reuse_horizon: int,
    force_fixed_sample: bool,
    dataset_fingerprint: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    matrix = _new_matrix(template)
    record_ids = tuple(matrix.records(str(plan["dataset"])))
    ordered = deterministic_record_order(record_ids, 5201)
    if strategy == SamplingStrategy.RANDOM_REPRESENTATIVE:
        certification = tuple(ordered[:budget_records])
        discovery: tuple[str, ...] = ()
    else:
        discovery_count = min(max(1, budget_records // 4), 30)
        discovery = tuple(ordered[:discovery_count])
        certification = tuple(ordered[discovery_count:budget_records])
    evidence = _quality_evidence(plan, dataset_fingerprint=dataset_fingerprint)
    fallback_prediction, candidate_prediction = _predictions(
        estimator,
        matrix,
        plan,
        fallback,
        record_count=len(record_ids),
        dataset_fingerprint=dataset_fingerprint,
    )
    config = M52ControllerConfig(
        mode=mode,
        sampling_strategy=strategy,
        qualification=SequentialQualificationConfig(
            maximum_checkpoints=8,
            family_candidate_count=4,
        ),
        budget=ProfilingBudget(
            maximum_physical_requests=max(1, budget_records * 6),
            maximum_cost_usd=0.02,
            maximum_records=budget_records,
            maximum_actions=max(1, budget_records * 2),
            maximum_certification_records=len(certification),
            maximum_discovery_records=len(discovery),
            maximum_positive_enrichment_records=len(discovery),
        ),
        representative_checkpoints=tuple(
            value for value in (40, 80, 100, 120, 160) if value <= len(certification)
        )
        or (len(certification),),
        reuse_horizon=reuse_horizon,
        minimum_expected_value_of_information_usd=(-1.0 if force_fixed_sample else 0.0),
    )
    result = M52Optimizer(config).run(
        matrix=matrix,
        evidence=evidence,
        fallback_prediction=fallback_prediction,
        candidate_prediction=candidate_prediction,
        certification_pool=certification,
        discovery_pool=discovery,
        evaluation_record_count=len(record_ids),
    )
    score = _score_candidate(
        stage_d, str(workload["id"]), str(plan["plan_id"]), str(fallback["plan_id"])
    )
    selected_candidate = not result.decision.used_fallback
    unsafe = selected_candidate and not score["heldout_point_gate_eligible"]
    missed = (
        not selected_candidate
        and score["heldout_point_gate_eligible"]
        and score["execution_saving_usd"] > 0
    )
    row = {
        "dataset": workload["dataset"],
        "workload_id": workload["id"],
        "policy_mode": mode.value,
        "sampling_strategy": strategy.value,
        "budget_records": budget_records,
        "reuse_horizon": reuse_horizon,
        "candidate_plan_id": plan["plan_id"],
        "selected_plan_id": result.decision.selected_plan_id,
        "used_fallback": result.decision.used_fallback,
        "stop_reason": result.final_state.stopping_reason,
        "quality_state": result.final_state.quality_estimate.state.value,
        "representative_samples": result.final_state.usage.certification_records,
        "discovery_samples": result.final_state.usage.discovery_records,
        "profiling_requests": result.final_state.usage.physical_requests,
        "profiling_named_question_evaluations": (
            result.final_state.usage.named_question_evaluations
        ),
        "profiling_input_tokens": result.final_state.usage.input_tokens,
        "profiling_output_tokens": result.final_state.usage.output_tokens,
        "profiling_cost_usd": result.final_state.usage.token_priced_cost_usd,
        "fallback_realized_execution_cost_usd": score["fallback_cost_usd"],
        "candidate_realized_execution_cost_usd": score["candidate_cost_usd"],
        "predicted_execution_saving_usd": (
            fallback_prediction.predicted_cost_usd - candidate_prediction.predicted_cost_usd
        ),
        "realized_execution_saving_usd": score["execution_saving_usd"] if selected_candidate else 0,
        "profiling_inclusive_net_benefit_usd": (
            reuse_horizon * (score["execution_saving_usd"] if selected_candidate else 0)
            - result.final_state.usage.token_priced_cost_usd
        ),
        "unsafe_selection": unsafe,
        "missed_admissible_opportunity": missed,
        "heldout_point_gate_eligible": score["heldout_point_gate_eligible"],
        "historical_stage_d_is_independent_m52_test": False,
        "policy_fingerprint": result.policy_fingerprint,
    }
    trace = result.model_dump(mode="json")
    trace["retrospective_scoring"] = score
    return row, trace


def _direct_full_evidence(
    template: HistoricalReplayMatrix,
    plan: Mapping[str, Any],
    fallback: Mapping[str, Any],
    *,
    dataset_fingerprint: str,
    track: EvidenceTrack = EvidenceTrack.CERTIFICATION,
) -> QualityEvidence:
    matrix = _new_matrix(template)
    evidence = _quality_evidence(plan, dataset_fingerprint=dataset_fingerprint)
    for index, record_id in enumerate(matrix.records(str(plan["dataset"]))):
        action = ProfilingAction(
            action_id=f"m52direct_{plan['plan_id']}_{index}",
            kind=ActionKind.ACQUIRE_REPRESENTATIVE_PAIR,
            record_id=record_id,
            plan_id=str(plan["plan_id"]),
            track=track,
            acquisition_method="historical_complete_reconstruction",
            score_components={"checkpoint": 0, "inclusion_probability": 1.0},
        )
        acquired = matrix.acquire(action)
        if acquired.quality_measurement is not None:
            evidence = evidence.append(acquired.quality_measurement)
    return evidence


def _pilot_budget(selection_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mean_input = sum(int(row["input_tokens"]) for row in selection_rows) / len(selection_rows)
    mean_output = sum(int(row["output_tokens"]) for row in selection_rows) / len(selection_rows)
    workloads = [
        "m51_banking_card_unauthorized_refund",
        "m51_beauty_failure_usability",
        "m51_sms_prize_urgency_contact",
    ]
    per_workload = {
        "representative_records": 60,
        "discovery_screen_records": 20,
        "discovery_candidate_records_maximum": 20,
        "maximum_physical_requests_per_record": 5,
    }
    request_ceiling = len(workloads) * 80 * 5
    expected_requests = len(workloads) * (60 * 2.4 + 20 * 2.4)
    input_forecast = round(expected_requests * mean_input)
    output_forecast = round(expected_requests * mean_output)
    return {
        "schema_version": "m5.2-future-pilot-budget-proposal-v1",
        "status": "proposal_only_not_authorized",
        "datasets": 3,
        "workloads": workloads,
        "per_workload_schedule": per_workload,
        "expected_physical_requests": math.ceil(expected_requests),
        "physical_request_ceiling": request_ceiling,
        "expected_named_question_evaluations": math.ceil(expected_requests * 1.7),
        "expected_input_tokens": input_forecast,
        "expected_output_tokens": output_forecast,
        "expected_cost_proxy_usd": input_forecast * 0.042 / 1_000_000,
        "conservative_input_tokens": math.ceil(request_ceiling * mean_input * 1.25),
        "conservative_cost_proxy_usd": request_ceiling * mean_input * 1.25 * 0.042 / 1_000_000,
        "maximum_retries": 2,
        "retry_inclusive_attempt_ceiling": request_ceiling * 3,
        "screening_calls_included": True,
        "provider_initialization_requires_new_exact_authorization": True,
        "authorization_phrase": "AUTHORIZE M5.2 SMALL LIVE PILOT <FROZEN_SCOPE_FINGERPRINT>",
        "historical_authorization_reuse_allowed": False,
        "automatic_paid_continuation": False,
        "pricing": {"input_per_million_usd": 0.042, "output_per_million_usd": 0.0},
        "forecast_basis": "M5.1 selection-stage request-level mean tokens; not future measured costs",
    }


def build_offline_results() -> dict[str, Any]:
    selection_evidence = load_evidence()
    selection = HistoricalReplayMatrix.selection()
    stage_d = HistoricalReplayMatrix.stage_d()
    estimator = M52CostEstimator(selection_evidence["normalized"], allowed_partition="selection")
    selections = _json(M51 / "frozen_strategy_selections.json")["strategies"]
    adaptive = _strategy_map(selections, "existing_adaptive")
    full = _strategy_map(selections, "full_multi_decision_batch")
    fingerprints = _dataset_fingerprints()
    policy_definitions = [
        (
            "existing_adaptive",
            PolicyMode.EXISTING_ADAPTIVE,
            SamplingStrategy.RANDOM_REPRESENTATIVE,
            0,
            False,
            10,
        ),
        (
            "m51_fixed_sample",
            PolicyMode.M51_FIXED_GATE,
            SamplingStrategy.RANDOM_REPRESENTATIVE,
            100,
            True,
            1000,
        ),
        (
            "random_original_gate",
            PolicyMode.M51_FIXED_GATE,
            SamplingStrategy.RANDOM_REPRESENTATIVE,
            80,
            True,
            1000,
        ),
        (
            "random_uncertainty",
            PolicyMode.STRICT_CERTIFICATION,
            SamplingStrategy.RANDOM_REPRESENTATIVE,
            80,
            True,
            1000,
        ),
        (
            "active_original_gate",
            PolicyMode.M51_FIXED_GATE,
            SamplingStrategy.HYBRID_ACTIVE,
            100,
            True,
            1000,
        ),
        (
            "active_uncertainty",
            PolicyMode.STRICT_CERTIFICATION,
            SamplingStrategy.HYBRID_ACTIVE,
            100,
            True,
            1000,
        ),
        (
            "cost_aware_uncertainty",
            PolicyMode.STRICT_CERTIFICATION,
            SamplingStrategy.COST_AWARE,
            100,
            True,
            1000,
        ),
        (
            "risk_budget",
            PolicyMode.RISK_BUDGET,
            SamplingStrategy.RANDOM_REPRESENTATIVE,
            100,
            True,
            1000,
        ),
        (
            "cost_only",
            PolicyMode.COST_ONLY,
            SamplingStrategy.RANDOM_REPRESENTATIVE,
            0,
            False,
            1000,
        ),
        (
            "full_m52",
            PolicyMode.STRICT_CERTIFICATION,
            SamplingStrategy.HYBRID_ACTIVE,
            100,
            False,
            10,
        ),
    ]
    ablations: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for policy_name, mode, strategy, budget, fixed, horizon in policy_definitions:
        for workload in selection.config["workloads"]:
            workload_id = str(workload["id"])
            fallback = selection.plans[str(adaptive[workload_id]["plan_id"])]
            candidate = selection.plans[str(full[workload_id]["plan_id"])]
            if budget == 0:
                score = _score_candidate(
                    stage_d, workload_id, str(candidate["plan_id"]), str(fallback["plan_id"])
                )
                fallback_prediction, candidate_prediction = _predictions(
                    estimator,
                    selection,
                    candidate,
                    fallback,
                    record_count=len(selection.records(str(workload["dataset"]))),
                    dataset_fingerprint=fingerprints[str(workload["dataset"])],
                )
                cost_only_selected = bool(
                    mode == PolicyMode.COST_ONLY
                    and candidate_prediction.predicted_cost_usd
                    < fallback_prediction.predicted_cost_usd
                )
                row = {
                    "dataset": workload["dataset"],
                    "workload_id": workload_id,
                    "policy_name": policy_name,
                    "policy_mode": mode.value,
                    "sampling_strategy": strategy.value,
                    "budget_records": 0,
                    "reuse_horizon": horizon,
                    "candidate_plan_id": candidate["plan_id"],
                    "selected_plan_id": (
                        candidate["plan_id"] if cost_only_selected else fallback["plan_id"]
                    ),
                    "used_fallback": not cost_only_selected,
                    "stop_reason": (
                        "cost_only_prediction" if mode == PolicyMode.COST_ONLY
                        else "control_baseline"
                    ),
                    "quality_state": "not_evaluated",
                    "representative_samples": 0,
                    "discovery_samples": 0,
                    "profiling_requests": 0,
                    "profiling_named_question_evaluations": 0,
                    "profiling_input_tokens": 0,
                    "profiling_output_tokens": 0,
                    "profiling_cost_usd": 0.0,
                    "fallback_realized_execution_cost_usd": score["fallback_cost_usd"],
                    "candidate_realized_execution_cost_usd": score["candidate_cost_usd"],
                    "predicted_execution_saving_usd": (
                        fallback_prediction.predicted_cost_usd
                        - candidate_prediction.predicted_cost_usd
                    ),
                    "realized_execution_saving_usd": (
                        score["execution_saving_usd"] if cost_only_selected else 0.0
                    ),
                    "profiling_inclusive_net_benefit_usd": (
                        horizon * score["execution_saving_usd"] if cost_only_selected else 0.0
                    ),
                    "unsafe_selection": (
                        cost_only_selected and not score["heldout_point_gate_eligible"]
                    ),
                    "missed_admissible_opportunity": (
                        not cost_only_selected
                        and score["heldout_point_gate_eligible"]
                        and score["execution_saving_usd"] > 0
                    ),
                    "heldout_point_gate_eligible": score["heldout_point_gate_eligible"],
                    "historical_stage_d_is_independent_m52_test": False,
                    "policy_fingerprint": _canonical_hash({"policy": policy_name}),
                }
                trace = {"policy_name": policy_name, "workload_id": workload_id, "control": True}
            else:
                row, trace = _run_policy(
                    template=selection,
                    estimator=estimator,
                    stage_d=stage_d,
                    workload=workload,
                    plan=candidate,
                    fallback=fallback,
                    mode=mode,
                    strategy=strategy,
                    budget_records=budget,
                    reuse_horizon=horizon,
                    force_fixed_sample=fixed,
                    dataset_fingerprint=fingerprints[str(workload["dataset"])],
                )
                row["policy_name"] = policy_name
                trace["policy_name"] = policy_name
            ablations.append(row)
            traces.append(trace)

    primary_id = "m51_banking_card_unauthorized_refund"
    beauty_id = "m51_beauty_failure_usability"
    controlled = {}
    for label, workload_id in (("banking77", primary_id), ("beauty", beauty_id)):
        plan = selection.plans[str(full[workload_id]["plan_id"])]
        fallback = selection.plans[str(adaptive[workload_id]["plan_id"])]
        evidence = _direct_full_evidence(
            selection,
            plan,
            fallback,
            dataset_fingerprint=fingerprints[str(plan["dataset"])],
        )
        strict = qualify_evidence(
            evidence,
            SequentialQualificationConfig(family_candidate_count=1, maximum_checkpoints=1),
            checkpoint=0,
        )
        score = _score_candidate(
            stage_d, workload_id, str(plan["plan_id"]), str(fallback["plan_id"])
        )
        controlled[label] = {
            "workload_id": workload_id,
            "selection_evidence": evidence.counts().model_dump(mode="json"),
            "selection_measurement_cost_usd": evidence.measurement_cost_usd,
            "m51_point_gate_passed": _point_gate_for_report(evidence),
            "m52_strict_estimate": strict.model_dump(mode="json"),
            "retrospective_stage_d_score": score,
            "conclusion": (
                "M5.2 remains uncertain and cannot certify; cost-aware stopping may fall back"
                if label == "banking77"
                else "selection evidence fails the point gate and strict bounds; held-out success cannot be leaked into the decision"
            ),
            "independent_m52_validation": False,
        }

    sparse = []
    stage_d_quality = _json(M51_AUDIT / "canonical_plan_quality_results.json")[
        "all_measured_plan_quality"
    ]
    quality_by_id = {row["plan_id"]: row for row in stage_d_quality}
    for workload in selection.config["workloads"]:
        workload_id = str(workload["id"])
        plan_id = str(full[workload_id]["plan_id"])
        quality = quality_by_id[plan_id]
        if quality["reference_positive_count"] == 0:
            sparse.append(
                {
                    "workload_id": workload_id,
                    "observations": quality["sample_count"],
                    "reference_positives": 0,
                    "recall": None,
                    "state": "INSUFFICIENT_SUPPORT",
                    "certified": False,
                    "safe_fallback": True,
                    "additional_representative_evidence_required": True,
                }
            )

    budgets = []
    for budget in (20, 40, 80, 100):
        for workload_id in (primary_id, beauty_id):
            workload = selection.workload(workload_id)
            plan = selection.plans[str(full[workload_id]["plan_id"])]
            fallback = selection.plans[str(adaptive[workload_id]["plan_id"])]
            row, _ = _run_policy(
                template=selection,
                estimator=estimator,
                stage_d=stage_d,
                workload=workload,
                plan=plan,
                fallback=fallback,
                mode=PolicyMode.STRICT_CERTIFICATION,
                strategy=SamplingStrategy.HYBRID_ACTIVE,
                budget_records=budget,
                reuse_horizon=10,
                force_fixed_sample=False,
                dataset_fingerprint=fingerprints[str(workload["dataset"])],
            )
            budgets.append(row)

    horizons = []
    for horizon in REUSE_HORIZONS:
        for workload_id in (primary_id, beauty_id):
            workload = selection.workload(workload_id)
            plan = selection.plans[str(full[workload_id]["plan_id"])]
            fallback = selection.plans[str(adaptive[workload_id]["plan_id"])]
            row, _ = _run_policy(
                template=selection,
                estimator=estimator,
                stage_d=stage_d,
                workload=workload,
                plan=plan,
                fallback=fallback,
                mode=PolicyMode.STRICT_CERTIFICATION,
                strategy=SamplingStrategy.HYBRID_ACTIVE,
                budget_records=100,
                reuse_horizon=horizon,
                force_fixed_sample=False,
                dataset_fingerprint=fingerprints[str(workload["dataset"])],
            )
            horizons.append(row)

    config = SequentialQualificationConfig()
    uncertainty_validation = {
        "schema_version": "m5.2-quality-uncertainty-validation-v1",
        "method": "Clopper-Pearson one-sided exact bounds with preregistered Bonferroni checkpoint/candidate/metric allocation",
        "family_alpha": config.family_alpha,
        "allocated_one_sided_alpha": config.allocated_one_sided_alpha,
        "wilson_examples": {
            "3_of_3": wilson_interval(3, 3),
            "21_of_22": wilson_interval(21, 22),
        },
        "exact_examples": {
            "3_of_3": exact_one_sided_bounds(3, 3, config.allocated_one_sided_alpha),
            "21_of_22": exact_one_sided_bounds(21, 22, config.allocated_one_sided_alpha),
        },
        "ordinary_wilson_used_for_sequential_certification": False,
        "weighted_active_sample_certification_implemented": False,
    }
    aggregate_by_policy = []
    for policy_name in [item[0] for item in policy_definitions]:
        rows = [row for row in ablations if row["policy_name"] == policy_name]
        aggregate_by_policy.append(
            {
                "policy_name": policy_name,
                "workloads": len(rows),
                "unsafe_selections": sum(row["unsafe_selection"] for row in rows),
                "missed_admissible_opportunities": sum(
                    row["missed_admissible_opportunity"] for row in rows
                ),
                "selected_candidates": sum(not row["used_fallback"] for row in rows),
                "fallbacks": sum(row["used_fallback"] for row in rows),
                "profiling_requests": sum(row["profiling_requests"] for row in rows),
                "profiling_named_question_evaluations": sum(
                    row["profiling_named_question_evaluations"] for row in rows
                ),
                "profiling_input_tokens": sum(row["profiling_input_tokens"] for row in rows),
                "profiling_output_tokens": sum(row["profiling_output_tokens"] for row in rows),
                "profiling_cost_usd": sum(row["profiling_cost_usd"] for row in rows),
                "profiling_inclusive_net_benefit_usd": sum(
                    row["profiling_inclusive_net_benefit_usd"] for row in rows
                ),
            }
        )
    return {
        "integrity": {
            "schema_version": "m5.2-historical-integrity-audit-v1",
            "source_fingerprints": _manifest_fingerprints(),
            "m51_integrity_valid": _json(M51_AUDIT / "historical_integrity_audit.json")["valid"],
            "historical_artifacts_modified": False,
            "provider_initialized": False,
            "new_jev_requests": 0,
        },
        "policy_configs": {
            "schema_version": "m5.2-policy-configurations-v1",
            "qualification": config.model_dump(mode="json"),
            "policies": [
                {
                    "name": name,
                    "mode": mode.value,
                    "sampling": strategy.value,
                    "budget_records": budget,
                    "fixed_sample": fixed,
                    "reuse_horizon": horizon,
                    "fingerprint": _canonical_hash(
                        {
                            "name": name,
                            "mode": mode.value,
                            "sampling": strategy.value,
                            "budget": budget,
                            "fixed": fixed,
                            "horizon": horizon,
                            "qualification": config.model_dump(mode="json"),
                        }
                    ),
                }
                for name, mode, strategy, budget, fixed, horizon in policy_definitions
            ],
        },
        "uncertainty_validation": uncertainty_validation,
        "sequential_tests": {
            "schema_version": "m5.2-sequential-qualification-tests-v1",
            "maximum_checkpoints": config.maximum_checkpoints,
            "family_candidate_count": config.family_candidate_count,
            "simultaneous_bounds": config.simultaneous_bounds,
            "family_wise_union_bound": config.allocated_one_sided_alpha
            * config.maximum_checkpoints
            * config.family_candidate_count
            * config.simultaneous_bounds,
            "adaptive_stopping_supported": True,
            "assumption": "all potential inspections are included in the frozen finite family",
        },
        "ablations": ablations,
        "aggregate_by_policy": aggregate_by_policy,
        "traces": traces,
        "banking": controlled["banking77"],
        "beauty": controlled["beauty"],
        "sparse": sparse,
        "budgets": budgets,
        "horizons": horizons,
        "pilot_budget": _pilot_budget(selection_evidence["normalized"]),
        "future_protocol": _future_protocol(config),
        "cost_provenance": {
            "schema_version": "m5.2-cost-prediction-provenance-v1",
            "estimator_version": estimator.estimator_version,
            "allowed_partition": estimator.allowed_partition,
            "future_evaluation_costs_visible": False,
            "predictions_persisted_before_replay_reveal": True,
            "operator_statistics": [
                value.model_dump(mode="json") for value in estimator.statistics.values()
            ],
        },
    }


def _point_gate_for_report(evidence: QualityEvidence) -> bool:
    counts = evidence.counts()
    if counts.observations < 40 or counts.reference_positives < 3:
        return False
    return (
        (counts.true_positive + counts.true_negative) / counts.observations >= 0.98
        and counts.true_positive / counts.reference_positives >= 0.95
        and counts.false_negative / counts.reference_positives <= 0.05
        and (counts.false_positive + counts.false_negative) / counts.observations <= 0.02
    )


def _future_protocol(config: SequentialQualificationConfig) -> dict[str, Any]:
    return {
        "schema_version": "m5.2-future-independent-evaluation-protocol-v1",
        "status": "not_executed_requires_freeze_and_authorization",
        "independent_data": [
            "new unseen records from existing domains",
            "at least one new semantic domain",
            "independently annotated stratified subset",
        ],
        "quality_rule": config.model_dump(mode="json"),
        "positive_support_feasibility": (
            "must be estimated before scheduling; 400 records is not assumed sufficient"
        ),
        "human_annotation_strata": [
            "representative random records",
            "clear predicted positives",
            "clear predicted negatives",
            "near-threshold diagnostics",
            "separate/batch disagreements",
        ],
        "annotation_controls": [
            "independent double annotation",
            "adjudication without Jev reference disclosure",
            "separate representative and enriched estimates",
        ],
        "freeze_requirements": [
            "policy hash",
            "qualification rule",
            "sampling and stopping rule",
            "candidate catalog",
            "cost estimator",
            "quality thresholds",
            "budget and schedule",
            "dataset partition fingerprints",
        ],
        "m51_stage_d_is_independent_test": False,
    }


def generate(output: Path = OUTPUT, docs: Path = DOCS) -> dict[str, Any]:
    data = build_offline_results()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "historical_integrity_audit.json", data["integrity"])
    _write_json(output / "m52_policy_configurations.json", data["policy_configs"])
    _write_json(output / "quality_uncertainty_validation.json", data["uncertainty_validation"])
    _write_json(output / "sequential_qualification_tests.json", data["sequential_tests"])
    _write_json(output / "profiling_action_traces.json", data["traces"])
    _write_json(
        output / "unsafe_selection_analysis.json",
        {
            "schema_version": "m5.2-unsafe-selection-analysis-v1",
            "policies": data["aggregate_by_policy"],
            "retrospective_only": True,
        },
    )
    _write_json(
        output / "missed_opportunity_analysis.json",
        {
            "schema_version": "m5.2-missed-opportunity-analysis-v1",
            "policies": data["aggregate_by_policy"],
            "oracle_hindsight": True,
        },
    )
    _write_json(output / "banking77_failure_replay.json", data["banking"])
    _write_json(output / "beauty_missed_opportunity_replay.json", data["beauty"])
    _write_json(output / "sparse_positive_stress_tests.json", data["sparse"])
    _write_json(output / "cost_prediction_provenance.json", data["cost_provenance"])
    _write_json(output / "replay_limitations.json", _limitations())
    _write_json(output / "future_pilot_budget.json", data["pilot_budget"])
    _write_json(output / "future_evaluation_protocol.json", data["future_protocol"])
    _write_csv(output / "sampling_strategy_comparison.csv", data["aggregate_by_policy"])
    _write_csv(output / "profiling_costs.csv", data["ablations"])
    _write_csv(output / "quality_qualification_results.csv", data["ablations"])
    _write_csv(output / "selection_regret.csv", data["ablations"])
    _write_csv(output / "policy_ablation_results.csv", data["ablations"])
    _write_csv(output / "reuse_horizon_analysis.csv", data["horizons"])
    _write_csv(output / "budget_sensitivity.csv", data["budgets"])
    _write_docs(docs, data)
    artifacts = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_integrity_manifest.json"
    }
    documents = {
        name: sha256_file(docs / name)
        for name in (
            "milestone_5_2_system_design.md",
            "milestone_5_2_statistical_protocol.md",
            "milestone_5_2_offline_results.md",
            "milestone_5_2_live_pilot_proposal.md",
        )
    }
    manifest = {
        "schema_version": "m5.2-offline-artifact-integrity-v1",
        "artifacts": artifacts,
        "documents": documents,
        "historical_source_fingerprints": _manifest_fingerprints(),
        "provider_initialized": False,
        "new_jev_requests": 0,
        "historical_m51_artifacts_modified": False,
    }
    _write_json(output / "artifact_integrity_manifest.json", manifest)
    return {
        "status": "complete",
        "output": output.as_posix(),
        "policies": len(data["aggregate_by_policy"]),
        "workloads": 9,
        "new_jev_requests": 0,
    }


def _limitations() -> dict[str, Any]:
    return {
        "schema_version": "m5.2-replay-limitations-v1",
        "retrospective_only": True,
        "limitations": [
            "M5.1 Stage D outcomes are known development evidence, not an independent M5.2 test.",
            "Only physical operators present in the historical matrices are replayable.",
            "Text-only active features were not claimed to be calibrated semantic predictors.",
            "Discovery enrichment is never used as unweighted certification evidence.",
            "Separate Jev inference is a consistency reference, not human ground truth.",
            "Sequential exact bounds are intentionally conservative under rare positives.",
            "Policy comparisons share records, predicates, and physical measurements.",
            "Local optimization cost remains unmetered and is reported as unknown.",
        ],
        "unsupported_actions": [
            "unmeasured operator combinations",
            "new LLM screening questions",
            "human labels",
            "identical-request repeatability",
        ],
    }


def _write_docs(docs: Path, data: Mapping[str, Any]) -> None:
    docs.mkdir(parents=True, exist_ok=True)
    aggregate = {row["policy_name"]: row for row in data["aggregate_by_policy"]}
    policy_table_rows = "".join(
        "| {name} | {unsafe} | {missed} | {selected} | {requests} | ${cost:.9f} | "
        "${net:.9f} |\n".format(
            name=name,
            unsafe=row["unsafe_selections"],
            missed=row["missed_admissible_opportunities"],
            selected=row["selected_candidates"],
            requests=row["profiling_requests"],
            cost=row["profiling_cost_usd"],
            net=row["profiling_inclusive_net_benefit_usd"],
        )
        for name, row in aggregate.items()
    )
    system = """# SemanticFlow Milestone 5.2 System Design

## Problem and integration boundary

M5.2 adds uncertainty-aware quality qualification and active profiling around the unchanged M5.1 physical plan representation and `existing_adaptive` baseline. It does not change executor semantics, frozen M5.1 plans, the M5.1 point gate, or provider integration.

## Components

- `m52_evidence.py`: immutable paired evidence, discovery/certification isolation, confusion counts, and qualification states.
- `m52_quality_uncertainty.py`: Wilson diagnostics plus exact one-sided sequential bounds with family-wise spending.
- `m52_cost_model.py`: selection-partition-locked predictions with input measurement provenance.
- `m52_replay.py`: reveal-on-acquisition historical matrix and exact recorded charging.
- `m52_active_profiler.py`: random, positive-enriched, threshold, disagreement, cost-aware, and hybrid policies.
- `m52_controller.py`: traceable state machine, budgets, hard quality constraint, EVSI approximation, and stopping.
- `m52_planner.py`: strict, risk-budget, cost-only, M5.1-gate, and existing-adaptive modes.

## Evidence lifecycle

Discovery actions may screen a record using acquired separate inference and then acquire a candidate measurement. Certification actions use an independently frozen representative sequence. A record cannot occur in both tracks. Diagnostic observations never enter ordinary certification counts. Every request ID is charged once and remains invisible until acquired.

## State transitions

The controller records INITIALIZE, EVALUATE_PRIOR, SELECT_ACTION, ACQUIRE_EVIDENCE, UPDATE_QUALITY_ESTIMATES, CHECK_QUALIFICATION, CHECK_ECONOMIC_VALUE, CONTINUE, SELECT_PLAN/FALLBACK, and TERMINATE transitions. Stop reasons distinguish certification, rejection, insufficient support, budget exhaustion, unsupported evidence, and uneconomic information.

## Expected value of information

Semantic quality is a hard constraint. No dollar penalty is assigned to semantic errors. The tractable EVSI approximation multiplies the remaining optimistic reuse-horizon saving by an evidence-gap-based probability of changing the decision, then subtracts predicted acquisition cost. It uses only frozen selection statistics and currently visible counts. It is a stopping heuristic, not a Bayesian optimal-design solution.

## Failure handling and auditability

Missing historical arms are UNSUPPORTED. Provider failures are distinct. Prediction, action, evidence, plan, configuration, and policy fingerprints are persisted. The offline runner contains no provider initialization path.
"""
    statistical = f"""# SemanticFlow Milestone 5.2 Statistical Protocol

## Hypothesis and estimands

The hypothesis is that uncertainty-aware active qualification can reduce unsafe selection and wasted profiling while preserving economic opportunities. Primary quality estimands are representative-distribution recall, FNR, agreement, and changed-decision rate relative to separate Jev inference. They are consistency estimands, not human accuracy.

## Bounds and sequential validity

Wilson intervals are descriptive. Certification uses one-sided Clopper-Pearson exact bounds. Family alpha {data["uncertainty_validation"]["family_alpha"]:.3f} is allocated in advance across candidates, checkpoints, and four simultaneous bounds by Bonferroni. The per-bound allocation is {data["uncertainty_validation"]["allocated_one_sided_alpha"]:.8f}. This finite-family construction remains valid under outcome-dependent stopping because every possible inspection is included before replay.

## Qualification

At least 40 representative records and three representative positives remain necessary. Certification additionally requires lower recall and agreement bounds and upper FNR and change-rate bounds to cross the original thresholds. Zero positives leave recall undefined. Evidence may be QUALIFIED, REJECTED, INSUFFICIENT_SUPPORT, UNCERTAIN, UNSUPPORTED, or PROVIDER_FAILURE.

## Active sampling bias

Positive, threshold, disagreement, and cost-focused discovery samples are diagnostic only. Weighted certification is deliberately not implemented: the historical replay does not provide a sufficiently justified sequential unequal-probability design. Representative certification is isolated by record identity.

## Dependence and interpretation

Workloads share records, predicates, and provider measurements; offline policy rows are not independent confirmations. M5.1 Stage D is retrospective M5.2 development evidence. A future evaluation must freeze all rules and use new records before outcomes are available.
"""
    results = f"""# SemanticFlow Milestone 5.2 Offline Results

## Status

All requested subsystems are implemented and exercised against stored real Jev evidence. No provider calls occurred. Results are retrospective development diagnostics.

## Policy comparison

| Policy | Unsafe | Missed opportunities | Selected | Profiling requests | Profiling cost | Net benefit |
|---|---:|---:|---:|---:|---:|---:|
{policy_table_rows}

## Banking77 replay

The historical selection evidence contained {data["banking"]["selection_evidence"]["reference_positives"]} positives and {data["banking"]["selection_evidence"]["false_negative"]} false negatives. The old point gate passed: {data["banking"]["m51_point_gate_passed"]}. The strict state is **{data["banking"]["m52_strict_estimate"]["state"]}** because 3/3 cannot establish 95% population recall under the frozen family-wise sequential rule. The controller therefore cannot certify the known failed batch; depending on reuse horizon it profiles further or falls back when information cannot repay its cost.

## Beauty replay

Selection evidence had {data["beauty"]["selection_evidence"]["reference_positives"]} positives and {data["beauty"]["selection_evidence"]["false_negative"]} false negatives. Both the old point gate and strict rule fail on permitted selection evidence. Active discovery can expose mechanisms and prioritize records, but cannot turn biased enrichment into representative certification. The later Stage D pass is not leaked back into the decision.

## Sparse positives

Four full-batch workloads have zero held-out complete-query positives. Recall remains undefined, strict qualification returns insufficient support, and the controller safely falls back or stops when further evidence is uneconomic.

## Main finding

The strict procedure removes unsafe historical certification but often abstains. This is statistically defensible and economically negative at short horizons. The offline evidence therefore does not establish that M5.2 improves deployment utility; it establishes a leakage-safe implementation and identifies the positive-support burden for a fresh pilot.

## Limitations

Stage D is known retrospective evidence, no human labels exist, weighted active certification is not justified, and the measured operator catalog bounds replay coverage. Optimization CPU cost is unmetered.
"""
    pilot = f"""# SemanticFlow Milestone 5.2 Small Live Pilot Proposal

## Status

Proposal only. No live execution is authorized.

## Scope

- Three workloads: Banking77 card/unauthorized/refund, Beauty failure/usability, and SMS prize/urgency/contact.
- Per workload: 60 representative paired records, up to 20 paid discovery screens, and up to 20 corresponding candidate measurements.
- Screening inference is explicitly counted; no free likely-positive or threshold ranking is assumed.

## Budget

- Expected physical requests: {data["pilot_budget"]["expected_physical_requests"]}.
- Physical request ceiling: {data["pilot_budget"]["physical_request_ceiling"]}.
- Expected named-question evaluations: {data["pilot_budget"]["expected_named_question_evaluations"]}.
- Expected input tokens: {data["pilot_budget"]["expected_input_tokens"]:,}.
- Expected cost proxy: ${data["pilot_budget"]["expected_cost_proxy_usd"]:.6f}.
- Conservative input tokens: {data["pilot_budget"]["conservative_input_tokens"]:,}.
- Conservative cost proxy: ${data["pilot_budget"]["conservative_cost_proxy_usd"]:.6f}.
- Retry-inclusive attempt ceiling: {data["pilot_budget"]["retry_inclusive_attempt_ceiling"]} with at most two retries per request.

## Independent evaluation and positive feasibility

The pilot is development profiling, not the final independent test. Before scheduling, prevalence estimates must show whether representative certification is feasible; 400 records is not presumed sufficient. A later frozen evaluation needs new records, at least one new domain, and independently double-annotated representative and diagnostic strata.

## Authorization checkpoint

Freeze the policy hash, qualification and sampling rules, candidate catalog, cost estimator, budgets, schedule, model, and dataset fingerprints. Validate them fail-closed before provider initialization. Historical authorization cannot be reused. Execution requires the exact new phrase `AUTHORIZE M5.2 SMALL LIVE PILOT <FROZEN_SCOPE_FINGERPRINT>`. Attempts must be reserved atomically, interruption must not auto-resume, and no automatic paid continuation is allowed.
"""
    for name, text in {
        "milestone_5_2_system_design.md": system,
        "milestone_5_2_statistical_protocol.md": statistical,
        "milestone_5_2_offline_results.md": results,
        "milestone_5_2_live_pilot_proposal.md": pilot,
    }.items():
        (docs / name).write_text(text, encoding="utf-8")
