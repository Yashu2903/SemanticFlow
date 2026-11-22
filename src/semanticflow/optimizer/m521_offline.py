# ruff: noqa: E501
from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from semanticflow.optimizer.m51_final_audit import sha256_file
from semanticflow.optimizer.m52_evidence import EvidenceCounts
from semanticflow.optimizer.m521_error_budget import (
    QualificationRequirements,
    legacy_requirements,
    nonredundant_requirements,
    redundancy_audit,
    unequal_requirements,
)
from semanticflow.optimizer.m521_feasibility import (
    EvidenceBudget,
    analyze_feasibility,
    maximum_tolerable_errors,
    zero_error_minimum,
)
from semanticflow.optimizer.m521_risk_estimation import exploratory_beta_risk
from semanticflow.optimizer.m521_statistical_power import (
    certification_power,
    estimate_prevalence,
    expected_records_for_positives,
    joint_certification_power_approximation,
    minimum_records_for_support_probability,
    positive_support_probability,
)

ROOT = Path("results/milestone_5_2_1/offline_development")
DOCS = Path("docs")
M51 = Path("results/milestone_5_1/selection_stage_results")
M51_AUDIT = Path("results/milestone_5_1/final_research_audit")
M52 = Path("results/milestone_5_2/offline_development")
HORIZONS = (1, 5, 10, 25, 50, 100, 250, 1000)


def _json(path: Path) -> Any:
    return json.loads(path.read_text("utf-8"))


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = list(rows)
    fields = list(dict.fromkeys(key for row in values for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def _policy_rows() -> tuple[list[QualificationRequirements], list[dict[str, Any]]]:
    policies = [legacy_requirements(), nonredundant_requirements(), unequal_requirements()]
    rows = []
    sensitivity = [
        QualificationRequirements(
            policy_version=policy_version,
            distinct_estimands=bounds,
            candidate_count=candidates,
            checkpoint_count=checkpoints,
            recall_alpha_fraction=recall_fraction,
        )
        for policy_version, bounds, recall_fraction in (
            (policies[0].policy_version, 4, 0.5),
            (policies[1].policy_version, 2, 0.5),
        )
        for candidates in (1, 4, 8)
        for checkpoints in (1, 4, 8)
    ]
    configurations = [*policies, *sensitivity]
    seen: set[str] = set()
    for policy in configurations:
        if policy.policy_fingerprint in seen:
            continue
        seen.add(policy.policy_fingerprint)
        recall_min = zero_error_minimum(policy.minimum_recall, policy.recall_alpha)
        agreement_min = zero_error_minimum(policy.minimum_agreement, policy.agreement_alpha)
        rows.append(
            {
                "policy_version": policy.policy_version.value,
                "scenario": (
                    "primary"
                    if policy.policy_fingerprint
                    in {item.policy_fingerprint for item in policies}
                    else "candidate_checkpoint_sensitivity"
                ),
                "family_alpha": policy.family_alpha,
                "candidate_count": policy.candidate_count,
                "checkpoint_count": policy.checkpoint_count,
                "reported_bounds": policy.distinct_estimands,
                "recall_alpha": policy.recall_alpha,
                "agreement_alpha": policy.agreement_alpha,
                "minimum_positive_references_zero_error": recall_min,
                "minimum_representative_records_zero_error": agreement_min,
                "binding_constraint_at_100_percent_prevalence": (
                    "agreement_changed_decision" if agreement_min >= recall_min else "recall_fnr"
                ),
                "policy_fingerprint": policy.policy_fingerprint,
            }
        )
    return policies, rows


def _historical_quality() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = _json(M51_AUDIT / "canonical_plan_quality_results.json")[
        "all_measured_plan_quality"
    ]
    full = {}
    selections = _json(M51 / "frozen_strategy_selections.json")["strategies"]
    full_ids = {
        row["workload_id"]: row["plan_id"]
        for row in selections
        if row["strategy"] == "full_multi_decision_batch"
    }
    for row in rows:
        if full_ids.get(row["workload_id"]) == row["plan_id"]:
            full[row["workload_id"]] = row
    return rows, full


def _feasibility_certificates(
    policies: Sequence[QualificationRequirements], full_quality: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    catalog = _json(M51 / "frozen_physical_plan_catalog.json")["plans"]
    plans = {row["plan_id"]: row for row in catalog}
    selections = _json(M51 / "frozen_strategy_selections.json")["strategies"]
    full_ids = {
        row["workload_id"]: row["plan_id"]
        for row in selections
        if row["strategy"] == "full_multi_decision_batch"
    }
    output = []
    for workload_id, plan_id in full_ids.items():
        quality = full_quality[workload_id]
        for policy in policies:
            certificate = analyze_feasibility(
                candidate_plan_fingerprint=plans[plan_id]["plan_fingerprint"],
                requirements=policy,
                counts=EvidenceCounts(
                    observations=0,
                    reference_positives=0,
                    reference_negatives=0,
                    true_positive=0,
                    true_negative=0,
                    false_positive=0,
                    false_negative=0,
                ),
                budget=EvidenceBudget(
                    maximum_representative_records=60,
                    remaining_representative_records=60,
                    maximum_positive_references=60,
                    remaining_physical_requests=1200,
                    remaining_input_tokens=532_412,
                    remaining_cost_usd=0.022361304,
                    requests_per_representative_pair=2,
                    expected_input_tokens_per_pair=720,
                    expected_cost_per_pair_usd=0.00003024,
                ),
            )
            output.append(
                {
                    "workload_id": workload_id,
                    "plan_id": plan_id,
                    "historical_stage_d_positives": quality["reference_positive_count"],
                    "retrospective_only": True,
                    **certificate.model_dump(mode="json"),
                }
            )
    return output


def _prevalence(full_quality: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for workload_id, row in full_quality.items():
        estimate = estimate_prevalence(
            int(row["reference_positive_count"]),
            int(row["sample_count"]),
            provenance="M5.1 Stage D retrospective development evidence",
        )
        required = 153
        output.append(
            {
                "workload_id": workload_id,
                **estimate.model_dump(mode="json"),
                "required_positive_references": required,
                "expected_records_at_point_estimate": expected_records_for_positives(
                    required, estimate.point_estimate
                ),
                "support_probability_at_400_point": positive_support_probability(
                    400, estimate.point_estimate, required
                ),
                "support_probability_at_400_lower": positive_support_probability(
                    400, estimate.lower, required
                ),
                "support_probability_at_400_upper": positive_support_probability(
                    400, estimate.upper, required
                ),
            }
        )
    return output


def _workload_diagnostics() -> list[dict[str, Any]]:
    from semanticflow.optimizer.m52_replay import HistoricalReplayMatrix

    matrix = HistoricalReplayMatrix.stage_d()
    selections = _json(M51 / "frozen_strategy_selections.json")["strategies"]
    full_ids = {
        row["workload_id"]: row["plan_id"]
        for row in selections
        if row["strategy"] == "full_multi_decision_batch"
    }
    singles: dict[tuple[str, str], dict[str, bool]] = {}
    for row in matrix.rows:
        if len(row["predicate_ids"]) != 1:
            continue
        predicate = row["predicate_ids"][0]
        singles[(row["dataset"], predicate)] = singles.get(
            (row["dataset"], predicate), {}
        ) | {row["record_id"]: bool(row["predicate_decisions"][predicate])}
    output = []
    for workload_id, plan_id in full_ids.items():
        plan = matrix.plans[str(plan_id)]
        predicates = list(plan["logical_predicate_ids"])
        record_ids = matrix.records(str(plan["dataset"]))
        decisions = {
            predicate: singles[(str(plan["dataset"]), predicate)] for predicate in predicates
        }
        individual = {
            predicate: sum(decisions[predicate][record] for record in record_ids)
            / len(record_ids)
            for predicate in predicates
        }
        conditionals = []
        reached = list(record_ids)
        for predicate in predicates:
            passed = [record for record in reached if decisions[predicate][record]]
            conditionals.append(
                {
                    "predicate_id": predicate,
                    "conditioning_count": len(reached),
                    "conditional_prevalence": len(passed) / len(reached) if reached else None,
                }
            )
            reached = passed
        correlations = []
        for index, left in enumerate(predicates):
            for right in predicates[index + 1 :]:
                left_values = [int(decisions[left][record]) for record in record_ids]
                right_values = [int(decisions[right][record]) for record in record_ids]
                mean_left = sum(left_values) / len(left_values)
                mean_right = sum(right_values) / len(right_values)
                covariance = sum(
                    (a - mean_left) * (b - mean_right)
                    for a, b in zip(left_values, right_values, strict=True)
                ) / len(left_values)
                denominator = math.sqrt(
                    mean_left * (1 - mean_left) * mean_right * (1 - mean_right)
                )
                correlations.append(
                    {
                        "left": left,
                        "right": right,
                        "phi": covariance / denominator if denominator else None,
                    }
                )
        output.append(
            {
                "workload_id": workload_id,
                "dataset": plan["dataset"],
                "predicates": predicates,
                "individual_prevalence": individual,
                "conditional_prevalence_in_declared_order": conditionals,
                "pairwise_phi": correlations,
                "complete_query_positives": len(reached),
                "complete_query_prevalence": len(reached) / len(record_ids),
                "retrospective_development_only": True,
                "future_workload_selection_must_not_use_test_outcomes": True,
            }
        )
    return output


def _support_curves(required: int) -> list[dict[str, Any]]:
    rows = []
    for prevalence in (0.005, 0.01, 0.02, 0.05, 0.10, 0.20):
        for records in (60, 100, 400, 1000, 3000, 10_000, 20_400, 30_000):
            rows.append(
                {
                    "prevalence": prevalence,
                    "representative_records": records,
                    "required_positive_references": required,
                    "expected_positive_references": records * prevalence,
                    "probability_sufficient_support": positive_support_probability(
                        records, prevalence, required
                    ),
                    "calculation": "exact_binomial",
                }
            )
    return rows


def _power_curves(policies: Sequence[QualificationRequirements]) -> list[dict[str, Any]]:
    rows = []
    for policy in policies:
        recall_trials = zero_error_minimum(policy.minimum_recall, policy.recall_alpha)
        agreement_trials = zero_error_minimum(policy.minimum_agreement, policy.agreement_alpha)
        for quality in (0.95, 0.97, 0.98, 0.99, 0.995, 1.0):
            recall = certification_power(
                trials=recall_trials,
                true_quality=quality,
                threshold=policy.minimum_recall,
                alpha=policy.recall_alpha,
            )
            agreement = certification_power(
                trials=agreement_trials,
                true_quality=quality,
                threshold=policy.minimum_agreement,
                alpha=policy.agreement_alpha,
            )
            rows.extend(
                [
                    {
                        "policy_version": policy.policy_version.value,
                        "estimand": "recall",
                        **recall.model_dump(mode="json"),
                    },
                    {
                        "policy_version": policy.policy_version.value,
                        "estimand": "agreement",
                        **agreement.model_dump(mode="json"),
                    },
                ]
            )
    return rows


def _same_horizon() -> list[dict[str, Any]]:
    rows = _csv(M52 / "policy_ablation_results.csv")
    policies = sorted({row["policy_name"] for row in rows})
    output = []
    for policy in policies:
        selected = [row for row in rows if row["policy_name"] == policy]
        profiling_cost = sum(float(row["profiling_cost_usd"]) for row in selected)
        realized = sum(float(row["realized_execution_saving_usd"]) for row in selected)
        admissible = sum(
            float(row["realized_execution_saving_usd"])
            for row in selected
            if row["unsafe_selection"].lower() == "false"
        )
        unsafe = sum(row["unsafe_selection"].lower() == "true" for row in selected)
        missed = sum(
            row["missed_admissible_opportunity"].lower() == "true" for row in selected
        )
        for horizon in HORIZONS:
            output.append(
                {
                    "policy_name": policy,
                    "reuse_horizon": horizon,
                    "profiling_scenario": "standalone_historical_replay",
                    "profiling_cost_usd": profiling_cost,
                    "selected_plans": sum(row["used_fallback"].lower() == "false" for row in selected),
                    "realized_execution_saving_per_run_usd": realized,
                    "gross_execution_benefit_usd": horizon * realized,
                    "net_benefit_usd": horizon * realized - profiling_cost,
                    "quality_admissible_gross_benefit_usd": horizon * admissible,
                    "quality_admissible_net_benefit_usd": horizon * admissible - profiling_cost,
                    "unsafe_selections": unsafe,
                    "missed_opportunities": missed,
                    "optimization_cpu_cost_usd": None,
                }
            )
    return output


def _economics(policies: Sequence[QualificationRequirements]) -> dict[str, Any]:
    controlled = {
        "banking77": _json(M52 / "banking77_failure_replay.json"),
        "beauty": _json(M52 / "beauty_missed_opportunity_replay.json"),
    }
    output = {}
    horizon_rows = _csv(M52 / "reuse_horizon_analysis.csv")
    for label, item in controlled.items():
        score = item["retrospective_stage_d_score"]
        prevalence = score["reference_positive_count"] / score["sample_count"]
        required = zero_error_minimum(
            policies[0].minimum_recall, policies[0].recall_alpha
        )
        records = expected_records_for_positives(required, prevalence)
        records_95 = minimum_records_for_support_probability(required, prevalence, 0.95)
        pair_cost = item["selection_measurement_cost_usd"] / 100
        profile_cost = None if records is None else records * pair_cost
        profile_cost_95 = None if records_95 is None else records_95 * pair_cost
        saving = score["execution_saving_usd"]
        acquisition = next(
            row
            for row in horizon_rows
            if row["workload_id"] == item["workload_id"]
            and row["reuse_horizon"] == "1000"
        )
        output[label] = {
            "workload_id": item["workload_id"],
            "retrospective_prevalence": prevalence,
            "required_positive_references": required,
            "expected_records_for_support": records,
            "records_for_95_percent_support_probability": records_95,
            "support_probability_at_expected_records": (
                None
                if records is None
                else positive_support_probability(records, prevalence, required)
            ),
            "pair_cost_usd": pair_cost,
            "historical_requests_per_representative_record": (
                int(acquisition["profiling_requests"]) / 100
            ),
            "historical_named_evaluations_per_representative_record": (
                int(acquisition["profiling_named_question_evaluations"]) / 100
            ),
            "historical_input_tokens_per_representative_record": (
                int(acquisition["profiling_input_tokens"]) / 100
            ),
            "historical_output_tokens_per_representative_record": (
                int(acquisition["profiling_output_tokens"]) / 100
            ),
            "expected_profile_cost_usd": profile_cost,
            "profile_cost_for_95_percent_support_usd": profile_cost_95,
            "execution_saving_per_run_usd": saving,
            "break_even_reuses_if_certified": (
                None if profile_cost is None or saving <= 0 else math.ceil(profile_cost / saving)
            ),
            "break_even_reuses_at_95_percent_support_design": (
                None
                if profile_cost_95 is None or saving <= 0
                else math.ceil(profile_cost_95 / saving)
            ),
            "quality_status": (
                "KNOWN_HISTORICAL_FAILURE" if label == "banking77" else "RETROSPECTIVE_PASS"
            ),
            "not_a_prospective_forecast": True,
        }
    return output


def _stop_before_spend() -> list[dict[str, Any]]:
    rows = _csv(M52 / "policy_ablation_results.csv")
    output = []
    for row in rows:
        if row["policy_name"] != "full_m52":
            continue
        output.append(
            {
                "workload_id": row["workload_id"],
                "m52_requests": int(row["profiling_requests"]),
                "m52_cost_usd": float(row["profiling_cost_usd"]),
                "m521_requests": 0,
                "m521_cost_usd": 0.0,
                "requests_avoided_mathematical_impossibility": int(row["profiling_requests"]),
                "cost_avoided_mathematical_impossibility_usd": float(row["profiling_cost_usd"]),
                "requests_avoided_negative_net_value": 0,
                "requests_avoided_unsupported_coverage": 0,
                "requests_avoided_budget_exhaustion": 0,
                "selected_plan": "existing_adaptive",
                "quality_admissible_net_benefit_usd": 0.0,
            }
        )
    return output


def _information_efficiency() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from semanticflow.optimizer.m51_selection_analysis import load_evidence
    from semanticflow.optimizer.m52_active_profiler import SamplingStrategy
    from semanticflow.optimizer.m52_cost_model import M52CostEstimator
    from semanticflow.optimizer.m52_offline import (
        _dataset_fingerprints,
        _run_policy,
        _strategy_map,
    )
    from semanticflow.optimizer.m52_planner import PolicyMode
    from semanticflow.optimizer.m52_replay import HistoricalReplayMatrix

    selection = HistoricalReplayMatrix.selection()
    stage_d = HistoricalReplayMatrix.stage_d()
    estimator = M52CostEstimator(load_evidence()["normalized"], allowed_partition="selection")
    selections = _json(M51 / "frozen_strategy_selections.json")["strategies"]
    adaptive = _strategy_map(selections, "existing_adaptive")
    full = _strategy_map(selections, "full_multi_decision_batch")
    fingerprints = _dataset_fingerprints()
    policy = []
    info = []
    conditions = (
        ("random_fixed_budget", SamplingStrategy.RANDOM_REPRESENTATIVE, True, 1000),
        ("active_fixed_budget", SamplingStrategy.HYBRID_ACTIVE, True, 1000),
        ("random_identical_stopping", SamplingStrategy.RANDOM_REPRESENTATIVE, False, 10),
        ("active_identical_stopping", SamplingStrategy.HYBRID_ACTIVE, False, 10),
    )
    for label, strategy, fixed, horizon in conditions:
        selected = []
        traces = []
        for workload in selection.config["workloads"]:
            workload_id = str(workload["id"])
            plan = selection.plans[str(full[workload_id]["plan_id"])]
            fallback = selection.plans[str(adaptive[workload_id]["plan_id"])]
            row, trace = _run_policy(
                template=selection,
                estimator=estimator,
                stage_d=stage_d,
                workload=workload,
                plan=plan,
                fallback=fallback,
                mode=PolicyMode.STRICT_CERTIFICATION,
                strategy=strategy,
                budget_records=80,
                reuse_horizon=horizon,
                force_fixed_sample=fixed,
                dataset_fingerprint=fingerprints[str(workload["dataset"])],
            )
            selected.append(row)
            traces.append(trace)
        requests = sum(int(row["profiling_requests"]) for row in selected)
        reps = sum(int(row["representative_samples"]) for row in selected)
        discoveries = sum(int(row["discovery_samples"]) for row in selected)
        cost = sum(float(row["profiling_cost_usd"]) for row in selected)
        measurements = [
            measurement
            for trace in traces
            for measurement in trace["final_state"]["evidence"]["measurements"]
        ]
        representative_positives = sum(
            measurement["track"] == "certification"
            and measurement["reference_decision"] is True
            for measurement in measurements
        )
        discovery_positives = sum(
            measurement["track"] == "discovery"
            and measurement["reference_decision"] is True
            for measurement in measurements
        )
        disagreements = sum(
            measurement["reference_decision"] != measurement["candidate_decision"]
            for measurement in measurements
        )
        policy.append(
            {
                "policy": label,
                "source_policy": "controlled_m5.2_replay",
                "requests": requests,
                "profiling_cost_usd": cost,
                "representative_samples": reps,
                "discovery_samples": discoveries,
                "qualified_plans": 0,
                "fixed_record_budget": 80,
                "reuse_horizon": horizon,
                "identical_stopping_rule": not fixed,
            }
        )
        info.append(
            {
                "policy": label,
                "representative_samples_per_request": reps / requests if requests else None,
                "discovery_samples_per_request": discoveries / requests if requests else None,
                "representative_positives": representative_positives,
                "discovery_positives": discovery_positives,
                "total_positives_per_request": (
                    (representative_positives + discovery_positives) / requests
                    if requests
                    else None
                ),
                "confirmed_disagreements": disagreements,
                "confirmed_disagreements_per_request": disagreements / requests if requests else None,
                "profiling_cost_usd": cost,
                "representative_certification_progress": reps,
                "interpretation": "discovery positives are diagnostic, never certification support",
            }
        )
    for strategy in ("random_feasibility_stop", "active_feasibility_stop"):
        policy.append(
            {
                "policy": strategy,
                "source_policy": "m5.2.1",
                "requests": 0,
                "profiling_cost_usd": 0.0,
                "representative_samples": 0,
                "discovery_samples": 0,
                "qualified_plans": 0,
                "fixed_record_budget": 80,
                "reuse_horizon": 10,
                "identical_stopping_rule": True,
            }
        )
        info.append(
            {
                "policy": strategy,
                "representative_samples_per_request": None,
                "discovery_samples_per_request": None,
                "uncertainty_reduction_per_dollar": None,
                "representative_positives": 0,
                "discovery_positives": 0,
                "total_positives_per_request": None,
                "confirmed_disagreements": 0,
                "confirmed_disagreements_per_request": None,
                "profiling_cost_usd": 0.0,
                "representative_certification_progress": 0,
                "interpretation": "identical feasibility stopping dominates acquisition strategy",
            }
        )
    return policy, info


def build() -> dict[str, Any]:
    policies, allocation_rows = _policy_rows()
    _, full_quality = _historical_quality()
    prevalence = _prevalence(full_quality)
    economics = _economics(policies)
    certificates = _feasibility_certificates(policies, full_quality)
    minimum_rows = []
    for row in allocation_rows:
        for denominator in (
            row["minimum_positive_references_zero_error"],
            row["minimum_representative_records_zero_error"],
        ):
            minimum_rows.append(
                {
                    **row,
                    "evaluated_denominator": denominator,
                    "maximum_recall_errors_at_denominator": maximum_tolerable_errors(
                        denominator, 0.95, row["recall_alpha"]
                    ),
                    "maximum_agreement_errors_at_denominator": maximum_tolerable_errors(
                        denominator, 0.98, row["agreement_alpha"]
                    ),
                }
            )
    banking = economics["banking77"]
    beauty = economics["beauty"]
    legacy = policies[0]
    banking_power = joint_certification_power_approximation(
        representative_records=banking["expected_records_for_support"],
        positive_prevalence=banking["retrospective_prevalence"],
        required_positive_trials=banking["required_positive_references"],
        true_recall=0.99,
        recall_threshold=0.95,
        recall_alpha=legacy.recall_alpha,
        true_agreement=0.995,
        agreement_threshold=0.98,
        agreement_alpha=legacy.agreement_alpha,
    )
    beauty_power = joint_certification_power_approximation(
        representative_records=beauty["expected_records_for_support"],
        positive_prevalence=beauty["retrospective_prevalence"],
        required_positive_trials=beauty["required_positive_references"],
        true_recall=0.99,
        recall_threshold=0.95,
        recall_alpha=legacy.recall_alpha,
        true_agreement=0.995,
        agreement_threshold=0.98,
        agreement_alpha=legacy.agreement_alpha,
    )
    ablation, info = _information_efficiency()
    stop = _stop_before_spend()
    controlled = {
        "banking77": {
            **banking,
            "qualification_power_scenario": banking_power,
            "sixty_record_pilot_feasible": False,
            "feasibility_decision": "fallback_before_spend",
            "unsafe_selection_mechanism": "3/3 positives concealed uncertainty; held-out miss was 0.50 to 0.49 threshold crossing",
        },
        "beauty": {
            **beauty,
            "qualification_power_scenario": beauty_power,
            "sixty_record_pilot_feasible": False,
            "selection_time_conclusion": "additional evidence could help but expected certification expense exceeds small-pilot budget",
            "heldout_result_not_leaked": True,
        },
    }
    risk = []
    for label in controlled:
        replay = _json(
            M52
            / (
                "banking77_failure_replay.json"
                if label == "banking77"
                else "beauty_missed_opportunity_replay.json"
            )
        )
        counts = replay["selection_evidence"]
        estimate = exploratory_beta_risk(counts["false_negative"], counts["reference_positives"])
        risk.append({"workload": label, **estimate.model_dump(mode="json")})
    mean_request_cost = 0.008586732 / 576
    pilot_a = {
        "status": "proposal_only_not_authorized",
        "objective": "instrumentation, acquisition accounting, provenance, disagreement diagnostics, restart safety",
        "formal_qualification_claim": False,
        "workloads": 3,
        "representative_records_per_workload": 20,
        "discovery_screens_per_workload": 10,
        "paired_representative_requests": 220,
        "discovery_screen_requests": 30,
        "discovery_candidate_requests": 30,
        "expected_physical_requests": 280,
        "expected_named_question_evaluations": 430,
        "physical_request_ceiling": 360,
        "expected_cost_proxy_usd": 280 * mean_request_cost,
        "conservative_cost_proxy_usd": 360 * mean_request_cost * 1.25,
        "retry_inclusive_attempt_ceiling": 1080,
        "authorization_required": True,
    }
    pilot_b = {
        "status": "not_recommended_on_current_workloads",
        "objective": "formal frequentist qualification",
        "banking77": banking,
        "beauty": beauty,
        "design_support_probability": 0.95,
        "banking77_design_physical_requests": math.ceil(
            banking["records_for_95_percent_support_probability"]
            * banking["historical_requests_per_representative_record"]
        ),
        "beauty_design_physical_requests": math.ceil(
            beauty["records_for_95_percent_support_probability"]
            * beauty["historical_requests_per_representative_record"]
        ),
        "combined_design_physical_requests": (
            math.ceil(
                banking["records_for_95_percent_support_probability"]
                * banking["historical_requests_per_representative_record"]
            )
            + math.ceil(
                beauty["records_for_95_percent_support_probability"]
                * beauty["historical_requests_per_representative_record"]
            )
        ),
        "combined_design_named_question_evaluations": math.ceil(
            banking["records_for_95_percent_support_probability"]
            * banking["historical_named_evaluations_per_representative_record"]
            + beauty["records_for_95_percent_support_probability"]
            * beauty["historical_named_evaluations_per_representative_record"]
        ),
        "combined_design_input_tokens": round(
            (
                banking["profile_cost_for_95_percent_support_usd"]
                + beauty["profile_cost_for_95_percent_support_usd"]
            )
            / 0.042
            * 1_000_000
        ),
        "combined_design_cost_proxy_usd": (
            banking["profile_cost_for_95_percent_support_usd"]
            + beauty["profile_cost_for_95_percent_support_usd"]
        ),
        "conservative_cost_proxy_usd": 1.5
        * (
            banking["profile_cost_for_95_percent_support_usd"]
            + beauty["profile_cost_for_95_percent_support_usd"]
        ),
        "retry_inclusive_attempt_ceiling": 3
        * math.ceil(
            1.5
            * (
                banking["records_for_95_percent_support_probability"]
                * banking["historical_requests_per_representative_record"]
                + beauty["records_for_95_percent_support_probability"]
                * beauty["historical_requests_per_representative_record"]
            )
        ),
        "reason": "expected representative evidence costs are large relative to per-run savings and true-threshold power is low",
        "authorization_required": True,
    }
    pilot_c = {
        "status": "proposal_only_not_authorized",
        "objective": "independent validation of an exploratory risk predictor; no formal guarantee",
        "workloads": 3,
        "fresh_representative_records_per_workload": 200,
        "calibration_fraction": 0.5,
        "test_fraction": 0.5,
        "expected_physical_requests": 2200,
        "expected_named_question_evaluations": 3200,
        "physical_request_ceiling": 3300,
        "expected_cost_proxy_usd": 2200 * mean_request_cost,
        "conservative_cost_proxy_usd": 3300 * mean_request_cost * 1.25,
        "retry_inclusive_attempt_ceiling": 9900,
        "metrics": ["Brier score", "log loss", "calibration error", "recall/FNR error"],
        "authorization_required": True,
    }
    return {
        "policies": policies,
        "allocation_rows": allocation_rows,
        "minimum_rows": minimum_rows,
        "certificates": certificates,
        "prevalence": prevalence,
        "workload_diagnostics": _workload_diagnostics(),
        "support": _support_curves(153),
        "power": _power_curves(policies),
        "economics": economics,
        "stop": stop,
        "ablation": ablation,
        "information": info,
        "same_horizon": _same_horizon(),
        "controlled": controlled,
        "risk": risk,
        "pilot_a": pilot_a,
        "pilot_b": pilot_b,
        "pilot_c": pilot_c,
        "future_protocol": {
            "policy_freeze_before_fresh_outcomes": True,
            "fresh_records_required": True,
            "calibration_test_isolation": True,
            "human_labels_separate_study": True,
            "historical_stage_d_independent_for_m521": False,
            "frozen_items": [
                "policy fingerprint",
                "error allocation",
                "checkpoints",
                "candidate family",
                "sampling policy",
                "stopping rule",
                "estimator",
                "schedule",
                "dataset partitions",
            ],
        },
    }


def generate(output: Path = ROOT, docs: Path = DOCS) -> dict[str, Any]:
    data = build()
    output.mkdir(parents=True, exist_ok=True)
    integrity = {
        "schema_version": "m5.2.1-historical-integrity-v1",
        "m51_audit_valid": _json(M51_AUDIT / "historical_integrity_audit.json")["valid"],
        "m52_manifest_sha256": sha256_file(M52 / "artifact_integrity_manifest.json"),
        "m51_frozen_catalog_sha256": sha256_file(M51 / "frozen_physical_plan_catalog.json"),
        "m51_frozen_selections_sha256": sha256_file(M51 / "frozen_strategy_selections.json"),
        "historical_measurements_modified": False,
        "new_jev_requests": 0,
        "provider_initialized": False,
    }
    _write_json(output / "historical_integrity_audit.json", integrity)
    _write_json(output / "qualification_feasibility.json", data["certificates"])
    _write_json(output / "statistical_redundancy_audit.json", redundancy_audit())
    _write_csv(output / "error_allocation_comparison.csv", data["allocation_rows"])
    _write_csv(output / "minimum_evidence_requirements.csv", data["minimum_rows"])
    _write_json(
        output / "positive_prevalence_estimates.json",
        {
            "full_conjunction_prevalence": data["prevalence"],
            "predicate_and_correlation_diagnostics": data["workload_diagnostics"],
            "workload_selection_guidelines": [
                "require realistic semantic meaning and nontrivial conjunction prevalence",
                "preserve early-rejection opportunity and plausible batching benefit",
                "estimate conditional prevalence and correlation on development data only",
                "retain rare-positive conjunctions as stress tests, not certification targets",
                "freeze workload definitions before future test outcomes are visible",
            ],
        },
    )
    _write_csv(output / "positive_support_probability.csv", data["support"])
    _write_csv(output / "certification_power_curves.csv", data["power"])
    _write_json(output / "economic_feasibility.json", data["economics"])
    _write_csv(output / "stop_before_spend_results.csv", data["stop"])
    _write_csv(output / "profiling_policy_ablation.csv", data["ablation"])
    _write_csv(output / "information_efficiency.csv", data["information"])
    _write_json(output / "banking77_feasibility_replay.json", data["controlled"]["banking77"])
    _write_json(output / "beauty_feasibility_replay.json", data["controlled"]["beauty"])
    rare = [item for item in data["prevalence"] if item["positives"] <= 1]
    _write_json(
        output / "rare_positive_stress_test.json",
        {
            "workloads": rare,
            "zero_positive_is_not_zero_prevalence": True,
            "formal_recall_undefined": True,
            "decision": "fallback_before_spend_when_record_budget_cannot_certify",
        },
    )
    _write_csv(output / "same_horizon_economics.csv", data["same_horizon"])
    _write_json(output / "risk_policy_analysis.json", data["risk"])
    _write_json(output / "pilot_a_instrumentation_proposal.json", data["pilot_a"])
    _write_json(output / "pilot_b_qualification_proposal.json", data["pilot_b"])
    _write_json(output / "pilot_c_risk_validation_proposal.json", data["pilot_c"])
    _write_json(output / "future_independent_evaluation_protocol.json", data["future_protocol"])
    _write_docs(docs, data)
    artifacts = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_integrity_manifest.json"
    }
    documents = {
        name: sha256_file(docs / name)
        for name in (
            "milestone_5_2_1_system_design.md",
            "milestone_5_2_1_statistical_feasibility.md",
            "milestone_5_2_1_offline_results.md",
            "milestone_5_2_1_future_pilot_protocol.md",
        )
    }
    manifest = {
        "schema_version": "m5.2.1-artifact-integrity-v1",
        "artifacts": artifacts,
        "documents": documents,
        "historical_integrity": integrity,
        "new_jev_requests": 0,
    }
    _write_json(output / "artifact_integrity_manifest.json", manifest)
    return {"status": "complete", "artifacts": len(artifacts) + 1, "new_jev_requests": 0}


def _write_docs(docs: Path, data: Mapping[str, Any]) -> None:
    docs.mkdir(parents=True, exist_ok=True)
    allocation = data["allocation_rows"]
    table = "\n".join(
        f"| {row['policy_version']} | {row['recall_alpha']:.9f} | {row['agreement_alpha']:.9f} | {row['minimum_positive_references_zero_error']} | {row['minimum_representative_records_zero_error']} |"
        for row in allocation
    )
    legacy_recall_power = "\n".join(
        f"| {float(row['true_quality']):.3f} | {float(row['certification_probability']):.6f} | {row['maximum_errors']} |"
        for row in data["power"]
        if row["policy_version"] == "m52_legacy_strict" and row["estimand"] == "recall"
    )
    active_table = "\n".join(
        f"| {row['policy']} | {row['requests']} | ${float(row['profiling_cost_usd']):.9f} | {row['representative_samples']} | {row['discovery_samples']} |"
        for row in data["ablation"]
    )
    horizons_table = "\n".join(
        f"| {row['policy_name']} | {row['reuse_horizon']} | ${float(row['profiling_cost_usd']):.9f} | ${float(row['net_benefit_usd']):.9f} | ${float(row['quality_admissible_net_benefit_usd']):.9f} | {row['unsafe_selections']} |"
        for row in data["same_horizon"]
        if row["policy_name"] in {"existing_adaptive", "m51_fixed_sample", "full_m52"}
        and row["reuse_horizon"] in {10, 1000}
    )
    system = """# SemanticFlow Milestone 5.2.1 System Design

## Purpose

M5.2.1 adds a qualification feasibility gate ahead of the unchanged M5.2 controller. It separates mathematical possibility, statistical power, economic feasibility, and exploratory risk estimation.

## Components and lifecycle

`m521_error_budget.py` versions the preserved legacy and prospective nonredundant policies. `m521_feasibility.py` computes exact best-case attainability from current confusion counts. `m521_statistical_power.py` computes positive-support and certification probabilities. `m521_economic_feasibility.py` enforces optimistic-benefit and expected-value stops. `m521_controller.py` checks feasibility before acquisition and through the M5.2 continuation guard.

Evidence provenance is validated first. The controller then checks qualification, exact attainability, measurement coverage, power, remaining expenditure, and economic upside. Hard impossibility cannot be overridden by an EVSI heuristic. Discovery evidence remains isolated from representative certification.

## Policy types

Formal certification retains frequentist assumptions. Exploratory Jeffreys beta-binomial risk estimates are explicitly `ESTIMATED_RISK_ONLY`; they never override strict deployment constraints. `existing_adaptive`, M5.1, and legacy M5.2 remain unchanged baselines.

## Failure handling

Distinct reasons cover record, call, token, positive-support, structural, and economic infeasibility. All certificates and policy configurations are fingerprinted. Offline execution never imports or initializes a provider.
"""
    statistical = f"""# SemanticFlow Milestone 5.2.1 Statistical Feasibility

## Exact construction

For all successes, the one-sided exact lower bound is `alpha^(1/n)`, giving `ceil(log(alpha)/log(q))`. For nonzero errors, M5.2.1 evaluates the exact binomial tail at the threshold. Qualification is attainable exactly when that tail is at most the allocated alpha. Log-space binomial calculations avoid small-alpha underflow.

| Policy | Recall alpha | Agreement alpha | Positive minimum | Record minimum |
|---|---:|---:|---:|---:|
{table}

## Redundancy

The implementation audit verified that recall and FNR share the positive denominator and are exact complements; agreement and changed-decision rate likewise share the complete-query denominator. Legacy M5.2 still allocates four bounds. The prospective nonredundant policy controls recall and agreement once and reports their complements, while retaining candidate and checkpoint multiplicity.

## Power and prevalence

Positive support is calculated exactly under a stated binomial prevalence. Wilson intervals show prevalence uncertainty, including nonzero upper bounds after zero observed positives. Certification power is exact conditional on a fixed denominator. Joint qualification uses a labeled dependence approximation rather than claiming independence.

### Legacy recall power at the best-case 153-positive denominator

| True recall | Certification probability | Maximum errors |
|---:|---:|---:|
{legacy_recall_power}

## Validity limitations

Fixed checkpoint spending covers every predeclared inspection. Representative records must be sampled independently of outcomes. Active discovery is diagnostic only. Stage D is retrospective development evidence, and separate Jev inference is not human truth.
"""
    banking = data["controlled"]["banking77"]
    beauty = data["controlled"]["beauty"]
    stop_requests = sum(row["requests_avoided_mathematical_impossibility"] for row in data["stop"])
    stop_cost = sum(row["cost_avoided_mathematical_impossibility_usd"] for row in data["stop"])
    results = f"""# SemanticFlow Milestone 5.2.1 Offline Results

## Central result

The legacy M5.2 rule needs 153 positive references and 389 representative records even with zero errors. The proposed 60-record pilot is mathematically incapable of certification. The nonredundant prospective rule lowers the limits but remains infeasible at 60 records.

Feasibility-aware stopping avoids {stop_requests} historical M5.2 profiling requests and ${stop_cost:.9f} before acquisition. This is cost containment relative to M5.2 profiling, not a deployment improvement over immediate zero-cost fallback.

## Banking77

At the retrospective 0.75% positive rate, 153 positives require about {banking['expected_records_for_support']:,} representative records in expectation and about ${banking['expected_profile_cost_usd']:.6f}. The 60-record proposal cannot qualify it. Its known quality failure remains unchanged.

## Beauty

At the retrospective 5.5% rate, 153 positives require about {beauty['expected_records_for_support']:,} records in expectation and about ${beauty['expected_profile_cost_usd']:.6f}. That may be mathematically possible, but is not supported as an economical prospective selection-time decision.

## Active sampling

Historical active and random policies used different evidence mixes, so raw request reductions cannot identify an active-sampling effect. Under identical feasibility stopping both acquire zero records. Active discovery therefore shows no incremental information-efficiency benefit once impossibility is known; this is a negative finding.

| Controlled policy | Requests | Profiling cost | Representative | Discovery |
|---|---:|---:|---:|---:|
{active_table}

## Economics

All policies are recomputed at R=1,5,10,25,50,100,250,1000. Economic-only benefits from the unsafe Banking77 batch are separated from quality-admissible benefit. Immediate `existing_adaptive` fallback remains net zero; any policy that profiles and selects nothing is economically worse.

| Policy | R | Profiling cost | Economic-only net | Quality-admissible net | Unsafe |
|---|---:|---:|---:|---:|---:|
{horizons_table}

## Conclusion

M5.2.1 solves stop-before-spend feasibility detection. It does not solve rare-positive formal certification. The central obstacle is representative positive support at a cost that execution savings can amortize.
"""
    pilot = f"""# SemanticFlow Milestone 5.2.1 Future Pilot Protocol

## Pilot A — instrumentation

Objective: validate acquisition behavior, paid screening, provenance, actual costs, disagreement diagnostics, and restart safety. It cannot establish formal quality qualification. Expected {data['pilot_a']['expected_physical_requests']} requests, ceiling {data['pilot_a']['physical_request_ceiling']}, expected proxy ${data['pilot_a']['expected_cost_proxy_usd']:.6f}.

## Pilot B — formal qualification

Not recommended for the current Banking77 and Beauty workloads. A design with 95% probability of merely obtaining 153 positives requires an estimated {data['pilot_b']['combined_design_physical_requests']:,} physical requests, {data['pilot_b']['combined_design_named_question_evaluations']:,} named evaluations, ${data['pilot_b']['combined_design_cost_proxy_usd']:.6f} proxy, and ${data['pilot_b']['conservative_cost_proxy_usd']:.6f} conservative proxy. Expected representative counts and costs are too large relative to execution savings, and a plan at exactly the 95% recall threshold has very low certification power. No paid schedule is authorized.

## Pilot C — exploratory risk validation

Objective: independently evaluate a frozen risk predictor on fresh calibration/test partitions without guarantee claims. Expected {data['pilot_c']['expected_physical_requests']} requests, ceiling {data['pilot_c']['physical_request_ceiling']}, expected proxy ${data['pilot_c']['expected_cost_proxy_usd']:.6f}.

## Independent evaluation freeze

Freeze policy hashes, alpha allocation, checkpoints, candidates, sampling, stopping, estimator, schedule, and dataset partitions before fresh outcomes. Human annotation is a separate study. Every pilot requires a new exact authorization; historical authorization cannot be reused.
"""
    for name, text in (
        ("milestone_5_2_1_system_design.md", system),
        ("milestone_5_2_1_statistical_feasibility.md", statistical),
        ("milestone_5_2_1_offline_results.md", results),
        ("milestone_5_2_1_future_pilot_protocol.md", pilot),
    ):
        (docs / name).write_text(text, encoding="utf-8")
