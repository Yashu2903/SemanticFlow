"""Post-authorization analysis for the Milestone 4.6 live validation."""

from __future__ import annotations

import itertools
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from statistics import mean, median
from typing import Any

from semanticflow.optimizer.bootstrap import collect_paired_bootstrap, quantile
from semanticflow.optimizer.generalization import (
    WorkloadSuite,
    economics_table,
    records_for_partition,
    subset_matrix,
)
from semanticflow.optimizer.matrix import ProfileMatrix
from semanticflow.optimizer.models import load_optimizer_settings
from semanticflow.optimizer.planner import predicate_hash


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _key(order: list[str] | tuple[str, ...]) -> str:
    return " -> ".join(order)


def _profile_matrix(
    suite: WorkloadSuite, dataset_id: str, profile_root: Path
) -> ProfileMatrix:
    dataset = suite.datasets[dataset_id]
    report = _json(profile_root / dataset_id / "report.json")
    if not report.get("complete"):
        raise ValueError(f"incomplete profile for {dataset_id}")
    profiling = records_for_partition(dataset.path, "profiling")
    evaluation = records_for_partition(dataset.path, "evaluation")
    predicate_ids = list(dataset.predicates)
    hashes = {
        predicate_id: predicate_hash(
            predicate_id, dataset.predicates[predicate_id].instructions
        )
        for predicate_id in predicate_ids
    }
    resolved = report["resolved_model_versions"]
    if len(resolved) != 1:
        raise ValueError(f"profile resolved multiple models for {dataset_id}")
    return ProfileMatrix.from_traces(
        _jsonl(profile_root / dataset_id / "decisions.jsonl"),
        predicate_ids,
        review_ids=[record.review_id for record in profiling],
        forbidden_review_ids=[record.review_id for record in evaluation],
        model_identifier=resolved[0],
        question_hashes=hashes,
    )


def _concordance(predicted: Mapping[str, float], observed: Mapping[str, float]) -> float | None:
    comparable = concordant = 0
    keys = sorted(set(predicted) & set(observed))
    for left, right in itertools.combinations(keys, 2):
        predicted_delta = predicted[left] - predicted[right]
        observed_delta = observed[left] - observed[right]
        if math.isclose(predicted_delta, 0.0) or math.isclose(observed_delta, 0.0):
            continue
        comparable += 1
        concordant += (predicted_delta > 0) == (observed_delta > 0)
    return concordant / comparable if comparable else None


def _match_consistency(run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons = []
    for left, right in itertools.combinations(run_rows, 2):
        left_ids, right_ids = left["matching_ids"], right["matching_ids"]
        union = left_ids | right_ids
        comparisons.append(
            {
                "left_run": left["run_id"],
                "right_run": right["run_id"],
                "exact": left_ids == right_ids,
                "jaccard": len(left_ids & right_ids) / len(union) if union else 1.0,
                "symmetric_difference_count": len(left_ids ^ right_ids),
            }
        )
    return {
        "all_exact": all(item["exact"] for item in comparisons),
        "pairwise": comparisons,
        "accuracy_or_recall_established": False,
    }


def _decision_consistency(run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons = []
    for left, right in itertools.combinations(run_rows, 2):
        left_cells = {
            (row["review_id"], row["predicate_id"]): row for row in left["decision_rows"]
        }
        right_cells = {
            (row["review_id"], row["predicate_id"]): row for row in right["decision_rows"]
        }
        overlap = sorted(set(left_cells) & set(right_cells))
        different = [
            key
            for key in overlap
            if bool(left_cells[key]["passed"]) != bool(right_cells[key]["passed"])
        ]
        comparisons.append(
            {
                "left_run": left["run_id"],
                "right_run": right["run_id"],
                "overlapping_record_predicate_evaluations": len(overlap),
                "different_threshold_decisions": len(different),
                "agreement": (
                    1 - len(different) / len(overlap) if overlap else None
                ),
                "threshold_crossing_examples": [
                    {
                        "review_id": review_id,
                        "predicate_id": predicate_id,
                        "left_probability": left_cells[(review_id, predicate_id)][
                            "probability"
                        ],
                        "right_probability": right_cells[(review_id, predicate_id)][
                            "probability"
                        ],
                    }
                    for review_id, predicate_id in different[:20]
                ],
            }
        )
    return {"pairwise": comparisons, "reference_labels_used": False}


def _aggregate(rows: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
    savings = [row["strategies"][strategy]["execution_saving_usd"] for row in rows]
    return {
        "strategy": strategy,
        "workloads": len(savings),
        "improved": sum(value > 0 for value in savings),
        "worse": sum(value < 0 for value in savings),
        "tied": sum(value == 0 for value in savings),
        "mean_execution_saving_usd": mean(savings),
        "median_execution_saving_usd": median(savings),
        "cost_weighted_execution_saving_usd": sum(savings),
        "maximum_downside_usd": min(savings),
        "leave_one_workload_out_total_saving_usd": {
            row["workload_id"]: sum(savings) - row["strategies"][strategy]["execution_saving_usd"]
            for row in rows
        },
    }


def analyze_live_validation(
    *,
    suite: WorkloadSuite,
    protocol: Mapping[str, Any],
    plans: Mapping[str, Any],
    profile_root: str | Path,
    session_root: str | Path,
    profiling_cost_accounting: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare measured orders; never substitute estimates for missing live results."""
    session = Path(session_root)
    profile = Path(profile_root)
    reports = sorted(session.glob("*/*/report.json"))
    incomplete = [
        path.as_posix()
        for path in reports
        if _json(path)["completion_status"] != "complete"
    ]
    expected_runs = sum(
        len(item["schedule"]["runs"])
        for item in _json(session / "authorized_budget.json")["workloads"]
    )
    if incomplete or len(reports) != expected_runs:
        return {
            "schema_version": "m46-live-analysis-v1",
            "status": "incomplete",
            "expected_runs": expected_runs,
            "observed_reports": len(reports),
            "incomplete_reports": incomplete,
            "performance_claims_permitted": False,
        }

    matrices = {
        dataset_id: _profile_matrix(suite, dataset_id, profile)
        for dataset_id in suite.datasets
    }
    optimizer = load_optimizer_settings(protocol["strategy_sources"]["milestone_4"])
    input_price = float(protocol["provider"]["pricing"]["input_per_million_usd"])
    output_price = float(protocol["provider"]["pricing"]["output_per_million_usd"])
    horizons = protocol["analysis"]["reuse_horizons"]
    interval_levels = protocol["analysis"]["interval_levels"]
    iterations = int(protocol["analysis"]["bootstrap_iterations"])
    by_workload: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in reports:
        report = _json(path)
        matches_path = path.with_name("matches.jsonl")
        decisions_path = path.with_name("decisions.jsonl")
        matches = {row["review_id"] for row in _jsonl(matches_path)}
        by_workload[report["workload_id"]].append(
            {
                "run_id": report["run_id"],
                "order": report["predicate_order"],
                "order_key": _key(report["predicate_order"]),
                "estimated_cost_usd": report["estimated_cost_usd"],
                "completed_decisions": report["completed_decisions"],
                "http_attempts": report["http_attempts"],
                "retries": report["retry_attempts"],
                "failures": report["failed_http_attempts"],
                "input_tokens": report["input_tokens"],
                "output_tokens": report["output_tokens"],
                "summed_inference_latency_seconds": report[
                    "summed_inference_latency_seconds"
                ],
                "wall_time_seconds": report["wall_time_seconds"],
                "stage_input_records": [
                    stage["input_record_count"] for stage in report["predicates"]
                ],
                "matching_ids": matches,
                "decision_rows": _jsonl(decisions_path),
                "comparison_role": report.get("m46_schedule_role"),
                "measured": True,
            }
        )

    workload_rows = []
    coverage = {str(level): {"hits": 0, "checks": 0} for level in interval_levels}
    for workload in suite.workloads:
        runs = by_workload[workload.id]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for run in runs:
            grouped[run["order_key"]].append(run)
        measured = {
            key: mean(run["estimated_cost_usd"] for run in group)
            for key, group in grouped.items()
        }
        original_key = _key(workload.original_order)
        fixed_cost = measured[original_key]
        cheapest = min(measured.values())
        plan = plans["workloads"][workload.id]
        predictions = {
            estimator: {
                _key(item["order"]): item["estimated_cost_usd"]
                for item in candidates
            }
            for estimator, candidates in plan["candidate_predictions"].items()
        }
        prediction_metrics = {}
        representative = {key: group[0] for key, group in grouped.items()}
        for estimator, predicted in predictions.items():
            stage_errors = []
            cost_errors = []
            evaluation_errors = []
            for candidate in plan["candidate_predictions"][estimator]:
                key = _key(candidate["order"])
                if key not in representative:
                    continue
                actual = representative[key]
                stage_errors.extend(
                    predicted_value - observed_value
                    for predicted_value, observed_value in zip(
                        candidate["estimated_stage_input_records"],
                        actual["stage_input_records"],
                        strict=True,
                    )
                )
                cost_errors.append(candidate["estimated_cost_usd"] - measured[key])
                evaluation_errors.append(
                    candidate["estimated_evaluations"] - actual["completed_decisions"]
                )
            prediction_metrics[estimator] = {
                "stage_cardinality_mae": mean(abs(value) for value in stage_errors),
                "stage_cardinality_mape": mean(
                    abs(value) / observed
                    for value, observed in (
                        (predicted_value - observed_value, observed_value)
                        for candidate in plan["candidate_predictions"][estimator]
                        if _key(candidate["order"]) in representative
                        for predicted_value, observed_value in zip(
                            candidate["estimated_stage_input_records"],
                            representative[_key(candidate["order"])][
                                "stage_input_records"
                            ],
                            strict=True,
                        )
                    )
                    if observed
                ),
                "stage_cardinality_signed_bias": mean(stage_errors),
                "total_evaluation_mae": mean(abs(value) for value in evaluation_errors),
                "cost_prediction_mae_usd": mean(abs(value) for value in cost_errors),
                "cost_prediction_signed_bias_usd": mean(cost_errors),
                "ranking_concordance_executed_orders": _concordance(predicted, measured),
            }

        matrix = subset_matrix(matrices[workload.dataset], workload.predicate_ids)
        dataset = suite.datasets[workload.dataset]
        question_hashes = {
            predicate_id: predicate_hash(
                predicate_id, dataset.predicates[predicate_id].instructions
            )
            for predicate_id in workload.predicate_ids
        }
        bootstrap = collect_paired_bootstrap(
            matrix,
            estimator_name="correlation_aware",
            original_order=workload.original_order,
            objective="estimated_cost_usd",
            evaluation_record_count=800,
            question_hashes=question_hashes,
            settings=optimizer.bootstrap,
            correlation=optimizer.correlation,
            input_price_per_million=input_price,
            output_price_per_million=output_price,
            iterations=iterations,
            seed=int(protocol["analysis"]["bootstrap_seed"]),
        )
        uncertainty = {}
        original_tuple = tuple(workload.original_order)
        for key in measured:
            order = tuple(key.split(" -> "))
            saving_draws = [
                left - right
                for left, right in zip(
                    bootstrap["draws"][original_tuple],
                    bootstrap["draws"][order],
                    strict=True,
                )
            ]
            observed_saving = fixed_cost - measured[key]
            intervals = {}
            for level in interval_levels:
                tail = (1 - level) / 2
                lower, upper = quantile(saving_draws, tail), quantile(
                    saving_draws, 1 - tail
                )
                hit = lower <= observed_saving <= upper
                intervals[str(level)] = {"lower": lower, "upper": upper, "covered": hit}
                coverage[str(level)]["checks"] += 1
                coverage[str(level)]["hits"] += int(hit)
            uncertainty[key] = {
                "observed_saving_usd": observed_saving,
                "intervals": intervals,
            }

        strategies = {}
        for strategy, selection in plan["strategies"].items():
            key = _key(selection["selected_order"])
            selected_cost = measured[key]
            standalone_profile = selection["profiling_cost_usd"]
            shared_profile = standalone_profile
            if profiling_cost_accounting is not None:
                accounting = profiling_cost_accounting["standalone_workload_economics"][
                    workload.id
                ][strategy]
                standalone_profile = accounting["standalone_cost_usd"]
                shared_profile = accounting["shared_incremental_cost_usd"]
            strategies[strategy] = {
                "selected_order": selection["selected_order"],
                "measured_execution_cost_usd": selected_cost,
                "execution_saving_usd": fixed_cost - selected_cost,
                "selection_regret_usd": selected_cost - cheapest,
                "selected_cheapest_executed_order": math.isclose(selected_cost, cheapest),
                "economics": economics_table(
                    fixed_cost=fixed_cost,
                    selected_cost=selected_cost,
                    profiling_cost=standalone_profile,
                    optimization_cost=0.0,
                    horizons=horizons,
                ),
                "economic_scenarios": {
                    "standalone_workload": economics_table(
                        fixed_cost=fixed_cost,
                        selected_cost=selected_cost,
                        profiling_cost=standalone_profile,
                        optimization_cost=0.0,
                        horizons=horizons,
                    ),
                    "shared_statistics_incremental": economics_table(
                        fixed_cost=fixed_cost,
                        selected_cost=selected_cost,
                        profiling_cost=shared_profile,
                        optimization_cost=0.0,
                        horizons=horizons,
                    ),
                    "warm_start": economics_table(
                        fixed_cost=fixed_cost,
                        selected_cost=selected_cost,
                        profiling_cost=0.0,
                        optimization_cost=0.0,
                        horizons=horizons,
                    ),
                },
            }
        risk = strategies["risk_aware"]
        risk_reordered = risk["selected_order"] != workload.original_order
        workload_rows.append(
            {
                "workload_id": workload.id,
                "dataset": workload.dataset,
                "fixed_measured_cost_usd": fixed_cost,
                "cheapest_executed_cost_usd": cheapest,
                "executed_order_costs_usd": measured,
                "strategies": strategies,
                "prediction_metrics": prediction_metrics,
                "uncertainty": uncertainty,
                "risk_policy": {
                    "harmful_reordering_accepted": risk_reordered
                    and risk["execution_saving_usd"] < 0,
                    "beneficial_reordering_rejected": not risk_reordered and cheapest < fixed_cost,
                    "observed_downside_avoided_usd": (
                        max(0.0, max(measured.values()) - fixed_cost)
                        if not risk_reordered
                        else 0.0
                    ),
                    "conservative_fallback_regret_usd": risk["selection_regret_usd"],
                },
                "output_consistency": _match_consistency(runs),
                "overlapping_decision_consistency": _decision_consistency(runs),
                "comparison_strength": {
                    "repeated_controlled": sum(
                        run["comparison_role"]
                        in {"abba_fixed_a", "abba_comparator_b"}
                        for run in runs
                    ),
                    "single_run_exploratory": sum(
                        run["comparison_role"] == "selected_order_coverage"
                        for run in runs
                    ),
                },
                "runs": [
                    {
                        key: value
                        for key, value in run.items()
                        if key not in {"matching_ids", "decision_rows"}
                    }
                    for run in runs
                ],
            }
        )

    strategies = list(protocol["strategies"])
    return {
        "schema_version": "m46-live-analysis-v1",
        "status": "complete",
        "performance_claims_permitted": True,
        "cost_fields": {
            "measured_execution": "provider-reported tokens priced with frozen rates",
            "profiling": "provider-reported profiling tokens",
            "optimization": "local compute recorded separately as zero monetary API cost",
            "simulated_repetitions": "economics reuse horizons only",
        },
        "profiling_cost_accounting": {
            "modes": [
                "actual_experiment_expenditure",
                "standalone_workload",
                "shared_statistics_incremental",
                "warm_start",
            ],
            "actual_stage_b_expenditure_usd": (
                profiling_cost_accounting["actual_experiment_expenditure"]["cost_usd"]
                if profiling_cost_accounting is not None
                else None
            ),
            "actual_and_counterfactual_costs_combined": False,
        },
        "workloads": workload_rows,
        "aggregate": {
            strategy: _aggregate(workload_rows, strategy) for strategy in strategies
        },
        "by_dataset": {
            dataset_id: {
                strategy: _aggregate(
                    [row for row in workload_rows if row["dataset"] == dataset_id], strategy
                )
                for strategy in strategies
            }
            for dataset_id in suite.datasets
        },
        "interval_coverage": {
            level: {
                **counts,
                "coverage": counts["hits"] / counts["checks"] if counts["checks"] else None,
            }
            for level, counts in coverage.items()
        },
        "statistical_units": {
            "primary": "workload",
            "record_dependence": "records are paired within workload",
            "dataset_dependence": "aggregate sensitivity is grouped by dataset and workload",
            "individual_requests_treated_as_independent_experiments": False,
        },
        "semantic_quality_limitation": (
            "Match-set agreement is measured, but accuracy and recall are not established "
            "without human labels."
        ),
    }
