from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from semanticflow.optimizer.adaptive import simulate_adaptive_profiling
from semanticflow.optimizer.bootstrap import order_key, paired_bootstrap
from semanticflow.optimizer.calibration import (
    CALIBRATION_SCHEMA_VERSION,
    ESTIMATOR_NAMES,
    PREDICTIVE_METHOD,
    calibrate_orders,
    coverage_summary,
    cross_validated_calibration,
    two_proportion_test,
)
from semanticflow.optimizer.cardinality import (
    CorrelationAwareEstimator,
    IndependentEstimator,
    exact_matrix_cardinalities,
    predicate_statistics_from_matrix,
)
from semanticflow.optimizer.cost_model import estimate_plan
from semanticflow.optimizer.economics import (
    SAVINGS_BASES,
    break_even_executions,
    break_even_range,
    decide_optimization,
    profiling_cost_accounting,
)
from semanticflow.optimizer.evidence import DecisionEvidence, file_sha256, read_jsonl
from semanticflow.optimizer.matrix import ProfileMatrix, joint_statistics
from semanticflow.optimizer.models import (
    OptimizerSettings,
    PlanSelection,
    ProfilingResult,
    SplitManifest,
)
from semanticflow.optimizer.planner import CostBasedPlanner, predicate_hash
from semanticflow.optimizer.replay import (
    REPLAY_EVIDENCE,
    PatternPrior,
    ReplayPricing,
    ReplayTarget,
    ResolutionPolicy,
    compare_orders,
    replay_order,
)
from semanticflow.query.models import QuerySpec
from semanticflow.storage.results import ResultStore, git_commit, git_worktree_dirty

ANALYSIS_SCHEMA_VERSION = "m3-offline-analysis-v2"
FIXED_SOURCE = "m1_fixed_full_baseline"
OPTIMIZED_SOURCE = "m2_optimized_heldout"
PROFILING_SOURCE = "m2_profiling"
ADAPTIVE_HORIZONS = (0, 1, 10, 113, 1000)
PRIOR_STRENGTH_GRID = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0)


@dataclass(slots=True)
class AnalysisInputs:
    experiment_dir: Path
    historical_baseline_dir: Path
    query: QuerySpec
    optimizer: OptimizerSettings
    extra_sources: list[tuple[str, Path, str]] = field(default_factory=list)
    horizons: list[int] = field(default_factory=lambda: [0, 1, 10, 100, 113, 1000])
    cross_validation_folds: int = 5
    config_paths: dict[str, str] = field(default_factory=dict)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _errors(pairs: Sequence[tuple[float, float]]) -> dict[str, Any]:
    """MAE, MAPE (observed > 0 only), and signed mean error of predicted-observed pairs."""
    if not pairs:
        return {"count": 0, "mae": None, "mape_percent": None, "mean_signed_error": None}
    absolute = [abs(predicted - observed) for predicted, observed in pairs]
    relative = [
        abs(predicted - observed) / observed * 100 for predicted, observed in pairs if observed > 0
    ]
    return {
        "count": len(pairs),
        "mae": sum(absolute) / len(absolute),
        "mape_percent": sum(relative) / len(relative) if relative else None,
        "mape_defined_pairs": len(relative),
        "mean_signed_error": sum(p - o for p, o in pairs) / len(pairs),
    }


def _stage_pairs(
    predicted: Sequence[float], observed: Sequence[float]
) -> list[tuple[float, float]]:
    # Stage 1 always receives every record, so it carries no estimation error.
    return list(zip(predicted[1:], observed[1:], strict=True))


def _estimators(
    matrix: ProfileMatrix,
    optimizer: OptimizerSettings,
    *,
    question_hashes: dict[str, str],
    input_rate: float | None,
    output_rate: float | None,
) -> dict[str, Any]:
    statistics = predicate_statistics_from_matrix(
        matrix,
        question_hashes=question_hashes,
        input_price_per_million=input_rate,
        output_price_per_million=output_rate,
    )
    return {
        "statistics": statistics,
        "independent": IndependentEstimator(statistics),
        "correlation_aware": CorrelationAwareEstimator(
            matrix,
            optimizer.correlation,
            input_price_per_million=input_rate,
            output_price_per_million=output_rate,
        ),
    }


def _predict(
    estimators: dict[str, Any], order: Sequence[str], records: int, original: Sequence[str]
) -> dict[str, Any]:
    output = {}
    for name in ("independent", "correlation_aware"):
        plan = estimate_plan(
            order,
            estimators["statistics"],
            input_record_count=records,
            objective="evaluations",
            original_order=original,
            estimator=estimators[name],
        )
        output[name] = plan
    return output


def _sorted_by_rank(review_ids: Sequence[str], seed: int) -> list[str]:
    return sorted(
        review_ids,
        key=lambda item: (hashlib.sha256(f"{seed}:{item}".encode()).hexdigest(), item),
    )


def cross_validate_cardinality(
    matrix: ProfileMatrix,
    optimizer: OptimizerSettings,
    *,
    question_hashes: dict[str, str],
    folds: int,
    seed: int,
) -> dict[str, Any]:
    """K-fold out-of-sample check inside the profiling split; held-out data is not used."""
    if folds < 2:
        raise ValueError("cross-validation requires at least two folds")
    ranked = _sorted_by_rank([row.review_id for row in matrix.rows], seed)
    assignments = {review_id: index % folds for index, review_id in enumerate(ranked)}
    original = list(matrix.predicate_ids)
    pairs: dict[str, list[tuple[float, float]]] = {"independent": [], "correlation_aware": []}
    totals: dict[str, list[tuple[float, float]]] = {"independent": [], "correlation_aware": []}
    fold_rows = []
    for fold in range(folds):
        train_ids = [rid for rid in ranked if assignments[rid] != fold]
        test_ids = [rid for rid in ranked if assignments[rid] == fold]
        train = matrix.select_reviews(train_ids)
        test = matrix.select_reviews(test_ids)
        estimators = _estimators(
            train, optimizer, question_hashes=question_hashes, input_rate=None, output_rate=None
        )
        for order in itertools.permutations(original):
            observed = exact_matrix_cardinalities(test, order)
            predictions = _predict(estimators, order, test.size, original)
            for name, plan in predictions.items():
                stages = plan.estimated_stage_input_records or []
                pairs[name].extend(_stage_pairs(stages, observed))
                totals[name].append((plan.estimated_evaluations, float(sum(observed))))
        fold_rows.append({"fold": fold, "train_reviews": train.size, "test_reviews": test.size})
    return {
        "method": "k_fold_cross_validation_within_profiling_split",
        "folds": folds,
        "fold_assignment": f"sha256({seed}:review_id) rank modulo folds",
        "fold_sizes": fold_rows,
        "stage_cardinality_errors": {name: _errors(values) for name, values in pairs.items()},
        "total_evaluation_errors": {name: _errors(values) for name, values in totals.items()},
    }


def _observed_heldout_usage(report: dict[str, Any]) -> dict[str, Any]:
    stages = report["predicates"]
    return {
        "stage_input_records": [float(item["input_record_count"]) for item in stages],
        "logical_evaluations": float(report["completed_decisions"]),
        "input_tokens": float(report["input_tokens"]),
        "output_tokens": float(report["output_tokens"]),
        "estimated_cost_usd": report.get("estimated_cost_usd"),
        "stage_mean_input_tokens": [
            item["input_tokens"] / item["completed_decisions"] if item["completed_decisions"]
            else None
            for item in stages
        ],
    }


def _historical_fixed_report(
    evidence: DecisionEvidence,
    target: ReplayTarget,
    evaluation_ids: list[str],
    order: list[str],
    pricing: ReplayPricing,
) -> dict[str, Any]:
    replay = replay_order(
        evidence,
        target,
        evaluation_ids,
        order,
        ResolutionPolicy(name="fixed_trace_only", source_priority=[FIXED_SOURCE]),
        pricing=pricing,
    )
    if replay["observability"] != "fully_observable":
        raise ValueError("historical fixed-order trace does not cover its own execution path")
    stage_tokens = []
    for stage in order:
        rows = [
            item
            for item in evidence.all_observations()
            if item.source == FIXED_SOURCE
            and item.predicate_id == stage
            and item.review_id in set(evaluation_ids)
        ]
        stage_tokens.append(
            sum(item.input_tokens or 0 for item in rows) / len(rows) if rows else None
        )
    return {
        "stage_input_records": [float(s["input_records"]["lower"]) for s in replay["stages"]],
        "logical_evaluations": float(replay["logical_evaluations"]["lower"]),
        "input_tokens": replay["usage"]["input_tokens"]["lower"],
        "output_tokens": replay["usage"]["output_tokens"]["lower"],
        "estimated_cost_usd": (replay["usage"]["estimated_cost_usd"] or {}).get("lower"),
        "stage_mean_input_tokens": stage_tokens,
    }


def _prediction_record(plan: Any, observed: dict[str, Any]) -> dict[str, Any]:
    stages = plan.estimated_stage_input_records or []
    evaluation_error = plan.estimated_evaluations - observed["logical_evaluations"]
    record = {
        "predicted_stage_input_records": stages,
        "observed_stage_input_records": observed["stage_input_records"],
        "stage_errors": _errors(_stage_pairs(stages, observed["stage_input_records"])),
        "predicted_evaluations": plan.estimated_evaluations,
        "observed_evaluations": observed["logical_evaluations"],
        "evaluation_absolute_error": abs(evaluation_error),
        "evaluation_signed_error": evaluation_error,
        "evaluation_relative_error_percent": (
            abs(evaluation_error) / observed["logical_evaluations"]
            * 100
        ),
        "predicted_stage_mean_input_tokens": [
            item["average_input_tokens"] for item in plan.stage_estimates or []
        ],
        "observed_stage_mean_input_tokens": observed["stage_mean_input_tokens"],
    }
    if plan.estimated_input_tokens is not None:
        record.update(
            {
                "predicted_input_tokens": plan.estimated_input_tokens,
                "observed_input_tokens": observed["input_tokens"],
                "input_token_relative_error_percent": abs(
                    plan.estimated_input_tokens - observed["input_tokens"]
                )
                / observed["input_tokens"]
                * 100,
            }
        )
    if plan.estimated_cost_usd is not None and observed["estimated_cost_usd"] is not None:
        record.update(
            {
                "predicted_cost_usd": plan.estimated_cost_usd,
                "observed_cost_usd": observed["estimated_cost_usd"],
                "cost_signed_error_usd": plan.estimated_cost_usd - observed["estimated_cost_usd"],
                "cost_relative_error_percent": abs(
                    plan.estimated_cost_usd - observed["estimated_cost_usd"]
                )
                / observed["estimated_cost_usd"]
                * 100,
            }
        )
    return record


def _candidates_payload(plan: PlanSelection) -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in plan.candidates]


def _span_text(span: dict[str, Any] | None) -> str:
    if span is None:
        return "unavailable"
    return f"{span['fewest_executions']}..{span['most_executions']}"


def _summary_markdown(result: dict[str, Any]) -> str:
    phase1 = result["counterfactual"]
    phase2 = result["cardinality"]
    plans = result["planning"]
    boot = result["bootstrap"]["correlation_aware"]
    adaptive = result["adaptive"]
    lines = [
        "# Milestone 3 offline analysis",
        "",
        "Generated from stored traces only. No provider was contacted.",
        "",
        "## Counterfactual replay of the Milestone 2 execution difference",
        "",
        f"- Observed calls: fixed {phase1['observed_fixed_evaluations']}, optimized "
        f"{phase1['observed_optimized_evaluations']} (difference "
        f"{phase1['observed_difference']}).",
    ]
    lines.append(
        "- Ordering effects below are offline replay estimates, not causal measurements."
    )
    for item in phase1["decomposition"]:
        effect = item["replay_estimated_ordering_effect_evaluations"]
        residual = item["residual_not_explained_by_replay_evaluations"]
        lines.append(
            f"- Policy `{item['policy']}`: replay-estimated ordering effect "
            f"{effect['lower']}..{effect['upper']} calls; residual not explained by replay "
            f"{residual['lower']}..{residual['upper']} calls."
        )
    lines += ["", "## Cardinality prediction (stage inputs after stage 1)", ""]
    lines.append("| Evaluation | Estimator | MAE | MAPE % | Mean signed error |")
    lines.append("|---|---|---:|---:|---:|")
    for label, block in (
        ("Profiling in-sample", phase2["profiling_in_sample"]["stage_cardinality_errors"]),
        ("Profiling 5-fold CV", phase2["profiling_cross_validation"]["stage_cardinality_errors"]),
        ("Held-out observed orders", phase2["held_out"]["stage_cardinality_errors"]),
    ):
        for name in ("independent", "correlation_aware"):
            item = block[name]
            mape = item["mape_percent"]
            lines.append(
                f"| {label} | {name} | {item['mae']:.3f} | "
                f"{'n/a' if mape is None else f'{mape:.2f}'} | {item['mean_signed_error']:.3f} |"
            )
    lines += ["", "## Plans", ""]
    for name, plan in plans["selections"].items():
        lines.append(
            f"- `{name}`: selected `{' -> '.join(plan['selected_predicate_order'])}`; "
            f"minimum `{' -> '.join(plan['minimum_objective_predicate_order'])}`"
        )
    lines += [
        "",
        "## Bootstrap (sampling-uncertainty diagnostic)",
        "",
        f"- Current plan selection frequency: "
        f"{boot['selection_frequency'][order_key(boot['reference_order'])]:.3f}",
        f"- P(current plan cheaper than original): "
        f"{boot['probability_reference_cheaper_than_original']:.3f}",
        f"- Stable at threshold {boot['stability_threshold']}: {boot['stable_selection']}",
        "",
        "## Calibration of predictive cardinality intervals",
        "",
        "| Evaluation | Estimator | Checks | 50% coverage | 90% coverage | Mean PIT |",
        "|---|---|---:|---:|---:|---:|",
    ]
    calibration = result["calibration"]
    for label, summaries in (
        (
            "Held-out observed orders",
            {
                name: block["summary"]
                for name, block in calibration["held_out"]["estimators"].items()
            },
        ),
        ("Profiling 5-fold CV", calibration["profiling_cross_validation"]["summary"]),
    ):
        for name, summary in summaries.items():
            item = summary["all_quantities"]
            lines.append(
                f"| {label} | {name} | {item['checks']} | {item['coverage_50']:.3f} | "
                f"{item['coverage_90']:.3f} | {item['mean_pit']:.3f} |"
            )
    break_even = result["economics"]["break_even"]
    lines += [
        "",
        "## Break-even executions (sunk full profile)",
        "",
        f"- Observed, confounded: "
        f"{break_even['observed']['break_even_executions']['sunk_full_profile']}",
    ]
    for policy, item in break_even["replay_estimated"]["by_policy"].items():
        if item is not None:
            span = _span_text(item["break_even_executions"]["sunk_full_profile"])
            lines.append(f"- Replay estimate `{policy}`: {span}")
    predicted = break_even["predicted"]
    interval = _span_text(
        predicted["correlation_aware_bootstrap_interval"]["break_even_executions"][
            "sunk_full_profile"
        ]
    )
    lines += [
        f"- Predicted, correlation-aware: "
        f"{predicted['correlation_aware']['break_even_executions']['sunk_full_profile']}",
        f"- Predicted, independence: "
        f"{predicted['independent']['break_even_executions']['sunk_full_profile']}",
        f"- Predicted, bootstrap interval: {interval} (None means never)",
        "",
        "## Adaptive profiling (offline simulation)",
        "",
    ]
    for basis, runs in adaptive["runs"].items():
        for horizon, item in runs.items():
            if not item["profiling_started"]:
                lines.append(
                    f"- `{basis}`, horizon {horizon}: profiling skipped "
                    f"({item['start_decision']['reasons']})"
                )
                continue
            stop = item["stopping_checkpoint"]
            lines.append(
                f"- `{basis}`, horizon {horizon}: stop after {stop['profiled_reviews']} "
                f"reviews ({stop['profiling_decisions']} decisions); agrees with full "
                f"profile: {item['stopped_plan_agrees_with_full_profile']}; reasons "
                f"{stop['reasons']}"
            )
    lines.append("")
    return "\n".join(lines)


def run_offline_analysis(inputs: AnalysisInputs, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    experiment = inputs.experiment_dir
    query = inputs.query
    optimizer = inputs.optimizer
    profile = ProfilingResult.model_validate(_load_json(experiment / "profiling" / "report.json"))
    split = SplitManifest.model_validate(_load_json(experiment / "split_manifest.json"))
    stored_selected = _load_json(experiment / "planning" / "selected_plan.json")
    stored_selected["candidates"] = _load_json(experiment / "planning" / "candidate_plans.json")[
        "candidates"
    ]
    stored_plan = PlanSelection.model_validate(stored_selected)
    fixed_manifest = _load_json(inputs.historical_baseline_dir / "manifest.json")
    optimized_report = _load_json(experiment / "optimized" / "report.json")
    optimized_manifest = _load_json(experiment / "optimized" / "manifest.json")

    hashes = {item.id: predicate_hash(item.id, item.instructions) for item in query.predicates}
    if hashes != profile.predicate_hashes:
        raise ValueError("query predicate definitions do not match profiling hashes")
    if fixed_manifest.get("question_hashes") != hashes:
        raise ValueError("historical baseline used different predicate definitions")
    if fixed_manifest.get("dataset_fingerprint") != split.dataset_fingerprint:
        raise ValueError("historical baseline used a different dataset")
    if optimized_manifest.get("evaluation_fingerprint") != split.evaluation_fingerprint:
        raise ValueError("optimized run did not use the recorded evaluation split")
    if len(profile.resolved_model_versions) != 1:
        raise ValueError("analysis requires one resolved model version")
    model = profile.resolved_model_versions[0]
    original = [item.id for item in query.predicates]
    selected_m2 = stored_plan.selected_predicate_order
    input_rate = profile.pricing.get("input_per_million_usd")
    output_rate = profile.pricing.get("output_per_million_usd")
    input_rate = float(input_rate) if input_rate is not None else None
    output_rate = float(output_rate) if output_rate is not None else None
    pricing = ReplayPricing(input_per_million_usd=input_rate, output_per_million_usd=output_rate)

    evidence = DecisionEvidence()
    evidence.load_jsonl(
        inputs.historical_baseline_dir / "decisions.jsonl",
        FIXED_SOURCE,
        description="Milestone 1 fixed-order Jev run over all 1,000 reviews",
    )
    evidence.load_jsonl(
        experiment / "optimized" / "decisions.jsonl",
        OPTIMIZED_SOURCE,
        description="Milestone 2 fresh optimized-order Jev run over 800 held-out reviews",
    )
    evidence.load_jsonl(
        experiment / "profiling" / "decisions.jsonl",
        PROFILING_SOURCE,
        description="Milestone 2 unconditional Jev profiling over 200 profiling reviews",
    )
    for label, path, description in inputs.extra_sources:
        evidence.load_jsonl(path, label, description=description)
    target = ReplayTarget(
        model_identifier=model, threshold=query.threshold, question_hashes=hashes
    )
    matrix = ProfileMatrix.from_traces(
        read_jsonl(experiment / "profiling" / "decisions.jsonl"),
        original,
        review_ids=split.profiling_ids,
        forbidden_review_ids=split.evaluation_ids,
        model_identifier=model,
        question_hashes=hashes,
    )
    evaluation_ids = list(split.evaluation_ids)

    # Phase 1: counterfactual replay.
    reproduced_fixed = replay_order(
        evidence,
        target,
        evaluation_ids,
        original,
        ResolutionPolicy(name="fixed_trace_only", source_priority=[FIXED_SOURCE]),
        pricing=pricing,
    )
    reproduced_optimized = replay_order(
        evidence,
        target,
        evaluation_ids,
        selected_m2,
        ResolutionPolicy(name="optimized_trace_only", source_priority=[OPTIMIZED_SOURCE]),
        pricing=pricing,
    )
    optimized_matches = {
        row["review_id"] for row in read_jsonl(experiment / "optimized" / "matches.jsonl")
    }
    fixed_matches = {
        row["review_id"]
        for row in read_jsonl(inputs.historical_baseline_dir / "matches.jsonl")
        if row["review_id"] in set(evaluation_ids)
    }
    observed_fixed = int(reproduced_fixed["logical_evaluations"]["lower"])
    observed_optimized = int(optimized_report["completed_decisions"])
    observed_fixed_cost = (reproduced_fixed["usage"]["estimated_cost_usd"] or {}).get("lower")
    observed_optimized_cost = optimized_report.get("estimated_cost_usd")
    policies = [
        ResolutionPolicy(
            name="historical_first", source_priority=[FIXED_SOURCE, OPTIMIZED_SOURCE]
        ),
        ResolutionPolicy(name="fresh_first", source_priority=[OPTIMIZED_SOURCE, FIXED_SOURCE]),
        ResolutionPolicy(
            name="conflicts_unknown",
            source_priority=[FIXED_SOURCE, OPTIMIZED_SOURCE],
            conflict_mode="treat_as_unknown",
        ),
    ]
    decomposition = []
    for policy in policies:
        comparison = compare_orders(
            evidence, target, evaluation_ids, original, selected_m2, policy, pricing=pricing
        )
        effect = comparison["evaluation_difference_first_minus_second"]
        observed_difference = observed_fixed - observed_optimized
        cost_effect = comparison["estimated_cost_difference_usd"]
        entry = {
            "policy": policy.name,
            "policy_definition": policy.model_dump(mode="json"),
            "evidence_kind": REPLAY_EVIDENCE["evidence_kind"],
            "causal_measurement": False,
            "fixed_order_replay_evaluations": comparison["first"]["logical_evaluations"],
            "optimized_order_replay_evaluations": comparison["second"]["logical_evaluations"],
            "replay_estimated_ordering_effect_evaluations": effect,
            "residual_not_explained_by_replay_evaluations": {
                "lower": observed_difference - effect["upper"],
                "upper": observed_difference - effect["lower"],
            },
            "replay_estimated_ordering_effect_cost_usd": cost_effect,
            "residual_not_explained_by_replay_cost_usd": (
                {
                    "lower": observed_fixed_cost - observed_optimized_cost - cost_effect["upper"],
                    "upper": observed_fixed_cost - observed_optimized_cost - cost_effect["lower"],
                }
                if cost_effect is not None
                and observed_fixed_cost is not None
                and observed_optimized_cost is not None
                else None
            ),
            "fully_determined_by_stored_decisions": comparison[
                "difference_fully_determined_by_stored_decisions"
            ],
            "priority_resolved_conflicts_on_paths": {
                "fixed_order": comparison["first"]["priority_resolved_conflicts_on_paths"],
                "optimized_order": comparison["second"]["priority_resolved_conflicts_on_paths"],
            },
            "final_matches_fixed_order": comparison["first"]["final_matches"]["bounds"],
            "final_matches_optimized_order": comparison["second"]["final_matches"]["bounds"],
        }
        decomposition.append(entry)
    prior = PatternPrior(matrix, original)
    all_orders_heldout = {}
    for order in itertools.permutations(original):
        replay = replay_order(
            evidence,
            target,
            evaluation_ids,
            order,
            policies[0],
            pricing=pricing,
            prior=prior,
        )
        all_orders_heldout[order_key(order)] = {
            key: replay[key]
            for key in (
                "observability",
                "resolved_reviews",
                "unresolved_reviews",
                "logical_evaluations",
                "stages",
                "usage",
                "missing_decisions_on_possible_paths",
                "estimate",
            )
        } | {
            "final_matches": {
                key: replay["final_matches"][key]
                for key in ("definite_count", "possible_additional_count", "bounds")
            }
        }
    profiling_replay = {}
    for order in itertools.permutations(original):
        replay = replay_order(
            evidence,
            target,
            list(split.profiling_ids),
            order,
            ResolutionPolicy(name="profiling_only", source_priority=[PROFILING_SOURCE]),
            pricing=pricing,
        )
        profiling_replay[order_key(order)] = {
            key: replay[key] for key in ("observability", "logical_evaluations", "stages", "usage")
        } | {"final_matches": replay["final_matches"]["bounds"]}
    variability = evidence.repeated_decision_report(
        model_identifier=model,
        question_hashes=hashes,
        threshold=query.threshold,
        sources=[FIXED_SOURCE, PROFILING_SOURCE, OPTIMIZED_SOURCE]
        + [label for label, _, _ in inputs.extra_sources],
    )
    counterfactual = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "evaluation_record_count": len(evaluation_ids),
        "reproduction": {
            "fixed_from_trace": {
                "observability": reproduced_fixed["observability"],
                "logical_evaluations": reproduced_fixed["logical_evaluations"],
                "input_tokens": reproduced_fixed["usage"]["input_tokens"],
                "estimated_cost_usd": reproduced_fixed["usage"]["estimated_cost_usd"],
                "final_match_ids_equal_recorded": set(
                    reproduced_fixed["final_matches"]["definite_review_ids"]
                )
                == fixed_matches,
            },
            "optimized_from_trace": {
                "observability": reproduced_optimized["observability"],
                "logical_evaluations": reproduced_optimized["logical_evaluations"],
                "recorded_completed_decisions": optimized_report["completed_decisions"],
                "input_tokens": reproduced_optimized["usage"]["input_tokens"],
                "recorded_input_tokens": optimized_report["input_tokens"],
                "estimated_cost_usd": reproduced_optimized["usage"]["estimated_cost_usd"],
                "recorded_estimated_cost_usd": optimized_report["estimated_cost_usd"],
                "final_match_ids_equal_recorded": set(
                    reproduced_optimized["final_matches"]["definite_review_ids"]
                )
                == optimized_matches,
            },
        },
        "observed_fixed_evaluations": observed_fixed,
        "observed_optimized_evaluations": observed_optimized,
        "observed_difference": observed_fixed - observed_optimized,
        "observed_cost_difference_usd": (
            observed_fixed_cost - observed_optimized_cost
            if observed_fixed_cost is not None and observed_optimized_cost is not None
            else None
        ),
        "decomposition": decomposition,
        "evidence": REPLAY_EVIDENCE,
        "interpretation": (
            "Both orders are replayed over one resolved decision set, so their replayed "
            "difference estimates the ordering effect under the stated assumptions. It is "
            "not a causal measurement: no randomized or contemporaneous paired execution of "
            "both orders exists. The residual between the observed difference and the replay "
            "estimate is consistent with response variability between the historical and "
            "fresh calls, but also absorbs any violation of the replay assumptions. Bounds "
            "come from treating conflicting repeats as unknown."
        ),
        "all_orders_held_out_historical_first": all_orders_heldout,
        "all_orders_profiling_exact": profiling_replay,
        "variability": variability,
        "evidence_sources": {label: item.to_dict() for label, item in evidence.sources.items()},
    }

    # Phase 2: cardinality estimation.
    estimators = _estimators(
        matrix, optimizer, question_hashes=hashes, input_rate=input_rate, output_rate=output_rate
    )
    statistics_match = all(
        abs(
            (estimators["statistics"][item.predicate_id].average_estimated_cost_usd or 0)
            - (item.average_estimated_cost_usd or 0)
        )
        < 1e-15
        and estimators["statistics"][item.predicate_id].passing_decisions
        == item.passing_decisions
        for item in profile.statistics
    )
    in_sample_pairs: dict[str, list[tuple[float, float]]] = {
        "independent": [],
        "correlation_aware": [],
    }
    in_sample_totals: dict[str, list[tuple[float, float]]] = {
        "independent": [],
        "correlation_aware": [],
    }
    in_sample_rows = []
    for order in itertools.permutations(original):
        observed = exact_matrix_cardinalities(matrix, order)
        replayed = profiling_replay[order_key(order)]
        if [s["input_records"]["lower"] for s in replayed["stages"]] != observed:
            raise ValueError("matrix replay and evidence replay disagree")
        predictions = _predict(estimators, order, matrix.size, original)
        row = {"order": list(order), "observed_stage_input_records": observed}
        for name, plan in predictions.items():
            stages = plan.estimated_stage_input_records or []
            in_sample_pairs[name].extend(_stage_pairs(stages, observed))
            in_sample_totals[name].append((plan.estimated_evaluations, float(sum(observed))))
            row[f"{name}_stage_input_records"] = stages
        in_sample_rows.append(row)
    cross_validation = cross_validate_cardinality(
        matrix,
        optimizer,
        question_hashes=hashes,
        folds=inputs.cross_validation_folds,
        seed=optimizer.split.seed,
    )
    heldout_observed = {
        order_key(original): {
            "source": "historical Milestone 1 fixed-order trace restricted to held-out IDs",
            **_historical_fixed_report(evidence, target, evaluation_ids, original, pricing),
        },
        order_key(selected_m2): {
            "source": "fresh Milestone 2 optimized-order Jev execution",
            **_observed_heldout_usage(optimized_report),
        },
    }
    heldout_rows = {}
    heldout_pairs: dict[str, list[tuple[float, float]]] = {
        "independent": [],
        "correlation_aware": [],
    }
    heldout_totals: dict[str, list[tuple[float, float]]] = {
        "independent": [],
        "correlation_aware": [],
    }
    for key, observed in heldout_observed.items():
        order = key.split(" -> ")
        plans = {
            name: estimate_plan(
                order,
                estimators["statistics"],
                input_record_count=len(evaluation_ids),
                objective="estimated_cost_usd" if input_rate is not None else "evaluations",
                original_order=original,
                estimator=estimators[name],
            )
            for name in ("independent", "correlation_aware")
        }
        heldout_rows[key] = {"observed_source": observed["source"]} | {
            name: _prediction_record(plan, observed) for name, plan in plans.items()
        }
        for name, plan in plans.items():
            heldout_pairs[name].extend(
                _stage_pairs(plan.estimated_stage_input_records or [],
                             observed["stage_input_records"])
            )
            heldout_totals[name].append(
                (plan.estimated_evaluations, observed["logical_evaluations"])
            )
    bound_checks = {}
    for key, replay in all_orders_heldout.items():
        order = key.split(" -> ")
        predictions = _predict(estimators, order, len(evaluation_ids), original)
        bounds = replay["logical_evaluations"]
        bound_checks[key] = {
            "replay_observability": replay["observability"],
            "replay_bounds": bounds,
            "imputation_estimate": replay["estimate"]["logical_evaluations"],
            **{
                f"{name}_predicted": plan.estimated_evaluations
                for name, plan in predictions.items()
            },
            **{
                f"{name}_within_bounds": bounds["lower"]
                <= plan.estimated_evaluations
                <= bounds["upper"]
                for name, plan in predictions.items()
            },
        }
    cardinality = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "estimators_fit_on": "profiling split only",
        "held_out_outcomes_used_for_fitting": False,
        "matrix_statistics_reproduce_profile_statistics": statistics_match,
        "correlation_settings": optimizer.correlation.model_dump(mode="json"),
        "profiling_in_sample": {
            "note": (
                "In-sample comparison; agreement of the correlation-aware model here is "
                "largely by construction and is not evidence of out-of-sample accuracy."
            ),
            "orders": in_sample_rows,
            "stage_cardinality_errors": {k: _errors(v) for k, v in in_sample_pairs.items()},
            "total_evaluation_errors": {k: _errors(v) for k, v in in_sample_totals.items()},
        },
        "profiling_cross_validation": cross_validation,
        "held_out": {
            "observed_orders": heldout_rows,
            "stage_cardinality_errors": {k: _errors(v) for k, v in heldout_pairs.items()},
            "total_evaluation_errors": {k: _errors(v) for k, v in heldout_totals.items()},
            "replay_bound_checks_all_orders": bound_checks,
            "fairness_limitations": [
                "Only two held-out orders were executed; four are bounded by replay only.",
                "The fixed-order observation is a historical trace and the optimized-order "
                "observation is a fresh run, so response variability enters both.",
                "Orders that evaluate dissatisfaction before stage 3 lack dissatisfaction "
                "decisions for most held-out reviews.",
            ],
        },
    }

    # Calibration of predictive cardinality intervals (fit on profiling rows only).
    heldout_counts = {
        tuple(original): heldout_observed[order_key(original)]["stage_input_records"][1:]
        + [float(len(fixed_matches))],
        tuple(selected_m2): heldout_observed[order_key(selected_m2)]["stage_input_records"][1:]
        + [float(len(optimized_matches))],
    }
    heldout_calibration = {}
    for name in ESTIMATOR_NAMES:
        records = calibrate_orders(
            matrix,
            heldout_counts,
            estimator_name=name,
            question_hashes=hashes,
            correlation=optimizer.correlation,
            record_count=len(evaluation_ids),
            iterations=optimizer.bootstrap.iterations,
            seed=optimizer.bootstrap.seed,
        )
        heldout_calibration[name] = {"summary": coverage_summary(records), "records": records}
    marginal_shift = {}
    for order in heldout_counts:
        first = order[0]
        item = estimators["statistics"][first]
        marginal_shift[first] = {
            "profiling_passing": item.passing_decisions,
            "profiling_records": matrix.size,
            "held_out_passing": heldout_counts[order][0],
            "held_out_records": len(evaluation_ids),
            "held_out_source": heldout_observed[order_key(order)]["source"],
            **two_proportion_test(
                item.passing_decisions,
                matrix.size,
                int(heldout_counts[order][0]),
                len(evaluation_ids),
            ),
        }
    prior_sensitivity = {
        str(alpha): cross_validate_cardinality(
            matrix,
            optimizer.model_copy(
                update={
                    "correlation": optimizer.correlation.model_copy(
                        update={"prior_strength": alpha}
                    )
                }
            ),
            question_hashes=hashes,
            folds=inputs.cross_validation_folds,
            seed=optimizer.split.seed,
        )["stage_cardinality_errors"]["correlation_aware"]
        for alpha in PRIOR_STRENGTH_GRID
    }
    calibration = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "method": PREDICTIVE_METHOD,
        "estimators_fit_on": "profiling split only",
        "held_out_outcomes_used_for_fitting": False,
        "iterations": optimizer.bootstrap.iterations,
        "seed": optimizer.bootstrap.seed,
        "held_out": {
            "observed_sources": {
                order_key(order): heldout_observed[order_key(order)]["source"]
                for order in heldout_counts
            },
            "estimators": heldout_calibration,
        },
        "profiling_cross_validation": cross_validated_calibration(
            matrix,
            question_hashes=hashes,
            correlation=optimizer.correlation,
            folds=inputs.cross_validation_folds,
            fold_seed=optimizer.split.seed,
            iterations=optimizer.bootstrap.iterations,
            seed=optimizer.bootstrap.seed,
        ),
        "first_stage_marginal_shift": marginal_shift,
        "prior_strength_sensitivity_cv": {
            "configured_prior_strength": optimizer.correlation.prior_strength,
            "selection_uses_held_out": False,
            "configuration_changed": False,
            "stage_cardinality_errors_by_prior_strength": prior_sensitivity,
        },
    }

    # Planning with both estimators and the confidence-aware policy.
    reproduction = CostBasedPlanner(
        optimizer.model_copy(
            update={"planner": stored_plan_settings(optimizer, stored_plan, "independent")}
        )
    ).plan(query, profile, split, matrix=matrix)
    stored_values = {
        tuple(item.predicate_order): item.objective_value for item in stored_plan.candidates
    }
    reproduction_difference = max(
        abs(item.objective_value - stored_values[tuple(item.predicate_order)])
        for item in reproduction.candidates
    )
    selections = {}
    for name, estimator, policy in (
        ("independent", "independent", "minimum_expected_objective"),
        ("correlation_aware", "correlation_aware", "minimum_expected_objective"),
        ("correlation_aware_confidence", "correlation_aware", "confidence_aware"),
    ):
        settings = optimizer.model_copy(
            update={
                "planner": stored_plan_settings(optimizer, stored_plan, estimator).model_copy(
                    update={"selection_policy": policy}
                )
            }
        )
        plan = CostBasedPlanner(settings).plan(query, profile, split, matrix=matrix)
        selections[name] = plan.artifact_dict(include_candidates=True)
        ResultStore.write_json(output_dir / f"plan_{name}.json", selections[name])
    planning = {
        "independent_reproduces_milestone_2_candidates": reproduction_difference == 0.0,
        "maximum_reproduction_difference": reproduction_difference,
        "milestone_2_selected_order": selected_m2,
        "selections": selections,
    }

    # Phase 4: bootstrap for both estimators.
    bootstrap = {
        name: paired_bootstrap(
            matrix,
            estimator_name=name,
            original_order=original,
            objective=stored_plan.objective,
            evaluation_record_count=split.evaluation_size,
            question_hashes=hashes,
            settings=optimizer.bootstrap,
            correlation=optimizer.correlation,
            input_price_per_million=input_rate,
            output_price_per_million=output_rate,
            tie_relative_tolerance=optimizer.planner.tie_relative_tolerance,
            reference_order=selected_m2,
        )
        for name in ("independent", "correlation_aware")
    }

    # Phase 3: economics.
    corr_plan = selections["correlation_aware"]
    by_order = {
        order_key(item["predicate_order"]): item for item in corr_plan["candidates"]
    }
    original_cost = by_order[order_key(original)]["estimated_cost_usd"]
    optimized_cost = by_order[order_key(selected_m2)]["estimated_cost_usd"]
    profiling_cost = profile.estimated_cost_usd
    average_decision_cost = (
        profiling_cost / profile.completed_decisions
        if profiling_cost is not None and profile.completed_decisions
        else None
    )
    cached = sum(
        1
        for review_id in split.profiling_ids
        for predicate_id in original
        if evidence.observations(
            review_id, predicate_id, model, hashes[predicate_id], sources=[FIXED_SOURCE]
        )
    )
    accounting = profiling_cost_accounting(
        required_decisions=len(split.profiling_ids) * len(original),
        reusable_decisions=cached,
        average_cost_per_decision=average_decision_cost,
        incurred_cost=profiling_cost or 0.0,
    )
    savings_ci = bootstrap["correlation_aware"]["reference_savings_vs_original"]
    savings_probability = bootstrap["correlation_aware"][
        "probability_reference_cheaper_than_original"
    ]
    per_evaluation_costs = [item.average_estimated_cost_usd for item in profile.statistics]
    structural_bound = (
        len(evaluation_ids) * (sum(per_evaluation_costs) - min(per_evaluation_costs))
        if all(value is not None for value in per_evaluation_costs)
        else None
    )
    replay_savings = {
        item["policy"]: item["replay_estimated_ordering_effect_cost_usd"]
        for item in decomposition
    }
    observed_saving = (
        observed_fixed_cost - observed_optimized_cost
        if observed_fixed_cost is not None and observed_optimized_cost is not None
        else None
    )
    predicted_saving = (
        original_cost - optimized_cost
        if original_cost is not None and optimized_cost is not None
        else None
    )
    independent_saving = (
        stored_plan.original_estimated_objective - stored_plan.selected_estimated_objective
        if stored_plan.objective == "estimated_cost_usd"
        else None
    )
    overhead = optimizer.economics.optimization_overhead_usd
    cost_bases = {
        "sunk_full_profile": profiling_cost,
        "new_spend_with_cached_decisions": accounting["new_profiling_cost_usd"],
    }

    def break_even_point(saving: float | None) -> dict[str, int | None]:
        return {
            name: break_even_executions(cost + overhead, saving)
            if cost is not None and saving is not None
            else None
            for name, cost in cost_bases.items()
        }

    def break_even_span(lower: float, upper: float) -> dict[str, Any]:
        return {
            name: break_even_range(cost + overhead, lower, upper) if cost is not None else None
            for name, cost in cost_bases.items()
        }

    scenarios = []
    for horizon in inputs.horizons:
        base = optimizer.economics.model_copy(update={"expected_executions": horizon})
        gated = base.model_copy(
            update={"minimum_savings_probability": optimizer.bootstrap.confidence_level}
        )
        scenarios.append(
            {
                "expected_executions": horizon,
                "existing_statistics_sunk_profile": decide_optimization(
                    settings=base,
                    original_cost_per_execution=original_cost,
                    optimized_cost_per_execution=optimized_cost,
                    statistics_state="available",
                    new_profiling_cost=profiling_cost,
                    sunk_profiling_cost=profiling_cost or 0.0,
                    existing_plan_available=True,
                    savings_interval=(savings_ci["lower"], savings_ci["upper"]),
                    savings_probability=savings_probability,
                    savings_basis="predicted",
                ),
                "existing_statistics_confidence_gated": decide_optimization(
                    settings=gated,
                    original_cost_per_execution=original_cost,
                    optimized_cost_per_execution=optimized_cost,
                    statistics_state="available",
                    new_profiling_cost=profiling_cost,
                    sunk_profiling_cost=profiling_cost or 0.0,
                    existing_plan_available=True,
                    savings_interval=(savings_ci["lower"], savings_ci["upper"]),
                    savings_probability=savings_probability,
                    savings_basis="predicted",
                ),
                "new_comparable_workload_full_profile": decide_optimization(
                    settings=base,
                    original_cost_per_execution=original_cost,
                    statistics_state="none",
                    new_profiling_cost=profiling_cost,
                    maximum_savings_per_execution=predicted_saving,
                    savings_basis="transferred_prediction",
                ),
                "new_workload_structural_upper_bound": decide_optimization(
                    settings=base,
                    original_cost_per_execution=original_cost,
                    statistics_state="none",
                    new_profiling_cost=profiling_cost,
                    maximum_savings_per_execution=structural_bound,
                    savings_basis="structural_upper_bound",
                ),
            }
        )
    economics = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "objective": "estimated_cost_usd",
        "pricing": profile.pricing,
        "cost_source": "correlation_aware predicted per-execution cost on 800 records",
        "original_predicted_cost_usd": original_cost,
        "optimized_predicted_cost_usd": optimized_cost,
        "predicted_savings_per_execution_usd": predicted_saving,
        "independent_predicted_savings_per_execution_usd": independent_saving,
        "milestone_2_observed_savings_usd_confounded": observed_saving,
        "replay_estimated_ordering_savings_usd": replay_savings,
        "bootstrap_savings_interval_usd": savings_ci,
        "bootstrap_savings_probability": savings_probability,
        "profiling_accounting": accounting,
        "structural_savings_upper_bound_per_execution_usd": structural_bound,
        "structural_savings_upper_bound_definition": (
            "N times (sum of per-evaluation costs minus the cheapest): the original order "
            "evaluates at most every predicate on every record, and every order evaluates "
            "all N records at stage 1, so no order can save more under any selectivities"
        ),
        "savings_bases": SAVINGS_BASES,
        "break_even": {
            "definition": (
                "smallest whole number of executions R with R * saving >= profiling cost + "
                "optimization overhead; null means the saving is not positive, so the "
                "profile never pays back"
            ),
            "execution_unit": f"one execution over the {len(evaluation_ids)} held-out reviews",
            "optimization_overhead_usd": overhead,
            "profiling_cost_bases": {
                "sunk_full_profile": {
                    "usd": profiling_cost,
                    "kind": "recorded cost of the Milestone 2 profile",
                },
                "new_spend_with_cached_decisions": {
                    "usd": accounting["new_profiling_cost_usd"],
                    "kind": (
                        "hypothetical: the same profile reusing the "
                        f"{accounting['reusable_cached_decisions']} decisions already stored "
                        "in Milestone 1 traces"
                    ),
                },
            },
            "observed": {
                "savings_basis": "observed_confounded",
                "causal_measurement": False,
                "savings_per_execution_usd": observed_saving,
                "break_even_executions": break_even_point(observed_saving),
            },
            "replay_estimated": {
                "savings_basis": "replay_estimate",
                "causal_measurement": False,
                "by_policy": {
                    name: (
                        {
                            "savings_per_execution_usd": value,
                            "break_even_executions": break_even_span(
                                value["lower"], value["upper"]
                            ),
                        }
                        if value is not None
                        else None
                    )
                    for name, value in replay_savings.items()
                },
            },
            "predicted": {
                "savings_basis": "predicted",
                "causal_measurement": False,
                "correlation_aware": {
                    "savings_per_execution_usd": predicted_saving,
                    "break_even_executions": break_even_point(predicted_saving),
                },
                "independent": {
                    "savings_per_execution_usd": independent_saving,
                    "break_even_executions": break_even_point(independent_saving),
                },
                "correlation_aware_bootstrap_interval": {
                    "savings_basis": "bootstrap_interval",
                    "interval_level": optimizer.bootstrap.interval_level,
                    "savings_per_execution_usd": {
                        "lower": savings_ci["lower"],
                        "upper": savings_ci["upper"],
                    },
                    "break_even_executions": break_even_span(
                        savings_ci["lower"], savings_ci["upper"]
                    ),
                },
            },
        },
        "scenarios": scenarios,
    }

    # Phase 5: adaptive profiling simulation, gated by profiling economics before it starts.
    gate_bounds = {
        "structural_upper_bound": structural_bound,
        "transferred_prediction": predicted_saving,
    }
    adaptive = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "evidence": "offline simulation over the recorded profiling matrix; no provider calls",
        "start_gate_bases": {
            basis: {
                "definition": SAVINGS_BASES[basis],
                "savings_upper_bound_per_execution_usd": bound,
            }
            for basis, bound in gate_bounds.items()
        },
        "first_checkpoint_cost_basis": (
            "average recorded profiling cost per decision times first-checkpoint decisions"
        ),
        "runs": {
            basis: {
                str(horizon): simulate_adaptive_profiling(
                    matrix,
                    estimator_name="correlation_aware",
                    original_order=original,
                    objective=stored_plan.objective,
                    evaluation_record_count=split.evaluation_size,
                    question_hashes=hashes,
                    settings=optimizer.adaptive,
                    bootstrap=optimizer.bootstrap,
                    correlation=optimizer.correlation,
                    input_price_per_million=input_rate,
                    output_price_per_million=output_rate,
                    expected_executions=horizon,
                    tie_relative_tolerance=optimizer.planner.tie_relative_tolerance,
                    economics=optimizer.economics,
                    savings_upper_bound_per_execution=bound,
                    savings_bound_basis=basis,
                    estimated_cost_per_decision=average_decision_cost,
                )
                for horizon in ADAPTIVE_HORIZONS
            }
            for basis, bound in gate_bounds.items()
        },
    }

    input_files = {
        "profiling_decisions": experiment / "profiling" / "decisions.jsonl",
        "profiling_report": experiment / "profiling" / "report.json",
        "split_manifest": experiment / "split_manifest.json",
        "milestone_2_selected_plan": experiment / "planning" / "selected_plan.json",
        "milestone_2_candidate_plans": experiment / "planning" / "candidate_plans.json",
        "optimized_decisions": experiment / "optimized" / "decisions.jsonl",
        "optimized_report": experiment / "optimized" / "report.json",
        "optimized_matches": experiment / "optimized" / "matches.jsonl",
        "historical_decisions": inputs.historical_baseline_dir / "decisions.jsonl",
        "historical_manifest": inputs.historical_baseline_dir / "manifest.json",
        "historical_matches": inputs.historical_baseline_dir / "matches.jsonl",
    }
    manifest = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "live_provider_calls": 0,
        "evidence": "stored traces, offline replay, and deterministic computation only",
        "query_id": query.query_id,
        "query_definition": query.model_dump(mode="json"),
        "predicate_hashes": hashes,
        "model_identifier": model,
        "threshold": query.threshold,
        "dataset_fingerprint": split.dataset_fingerprint,
        "profiling_fingerprint": split.profiling_fingerprint,
        "evaluation_fingerprint": split.evaluation_fingerprint,
        "profile_matrix_fingerprint": matrix.fingerprint(),
        "optimizer_settings": optimizer.model_dump(mode="json"),
        "config_paths": inputs.config_paths,
        "input_files": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in input_files.items()
        },
        "extra_sources": [
            {"label": label, "path": str(path), "sha256": file_sha256(path)}
            for label, path, _ in inputs.extra_sources
        ],
        "git_commit": git_commit(),
        "git_worktree_dirty": git_worktree_dirty(),
    }
    result = {
        "manifest": manifest,
        "counterfactual": counterfactual,
        "joint_statistics": joint_statistics(
            matrix, version=optimizer.correlation.joint_statistics_version
        ),
        "cardinality": cardinality,
        "calibration": calibration,
        "planning": planning,
        "bootstrap": bootstrap,
        "economics": economics,
        "adaptive": adaptive,
    }
    for name in (
        "manifest",
        "counterfactual",
        "joint_statistics",
        "cardinality",
        "calibration",
        "planning",
        "bootstrap",
        "economics",
        "adaptive",
    ):
        ResultStore.write_json(output_dir / f"{name}.json", result[name])
    (output_dir / "summary.md").write_text(_summary_markdown(result), encoding="utf-8")
    return result


def stored_plan_settings(
    optimizer: OptimizerSettings, stored_plan: PlanSelection, estimator: str
) -> Any:
    """Planner settings matching the historical objective, for a named estimator."""
    version = (
        stored_plan.cost_model_version
        if estimator == "independent"
        else optimizer.planner.cost_model_version
        if optimizer.planner.cardinality_estimator == "correlation_aware"
        else "correlation-aware-and-v1"
    )
    return optimizer.planner.model_copy(
        update={
            "objective": stored_plan.objective,
            "cardinality_estimator": estimator,
            "cost_model_version": version,
            "statistics_version": stored_plan.statistics_version,
            "optimizer_version": stored_plan.optimizer_version,
        }
    )
