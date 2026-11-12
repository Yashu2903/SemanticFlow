"""Milestone 4.5 profiling-only calibration and optimizer tuning.

Hyperparameters are selected from profiling-validation evidence only.  The 800-review live
traces are used solely for a frozen retrospective report and never enter the selection
objective.  The four workloads already informed earlier milestones, so the held-out
comparison remains exploratory.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from semanticflow.optimizer.adaptive import simulate_adaptive_profiling
from semanticflow.optimizer.bootstrap import (
    collect_paired_bootstrap,
    order_costs,
    quantile,
    select_minimum,
)
from semanticflow.optimizer.calibration import (
    calibrate_orders,
    coverage_summary,
    observed_matrix_counts,
)
from semanticflow.optimizer.cardinality import (
    build_estimator,
    predicate_statistics_from_matrix,
)
from semanticflow.optimizer.cost_model import estimate_plan
from semanticflow.optimizer.joint import simulate_joint_adaptive_risk
from semanticflow.optimizer.joint_benchmark import (
    DEFAULT_HORIZONS as JOINT_HORIZONS,
)
from semanticflow.optimizer.joint_benchmark import (
    evaluate_joint_workload,
)
from semanticflow.optimizer.models import (
    BootstrapSettings,
    CorrelationSettings,
    OptimizerSettings,
    ProfilingResult,
    RiskSettings,
    SplitManifest,
)
from semanticflow.optimizer.profiling_splits import (
    assert_consistent_partitions,
    split_matrix,
)
from semanticflow.optimizer.risk import select_risk_aware_plan
from semanticflow.optimizer.risk_benchmark import DEFAULT_WORKLOADS, load_observations
from semanticflow.optimizer.workflow import (
    _load_json,
    _profiling_report_path,
    load_profile_matrix,
)
from semanticflow.query.models import load_query
from semanticflow.storage.results import ResultStore, git_commit, git_worktree_dirty

CALIBRATION_SCHEMA_VERSION = "m45-calibration-v1"
FROZEN_SCHEMA_VERSION = "m45-frozen-config-v1"
INTERVAL_LEVELS = (0.5, 0.8, 0.9, 0.95)
BOOTSTRAP_ITERATION_SWEEP = (500, 1000, 2000)
PRIOR_STRENGTHS = (0.5, 1.0, 2.0)
SAVING_PROBABILITIES = (0.9, 0.95)
RELATIVE_SAVINGS = (0.01,)
CHECKPOINT_SCHEDULES = (
    (25, 50, 100, 150, 200),
    (25, 50, 100),
    (50, 100, 200),
)
SELECTION_HORIZON = 10
TUNING_JOINT_ITERATIONS = 120
TUNING_VOI_SCENARIOS = 12
TUNING_FOLDS = 2


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fingerprint_mapping(payload: Mapping[str, Any]) -> str:
    return _sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def _exact_cost_usd(
    matrix,
    order: Sequence[str],
    *,
    input_price: float,
    output_price: float,
) -> float:
    """Exact early-rejection cost over a complete matrix using recorded token usage."""
    if not matrix.complete:
        raise ValueError("exact cost requires a complete matrix")
    indices = [matrix.index(item) for item in order]
    total = 0.0
    for row in matrix.rows:
        for index in indices:
            input_tokens = row.input_tokens[index]
            output_tokens = row.output_tokens[index]
            if input_tokens is None or output_tokens is None:
                raise ValueError("exact cost requires recorded token usage")
            total += (
                input_tokens * input_price + output_tokens * output_price
            ) / 1_000_000
            if not row.outcomes[index]:
                break
    return total


def _profile_cost(matrix, input_price: float | None, output_price: float | None) -> float | None:
    if input_price is None or output_price is None:
        return None
    total = 0.0
    for row in matrix.rows:
        for input_tokens, output_tokens in zip(row.input_tokens, row.output_tokens, strict=True):
            if input_tokens is None or output_tokens is None:
                return None
            total += (input_tokens * input_price + output_tokens * output_price) / 1_000_000
    return total


def _context(spec: Mapping[str, str], optimizer: OptimizerSettings) -> dict[str, Any]:
    query = load_query(spec["query"])
    profile_dir = Path(spec["profile_dir"])
    profile = ProfilingResult.model_validate(_load_json(_profiling_report_path(profile_dir)))
    split = SplitManifest.model_validate(_load_json(profile_dir / "split_manifest.json"))
    matrix = load_profile_matrix(profile_dir, profile, split)
    analysis = _load_json(Path(spec["analysis"]))
    observations = load_observations(analysis)
    original = [item.id for item in query.predicates]
    pricing = profile.pricing or {}
    return {
        "id": spec["id"],
        "query": query,
        "profile": profile,
        "split": split,
        "matrix": matrix,
        "observations": observations,
        "original": original,
        "input_rate": pricing.get("input_per_million_usd"),
        "output_rate": pricing.get("output_per_million_usd"),
        "profile_cost": float(profile.estimated_cost_usd or 0.0),
        "optimizer": optimizer,
    }


def reproduce_baseline_fingerprints(output_dir: Path) -> dict[str, Any]:
    """Record hashes of the canonical Milestone 4 artifacts without modifying them."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    sources = {
        "offline_benchmark": Path(
            "results/milestone_4/offline_benchmark_20261106/benchmark.json"
        ),
        "offline_benchmark_reproduction": Path(
            "results/milestone_4/offline_benchmark_joint_baseline_reproduction_20261106/"
            "benchmark.json"
        ),
        "joint_adaptive_risk": Path(
            "results/milestone_4/joint_adaptive_risk_20261106_2000/benchmark.json"
        ),
    }
    fingerprints = {}
    metrics = {}
    for name, path in sources.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        created = payload.pop("created_at", None)
        fingerprints[name] = {
            "path": str(path),
            "sha256_without_created_at": fingerprint_mapping(payload),
            "created_at_recorded": created,
            "live_jev_calls": payload.get("live_jev_calls", payload.get("paid_api_calls")),
        }
        if name.startswith("offline"):
            risk = payload["strategies"]["risk_aware"]
            metrics[name] = {
                "risk_aware_mean_saving_percent": risk["mean_saving_percent"],
                "risk_aware_keep_original": risk["workloads_retaining_original"],
            }
        if name == "joint_adaptive_risk":
            metrics[name] = {
                "R10_existing_adaptive_net": payload["aggregates"]["10"]["existing_adaptive"][
                    "total_net_saving_usd"
                ],
                "R10_existing_adaptive_risk_net": payload["aggregates"]["10"][
                    "existing_adaptive_risk"
                ]["total_net_saving_usd"],
                "R10_joint_net": payload["aggregates"]["10"]["joint"]["total_net_saving_usd"],
            }
    per_workload = {}
    for workload in (
        "product_quality",
        "listing_mismatch",
        "positive_experience",
        "mixed_feedback",
    ):
        left = Path(f"results/milestone_4/offline_benchmark_20261106/{workload}.json")
        right = Path(
            "results/milestone_4/offline_benchmark_joint_baseline_reproduction_20261106/"
            f"{workload}.json"
        )
        left_payload = json.loads(left.read_text(encoding="utf-8"))
        right_payload = json.loads(right.read_text(encoding="utf-8"))
        per_workload[workload] = {
            "identical_to_reproduction": left_payload == right_payload,
            "sha256": fingerprint_mapping(left_payload),
        }
    report = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "phase": "baseline_reproduction",
        "created_at": _now(),
        "git_commit": git_commit(),
        "git_worktree_dirty": git_worktree_dirty(),
        "live_jev_calls": 0,
        "fingerprints": fingerprints,
        "metrics": metrics,
        "per_workload_offline_benchmark": per_workload,
        "reproduction_status": (
            "confirmed"
            if all(item["identical_to_reproduction"] for item in per_workload.values())
            else "mismatch"
        ),
    }
    ResultStore.write_json(output_dir / "baseline_reproduction.json", report)
    return report


def calibrate_cardinality_fold(
    train,
    validation,
    *,
    question_hashes: Mapping[str, str],
    correlation: CorrelationSettings,
    prior_strength: float,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    settings = correlation.model_copy(update={"prior_strength": prior_strength})
    orders = list(itertools.permutations(train.predicate_ids))
    observations = {order: observed_matrix_counts(validation, order) for order in orders}
    by_estimator = {}
    for name in ("independent", "correlation_aware"):
        records = calibrate_orders(
            train,
            observations,
            estimator_name=name,
            question_hashes=question_hashes,
            correlation=settings,
            record_count=validation.size,
            iterations=iterations,
            seed=seed,
        )
        ranking = _plan_ranking(
            train,
            validation,
            estimator_name=name,
            question_hashes=question_hashes,
            correlation=settings,
            input_price=1.0,
            output_price=0.0,
            objective="total_tokens",
        )
        by_estimator[name] = {
            "coverage": coverage_summary(records),
            "plan_ranking": ranking,
            "record_count": len(records),
        }
    return {
        "prior_strength": prior_strength,
        "estimators": by_estimator,
        "correlation_lower_mae_than_independent": (
            abs(
                by_estimator["correlation_aware"]["coverage"]["all_quantities"][
                    "mean_signed_error_of_point_prediction"
                ]
            )
            <= abs(
                by_estimator["independent"]["coverage"]["all_quantities"][
                    "mean_signed_error_of_point_prediction"
                ]
            )
        ),
        "correlation_better_ranking_concordance": (
            by_estimator["correlation_aware"]["plan_ranking"]["pairwise_concordance"]
            >= by_estimator["independent"]["plan_ranking"]["pairwise_concordance"]
        ),
        "correlation_lower_selection_regret": (
            by_estimator["correlation_aware"]["plan_ranking"]["selection_regret_tokens"]
            <= by_estimator["independent"]["plan_ranking"]["selection_regret_tokens"]
        ),
    }


def _plan_ranking(
    train,
    validation,
    *,
    estimator_name: str,
    question_hashes: Mapping[str, str],
    correlation: CorrelationSettings,
    input_price: float,
    output_price: float,
    objective: str,
) -> dict[str, Any]:
    statistics = predicate_statistics_from_matrix(
        train,
        question_hashes=question_hashes,
        input_price_per_million=input_price,
        output_price_per_million=output_price,
    )
    estimator = build_estimator(
        estimator_name,
        train,
        question_hashes=question_hashes,
        correlation=correlation,
        input_price_per_million=input_price,
        output_price_per_million=output_price,
    )
    predicted = {}
    observed = {}
    for order in itertools.permutations(train.predicate_ids):
        plan = estimate_plan(
            order,
            statistics,
            input_record_count=validation.size,
            objective=objective,
            original_order=train.predicate_ids,
            estimator=estimator,
        )
        predicted[order] = plan.objective_value
        total = 0.0
        indices = [validation.index(item) for item in order]
        for row in validation.rows:
            for index in indices:
                total += float(row.input_tokens[index] or 0) + float(
                    row.output_tokens[index] or 0
                )
                if not row.outcomes[index]:
                    break
        observed[order] = total
    original = tuple(train.predicate_ids)
    predicted_selected = select_minimum(predicted, original, 1e-9)
    observed_selected = select_minimum(observed, original, 1e-9)
    pairs = list(itertools.combinations(sorted(predicted), 2))
    concordant = 0
    for left, right in pairs:
        pred = predicted[left] - predicted[right]
        obs = observed[left] - observed[right]
        if pred == 0 and obs == 0 or pred * obs > 0:
            concordant += 1
    regret = observed[predicted_selected] - observed[observed_selected]
    abs_errors = [abs(predicted[order] - observed[order]) for order in predicted]
    return {
        "predicted_selected_order": list(predicted_selected),
        "observed_selected_order": list(observed_selected),
        "selected_matches_observed_cheapest": predicted_selected == observed_selected,
        "selection_regret_tokens": regret,
        "pairwise_concordance": concordant / len(pairs) if pairs else 1.0,
        "mean_absolute_cost_error": sum(abs_errors) / len(abs_errors),
        "mean_signed_cost_error": sum(
            predicted[order] - observed[order] for order in predicted
        )
        / len(predicted),
    }


def calibrate_bootstrap_fold(
    train,
    validation,
    *,
    question_hashes: Mapping[str, str],
    correlation: CorrelationSettings,
    iterations: int,
    seed: int,
    input_price: float,
    output_price: float,
) -> dict[str, Any]:
    settings = BootstrapSettings(iterations=iterations, seed=seed, interval_level=0.95)
    collected = collect_paired_bootstrap(
        train,
        estimator_name="correlation_aware",
        original_order=train.predicate_ids,
        objective="estimated_cost_usd",
        evaluation_record_count=validation.size,
        question_hashes=question_hashes,
        settings=settings,
        correlation=correlation,
        input_price_per_million=input_price,
        output_price_per_million=output_price,
    )
    coverage = {str(level): {"hits": 0, "checks": 0} for level in INTERVAL_LEVELS}
    # Interval calibration on train-resample vs validation exact relative savings.
    original = tuple(train.predicate_ids)
    original_val = _exact_cost_usd(
        validation, original, input_price=input_price, output_price=output_price
    )
    for order, draws in collected["draws"].items():
        observed_cost = _exact_cost_usd(
            validation, order, input_price=input_price, output_price=output_price
        )
        observed_saving = original_val - observed_cost
        predicted_savings = [
            collected["draws"][original][index] - draws[index] for index in range(len(draws))
        ]
        for level in INTERVAL_LEVELS:
            tail = (1 - level) / 2
            lower = quantile(predicted_savings, tail)
            upper = quantile(predicted_savings, 1 - tail)
            coverage[str(level)]["checks"] += 1
            if lower <= observed_saving <= upper:
                coverage[str(level)]["hits"] += 1
    counts = Counter(collected["selected_orders"])
    modal_order, modal_count = max(sorted(counts.items()), key=lambda item: item[1])
    return {
        "iterations": iterations,
        "seed": seed,
        "interval_coverage": {
            level: (
                None
                if payload["checks"] == 0
                else payload["hits"] / payload["checks"]
            )
            for level, payload in coverage.items()
        },
        "interval_checks": {level: payload["checks"] for level, payload in coverage.items()},
        "modal_order": list(modal_order),
        "modal_frequency": modal_count / collected["iterations"],
        "interpretation": "sampling_uncertainty_diagnostic",
        "not_a_calibrated_production_probability": True,
        "note": (
            "coverage compares train-bootstrap savings intervals with exact validation-matrix "
            "savings; this is profiling-only calibration, not production calibration"
        ),
    }


def evaluate_policy_on_fold(
    train,
    validation,
    *,
    original: Sequence[str],
    question_hashes: Mapping[str, str],
    optimizer: OptimizerSettings,
    policy: str,
    horizon: int,
    input_price: float,
    output_price: float,
) -> dict[str, Any]:
    """Select a plan using train evidence only; score exact validation execution cost."""
    economics = optimizer.economics.model_copy(update={"expected_executions": horizon})
    width = len(original)
    structural_bound = validation.size * (width - 1) * (
        (_profile_cost(train, input_price, output_price) or 0.0) / max(1, train.size * width)
    )
    if policy == "fixed":
        selected = list(original)
        profiling_cost = 0.0
        decisions = 0
        action = "KEEP_ORIGINAL"
    elif policy == "independent_full":
        costs = order_costs(
            train,
            estimator_name="independent",
            original_order=original,
            objective="estimated_cost_usd",
            evaluation_record_count=validation.size,
            question_hashes=question_hashes,
            correlation=optimizer.correlation,
            input_price_per_million=input_price,
            output_price_per_million=output_price,
        )
        selected = list(select_minimum(costs, tuple(original), 1e-9))
        profiling_cost = _profile_cost(train, input_price, output_price) or 0.0
        decisions = train.size * width
        action = "ADOPT_REORDER" if selected != list(original) else "KEEP_ORIGINAL"
    elif policy == "correlation_full":
        costs = order_costs(
            train,
            estimator_name="correlation_aware",
            original_order=original,
            objective="estimated_cost_usd",
            evaluation_record_count=validation.size,
            question_hashes=question_hashes,
            correlation=optimizer.correlation,
            input_price_per_million=input_price,
            output_price_per_million=output_price,
        )
        selected = list(select_minimum(costs, tuple(original), 1e-9))
        profiling_cost = _profile_cost(train, input_price, output_price) or 0.0
        decisions = train.size * width
        action = "ADOPT_REORDER" if selected != list(original) else "KEEP_ORIGINAL"
    elif policy == "confidence":
        selected, explanation = select_risk_aware_plan(
            train,
            original_order=original,
            objective="estimated_cost_usd",
            evaluation_record_count=validation.size,
            question_hashes=question_hashes,
            bootstrap=optimizer.bootstrap.model_copy(
                update={"iterations": min(optimizer.bootstrap.iterations, 200)}
            ),
            correlation=optimizer.correlation,
            risk=RiskSettings(
                minimum_relative_saving=optimizer.bootstrap.minimum_relative_saving
                or optimizer.risk.minimum_relative_saving,
                minimum_saving_probability=optimizer.bootstrap.confidence_level,
            ),
            economics=economics,
            input_price_per_million=input_price,
            output_price_per_million=output_price,
            tie_relative_tolerance=1e-9,
            primary_estimator="correlation_aware",
            statistics_state="available",
            new_profiling_cost=_profile_cost(train, input_price, output_price),
            sunk_profiling_cost=_profile_cost(train, input_price, output_price) or 0.0,
        )
        profiling_cost = _profile_cost(train, input_price, output_price) or 0.0
        decisions = train.size * width
        action = explanation["decision"]
    elif policy == "risk_aware":
        selected, explanation = select_risk_aware_plan(
            train,
            original_order=original,
            objective="estimated_cost_usd",
            evaluation_record_count=validation.size,
            question_hashes=question_hashes,
            bootstrap=optimizer.bootstrap.model_copy(
                update={"iterations": min(optimizer.bootstrap.iterations, TUNING_JOINT_ITERATIONS)}
            ),
            correlation=optimizer.correlation,
            risk=optimizer.risk,
            economics=economics,
            input_price_per_million=input_price,
            output_price_per_million=output_price,
            tie_relative_tolerance=1e-9,
            primary_estimator="correlation_aware",
            statistics_state="available",
            new_profiling_cost=_profile_cost(train, input_price, output_price),
            sunk_profiling_cost=_profile_cost(train, input_price, output_price) or 0.0,
        )
        profiling_cost = _profile_cost(train, input_price, output_price) or 0.0
        decisions = train.size * width
        action = explanation["decision"]
    elif policy == "adaptive":
        result = simulate_adaptive_profiling(
            train,
            estimator_name="correlation_aware",
            original_order=original,
            objective="estimated_cost_usd",
            evaluation_record_count=validation.size,
            question_hashes=question_hashes,
            settings=optimizer.adaptive,
            bootstrap=optimizer.bootstrap.model_copy(
                update={"iterations": min(200, optimizer.bootstrap.iterations)}
            ),
            correlation=optimizer.correlation,
            input_price_per_million=input_price,
            output_price_per_million=output_price,
            expected_executions=horizon,
            economics=economics,
            savings_upper_bound_per_execution=structural_bound,
            savings_bound_basis="structural_upper_bound",
            estimated_cost_per_decision=(
                (_profile_cost(train, input_price, output_price) or 0.0)
                / max(1, train.size * width)
            ),
            statistics_available=False,
        )
        selected = list(result["selected_order"])
        if result.get("profiling_started") and result.get("stopping_checkpoint"):
            profiling_cost = float(
                result["stopping_checkpoint"]["profiling_cost_usd_offline_estimate"] or 0.0
            )
            decisions = int(result["stopping_checkpoint"]["profiling_decisions"])
        else:
            profiling_cost = 0.0
            decisions = 0
        action = "ADOPT_REORDER" if selected != list(original) else "KEEP_ORIGINAL"
    elif policy == "joint":
        result = simulate_joint_adaptive_risk(
            train,
            estimator_name="correlation_aware",
            original_order=original,
            objective="estimated_cost_usd",
            evaluation_record_count=validation.size,
            question_hashes=question_hashes,
            checkpoints=[
                size for size in optimizer.adaptive.checkpoints if size <= train.size
            ]
            or [max(5, train.size // 4), max(10, train.size // 2), train.size],
            sampling_seed=optimizer.adaptive.seed,
            bootstrap=optimizer.bootstrap.model_copy(
                update={"iterations": TUNING_JOINT_ITERATIONS}
            ),
            correlation=optimizer.correlation,
            risk=optimizer.risk,
            economics=economics,
            joint=optimizer.joint.model_copy(
                update={
                    "bootstrap_iterations": TUNING_JOINT_ITERATIONS,
                    "voi_scenarios": TUNING_VOI_SCENARIOS,
                }
            ),
            input_price_per_million=input_price,
            output_price_per_million=output_price,
            expected_executions=horizon,
            savings_upper_bound_per_execution=structural_bound,
            estimated_cost_per_decision=(
                (_profile_cost(train, input_price, output_price) or 0.0)
                / max(1, train.size * width)
            ),
            statistics_available=False,
        )
        selected = list(result["selected_order"])
        profiling_cost = float(result.get("profiling_cost_usd") or 0.0)
        decisions = int(result.get("profiling_decisions_consumed") or 0)
        action = str(result.get("action") or "KEEP_ORIGINAL")
    else:
        raise ValueError(f"unknown policy {policy}")

    original_cost = _exact_cost_usd(
        validation, original, input_price=input_price, output_price=output_price
    )
    selected_cost = _exact_cost_usd(
        validation, selected, input_price=input_price, output_price=output_price
    )
    cheapest = min(
        _exact_cost_usd(validation, order, input_price=input_price, output_price=output_price)
        for order in itertools.permutations(original)
    )
    saving = original_cost - selected_cost
    return {
        "policy": policy,
        "action": action,
        "selected_order": selected,
        "profiling_decisions": decisions,
        "profiling_cost_usd": profiling_cost,
        "validation_original_cost_usd": original_cost,
        "validation_selected_cost_usd": selected_cost,
        "validation_saving_usd": saving,
        "validation_regret_usd": selected_cost - cheapest,
        "cold_start_net_usd": horizon * saving - profiling_cost,
        "warm_start_net_usd": horizon * saving,
        "horizon": horizon,
    }


def run_profiling_only_calibration(
    optimizer: OptimizerSettings,
    *,
    folds: int = 4,
    fold_seed: int = 7073,
    predictive_iterations: int = 40,
    workloads: Sequence[Mapping[str, str]] = DEFAULT_WORKLOADS,
) -> dict[str, Any]:
    contexts = [_context(spec, optimizer) for spec in workloads]
    consistency = assert_consistent_partitions(
        [ctx["matrix"] for ctx in contexts], folds=folds, seed=fold_seed
    )
    cardinality_rows = []
    bootstrap_rows = []
    policy_rows = []
    for fold in range(folds):
        for ctx in contexts:
            train, validation, meta = split_matrix(
                ctx["matrix"], folds=folds, fold=fold, seed=fold_seed
            )
            for prior in PRIOR_STRENGTHS:
                cardinality_rows.append(
                    {
                        "workload_id": ctx["id"],
                        "fold": fold,
                        **calibrate_cardinality_fold(
                            train,
                            validation,
                            question_hashes=ctx["profile"].predicate_hashes,
                            correlation=optimizer.correlation,
                            prior_strength=prior,
                            iterations=predictive_iterations,
                            seed=optimizer.bootstrap.seed + fold,
                        ),
                        "split": meta,
                    }
                )
            for iterations in (500, 1000):
                bootstrap_rows.append(
                    {
                        "workload_id": ctx["id"],
                        "fold": fold,
                        **calibrate_bootstrap_fold(
                            train,
                            validation,
                            question_hashes=ctx["profile"].predicate_hashes,
                            correlation=optimizer.correlation,
                            iterations=iterations,
                            seed=optimizer.bootstrap.seed + fold,
                            input_price=float(ctx["input_rate"]),
                            output_price=float(ctx["output_rate"]),
                        ),
                        "requested_iterations": iterations,
                    }
                )
            for policy in (
                "fixed",
                "independent_full",
                "correlation_full",
                "confidence",
                "risk_aware",
                "adaptive",
                "joint",
            ):
                policy_rows.append(
                    {
                        "workload_id": ctx["id"],
                        "fold": fold,
                        **evaluate_policy_on_fold(
                            train,
                            validation,
                            original=ctx["original"],
                            question_hashes=ctx["profile"].predicate_hashes,
                            optimizer=optimizer,
                            policy=policy,
                            horizon=SELECTION_HORIZON,
                            input_price=float(ctx["input_rate"]),
                            output_price=float(ctx["output_rate"]),
                        ),
                    }
                )
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "phase": "profiling_only_calibration",
        "created_at": _now(),
        "folds": folds,
        "fold_seed": fold_seed,
        "selection_horizon": SELECTION_HORIZON,
        "held_out_live_reviews_used_for_selection": False,
        "partition_consistency": consistency,
        "cardinality": cardinality_rows,
        "bootstrap": bootstrap_rows,
        "policies": policy_rows,
        "summaries": {
            "cardinality": _summarize_cardinality(cardinality_rows),
            "bootstrap": _summarize_bootstrap(bootstrap_rows),
            "policies": _summarize_policies(policy_rows),
        },
    }


def _summarize_cardinality(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_prior: dict[float, list] = {}
    for row in rows:
        by_prior.setdefault(row["prior_strength"], []).append(row)
    summary = {}
    for prior, items in sorted(by_prior.items()):
        corr_regret = [
            item["estimators"]["correlation_aware"]["plan_ranking"]["selection_regret_tokens"]
            for item in items
        ]
        ind_regret = [
            item["estimators"]["independent"]["plan_ranking"]["selection_regret_tokens"]
            for item in items
        ]
        corr_conc = [
            item["estimators"]["correlation_aware"]["plan_ranking"]["pairwise_concordance"]
            for item in items
        ]
        ind_conc = [
            item["estimators"]["independent"]["plan_ranking"]["pairwise_concordance"]
            for item in items
        ]
        summary[str(prior)] = {
            "folds": len(items),
            "mean_correlation_regret_tokens": sum(corr_regret) / len(corr_regret),
            "mean_independent_regret_tokens": sum(ind_regret) / len(ind_regret),
            "mean_correlation_concordance": sum(corr_conc) / len(corr_conc),
            "mean_independent_concordance": sum(ind_conc) / len(ind_conc),
            "correlation_wins_regret": sum(
                c <= i for c, i in zip(corr_regret, ind_regret, strict=True)
            ),
            "correlation_wins_concordance": sum(
                c >= i for c, i in zip(corr_conc, ind_conc, strict=True)
            ),
        }
    return summary


def _summarize_bootstrap(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_iter: dict[int, list] = {}
    for row in rows:
        by_iter.setdefault(int(row["requested_iterations"]), []).append(row)
    summary = {}
    for iterations, items in sorted(by_iter.items()):
        coverages = {str(level): [] for level in INTERVAL_LEVELS}
        for item in items:
            for level, value in item["interval_coverage"].items():
                if value is not None:
                    coverages[level].append(value)
        summary[str(iterations)] = {
            "mean_coverage": {
                level: (sum(values) / len(values) if values else None)
                for level, values in coverages.items()
            },
            "mean_modal_frequency": sum(item["modal_frequency"] for item in items) / len(items),
            "folds": len(items),
        }
    return summary


def _summarize_policies(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_policy: dict[str, list] = {}
    for row in rows:
        by_policy.setdefault(row["policy"], []).append(row)
    summary = {}
    for policy, items in sorted(by_policy.items()):
        nets = [item["cold_start_net_usd"] for item in items]
        regrets = [item["validation_regret_usd"] for item in items]
        profiling = [item["profiling_cost_usd"] for item in items]
        leave = {}
        workloads = sorted({item["workload_id"] for item in items})
        for left in workloads:
            others = [
                item["cold_start_net_usd"]
                for item in items
                if item["workload_id"] != left
            ]
            leave[left] = sum(others) / len(others) if others else None
        summary[policy] = {
            "mean_cold_start_net_usd": sum(nets) / len(nets),
            "median_cold_start_net_usd": median(nets),
            "mean_validation_regret_usd": sum(regrets) / len(regrets),
            "mean_profiling_cost_usd": sum(profiling) / len(profiling),
            "leave_one_workload_out_mean_cold_net_usd": leave,
            "folds": len(items),
        }
    return summary


def tune_configuration(
    optimizer: OptimizerSettings,
    *,
    folds: int = TUNING_FOLDS,
    fold_seed: int = 7073,
    workloads: Sequence[Mapping[str, str]] = DEFAULT_WORKLOADS,
) -> dict[str, Any]:
    """Predefined grid search scored only on profiling-validation cold-start net at R=10."""
    contexts = [_context(spec, optimizer) for spec in workloads]
    assert_consistent_partitions(
        [ctx["matrix"] for ctx in contexts], folds=folds, seed=fold_seed
    )
    candidates = []
    for prior in PRIOR_STRENGTHS:
        for probability in SAVING_PROBABILITIES:
            for relative in RELATIVE_SAVINGS:
                for checkpoints in CHECKPOINT_SCHEDULES:
                    candidates.append(
                        {
                            "prior_strength": prior,
                            "minimum_saving_probability": probability,
                            "minimum_relative_saving": relative,
                            "checkpoints": list(checkpoints),
                            "voi_decision_rule": "point_estimate",
                            "uncertainty_contraction_exponent": 0.5,
                        }
                    )

    def score_policy(configured: OptimizerSettings, policy: str) -> list[dict[str, Any]]:
        rows = []
        for fold in range(folds):
            for ctx in contexts:
                train, validation, _ = split_matrix(
                    ctx["matrix"], folds=folds, fold=fold, seed=fold_seed
                )
                rows.append(
                    evaluate_policy_on_fold(
                        train,
                        validation,
                        original=ctx["original"],
                        question_hashes=ctx["profile"].predicate_hashes,
                        optimizer=configured,
                        policy=policy,
                        horizon=SELECTION_HORIZON,
                        input_price=float(ctx["input_rate"]),
                        output_price=float(ctx["output_rate"]),
                    )
                    | {"workload_id": ctx["id"], "fold": fold}
                )
        return rows

    adaptive_rows = score_policy(optimizer, "adaptive")
    adaptive_net = sum(row["cold_start_net_usd"] for row in adaptive_rows) / len(adaptive_rows)
    adaptive_leave = {}
    for left in sorted({row["workload_id"] for row in adaptive_rows}):
        others = [
            row["cold_start_net_usd"] for row in adaptive_rows if row["workload_id"] != left
        ]
        adaptive_leave[left] = sum(others) / len(others)
    reference_adaptive = {
        "adaptive_mean_cold_net_usd": adaptive_net,
        "leave_one_workload_out_adaptive_cold_net_usd": adaptive_leave,
        "adaptive_mean_profiling_cost_usd": sum(
            row["profiling_cost_usd"] for row in adaptive_rows
        )
        / len(adaptive_rows),
        "adaptive_mean_regret_usd": sum(row["validation_regret_usd"] for row in adaptive_rows)
        / len(adaptive_rows),
    }

    scored = []
    for candidate in candidates:
        configured = optimizer.model_copy(
            update={
                "correlation": optimizer.correlation.model_copy(
                    update={"prior_strength": candidate["prior_strength"]}
                ),
                "risk": optimizer.risk.model_copy(
                    update={
                        "minimum_saving_probability": candidate[
                            "minimum_saving_probability"
                        ],
                        "minimum_relative_saving": candidate["minimum_relative_saving"],
                    }
                ),
                "adaptive": optimizer.adaptive.model_copy(
                    update={"checkpoints": candidate["checkpoints"]}
                ),
                "joint": optimizer.joint.model_copy(
                    update={
                        "voi_decision_rule": candidate["voi_decision_rule"],
                        "uncertainty_contraction_exponent": candidate[
                            "uncertainty_contraction_exponent"
                        ],
                        "bootstrap_iterations": TUNING_JOINT_ITERATIONS,
                        "voi_scenarios": TUNING_VOI_SCENARIOS,
                    }
                ),
                "bootstrap": optimizer.bootstrap.model_copy(
                    update={"iterations": TUNING_JOINT_ITERATIONS}
                ),
            }
        )
        joint_rows = score_policy(configured, "joint")
        joint_net = sum(row["cold_start_net_usd"] for row in joint_rows) / len(joint_rows)
        leave = {}
        for left in sorted({row["workload_id"] for row in joint_rows}):
            others = [
                row["cold_start_net_usd"]
                for row in joint_rows
                if row["workload_id"] != left
            ]
            leave[left] = sum(others) / len(others)
        scored.append(
            {
                "candidate": candidate,
                "joint_mean_cold_net_usd": joint_net,
                "adaptive_mean_cold_net_usd": adaptive_net,
                "joint_mean_profiling_cost_usd": sum(
                    row["profiling_cost_usd"] for row in joint_rows
                )
                / len(joint_rows),
                "joint_mean_regret_usd": sum(
                    row["validation_regret_usd"] for row in joint_rows
                )
                / len(joint_rows),
                "leave_one_workload_out_joint_cold_net_usd": leave,
                "complexity_score": (
                    len(candidate["checkpoints"])
                    + (0 if candidate["minimum_saving_probability"] == 0.95 else 1)
                    + (0 if candidate["minimum_relative_saving"] == 0.01 else 1)
                    + (0 if candidate["prior_strength"] == 1.0 else 1)
                ),
            }
        )

    def sort_key(item: Mapping[str, Any]) -> tuple:
        leave_values = list(item["leave_one_workload_out_joint_cold_net_usd"].values())
        return (
            item["joint_mean_cold_net_usd"],
            min(leave_values),
            -item["complexity_score"],
            -item["joint_mean_profiling_cost_usd"],
        )

    ranked = sorted(scored, key=sort_key, reverse=True)
    best_joint = ranked[0]
    margin = best_joint["joint_mean_cold_net_usd"] - adaptive_net
    leave_mean = sum(best_joint["leave_one_workload_out_joint_cold_net_usd"].values()) / len(
        best_joint["leave_one_workload_out_joint_cold_net_usd"]
    )
    adopt_tuned_joint = margin > 0.00005 and leave_mean > adaptive_net - 0.0001
    selected = {
        "recommended_default": (
            "tuned_joint_adaptive_risk" if adopt_tuned_joint else "existing_adaptive"
        ),
        "experimental_strategy": "tuned_joint_adaptive_risk",
        "selection_objective": (
            f"maximize mean profiling-validation cold-start net at R={SELECTION_HORIZON}; "
            "tie-break by leave-one-workload-out floor then lower complexity"
        ),
        "adopt_tuned_as_default": adopt_tuned_joint,
        "margin_vs_best_adaptive_usd": margin,
        "best_joint_candidate": best_joint["candidate"],
        "best_joint_score": {
            key: value for key, value in best_joint.items() if key != "candidate"
        },
        "reference_adaptive_score": reference_adaptive,
        "grid_size": len(candidates),
        "folds": folds,
        "fold_seed": fold_seed,
        "held_out_live_reviews_used_for_selection": False,
    }
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "phase": "configuration_tuning",
        "created_at": _now(),
        "selection": selected,
        "ranked_candidates": ranked[:12],
        "all_candidate_count": len(ranked),
    }


def freeze_configuration(
    optimizer: OptimizerSettings,
    selection: Mapping[str, Any],
    *,
    calibration_fingerprint: str,
) -> tuple[OptimizerSettings, dict[str, Any]]:
    candidate = selection["best_joint_candidate"]
    frozen = optimizer.model_copy(
        update={
            "correlation": optimizer.correlation.model_copy(
                update={"prior_strength": candidate["prior_strength"]}
            ),
            "risk": optimizer.risk.model_copy(
                update={
                    "minimum_saving_probability": candidate["minimum_saving_probability"],
                    "minimum_relative_saving": candidate["minimum_relative_saving"],
                }
            ),
            "adaptive": optimizer.adaptive.model_copy(
                update={"checkpoints": candidate["checkpoints"]}
            ),
            "joint": optimizer.joint.model_copy(
                update={
                    "voi_decision_rule": candidate["voi_decision_rule"],
                    "uncertainty_contraction_exponent": candidate[
                        "uncertainty_contraction_exponent"
                    ],
                    # Restore full resolution for frozen retrospective evaluation.
                    "bootstrap_iterations": 2000,
                    "voi_scenarios": 32,
                }
            ),
            "bootstrap": optimizer.bootstrap.model_copy(update={"iterations": 2000}),
        }
    )
    manifest = {
        "schema_version": FROZEN_SCHEMA_VERSION,
        "created_at": _now(),
        "git_commit": git_commit(),
        "calibration_fingerprint": calibration_fingerprint,
        "recommended_default": selection["recommended_default"],
        "adopt_tuned_as_default": selection["adopt_tuned_as_default"],
        "selection_objective": selection["selection_objective"],
        "parameters": {
            "correlation.prior_strength": candidate["prior_strength"],
            "risk.minimum_saving_probability": candidate["minimum_saving_probability"],
            "risk.minimum_relative_saving": candidate["minimum_relative_saving"],
            "adaptive.checkpoints": candidate["checkpoints"],
            "joint.voi_decision_rule": candidate["voi_decision_rule"],
            "joint.uncertainty_contraction_exponent": candidate[
                "uncertainty_contraction_exponent"
            ],
            "joint.bootstrap_iterations": 2000,
            "joint.voi_scenarios": 32,
            "bootstrap.iterations": 2000,
        },
        "held_out_live_reviews_used_for_selection": False,
        "note": (
            "Frozen before any retrospective use of the 800-review live outcomes. "
            "Do not edit after examining held-out results."
        ),
    }
    manifest["config_fingerprint"] = fingerprint_mapping(manifest["parameters"])
    return frozen, manifest


def run_retrospective_evaluation(
    base_optimizer: OptimizerSettings,
    frozen_optimizer: OptimizerSettings,
    output_dir: Path,
    *,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Score frozen and baseline strategies on live traces after the config is frozen."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    workloads = []
    for spec in DEFAULT_WORKLOADS:
        base = evaluate_joint_workload(spec, base_optimizer)
        tuned = evaluate_joint_workload(spec, frozen_optimizer)
        # Attach tuned joint under a distinct key while preserving baseline fields.
        workloads.append(
            {
                "workload_id": spec["id"],
                "baseline": base,
                "tuned_joint": {
                    "horizons": {
                        str(horizon): {
                            "joint": tuned["horizons"][str(horizon)]["joint"],
                        }
                        for horizon in JOINT_HORIZONS
                    },
                    "economic_scenarios": {
                        str(horizon): {
                            "joint": tuned["economic_scenarios"][str(horizon)]["joint"],
                        }
                        for horizon in JOINT_HORIZONS
                    },
                },
            }
        )
    aggregates = {}
    for horizon in JOINT_HORIZONS:
        aggregates[str(horizon)] = {}
        for strategy in ("existing_adaptive", "existing_adaptive_risk", "joint"):
            rows = [
                workload["baseline"]["horizons"][str(horizon)][strategy]["score"]
                for workload in workloads
            ]
            measured = [row for row in rows if row["observed_result"] == "measured"]
            savings = [row["observed_saving_percent"] for row in measured]
            nets = [row["net_saving_usd"] for row in measured]
            aggregates[str(horizon)][strategy] = {
                "mean_execution_saving_percent": sum(savings) / len(savings),
                "median_execution_saving_percent": median(savings),
                "total_net_saving_usd": sum(nets),
                "total_profiling_cost_usd": sum(row["profiling_cost_usd"] for row in rows),
            }
        tuned_rows = [
            workload["tuned_joint"]["horizons"][str(horizon)]["joint"]["score"]
            for workload in workloads
        ]
        measured = [row for row in tuned_rows if row["observed_result"] == "measured"]
        savings = [row["observed_saving_percent"] for row in measured]
        nets = [row["net_saving_usd"] for row in measured]
        aggregates[str(horizon)]["tuned_joint"] = {
            "mean_execution_saving_percent": sum(savings) / len(savings),
            "median_execution_saving_percent": median(savings),
            "total_net_saving_usd": sum(nets),
            "total_profiling_cost_usd": sum(row["profiling_cost_usd"] for row in tuned_rows),
            "leave_one_workload_out_mean_execution_saving_percent": {
                workload["workload_id"]: sum(
                    other["tuned_joint"]["horizons"][str(horizon)]["joint"]["score"][
                        "observed_saving_percent"
                    ]
                    for other in workloads
                    if other["workload_id"] != workload["workload_id"]
                )
                / (len(workloads) - 1)
                for workload in workloads
            },
        }
    report = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "phase": "retrospective_held_out_evaluation",
        "created_at": _now(),
        "git_commit": git_commit(),
        "config_path": config_path,
        "label": "exploratory_retrospective_after_freeze",
        "not_a_confirmatory_test": True,
        "live_jev_calls": 0,
        "held_out_outcomes_used_for_selection": False,
        "held_out_outcomes_used_for_reporting_only": True,
        "horizons": list(JOINT_HORIZONS),
        "aggregates": aggregates,
        "workloads": [
            {
                "workload_id": workload["workload_id"],
                "baseline_actions": {
                    str(horizon): {
                        strategy: workload["baseline"]["horizons"][str(horizon)][strategy][
                            "score"
                        ]["action"]
                        for strategy in (
                            "existing_adaptive",
                            "existing_adaptive_risk",
                            "joint",
                        )
                    }
                    for horizon in (1, 10, 100)
                },
                "tuned_joint_actions": {
                    str(horizon): workload["tuned_joint"]["horizons"][str(horizon)]["joint"][
                        "score"
                    ]["action"]
                    for horizon in (1, 10, 100)
                },
                "tuned_joint_scores": {
                    str(horizon): workload["tuned_joint"]["horizons"][str(horizon)]["joint"][
                        "score"
                    ]
                    for horizon in JOINT_HORIZONS
                },
            }
            for workload in workloads
        ],
    }
    ResultStore.write_json(output_dir / "retrospective.json", report)
    return report


def run_milestone_45(
    optimizer: OptimizerSettings,
    output_dir: Path,
    *,
    folds: int = 4,
    fold_seed: int = 7073,
    predictive_iterations: int = 40,
    skip_retrospective: bool = False,
    config_path: str | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    baseline = reproduce_baseline_fingerprints(output_dir / "baseline")
    calibration = run_profiling_only_calibration(
        optimizer,
        folds=folds,
        fold_seed=fold_seed,
        predictive_iterations=predictive_iterations,
    )
    ResultStore.write_json(output_dir / "profiling_calibration.json", calibration)
    # Keep the expensive joint grid on a fixed, smaller fold count so nested selection
    # stays tractable; cardinality/bootstrap calibration still uses the requested folds.
    tuning_folds = min(folds, TUNING_FOLDS)
    tuning = tune_configuration(optimizer, folds=tuning_folds, fold_seed=fold_seed)
    ResultStore.write_json(output_dir / "tuning.json", tuning)
    calibration_fingerprint = fingerprint_mapping(
        {
            "cardinality": calibration["summaries"]["cardinality"],
            "bootstrap": calibration["summaries"]["bootstrap"],
            "policies": calibration["summaries"]["policies"],
            "selection": tuning["selection"],
        }
    )
    frozen_settings, manifest = freeze_configuration(
        optimizer, tuning["selection"], calibration_fingerprint=calibration_fingerprint
    )
    ResultStore.write_json(output_dir / "frozen_config.json", manifest)
    # Write the frozen YAML companion for reproducibility.
    yaml_path = output_dir / "optimizer.milestone_4_5_frozen.yaml"
    _write_frozen_yaml(frozen_settings, yaml_path, manifest)
    retrospective = None
    if not skip_retrospective:
        retrospective = run_retrospective_evaluation(
            optimizer,
            frozen_settings,
            output_dir / "retrospective",
            config_path=config_path,
        )
    report = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "created_at": _now(),
        "git_commit": git_commit(),
        "git_worktree_dirty": git_worktree_dirty(),
        "live_jev_calls": 0,
        "paid_api_calls": 0,
        "baseline_reproduction": baseline["reproduction_status"],
        "baseline_metrics": baseline["metrics"],
        "calibration_summaries": calibration["summaries"],
        "tuning_selection": tuning["selection"],
        "frozen_config": manifest,
        "retrospective_aggregates": None if retrospective is None else retrospective["aggregates"],
        "label": "exploratory_milestone_4_5",
        "not_a_confirmatory_test": True,
    }
    ResultStore.write_json(output_dir / "report.json", report)
    _write_summary(report, output_dir / "summary.md")
    return report


def _write_frozen_yaml(
    settings: OptimizerSettings, path: Path, manifest: Mapping[str, Any]
) -> None:
    payload = settings.model_dump(mode="json")
    text = (
        "# Frozen Milestone 4.5 candidate configuration.\n"
        f"# fingerprint: {manifest['config_fingerprint']}\n"
        f"# recommended_default: {manifest['recommended_default']}\n"
        "# Selected from profiling-only validation before held-out retrospective scoring.\n"
        + _to_yaml(payload)
    )
    path.write_text(text, encoding="utf-8")


def _to_yaml(value: Any, indent: int = 0) -> str:
    space = "  " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, dict | list):
                lines.append(f"{space}{key}:")
                lines.append(_to_yaml(item, indent + 1).rstrip())
            else:
                lines.append(f"{space}{key}: {_yaml_scalar(item)}")
        return "\n".join(lines) + "\n"
    if isinstance(value, list):
        if not value:
            return f"{space}[]\n"
        lines = []
        for item in value:
            if isinstance(item, dict | list):
                lines.append(f"{space}-")
                lines.append(_to_yaml(item, indent + 1).rstrip())
            else:
                lines.append(f"{space}- {_yaml_scalar(item)}")
        return "\n".join(lines) + "\n"
    return f"{space}{_yaml_scalar(value)}\n"


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, int):
        return str(value)
    return json.dumps(str(value))


def _write_summary(report: Mapping[str, Any], path: Path) -> None:
    selection = report["tuning_selection"]
    lines = [
        "# Milestone 4.5 calibration summary",
        "",
        f"Baseline reproduction: {report['baseline_reproduction']}",
        f"Recommended default: {selection['recommended_default']}",
        f"Adopt tuned joint as default: {selection['adopt_tuned_as_default']}",
        f"Margin vs adaptive (profiling validation): {selection['margin_vs_best_adaptive_usd']}",
        "",
        "Held-out live outcomes were not used for selection.",
        "Bootstrap probabilities remain resampling diagnostics.",
        "",
    ]
    if report.get("retrospective_aggregates"):
        lines.extend(
            [
                "## Retrospective cold-start net at R=10",
                "",
                "| Strategy | Net USD | Mean saving % | Profiling USD |",
                "|---|---:|---:|---:|",
            ]
        )
        for name in (
            "existing_adaptive",
            "existing_adaptive_risk",
            "joint",
            "tuned_joint",
        ):
            row = report["retrospective_aggregates"]["10"][name]
            lines.append(
                f"| {name} | {row['total_net_saving_usd']:.6f} | "
                f"{row['mean_execution_saving_percent']:.2f} | "
                f"{row['total_profiling_cost_usd']:.6f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
