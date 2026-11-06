"""Milestone 3 strategy benchmark: offline strategy selection and counterbalanced live sessions.

Strategy selection reads only the profiling split. Sessions execute predicate orders on the
held-out split, one run directory per execution, in a declared counterbalanced schedule so that
fixed and optimized orders run close together in time.
"""

from __future__ import annotations

import itertools
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from semanticflow.optimizer.adaptive import simulate_adaptive_profiling
from semanticflow.optimizer.bootstrap import order_key
from semanticflow.optimizer.economics import decide_optimization
from semanticflow.optimizer.evidence import file_sha256
from semanticflow.optimizer.models import (
    OptimizerSettings,
    ProfilingResult,
    SplitManifest,
)
from semanticflow.optimizer.planner import CostBasedPlanner
from semanticflow.optimizer.workflow import (
    _load_json,
    _profiling_decisions_path,
    _profiling_report_path,
    execute_order_run,
    load_profile_matrix,
)
from semanticflow.providers.base import InferenceProvider
from semanticflow.query.models import QuerySpec
from semanticflow.storage.results import ResultStore, git_commit, git_worktree_dirty

BENCHMARK_SCHEMA_VERSION = "m3-strategy-benchmark-v1"
DEFAULT_HORIZONS = (1, 10, 100, 1000)
PLANNED_STRATEGIES = {
    "independent": ("independent", "independent-and-v1", "minimum_expected_objective"),
    "correlation_aware": (
        "correlation_aware",
        "correlation-aware-and-v1",
        "minimum_expected_objective",
    ),
    "correlation_aware_confidence": (
        "correlation_aware",
        "correlation-aware-and-v1",
        "confidence_aware",
    ),
}
STRATEGY_DEFINITIONS = {
    "fixed": "query-file order; no profiling",
    "independent": "Milestone 2 independence estimator on the full profile",
    "correlation_aware": "chain-rule correlation-aware estimator on the full profile",
    "correlation_aware_confidence": (
        "correlation-aware estimator; reorders only if the bootstrap savings probability "
        "reaches the configured confidence level"
    ),
    "economics_gated": (
        "decides before profiling, from a structural savings bound and a prior per-decision "
        "cost, whether profiling can pay off within the declared horizon; then reorders only "
        "if the predicted forward savings are positive"
    ),
    "adaptive": (
        "economics-gated sequential profiling simulated over the recorded profile; charged "
        "only for the profiling decisions consumed up to its stopping checkpoint"
    ),
}
HORIZON_STRATEGIES = ("economics_gated", "adaptive")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def permutation_labels(original: list[str]) -> dict[str, str]:
    """Stable short labels ``p1``..``pk!`` in ``itertools.permutations`` order."""
    return {
        order_key(order): f"p{index}"
        for index, order in enumerate(itertools.permutations(original), start=1)
    }


def plan_strategies(
    *,
    query: QuerySpec,
    optimizer: OptimizerSettings,
    profile_dir: Path,
    prior_cost_per_decision: float,
    prior_cost_source: str,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> dict[str, Any]:
    """Select an order for every strategy from profiling data only."""
    if prior_cost_per_decision <= 0:
        raise ValueError("prior cost per decision must be positive")
    if any(value <= 0 for value in horizons):
        raise ValueError("horizons must be positive")
    if optimizer.planner.objective != "estimated_cost_usd":
        raise ValueError("the strategy benchmark compares dollar cost; use estimated_cost_usd")
    profile = ProfilingResult.model_validate(_load_json(_profiling_report_path(profile_dir)))
    split = SplitManifest.model_validate(_load_json(profile_dir / "split_manifest.json"))
    profile.require_complete()
    original = [item.id for item in query.predicates]
    if profile.predicate_order != original:
        raise ValueError("profile predicate order does not match the query file order")
    matrix = load_profile_matrix(profile_dir, profile, split)
    if not matrix.complete:
        raise ValueError("strategy benchmark requires a complete profiling matrix")
    pricing = profile.pricing or {}
    input_rate = pricing.get("input_per_million_usd")
    output_rate = pricing.get("output_per_million_usd")
    if input_rate is None or output_rate is None or profile.estimated_cost_usd is None:
        raise ValueError("profile lacks pricing; dollar costs cannot be compared")
    width = len(original)
    evaluation_size = split.evaluation_size
    profiling_cost = float(profile.estimated_cost_usd)
    profiling_decisions = profile.completed_decisions
    plans: dict[str, dict[str, Any]] = {}
    for name, (estimator, version, policy) in PLANNED_STRATEGIES.items():
        settings = optimizer.model_copy(
            update={
                "planner": optimizer.planner.model_copy(
                    update={
                        "cardinality_estimator": estimator,
                        "cost_model_version": version,
                        "selection_policy": policy,
                    }
                )
            }
        )
        plan = CostBasedPlanner(settings).plan(query, profile, split, matrix=matrix)
        plans[name] = plan.artifact_dict(include_candidates=True)

    full = {"profiling_decisions": profiling_decisions, "profiling_cost_usd": profiling_cost}
    strategies: dict[str, Any] = {
        "fixed": {
            "horizon_dependent": False,
            "selected_order": original,
            "profiling_decisions": 0,
            "profiling_cost_usd": 0.0,
        }
    }
    for name in PLANNED_STRATEGIES:
        strategies[name] = {
            "horizon_dependent": False,
            "selected_order": plans[name]["selected_predicate_order"],
            **full,
        }

    corr = {
        order_key(item["predicate_order"]): item
        for item in plans["correlation_aware"]["candidates"]
    }
    corr_selected = plans["correlation_aware"]["selected_predicate_order"]
    predicted_original = corr[order_key(original)]["estimated_cost_usd"]
    predicted_selected = corr[order_key(corr_selected)]["estimated_cost_usd"]
    prior_profile_cost = profiling_decisions * prior_cost_per_decision
    prior_original_cost = evaluation_size * prior_cost_per_decision
    # Before profiling, every evaluation is assumed to cost the prior; no order can then avoid
    # more than (k - 1) evaluations per held-out record.
    prior_bound = evaluation_size * (width - 1) * prior_cost_per_decision
    gated: dict[str, Any] = {}
    adaptive: dict[str, Any] = {}
    for horizon in horizons:
        economics = optimizer.economics.model_copy(update={"expected_executions": horizon})
        before = decide_optimization(
            settings=economics,
            original_cost_per_execution=prior_original_cost,
            statistics_state="none",
            new_profiling_cost=prior_profile_cost,
            maximum_savings_per_execution=prior_bound,
            savings_basis="structural_upper_bound",
        )
        after = None
        order, decisions, cost = original, 0, 0.0
        if before["action"] == "collect_additional_statistics":
            after = decide_optimization(
                settings=economics,
                original_cost_per_execution=predicted_original,
                optimized_cost_per_execution=predicted_selected,
                statistics_state="available",
                new_profiling_cost=profiling_cost,
                sunk_profiling_cost=profiling_cost,
                savings_basis="predicted",
            )
            decisions, cost = profiling_decisions, profiling_cost
            if after["action"] in {"optimize_with_existing_statistics", "reuse_existing_plan"}:
                order = corr_selected
        gated[str(horizon)] = {
            "selected_order": order,
            "profiling_decisions": decisions,
            "profiling_cost_usd": cost,
            "before_profiling": before,
            "after_profiling": after,
        }
        simulation = simulate_adaptive_profiling(
            matrix,
            estimator_name="correlation_aware",
            original_order=original,
            objective="estimated_cost_usd",
            evaluation_record_count=evaluation_size,
            question_hashes=profile.predicate_hashes,
            settings=optimizer.adaptive,
            bootstrap=optimizer.bootstrap,
            correlation=optimizer.correlation,
            input_price_per_million=input_rate,
            output_price_per_million=output_rate,
            expected_executions=horizon,
            tie_relative_tolerance=optimizer.planner.tie_relative_tolerance,
            economics=optimizer.economics,
            savings_upper_bound_per_execution=prior_bound,
            savings_bound_basis="structural_upper_bound",
            estimated_cost_per_decision=prior_cost_per_decision,
        )
        stop = simulation["stopping_checkpoint"]
        adaptive[str(horizon)] = {
            "selected_order": simulation["selected_order"],
            "profiling_decisions": simulation["profiling_decisions_used"],
            "profiling_cost_usd": (
                stop["profiling_cost_usd_offline_estimate"] if stop is not None else 0.0
            ),
            "profiling_started": simulation["profiling_started"],
            "start_decision": simulation["start_decision"],
            "stopping_checkpoint": stop,
        }
    strategies["economics_gated"] = {"horizon_dependent": True, "by_horizon": gated}
    strategies["adaptive"] = {"horizon_dependent": True, "by_horizon": adaptive}

    selected = {order_key(original)}
    for item in strategies.values():
        entries = item["by_horizon"].values() if item["horizon_dependent"] else [item]
        selected.update(order_key(entry["selected_order"]) for entry in entries)
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "query_id": query.query_id,
        "original_order": original,
        "permutation_labels": permutation_labels(original),
        "profile_dir": profile_dir.as_posix(),
        "profile_report_sha256": file_sha256(_profiling_report_path(profile_dir)),
        "profile_decisions_sha256": file_sha256(_profiling_decisions_path(profile_dir)),
        "split_manifest_sha256": file_sha256(profile_dir / "split_manifest.json"),
        "dataset_fingerprint": split.dataset_fingerprint,
        "profiling_fingerprint": split.profiling_fingerprint,
        "evaluation_fingerprint": split.evaluation_fingerprint,
        "evaluation_size": evaluation_size,
        "model": profile.resolved_model_versions,
        "pricing": pricing,
        "profiling_decisions": profiling_decisions,
        "profiling_cost_usd": profiling_cost,
        "prior_cost_per_decision_usd": prior_cost_per_decision,
        "prior_cost_source": prior_cost_source,
        "prior_structural_savings_bound_per_execution_usd": prior_bound,
        "horizons": list(horizons),
        "held_out_outcomes_used": False,
        "strategy_definitions": STRATEGY_DEFINITIONS,
        "strategies": strategies,
        "distinct_selected_orders": sorted(selected),
        "plans": plans,
    }


def build_schedule(
    strategies: dict[str, Any],
    *,
    orders: str = "all",
    abba_blocks: int = 1,
    seed: int = 5051,
) -> dict[str, Any]:
    """Counterbalanced schedule: ABBA blocks of fixed vs comparator, then the other orders.

    The comparator is the correlation-aware selection; if that equals the fixed order, the
    independence selection, and otherwise the best-ranked other correlation-aware candidate.
    """
    if orders not in {"all", "selected"}:
        raise ValueError("orders must be 'all' or 'selected'")
    if abba_blocks < 1:
        raise ValueError("at least one ABBA block is required")
    original = list(strategies["original_order"])
    labels = strategies["permutation_labels"]
    choice = strategies["strategies"]["correlation_aware"]["selected_order"]
    source = "correlation_aware_selection"
    if choice == original:
        choice = strategies["strategies"]["independent"]["selected_order"]
        source = "independent_selection"
    if choice == original:
        ranked = sorted(
            strategies["plans"]["correlation_aware"]["candidates"],
            key=lambda item: item["rank"],
        )
        choice = next(
            item["predicate_order"] for item in ranked if item["predicate_order"] != original
        )
        source = "best_ranked_non_original_correlation_aware_candidate"
    fixed, comparator = order_key(original), order_key(choice)
    runs: list[dict[str, Any]] = []
    for block in range(abba_blocks):
        pattern = (fixed, comparator, comparator, fixed)
        if block % 2:
            pattern = (comparator, fixed, fixed, comparator)
        for key in pattern:
            role = "abba_fixed" if key == fixed else "abba_comparator"
            runs.append({"order_key": key, "role": role, "block": block + 1})
    if orders == "all":
        remaining = [key for key in labels if key not in {fixed, comparator}]
    else:
        remaining = [
            key for key in strategies["distinct_selected_orders"]
            if key not in {fixed, comparator}
        ]
    remaining = sorted(remaining, key=lambda key: labels[key])
    random.Random(seed).shuffle(remaining)
    runs.extend({"order_key": key, "role": "coverage", "block": None} for key in remaining)
    for index, run in enumerate(runs, start=1):
        run["index"] = index
        run["label"] = labels[run["order_key"]]
        run["order"] = run["order_key"].split(" -> ")
        run["run_dir"] = f"run_{index:02d}_{run['label']}"
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "query_id": strategies["query_id"],
        "design": "abba_then_shuffled_remaining_orders",
        "orders": orders,
        "abba_blocks": abba_blocks,
        "seed": seed,
        "fixed_order": original,
        "comparator_order": choice,
        "comparator_source": source,
        "runs": runs,
    }


def schedule_preflight(
    strategies: dict[str, Any],
    schedule: dict[str, Any],
    optimizer: OptimizerSettings,
    *,
    cost_per_decision: float,
) -> dict[str, Any]:
    """Offline cost and attempt bounds for a schedule; no provider is constructed."""
    candidates = {
        order_key(item["predicate_order"]): item
        for item in strategies["plans"]["correlation_aware"]["candidates"]
    }
    width = len(strategies["original_order"])
    worst_decisions = strategies["evaluation_size"] * width
    per_run_cap = optimizer.execution.max_attempts_per_run
    if per_run_cap < worst_decisions:
        raise ValueError("per-run attempt cap is below the worst-case decision count")
    runs = []
    for run in schedule["runs"]:
        candidate = candidates[run["order_key"]]
        runs.append(
            {
                "index": run["index"],
                "label": run["label"],
                "order_key": run["order_key"],
                "predicted_decisions": candidate["estimated_evaluations"],
                "predicted_cost_usd": candidate["estimated_cost_usd"],
                "worst_case_decisions": worst_decisions,
                "worst_case_cost_usd": worst_decisions * cost_per_decision,
                "max_http_attempts": per_run_cap,
            }
        )
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "network_requests_made": 0,
        "cost_per_decision_usd": cost_per_decision,
        "prediction_source": "correlation-aware candidate estimates from the profile",
        "runs": runs,
        "run_count": len(runs),
        "predicted_decisions": sum(item["predicted_decisions"] for item in runs),
        "predicted_cost_usd": sum(item["predicted_cost_usd"] for item in runs),
        "worst_case_decisions": len(runs) * worst_decisions,
        "worst_case_cost_usd": len(runs) * worst_decisions * cost_per_decision,
        "max_http_attempts": len(runs) * per_run_cap,
        "cost_if_every_allowed_attempt_were_billed_usd": (
            len(runs) * per_run_cap * cost_per_decision
        ),
    }


def _run_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report.get(key)
        for key in (
            "completion_status",
            "incomplete_reason",
            "completed_decisions",
            "provider_attempts",
            "http_attempts",
            "retry_attempts",
            "input_tokens",
            "output_tokens",
            "estimated_cost_usd",
            "final_record_count",
            "wall_time_seconds",
            "run_id",
        )
    }


async def run_session(
    *,
    query: QuerySpec,
    settings: dict[str, Any],
    optimizer: OptimizerSettings,
    provider: InferenceProvider,
    profile_dir: Path,
    schedule: dict[str, Any],
    schedule_path: Path | None,
    session_dir: Path,
    max_session_attempts: int | None = None,
) -> dict[str, Any]:
    """Run every scheduled order sequentially; stop at the first incomplete run."""
    if optimizer.execution.cache_enabled:
        raise ValueError("paired sessions require the response cache to be disabled")
    if schedule["query_id"] != query.query_id:
        raise ValueError("schedule was built for a different query")
    original = [item.id for item in query.predicates]
    if schedule["fixed_order"] != original:
        raise ValueError("schedule fixed order does not match the query file order")
    for run in schedule["runs"]:
        if sorted(run["order"]) != sorted(original):
            raise ValueError(f"scheduled order {run['order_key']} is not a permutation")
    per_run_cap = optimizer.execution.max_attempts_per_run
    planned_cap = per_run_cap * len(schedule["runs"])
    session_cap = planned_cap if max_session_attempts is None else max_session_attempts
    if session_cap <= 0:
        raise ValueError("session attempt cap must be positive")
    session_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "query_id": query.query_id,
        "profile_dir": profile_dir.as_posix(),
        "split_manifest_sha256": file_sha256(profile_dir / "split_manifest.json"),
        "schedule_path": schedule_path.as_posix() if schedule_path else None,
        "schedule_sha256": file_sha256(schedule_path) if schedule_path else None,
        "schedule": schedule,
        "provider": provider.name,
        "requested_model": provider.model_identifier,
        "cache_enabled": False,
        "concurrency": int(settings.get("executor", {}).get("concurrency", 4)),
        "pricing_configuration": settings.get("pricing", {}),
        "per_run_attempt_cap": per_run_cap,
        "session_attempt_cap": session_cap,
        "git_commit": git_commit(),
        "git_worktree_dirty": git_worktree_dirty(),
        "started_at": _now(),
        "finished_at": None,
        "status": "running",
        "attempts_used": 0,
        "runs": [],
    }
    path = session_dir / "session_manifest.json"
    ResultStore.write_json(path, manifest)
    used = 0
    for run in schedule["runs"]:
        remaining = session_cap - used
        if remaining <= 0:
            manifest["status"] = "stopped_session_attempt_cap_reached"
            break
        started = _now()
        try:
            outcome = await execute_order_run(
                order=run["order"],
                label=run["label"],
                query=query,
                settings=settings,
                optimizer=optimizer,
                provider=provider,
                experiment_dir=profile_dir,
                run_dir=session_dir / run["run_dir"],
                max_attempts=min(per_run_cap, remaining),
                metadata={
                    "session_run_index": run["index"],
                    "session_run_role": run["role"],
                    "session_dir": session_dir.as_posix(),
                },
            )
        except Exception as exc:
            manifest["runs"].append(
                {**run, "started_at": started, "finished_at": _now(), "error": repr(exc)}
            )
            manifest["status"] = "stopped_run_error"
            manifest["finished_at"] = _now()
            ResultStore.write_json(path, manifest)
            raise
        summary = _run_summary(outcome.report)
        used += int(summary["provider_attempts"] or 0)
        manifest["attempts_used"] = used
        manifest["runs"].append(
            {**run, "started_at": started, "finished_at": _now(), **summary}
        )
        ResultStore.write_json(path, manifest)
        if summary["completion_status"] != "complete":
            manifest["status"] = "stopped_incomplete_run"
            break
    else:
        manifest["status"] = "complete"
    manifest["finished_at"] = _now()
    ResultStore.write_json(path, manifest)
    return manifest
