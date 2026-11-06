from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from semanticflow.dataset.loader import ReviewRecord, load_dataset
from semanticflow.optimizer.comparison import compare_execution_reports
from semanticflow.optimizer.evidence import read_jsonl
from semanticflow.optimizer.matrix import ProfileMatrix
from semanticflow.optimizer.models import (
    OptimizerSettings,
    PlanSelection,
    ProfilingResult,
    SplitManifest,
)
from semanticflow.optimizer.planner import CostBasedPlanner
from semanticflow.optimizer.profiler import PredicateProfiler
from semanticflow.optimizer.split import deterministic_split
from semanticflow.providers.base import InferenceProvider
from semanticflow.query.models import QuerySpec, load_query
from semanticflow.runtime import RunOutcome, run_query_records
from semanticflow.storage.results import (
    ResultStore,
    dataset_fingerprint,
    git_commit,
    git_worktree_dirty,
)


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _profiling_report_path(experiment_dir: Path) -> Path:
    nested = experiment_dir / "profiling" / "report.json"
    return nested if nested.is_file() else experiment_dir / "profiling_report.json"


def _candidate_plans_path(experiment_dir: Path) -> Path:
    nested = experiment_dir / "planning" / "candidate_plans.json"
    return nested if nested.is_file() else experiment_dir / "candidate_plans.json"


def _selected_plan_path(experiment_dir: Path) -> Path:
    nested = experiment_dir / "planning" / "selected_plan.json"
    return nested if nested.is_file() else experiment_dir / "selected_plan.json"


def _profiling_decisions_path(experiment_dir: Path) -> Path:
    nested = experiment_dir / "profiling" / "decisions.jsonl"
    return nested if nested.is_file() else experiment_dir / "profiling_decisions.jsonl"


def load_profile_matrix(
    experiment_dir: Path, profile: ProfilingResult, split: SplitManifest
) -> ProfileMatrix:
    """Rebuild the complete profiling matrix from recorded profiling decisions only."""
    return ProfileMatrix.from_traces(
        read_jsonl(_profiling_decisions_path(experiment_dir)),
        profile.predicate_order,
        review_ids=split.profiling_ids,
        forbidden_review_ids=split.evaluation_ids,
        model_identifier=(
            profile.resolved_model_versions[0]
            if len(profile.resolved_model_versions) == 1
            else None
        ),
        question_hashes=profile.predicate_hashes,
    )


def _needs_matrix(optimizer: OptimizerSettings) -> bool:
    return (
        optimizer.planner.cardinality_estimator != "independent"
        or optimizer.planner.selection_policy != "minimum_expected_objective"
    )


def _records_by_ids(records: list[ReviewRecord], ids: list[str]) -> list[ReviewRecord]:
    by_id = {record.review_id: record for record in records}
    missing = set(ids) - set(by_id)
    if missing:
        raise ValueError(f"Split references missing review IDs: {sorted(missing)[:3]}")
    return [by_id[review_id] for review_id in ids]


def _ordered_query(query: QuerySpec, order: list[str], max_calls: int) -> QuerySpec:
    by_id = {predicate.id: predicate for predicate in query.predicates}
    if set(order) != set(by_id):
        raise ValueError("selected order does not match query predicates")
    return query.model_copy(
        deep=True,
        update={"predicates": [by_id[item] for item in order], "max_calls": max_calls},
    )


def prepare_split(
    query: QuerySpec, optimizer: OptimizerSettings
) -> tuple[list[ReviewRecord], list[ReviewRecord], SplitManifest]:
    records = load_dataset(query.dataset_path, limit=query.limit)
    required = optimizer.split.profiling_size + optimizer.split.evaluation_size
    if len(records) < required:
        raise ValueError(f"Optimizer requires {required} usable records; found {len(records)}")
    return deterministic_split(
        records,
        profiling_size=optimizer.split.profiling_size,
        evaluation_size=optimizer.split.evaluation_size,
        seed=optimizer.split.seed,
        dataset_path=query.dataset_path,
        dataset_fingerprint=dataset_fingerprint(query.dataset_path),
        dataset_version=query.dataset_version,
    )


async def profile_experiment(
    *,
    query: QuerySpec,
    settings: dict[str, Any],
    optimizer: OptimizerSettings,
    provider: InferenceProvider,
    experiment_dir: Path,
    research_layout: bool = False,
) -> tuple[ProfilingResult, SplitManifest]:
    experiment_dir.mkdir(parents=True, exist_ok=False)
    profiling_records, _, split = prepare_split(query, optimizer)
    profile = await PredicateProfiler(provider, optimizer).profile(
        profiling_records,
        query,
        run_id=f"profile-{uuid.uuid4().hex[:12]}",
        dataset_fingerprint=split.dataset_fingerprint,
        profiling_fingerprint=split.profiling_fingerprint,
        pricing=settings.get("pricing", {}),
    )
    ResultStore.write_json(experiment_dir / "split_manifest.json", split.model_dump(mode="json"))
    artifact_dir = experiment_dir / "profiling" if research_layout else experiment_dir
    if research_layout:
        artifact_dir.mkdir()
    report_name = "report.json" if research_layout else "profiling_report.json"
    statistics_name = "statistics.json" if research_layout else "predicate_statistics.json"
    decisions_name = "decisions.jsonl" if research_layout else "profiling_decisions.jsonl"
    ResultStore.write_json(artifact_dir / report_name, profile.artifact_dict())
    ResultStore.write_json(
        artifact_dir / statistics_name,
        {
            "schema_version": profile.schema_version,
            "statistics": [item.model_dump(mode="json") for item in profile.statistics],
        },
    )
    ResultStore.write_jsonl(
        artifact_dir / decisions_name, [trace.to_dict() for trace in profile.traces]
    )
    if research_layout:
        provider_settings = settings.get("providers", {}).get(provider.name, {})
        manifest = {
            "schema_version": "m2-profile-manifest-v1",
            "experiment_id": experiment_dir.name,
            "timestamp": _timestamp(),
            "dataset_path": str(query.dataset_path),
            "dataset_fingerprint": split.dataset_fingerprint,
            "dataset_version": query.dataset_version,
            "dataset_preparation_version": query.dataset_preparation_version,
            "split_manifest": "../split_manifest.json",
            "profiling_fingerprint": split.profiling_fingerprint,
            "profiling_size": split.profiling_size,
            "sampling_seed": split.seed,
            "split_algorithm": split.split_algorithm,
            "query_id": query.query_id,
            "predicate_order": profile.predicate_order,
            "predicate_instructions": profile.predicate_instructions,
            "predicate_hashes": profile.predicate_hashes,
            "threshold": query.threshold,
            "provider": provider.name,
            "requested_model": profile.requested_model,
            "resolved_model_versions": profile.resolved_model_versions,
            "concurrency": optimizer.profiling.concurrency,
            "timeout_seconds": provider_settings.get("timeout_seconds"),
            "max_retries": provider_settings.get("max_retries"),
            "max_http_attempts": optimizer.profiling.max_attempts,
            "cache_enabled": False,
            "pricing_configuration": settings.get("pricing", {}),
            "statistics_version": profile.statistics_version,
            "completion_status": "complete" if profile.complete else "incomplete",
            "incomplete_reason": profile.incomplete_reason,
            "git_commit": git_commit(),
            "git_worktree_dirty": git_worktree_dirty(),
        }
        ResultStore.write_json(artifact_dir / "manifest.json", manifest)
    return profile, split


def plan_experiment(
    *,
    query: QuerySpec,
    optimizer: OptimizerSettings,
    experiment_dir: Path,
    provider_model: str | None = None,
) -> PlanSelection:
    profile = ProfilingResult.model_validate(_load_json(_profiling_report_path(experiment_dir)))
    split = SplitManifest.model_validate(_load_json(experiment_dir / "split_manifest.json"))
    matrix = (
        load_profile_matrix(experiment_dir, profile, split) if _needs_matrix(optimizer) else None
    )
    plan = CostBasedPlanner(optimizer).plan(
        query, profile, split, provider_model=provider_model, matrix=matrix
    )
    nested_layout = _profiling_report_path(experiment_dir).parent.name == "profiling"
    artifact_dir = experiment_dir / "planning" if nested_layout else experiment_dir
    if nested_layout:
        artifact_dir.mkdir(exist_ok=False)
    ResultStore.write_json(
        artifact_dir / "candidate_plans.json",
        {
            "schema_version": plan.schema_version,
            "objective": plan.objective,
            "candidates": [item.model_dump(mode="json") for item in plan.candidates],
        },
    )
    ResultStore.write_json(
        artifact_dir / "selected_plan.json",
        plan.artifact_dict(include_candidates=False),
    )
    if nested_layout and plan.cardinality_estimator != "independent":
        selected = plan.candidates[0]
        ResultStore.write_json(
            artifact_dir / "report.json",
            {
                **plan.artifact_dict(include_candidates=False),
                "profiling_statistics_source": "../profiling/statistics.json",
                "profiling_decisions_source": "../profiling/decisions.jsonl",
                "held_out_evaluation_outcomes_used": False,
                "selected_is_minimum_objective": plan.selected_estimated_objective
                == min(candidate.objective_value for candidate in plan.candidates),
                "selected_plan_stage_breakdown": selected.stage_estimates,
                "limitation": (
                    "joint selectivities come from one finite profiling sample; sparse "
                    "conditioning sets are smoothed and labelled low confidence"
                ),
            },
        )
    elif nested_layout:
        statistics = {item.predicate_id: item for item in profile.statistics}
        reach_probability = 1.0
        breakdown: list[dict[str, Any]] = []
        for stage, predicate_id in enumerate(plan.selected_predicate_order, start=1):
            item = statistics[predicate_id]
            evaluations = plan.evaluation_record_count * reach_probability
            breakdown.append(
                {
                    "stage": stage,
                    "predicate_id": predicate_id,
                    "estimated_input_records": evaluations,
                    "estimated_selectivity": item.estimated_selectivity,
                    "average_input_tokens": item.average_input_tokens,
                    "average_total_tokens": item.average_total_tokens,
                    "average_cost_usd": item.average_estimated_cost_usd,
                    "estimated_input_tokens": (
                        evaluations * item.average_input_tokens
                        if item.average_input_tokens is not None
                        else None
                    ),
                    "estimated_total_tokens": (
                        evaluations * item.average_total_tokens
                        if item.average_total_tokens is not None
                        else None
                    ),
                    "estimated_cost_usd": (
                        evaluations * item.average_estimated_cost_usd
                        if item.average_estimated_cost_usd is not None
                        else None
                    ),
                }
            )
            reach_probability *= item.estimated_selectivity or 0.0
        ResultStore.write_json(
            artifact_dir / "report.json",
            {
                **plan.artifact_dict(include_candidates=False),
                "profiling_statistics_source": "../profiling/statistics.json",
                "held_out_evaluation_outcomes_used": False,
                "selected_is_minimum_objective": plan.selected_estimated_objective
                == min(candidate.objective_value for candidate in plan.candidates),
                "selected_plan_stage_breakdown": breakdown,
                "limitation": "unconditional selectivity assumes predicate independence",
            },
        )
    return plan


async def execute_experiment_strategy(
    *,
    strategy: str,
    query: QuerySpec,
    settings: dict[str, Any],
    optimizer: OptimizerSettings,
    provider: InferenceProvider,
    experiment_dir: Path,
    split: SplitManifest | None = None,
    plan: PlanSelection | None = None,
) -> RunOutcome:
    if strategy not in {"fixed", "optimized"}:
        raise ValueError("strategy must be fixed or optimized")
    split = split or SplitManifest.model_validate(
        _load_json(experiment_dir / "split_manifest.json")
    )
    profile = ProfilingResult.model_validate(_load_json(_profiling_report_path(experiment_dir)))
    CostBasedPlanner(optimizer).validate_compatibility(
        query, profile, split, provider_model=provider.model_identifier
    )
    if plan is None:
        selected_payload = _load_json(_selected_plan_path(experiment_dir))
        candidates_payload = _load_json(_candidate_plans_path(experiment_dir))
        selected_payload["candidates"] = candidates_payload["candidates"]
        plan = PlanSelection.model_validate(selected_payload)
    if (
        plan.dataset_fingerprint != split.dataset_fingerprint
        or plan.profiling_fingerprint != split.profiling_fingerprint
        or plan.predicate_hashes != profile.predicate_hashes
    ):
        raise ValueError("selected plan is incompatible with profiling artifacts")
    order = plan.original_predicate_order if strategy == "fixed" else plan.selected_predicate_order
    return await execute_order_run(
        order=order,
        label=strategy,
        query=query,
        settings=settings,
        optimizer=optimizer,
        provider=provider,
        experiment_dir=experiment_dir,
        run_dir=experiment_dir / strategy,
        split=split,
        run_id_prefix="m2",
        metadata={
            "original_predicate_order": plan.original_predicate_order,
            "selected_predicate_order": plan.selected_predicate_order,
            "optimization_objective": plan.objective,
            "original_estimated_objective": plan.original_estimated_objective,
            "selected_estimated_objective": plan.selected_estimated_objective,
            "cost_model_version": plan.cost_model_version,
            "optimizer_version": plan.optimizer_version,
        },
    )


async def execute_order_run(
    *,
    order: list[str],
    label: str,
    query: QuerySpec,
    settings: dict[str, Any],
    optimizer: OptimizerSettings,
    provider: InferenceProvider,
    experiment_dir: Path,
    run_dir: Path,
    split: SplitManifest | None = None,
    max_attempts: int | None = None,
    run_id_prefix: str = "m3",
    metadata: dict[str, Any] | None = None,
) -> RunOutcome:
    """Execute one predicate order on the held-out split of a profiled experiment.

    ``experiment_dir`` is only read (split and profile); results go to ``run_dir``.
    """
    split = split or SplitManifest.model_validate(
        _load_json(experiment_dir / "split_manifest.json")
    )
    profile = ProfilingResult.model_validate(_load_json(_profiling_report_path(experiment_dir)))
    CostBasedPlanner(optimizer).validate_compatibility(
        query, profile, split, provider_model=provider.model_identifier
    )
    cap = optimizer.execution.max_attempts_per_run
    if max_attempts is not None:
        if not 0 < max_attempts <= cap:
            raise ValueError(f"run attempt cap must be in 1..{cap}")
        cap = max_attempts
    records = _records_by_ids(load_dataset(query.dataset_path), split.evaluation_ids)
    execution_query = _ordered_query(query, list(order), cap)
    run_metadata = {
        "execution_strategy": label,
        "executed_predicate_order": list(order),
        "original_predicate_order": [item.id for item in query.predicates],
        "profiling_fingerprint": split.profiling_fingerprint,
        "evaluation_fingerprint": split.evaluation_fingerprint,
        "evaluation_record_ids": split.evaluation_ids,
        **(metadata or {}),
    }
    return await run_query_records(
        query=execution_query,
        records=records,
        settings=settings,
        provider=provider,
        run_dir=run_dir,
        run_id=f"{run_id_prefix}-{label}-{uuid.uuid4().hex[:12]}",
        timestamp=_timestamp(),
        cache_enabled=optimizer.execution.cache_enabled,
        report_metadata=run_metadata,
        manifest_metadata=run_metadata,
    )


def compare_experiment(experiment_dir: Path) -> dict[str, Any]:
    profile = ProfilingResult.model_validate(_load_json(_profiling_report_path(experiment_dir)))
    selected_payload = _load_json(_selected_plan_path(experiment_dir))
    selected_payload["candidates"] = _load_json(_candidate_plans_path(experiment_dir))[
        "candidates"
    ]
    plan = PlanSelection.model_validate(selected_payload)
    fixed = _load_json(experiment_dir / "fixed" / "report.json")
    optimized = _load_json(experiment_dir / "optimized" / "report.json")
    fixed_manifest = _load_json(experiment_dir / "fixed" / "manifest.json")
    optimized_manifest = _load_json(experiment_dir / "optimized" / "manifest.json")
    if fixed_manifest.get("evaluation_record_ids") != optimized_manifest.get(
        "evaluation_record_ids"
    ):
        raise ValueError("fixed and optimized runs do not use identical held-out IDs")
    if fixed_manifest.get("evaluation_fingerprint") != optimized_manifest.get(
        "evaluation_fingerprint"
    ):
        raise ValueError("fixed and optimized evaluation fingerprints differ")

    def match_ids(path: Path) -> set[str]:
        return {
            json.loads(line)["review_id"]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    comparison = compare_execution_reports(
        fixed,
        optimized,
        profile,
        plan,
        fixed_match_ids=match_ids(experiment_dir / "fixed" / "matches.jsonl"),
        optimized_match_ids=match_ids(experiment_dir / "optimized" / "matches.jsonl"),
    )
    comparison["evaluation_fingerprint"] = fixed_manifest["evaluation_fingerprint"]
    comparison["identical_evaluation_record_ids"] = True
    ResultStore.write_json(experiment_dir / "comparison.json", comparison)
    return comparison


def _summary(profile: ProfilingResult, plan: PlanSelection, comparison: dict[str, Any]) -> str:
    stats = "\n".join(
        f"| {item.predicate_id} | {item.completed_decisions} | "
        f"{item.estimated_selectivity:.4f} | {item.average_total_tokens:.2f} |"
        for item in profile.statistics
        if item.estimated_selectivity is not None and item.average_total_tokens is not None
    )
    return f"""# Milestone 2 Offline Experiment

This is a deterministic mock-provider infrastructure experiment on genuine SemCEB
reviews. It is not a Jev quality or cost result.

## Plan

- Original order: `{" -> ".join(plan.original_predicate_order)}`
- Selected order: `{" -> ".join(plan.selected_predicate_order)}`
- Objective: `{plan.objective}`
- Predicted reduction: `{plan.estimated_reduction_percent}` percent

## Unconditional profiling statistics

| Predicate | Decisions | Selectivity | Average total tokens |
|---|---:|---:|---:|
{stats}

## Held-out comparison

- Fixed decisions: {comparison["fixed"]["completed_decisions"]}
- Optimized decisions: {comparison["optimized"]["completed_decisions"]}
- Request reduction: {comparison["request_reduction_percent"]} percent
- Token reduction: {comparison["token_reduction_percent"]} percent
- Equivalent final match IDs: {comparison["output_equivalence"]["equivalent"]}
- Profiling decisions: {comparison["profiling_overhead"]["requests"]}
- Break-even executions: {comparison["break_even_execution_count"]}

The analytical model assumes unconditional, independent predicate selectivities.
Correlated semantic predicates can make its predictions inaccurate. Mock token usage
is synthetic and exists only to test the planner pipeline.
"""


async def run_offline_experiment(
    *,
    query_path: Path,
    settings: dict[str, Any],
    optimizer: OptimizerSettings,
    provider: InferenceProvider,
    output_root: Path,
    experiment_id: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    query = load_query(query_path)
    experiment_id = experiment_id or f"{query.query_id}_{_timestamp()}_{uuid.uuid4().hex[:8]}"
    experiment_dir = output_root / experiment_id
    profile, split = await profile_experiment(
        query=query,
        settings=settings,
        optimizer=optimizer,
        provider=provider,
        experiment_dir=experiment_dir,
    )
    profile.require_complete()
    plan = plan_experiment(
        query=query,
        optimizer=optimizer,
        experiment_dir=experiment_dir,
        provider_model=provider.model_identifier,
    )
    fixed = await execute_experiment_strategy(
        strategy="fixed",
        query=query,
        settings=settings,
        optimizer=optimizer,
        provider=provider,
        experiment_dir=experiment_dir,
        split=split,
        plan=plan,
    )
    optimized = await execute_experiment_strategy(
        strategy="optimized",
        query=query,
        settings=settings,
        optimizer=optimizer,
        provider=provider,
        experiment_dir=experiment_dir,
        split=split,
        plan=plan,
    )
    if (
        fixed.report["completion_status"] != "complete"
        or optimized.report["completion_status"] != "complete"
    ):
        raise RuntimeError("held-out execution did not complete")
    comparison = compare_experiment(experiment_dir)
    manifest = {
        "schema_version": "m2-experiment-v1",
        "experiment_id": experiment_id,
        "timestamp": _timestamp(),
        "query_path": str(query_path),
        "query_configuration": query.model_dump(mode="json"),
        "dataset_path": str(query.dataset_path),
        "dataset_version": query.dataset_version,
        "dataset_preparation_version": query.dataset_preparation_version,
        "dataset_fingerprint": split.dataset_fingerprint,
        "profiling_fingerprint": split.profiling_fingerprint,
        "evaluation_fingerprint": split.evaluation_fingerprint,
        "sampling_seed": split.seed,
        "profiling_size": split.profiling_size,
        "evaluation_size": split.evaluation_size,
        "provider": provider.name,
        "requested_model": provider.model_identifier,
        "resolved_model_versions": profile.resolved_model_versions,
        "threshold": query.threshold,
        "predicate_instructions": profile.predicate_instructions,
        "predicate_hashes": profile.predicate_hashes,
        "statistics_version": profile.statistics_version,
        "cost_model_version": plan.cost_model_version,
        "optimizer_version": plan.optimizer_version,
        "optimization_objective": plan.objective,
        "original_predicate_order": plan.original_predicate_order,
        "selected_predicate_order": plan.selected_predicate_order,
        "profiling_request_budget": optimizer.profiling.max_attempts,
        "execution_request_budget_per_run": optimizer.execution.max_attempts_per_run,
        "concurrency": int(settings.get("executor", {}).get("concurrency", 4)),
        "cache_enabled": optimizer.execution.cache_enabled,
        "pricing_configuration": settings.get("pricing", {}),
        "git_commit": git_commit(),
        "git_worktree_dirty": git_worktree_dirty(),
        "mock_token_usage_is_synthetic": provider.name == "mock",
    }
    ResultStore.write_json(experiment_dir / "experiment_manifest.json", manifest)
    (experiment_dir / "summary.md").write_text(
        _summary(profile, plan, comparison), encoding="utf-8"
    )
    return experiment_dir, comparison
