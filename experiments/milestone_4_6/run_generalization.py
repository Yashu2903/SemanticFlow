"""Milestone 4.6 offline preflight and separately authorized live stages."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from semanticflow.dataset.generalization import token_proxy
from semanticflow.optimizer.generalization import (
    STAGE_B_AUTHORIZATION,
    canonical_m45_reproduction,
    dataset_query,
    environment_manifest,
    heldout_budget,
    historical_artifact_manifest,
    load_protocol,
    load_workload_suite,
    mock_preflight,
    profiling_budget,
    records_for_partition,
    require_paid_authorization,
    select_frozen_plans,
    sha256_file,
    strategy_freeze_manifest,
    validate_preflight_inputs,
    workload_freeze_manifest,
)
from semanticflow.optimizer.generalization_analysis import analyze_live_validation
from semanticflow.optimizer.matrix import ProfileMatrix
from semanticflow.optimizer.models import load_optimizer_settings
from semanticflow.optimizer.planner import predicate_hash
from semanticflow.optimizer.profiler import PredicateProfiler
from semanticflow.optimizer.stage_c_preflight import (
    baseline_integrity_audit,
    build_authorization_manifest,
    load_stage_b_matrix,
    per_run_cost_forecasts,
    profiling_cost_reconciliation,
    support_and_strategy_diagnostics,
    validate_schedule,
    validate_stage_c_launch,
)
from semanticflow.runtime import run_query_records
from semanticflow.storage.results import ResultStore

WORKLOADS = Path("configs/milestone_4_6/workloads.yaml")
PROTOCOL = Path("configs/milestone_4_6/protocol.yaml")
M45_MANIFEST = Path("results/milestone_4_5/calibration_20261111/frozen_config.json")
RESULTS = Path("results/milestone_4_6")


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")


def _new_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    ResultStore.write_json(path, payload)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _settings(protocol: dict[str, Any]) -> dict[str, Any]:
    provider = protocol["provider"]
    return {
        "executor": {"concurrency": provider["concurrency"]},
        "providers": {
            "jev": {
                "model": provider["model"],
                "api_url": provider["api_url"],
                "timeout_seconds": provider["timeout_seconds"],
                "max_retries": provider["max_retries"],
                "backoff_seconds": 0.5,
            }
        },
        "pricing": {
            "id": "jev-pricing-2026-11-12",
            **provider["pricing"],
        },
    }


async def _preflight(output: Path) -> int:
    suite = load_workload_suite(WORKLOADS)
    protocol = load_protocol(PROTOCOL)
    destination = _new_directory(output)
    freeze = strategy_freeze_manifest(
        m4_config=protocol["strategy_sources"]["milestone_4"],
        m45_config=protocol["strategy_sources"]["milestone_4_5_frozen"],
        m45_frozen_manifest=M45_MANIFEST,
    )
    artifacts = {
        "input_validation.json": validate_preflight_inputs(suite, protocol),
        "strategy_freeze.json": freeze,
        "workload_freeze.json": workload_freeze_manifest(suite),
        "historical_integrity.json": historical_artifact_manifest(),
        "m45_reproduction.json": canonical_m45_reproduction(),
        "environment.json": environment_manifest(),
        "stage_b_budget.json": profiling_budget(suite, protocol),
        "mock_preflight.json": await mock_preflight(suite, protocol),
    }
    for filename, payload in artifacts.items():
        _write_json(destination / filename, payload)
    _write_json(
        destination / "manifest.json",
        {
            "schema_version": "m46-offline-preflight-run-v1",
            "created_at": _timestamp(),
            "status": "complete",
            "paid_jev_calls": 0,
            "held_out_outcomes_observed": False,
            "protocol_sha256": sha256_file(PROTOCOL),
            "workloads_sha256": sha256_file(WORKLOADS),
            "artifacts": {
                filename: sha256_file(destination / filename) for filename in artifacts
            },
        },
    )
    print(destination)
    return 0


async def _profile(output: Path, authorization: str | None) -> int:
    require_paid_authorization("stage_b", authorization)
    # Importing and initializing the paid provider happens only after the fail-closed guard.
    from semanticflow.providers.jev import JevProvider

    suite = load_workload_suite(WORKLOADS)
    protocol = load_protocol(PROTOCOL)
    budget = profiling_budget(suite, protocol)
    destination = _new_directory(output)
    _write_json(destination / "authorized_budget.json", budget)
    base_optimizer = load_optimizer_settings(protocol["strategy_sources"]["milestone_4"])
    provider_config = protocol["provider"]
    provider = JevProvider.from_env(
        model=provider_config["model"],
        api_url=provider_config["api_url"],
        timeout_seconds=provider_config["timeout_seconds"],
        max_retries=provider_config["max_retries"],
    )
    status = "complete"
    try:
        for dataset_id, dataset in suite.datasets.items():
            records = records_for_partition(dataset.path, "profiling")
            per_dataset = next(
                item for item in budget["datasets"] if item["dataset_id"] == dataset_id
            )
            query = dataset_query(
                suite,
                dataset_id,
                partition="profiling",
                max_calls=per_dataset["maximum_http_attempts"],
            )
            optimizer = base_optimizer.model_copy(
                update={
                    "profiling": base_optimizer.profiling.model_copy(
                        update={
                            "concurrency": provider_config["concurrency"],
                            "max_attempts": per_dataset["maximum_http_attempts"],
                        }
                    )
                }
            )
            dataset_manifest = _read_json(
                dataset.path.with_suffix(dataset.path.suffix + ".manifest.json")
            )
            result = await PredicateProfiler(provider, optimizer).profile(
                records,
                query,
                run_id=f"m46-stage-b-{dataset_id}-{_timestamp()}",
                dataset_fingerprint=dataset_manifest["dataset_fingerprint_sha256"],
                profiling_fingerprint=dataset_manifest["profiling_fingerprint_sha256"],
                pricing=provider_config["pricing"],
            )
            dataset_dir = _new_directory(destination / dataset_id)
            _write_json(dataset_dir / "report.json", result.model_dump(mode="json"))
            ResultStore.write_jsonl(
                dataset_dir / "decisions.jsonl",
                [trace.to_dict() for trace in result.traces],
            )
            _write_json(
                dataset_dir / "manifest.json",
                {
                    "schema_version": "m46-live-profile-v1",
                    "query": query.model_dump(mode="json"),
                    "authorized_logical_decisions": per_dataset["logical_decisions"],
                    "authorized_http_attempts": per_dataset["maximum_http_attempts"],
                    "complete": result.complete,
                    "decisions_sha256": sha256_file(dataset_dir / "decisions.jsonl"),
                },
            )
            if not result.complete:
                status = "incomplete"
                break
    finally:
        await provider.aclose()
    _write_json(
        destination / "session.json",
        {
            "schema_version": "m46-stage-b-session-v1",
            "status": status,
            "actual_http_attempts": provider.stats.attempts,
            "authorized_http_attempts": budget["maximum_http_attempts"],
            "authorization_phrase": STAGE_B_AUTHORIZATION,
        },
    )
    return 0 if status == "complete" else 2


def _matrix_from_profile(
    suite: Any,
    dataset_id: str,
    profile_root: Path,
) -> ProfileMatrix:
    dataset = suite.datasets[dataset_id]
    report = _read_json(profile_root / dataset_id / "report.json")
    if not report.get("complete"):
        raise ValueError(f"incomplete profile for {dataset_id}")
    records = records_for_partition(dataset.path, "profiling")
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
        raise ValueError(f"profile resolved multiple models for {dataset_id}: {resolved}")
    matrix = ProfileMatrix.from_traces(
        _read_jsonl(profile_root / dataset_id / "decisions.jsonl"),
        predicate_ids,
        review_ids=[record.review_id for record in records],
        forbidden_review_ids=[record.review_id for record in evaluation],
        model_identifier=resolved[0],
        question_hashes=hashes,
    )
    if not matrix.complete:
        raise ValueError(f"profile matrix is incomplete for {dataset_id}")
    return matrix


def _usage_calibration(suite: Any, profile_root: Path) -> dict[str, Any]:
    datasets = {}
    for dataset_id, dataset in suite.datasets.items():
        records = {
            record.review_id: record
            for record in records_for_partition(dataset.path, "profiling")
        }
        traces = _read_jsonl(profile_root / dataset_id / "decisions.jsonl")
        predicates = {}
        for predicate_id, predicate in dataset.predicates.items():
            predicate_traces = [
                trace for trace in traces if trace["predicate_id"] == predicate_id
            ]
            if len(predicate_traces) != len(records):
                raise ValueError(f"incomplete usage evidence for {dataset_id}/{predicate_id}")
            if any(
                trace.get("input_tokens") is None or trace.get("output_tokens") is None
                for trace in predicate_traces
            ):
                raise ValueError(f"missing provider usage for {dataset_id}/{predicate_id}")
            input_ratios = [
                trace["input_tokens"]
                / token_proxy(
                    records[trace["review_id"]].review_text, predicate.instructions
                )
                for trace in predicate_traces
            ]
            output_tokens = [trace["output_tokens"] for trace in predicate_traces]
            predicates[predicate_id] = {
                "profile_decisions": len(predicate_traces),
                "mean_input_token_ratio": sum(input_ratios) / len(input_ratios),
                "maximum_input_token_ratio": max(input_ratios),
                "mean_output_tokens": sum(output_tokens) / len(output_tokens),
                "maximum_output_tokens": max(output_tokens),
            }
        datasets[dataset_id] = {"predicates": predicates}
    return {
        "schema_version": "m46-stage-b-usage-calibration-v1",
        "method": (
            "provider input tokens divided by preregistered character proxy per profiling "
            "decision; mean estimates expected usage and maximum bounds the cost proxy"
        ),
        "changes_optimizer_or_selected_plans": False,
        "datasets": datasets,
    }


def _plan(profile_root: Path, output: Path) -> int:
    suite = load_workload_suite(WORKLOADS)
    protocol = load_protocol(PROTOCOL)
    matrices = {
        dataset_id: _matrix_from_profile(suite, dataset_id, profile_root)
        for dataset_id in suite.datasets
    }
    plans = select_frozen_plans(suite, protocol, matrices)
    plans["stage_b_usage_calibration"] = _usage_calibration(suite, profile_root)
    plans["profile_artifacts"] = {
        dataset_id: {
            "report_sha256": sha256_file(profile_root / dataset_id / "report.json"),
            "decisions_sha256": sha256_file(profile_root / dataset_id / "decisions.jsonl"),
            "matrix_fingerprint_sha256": matrix.fingerprint(),
        }
        for dataset_id, matrix in matrices.items()
    }
    plans["strategy_freeze"] = strategy_freeze_manifest(
        m4_config=protocol["strategy_sources"]["milestone_4"],
        m45_config=protocol["strategy_sources"]["milestone_4_5_frozen"],
        m45_frozen_manifest=M45_MANIFEST,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    _write_json(output, plans)
    budget_path = output.with_name("stage_c_budget.json")
    _write_json(budget_path, heldout_budget(suite, protocol, plans))
    print(output)
    print(budget_path)
    return 0


def _rebudget(profile_root: Path, plans_path: Path, output: Path) -> int:
    suite = load_workload_suite(WORKLOADS)
    protocol = load_protocol(PROTOCOL)
    plans = _read_json(plans_path)
    if "stage_b_usage_calibration" in plans:
        raise ValueError("plan artifact already contains Stage B usage calibration")
    corrected = {
        **plans,
        "budget_accounting_correction": {
            "reason": (
                "Stage B provider-reported input tokens exceeded the preregistered character "
                "proxy; selected orders and strategy decisions are unchanged"
            ),
            "supersedes_plans_sha256_for_budget_only": sha256_file(plans_path),
        },
        "stage_b_usage_calibration": _usage_calibration(suite, profile_root),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    _write_json(output, corrected)
    budget_path = output.with_name("stage_c_budget.json")
    _write_json(budget_path, heldout_budget(suite, protocol, corrected))
    print(output)
    print(budget_path)
    return 0


def _analyze(
    profile_root: Path,
    plans_path: Path,
    session_root: Path,
    output: Path,
    profiling_accounting_path: Path | None = None,
) -> int:
    suite = load_workload_suite(WORKLOADS)
    protocol = load_protocol(PROTOCOL)
    report = analyze_live_validation(
        suite=suite,
        protocol=protocol,
        plans=_read_json(plans_path),
        profile_root=profile_root,
        session_root=session_root,
        profiling_cost_accounting=(
            _read_json(profiling_accounting_path) if profiling_accounting_path else None
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    _write_json(output, report)
    return 0 if report["status"] == "complete" else 2


def _stage_c_preflight_hardening(
    *,
    profile_root: Path,
    plans_path: Path,
    budget_path: Path,
    output: Path,
    bootstrap_iterations: int,
) -> int:
    """Generate Stage C readiness artifacts without importing the paid provider."""
    if not output.is_dir() or not (output / "baseline_audit.json").is_file():
        raise FileNotFoundError(
            "pre-change baseline_audit.json must exist in the non-overwriting output directory"
        )
    suite = load_workload_suite(WORKLOADS)
    protocol = load_protocol(PROTOCOL)
    plans = _read_json(plans_path)
    matrices = {
        dataset_id: load_stage_b_matrix(suite, dataset_id, profile_root)
        for dataset_id in suite.datasets
    }
    verification = baseline_integrity_audit(
        suite=suite,
        protocol=protocol,
        plans_path=plans_path,
        budget_path=budget_path,
        profile_root=profile_root,
    )
    forecasts = per_run_cost_forecasts(
        suite=suite,
        protocol=protocol,
        plans=plans,
        matrices=matrices,
        bootstrap_iterations=bootstrap_iterations,
    )
    reconciliation = profiling_cost_reconciliation(
        suite=suite,
        protocol=protocol,
        plans=plans,
        profile_root=profile_root,
    )
    diagnostics = support_and_strategy_diagnostics(
        suite=suite,
        protocol=protocol,
        plans=plans,
        matrices=matrices,
    )
    schedule = validate_schedule(suite=suite, protocol=protocol, plans=plans)
    authorization = build_authorization_manifest(
        suite=suite,
        protocol=protocol,
        plans_path=plans_path,
        budget_path=budget_path,
    )
    conservative = {
        "schema_version": "m46-stage-c-conservative-budget-v1",
        "maximum_logical_decisions": forecasts["aggregate"][
            "maximum_logical_evaluations"
        ],
        "maximum_http_attempts": forecasts["aggregate"]["allowed_http_attempts"],
        "conservative_no_retry_cost_proxy_usd": forecasts["aggregate"][
            "conservative_no_retry_cost_proxy_usd"
        ],
        "retry_inclusive_cost_proxy_usd": forecasts["aggregate"][
            "retry_inclusive_cost_proxy_usd"
        ],
        "hard_limits": ["logical_decisions", "http_attempts"],
        "not_hard_monetary_limits": [
            "conservative_no_retry_cost_proxy_usd",
            "retry_inclusive_cost_proxy_usd",
        ],
        "original_corrected_budget_preserved": True,
        "paid_calls_executed": 0,
    }
    sparse = {
        "schema_version": diagnostics["schema_version"],
        "workloads": diagnostics["workloads"],
        "held_out_outcomes_used": False,
        "paid_calls_executed": 0,
    }
    ranking = {
        "schema_version": "m46-candidate-ranking-support-v1",
        "workloads": [
            {
                key: value
                for key, value in workload.items()
                if key != "prefix_diagnostics"
            }
            for workload in diagnostics["workloads"]
        ],
        "held_out_outcomes_used": False,
    }
    disagreements = {
        "schema_version": "m46-strategy-disagreement-v1",
        "strategy_comparison_matrix": diagnostics["strategy_comparison_matrix"],
        "strategy_disagreements": diagnostics["strategy_disagreements"],
        "negative_profiling_replay_reorderings": diagnostics[
            "negative_profiling_replay_reorderings"
        ],
        "profiling_replay_is_not_held_out_evidence": True,
    }
    safety = {
        "schema_version": "m46-stage-c-safety-checks-v1",
        "authorization_manifest_sha256": None,
        "checks": {
            "exact_authorization_phrase_required": True,
            "plans_sha256_pinned": True,
            "budget_sha256_pinned": True,
            "schedule_sha256_pinned": True,
            "dataset_files_and_fingerprints_pinned": True,
            "prompt_hashes_pinned": True,
            "model_pinned": True,
            "cache_disabled": True,
            "request_ceiling": authorization["maximum_http_attempts"],
            "provider_import_deferred_until_all_offline_checks_pass": True,
            "conflicting_output_refused": True,
            "resume_policy": "fail_closed_no_paid_resume",
        },
        "negative_tests_require_no_provider_initialization": True,
        "provider_imported": False,
        "paid_calls_executed": 0,
    }
    artifacts = {
        "baseline_verification.json": verification,
        "per_run_expected_cost_forecasts.json": forecasts,
        "conservative_budget_forecasts.json": conservative,
        "profiling_cost_reconciliation.json": reconciliation,
        "sparse_data_diagnostics.json": sparse,
        "candidate_ranking_support_diagnostics.json": ranking,
        "strategy_disagreement_analysis.json": disagreements,
        "abba_schedule_validation.json": schedule,
        "stage_c_authorization_manifest.json": authorization,
    }
    for filename, payload in artifacts.items():
        path = output / filename
        if path.exists():
            raise FileExistsError(path)
        _write_json(path, payload)
    safety["authorization_manifest_sha256"] = sha256_file(
        output / "stage_c_authorization_manifest.json"
    )
    safety_path = output / "authorization_and_safety_checks.json"
    if safety_path.exists():
        raise FileExistsError(safety_path)
    _write_json(safety_path, safety)
    artifacts[safety_path.name] = safety
    summary = {
        "schema_version": "m46-stage-c-complete-preflight-summary-v1",
        "status": "offline_checks_complete_testing_pending",
        "recommendation": "READY_WITH_DOCUMENTED_LIMITATIONS",
        "plans_sha256": authorization["plans_sha256"],
        "frozen_workload_payload_sha256": authorization[
            "selected_workloads_sha256"
        ],
        "expected_cost_usd": forecasts["aggregate"]["expected_estimated_cost_usd"],
        "conservative_no_retry_cost_proxy_usd": forecasts[
            "legacy_conservative_no_retry_cost_proxy_usd"
        ],
        "retry_inclusive_cost_proxy_usd": forecasts[
            "legacy_retry_inclusive_cost_proxy_usd"
        ],
        "per_run_raw_conservative_no_retry_sum_usd": conservative[
            "conservative_no_retry_cost_proxy_usd"
        ],
        "per_run_raw_retry_inclusive_sum_usd": conservative[
            "retry_inclusive_cost_proxy_usd"
        ],
        "expected_logical_evaluations": forecasts["aggregate"][
            "expected_logical_evaluations"
        ],
        "maximum_logical_decisions": conservative["maximum_logical_decisions"],
        "maximum_http_attempts": conservative["maximum_http_attempts"],
        "schedule_runs": schedule["total_runs"],
        "unique_workload_orders": schedule["unique_workload_order_combinations"],
        "actual_stage_b_cost_usd": reconciliation["actual_experiment_expenditure"][
            "cost_usd"
        ],
        "held_out_outcomes_observed": False,
        "provider_imported": False,
        "paid_calls_executed": 0,
        "artifacts": {
            filename: sha256_file(output / filename) for filename in artifacts
        },
    }
    summary_path = output / "complete_preflight_summary.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    _write_json(summary_path, summary)
    print(output)
    return 0


async def _heldout(
    plans_path: Path,
    budget_path: Path,
    authorization_manifest_path: Path,
    output: Path,
    authorization: str | None,
) -> int:
    suite = load_workload_suite(WORKLOADS)
    protocol = load_protocol(PROTOCOL)
    launch = validate_stage_c_launch(
        suite=suite,
        protocol=protocol,
        plans_path=plans_path,
        budget_path=budget_path,
        authorization_manifest_path=authorization_manifest_path,
        authorization_phrase=authorization,
        output_path=output,
    )
    # Importing and initializing the paid provider is the final step after every offline
    # fingerprint, schedule, output, and exact-scope authorization check has passed.
    from semanticflow.providers.jev import JevProvider

    budget = _read_json(budget_path)
    destination = _new_directory(output)
    _write_json(destination / "authorized_budget.json", budget)
    _write_json(destination / "validated_launch_scope.json", launch)
    provider_config = protocol["provider"]
    provider = JevProvider.from_env(
        model=provider_config["model"],
        api_url=provider_config["api_url"],
        timeout_seconds=provider_config["timeout_seconds"],
        max_retries=provider_config["max_retries"],
    )
    settings = _settings(protocol)
    status = "complete"
    failure: BaseException | None = None
    try:
        for workload in suite.workloads:
            entry = next(
                item for item in budget["workloads"] if item["workload_id"] == workload.id
            )
            records = records_for_partition(
                suite.datasets[workload.dataset].path, "evaluation"
            )
            workload_dir = _new_directory(destination / workload.id)
            for scheduled in entry["schedule"]["runs"]:
                per_run_ceiling = (
                    len(records)
                    * len(scheduled["order"])
                    * (provider_config["max_retries"] + 1)
                )
                query = dataset_query(
                    suite,
                    workload.dataset,
                    partition="evaluation",
                    predicate_order=scheduled["order"],
                    max_calls=per_run_ceiling,
                ).model_copy(
                    update={
                        "query_id": scheduled["run_id"],
                        "workload_id": workload.id,
                    }
                )
                store = ResultStore(workload_dir)
                run_id, run_dir, timestamp = store.create_run(scheduled["run_id"])
                outcome = await run_query_records(
                    query=query,
                    records=records,
                    settings=settings,
                    provider=provider,
                    run_dir=run_dir,
                    run_id=run_id,
                    timestamp=timestamp,
                    cache_enabled=False,
                    report_metadata={
                        "m46_schedule_role": scheduled["role"],
                        "measured_live_execution": True,
                        "simulated_repetition": False,
                    },
                    manifest_metadata={
                        "m46_plans_sha256": sha256_file(plans_path),
                        "m46_schedule": scheduled,
                    },
                )
                if outcome.report["completion_status"] != "complete":
                    status = "incomplete"
                    break
                if provider.stats.attempts > budget["maximum_http_attempts"]:
                    raise RuntimeError("Stage C authorized HTTP-attempt ceiling exceeded")
            if status != "complete":
                break
    except BaseException as exc:
        status = "incomplete"
        failure = exc
    finally:
        await provider.aclose()
        _write_json(
            destination / "session.json",
            {
                "schema_version": "m46-stage-c-session-v1",
                "status": status,
                "actual_http_attempts": provider.stats.attempts,
                "authorized_http_attempts": budget["maximum_http_attempts"],
                "authorization_phrase": authorization,
                "plans_sha256": sha256_file(plans_path),
                "failure": (
                    f"{type(failure).__name__}: {failure}" if failure is not None else None
                ),
                "resume_permitted": False,
            },
        )
    if failure is not None:
        raise RuntimeError("Stage C stopped incomplete; paid resume is refused") from failure
    return 0 if status == "complete" else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument(
        "--output",
        type=Path,
        default=RESULTS / f"offline_preflight_{_timestamp()}",
    )
    profile = subparsers.add_parser("profile")
    profile.add_argument("--output", type=Path, required=True)
    profile.add_argument("--authorization")
    plan = subparsers.add_parser("plan")
    plan.add_argument("--profile-root", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    rebudget = subparsers.add_parser("rebudget")
    rebudget.add_argument("--profile-root", type=Path, required=True)
    rebudget.add_argument("--plans", type=Path, required=True)
    rebudget.add_argument("--output", type=Path, required=True)
    heldout = subparsers.add_parser("heldout")
    heldout.add_argument("--plans", type=Path, required=True)
    heldout.add_argument("--budget", type=Path, required=True)
    heldout.add_argument("--authorization-manifest", type=Path, required=True)
    heldout.add_argument("--output", type=Path, required=True)
    heldout.add_argument("--authorization")
    hardening = subparsers.add_parser("stage-c-preflight-hardening")
    hardening.add_argument("--profile-root", type=Path, required=True)
    hardening.add_argument("--plans", type=Path, required=True)
    hardening.add_argument("--budget", type=Path, required=True)
    hardening.add_argument("--output", type=Path, required=True)
    hardening.add_argument("--bootstrap-iterations", type=int, default=2000)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--profile-root", type=Path, required=True)
    analyze.add_argument("--plans", type=Path, required=True)
    analyze.add_argument("--session-root", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--profiling-accounting", type=Path)
    return parser


async def _main() -> int:
    args = _parser().parse_args()
    if args.command == "preflight":
        return await _preflight(args.output)
    if args.command == "profile":
        return await _profile(args.output, args.authorization)
    if args.command == "plan":
        return _plan(args.profile_root, args.output)
    if args.command == "rebudget":
        return _rebudget(args.profile_root, args.plans, args.output)
    if args.command == "analyze":
        return _analyze(
            args.profile_root,
            args.plans,
            args.session_root,
            args.output,
            args.profiling_accounting,
        )
    if args.command == "stage-c-preflight-hardening":
        return _stage_c_preflight_hardening(
            profile_root=args.profile_root,
            plans_path=args.plans,
            budget_path=args.budget,
            output=args.output,
            bootstrap_iterations=args.bootstrap_iterations,
        )
    return await _heldout(
        args.plans,
        args.budget,
        args.authorization_manifest,
        args.output,
        args.authorization,
    )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
