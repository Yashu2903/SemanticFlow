from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from semanticflow.executor.fused import FusedExecutor
from semanticflow.optimizer.fusion import FusedOperator, ProviderFusionCapabilities
from semanticflow.optimizer.fusion_analysis import (
    existing_adaptive_order_from_reference,
    read_jsonl,
    summarize_measured_candidates,
)
from semanticflow.optimizer.fusion_experiment import (
    build_authorization_manifest,
    build_pilot_budget,
    load_fusion_protocol,
    load_fusion_suite,
    load_yaml,
    offline_audit,
    offline_preflight,
    records_for_partition,
    sha256_file,
    suite_predicates,
    validate_stage_b_launch,
)
from semanticflow.optimizer.fusion_planner import (
    FusionPhysicalPlan,
    enumerate_fusion_plans,
    full_fusion_plan,
    naive_adjacent_partial_plan,
    select_cost_only_plan,
    select_profiling_aware_plan,
    select_quality_constrained_plan,
    separate_plan,
)
from semanticflow.optimizer.fusion_quality import FusionQualityGate
from semanticflow.storage.results import ResultStore

DEFAULT_WORKLOADS = "configs/milestone_5/workloads.yaml"
DEFAULT_PROTOCOL = "configs/milestone_5/protocol.yaml"
DEFAULT_FUSION = "configs/milestone_5/fusion.yaml"
DEFAULT_PILOT_GROUPS = "configs/milestone_5/pilot_groups.yaml"


def _write_new_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    ResultStore.write_json(target, payload)


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SemanticFlow Milestone 5 fusion harness")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "prepare", "preflight", "simulate", "authorization-proposal"):
        command = subparsers.add_parser(name)
        command.add_argument("--workloads", default=DEFAULT_WORKLOADS)
        command.add_argument("--protocol", default=DEFAULT_PROTOCOL)
        command.add_argument("--fusion", default=DEFAULT_FUSION)
        command.add_argument("--pilot-groups", default=DEFAULT_PILOT_GROUPS)
        command.add_argument("--output", required=True)
    pilot = subparsers.add_parser("pilot-profile")
    pilot.add_argument("--workloads", default=DEFAULT_WORKLOADS)
    pilot.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    pilot.add_argument("--fusion", default=DEFAULT_FUSION)
    pilot.add_argument("--pilot-groups", default=DEFAULT_PILOT_GROUPS)
    pilot.add_argument("--authorization-manifest", required=True)
    pilot.add_argument("--authorization")
    pilot.add_argument("--output", required=True)
    freeze = subparsers.add_parser("freeze-plans")
    freeze.add_argument("--pilot-root", required=True)
    freeze.add_argument("--workloads", default=DEFAULT_WORKLOADS)
    freeze.add_argument("--fusion", default=DEFAULT_FUSION)
    freeze.add_argument("--output", required=True)
    heldout = subparsers.add_parser("heldout")
    heldout.add_argument("--plans", required=True)
    heldout.add_argument("--workloads", default=DEFAULT_WORKLOADS)
    heldout.add_argument("--fusion", default=DEFAULT_FUSION)
    heldout.add_argument("--authorization-manifest", required=True)
    heldout.add_argument("--authorization")
    heldout.add_argument("--output", required=True)
    stage_d = subparsers.add_parser("stage-d-proposal")
    stage_d.add_argument("--plans", required=True)
    stage_d.add_argument("--workloads", default=DEFAULT_WORKLOADS)
    stage_d.add_argument("--fusion", default=DEFAULT_FUSION)
    stage_d.add_argument("--output", required=True)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--session-root", required=True)
    analyze.add_argument("--output", required=True)
    return parser


def _offline_inputs(
    arguments: argparse.Namespace,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    suite = load_fusion_suite(arguments.workloads)
    protocol = load_fusion_protocol(arguments.protocol)
    fusion = load_yaml(arguments.fusion)
    pilot_groups = load_yaml(arguments.pilot_groups)
    return suite, protocol, fusion, pilot_groups


async def _run_preflight(arguments: argparse.Namespace) -> dict[str, Any]:
    suite, protocol, fusion, pilot_groups = _offline_inputs(arguments)
    budget = build_pilot_budget(suite, protocol, fusion, pilot_groups)
    return {
        "schema_version": "m5-stage-a-complete-preflight-v1",
        "audit": offline_audit(
            suite=suite,
            protocol_path=arguments.protocol,
            fusion_config_path=arguments.fusion,
        ),
        "mock_preflight": await offline_preflight(suite, fusion),
        "pilot_budget": budget,
        "stage_b_authorization_manifest": build_authorization_manifest(
            budget=budget,
            suite_path=arguments.workloads,
            protocol_path=arguments.protocol,
            fusion_config_path=arguments.fusion,
            pilot_groups_path=arguments.pilot_groups,
        ),
        "live_jev_calls": 0,
    }


def _freeze_from_pilot(arguments: argparse.Namespace) -> dict[str, Any]:
    root = Path(arguments.pilot_root)
    session = json.loads((root / "session.json").read_text("utf-8"))
    if session.get("status") != "complete":
        raise RuntimeError("cannot freeze plans from an incomplete pilot")
    suite = load_fusion_suite(arguments.workloads)
    fusion = load_yaml(arguments.fusion)
    provider = fusion["provider"]
    capabilities = ProviderFusionCapabilities(
        provider="jev",
        conjunctive_binary=True,
        multi_decision=True,
        evidence="TypeSafe OpenAPI contract plus completed Stage B pilot",
        live_validated=True,
    )
    frozen: dict[str, Any] = {}
    for workload in suite.workloads:
        dataset_root = root / workload.dataset
        reference_rows = read_jsonl(dataset_root / "separate_reference" / "decisions.jsonl")
        record_ids = sorted({str(row["review_id"]) for row in reference_rows})
        separate = {
            (str(row["review_id"]), str(row["predicate_id"])): row
            for row in reference_rows
        }
        fused: dict[tuple[str, str], dict[str, Any]] = {}
        profiling_cost_by_group: dict[str, float] = {}
        for operator_path in dataset_root.glob("conjunctive_binary_*/operator.json"):
            operator = FusedOperator.model_validate_json(operator_path.read_text("utf-8"))
            rows = read_jsonl(operator_path.parent / "decisions.jsonl")
            profiling_cost_by_group[operator.group_id] = sum(
                float(row.get("input_tokens") or 0)
                * float(provider["input_price_per_million_usd"])
                / 1_000_000
                for row in rows
            )
            fused.update(
                {
                    (str(row["review_id"]), operator.group_id): row
                    for row in rows
                }
            )
        predicates = suite_predicates(suite, workload.dataset, workload.predicate_ids)
        candidates = enumerate_fusion_plans(
            predicates,
            workload_id=workload.id,
            logical_query_id=workload.id,
            requested_model=provider["model"],
            threshold=suite.threshold,
            fusion_mode="conjunctive_binary",
            capabilities=capabilities,
            configuration_fingerprint=sha256_file(arguments.fusion),
            maximum_group_size=int(fusion["maximum_group_size"]),
        )
        adaptive = existing_adaptive_order_from_reference(
            traces=reference_rows,
            predicate_ids=workload.predicate_ids,
            original_order=workload.original_order,
            review_ids=record_ids,
            evaluation_record_count=100,
            expected_executions=int(fusion["reuse_horizon"]),
        )
        fallback = separate_plan(candidates, adaptive["selected_order"])
        replays = summarize_measured_candidates(
            candidates,
            record_ids=record_ids,
            separate_traces=separate,
            fused_traces=fused,
            quality_gate=FusionQualityGate.model_validate(fusion["quality_gate"]),
            input_price_per_million=float(provider["input_price_per_million_usd"]),
            output_price_per_million=float(provider["output_price_per_million_usd"]),
        )
        replay_by_id = {row["plan_id"]: row for row in replays}
        costs = {
            plan.plan_id: replay_by_id[plan.plan_id]["estimated_cost_usd"]
            for plan in candidates
        }
        quality = {
            plan.plan_id: bool(replay_by_id[plan.plan_id]["quality_gate"]["eligible"])
            for plan in candidates
        }
        profile_costs = {
            plan.plan_id: sum(
                profiling_cost_by_group.get(group.group_id, 0.0)
                for group in plan.groups
                if len(group.predicate_ids) > 1
            )
            for plan in candidates
        }
        selections = {
            "fixed_separate": {
                "selected_plan_id": separate_plan(
                    candidates, workload.original_order
                ).plan_id
            },
            "existing_adaptive_separate": {
                "selected_plan_id": fallback.plan_id,
                "selected_order": adaptive["selected_order"],
                "controller": adaptive,
            },
            "full_fusion": {"selected_plan_id": full_fusion_plan(candidates).plan_id},
            "naive_adjacent_partial_fusion": {
                "selected_plan_id": naive_adjacent_partial_plan(
                    candidates, workload.original_order
                ).plan_id
            },
            "cost_only_joint": select_cost_only_plan(
                candidates, costs, fallback=fallback
            ).model_dump(mode="json"),
            "quality_constrained_joint": select_quality_constrained_plan(
                candidates, costs, quality, fallback=fallback
            ).model_dump(mode="json"),
            "profiling_aware_joint": select_profiling_aware_plan(
                candidates,
                costs,
                profile_costs,
                quality,
                fallback=fallback,
                reuse_horizon=int(fusion["reuse_horizon"]),
            ).model_dump(mode="json"),
        }
        frozen[workload.id] = {
            "dataset": workload.dataset,
            "fallback_strategy": "existing_adaptive",
            "adaptive_baseline": adaptive,
            "candidates": replays,
            "selections": selections,
            "plans": {plan.plan_id: plan.model_dump(mode="json") for plan in candidates},
            "held_out_outcomes_observed": False,
        }
    return {
        "schema_version": "m5-frozen-plans-v1",
        "pilot_session_sha256": sha256_file(root / "session.json"),
        "workloads_sha256": sha256_file(arguments.workloads),
        "fusion_config_sha256": sha256_file(arguments.fusion),
        "analysis_source_sha256": sha256_file(
            "src/semanticflow/optimizer/fusion_analysis.py"
        ),
        "freeze_runner_sha256": sha256_file("experiments/milestone_5/run_fusion.py"),
        "quality_reference": "separate LLM conjunction; not human truth",
        "held_out_outcomes_observed": False,
        "post_selection_retuning_permitted": False,
        "workloads": frozen,
    }


def _selected_plan_id(selection: dict[str, Any]) -> str:
    return str(selection["selected_plan_id"])


def _build_stage_d_proposal(arguments: argparse.Namespace) -> dict[str, Any]:
    plans_path = Path(arguments.plans)
    frozen = json.loads(plans_path.read_text("utf-8"))
    if frozen.get("schema_version") != "m5-frozen-plans-v1":
        raise ValueError("Stage D requires a Milestone 5 frozen plan artifact")
    if frozen.get("held_out_outcomes_observed") is not False:
        raise ValueError("frozen plan artifact is not pre-held-out")
    suite = load_fusion_suite(arguments.workloads)
    fusion = load_yaml(arguments.fusion)
    schedule: list[dict[str, Any]] = []
    maximum_logical = 0
    expected_cost = 0.0
    for workload in suite.workloads:
        item = frozen["workloads"][workload.id]
        plan_ids = sorted(
            {_selected_plan_id(selection) for selection in item["selections"].values()}
        )
        replay_by_id = {row["plan_id"]: row for row in item["candidates"]}
        for plan_id in plan_ids:
            plan = FusionPhysicalPlan.model_validate(item["plans"][plan_id])
            maximum = 400 * len(plan.groups)
            maximum_logical += maximum
            pilot = replay_by_id[plan_id]
            if pilot["estimated_cost_usd"] is not None:
                expected_cost += float(pilot["estimated_cost_usd"]) * 10
            schedule.append(
                {
                    "run_id": f"m5-stage-d-{workload.id}-{plan_id}",
                    "workload_id": workload.id,
                    "dataset": workload.dataset,
                    "plan_id": plan_id,
                    "physical_groups": [list(group) for group in plan.canonical_grouping],
                    "evaluation_records": 400,
                    "maximum_logical_requests": maximum,
                    "strategies": sorted(
                        name
                        for name, selection in item["selections"].items()
                        if _selected_plan_id(selection) == plan_id
                    ),
                }
            )
    attempts = maximum_logical * (int(fusion["provider"]["max_retries"]) + 1)
    return {
        "schema_version": "m5-stage-d-authorization-manifest-v1",
        "stage": "stage_d_held_out",
        "authorization_phrase": "AUTHORIZE M5 STAGE D HELD-OUT FUSION EVALUATION",
        "paid_calls_authorized": False,
        "plans_sha256": sha256_file(plans_path),
        "workloads_sha256": sha256_file(arguments.workloads),
        "fusion_config_sha256": sha256_file(arguments.fusion),
        "dataset_sha256": {
            dataset_id: sha256_file(dataset.path)
            for dataset_id, dataset in suite.datasets.items()
        },
        "schedule": schedule,
        "schedule_sha256": sha256_file(plans_path)
        + ":"
        + str(len(schedule)),
        "unique_workload_plan_runs": len(schedule),
        "maximum_logical_request_ceiling": maximum_logical,
        "http_attempt_ceiling": attempts,
        "expected_cost_usd_from_pilot_scale": expected_cost,
        "expected_cost_is_not_a_hard_cap": True,
        "model": fusion["provider"]["model"],
    }


def _validate_stage_d(arguments: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    if arguments.authorization != "AUTHORIZE M5 STAGE D HELD-OUT FUSION EVALUATION":
        raise PermissionError("Stage D requires its exact separately approved phrase")
    output = Path(arguments.output)
    if output.exists():
        raise FileExistsError(f"held-out output already exists: {output}")
    manifest = json.loads(Path(arguments.authorization_manifest).read_text("utf-8"))
    if manifest.get("stage") != "stage_d_held_out":
        raise ValueError("wrong Stage D authorization manifest")
    if manifest["plans_sha256"] != sha256_file(arguments.plans):
        raise ValueError("frozen plan fingerprint mismatch")
    if manifest["workloads_sha256"] != sha256_file(arguments.workloads):
        raise ValueError("workload fingerprint mismatch")
    if manifest["fusion_config_sha256"] != sha256_file(arguments.fusion):
        raise ValueError("fusion configuration fingerprint mismatch")
    proposal = _build_stage_d_proposal(arguments)
    for key in (
        "schedule",
        "maximum_logical_request_ceiling",
        "http_attempt_ceiling",
        "dataset_sha256",
        "model",
    ):
        if manifest[key] != proposal[key]:
            raise ValueError(f"Stage D manifest mismatch: {key}")
    return manifest, json.loads(Path(arguments.plans).read_text("utf-8"))


async def _run_heldout(arguments: argparse.Namespace) -> int:
    manifest, frozen = _validate_stage_d(arguments)
    suite = load_fusion_suite(arguments.workloads)
    fusion = load_yaml(arguments.fusion)
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=False)
    session_path = output / "session.json"
    session = {
        "schema_version": "m5-stage-d-session-v1",
        "status": "running",
        "authorization_manifest_sha256": sha256_file(arguments.authorization_manifest),
        "actual_http_attempts": 0,
        "http_attempt_ceiling": manifest["http_attempt_ceiling"],
        "failure": None,
    }
    ResultStore.write_json(session_path, session)
    provider = None
    try:
        from semanticflow.providers.jev import JevProvider

        config = fusion["provider"]
        provider = JevProvider.from_env(
            config["model"],
            api_url=config["api_url"],
            timeout_seconds=float(config["timeout_seconds"]),
            max_retries=int(config["max_retries"]),
        )
        for scheduled in manifest["schedule"]:
            workload = next(
                item for item in suite.workloads if item.id == scheduled["workload_id"]
            )
            plan_payload = frozen["workloads"][workload.id]["plans"][
                scheduled["plan_id"]
            ]
            plan = FusionPhysicalPlan.model_validate(plan_payload)
            records = records_for_partition(
                suite.datasets[workload.dataset].path, "evaluation"
            )
            remaining = manifest["http_attempt_ceiling"] - provider.stats.attempts
            if remaining <= 0:
                raise RuntimeError("Stage D HTTP-attempt ceiling exhausted")
            result = await FusedExecutor(
                provider, concurrency=int(config["concurrency"])
            ).execute(
                records,
                plan,
                run_id=scheduled["run_id"],
                maximum_http_attempts=remaining,
            )
            run_dir = output / scheduled["run_id"]
            run_dir.mkdir(parents=True, exist_ok=False)
            ResultStore.write_jsonl(
                run_dir / "decisions.jsonl", [trace.to_dict() for trace in result.traces]
            )
            ResultStore.write_json(
                run_dir / "report.json",
                {
                    **scheduled,
                    "complete": result.complete,
                    "incomplete_reason": result.incomplete_reason,
                    "provider_attempts": result.provider_attempts,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "wall_time_seconds": result.wall_time_seconds,
                    "group_metrics": [group.to_dict() for group in result.groups],
                    "matching_record_ids": [
                        record.review_id for record in result.matching_records
                    ],
                    "errors": result.errors,
                },
            )
            if not result.complete:
                raise RuntimeError(f"incomplete held-out run: {scheduled['run_id']}")
        session["actual_http_attempts"] = provider.stats.attempts
        if provider.stats.attempts > manifest["http_attempt_ceiling"]:
            raise RuntimeError("Stage D HTTP-attempt ceiling exceeded")
        session["status"] = "complete"
        ResultStore.write_json(session_path, session)
        return 0
    except BaseException as exc:
        session["status"] = "incomplete"
        session["failure"] = f"{type(exc).__name__}: {exc}"
        if provider is not None:
            session["actual_http_attempts"] = provider.stats.attempts
        ResultStore.write_json(session_path, session)
        raise
    finally:
        if provider is not None:
            await provider.aclose()


def _analyze_stage_d(root: Path) -> dict[str, Any]:
    session = json.loads((root / "session.json").read_text("utf-8"))
    if session.get("status") != "complete":
        raise RuntimeError("incomplete sessions cannot produce performance claims")
    reports = [
        json.loads(path.read_text("utf-8"))
        for path in root.glob("m5-stage-d-*/report.json")
    ]
    if not reports or any(not report.get("complete") for report in reports):
        raise RuntimeError("analysis requires every scheduled run to be complete")
    by_workload: dict[str, list[dict[str, Any]]] = {}
    for report in reports:
        report["logical_group_evaluations"] = sum(
            group["requested_group_evaluations"] for group in report["group_metrics"]
        )
        report["summed_inference_latency_seconds"] = sum(
            group["summed_inference_latency_seconds"] for group in report["group_metrics"]
        )
        report["cost_usd"] = float(report["input_tokens"]) * 0.042 / 1_000_000
        by_workload.setdefault(report["workload_id"], []).append(report)
    strategy_rows: dict[str, list[dict[str, Any]]] = {}
    workload_results: dict[str, Any] = {}
    for workload_id, candidates in by_workload.items():
        cheapest = min(candidates, key=lambda row: (row["cost_usd"], row["plan_id"]))
        fixed = next(
            row for row in candidates if "fixed_separate" in row["strategies"]
        )
        fixed_matches = set(fixed["matching_record_ids"])
        details = []
        for row in candidates:
            matches = set(row["matching_record_ids"])
            union = fixed_matches | matches
            jaccard = len(fixed_matches & matches) / len(union) if union else 1.0
            item = {
                "plan_id": row["plan_id"],
                "strategies": row["strategies"],
                "provider_attempts": row["provider_attempts"],
                "logical_group_evaluations": row["logical_group_evaluations"],
                "input_tokens": row["input_tokens"],
                "output_tokens": row["output_tokens"],
                "cost_usd": row["cost_usd"],
                "summed_inference_latency_seconds": row[
                    "summed_inference_latency_seconds"
                ],
                "wall_time_seconds": row["wall_time_seconds"],
                "regret_vs_cheapest_measured_usd": row["cost_usd"]
                - cheapest["cost_usd"],
                "saving_vs_fixed_usd": fixed["cost_usd"] - row["cost_usd"],
                "fixed_match_set_equal": matches == fixed_matches,
                "fixed_match_set_jaccard": jaccard,
                "semantic_accuracy_established": False,
            }
            details.append(item)
            for strategy in row["strategies"]:
                strategy_rows.setdefault(strategy, []).append(
                    {"workload_id": workload_id, **item}
                )
        workload_results[workload_id] = {
            "cheapest_measured_valid_plan_id": cheapest["plan_id"],
            "global_optimum_claimed": False,
            "plans": details,
        }
    strategy_summary = {
        strategy: {
            "workloads": len(rows),
            "total_cost_usd": sum(row["cost_usd"] for row in rows),
            "total_saving_vs_fixed_usd": sum(
                row["saving_vs_fixed_usd"] for row in rows
            ),
            "total_selection_regret_usd": sum(
                row["regret_vs_cheapest_measured_usd"] for row in rows
            ),
            "workload_wins_vs_fixed": sum(
                row["saving_vs_fixed_usd"] > 0 for row in rows
            ),
            "workload_losses_vs_fixed": sum(
                row["saving_vs_fixed_usd"] < 0 for row in rows
            ),
        }
        for strategy, rows in strategy_rows.items()
    }
    return {
        "schema_version": "m5-stage-d-analysis-v1",
        "observed_runs": len(reports),
        "actual_http_attempts": session["actual_http_attempts"],
        "workloads": workload_results,
        "strategies": strategy_summary,
        "quality_reference": "fixed separate Jev result set; not human truth",
        "projected_reuse_horizons": [1, 5, 10, 25, 50, 100, 250, 1000],
        "profiling_economics_status": (
            "join with frozen pilot artifact for standalone/shared/warm/cold accounting"
        ),
    }


async def _run_pilot(arguments: argparse.Namespace) -> int:
    # This validation must finish before the paid-provider module is imported.
    manifest = validate_stage_b_launch(
        manifest_path=arguments.authorization_manifest,
        authorization=arguments.authorization,
        output=arguments.output,
        suite_path=arguments.workloads,
        protocol_path=arguments.protocol,
        fusion_config_path=arguments.fusion,
        pilot_groups_path=arguments.pilot_groups,
    )
    suite, protocol, fusion, pilot_groups = _offline_inputs(arguments)
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=False)
    session_path = output / "session.json"
    session = {
        "schema_version": "m5-stage-b-session-v1",
        "status": "running",
        "authorization_phrase": arguments.authorization,
        "authorization_manifest_sha256": sha256_file(arguments.authorization_manifest),
        "authorized_http_attempts": manifest["http_attempt_ceiling"],
        "actual_http_attempts": 0,
        "resume_permitted": False,
        "failure": None,
    }
    ResultStore.write_json(session_path, session)
    provider = None
    try:
        from semanticflow.optimizer.fusion import build_fused_operator
        from semanticflow.optimizer.fusion_planner import build_physical_plan
        from semanticflow.optimizer.models import load_optimizer_settings
        from semanticflow.optimizer.profiler import PredicateProfiler
        from semanticflow.providers.jev import JevProvider
        from semanticflow.query.models import QuerySpec

        provider_config = fusion["provider"]
        provider = JevProvider.from_env(
            provider_config["model"],
            api_url=provider_config["api_url"],
            timeout_seconds=float(provider_config["timeout_seconds"]),
            max_retries=int(provider_config["max_retries"]),
        )
        optimizer = load_optimizer_settings("configs/optimizer.milestone_4.yaml")
        pricing = {
            "input_per_million_usd": provider_config["input_price_per_million_usd"],
            "output_per_million_usd": provider_config["output_price_per_million_usd"],
        }
        for dataset_id, dataset in suite.datasets.items():
            records = records_for_partition(dataset.path, "pilot")
            dataset_manifest = json.loads(
                dataset.path.with_suffix(dataset.path.suffix + ".manifest.json").read_text("utf-8")
            )
            predicates = suite_predicates(suite, dataset_id)
            logical = len(records) * len(predicates)
            per_dataset_attempts = logical * (int(provider_config["max_retries"]) + 1)
            settings = optimizer.model_copy(
                update={
                    "profiling": optimizer.profiling.model_copy(
                        update={
                            "concurrency": int(provider_config["concurrency"]),
                            "max_attempts": per_dataset_attempts,
                        }
                    )
                }
            )
            query = QuerySpec(
                query_id=f"m5-{dataset_id}-pilot-reference",
                workload_id=f"m5-{dataset_id}-pilot-reference",
                dataset_path=dataset.path,
                dataset_version=dataset_manifest["dataset_fingerprint_sha256"],
                dataset_preparation_version=dataset_manifest["schema_version"],
                operation="and",
                limit=len(records),
                threshold=suite.threshold,
                max_calls=per_dataset_attempts,
                predicates=predicates,
                config_id="m5-stage-b-reference-v1",
            )
            profile = await PredicateProfiler(provider, settings).profile(
                records,
                query,
                run_id=f"m5-{dataset_id}-reference",
                dataset_fingerprint=dataset_manifest["dataset_fingerprint_sha256"],
                profiling_fingerprint=dataset_manifest["partitions"]["pilot"][
                    "fingerprint_sha256"
                ],
                pricing=pricing,
            )
            reference_dir = output / dataset_id / "separate_reference"
            reference_dir.mkdir(parents=True, exist_ok=False)
            ResultStore.write_json(reference_dir / "profile.json", profile.artifact_dict())
            ResultStore.write_jsonl(
                reference_dir / "decisions.jsonl", [trace.to_dict() for trace in profile.traces]
            )
            if not profile.complete:
                raise RuntimeError(f"separate reference profile incomplete for {dataset_id}")

            group_specs = [
                ("conjunctive_binary", ids, len(records))
                for ids in pilot_groups["conjunctive_binary"][dataset_id]
            ]
            group_specs.append(
                (
                    "multi_decision",
                    pilot_groups["multi_decision_smoke"]["groups"][dataset_id],
                    int(pilot_groups["multi_decision_smoke"]["records_per_group"]),
                )
            )
            for mode, ids, record_count in group_specs:
                operator = build_fused_operator(
                    suite_predicates(suite, dataset_id, ids),
                    mode=mode,
                    requested_model=provider.model_identifier,
                    threshold=suite.threshold,
                )
                plan = build_physical_plan(
                    workload_id=f"pilot-{dataset_id}-{operator.group_id}",
                    logical_query_id=f"pilot-{dataset_id}-{operator.group_id}",
                    logical_predicate_ids=ids,
                    groups=[operator],
                    configuration_fingerprint=manifest["configuration_sha256"]["fusion"],
                )
                selected_records = records[:record_count]
                maximum_attempts = record_count * (int(provider_config["max_retries"]) + 1)
                execution = await FusedExecutor(
                    provider, concurrency=int(provider_config["concurrency"])
                ).execute(
                    selected_records,
                    plan,
                    run_id=f"m5-{dataset_id}-{operator.group_id}",
                    maximum_http_attempts=maximum_attempts,
                )
                group_dir = output / dataset_id / f"{mode}_{operator.group_id}"
                group_dir.mkdir(parents=True, exist_ok=False)
                ResultStore.write_json(
                    group_dir / "operator.json", operator.model_dump(mode="json")
                )
                ResultStore.write_jsonl(
                    group_dir / "decisions.jsonl",
                    [trace.to_dict() for trace in execution.traces],
                )
                ResultStore.write_json(
                    group_dir / "report.json",
                    {
                        "complete": execution.complete,
                        "incomplete_reason": execution.incomplete_reason,
                        "plan_id": plan.plan_id,
                        "group_metrics": [group.to_dict() for group in execution.groups],
                        "logical_group_evaluations": execution.logical_group_evaluations,
                        "provider_attempts": execution.provider_attempts,
                        "input_tokens": execution.input_tokens,
                        "output_tokens": execution.output_tokens,
                        "matching_record_ids": [
                            record.review_id for record in execution.matching_records
                        ],
                        "errors": execution.errors,
                    },
                )
                if not execution.complete:
                    raise RuntimeError(f"fused pilot group incomplete: {operator.group_id}")
        session["actual_http_attempts"] = provider.stats.attempts
        if provider.stats.attempts > manifest["http_attempt_ceiling"]:
            raise RuntimeError("authorized HTTP-attempt ceiling exceeded")
        session["status"] = "complete"
        ResultStore.write_json(session_path, session)
        return 0
    except BaseException as exc:
        session["status"] = "incomplete"
        session["failure"] = f"{type(exc).__name__}: {exc}"
        if provider is not None:
            session["actual_http_attempts"] = provider.stats.attempts
        ResultStore.write_json(session_path, session)
        raise
    finally:
        if provider is not None:
            await provider.aclose()


def main() -> int:
    arguments = _base_parser().parse_args()
    if arguments.command == "prepare":
        from scripts.prepare_m5_datasets import prepare

        result = prepare("configs/milestone_5/datasets.yaml")
        _write_new_json(arguments.output, result)
        return 0
    if arguments.command == "audit":
        suite, _, _, _ = _offline_inputs(arguments)
        result = offline_audit(
            suite=suite,
            protocol_path=arguments.protocol,
            fusion_config_path=arguments.fusion,
        )
        _write_new_json(arguments.output, result)
        return 0
    if arguments.command in {"preflight", "simulate"}:
        result = asyncio.run(_run_preflight(arguments))
        _write_new_json(arguments.output, result)
        return 0
    if arguments.command == "authorization-proposal":
        suite, protocol, fusion, pilot_groups = _offline_inputs(arguments)
        budget = build_pilot_budget(suite, protocol, fusion, pilot_groups)
        _write_new_json(
            arguments.output,
            {
                **build_authorization_manifest(
                    budget=budget,
                    suite_path=arguments.workloads,
                    protocol_path=arguments.protocol,
                    fusion_config_path=arguments.fusion,
                    pilot_groups_path=arguments.pilot_groups,
                ),
                "budget": budget,
            },
        )
        return 0
    if arguments.command == "pilot-profile":
        return asyncio.run(_run_pilot(arguments))
    if arguments.command == "freeze-plans":
        _write_new_json(arguments.output, _freeze_from_pilot(arguments))
        return 0
    if arguments.command == "heldout":
        return asyncio.run(_run_heldout(arguments))
    if arguments.command == "stage-d-proposal":
        _write_new_json(arguments.output, _build_stage_d_proposal(arguments))
        return 0
    if arguments.command == "analyze":
        _write_new_json(arguments.output, _analyze_stage_d(Path(arguments.session_root)))
        return 0
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
