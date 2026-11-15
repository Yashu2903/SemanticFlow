from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from semanticflow.dataset.loader import ReviewRecord, load_dataset
from semanticflow.executor.hybrid import HybridExecutor
from semanticflow.optimizer.fusion import build_fused_operator
from semanticflow.optimizer.fusion_planner import build_physical_plan
from semanticflow.optimizer.m51_analysis import diagnose_stage_b
from semanticflow.optimizer.m51_experiment import (
    baseline_audit,
    build_authorization_manifest,
    build_pilot_budget,
    load_yaml,
    validate_stage_b_launch,
)
from semanticflow.storage.results import ResultStore

WORKLOADS = "configs/milestone_5_1/workloads.yaml"
PROTOCOL = "configs/milestone_5_1/protocol.yaml"
SCHEDULE = "configs/milestone_5_1/pilot_schedule.yaml"


def _write_new(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    ResultStore.write_json(target, payload)


def _predicates(workloads: dict[str, Any], dataset_id: str, ids: list[str]):
    from semanticflow.query.models import PredicateSpec

    definitions = workloads["datasets"][dataset_id]["predicates"]
    return [PredicateSpec(id=item, instructions=definitions[item]) for item in ids]


def _partition(path: str | Path, name: str) -> list[ReviewRecord]:
    return sorted(
        [record for record in load_dataset(path) if record.metadata.get("partition") == name],
        key=lambda record: record.review_id,
    )


def _enriched(records: list[ReviewRecord], predicate_ids: list[str], count: int):
    terms = {
        "beauty": ("fail", "broke", "return", "refund", "leak"),
        "banking": ("unauthor", "refund", "revers", "recogn", "cash"),
        "sms": ("win", "prize", "claim", "call", "text"),
    }
    family = next((key for key in terms if predicate_ids[0].startswith(key)), "sms")
    return sorted(
        records,
        key=lambda record: (
            -sum(record.review_text.casefold().count(term) for term in terms[family]),
            record.review_id,
        ),
    )[:count]


async def _execute_operator(
    *,
    provider: Any,
    records: list[ReviewRecord],
    operator: Any,
    output: Path,
    run_id: str,
    max_retries: int,
    concurrency: int,
) -> None:
    plan = build_physical_plan(
        workload_id=run_id,
        logical_query_id=run_id,
        logical_predicate_ids=operator.predicate_ids,
        groups=[operator],
        configuration_fingerprint="m5.1-stage-b-frozen-manifest",
    )
    result = await HybridExecutor(provider, concurrency=concurrency).execute(
        records,
        plan,
        run_id=run_id,
        maximum_http_attempts=len(records) * (max_retries + 1),
    )
    output.mkdir(parents=True, exist_ok=False)
    ResultStore.write_json(output / "operator.json", operator.model_dump(mode="json"))
    ResultStore.write_jsonl(
        output / "decisions.jsonl", [trace.to_dict() for trace in result.traces]
    )
    ResultStore.write_json(
        output / "report.json",
        {
            "complete": result.complete,
            "incomplete_reason": result.incomplete_reason,
            "logical_group_evaluations": result.logical_group_evaluations,
            "named_question_evaluations": result.named_question_evaluations,
            "provider_attempts": result.provider_attempts,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "matching_record_ids": [record.review_id for record in result.matching_records],
            "errors": result.errors,
        },
    )
    if not result.complete:
        raise RuntimeError(f"incomplete physical operator: {operator.group_id}")


async def _run_pilot(args: argparse.Namespace) -> int:
    manifest = json.loads(Path(args.authorization_manifest).read_text("utf-8"))
    validate_stage_b_launch(
        manifest,
        authorization=args.authorization,
        output_directory=args.output,
        workloads_path=args.workloads,
        protocol_path=args.protocol,
        schedule_path=args.schedule,
    )
    # Paid-provider import is deliberately after every authorization/fingerprint check.
    from semanticflow.providers.jev import JevProvider

    workloads = load_yaml(args.workloads)
    protocol = load_yaml(args.protocol)
    schedule = load_yaml(args.schedule)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    session_path = output / "session.json"
    session = {
        "schema_version": "m5.1-stage-b-session-v1",
        "status": "running",
        "actual_http_attempts": 0,
        "authorized_http_attempts": manifest["budget"]["retry_inclusive_http_attempts"],
        "resume_permitted": False,
        "failure": None,
    }
    ResultStore.write_json(session_path, session)
    provider = JevProvider.from_env(
        protocol["model"], timeout_seconds=30, max_retries=int(protocol["max_retries"])
    )
    try:
        for dataset_id, definition in workloads["datasets"].items():
            dataset_schedule = schedule["datasets"][dataset_id]
            representative = _partition(definition["path"], "pilot")[
                : int(schedule["representative_records_per_dataset"])
            ]
            development = _partition(definition["path"], "development")
            core = list(dataset_schedule["ordered_core"])
            dataset_output = output / dataset_id
            for predicate_id in core:
                operator = build_fused_operator(
                    _predicates(workloads, dataset_id, [predicate_id]),
                    mode="single",
                    requested_model=protocol["model"],
                    threshold=float(workloads["threshold"]),
                )
                await _execute_operator(
                    provider=provider,
                    records=representative,
                    operator=operator,
                    output=dataset_output / "representative" / f"single_{predicate_id}",
                    run_id=f"m51-{dataset_id}-representative-{predicate_id}",
                    max_retries=int(protocol["max_retries"]),
                    concurrency=int(protocol["concurrency"]),
                )
            for ids in dataset_schedule["batch_groups"]:
                operator = build_fused_operator(
                    _predicates(workloads, dataset_id, ids),
                    mode="multi_decision",
                    requested_model=protocol["model"],
                    threshold=float(workloads["threshold"]),
                )
                await _execute_operator(
                    provider=provider,
                    records=representative,
                    operator=operator,
                    output=dataset_output / "representative" / f"batch_{operator.group_id}",
                    run_id=f"m51-{dataset_id}-batch-{operator.group_id}",
                    max_retries=int(protocol["max_retries"]),
                    concurrency=int(protocol["concurrency"]),
                )
            conjunction = build_fused_operator(
                _predicates(workloads, dataset_id, core),
                mode="conjunctive_binary",
                requested_model=protocol["model"],
                threshold=float(workloads["threshold"]),
            )
            await _execute_operator(
                provider=provider,
                records=representative,
                operator=conjunction,
                output=dataset_output / "representative" / f"conjunction_{conjunction.group_id}",
                run_id=f"m51-{dataset_id}-conjunction",
                max_retries=int(protocol["max_retries"]),
                concurrency=int(protocol["concurrency"]),
            )
            enriched_ids = list(dataset_schedule["enriched_group"])
            enriched = _enriched(
                development, enriched_ids, int(schedule["enriched_records_per_dataset"])
            )
            ResultStore.write_json(
                dataset_output / "enriched_selection.json",
                {
                    "method": schedule["enrichment_method"],
                    "record_ids": [record.review_id for record in enriched],
                    "population_representative": False,
                },
            )
            for predicate_id in enriched_ids:
                operator = build_fused_operator(
                    _predicates(workloads, dataset_id, [predicate_id]),
                    mode="single",
                    requested_model=protocol["model"],
                    threshold=float(workloads["threshold"]),
                )
                await _execute_operator(
                    provider=provider,
                    records=enriched,
                    operator=operator,
                    output=dataset_output / "enriched" / f"single_{predicate_id}",
                    run_id=f"m51-{dataset_id}-enriched-{predicate_id}",
                    max_retries=int(protocol["max_retries"]),
                    concurrency=int(protocol["concurrency"]),
                )
            batch = build_fused_operator(
                _predicates(workloads, dataset_id, enriched_ids),
                mode="multi_decision",
                requested_model=protocol["model"],
                threshold=float(workloads["threshold"]),
            )
            await _execute_operator(
                provider=provider,
                records=enriched,
                operator=batch,
                output=dataset_output / "enriched" / f"batch_{batch.group_id}",
                run_id=f"m51-{dataset_id}-enriched-batch",
                max_retries=int(protocol["max_retries"]),
                concurrency=int(protocol["concurrency"]),
            )
            if provider.stats.attempts > session["authorized_http_attempts"]:
                raise RuntimeError("authorized global HTTP-attempt ceiling exceeded")
        session["actual_http_attempts"] = provider.stats.attempts
        session["status"] = "complete"
        ResultStore.write_json(session_path, session)
        return 0
    except BaseException as exc:
        session["actual_http_attempts"] = provider.stats.attempts
        session["status"] = "incomplete"
        session["failure"] = f"{type(exc).__name__}: {exc}"
        ResultStore.write_json(session_path, session)
        raise
    finally:
        await provider.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(description="SemanticFlow M5.1 hybrid harness")
    parser.add_argument(
        "command",
        choices=(
            "audit",
            "prepare",
            "preflight",
            "offline-simulation",
            "authorization-proposal",
            "pilot-profile",
            "freeze-plans",
            "held-out-evaluation",
            "analyze",
        ),
    )
    parser.add_argument("--workloads", default=WORKLOADS)
    parser.add_argument("--protocol", default=PROTOCOL)
    parser.add_argument("--schedule", default=SCHEDULE)
    parser.add_argument("--output")
    parser.add_argument("--authorization-manifest")
    parser.add_argument("--authorization")
    args = parser.parse_args()
    if args.command == "prepare":
        from scripts.prepare_m5_datasets import prepare

        if not args.output:
            raise ValueError("prepare requires --output")
        _write_new(args.output, prepare("configs/milestone_5_1/datasets.yaml"))
        return 0
    if args.command == "authorization-proposal":
        _write_new(
            args.output,
            build_authorization_manifest(
                workloads_path=args.workloads,
                protocol_path=args.protocol,
                schedule_path=args.schedule,
            ),
        )
        return 0
    if args.command in {"audit", "preflight", "offline-simulation"}:
        payload = {
            "schema_version": "m5.1-stage-a-preflight-v1",
            "baseline_audit": baseline_audit(),
            "budget": build_pilot_budget(
                workloads_path=args.workloads,
                protocol_path=args.protocol,
                schedule_path=args.schedule,
            ),
            "stage_b_diagnostics": diagnose_stage_b(
                "results/milestone_5/live_fusion_pilot_20261115_stage_b_approved",
                "data/milestone_5",
            ),
            "provider_initialized": False,
            "live_jev_calls": 0,
        }
        _write_new(args.output, payload)
        return 0
    if args.command == "pilot-profile":
        if not args.authorization_manifest or not args.output:
            raise ValueError("pilot-profile requires manifest and output")
        return asyncio.run(_run_pilot(args))
    if args.command in {"freeze-plans", "held-out-evaluation"}:
        raise PermissionError(
            "this stage requires a versioned post-pilot manifest and separate authorization"
        )
    if args.command == "analyze":
        _write_new(
            args.output,
            {
                "status": "analysis scaffold ready; supply a complete Stage B session",
                "live_jev_calls": 0,
            },
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
