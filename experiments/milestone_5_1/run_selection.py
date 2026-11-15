from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from experiments.milestone_5_1.run_hybrid import _execute_operator, _partition, _predicates
from semanticflow.optimizer.fusion import build_fused_operator
from semanticflow.optimizer.m51_experiment import load_yaml
from semanticflow.optimizer.m51_selection_preflight import (
    AUTHORIZATION_PHRASE,
    validate_selection_launch,
)
from semanticflow.storage.results import ResultStore


def _validate_operator(row: dict[str, Any], operator: Any) -> None:
    if operator.group_id != row["operator_id"]:
        raise ValueError("rebuilt operator ID differs from frozen schedule")
    if operator.fusion_prompt_hash != row["prompt_hash"]:
        raise ValueError("rebuilt prompt differs from frozen schedule")
    if list(operator.provider_questions()) != row["question_names"]:
        raise ValueError("rebuilt question names differ from frozen schedule")
    if operator.result_schema_version != row["expected_response_schema"]:
        raise ValueError("rebuilt response schema differs from frozen schedule")


def _verify_smoke(output: Path, request_ids: list[str]) -> None:
    for request_id in request_ids:
        root = output / "smoke" / request_id
        report = json.loads((root / "report.json").read_text("utf-8"))
        decisions = [
            json.loads(line)
            for line in (root / "decisions.jsonl").read_text("utf-8").splitlines()
            if line
        ]
        if not report["complete"] or len(decisions) != 1:
            raise RuntimeError(f"provider-contract smoke request failed: {request_id}")
        if report["input_tokens"] <= 0 or report["output_tokens"] <= 0:
            raise RuntimeError(f"provider usage missing in smoke request: {request_id}")
        if decisions[0]["model_identifier"] != "jev-1.14.0":
            raise RuntimeError(f"resolved model mismatch in smoke request: {request_id}")


async def _execute(
    args: argparse.Namespace, manifest: dict[str, Any], schedule: dict[str, Any]
) -> int:
    # The paid-provider import occurs only after the complete fail-closed launch validation.
    from semanticflow.providers.jev import JevProvider

    workloads = load_yaml("configs/milestone_5_1/workloads.yaml")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    session_path = output / "session.json"
    session: dict[str, Any] = {
        "schema_version": "m5.1-selection-session-v1",
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "schedule_fingerprint": schedule["schedule_fingerprint"],
        "logical_measurement_roles": schedule["counts"]["logical_measurement_roles"],
        "authorized_unique_physical_requests": schedule["counts"]["unique_physical_requests"],
        "authorized_http_attempts": manifest["deduplicated_http_attempt_ceiling"],
        "actual_http_attempts": 0,
        "completed_physical_requests": 0,
        "resume_permitted": False,
        "failure": None,
    }
    ResultStore.write_json(session_path, session)
    provider = JevProvider.from_env("jev-1.14.0", timeout_seconds=30, max_retries=2)
    records = {
        dataset: {
            record.review_id: record for record in _partition(definition["path"], "selection")
        }
        for dataset, definition in workloads["datasets"].items()
    }
    rows = {row["request_id"]: row for row in schedule["physical_requests"]}
    completed: set[str] = set()

    async def run_rows(selected: list[dict[str, Any]], destination: Path, run_id: str) -> None:
        if not selected:
            return
        first = selected[0]
        operator = build_fused_operator(
            _predicates(workloads, first["dataset"], list(first["predicate_ids"])),
            mode=first["execution_mode"],
            requested_model=first["requested_model"],
            threshold=float(first["threshold"]),
            execution_parameters=first["inference_settings"],
        )
        _validate_operator(first, operator)
        if any(
            (row["dataset"], row["operator_id"]) != (first["dataset"], first["operator_id"])
            for row in selected
        ):
            raise ValueError("attempted to combine incompatible frozen operators")
        remaining_worst_case = len(selected) * 3
        if (
            provider.stats.attempts + remaining_worst_case
            > manifest["deduplicated_http_attempt_ceiling"]
        ):
            raise RuntimeError("atomic worst-case HTTP-attempt reservation would exceed ceiling")
        await _execute_operator(
            provider=provider,
            records=[records[first["dataset"]][row["record_id"]] for row in selected],
            operator=operator,
            output=destination,
            run_id=run_id,
            max_retries=2,
            concurrency=4,
        )
        completed.update(row["request_id"] for row in selected)
        session["actual_http_attempts"] = provider.stats.attempts
        session["completed_physical_requests"] = len(completed)
        ResultStore.write_json(session_path, session)

    try:
        for request_id in schedule["smoke_checkpoint"]:
            await run_rows(
                [rows[request_id]],
                output / "smoke" / request_id,
                f"m51-selection-smoke-{request_id}",
            )
        _verify_smoke(output, schedule["smoke_checkpoint"])
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for request_id, row in rows.items():
            if request_id not in completed:
                grouped[(row["dataset"], row["operator_id"])].append(row)
        for (dataset, operator_id), selected in sorted(grouped.items()):
            await run_rows(
                sorted(selected, key=lambda row: row["record_id"]),
                output / "operators" / dataset / operator_id,
                f"m51-selection-{dataset}-{operator_id}",
            )
        if len(completed) != schedule["counts"]["unique_physical_requests"]:
            raise RuntimeError("selection execution did not persist every frozen physical request")
        session["status"] = "complete"
        session["completed_at"] = datetime.now(UTC).isoformat()
        ResultStore.write_json(session_path, session)
        return 0
    except BaseException as exc:
        session["status"] = "incomplete"
        session["actual_http_attempts"] = provider.stats.attempts
        session["failure"] = f"{type(exc).__name__}: {exc}"
        ResultStore.write_json(session_path, session)
        raise
    finally:
        await provider.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed launch gate for the frozen M5.1 selection schedule"
    )
    parser.add_argument("command", choices=("validate", "execute"))
    parser.add_argument(
        "--manifest",
        default="results/milestone_5_1/selection_stage_preflight/authorization_manifest.json",
    )
    parser.add_argument(
        "--schedule",
        default="results/milestone_5_1/selection_stage_preflight/selection_schedule.json",
    )
    parser.add_argument("--output", default="results/milestone_5_1/selection_stage_live")
    parser.add_argument("--authorization")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text("utf-8"))
    schedule = json.loads(Path(args.schedule).read_text("utf-8"))
    validate_selection_launch(
        manifest,
        authorization=args.authorization,
        schedule=schedule,
        output_directory=args.output,
    )
    if args.command == "execute":
        return asyncio.run(_execute(args, manifest, schedule))
    print(
        f"validated {schedule['counts']['logical_measurement_roles']} roles against "
        f"{len(schedule['physical_requests'])} unique requests; exact phrase: "
        f"{AUTHORIZATION_PHRASE}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
