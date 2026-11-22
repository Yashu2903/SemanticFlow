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
from semanticflow.optimizer.m51_stage_d_preflight import validate_stage_d_launch
from semanticflow.storage.results import ResultStore


def _load(path: str | Path) -> Any:
    return json.loads(Path(path).read_text("utf-8"))


def _validate_operator(row: dict[str, Any], operator: Any) -> None:
    if operator.group_id != row["operator_id"]:
        raise ValueError("rebuilt operator ID differs from frozen Stage D schedule")
    if operator.fusion_prompt_hash != row["prompt_hash"]:
        raise ValueError("rebuilt prompt differs from frozen Stage D schedule")
    if operator.fusion_mode != row["execution_mode"]:
        raise ValueError("rebuilt execution mode differs from frozen Stage D schedule")
    if list(operator.provider_questions()) != row["question_names"]:
        raise ValueError("rebuilt named questions differ from frozen Stage D schedule")
    if operator.result_schema_version != row["expected_response_schema"]:
        raise ValueError("rebuilt response schema differs from frozen Stage D schedule")
    if list(operator.predicate_ids) != row["predicate_ids"]:
        raise ValueError("rebuilt predicate order differs from frozen Stage D schedule")


def _smoke_request_ids(schedule: dict[str, Any]) -> list[str]:
    by_size: dict[int, str] = {}
    for row in schedule["physical_requests"]:
        by_size.setdefault(len(row["predicate_ids"]), row["request_id"])
    if set(by_size) != {1, 2, 3, 4}:
        raise ValueError("frozen Stage D schedule lacks a smoke member for each supported size")
    return [by_size[size] for size in sorted(by_size)]


def _verify_persisted_rows(output: Path, request_rows: list[dict[str, Any]]) -> None:
    decisions_path = output / "decisions.jsonl"
    report_path = output / "report.json"
    report = json.loads(report_path.read_text("utf-8"))
    decisions = [
        json.loads(line) for line in decisions_path.read_text("utf-8").splitlines() if line
    ]
    if not report["complete"] or len(decisions) != len(request_rows):
        raise RuntimeError("persisted Stage D operator result is incomplete")
    expected = {(row["record_id"], row["operator_id"]): row for row in request_rows}
    observed: set[tuple[str, str]] = set()
    for decision in decisions:
        key = (decision["review_id"], decision["group_id"])
        row = expected.get(key)
        if row is None or key in observed:
            raise RuntimeError("unexpected or duplicate persisted Stage D decision")
        observed.add(key)
        if decision["model_identifier"] != "jev-1.14.0":
            raise RuntimeError("resolved Stage D model mismatch")
        if set(decision["predicate_probabilities"]) != set(row["predicate_ids"]):
            raise RuntimeError("partial or misnamed Stage D probability response")
        if decision["input_tokens"] <= 0 or decision["output_tokens"] <= 0:
            raise RuntimeError("Stage D provider usage is missing")
        if decision["request_attempt_count"] < 1 or decision["request_attempt_count"] > 3:
            raise RuntimeError("Stage D request attempt count is out of bounds")
    if observed != set(expected):
        raise RuntimeError("not every Stage D request was persisted")


async def _execute(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    schedule: dict[str, Any],
) -> int:
    # The paid-provider module is deliberately imported only after the fail-closed launch gate.
    from semanticflow.providers.jev import JevProvider

    workloads = load_yaml("configs/milestone_5_1/workloads.yaml")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    session_path = output / "session.json"
    logical_plan_group_roles = sum(
        len(row["selected_plan_references"]) for row in schedule["physical_requests"]
    )
    session: dict[str, Any] = {
        "schema_version": "m5.1-stage-d-session-v1",
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "schedule_fingerprint": schedule["schedule_fingerprint"],
        "authorization_scope_fingerprint": manifest["scope_fingerprint"],
        "authorized_unique_physical_requests": manifest["physical_request_ceiling"],
        "authorized_http_attempts": manifest["http_attempt_ceiling"],
        "logical_plan_group_roles": logical_plan_group_roles,
        "actual_http_attempts": 0,
        "completed_physical_requests": 0,
        "completed_request_ids": [],
        "resume_permitted": False,
        "failure": None,
    }
    ResultStore.write_json(session_path, session)
    provider = JevProvider.from_env("jev-1.14.0", timeout_seconds=30, max_retries=2)
    records = {
        dataset: {
            record.review_id: record for record in _partition(definition["path"], "evaluation")
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
            requested_model=first["model"],
            threshold=float(first["threshold"]),
            execution_parameters=first["inference_settings"],
        )
        _validate_operator(first, operator)
        if any(
            (row["dataset"], row["operator_id"]) != (first["dataset"], first["operator_id"])
            for row in selected
        ):
            raise ValueError("attempted to combine incompatible frozen Stage D operators")
        remaining_worst_case = len(selected) * 3
        if provider.stats.attempts + remaining_worst_case > manifest["http_attempt_ceiling"]:
            raise RuntimeError("atomic worst-case attempt reservation would exceed Stage D ceiling")
        await _execute_operator(
            provider=provider,
            records=[records[first["dataset"]][row["record_id"]] for row in selected],
            operator=operator,
            output=destination,
            run_id=run_id,
            max_retries=2,
            concurrency=4,
        )
        _verify_persisted_rows(destination, selected)
        completed.update(row["request_id"] for row in selected)
        session["actual_http_attempts"] = provider.stats.attempts
        session["completed_physical_requests"] = len(completed)
        session["completed_request_ids"] = sorted(completed)
        ResultStore.write_json(session_path, session)

    try:
        for request_id in _smoke_request_ids(schedule):
            await run_rows(
                [rows[request_id]],
                output / "smoke" / request_id,
                f"m51-stage-d-smoke-{request_id}",
            )
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for request_id, row in rows.items():
            if request_id not in completed:
                grouped[(row["dataset"], row["operator_id"])].append(row)
        for (dataset, operator_id), selected in sorted(grouped.items()):
            await run_rows(
                sorted(selected, key=lambda row: row["record_id"]),
                output / "operators" / dataset / operator_id,
                f"m51-stage-d-{dataset}-{operator_id}",
            )
        if len(completed) != manifest["physical_request_ceiling"]:
            raise RuntimeError("Stage D did not persist every frozen physical request")
        session["status"] = "complete"
        session["completed_at"] = datetime.now(UTC).isoformat()
        ResultStore.write_json(session_path, session)
        return 0
    except BaseException as exc:
        session["status"] = "incomplete"
        session["actual_http_attempts"] = provider.stats.attempts
        session["completed_physical_requests"] = len(completed)
        session["completed_request_ids"] = sorted(completed)
        session["failure"] = f"{type(exc).__name__}: {exc}"
        session["restart_requires_versioned_reauthorization"] = True
        ResultStore.write_json(session_path, session)
        raise
    finally:
        await provider.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed M5.1 Stage D held-out runner")
    parser.add_argument("command", choices=("validate", "execute"))
    parser.add_argument(
        "--manifest",
        default=(
            "results/milestone_5_1/stage_d_final_preflight/"
            "stage_d_authorization_manifest.json"
        ),
    )
    parser.add_argument(
        "--schedule",
        default=(
            "results/milestone_5_1/selection_stage_results/"
            "stage_d_execution_schedule.json"
        ),
    )
    parser.add_argument(
        "--catalog",
        default=(
            "results/milestone_5_1/selection_stage_results/"
            "frozen_physical_plan_catalog.json"
        ),
    )
    parser.add_argument(
        "--selections",
        default=(
            "results/milestone_5_1/selection_stage_results/"
            "frozen_strategy_selections.json"
        ),
    )
    parser.add_argument("--output", default="results/milestone_5_1/stage_d_live")
    parser.add_argument("--authorization")
    args = parser.parse_args()
    manifest = _load(args.manifest)
    schedule = _load(args.schedule)
    catalog = _load(args.catalog)["plans"]
    selections = _load(args.selections)["strategies"]
    validate_stage_d_launch(
        manifest,
        authorization=args.authorization,
        schedule=schedule,
        catalog=catalog,
        selections=selections,
        output_directory=args.output,
    )
    if args.command == "execute":
        return asyncio.run(_execute(args, manifest, schedule))
    print(
        f"validated {len(schedule['physical_requests'])} frozen Stage D requests; "
        "provider not initialized"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
