from __future__ import annotations

import asyncio
import csv
import json
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from semanticflow.optimizer.m522_action_universe import AuthorizedActionUniverse
from semanticflow.optimizer.m522_live_acquisition import (
    AmbiguousDispatchError,
    LiveAcquisitionAdapter,
    ProviderResponseRecord,
)
from semanticflow.optimizer.m522_session import AppendOnlyEventLog
from semanticflow.optimizer.m531_protocol import (
    DEFAULT_HARD_COST_USD,
    HARD_ATTEMPT_LIMIT,
    HARD_REQUEST_LIMIT,
    FreshRecord,
    build_experiment,
    canonical_hash,
    file_hash,
)

ProviderFactory = Callable[[], Any]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(
    path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _selection_rows(records: tuple[FreshRecord, ...]) -> list[dict[str, Any]]:
    return [
        {
            "workload_id": row.workload_id,
            "dataset": row.dataset,
            "record_id": row.record_id,
            "source_record_id": row.source_record_id,
            "content_sha256": row.text_sha256,
            "original_partition": row.original_partition,
            "validation_stratum": row.stratum,
            "pre_provider_enrichment_score": row.enrichment_score,
        }
        for row in records
    ]


def _write_artifact_manifest(output: Path, config_fingerprint: str) -> None:
    files = sorted(
        path
        for path in output.iterdir()
        if path.is_file() and path.name != "artifact_integrity_manifest.json"
    )
    _write_json(
        output / "artifact_integrity_manifest.json",
        {
            "schema_version": "m5.3.1-artifact-integrity-v1",
            "configuration_fingerprint": config_fingerprint,
            "artifacts": {
                path.name: {"sha256": file_hash(path), "bytes": path.stat().st_size}
                for path in files
            },
        },
    )


def prepare_dry_run(output: Path) -> dict[str, Any]:
    config, records, universe, plan = build_experiment()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "frozen_experiment_config.json", config)
    _write_json(output / "authorized_action_universe.json", universe.model_dump(mode="json"))
    _write_csv(output / "fresh_record_selection_manifest.csv", _selection_rows(records))
    _write_csv(output / "request_plan.csv", plan)
    by_workload = Counter(row["workload_id"] for row in plan)
    by_role = Counter(row["role"] for row in plan)
    coverage = {
        "schema_version": "m5.3.1-provider-evidence-coverage-v1",
        "status": "PENDING_LIVE_EXECUTION",
        "selected_records": len(records),
        "fully_paired_records": 0,
        "completed_requests": 0,
        "planned_requests": len(plan),
        "missing_requests": len(plan),
        "live_provider_execution_complete": False,
        "independent_evaluation_complete": False,
    }
    _write_json(output / "provider_evidence_coverage_report.json", coverage)
    report = {
        "status": "M531_DRY_RUN_VALIDATED",
        "paid_provider_initialized": False,
        "paid_requests": 0,
        "configuration_fingerprint": config["configuration_fingerprint"],
        "action_universe_fingerprint": universe.universe_fingerprint,
        "selected_records": len(records),
        "independently_eligible_by_workload": {
            key: value["independently_eligible"]
            for key, value in config["independence"]["eligibility"].items()
        },
        "planned_requests": len(plan),
        "planned_requests_by_workload": dict(sorted(by_workload.items())),
        "planned_requests_by_operation": dict(sorted(by_role.items())),
        "estimated_token_priced_cost_usd": config["planning_cost_usd"],
        "summed_action_cost_forecast_usd": config[
            "summed_action_cost_forecast_usd"
        ],
        "default_hard_cost_usd": config["default_hard_cost_usd"],
        "maximum_http_attempts": config["maximum_http_attempts"],
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    _write_json(output / "dry_run_report.json", report)
    _write_artifact_manifest(output, config["configuration_fingerprint"])
    return report


def _event_state(events: tuple[Any, ...]) -> dict[str, Any]:
    completed: set[str] = set()
    started: set[str] = set()
    responses: dict[str, dict[str, Any]] = {}
    receipts: dict[str, dict[str, Any]] = {}
    status = "CREATED"
    for event in events:
        payload = event.payload
        if event.event_type == "session_created":
            status = "AUTHORIZED"
        elif event.event_type == "status_changed":
            status = str(payload["status"])
        elif event.event_type == "action_started":
            started.add(str(payload["action_id"]))
        elif event.event_type == "provider_response_persisted":
            responses[str(payload["action_id"])] = dict(payload["response"])
        elif event.event_type == "receipt_persisted":
            receipts[str(payload["action_id"])] = dict(payload["receipt"])
        elif event.event_type == "action_completed":
            action_id = str(payload["action_id"])
            completed.add(action_id)
            started.discard(action_id)
    attempts = sum(int(receipt["http_attempts"]) for receipt in receipts.values())
    cost = sum(float(receipt["token_priced_cost_usd"]) for receipt in receipts.values())
    return {
        "status": status,
        "completed": completed,
        "pending": started,
        "responses": responses,
        "receipts": receipts,
        "attempts": attempts,
        "cost_usd": cost,
    }


def _validate_session_creation(
    events: tuple[Any, ...], config: dict[str, Any], universe: AuthorizedActionUniverse
) -> None:
    if not events:
        return
    first = events[0]
    if first.event_type != "session_created":
        raise ValueError("live session lacks creation event")
    if first.payload["configuration_fingerprint"] != config["configuration_fingerprint"]:
        raise PermissionError("existing session uses a different frozen configuration")
    if first.payload["action_universe_fingerprint"] != universe.universe_fingerprint:
        raise PermissionError("existing session uses a different action universe")


def executable_actions(
    universe: AuthorizedActionUniverse, state: dict[str, Any]
) -> tuple[Any, ...]:
    if state["status"] == "FAILED_CLOSED":
        raise PermissionError("FAILED_CLOSED sessions are not executable")
    return tuple(
        action for action in universe.actions if action.action_id not in state["completed"]
    )


async def execute_live(
    output: Path,
    *,
    authorize_paid: bool,
    provider_factory: ProviderFactory | None = None,
    hard_request_limit: int = HARD_REQUEST_LIMIT,
    hard_attempt_limit: int = HARD_ATTEMPT_LIMIT,
    hard_cost_usd: float = DEFAULT_HARD_COST_USD,
) -> dict[str, Any]:
    if not authorize_paid:
        return prepare_dry_run(output)
    config, records, universe, plan = build_experiment()
    if hard_request_limit != config["maximum_unique_requests"]:
        raise PermissionError("paid request limit differs from frozen scope")
    if hard_attempt_limit != config["maximum_http_attempts"]:
        raise PermissionError("HTTP attempt limit differs from frozen scope")
    if hard_cost_usd <= 0 or hard_cost_usd > config["default_hard_cost_usd"]:
        raise PermissionError("cost limit must be positive and no larger than the frozen cap")
    prepare_dry_run(output)
    event_log = AppendOnlyEventLog(output / "execution_events.jsonl")
    events = event_log.read()
    _validate_session_creation(events, config, universe)
    if not events:
        event_log.append(
            "session_created",
            {
                "experiment_id": config["experiment_id"],
                "configuration_fingerprint": config["configuration_fingerprint"],
                "action_universe_fingerprint": universe.universe_fingerprint,
                "hard_request_limit": hard_request_limit,
                "hard_attempt_limit": hard_attempt_limit,
                "hard_cost_usd": hard_cost_usd,
                "authorization_method": "explicit_cli_flag_--authorize-paid",
            },
        )
    state = _event_state(event_log.read())
    executable_actions(universe, state)
    # A started action without a durable response is ambiguous and must never be resent.
    unresolved = state["pending"] - set(state["responses"])
    if unresolved:
        event_log.append(
            "status_changed",
            {
                "status": "FAILED_CLOSED",
                "reason": "ambiguous_pending_dispatch",
                "actions": sorted(unresolved),
            },
        )
        raise RuntimeError("ambiguous prior dispatch requires external reconciliation")
    record_text = {record.record_id: record.text for record in records}
    plan_index = {row["action_id"]: row for row in plan}
    # Recover only persisted responses; this makes a crash between durability and validation safe.
    for action_id in sorted(state["pending"] & set(state["responses"])):
        action = universe.action(action_id)
        response = ProviderResponseRecord.model_validate(state["responses"][action_id])
        receipt = LiveAcquisitionAdapter.validate_response(action, response)
        if action_id not in state["receipts"]:
            event_log.append(
                "receipt_persisted",
                {"action_id": action_id, "receipt": receipt.model_dump(mode="json")},
            )
        event_log.append("action_completed", {"action_id": action_id})
    state = _event_state(event_log.read())
    if len(state["completed"]) >= len(universe.actions):
        return finalize_live_artifacts(output, config, records, universe, plan, event_log)
    if provider_factory is None:
        raise RuntimeError("authorized live execution requires a provider factory")
    provider = provider_factory()
    adapter = LiveAcquisitionAdapter(provider)
    event_log.append("status_changed", {"status": "EXECUTING"})
    try:
        for action in executable_actions(universe, state):
            state = _event_state(event_log.read())
            if action.action_id in state["completed"]:
                continue
            if len(state["completed"]) + 1 > hard_request_limit:
                raise RuntimeError("hard physical-request limit reached")
            if state["attempts"] + action.maximum_attempts > hard_attempt_limit:
                raise RuntimeError("hard HTTP-attempt reservation limit reached")
            conservative_reservation = float(
                plan_index[action.action_id]["maximum_reserved_cost_usd"]
            )
            if state["cost_usd"] + conservative_reservation > hard_cost_usd:
                raise RuntimeError("hard token-priced cost reservation limit reached")
            event_log.append(
                "action_started",
                {
                    "action_id": action.action_id,
                    "completed_requests_before": len(state["completed"]),
                    "attempts_before": state["attempts"],
                    "cost_before_usd": state["cost_usd"],
                    "reserved_cost_usd": conservative_reservation,
                },
            )
            try:
                response = await adapter.dispatch(action, record_text[action.record_id])
            except AmbiguousDispatchError as exc:
                event_log.append(
                    "ambiguous_attempt",
                    {
                        "action_id": action.action_id,
                        "attempts": exc.attempts,
                        "error": type(exc).__name__,
                    },
                )
                raise
            # Raw response and usage become durable before any semantic validation.
            event_log.append(
                "provider_response_persisted",
                {"action_id": action.action_id, "response": response.model_dump(mode="json")},
            )
            receipt = adapter.validate_response(action, response)
            event_log.append(
                "receipt_persisted",
                {"action_id": action.action_id, "receipt": receipt.model_dump(mode="json")},
            )
            event_log.append("action_completed", {"action_id": action.action_id})
            if _event_state(event_log.read())["cost_usd"] > hard_cost_usd:
                raise RuntimeError("actual token-priced cost exceeded frozen hard limit")
        event_log.append("status_changed", {"status": "COMPLETE"})
    except Exception as exc:
        event_log.append(
            "status_changed",
            {"status": "FAILED_CLOSED", "reason": type(exc).__name__, "message": str(exc)},
        )
        raise
    finally:
        close = getattr(provider, "aclose", None)
        if close is not None:
            await close()
    return finalize_live_artifacts(output, config, records, universe, plan, event_log)


def finalize_live_artifacts(
    output: Path,
    config: dict[str, Any],
    records: tuple[FreshRecord, ...],
    universe: AuthorizedActionUniverse,
    plan: list[dict[str, Any]],
    event_log: AppendOnlyEventLog,
) -> dict[str, Any]:
    state = _event_state(event_log.read())
    action_index = {action.action_id: action for action in universe.actions}
    plan_index = {row["action_id"]: row for row in plan}
    started_at = {
        str(event.payload["action_id"]): event.timestamp
        for event in event_log.read()
        if event.event_type == "action_started"
    }
    completed_at = {
        str(event.payload["action_id"]): event.timestamp
        for event in event_log.read()
        if event.event_type == "action_completed"
    }
    execution_rows = []
    for action_id, receipt in sorted(state["receipts"].items()):
        action = action_index[action_id]
        execution_rows.append(
            {
                "action_id": action_id,
                "workload_id": action.workload_id,
                "record_id": action.record_id,
                "stratum": action.source_partition,
                "role": plan_index[action_id]["role"],
                "predicate_group": plan_index[action_id]["predicate_group"],
                "resolved_model": receipt["resolved_model"],
                "input_tokens": receipt["input_tokens"],
                "output_tokens": receipt["output_tokens"],
                "http_attempts": receipt["http_attempts"],
                "retries": receipt["retries"],
                "token_priced_cost_usd": receipt["token_priced_cost_usd"],
                "response_identity": receipt["response_identity"],
                "execution_started_at_utc": started_at[action_id],
                "execution_completed_at_utc": completed_at[action_id],
            }
        )
    if execution_rows:
        _write_csv(output / "per_request_execution_records.csv", execution_rows)
    complete_records = sum(
        all(
            action.action_id in state["completed"]
            for action in universe.actions
            if action.record_id == record.record_id
        )
        for record in records
    )
    coverage = {
        "schema_version": "m5.3.1-provider-evidence-coverage-v1",
        "status": "COMPLETE" if complete_records == len(records) else state["status"],
        "selected_records": len(records),
        "fully_paired_records": complete_records,
        "completed_requests": len(state["completed"]),
        "planned_requests": len(universe.actions),
        "missing_requests": len(universe.actions) - len(state["completed"]),
        "http_attempts": state["attempts"],
        "token_priced_cost_usd": state["cost_usd"],
        "live_provider_execution_complete": complete_records == len(records),
        "independent_evaluation_complete": False,
    }
    _write_json(output / "provider_evidence_coverage_report.json", coverage)
    _write_json(
        output / "session_state.json",
        {
            **coverage,
            "configuration_fingerprint": config["configuration_fingerprint"],
            "action_universe_fingerprint": universe.universe_fingerprint,
            "event_log_head": event_log.read()[-1].event_hash,
            "event_count": len(event_log.read()),
        },
    )
    _write_artifact_manifest(output, config["configuration_fingerprint"])
    return coverage


def run_live(**kwargs: Any) -> dict[str, Any]:
    return asyncio.run(execute_live(**kwargs))


def verify_output_matches_frozen_scope(output: Path) -> None:
    config, _, universe, _ = build_experiment()
    persisted = json.loads((output / "frozen_experiment_config.json").read_text("utf-8"))
    persisted_universe = json.loads((output / "authorized_action_universe.json").read_text("utf-8"))
    if (
        canonical_hash(
            {key: value for key, value in persisted.items() if key != "configuration_fingerprint"}
        )
        != persisted["configuration_fingerprint"]
    ):
        raise ValueError("persisted configuration fingerprint is invalid")
    if persisted["configuration_fingerprint"] != config["configuration_fingerprint"]:
        raise ValueError("persisted configuration differs from source-derived frozen scope")
    if persisted_universe["universe_fingerprint"] != universe.universe_fingerprint:
        raise ValueError("persisted action universe differs from frozen scope")
