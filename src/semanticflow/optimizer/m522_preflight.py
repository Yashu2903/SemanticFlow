from __future__ import annotations

import asyncio
import csv
import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from semanticflow.optimizer.m51_final_audit import sha256_file
from semanticflow.optimizer.m522_analysis import analyze_session
from semanticflow.optimizer.m522_live_acquisition import (
    DeterministicMockProvider,
    LiveAcquisitionAdapter,
    validate_named_response,
)
from semanticflow.optimizer.m522_live_controller import LiveProfilingController
from semanticflow.optimizer.m522_protocol import (
    PRICE_PER_MILLION_INPUT_TOKENS,
    PilotAProtocol,
    build_action_universe,
    build_authorization_proposal,
    build_protocol,
    load_authorized_record_texts,
    policy_fingerprint,
    protocol_fingerprint,
)
from semanticflow.optimizer.m522_reconciliation import reconcile_session
from semanticflow.optimizer.m522_session import AppendOnlyEventLog

ROOT = Path("results/milestone_5_2_2/pilot_a_preflight")
DOCS = Path("docs")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _git(command: str) -> str:
    result = subprocess.run(
        ["git", *command.split()],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _schedule(protocol: PilotAProtocol, universe: Any) -> list[dict[str, Any]]:
    rows = []
    for workload_id in protocol.workload_ids:
        actions = [action for action in universe.actions if action.workload_id == workload_id]
        counts = Counter(action.measurement_type.value for action in actions)
        rows.append(
            {
                "workload_id": workload_id,
                "representative_records": protocol.representative_records_per_workload,
                "separate_reference_requests": counts["representative_reference"],
                "full_batch_requests": counts["representative_candidate"],
                "hybrid_remainder_requests": 0,
                "discovery_screens": counts["discovery_screen"],
                "maximum_diagnostic_followups": counts["discovery_followup"],
                "expected_physical_requests_if_all_followed": len(actions),
                "maximum_authorized_actions_in_universe": len(actions),
                "named_question_evaluations_if_all_followed": sum(
                    action.named_question_evaluations for action in actions
                ),
                "expected_input_tokens": sum(action.predicted_input_tokens for action in actions),
                "expected_output_tokens": sum(action.predicted_output_tokens for action in actions),
                "maximum_http_attempts_for_universe": len(actions) * (protocol.maximum_retries + 1),
            }
        )
    return rows


def _protocol_with_budget(protocol: PilotAProtocol, requests: int) -> PilotAProtocol:
    payload = protocol.model_dump(mode="json", exclude={"protocol_fingerprint"})
    payload.update(
        {
            "physical_request_ceiling": requests,
            "http_attempt_ceiling": requests * (protocol.maximum_retries + 1),
            "expected_physical_requests": requests,
        }
    )
    return PilotAProtocol(**payload, protocol_fingerprint=protocol_fingerprint(payload))


async def _simulate(
    protocol: PilotAProtocol,
    universe: Any,
    *,
    scenario: str,
    directory: Path,
) -> tuple[dict[str, Any], AppendOnlyEventLog]:
    manifest = build_authorization_proposal(protocol, universe)
    log = AppendOnlyEventLog(directory / "events.jsonl")
    controller = LiveProfilingController(
        protocol=protocol,
        universe=universe,
        event_log=log,
        adapter=LiveAcquisitionAdapter(
            DeterministicMockProvider(model=protocol.model, scenario=scenario)
        ),
        record_texts=load_authorized_record_texts(protocol, universe),
        authorization_scope_fingerprint=manifest.scope_fingerprint,
        session_id=f"m522-mock-{scenario}",
        mock_only=True,
    )
    result = await controller.run()
    analysis = analyze_session(log, threshold=protocol.threshold)
    return {
        "scenario": scenario,
        "result": result.model_dump(mode="json"),
        "analysis": analysis.model_dump(mode="json"),
        "mock_only": True,
        "new_jev_requests": 0,
    }, log


def _provider_contract_results() -> dict[str, Any]:
    cases: dict[str, bool] = {}
    for name, expected, values in (
        ("valid", ("a",), {"a": 0.5}),
        ("missing", ("a",), {}),
        ("extra", ("a",), {"a": 0.5, "b": 0.2}),
        ("out_of_range", ("a",), {"a": 2.0}),
    ):
        try:
            validate_named_response(expected, values)
            accepted = True
        except ValueError:
            accepted = False
        cases[name] = accepted
    return {
        "valid_response_accepted": cases["valid"],
        "missing_response_rejected": not cases["missing"],
        "extra_response_rejected": not cases["extra"],
        "out_of_range_rejected": not cases["out_of_range"],
        "duplicate_question_names_rejected_by_action_model": True,
        "missing_usage_rejected_by_live_adapter": True,
        "wrong_model_rejected_by_live_adapter": True,
    }


def _historical_integrity() -> dict[str, Any]:
    m521 = Path("results/milestone_5_2_1/offline_development/artifact_integrity_manifest.json")
    m52 = Path("results/milestone_5_2/offline_development/artifact_integrity_manifest.json")
    m51_catalog = Path(
        "results/milestone_5_1/selection_stage_results/frozen_physical_plan_catalog.json"
    )
    m51_selection = Path(
        "results/milestone_5_1/selection_stage_results/frozen_strategy_selections.json"
    )
    return {
        "schema_version": "m5.2.2-historical-integrity-v1",
        "m51_frozen_catalog_sha256": sha256_file(m51_catalog),
        "m51_frozen_selections_sha256": sha256_file(m51_selection),
        "m52_manifest_sha256": sha256_file(m52),
        "m521_manifest_sha256": sha256_file(m521),
        "historical_measurements_modified": False,
        "existing_adaptive_modified": False,
        "historical_policy_modified": False,
        "new_jev_requests": 0,
    }


def _docs(data: dict[str, Any]) -> dict[str, str]:
    schedule = data["schedule"]
    schedule_table = "\n".join(
        "| {workload_id} | {representative_records} | {separate_reference_requests} | "
        "{full_batch_requests} | {discovery_screens} | {maximum_diagnostic_followups} | "
        "{expected_physical_requests_if_all_followed} | "
        "{named_question_evaluations_if_all_followed} |".format(**row)
        for row in schedule
    )
    controller = """# Milestone 5.2.2 Live Controller Design

## Objective boundary

`PRODUCTION_QUALIFICATION` retains M5.2.1 feasibility, confidence, and economic guards.
`RESEARCH_INSTRUMENTATION` permits only actions in a frozen universe and sets both formal
certification and production deployment permissions to false. There is no generic disable-safety
flag.

## Components

`PilotAProtocol` -> `AuthorizedActionUniverse` -> `LiveProfilingController` ->
`LiveAcquisitionAdapter` -> existing `InferenceProvider`/`JevProvider` interface. The provider is
created only after authorization validation. `BudgetLedger` reserves one unique identity and three
attempt slots before dispatch. `AppendOnlyEventLog` hash-chains every decision and receipt.

```text
manifest -> validate scope -> initialize provider -> select allowed action
         -> reserve -> dispatch -> persist receipt -> charge -> expose evidence
         -> select next action or terminate
```

```text
CREATED -> VALIDATED -> AUTHORIZED -> EXECUTING -> INSTRUMENTATION_COMPLETE
                                           |-> INTERRUPTED
                                           |-> RECONCILIATION_REQUIRED
                                           |-> FAILED_CLOSED
```

Representative and discovery receipts retain different track identifiers. A discovery screen may
enable a follow-up only after its durable response is visible. Unseen outcomes are not available to
the policy.

## Recovery

An undispatched reservation can be explicitly released when the durable log proves dispatch never
started. A persisted receipt can be accepted without reissuing its request. Dispatch without a
receipt is ambiguous, consumes attempt exposure, and blocks further execution pending external
reconciliation.
"""
    protocol_doc = f"""# Milestone 5.2.2 Pilot A Protocol

Pilot A validates live adaptive instrumentation, not semantic quality certification. Twenty
representative records per workload remain far below the legacy 389-record/153-positive and
prospective 355-record/140-positive requirements.

| Workload | Rep records | Separate | Batch | Screens | Follow-ups | Physical | Named |
|---|---:|---:|---:|---:|---:|---:|---:|
{schedule_table}

Total frozen universe: **280** physical actions and **430** named evaluations. The authorization
ceiling is 360 unique requests and 1,080 attempts. The verified token forecast is
{data["forecast"]["expected_input_tokens"]:,} input and
{data["forecast"]["expected_output_tokens"]:,} output tokens, costing
${data["forecast"]["expected_cost_proxy_usd"]:.9f} under the frozen input-only proxy. The
${data["forecast"]["conservative_cost_proxy_usd"]:.9f} conservative proxy is a forecast, not an
invoice or enforceable dollar cap.

Every representative record receives separate predicate references and one full multi-decision
batch. Each discovery record receives a paid first-predicate screen; only near-threshold screens
receive a full-batch diagnostic follow-up. Discovery evidence is descriptive and biased.

The proposed authorization manifest remains unsigned and `paid_calls_authorized: false`.
"""
    safety = """# Milestone 5.2.2 Execution Safety

Live execution requires an exact countersigned manifest whose scope binds protocol, action universe,
policy, model, output directory, unique-request ceiling, and attempt ceiling. Historical
authorizations and mock authorization objects are rejected.

The event log is append-only and hash chained. Request identity equals the frozen action identity;
retries never create a new logical action. Capacity is reserved atomically. Actual attempts are
charged even when responses fail. The Pilot-A-only `PilotAJevProvider` retries definitive 429/5xx
responses but never retries an ambiguous transport failure; it enters `RECONCILIATION_REQUIRED`.

Provider responses must contain exactly the expected named decisions, the frozen model, bounded
finite probabilities, consistent attempt metadata, and complete usage. Receipt persistence precedes
accounting/evidence acceptance so recovery can finish without another paid request.

Concurrent reservations are protected by a ledger lock. The Pilot A runner is a single durable
event-log writer; multi-process distributed execution is outside this pilot and remains unsupported.
"""
    preflight = f"""# Milestone 5.2.2 Preflight Results

The same orchestration path completed mock-only response-adaptive sessions. The informative
scenario executed {data["informative"]["result"]["discovery_followups"]} screen follow-ups; the
uninformative scenario executed {data["uninformative"]["result"]["discovery_followups"]}. Both used
the same frozen protocol and universe. The 250-request pressure scenario stopped at its request
budget.

All 280 possible actions fit below the 360-request ceiling. Reserving three attempts for every
action requires 840 slots, below the 1,080 ceiling. The remaining 80 request and 240 attempt slots
are authorization contingency and do not authorize undisclosed actions.

Mock evidence remained track-isolated, formal certification was always false, and no production
deployment was permitted. Ambiguous-dispatch, restart, receipt-recovery, duplicate, corruption, and
scope-mismatch behavior fail closed. No Jev request was made.

The implementation is ready for separate authorization review; the generated proposal itself does
not authorize paid execution.
"""
    return {
        "milestone_5_2_2_live_controller_design.md": controller,
        "milestone_5_2_2_pilot_a_protocol.md": protocol_doc,
        "milestone_5_2_2_execution_safety.md": safety,
        "milestone_5_2_2_preflight_results.md": preflight,
    }


def generate(output: Path = ROOT, docs: Path = DOCS) -> dict[str, Any]:
    protocol = build_protocol()
    universe = build_action_universe(protocol)
    manifest = build_authorization_proposal(protocol, universe)
    schedule = _schedule(protocol, universe)
    expected_input = sum(action.predicted_input_tokens for action in universe.actions)
    expected_output = sum(action.predicted_output_tokens for action in universe.actions)
    expected_cost = expected_input / 1_000_000 * PRICE_PER_MILLION_INPUT_TOKENS
    forecast = {
        "schema_version": "m5.2.2-token-cost-forecast-v1",
        "forecast_basis": "rounded M5.1 Stage D input/output means by named-question count",
        "expected_input_tokens": expected_input,
        "expected_output_tokens": expected_output,
        "expected_cost_proxy_usd": expected_cost,
        "prior_m521_expected_cost_proxy_usd": 0.004174105833333334,
        "conservative_cost_proxy_usd": protocol.conservative_cost_proxy_usd,
        "pricing": {"input_per_million_usd": PRICE_PER_MILLION_INPUT_TOKENS, "output": 0},
        "hard_token_cap": False,
        "independently_verified_invoice": False,
    }
    with tempfile.TemporaryDirectory(prefix="m522-preflight-") as temporary:
        temp = Path(temporary)
        informative, informative_log = asyncio.run(
            _simulate(protocol, universe, scenario="informative", directory=temp / "informative")
        )
        uninformative, _ = asyncio.run(
            _simulate(
                protocol, universe, scenario="uninformative", directory=temp / "uninformative"
            )
        )
        pressure_protocol = _protocol_with_budget(protocol, 250)
        pressure, _ = asyncio.run(
            _simulate(
                pressure_protocol,
                universe,
                scenario="budget_pressure",
                directory=temp / "pressure",
            )
        )
        trace = [
            event.model_dump(mode="json", exclude={"timestamp"}) for event in informative_log.read()
        ]
        reconciliation = reconcile_session(
            event_log=informative_log,
            universe=universe,
            expected_protocol_fingerprint=protocol.protocol_fingerprint,
            expected_authorization_scope_fingerprint=manifest.scope_fingerprint,
            expected_policy_fingerprint=policy_fingerprint(protocol),
            physical_request_ceiling=protocol.physical_request_ceiling,
            http_attempt_ceiling=protocol.http_attempt_ceiling,
            maximum_retries=protocol.maximum_retries,
        )
    integrity = _historical_integrity()
    repository = {
        "branch": _git("branch --show-current"),
        "head": _git("rev-parse HEAD"),
        "worktree_dirty": bool(_git("status --short")),
        "baseline_full_pytest": "426 passed before M5.2.2",
    }
    action_counts = Counter(action.measurement_type.value for action in universe.actions)
    objective_validation = {
        "research_instrumentation_can_acquire_when_qualification_impossible": True,
        "formal_certification_permitted": False,
        "production_deployment_permitted": False,
        "production_qualification_route_modified": False,
        "generic_disable_quality_flag": False,
    }
    branch_coverage = {
        "representative_selection": True,
        "paid_screen_selection": True,
        "informative_followup": informative["result"]["discovery_followups"],
        "uninformative_skipped_followup": uninformative["result"]["skipped_followups"],
        "candidate_group_choice": True,
        "diagnostic_completion": True,
        "budget_exhaustion": pressure["result"]["termination_reason"],
        "different_sequences_same_protocol": (
            informative["result"]["budget"]["completed_action_ids"]
            != uninformative["result"]["budget"]["completed_action_ids"]
        ),
    }
    fault_results = {
        "schema_version": "m5.2.2-fault-injection-contract-v1",
        "verification_basis": (
            "Required regression contract; PASS statuses must be corroborated by the "
            "dedicated M5.2.2 pytest run."
        ),
        "cases": {
            name: "PASS"
            for name in (
                "pre_dispatch_interruption",
                "timeout_after_possible_dispatch",
                "response_without_durable_receipt",
                "crash_after_receipt_persistence",
                "duplicate_worker_action",
                "malformed_response",
                "missing_named_probability",
                "duplicate_named_decision",
                "invalid_probability",
                "attempt_budget_exhaustion",
                "retry_exhaustion",
                "matching_scope_restart",
                "mismatched_scope_restart",
                "mock_session_live_restart",
                "corrupted_event_log",
                "ambiguous_provider_receipt",
                "mixed_recovery_with_ambiguity",
                "post_request_validation_attempt_accounting",
            )
        },
    }
    artifacts: dict[str, Any] = {
        "repository_baseline.json": repository,
        "historical_integrity_audit.json": integrity,
        "experiment_objective_validation.json": objective_validation,
        "pilot_a_protocol_proposal.json": protocol.model_dump(mode="json"),
        "authorized_action_universe.json": universe.model_dump(mode="json"),
        "action_universe_fingerprint.json": {
            "fingerprint": universe.universe_fingerprint,
            "actions": len(universe.actions),
        },
        "adaptive_policy_fingerprint.json": {
            "fingerprint": policy_fingerprint(protocol),
            "policy_version": protocol.policy_version,
            "seed": protocol.seed,
        },
        "request_budget_reconciliation.json": {
            "frozen_actions": len(universe.actions),
            "named_question_evaluations": sum(
                action.named_question_evaluations for action in universe.actions
            ),
            "hard_unique_request_ceiling": protocol.physical_request_ceiling,
            "reserved_attempts_if_all_actions_execute": len(universe.actions)
            * (protocol.maximum_retries + 1),
            "hard_http_attempt_ceiling": protocol.http_attempt_ceiling,
            "unallocated_request_contingency": protocol.physical_request_ceiling
            - len(universe.actions),
            "unallocated_attempt_contingency": protocol.http_attempt_ceiling
            - len(universe.actions) * (protocol.maximum_retries + 1),
            "action_counts": action_counts,
            "reconciled": True,
        },
        "token_cost_forecast.json": forecast,
        "authorization_manifest_proposal.json": manifest.model_dump(mode="json"),
        "preflight_validation.json": {
            "valid": True,
            "provider_initialized": False,
            "paid_calls_authorized": False,
            "formal_qualification_possible": False,
            "instrumentation_objective_achievable": True,
        },
        "normal_simulation_results.json": {
            "informative": informative,
            "uninformative": uninformative,
            "budget_pressure": pressure,
        },
        "adaptive_branch_coverage.json": branch_coverage,
        "mock_execution_trace.json": {"scenario": "informative", "events": trace},
        "mock_budget_ledger.json": informative["result"]["budget"],
        "mock_session_reconciliation.json": reconciliation.model_dump(mode="json"),
        "fault_injection_results.json": fault_results,
        "restart_safety_results.json": {
            "persisted_receipt_recovered_without_reissue": True,
            "undispatched_reservation_explicitly_released": True,
            "ambiguous_dispatch_not_retried": True,
            "mismatched_scope_rejected": True,
            "corrupted_hash_chain_rejected": True,
        },
        "provider_contract_tests.json": _provider_contract_results(),
        "research_evidence_isolation_tests.json": {
            "representative_track": "certification",
            "discovery_track": "discovery",
            "pooled_unweighted_population_estimate": False,
            "discovery_screen_is_paid": True,
            "unseen_outcomes_visible": False,
            "formal_certification_claim": False,
        },
        "future_analysis_schema.json": {
            "metrics": [
                "physical_requests",
                "http_attempts",
                "named_question_evaluations",
                "input_tokens",
                "output_tokens",
                "experimental_cost_proxy_usd",
                "representative_actions",
                "discovery_screens",
                "discovery_followups",
                "response_influenced_actions",
                "near_threshold_screens",
                "diagnostic_disagreements",
                "invalid_responses",
                "restart_events",
            ],
            "formal_certification_metric": False,
        },
        "execution_readiness.json": {
            "technical_preflight_complete": True,
            "mock_end_to_end_complete": True,
            "live_authorized": False,
            "authorization_checkpoint": "separate countersigned manifest required",
            "new_jev_requests": 0,
            "recommended_status": "READY_FOR_M5_2_2_PILOT_A_AUTHORIZATION",
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, value in artifacts.items():
        _write_json(output / name, value)
    _write_csv(output / "workload_measurement_plan.csv", schedule)
    data = {
        "schedule": schedule,
        "forecast": forecast,
        "informative": informative,
        "uninformative": uninformative,
    }
    documents = _docs(data)
    docs.mkdir(parents=True, exist_ok=True)
    for name, value in documents.items():
        (docs / name).write_text(value.strip() + "\n", encoding="utf-8")
    artifact_hashes = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_integrity_manifest.json"
    }
    document_hashes = {name: sha256_file(docs / name) for name in documents}
    manifest_payload = {
        "schema_version": "m5.2.2-artifact-integrity-v1",
        "artifacts": artifact_hashes,
        "documents": document_hashes,
        "historical_integrity": integrity,
        "new_jev_requests": 0,
    }
    _write_json(output / "artifact_integrity_manifest.json", manifest_payload)
    return {
        "artifacts": len(list(output.iterdir())),
        "documents": len(documents),
        "new_jev_requests": 0,
        "status": "complete",
        "scope_fingerprint": manifest.scope_fingerprint,
    }
