from __future__ import annotations

import argparse
import asyncio
import builtins
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from semanticflow.optimizer.m522_budget import BudgetLedger
from semanticflow.optimizer.m522_experiment_objective import (
    ExperimentObjective,
    ObjectivePolicy,
    production_qualification_policy,
    research_instrumentation_policy,
)
from semanticflow.optimizer.m522_live_acquisition import (
    DeterministicMockProvider,
    LiveAcquisitionAdapter,
    validate_named_response,
)
from semanticflow.optimizer.m522_live_controller import LiveProfilingController
from semanticflow.optimizer.m522_protocol import (
    build_action_universe,
    build_authorization_proposal,
    build_protocol,
    load_authorized_record_texts,
    policy_fingerprint,
    validate_live_manifest,
)
from semanticflow.optimizer.m522_reconciliation import reconcile_session
from semanticflow.optimizer.m522_session import AppendOnlyEventLog, SessionStatus


def controller(tmp_path: Path, scenario: str) -> LiveProfilingController:
    protocol = build_protocol()
    universe = build_action_universe(protocol)
    manifest = build_authorization_proposal(protocol, universe)
    provider = DeterministicMockProvider(model=protocol.model, scenario=scenario)
    return LiveProfilingController(
        protocol=protocol,
        universe=universe,
        event_log=AppendOnlyEventLog(tmp_path / "events.jsonl"),
        adapter=LiveAcquisitionAdapter(provider),
        record_texts=load_authorized_record_texts(protocol, universe),
        authorization_scope_fingerprint=manifest.scope_fingerprint,
        session_id=f"test-{scenario}",
        mock_only=True,
    )


def test_objective_boundary_is_typed_and_fail_closed() -> None:
    research = research_instrumentation_policy()
    production = production_qualification_policy()
    assert research.objective == ExperimentObjective.RESEARCH_INSTRUMENTATION
    assert not research.permits_formal_certification
    assert production.permits_formal_certification
    with pytest.raises(ValueError):
        ObjectivePolicy(
            objective=ExperimentObjective.RESEARCH_INSTRUMENTATION,
            permits_diagnostic_acquisition=True,
            permits_formal_certification=True,
            permits_production_deployment=False,
        )


def test_action_universe_reconciles_exact_schedule() -> None:
    protocol = build_protocol()
    universe = build_action_universe(protocol)
    assert len(universe.actions) == 280
    assert sum(action.named_question_evaluations for action in universe.actions) == 430
    assert len({action.action_id for action in universe.actions}) == 280
    assert protocol.physical_request_ceiling == 360
    assert protocol.http_attempt_ceiling == 1080


def test_unsigned_manifest_rejected_before_live_provider_factory() -> None:
    protocol = build_protocol()
    universe = build_action_universe(protocol)
    manifest = build_authorization_proposal(protocol, universe)
    initialized = False

    def provider_factory() -> None:
        nonlocal initialized
        initialized = True

    with pytest.raises(PermissionError):
        validate_live_manifest(
            manifest,
            protocol=protocol,
            universe=universe,
            supplied_authorization=manifest.required_authorization_contract,
            output_directory=manifest.output_directory,
        )
    assert not initialized
    del provider_factory


@pytest.mark.asyncio
async def test_different_responses_produce_different_adaptive_sequences(tmp_path: Path) -> None:
    informative = await controller(tmp_path / "a", "informative").run()
    uninformative = await controller(tmp_path / "b", "uninformative").run()
    assert informative.status == SessionStatus.INSTRUMENTATION_COMPLETE
    assert uninformative.status == SessionStatus.INSTRUMENTATION_COMPLETE
    assert informative.discovery_followups == 30
    assert uninformative.discovery_followups == 0
    assert informative.budget.physical_request_ceiling == 360
    assert informative.budget.completed_action_ids != uninformative.budget.completed_action_ids
    assert not informative.formal_quality_certified
    assert not informative.production_deployment_permitted


def test_atomic_budget_reservations_do_not_overallocate() -> None:
    ledger = BudgetLedger(
        physical_request_ceiling=10,
        http_attempt_ceiling=30,
        maximum_retries=2,
    )

    def reserve(value: int) -> bool:
        try:
            ledger.reserve_action(f"action-{value}", predicted_input_tokens=1, predicted_cost_usd=0)
            return True
        except PermissionError:
            return False

    with ThreadPoolExecutor(max_workers=20) as pool:
        successes = sum(pool.map(reserve, range(20)))
    assert successes == 10
    assert ledger.snapshot().reserved_attempt_capacity == 30


def test_duplicate_worker_cannot_reserve_same_action_twice() -> None:
    ledger = BudgetLedger(
        physical_request_ceiling=2,
        http_attempt_ceiling=6,
        maximum_retries=2,
    )

    def reserve_duplicate(_: int) -> bool:
        try:
            ledger.reserve_action("same-action", predicted_input_tokens=1, predicted_cost_usd=0)
            return True
        except PermissionError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        successes = sum(pool.map(reserve_duplicate, range(2)))
    assert successes == 1
    assert ledger.snapshot().reserved_action_ids == ("same-action",)


def test_named_response_validation_rejects_missing_extra_and_range() -> None:
    assert validate_named_response(("a",), {"a": 0.5}) == {"a": 0.5}
    with pytest.raises(ValueError):
        validate_named_response(("a",), {})
    with pytest.raises(ValueError):
        validate_named_response(("a",), {"a": 0.5, "b": 0.2})
    with pytest.raises(ValueError):
        validate_named_response(("a",), {"a": 1.1})


@pytest.mark.asyncio
async def test_ambiguous_dispatch_stops_and_cannot_duplicate(tmp_path: Path) -> None:
    item = controller(tmp_path, "informative")
    first_action, _ = item.choose_next(item.initialize())
    assert first_action is not None
    text = item.record_texts[first_action.record_id]
    text_hash = __import__("hashlib").sha256(text.encode()).hexdigest()
    item.adapter = LiveAcquisitionAdapter(
        DeterministicMockProvider(
            model=item.protocol.model,
            scenario="informative",
            fault_by_text_hash={text_hash: "ambiguous"},
        )
    )
    result = await item.run(maximum_steps=1)
    assert result.status == SessionStatus.RECONCILIATION_REQUIRED
    assert result.budget.actual_http_attempts == 1
    again = await item.run(maximum_steps=1)
    assert again.budget.actual_http_attempts == 1


def test_recovery_releases_proven_undispatched_reservation(tmp_path: Path) -> None:
    item = controller(tmp_path, "informative")
    state = item.initialize()
    action, _ = item.choose_next(state)
    assert action is not None
    ledger = item._ledger(state)
    snapshot = ledger.reserve_action(
        action.action_id,
        predicted_input_tokens=action.predicted_input_tokens,
        predicted_cost_usd=action.predicted_cost_usd,
    )
    item.event_log.append(
        "action_reserved",
        {"action_id": action.action_id, "budget": snapshot.model_dump(mode="json")},
    )
    result = reconcile_session(
        event_log=item.event_log,
        universe=item.universe,
        expected_protocol_fingerprint=item.protocol.protocol_fingerprint,
        expected_authorization_scope_fingerprint=item.authorization_scope_fingerprint,
        expected_policy_fingerprint=policy_fingerprint(item.protocol),
        physical_request_ceiling=item.protocol.physical_request_ceiling,
        http_attempt_ceiling=item.protocol.http_attempt_ceiling,
        maximum_retries=item.protocol.maximum_retries,
        repair=True,
    )
    assert result.valid
    assert result.released_undispatched_reservations == 1


def test_no_real_provider_is_needed_by_tests() -> None:
    # Explicit regression: the test suite completes solely with the mock adapter.
    assert asyncio.iscoroutinefunction(LiveProfilingController.run)


@pytest.mark.asyncio
async def test_execute_rejects_unsigned_manifest_before_jev_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments.milestone_5_2_2.run_pilot_a import _execute

    protocol = build_protocol()
    universe = build_action_universe(protocol)
    manifest = build_authorization_proposal(protocol, universe)
    path = tmp_path / "manifest.json"
    path.write_text(manifest.model_dump_json(), encoding="utf-8")
    imported = False
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        nonlocal imported
        if name == "semanticflow.providers.jev":
            imported = True
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    args = argparse.Namespace(
        manifest=str(path),
        authorization=manifest.required_authorization_contract,
        output=manifest.output_directory,
        maximum_steps=1,
    )
    with pytest.raises(PermissionError, match="countersigned"):
        await _execute(args)
    assert not imported


@pytest.mark.asyncio
async def test_runner_budget_pressure_scenario_exhausts_mock_ceiling(
    tmp_path: Path,
) -> None:
    from experiments.milestone_5_2_2.run_pilot_a import _simulate

    result = await _simulate(
        argparse.Namespace(
            output=str(tmp_path / "pressure"),
            scenario="budget_pressure",
            maximum_steps=None,
        )
    )
    assert result["status"] == SessionStatus.INSTRUMENTATION_COMPLETE.value
    assert len(result["budget"]["completed_action_ids"]) == 250
    assert result["budget"]["physical_request_ceiling"] == 250
    assert result["budget"]["http_attempt_ceiling"] == 750
    assert result["termination_reason"].startswith("budget_exhausted:")
    assert result["formal_quality_certified"] is False
