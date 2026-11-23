from __future__ import annotations

import argparse
import builtins
import json
from pathlib import Path

import pytest

from experiments.milestone_5_2_2.run_pilot_a import _authorize, _execute, _validate
from semanticflow.optimizer.m522_live_acquisition import (
    DeterministicMockProvider,
    LiveAcquisitionAdapter,
)
from semanticflow.optimizer.m522_live_controller import LiveProfilingController
from semanticflow.optimizer.m522_protocol import (
    activate_live_manifest,
    build_action_universe,
    build_authorization_proposal,
    load_authorized_record_texts,
    policy_fingerprint,
    validate_live_manifest,
)
from semanticflow.optimizer.m522_protocol import build_protocol as build_m522_protocol
from semanticflow.optimizer.m522_session import AppendOnlyEventLog, SessionStatus
from semanticflow.optimizer.m523_preflight import generate
from semanticflow.optimizer.m523_protocol import LIVE_OUTPUT_DIRECTORY, build_protocol


def test_m523_scope_versions_only_experiment_identity_and_output() -> None:
    historical = build_m522_protocol()
    current = build_protocol()
    assert historical.output_directory == "results/milestone_5_2_2/pilot_a_live"
    assert current.output_directory == LIVE_OUTPUT_DIRECTORY
    assert current.experiment_id == "semanticflow-m5.2.3-live-pilot-a-instrumentation"
    assert current.workload_ids == historical.workload_ids
    assert current.model == historical.model
    assert current.threshold == historical.threshold
    assert current.physical_request_ceiling == historical.physical_request_ceiling
    assert current.http_attempt_ceiling == historical.http_attempt_ceiling
    assert policy_fingerprint(current) == policy_fingerprint(historical)


def test_m523_preflight_is_unsigned_mock_only_and_does_not_create_live_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = tmp_path / "live_pilot_a"
    monkeypatch.setattr("semanticflow.optimizer.m523_preflight.LIVE_OUTPUT_DIRECTORY", str(live))
    result = generate(tmp_path / "preflight")
    manifest = json.loads(
        (tmp_path / "preflight" / "authorization_manifest_proposal.json").read_text("utf-8")
    )
    scenarios = json.loads(
        (tmp_path / "preflight" / "mock_scenario_results.json").read_text("utf-8")
    )
    assert result["new_jev_requests"] == 0
    assert not live.exists()
    assert manifest["paid_calls_authorized"] is False
    assert manifest["countersigned_by"] is None
    assert scenarios["informative"]["discovery_followups"] == 30
    assert scenarios["uninformative"]["discovery_followups"] == 0
    assert len(scenarios["budget_pressure"]["budget"]["completed_action_ids"]) == 250


@pytest.mark.asyncio
async def test_m523_unsigned_execute_fails_before_provider_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = build_protocol()
    universe = build_action_universe(protocol)
    manifest = build_authorization_proposal(protocol, universe)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    imported = False
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        nonlocal imported
        if name == "semanticflow.optimizer.m522_jev_adapter":
            imported = True
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    args = argparse.Namespace(
        manifest=str(manifest_path),
        authorization=manifest.required_authorization_contract,
        output=manifest.output_directory,
        maximum_steps=1,
    )
    with pytest.raises(PermissionError, match="activated"):
        await _execute(args, build_protocol)
    assert not imported


def test_explicit_authorization_activates_exact_named_scope_without_phrase(
    tmp_path: Path,
) -> None:
    live_output = tmp_path / "live"
    protocol = build_m522_protocol(
        experiment_id="semanticflow-m5.2.3-live-pilot-a-test",
        output_directory=live_output.as_posix(),
    )
    def builder():
        return protocol
    universe = build_action_universe(protocol)
    proposal = build_authorization_proposal(protocol, universe)
    proposal_path = tmp_path / "proposal.json"
    active_path = tmp_path / "active.json"
    proposal_path.write_text(proposal.model_dump_json(), encoding="utf-8")
    result = _authorize(
        argparse.Namespace(
            manifest=str(proposal_path),
            output=str(active_path),
            experiment=protocol.experiment_id,
        ),
        builder,
    )
    active = json.loads(active_path.read_text("utf-8"))
    assert result["authorized"] is True
    assert result["provider_initialized"] is False
    assert active["paid_calls_authorized"] is True
    assert active["countersigned_by"] is None
    assert active["authorization_method"] == "explicit_live_authorization_instruction"
    assert active["authorized_experiment_id"] == protocol.experiment_id
    assert active["authorized_scope_fingerprint"] == proposal.scope_fingerprint
    assert proposal_path.read_text("utf-8") == proposal.model_dump_json()
    assert _validate(
        argparse.Namespace(
            manifest=str(active_path),
            output=protocol.output_directory,
            authorization=None,
        ),
        builder,
    )["valid"]
    assert not live_output.exists()


def test_authorization_rejects_wrong_named_experiment_or_material_scope_change() -> None:
    protocol = build_protocol()
    universe = build_action_universe(protocol)
    proposal = build_authorization_proposal(protocol, universe)
    with pytest.raises(PermissionError, match="named experiment"):
        activate_live_manifest(
            proposal,
            protocol=protocol,
            universe=universe,
            named_experiment="another-experiment",
        )
    changed = protocol.model_copy(update={"physical_request_ceiling": 361})
    with pytest.raises((PermissionError, ValueError)):
        activate_live_manifest(
            proposal,
            protocol=changed,
            universe=universe,
            named_experiment=changed.experiment_id,
        )


def test_historical_countersignature_and_phrase_cannot_replace_activation() -> None:
    protocol = build_protocol()
    universe = build_action_universe(protocol)
    proposal = build_authorization_proposal(protocol, universe)
    historical = proposal.model_copy(
        update={"paid_calls_authorized": True, "countersigned_by": "historical-user"}
    )
    with pytest.raises(PermissionError, match="activation"):
        validate_live_manifest(
            historical,
            protocol=protocol,
            universe=universe,
            supplied_authorization=proposal.required_authorization_contract,
            output_directory=protocol.output_directory,
        )


def test_failed_closed_session_is_never_executable(tmp_path: Path) -> None:
    output = tmp_path / "failed-live"
    protocol = build_m522_protocol(
        experiment_id="semanticflow-m5.2.3-failed-closed-test",
        output_directory=output.as_posix(),
    )
    def builder():
        return protocol
    universe = build_action_universe(protocol)
    proposal = build_authorization_proposal(protocol, universe)
    active = activate_live_manifest(
        proposal,
        protocol=protocol,
        universe=universe,
        named_experiment=protocol.experiment_id,
    )
    manifest_path = tmp_path / "active.json"
    manifest_path.write_text(active.model_dump_json(), encoding="utf-8")
    controller = LiveProfilingController(
        protocol=protocol,
        universe=universe,
        event_log=AppendOnlyEventLog(output / "events.jsonl"),
        adapter=LiveAcquisitionAdapter(
            DeterministicMockProvider(model=protocol.model, scenario="informative")
        ),
        record_texts=load_authorized_record_texts(protocol, universe),
        authorization_scope_fingerprint=active.scope_fingerprint,
        session_id="failed-closed-test",
        mock_only=False,
    )
    controller.initialize()
    controller.event_log.append(
        "status_changed",
        {"status": SessionStatus.FAILED_CLOSED.value, "reason": "test"},
    )

    with pytest.raises(PermissionError, match="FAILED_CLOSED"):
        _validate(
            argparse.Namespace(
                manifest=str(manifest_path),
                output=protocol.output_directory,
                authorization=None,
            ),
            builder,
        )
