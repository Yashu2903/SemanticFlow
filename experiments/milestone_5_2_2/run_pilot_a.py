from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from semanticflow.optimizer.m522_analysis import analyze_session
from semanticflow.optimizer.m522_live_acquisition import (
    DeterministicMockProvider,
    LiveAcquisitionAdapter,
)
from semanticflow.optimizer.m522_live_controller import LiveProfilingController
from semanticflow.optimizer.m522_preflight import generate
from semanticflow.optimizer.m522_protocol import (
    PilotAAuthorizationManifest,
    PilotAProtocol,
    activate_live_manifest,
    build_action_universe,
    build_authorization_proposal,
    build_protocol,
    load_authorized_record_texts,
    policy_fingerprint,
    protocol_fingerprint,
    validate_live_manifest,
)
from semanticflow.optimizer.m522_reconciliation import reconcile_session
from semanticflow.optimizer.m522_session import AppendOnlyEventLog, SessionStatus, rebuild_session


def _json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text("utf-8"))


ProtocolBuilder = Callable[[], PilotAProtocol]


def _scope(protocol_builder: ProtocolBuilder = build_protocol) -> tuple[PilotAProtocol, Any]:
    protocol = protocol_builder()
    return protocol, build_action_universe(protocol)


def _validate(
    args: argparse.Namespace, protocol_builder: ProtocolBuilder = build_protocol
) -> dict[str, Any]:
    protocol, universe = _scope(protocol_builder)
    manifest = PilotAAuthorizationManifest.model_validate(_json(args.manifest))
    validate_live_manifest(
        manifest,
        protocol=protocol,
        universe=universe,
        supplied_authorization=getattr(args, "authorization", None),
        output_directory=args.output,
    )
    output = Path(args.output)
    if output.exists():
        event_path = output / "events.jsonl"
        if not event_path.exists():
            raise FileExistsError("live output exists without a reconcilable session log")
        state = rebuild_session(AppendOnlyEventLog(event_path).read())
        expected = {
            "protocol": protocol.protocol_fingerprint,
            "authorization": manifest.scope_fingerprint,
            "policy": policy_fingerprint(protocol),
            "universe": universe.universe_fingerprint,
            "objective": protocol.objective.objective.value,
            "mock_only": False,
        }
        actual = {
            "protocol": state.protocol_fingerprint,
            "authorization": state.authorization_scope_fingerprint,
            "policy": state.policy_fingerprint,
            "universe": state.action_universe_fingerprint,
            "objective": state.objective,
            "mock_only": state.mock_only,
        }
        if actual != expected or state.pending_action_ids or state.ambiguous_action_ids:
            raise PermissionError("existing live session requires safe reconciliation")
        if state.status == SessionStatus.FAILED_CLOSED:
            raise PermissionError("FAILED_CLOSED session cannot execute")
    return {
        "valid": True,
        "scope_fingerprint": manifest.scope_fingerprint,
        "provider_initialized": False,
    }


def _authorize(
    args: argparse.Namespace, protocol_builder: ProtocolBuilder = build_protocol
) -> dict[str, Any]:
    protocol, universe = _scope(protocol_builder)
    source = Path(args.manifest)
    output = Path(args.output)
    if source.resolve() == output.resolve():
        raise PermissionError("active authorization must not overwrite the unsigned proposal")
    if output.exists():
        raise FileExistsError("active authorization manifest already exists")
    proposal = PilotAAuthorizationManifest.model_validate(_json(source))
    manifest = activate_live_manifest(
        proposal,
        protocol=protocol,
        universe=universe,
        named_experiment=args.experiment,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(manifest.model_dump_json(indent=2) + "\n")
    return {
        "authorized": True,
        "experiment_id": manifest.authorized_experiment_id,
        "scope_fingerprint": manifest.authorized_scope_fingerprint,
        "manifest": output.as_posix(),
        "provider_initialized": False,
    }


async def _simulate(
    args: argparse.Namespace, protocol_builder: ProtocolBuilder = build_protocol
) -> dict[str, Any]:
    protocol, universe = _scope(protocol_builder)
    if args.scenario == "budget_pressure":
        payload = protocol.model_dump(mode="json", exclude={"protocol_fingerprint"})
        payload.update(
            {
                "physical_request_ceiling": 250,
                "http_attempt_ceiling": 750,
                "expected_physical_requests": 250,
            }
        )
        protocol = PilotAProtocol(
            **payload,
            protocol_fingerprint=protocol_fingerprint(payload),
        )
    output = Path(args.output)
    live_output = Path(protocol.output_directory).resolve()
    if output.resolve() == live_output:
        raise PermissionError("mock simulation cannot use the live output directory")
    manifest = build_authorization_proposal(protocol, universe)
    controller = LiveProfilingController(
        protocol=protocol,
        universe=universe,
        event_log=AppendOnlyEventLog(output / "events.jsonl"),
        adapter=LiveAcquisitionAdapter(
            DeterministicMockProvider(model=protocol.model, scenario=args.scenario)
        ),
        record_texts=load_authorized_record_texts(protocol, universe),
        authorization_scope_fingerprint=manifest.scope_fingerprint,
        session_id=f"m522-cli-mock-{args.scenario}",
        mock_only=True,
    )
    result = await controller.run(maximum_steps=args.maximum_steps)
    return result.model_dump(mode="json")


async def _execute(
    args: argparse.Namespace, protocol_builder: ProtocolBuilder = build_protocol
) -> dict[str, Any]:
    # Authorization and scope validation deliberately precede the paid-provider import.
    validation = _validate(args, protocol_builder)
    protocol, universe = _scope(protocol_builder)
    from semanticflow.optimizer.m522_jev_adapter import PilotAJevProvider

    provider = PilotAJevProvider.from_env(
        protocol.model,
        max_retries=protocol.maximum_retries,
    )
    try:
        controller = LiveProfilingController(
            protocol=protocol,
            universe=universe,
            event_log=AppendOnlyEventLog(Path(args.output) / "events.jsonl"),
            adapter=LiveAcquisitionAdapter(provider),
            record_texts=load_authorized_record_texts(protocol, universe),
            authorization_scope_fingerprint=validation["scope_fingerprint"],
            session_id="m522-pilot-a-live",
            mock_only=False,
        )
        result = await controller.run(maximum_steps=args.maximum_steps)
        return result.model_dump(mode="json")
    finally:
        await provider.aclose()


def _status(args: argparse.Namespace) -> dict[str, Any]:
    log = AppendOnlyEventLog(Path(args.output) / "events.jsonl")
    return rebuild_session(log.read()).model_dump(mode="json")


def _reconcile(
    args: argparse.Namespace, protocol_builder: ProtocolBuilder = build_protocol
) -> dict[str, Any]:
    protocol, universe = _scope(protocol_builder)
    manifest = build_authorization_proposal(protocol, universe)
    result = reconcile_session(
        event_log=AppendOnlyEventLog(Path(args.output) / "events.jsonl"),
        universe=universe,
        expected_protocol_fingerprint=protocol.protocol_fingerprint,
        expected_authorization_scope_fingerprint=manifest.scope_fingerprint,
        expected_policy_fingerprint=policy_fingerprint(protocol),
        physical_request_ceiling=protocol.physical_request_ceiling,
        http_attempt_ceiling=protocol.http_attempt_ceiling,
        maximum_retries=protocol.maximum_retries,
        repair=args.repair,
    )
    return result.model_dump(mode="json")


def _analyze(
    args: argparse.Namespace, protocol_builder: ProtocolBuilder = build_protocol
) -> dict[str, Any]:
    protocol = protocol_builder()
    return analyze_session(
        AppendOnlyEventLog(Path(args.output) / "events.jsonl"),
        threshold=protocol.threshold,
    ).model_dump(mode="json")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight")
    authorize = commands.add_parser("authorize")
    authorize.add_argument("--manifest", required=True)
    authorize.add_argument("--output", required=True)
    authorize.add_argument("--experiment", required=True)
    simulate = commands.add_parser("simulate")
    simulate.add_argument("--output", required=True)
    simulate.add_argument(
        "--scenario",
        choices=("informative", "uninformative", "budget_pressure"),
        default="informative",
    )
    simulate.add_argument("--maximum-steps", type=int)
    validate = commands.add_parser("validate")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--authorization")
    validate.add_argument("--output", required=True)
    status = commands.add_parser("status")
    status.add_argument("--output", required=True)
    reconcile = commands.add_parser("reconcile")
    reconcile.add_argument("--output", required=True)
    reconcile.add_argument("--repair", action="store_true")
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--output", required=True)
    execute = commands.add_parser("execute")
    execute.add_argument("--manifest", required=True)
    execute.add_argument("--authorization")
    execute.add_argument("--output", required=True)
    execute.add_argument("--maximum-steps", type=int)
    return value


def main(
    *,
    protocol_builder: ProtocolBuilder = build_protocol,
    preflight_generator: Callable[[], dict[str, Any]] = generate,
) -> int:
    args = parser().parse_args()
    if args.command == "preflight":
        result = preflight_generator()
    elif args.command == "authorize":
        result = _authorize(args, protocol_builder)
    elif args.command == "simulate":
        result = asyncio.run(_simulate(args, protocol_builder))
    elif args.command == "validate":
        result = _validate(args, protocol_builder)
    elif args.command == "status":
        result = _status(args)
    elif args.command == "reconcile":
        result = _reconcile(args, protocol_builder)
    elif args.command == "analyze":
        result = _analyze(args, protocol_builder)
    else:
        result = asyncio.run(_execute(args, protocol_builder))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
