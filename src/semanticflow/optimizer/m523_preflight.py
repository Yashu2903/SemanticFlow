from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

from semanticflow.optimizer.m51_final_audit import sha256_file
from semanticflow.optimizer.m522_preflight import _protocol_with_budget, _simulate
from semanticflow.optimizer.m522_protocol import (
    PRICE_PER_MILLION_INPUT_TOKENS,
    build_action_universe,
    build_authorization_proposal,
    policy_fingerprint,
)
from semanticflow.optimizer.m523_protocol import LIVE_OUTPUT_DIRECTORY, build_protocol

ROOT = Path("results/milestone_5_2_3/pilot_a_preflight")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def generate(output: Path = ROOT) -> dict[str, Any]:
    """Freeze the M5.2.3 scope and exercise it with mock providers only."""

    protocol = build_protocol()
    universe = build_action_universe(protocol)
    manifest = build_authorization_proposal(protocol, universe)
    expected_input = sum(action.predicted_input_tokens for action in universe.actions)
    expected_output = sum(action.predicted_output_tokens for action in universe.actions)
    with tempfile.TemporaryDirectory(prefix="m523-preflight-") as temporary:
        root = Path(temporary)
        informative, _ = asyncio.run(
            _simulate(protocol, universe, scenario="informative", directory=root / "informative")
        )
        uninformative, _ = asyncio.run(
            _simulate(
                protocol,
                universe,
                scenario="uninformative",
                directory=root / "uninformative",
            )
        )
        pressure_protocol = _protocol_with_budget(protocol, 250)
        pressure, _ = asyncio.run(
            _simulate(
                pressure_protocol,
                universe,
                scenario="budget_pressure",
                directory=root / "budget_pressure",
            )
        )
    artifacts = {
        "frozen_protocol.json": protocol.model_dump(mode="json"),
        "authorized_action_universe.json": universe.model_dump(mode="json"),
        "authorization_manifest_proposal.json": manifest.model_dump(mode="json"),
        "preflight_validation.json": {
            "schema_version": "m5.2.3-preflight-validation-v1",
            "provider_initialized": False,
            "new_jev_requests": 0,
            "paid_calls_authorized": False,
            "live_output_directory": LIVE_OUTPUT_DIRECTORY,
            "live_output_directory_exists": Path(LIVE_OUTPUT_DIRECTORY).exists(),
            "action_count": len(universe.actions),
            "named_question_evaluations": sum(
                action.named_question_evaluations for action in universe.actions
            ),
            "policy_fingerprint": policy_fingerprint(protocol),
        },
        "token_cost_forecast.json": {
            "schema_version": "m5.2.3-token-cost-forecast-v1",
            "forecast_basis": "frozen M5.2.2 rounded Stage D means",
            "expected_input_tokens": expected_input,
            "expected_output_tokens": expected_output,
            "expected_cost_proxy_usd": expected_input / 1_000_000 * PRICE_PER_MILLION_INPUT_TOKENS,
            "prior_m521_expected_cost_proxy_usd": 0.004174105833333334,
            "conservative_cost_proxy_usd": protocol.conservative_cost_proxy_usd,
            "pricing": {
                "input_per_million_usd": PRICE_PER_MILLION_INPUT_TOKENS,
                "output": 0,
            },
            "hard_token_cap": False,
            "independently_verified_invoice": False,
        },
        "mock_scenario_results.json": {
            "informative": informative["result"],
            "uninformative": uninformative["result"],
            "budget_pressure": pressure["result"],
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, value in artifacts.items():
        _write_json(output / name, value)
    hashes = {name: sha256_file(output / name) for name in sorted(artifacts)}
    _write_json(
        output / "artifact_integrity_manifest.json",
        {
            "schema_version": "m5.2.3-preflight-artifact-integrity-v1",
            "artifacts": hashes,
            "new_jev_requests": 0,
        },
    )
    return {
        "status": "READY_FOR_EXPLICIT_PILOT_A_AUTHORIZATION",
        "scope_fingerprint": manifest.scope_fingerprint,
        "artifacts": len(artifacts) + 1,
        "new_jev_requests": 0,
        "live_output_directory_created": Path(LIVE_OUTPUT_DIRECTORY).exists(),
    }
