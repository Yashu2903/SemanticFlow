from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from semanticflow.optimizer import m531_live, m531_protocol
from semanticflow.optimizer.m53_analysis import matched_ablation_comparisons
from semanticflow.optimizer.m522_action_universe import (
    AuthorizedActionUniverse,
    universe_fingerprint,
)
from semanticflow.optimizer.m522_live_acquisition import DeterministicMockProvider
from semanticflow.optimizer.m531_analysis import load_fresh_evidence
from semanticflow.optimizer.m531_live import executable_actions, prepare_dry_run, run_live
from semanticflow.optimizer.m531_protocol import build_experiment, canonical_hash


def test_fresh_selection_is_independent_stratified_and_deterministic() -> None:
    first = build_experiment()
    second = build_experiment()
    config, records, universe, plan = first
    assert config["configuration_fingerprint"] == second[0]["configuration_fingerprint"]
    assert universe.universe_fingerprint == second[2].universe_fingerprint
    assert len(records) == 300
    assert len({record.record_id for record in records}) == 300
    assert len({record.text_sha256 for record in records}) == 300
    for workload in config["independence"]["eligibility"].values():
        assert workload["independently_eligible"] >= 100
        assert workload["representative"] == workload["enriched"] == 50
    assert len(plan) == 1100
    assert len(universe.actions) == 1100
    for record in records:
        actions = [action for action in universe.actions if action.record_id == record.record_id]
        assert all(action.record_text_sha256 == record.text_sha256 for action in actions)
        assert {action.requested_model for action in actions} == {"jev-1.14.0"}
        assert sum(len(action.predicate_ids) == 1 for action in actions) in {2, 3}
        assert sum(len(action.predicate_ids) > 1 for action in actions) == 1


def test_representative_random_identities_do_not_depend_on_enrichment_ranking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline, _ = m531_protocol.select_fresh_records()
    representative_ids = {
        record.record_id for record in baseline if record.stratum == "representative_random"
    }
    monkeypatch.setattr(
        m531_protocol,
        "_enrichment_score",
        lambda workload_id, text: float(
            int(hashlib.sha256(f"{workload_id}:{text}".encode()).hexdigest()[:8], 16)
        ),
    )
    reranked, _ = m531_protocol.select_fresh_records()
    reranked_representative_ids = {
        record.record_id for record in reranked if record.stratum == "representative_random"
    }
    assert representative_ids == reranked_representative_ids
    assert len(representative_ids) == 150


def test_default_dry_run_never_constructs_provider(tmp_path: Path) -> None:
    called = False

    def forbidden_provider() -> object:
        nonlocal called
        called = True
        raise AssertionError("provider must not be initialized")

    result = run_live(
        output=tmp_path,
        authorize_paid=False,
        provider_factory=forbidden_provider,
    )
    assert result["paid_requests"] == 0
    assert result["paid_provider_initialized"] is False
    assert not called
    assert not (tmp_path / "execution_events.jsonl").exists()


def test_paid_scope_limits_fail_before_provider_initialization(tmp_path: Path) -> None:
    called = False

    def forbidden_provider() -> object:
        nonlocal called
        called = True
        return object()

    with pytest.raises(PermissionError, match="request limit"):
        run_live(
            output=tmp_path,
            authorize_paid=True,
            provider_factory=forbidden_provider,
            hard_request_limit=1099,
        )
    with pytest.raises(PermissionError, match="attempt limit"):
        run_live(
            output=tmp_path,
            authorize_paid=True,
            provider_factory=forbidden_provider,
            hard_attempt_limit=3299,
        )
    with pytest.raises(PermissionError, match="cost limit"):
        run_live(
            output=tmp_path,
            authorize_paid=True,
            provider_factory=forbidden_provider,
            hard_cost_usd=0.025001,
        )
    assert not called


def test_completed_requests_are_not_executable_and_failed_closed_is_terminal() -> None:
    _, _, universe, _ = build_experiment()
    completed = universe.actions[0].action_id
    state = {"status": "EXECUTING", "completed": {completed}}
    remaining = executable_actions(universe, state)
    assert completed not in {action.action_id for action in remaining}
    assert len(remaining) == len(universe.actions) - 1
    with pytest.raises(PermissionError, match="FAILED_CLOSED"):
        executable_actions(universe, {"status": "FAILED_CLOSED", "completed": set()})


def test_live_resumption_does_not_repeat_completed_paid_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, records, universe, plan = build_experiment()
    record = records[0]
    actions = tuple(action for action in universe.actions if action.record_id == record.record_id)
    mini_universe = AuthorizedActionUniverse(
        experiment_id=universe.experiment_id,
        actions=actions,
        maximum_actions_per_record=len(actions),
        universe_fingerprint=universe_fingerprint(universe.experiment_id, actions),
    )
    mini_config = {
        **config,
        "maximum_unique_requests": len(actions),
        "maximum_http_attempts": len(actions) * 3,
        "request_count": len(actions),
        "action_universe_fingerprint": mini_universe.universe_fingerprint,
    }
    mini_config.pop("configuration_fingerprint")
    mini_config["configuration_fingerprint"] = canonical_hash(mini_config)
    action_ids = {action.action_id for action in actions}
    mini_plan = [row for row in plan if row["action_id"] in action_ids]
    monkeypatch.setattr(
        m531_live,
        "build_experiment",
        lambda: (mini_config, (record,), mini_universe, mini_plan),
    )
    provider_calls = 0

    def provider_factory() -> DeterministicMockProvider:
        nonlocal provider_calls
        provider_calls += 1
        return DeterministicMockProvider(
            model="jev-1.14.0", scenario="uninformative", attempts=3
        )

    first = run_live(
        output=tmp_path,
        authorize_paid=True,
        provider_factory=provider_factory,
        hard_request_limit=len(actions),
        hard_attempt_limit=len(actions) * 3,
    )
    assert first["completed_requests"] == len(actions)
    assert first["http_attempts"] == len(actions) * 3
    assert first["token_priced_cost_usd"] <= 0.025
    events = [
        json.loads(line)
        for line in (tmp_path / "execution_events.jsonl").read_text("utf-8").splitlines()
    ]
    for action in actions:
        types = [
            event["event_type"]
            for event in events
            if event["payload"].get("action_id") == action.action_id
        ]
        assert types.index("provider_response_persisted") < types.index("receipt_persisted")
        reservation = next(
            event["payload"]["reserved_cost_usd"]
            for event in events
            if event["event_type"] == "action_started"
            and event["payload"]["action_id"] == action.action_id
        )
        assert reservation > 0
    second = run_live(
        output=tmp_path,
        authorize_paid=True,
        provider_factory=provider_factory,
        hard_request_limit=len(actions),
        hard_attempt_limit=len(actions) * 3,
    )
    assert second["completed_requests"] == len(actions)
    assert provider_calls == 1


def test_partial_evidence_is_not_marked_fully_paired(tmp_path: Path) -> None:
    prepare_dry_run(tmp_path)
    records, coverage = load_fresh_evidence(tmp_path)
    assert records == ()
    assert coverage["fully_paired_records"] == 0
    assert coverage["partial_records"] == 300


def test_artifact_manifest_hashes_generated_files(tmp_path: Path) -> None:
    result = prepare_dry_run(tmp_path)
    manifest = json.loads((tmp_path / "artifact_integrity_manifest.json").read_text("utf-8"))
    assert manifest["configuration_fingerprint"] == result["configuration_fingerprint"]
    assert "request_plan.csv" in manifest["artifacts"]
    assert manifest["artifacts"]["request_plan.csv"]["bytes"] > 0


def test_matched_ablation_comparison_rejects_missing_baseline() -> None:
    candidate = {
        "setting": "end_to_end",
        "pool": "p",
        "budget_kind": "requests",
        "budget_value": 10,
        "seed": 1,
        "policy": "hybrid_no_cost",
        "predicate_disagreements": 1,
        "query_disagreements": 0,
        "false_negatives": 0,
        "positives": 1,
        "cost_usd": 0.1,
    }
    with pytest.raises(ValueError, match="unmatched"):
        matched_ablation_comparisons([candidate])


def test_matched_ablation_comparison_excludes_surplus_baseline_seeds() -> None:
    def row(policy: str, seed: int, positives: int) -> dict[str, object]:
        return {
            "setting": "end_to_end",
            "pool": "p",
            "budget_kind": "requests",
            "budget_value": 10,
            "seed": seed,
            "policy": policy,
            "predicate_disagreements": 0,
            "query_disagreements": 0,
            "false_negatives": 0,
            "positives": positives,
            "cost_usd": 0.1,
        }

    rows = [
        row("hybrid_weighted", 1, 1),
        row("hybrid_weighted", 2, 9),
        row("hybrid_no_cost", 1, 3),
    ]
    result = matched_ablation_comparisons(rows)[0]
    assert result["matched_seed_count"] == 1
    assert result["unmatched_baseline_seeds_excluded"] == 1
    assert result["positives_difference_mean"] == 2
