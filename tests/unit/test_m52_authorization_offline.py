from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from semanticflow.optimizer.m52_authorization import (
    AtomicAttemptBudget,
    M52LivePilotScope,
    schedule_fingerprint,
    validate_live_authorization,
)
from semanticflow.optimizer.m52_offline import generate


def scope(schedule: list[dict[str, object]]) -> M52LivePilotScope:
    fingerprint = schedule_fingerprint(schedule)
    return M52LivePilotScope(
        scope_fingerprint="scope",
        authorization_phrase="AUTHORIZE M5.2 SMALL LIVE PILOT scope",
        schedule_fingerprint=fingerprint,
        physical_request_ceiling=len(schedule),
        retry_inclusive_attempt_ceiling=len(schedule) * 3,
    )


def test_paid_execution_requires_exact_new_authorization() -> None:
    schedule = [{"request_id": "one"}]
    with pytest.raises(PermissionError, match="exact"):
        validate_live_authorization(
            scope(schedule),
            supplied_authorization=None,
            schedule=schedule,
            requested_model="jev-1.14.0",
            requested_maximum_retries=2,
        )


def test_historical_authorization_reuse_is_rejected() -> None:
    schedule = [{"request_id": "one"}]
    historical = scope(schedule).model_copy(update={"historical_authorization": True})
    with pytest.raises(PermissionError, match="cannot be reused"):
        validate_live_authorization(
            historical,
            supplied_authorization=historical.authorization_phrase,
            schedule=schedule,
            requested_model="jev-1.14.0",
            requested_maximum_retries=2,
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"requested_model": "other-model"}, "model differs"),
        ({"requested_maximum_retries": 3}, "retry policy"),
        ({"schedule": [{"request_id": "changed"}]}, "fingerprint mismatch"),
    ],
)
def test_scope_changes_fail_closed(change: dict[str, object], message: str) -> None:
    schedule = [{"request_id": "one"}]
    arguments = {
        "scope": scope(schedule),
        "supplied_authorization": "AUTHORIZE M5.2 SMALL LIVE PILOT scope",
        "schedule": schedule,
        "requested_model": "jev-1.14.0",
        "requested_maximum_retries": 2,
        **change,
    }
    with pytest.raises(PermissionError, match=message):
        validate_live_authorization(**arguments)


def test_atomic_attempt_budget_never_exceeds_ceiling() -> None:
    budget = AtomicAttemptBudget(3)
    assert budget.reserve(2) == 2
    assert budget.reserve() == 3
    with pytest.raises(PermissionError, match="exhausted"):
        budget.reserve()
    assert budget.reserved == 3


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@pytest.mark.research_artifacts(
    "results/milestone_5_1/selection_stage_results/frozen_strategy_selections.json",
    "results/milestone_5_1/stage_d_live/session.json",
)
def test_offline_generation_is_idempotent_and_provider_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from semanticflow.providers.jev import JevProvider

    def fail_provider(*args: object, **kwargs: object) -> None:
        raise AssertionError("offline M5.2 initialized the provider")

    monkeypatch.setattr(JevProvider, "__init__", fail_provider)
    output = tmp_path / "results"
    docs = tmp_path / "docs"
    first_result = generate(output, docs)
    first = (tree_hash(output), tree_hash(docs))
    second_result = generate(output, docs)
    second = (tree_hash(output), tree_hash(docs))
    assert first_result["new_jev_requests"] == 0
    assert second_result["new_jev_requests"] == 0
    assert first == second
    manifest = json.loads((output / "artifact_integrity_manifest.json").read_text("utf-8"))
    assert manifest["provider_initialized"] is False
    assert manifest["historical_m51_artifacts_modified"] is False
