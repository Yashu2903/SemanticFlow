from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from semanticflow.optimizer.m522_preflight import generate

pytestmark = pytest.mark.research_artifacts(
    "results/milestone_5_1/selection_stage_results/frozen_strategy_selections.json",
    "results/milestone_5_2/offline_development/artifact_integrity_manifest.json",
    "results/milestone_5_2_1/offline_development/artifact_integrity_manifest.json",
)


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_preflight_generation_is_idempotent_and_unauthorized(tmp_path: Path) -> None:
    output = tmp_path / "results"
    docs = tmp_path / "docs"
    first = generate(output, docs)
    first_hash = (tree_hash(output), tree_hash(docs))
    second = generate(output, docs)
    assert first == second
    assert first_hash == (tree_hash(output), tree_hash(docs))
    assert first["artifacts"] == 24
    assert first["documents"] == 4
    manifest = json.loads((output / "authorization_manifest_proposal.json").read_text())
    assert manifest["paid_calls_authorized"] is False
    assert manifest["countersigned_by"] is None
    assert first["new_jev_requests"] == 0


def test_preflight_reconciles_action_and_attempt_budgets(tmp_path: Path) -> None:
    output = tmp_path / "results"
    generate(output, tmp_path / "docs")
    budget = json.loads((output / "request_budget_reconciliation.json").read_text())
    assert budget["frozen_actions"] == 280
    assert budget["named_question_evaluations"] == 430
    assert budget["reserved_attempts_if_all_actions_execute"] == 840
    assert budget["hard_unique_request_ceiling"] == 360
    assert budget["hard_http_attempt_ceiling"] == 1080
