from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from semanticflow.optimizer import m53_analysis
from semanticflow.optimizer.m53_analysis import _wilson, generate
from semanticflow.optimizer.m53_evidence import (
    EvidenceRecord,
    Usage,
    coverage_rows,
    load_m523_representative_evidence,
    load_stage_d_evidence,
    m523_discovery_coverage,
    selection_stage_coverage,
)
from semanticflow.optimizer.m53_policies import (
    HybridPolicy,
    PositivePolicy,
    RandomPolicy,
    UncertaintyPolicy,
    VisibleCandidate,
)
from semanticflow.optimizer.m53_replay import replay_conditional, replay_end_to_end

pytestmark = pytest.mark.research_artifacts(
    "results/milestone_5_1/final_research_audit/per_request_expenditure.csv",
    "results/milestone_5_1/stage_d_live_v3/session.json",
    "results/milestone_5_2_3/live_pilot_a_repaired_session_01/representative_measurements.csv",
)


def _record(
    record_id: str,
    separate: tuple[float, float],
    batch: tuple[float, float],
    *,
    screen_cost: float = 1.0,
    followup_cost: float = 2.0,
) -> EvidenceRecord:
    predicates = ("p1", "p2")
    screen_usage = Usage(1, 1, 10, 2, 1, screen_cost)
    followup_usage = Usage(2, 2, 20, 4, 3, followup_cost)
    return EvidenceRecord(
        source="synthetic",
        partition="test",
        workload_id="workload",
        dataset="dataset",
        record_id=record_id,
        predicate_ids=predicates,
        batch_operator_id="batch",
        screen_predicate_id="p1",
        separate_probabilities=dict(zip(predicates, separate, strict=True)),
        batch_probabilities=dict(zip(predicates, batch, strict=True)),
        separate_decisions={
            predicate: probability >= 0.5
            for predicate, probability in zip(predicates, separate, strict=True)
        },
        batch_decisions={
            predicate: probability >= 0.5
            for predicate, probability in zip(predicates, batch, strict=True)
        },
        request_ids=(f"{record_id}-1", f"{record_id}-2", f"{record_id}-b"),
        screen_usage=screen_usage,
        followup_usage=followup_usage,
        full_usage=screen_usage + followup_usage,
    )


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_real_evidence_loading_and_exact_group_compatibility() -> None:
    records = load_stage_d_evidence()
    assert len(records) == 1200
    assert len({record.record_id for record in records}) == 1200
    assert {record.workload_id for record in records} == {
        "m51_banking_card_unauthorized_refund",
        "m51_beauty_failure_usability",
        "m51_sms_prize_urgency_contact",
    }
    assert all(
        set(record.separate_probabilities) == set(record.predicate_ids) for record in records
    )
    assert all(set(record.batch_probabilities) == set(record.predicate_ids) for record in records)
    assert sum(record.full_usage.requests for record in records) == 4400


def test_coverage_keeps_development_representative_and_discovery_isolated() -> None:
    development = selection_stage_coverage()
    representative = coverage_rows(load_m523_representative_evidence())
    discovery = m523_discovery_coverage()
    assert len(development) == 300
    assert len(representative) == 60
    assert len(discovery) == 30
    assert {row["partition"] for row in development} == {"historical_development"}
    assert {row["partition"] for row in representative} == {"representative"}
    assert {row["partition"] for row in discovery} == {"discovery_enriched"}
    assert sum(row["batch_probabilities"] for row in discovery) == 1
    assert not any(row["complete_counterfactual"] for row in discovery)


def test_policy_view_contains_no_hidden_outcome_labels() -> None:
    candidate = VisibleCandidate("r1", "w", 0.49, 0.001)
    assert set(candidate.__dataclass_fields__) == {
        "key",
        "workload_id",
        "screen_probability",
        "followup_cost_usd",
    }
    for policy in (UncertaintyPolicy(), PositivePolicy(), HybridPolicy()):
        assert policy.order((candidate,), 1) == ["r1"]


def test_random_sampling_is_reproducible_and_seed_sensitive() -> None:
    candidates = tuple(VisibleCandidate(f"r{i}", "w", i / 10, 1.0) for i in range(10))
    policy = RandomPolicy()
    assert policy.order(candidates, 7) == policy.order(candidates, 7)
    assert policy.order(candidates, 7) != policy.order(candidates, 8)


def test_disagreement_and_complete_query_positive_classification() -> None:
    false_negative = _record("fn", (0.9, 0.8), (0.9, 0.2))
    false_positive = _record("fp", (0.9, 0.2), (0.9, 0.8))
    agreement = _record("ok", (0.9, 0.8), (0.7, 0.6))
    assert false_negative.predicate_disagreement_count == 1
    assert false_negative.query_disagreement and false_negative.false_negative
    assert false_negative.separate_query_positive
    assert false_positive.query_disagreement and false_positive.false_positive
    assert not false_positive.separate_query_positive
    assert not agreement.query_disagreement and agreement.separate_query_positive


def test_budget_accounting_and_record_deduplication() -> None:
    records = (
        _record("r1", (0.49, 0.8), (0.2, 0.8)),
        _record("r2", (0.1, 0.2), (0.1, 0.2)),
    )
    conditional = replay_conditional(records, RandomPolicy(), 2, (2, 4), "test")
    assert conditional[0].requests == 2
    assert conditional[0].selected_records == 1
    assert conditional[1].requests == 4
    assert conditional[1].selected_records == 2
    assert conditional[1].duplicate_acquisitions == 0
    assert conditional[1].cost_usd == 4.0


def test_end_to_end_active_policy_pays_for_screens() -> None:
    records = tuple(_record(f"r{i}", (0.45 + i / 100, 0.8), (0.2, 0.8)) for i in range(5))
    active = replay_end_to_end(records, UncertaintyPolicy(), 1, (5, 7), "test", 5)
    assert active[0].screening_requests == 5
    assert active[0].selected_records == 0
    assert active[1].screening_requests == 5
    assert active[1].selected_records == 1
    assert active[1].cost_usd == 7.0


def test_undefined_discovery_fraction_remains_none() -> None:
    record = _record("r", (0.1, 0.2), (0.1, 0.2))
    row = replay_conditional((record,), RandomPolicy(), 1, (2,), "test")[0].row()
    assert row["query_disagreement_fraction"] is None
    assert row["false_negative_fraction"] is None
    assert row["cost_per_predicate_disagreement"] is None


def test_sparse_zero_event_interval_does_not_prove_zero() -> None:
    low, high = _wilson(0, 60)
    assert low == 0
    assert high > 0.05


def test_offline_generation_is_deterministic_and_provider_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from semanticflow.providers.jev import JevProvider

    def fail_provider(*args: object, **kwargs: object) -> None:
        raise AssertionError("M5.3 initialized the paid provider")

    monkeypatch.setattr(JevProvider, "__init__", fail_provider)
    monkeypatch.setattr(m53_analysis, "SEEDS", (5301, 5302))
    monkeypatch.setattr(m53_analysis, "ABLATION_SEEDS", (5301,))
    output = tmp_path / "results"
    docs = tmp_path / "docs"
    first = generate(output, docs)
    first_hash = (_tree_hash(output), _tree_hash(docs))
    second = generate(output, docs)
    assert first == second
    assert first_hash == (_tree_hash(output), _tree_hash(docs))
    assert first["new_jev_requests"] == 0
    assert (output / "policy_comparison_results.csv").is_file()
