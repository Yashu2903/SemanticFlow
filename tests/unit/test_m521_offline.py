from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from semanticflow.optimizer.m51_selection_analysis import load_evidence
from semanticflow.optimizer.m52_acquisition import ProfilingBudget
from semanticflow.optimizer.m52_active_profiler import SamplingStrategy, deterministic_record_order
from semanticflow.optimizer.m52_controller import M52ControllerConfig
from semanticflow.optimizer.m52_cost_model import M52CostEstimator
from semanticflow.optimizer.m52_offline import (
    M51,
    _dataset_fingerprints,
    _predictions,
    _quality_evidence,
    _strategy_map,
)
from semanticflow.optimizer.m52_planner import PolicyMode
from semanticflow.optimizer.m52_replay import HistoricalReplayMatrix
from semanticflow.optimizer.m521_controller import M521ControllerConfig, M521Optimizer
from semanticflow.optimizer.m521_error_budget import legacy_requirements
from semanticflow.optimizer.m521_offline import generate

pytestmark = pytest.mark.research_artifacts(
    "results/milestone_5_1/selection_stage_results/frozen_strategy_selections.json",
    "results/milestone_5_1/stage_d_live/session.json",
    "results/milestone_5_2/offline_development/artifact_integrity_manifest.json",
)


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_offline_generation_is_idempotent_and_provider_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from semanticflow.providers.jev import JevProvider

    def fail_provider(*args: object, **kwargs: object) -> None:
        raise AssertionError("M5.2.1 offline generation initialized Jev")

    monkeypatch.setattr(JevProvider, "__init__", fail_provider)
    output = tmp_path / "results"
    docs = tmp_path / "docs"
    first = generate(output, docs)
    first_hash = (tree_hash(output), tree_hash(docs))
    second = generate(output, docs)
    assert first == second
    assert first_hash == (tree_hash(output), tree_hash(docs))
    assert first["new_jev_requests"] == 0
    manifest = json.loads((output / "artifact_integrity_manifest.json").read_text("utf-8"))
    assert manifest["historical_integrity"]["historical_measurements_modified"] is False


def test_required_outputs_are_present(tmp_path: Path) -> None:
    output = tmp_path / "results"
    docs = tmp_path / "docs"
    generate(output, docs)
    assert len(list(output.iterdir())) == 22
    assert len(list(docs.iterdir())) == 4
    banking = json.loads((output / "banking77_feasibility_replay.json").read_text("utf-8"))
    assert banking["sixty_record_pilot_feasible"] is False
    assert banking["feasibility_decision"] == "fallback_before_spend"


def test_feasibility_controller_stops_before_first_acquisition() -> None:
    matrix = HistoricalReplayMatrix.selection()
    normalized = load_evidence()["normalized"]
    estimator = M52CostEstimator(normalized, allowed_partition="selection")
    selections = json.loads((M51 / "frozen_strategy_selections.json").read_text("utf-8"))[
        "strategies"
    ]
    adaptive = _strategy_map(selections, "existing_adaptive")
    full = _strategy_map(selections, "full_multi_decision_batch")
    workload = matrix.workload("m51_banking_card_unauthorized_refund")
    plan = matrix.plans[str(full[workload["id"]]["plan_id"])]
    fallback = matrix.plans[str(adaptive[workload["id"]]["plan_id"])]
    fingerprints = _dataset_fingerprints()
    evidence = _quality_evidence(
        plan, dataset_fingerprint=fingerprints[str(workload["dataset"])]
    )
    fallback_prediction, candidate_prediction = _predictions(
        estimator,
        matrix,
        plan,
        fallback,
        record_count=len(matrix.records(str(workload["dataset"]))),
        dataset_fingerprint=fingerprints[str(workload["dataset"])],
    )
    config = M521ControllerConfig(
        requirements=legacy_requirements(),
        legacy_controller=M52ControllerConfig(
            mode=PolicyMode.STRICT_CERTIFICATION,
            sampling_strategy=SamplingStrategy.HYBRID_ACTIVE,
            budget=ProfilingBudget(
                maximum_physical_requests=1200,
                maximum_cost_usd=0.022361304,
                maximum_records=100,
                maximum_actions=200,
                maximum_certification_records=60,
                maximum_discovery_records=20,
                maximum_positive_enrichment_records=20,
            ),
            representative_checkpoints=(40, 60),
            reuse_horizon=10,
        ),
    )
    records = deterministic_record_order(
        matrix.records(str(workload["dataset"])), 5201
    )
    result = M521Optimizer(config).run(
        matrix=matrix,
        evidence=evidence,
        fallback_prediction=fallback_prediction,
        candidate_prediction=candidate_prediction,
        certification_pool=tuple(records[:60]),
        discovery_pool=tuple(records[60:80]),
        evaluation_record_count=len(records),
        input_tokens_per_pair=800,
        cost_per_pair_usd=0.000034,
    )
    assert result.legacy_result is None
    assert result.stop_reason.startswith("mathematical_feasibility:")
    assert matrix.visibility.acquired_measurement_ids == set()


def test_horizons_are_matched_and_quality_inadmissible_savings_excluded(
    tmp_path: Path,
) -> None:
    output = tmp_path / "results"
    generate(output, tmp_path / "docs")
    import csv

    with (output / "same_horizon_economics.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    m51_r10 = next(
        row
        for row in rows
        if row["policy_name"] == "m51_fixed_sample" and row["reuse_horizon"] == "10"
    )
    m51_r1000 = next(
        row
        for row in rows
        if row["policy_name"] == "m51_fixed_sample" and row["reuse_horizon"] == "1000"
    )
    assert float(m51_r10["net_benefit_usd"]) < 0
    assert float(m51_r1000["net_benefit_usd"]) > 0
    assert float(m51_r1000["quality_admissible_net_benefit_usd"]) < 0
