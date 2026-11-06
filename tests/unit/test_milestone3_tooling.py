import argparse
import json

import pytest

from experiments.milestone_3.prepare_strategy_experiment import REQUIRED
from experiments.milestone_3.prepare_strategy_experiment import run as prepare
from semanticflow.optimizer.models import load_optimizer_settings


def test_milestone_3_configs_load_with_explicit_estimators() -> None:
    correlation = load_optimizer_settings("configs/optimizer.milestone_3.yaml")
    confidence = load_optimizer_settings("configs/optimizer.milestone_3_confidence.yaml")
    baseline = load_optimizer_settings("configs/optimizer.milestone_2_jev.yaml")
    assert correlation.planner.cardinality_estimator == "correlation_aware"
    assert correlation.planner.selection_policy == "minimum_expected_objective"
    assert confidence.planner.selection_policy == "confidence_aware"
    assert baseline.planner.cardinality_estimator == "independent"
    assert correlation.economics.expected_executions is None
    assert correlation.economics.horizon() == 1
    assert correlation.bootstrap.iterations == 1000
    assert correlation.adaptive.checkpoints == [25, 50, 100, 150, 200]


def test_prepare_strategy_experiment_copies_without_touching_the_source(tmp_path) -> None:
    source = tmp_path / "source"
    for name in REQUIRED:
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        (source / name).write_text(f"{name}\n", encoding="utf-8")
    before = {name: (source / name).read_bytes() for name in REQUIRED}
    destination = tmp_path / "copy"
    arguments = argparse.Namespace(source=source, destination=destination, copy_planning=False)
    assert prepare(arguments) == 0
    provenance = json.loads((destination / "provenance.json").read_text())
    assert set(provenance["copied_files_sha256"]) == set(REQUIRED)
    assert provenance["network_requests_made"] == 0
    assert {name: (source / name).read_bytes() for name in REQUIRED} == before
    with pytest.raises(FileExistsError):
        prepare(arguments)
    with pytest.raises(FileNotFoundError, match="planning"):
        prepare(
            argparse.Namespace(
                source=source, destination=tmp_path / "other", copy_planning=True
            )
        )
