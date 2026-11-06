import json
from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from semanticflow import cli
from semanticflow.optimizer.benchmark import run_session
from semanticflow.optimizer.models import OptimizerSettings
from semanticflow.optimizer.workflow import (
    execute_experiment_strategy,
    plan_experiment,
    profile_experiment,
)
from semanticflow.providers.mock import MockProvider
from semanticflow.query.models import load_query

OPTIMIZER = {
    "split": {"seed": 2027, "profiling_size": 30, "evaluation_size": 30},
    "profiling": {"concurrency": 2, "max_attempts": 90},
    "execution": {"max_attempts_per_run": 90, "cache_enabled": False},
    "planner": {"objective": "estimated_cost_usd"},
}
SETTINGS = {
    "executor": {"concurrency": 2},
    "pricing": {
        "id": "synthetic-test",
        "input_per_million_usd": 1.0,
        "output_per_million_usd": 0.0,
    },
}


def provider() -> MockProvider:
    probabilities = {}
    for index in range(60):
        text = f"review {index}"
        shared = index % 4 == 0
        probabilities[(text, "broad")] = 0.9 if shared or index % 3 == 0 else 0.1
        probabilities[(text, "twin")] = 0.9 if shared else 0.1
        probabilities[(text, "costly")] = 0.9 if index % 2 == 0 else 0.1
    return MockProvider(
        probabilities,
        usage_by_predicate={"broad": (40, 1), "twin": (30, 1), "costly": (300, 1)},
    )


@pytest.fixture
async def experiment(tmp_path):
    dataset = tmp_path / "reviews.parquet"
    rows = [{"review_id": f"r{index:02}", "review_text": f"review {index}"} for index in range(60)]
    pq.write_table(pa.Table.from_pylist(rows), dataset)
    query_path = tmp_path / "query.yaml"
    query_path.write_text(
        f"query_id: m3\ndataset_path: {dataset.as_posix()}\n"
        "dataset_version: fixture-v1\nlimit: 60\nthreshold: 0.5\nmax_calls: 200\n"
        "predicates:\n"
        "  - id: broad\n    instructions: Broad?\n"
        "  - id: twin\n    instructions: Twin?\n"
        "  - id: costly\n    instructions: Costly?\n",
        encoding="utf-8",
    )
    query = load_query(query_path)
    optimizer = OptimizerSettings.model_validate(OPTIMIZER)
    experiment_dir = tmp_path / "experiment"
    mock = provider()
    await profile_experiment(
        query=query,
        settings=SETTINGS,
        optimizer=optimizer,
        provider=mock,
        experiment_dir=experiment_dir,
        research_layout=True,
    )
    plan_experiment(
        query=query,
        optimizer=optimizer,
        experiment_dir=experiment_dir,
        provider_model=mock.model_identifier,
    )
    for strategy in ("optimized", "fixed"):
        await execute_experiment_strategy(
            strategy=strategy,
            query=query,
            settings=SETTINGS,
            optimizer=optimizer,
            provider=mock,
            experiment_dir=experiment_dir,
        )
    analysis_config = tmp_path / "optimizer.m3.yaml"
    analysis_config.write_text(
        yaml.safe_dump(
            {
                **OPTIMIZER,
                "planner": {
                    "objective": "estimated_cost_usd",
                    "cardinality_estimator": "correlation_aware",
                    "cost_model_version": "correlation-aware-and-v1",
                },
                "bootstrap": {"iterations": 40, "seed": 5},
                "adaptive": {"checkpoints": [10, 20, 30], "bootstrap_iterations": 20},
            }
        ),
        encoding="utf-8",
    )
    return {
        "tmp": tmp_path,
        "query": query_path,
        "experiment": experiment_dir,
        "config": analysis_config,
        "model": mock.model_identifier,
    }


@pytest.fixture
def no_provider(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("offline commands must not construct a provider")

    monkeypatch.setattr(cli, "_provider", refuse)
    monkeypatch.setattr(cli.JevProvider, "from_env", refuse)


async def test_analyze_cli_runs_offline_and_reproduces_stored_runs(experiment, no_provider):
    output = experiment["tmp"] / "analysis"
    arguments = [
        "analyze",
        "--query", str(experiment["query"]),
        "--optimizer-config", str(experiment["config"]),
        "--experiment-dir", str(experiment["experiment"]),
        "--historical-baseline-dir", str(experiment["experiment"] / "fixed"),
        "--output-dir", str(output),
    ]
    assert cli.main(arguments) == 0
    expected = {
        "manifest.json",
        "counterfactual.json",
        "joint_statistics.json",
        "cardinality.json",
        "calibration.json",
        "planning.json",
        "bootstrap.json",
        "economics.json",
        "adaptive.json",
        "summary.md",
    }
    assert expected <= {item.name for item in output.iterdir()}
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["live_provider_calls"] == 0
    counterfactual = json.loads((output / "counterfactual.json").read_text())
    fixed_report = json.loads((experiment["experiment"] / "fixed" / "report.json").read_text())
    optimized_report = json.loads(
        (experiment["experiment"] / "optimized" / "report.json").read_text()
    )
    checks = counterfactual["reproduction"]
    fixed_replay = checks["fixed_from_trace"]["logical_evaluations"]
    optimized_replay = checks["optimized_from_trace"]["logical_evaluations"]
    assert fixed_replay["lower"] == fixed_replay["upper"] == fixed_report["completed_decisions"]
    assert optimized_replay["lower"] == optimized_report["completed_decisions"]
    assert checks["fixed_from_trace"]["final_match_ids_equal_recorded"] is True
    assert checks["optimized_from_trace"]["final_match_ids_equal_recorded"] is True
    # A deterministic mock has no response variability, so the replay estimate accounts for
    # the whole observed difference under every resolution policy.
    observed = fixed_report["completed_decisions"] - optimized_report["completed_decisions"]
    assert counterfactual["evidence"]["causal_measurement"] is False
    for item in counterfactual["decomposition"]:
        assert item["evidence_kind"] == "offline_replay_estimate"
        assert item["causal_measurement"] is False
        assert item["replay_estimated_ordering_effect_evaluations"]["lower"] == observed
        assert item["replay_estimated_ordering_effect_evaluations"]["upper"] == observed
        assert item["residual_not_explained_by_replay_evaluations"] == {"lower": 0, "upper": 0}
    cardinality = json.loads((output / "cardinality.json").read_text())
    assert cardinality["held_out_outcomes_used_for_fitting"] is False
    calibration = json.loads((output / "calibration.json").read_text())
    assert calibration["held_out_outcomes_used_for_fitting"] is False
    assert set(calibration["held_out"]["estimators"]) == {"independent", "correlation_aware"}
    assert calibration["prior_strength_sensitivity_cv"]["selection_uses_held_out"] is False
    planning = json.loads((output / "planning.json").read_text())
    assert planning["independent_reproduces_milestone_2_candidates"] is True

    economics = json.loads((output / "economics.json").read_text())
    break_even = economics["break_even"]
    assert break_even["observed"]["savings_basis"] == "observed_confounded"
    assert break_even["replay_estimated"]["savings_basis"] == "replay_estimate"
    assert break_even["predicted"]["savings_basis"] == "predicted"
    assert all(
        break_even[kind]["causal_measurement"] is False
        for kind in ("observed", "replay_estimated", "predicted")
    )
    assert "replay_measured_ordering_savings_usd" not in economics

    adaptive = json.loads((output / "adaptive.json").read_text())
    for runs in adaptive["runs"].values():
        skipped = runs["0"]
        assert skipped["profiling_started"] is False
        assert skipped["start_decision"]["reasons"] == ["no_expected_executions"]
        assert skipped["profiling_decisions_used"] == 0
        assert skipped["trajectory"] == []
    assert cli.main(arguments) == 1


async def test_replay_cli_reproduces_a_stored_run_without_a_provider(
    experiment, no_provider, capsys, caplog
):
    fixed_dir = experiment["experiment"] / "fixed"
    fixed_report = json.loads((fixed_dir / "report.json").read_text())
    output = experiment["tmp"] / "replay.json"
    arguments = [
        "replay",
        "--query", str(experiment["query"]),
        "--decisions", f"fixed={fixed_dir / 'decisions.jsonl'}",
        "--split-manifest", str(experiment["experiment"] / "split_manifest.json"),
        "--split-part", "evaluation",
        "--model", experiment["model"],
        "--order", "broad,twin,costly",
        "--output", str(output),
    ]
    assert cli.main(arguments) == 0
    report = json.loads(output.read_text())
    assert report["observability"] == "fully_observable"
    assert report["logical_evaluations"]["lower"] == fixed_report["completed_decisions"]
    assert report["evidence"]["causal_measurement"] is False
    capsys.readouterr()
    assert cli.main(arguments) == 1
    assert "Refusing to overwrite" in caplog.text
    compare = [*arguments[:-2], "--order", "costly,twin,broad"]
    assert cli.main(compare) == 0
    printed = capsys.readouterr().out
    assert "Evaluation difference" in printed
    assert "not a causal measurement" in printed
    compared = json.loads(printed[printed.index("\n{") + 1 :])
    assert compared["evidence"]["evidence_kind"] == "offline_replay_estimate"
    assert "difference_is_exact" not in compared


async def test_plan_experiment_with_correlation_aware_estimator(experiment) -> None:
    source = experiment["experiment"]
    copy = experiment["tmp"] / "replan"
    (copy / "profiling").mkdir(parents=True)
    for name in ("decisions.jsonl", "report.json", "statistics.json", "manifest.json"):
        (copy / "profiling" / name).write_bytes((source / "profiling" / name).read_bytes())
    (copy / "split_manifest.json").write_bytes((source / "split_manifest.json").read_bytes())
    optimizer = OptimizerSettings.model_validate(
        yaml.safe_load(experiment["config"].read_text(encoding="utf-8"))
    )
    selection = plan_experiment(
        query=load_query(experiment["query"]),
        optimizer=optimizer,
        experiment_dir=copy,
        provider_model=experiment["model"],
    )
    selected = json.loads((copy / "planning" / "selected_plan.json").read_text())
    report = json.loads((copy / "planning" / "report.json").read_text())
    assert selected["cardinality_estimator"] == "correlation_aware"
    assert selected["joint_matrix_fingerprint"] == selection.joint_matrix_fingerprint
    assert selected["joint_matrix_fingerprint"].startswith("sha256:")
    assert report["held_out_evaluation_outcomes_used"] is False
    assert len(report["selected_plan_stage_breakdown"]) == 3


async def test_benchmark_session_pairs_orders_and_offline_cli_never_builds_a_provider(
    experiment, no_provider, capsys
):
    root = experiment["tmp"] / "benchmark"
    strategies_path = root / "strategies.json"
    schedule_path = root / "schedule.json"
    session_dir = root / "session"
    analysis_path = root / "analysis.json"
    common = [
        "--query", str(experiment["query"]),
        "--optimizer-config", str(experiment["config"]),
        "--experiment-dir", str(experiment["experiment"]),
    ]
    assert cli.main(
        [
            "benchmark-plan",
            *common,
            "--output", str(strategies_path),
            "--prior-cost-per-decision", "0.00001",
            "--prior-cost-source", "unit-test",
        ]
    ) == 0
    strategies = json.loads(strategies_path.read_text())
    assert strategies["held_out_outcomes_used"] is False
    assert cli.main(
        [
            "benchmark-schedule",
            "--strategies", str(strategies_path),
            "--output", str(schedule_path),
            "--orders", "selected",
        ]
    ) == 0
    schedule = json.loads(schedule_path.read_text())
    assert len(schedule["runs"]) == 4
    assert schedule["runs"][0]["order"] != schedule["runs"][1]["order"]
    assert cli.main(
        [
            "benchmark-preflight",
            "--strategies", str(strategies_path),
            "--schedule", str(schedule_path),
            "--optimizer-config", str(experiment["config"]),
            "--cost-per-decision", "0.00001",
        ]
    ) == 0
    printed = capsys.readouterr().out
    assert "network_requests_made" in printed
    manifest = await run_session(
        query=load_query(experiment["query"]),
        settings=SETTINGS,
        optimizer=OptimizerSettings.model_validate(
            yaml.safe_load(experiment["config"].read_text(encoding="utf-8"))
        ),
        provider=provider(),
        profile_dir=experiment["experiment"],
        schedule=schedule,
        schedule_path=schedule_path,
        session_dir=session_dir,
        max_session_attempts=400,
    )
    assert manifest["status"] == "complete"
    assert manifest["cache_enabled"] is False
    assert len(manifest["runs"]) == 4
    for first, second in zip(manifest["runs"], manifest["runs"][1:], strict=False):
        gap = datetime.fromisoformat(second["started_at"]) - datetime.fromisoformat(
            first["finished_at"]
        )
        assert gap.total_seconds() >= 0
    assert cli.main(
        [
            "benchmark-analyze",
            "--workload", "synthetic",
            "--strategies", str(strategies_path),
            "--session-dir", str(session_dir),
            "--output", str(analysis_path),
            "--iterations", "40",
        ]
    ) == 0
    analysis = json.loads(analysis_path.read_text())
    assert analysis["evidence"]["causal_measurement"] is True
    assert analysis["abba"] is not None
    assert analysis["strategies"]["fixed"]["executed"] is True
    assert cli.main(
        [
            "benchmark-compare",
            "--analysis", str(analysis_path),
            "--output", str(root / "pooled.json"),
        ]
    ) == 0
    assert cli.main(
        [
            "benchmark-plan",
            *common,
            "--output", str(strategies_path),
            "--prior-cost-per-decision", "0.00001",
            "--prior-cost-source", "unit-test",
        ]
    ) == 1
