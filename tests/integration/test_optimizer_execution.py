import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from semanticflow.optimizer.models import OptimizerSettings
from semanticflow.optimizer.workflow import (
    plan_experiment,
    profile_experiment,
    run_offline_experiment,
)
from semanticflow.providers.mock import MockProvider
from semanticflow.query.models import QuerySpec


@pytest.mark.asyncio
async def test_complete_offline_optimizer_experiment(tmp_path) -> None:
    dataset = tmp_path / "reviews.parquet"
    rows = [{"review_id": f"r{index:02}", "review_text": f"review {index}"} for index in range(20)]
    pq.write_table(pa.Table.from_pylist(rows), dataset)
    query = tmp_path / "query.yaml"
    query.write_text(
        f"query_id: optimizer\ndataset_path: {dataset.as_posix()}\n"
        "dataset_version: fixture-v1\nlimit: 20\nthreshold: 0.5\nmax_calls: 100\n"
        "predicates:\n"
        "  - id: broad\n    instructions: Broad?\n"
        "  - id: selective\n    instructions: Selective?\n"
        "  - id: medium\n    instructions: Medium?\n",
        encoding="utf-8",
    )
    probabilities = {}
    for index in range(20):
        text = f"review {index}"
        probabilities[(text, "broad")] = 0.9 if index % 5 else 0.1
        probabilities[(text, "selective")] = 0.9 if index % 5 == 1 else 0.1
        probabilities[(text, "medium")] = 0.9 if index % 2 else 0.1
    provider = MockProvider(
        probabilities,
        usage_by_predicate={
            "broad": (400, 20),
            "selective": (30, 5),
            "medium": (120, 10),
        },
    )
    optimizer = OptimizerSettings.model_validate(
        {
            "split": {"seed": 2027, "profiling_size": 5, "evaluation_size": 15},
            "profiling": {"concurrency": 2, "max_attempts": 15},
            "execution": {"max_attempts_per_run": 45, "cache_enabled": False},
            "planner": {"objective": "estimated_cost_usd"},
        }
    )
    experiment_dir, comparison = await run_offline_experiment(
        query_path=query,
        settings={
            "executor": {"concurrency": 2},
            "pricing": {
                "id": "synthetic-test",
                "input_per_million_usd": 2.0,
                "output_per_million_usd": 4.0,
            },
        },
        optimizer=optimizer,
        provider=provider,
        output_root=tmp_path / "results",
        experiment_id="offline",
    )
    assert comparison["selected_predicate_order"] != comparison[
        "original_predicate_order"
    ]
    candidate_plans = json.loads(
        (experiment_dir / "candidate_plans.json").read_text()
    )["candidates"]
    assert candidate_plans[0]["predicate_order"] == comparison[
        "selected_predicate_order"
    ]
    assert candidate_plans[0]["objective_value"] == min(
        candidate["objective_value"] for candidate in candidate_plans
    )
    assert (
        comparison["optimized"]["completed_decisions"]
        < comparison["fixed"]["completed_decisions"]
    )
    assert comparison["optimized"]["total_tokens"] < comparison["fixed"][
        "total_tokens"
    ]
    assert comparison["optimized"]["estimated_cost_usd"] < comparison["fixed"][
        "estimated_cost_usd"
    ]
    assert comparison["output_equivalence"]["equivalent"]
    assert comparison["fixed"]["completion_status"] == "complete"
    assert comparison["optimized"]["completion_status"] == "complete"
    assert comparison["profiling_overhead"]["requests"] == 15
    assert comparison["prediction_error"]["original"]["observed"] is not None
    assert comparison["prediction_error"]["selected"]["observed"] is not None
    assert comparison["net_objective_savings_after_profiling"] == pytest.approx(
        comparison["execution_only_objective_savings"]
        - comparison["profiling_overhead"]["objective_value"]
    )
    expected = {
        "split_manifest.json",
        "profiling_decisions.jsonl",
        "profiling_report.json",
        "predicate_statistics.json",
        "candidate_plans.json",
        "selected_plan.json",
        "experiment_manifest.json",
        "comparison.json",
        "summary.md",
        "fixed",
        "optimized",
    }
    assert {item.name for item in experiment_dir.iterdir()} == expected
    split = json.loads((experiment_dir / "split_manifest.json").read_text())
    assert len(split["profiling_ids"]) == 5
    assert len(split["evaluation_ids"]) == 15
    assert not set(split["profiling_ids"]) & set(split["evaluation_ids"])
    fixed_manifest = json.loads((experiment_dir / "fixed" / "manifest.json").read_text())
    optimized_manifest = json.loads((experiment_dir / "optimized" / "manifest.json").read_text())
    assert fixed_manifest["evaluation_record_ids"] == optimized_manifest["evaluation_record_ids"]


@pytest.mark.asyncio
async def test_staged_research_layout_writes_profile_and_planning_manifests(
    tmp_path,
) -> None:
    dataset = tmp_path / "reviews.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"review_id": f"r{index}", "review_text": f"review {index}"}
                for index in range(6)
            ]
        ),
        dataset,
    )
    query = QuerySpec.model_validate(
        {
            "query_id": "staged",
            "dataset_path": dataset,
            "limit": 6,
            "threshold": 0.5,
            "max_calls": 20,
            "predicates": [
                {"id": "A", "instructions": "A?"},
                {"id": "B", "instructions": "B?"},
            ],
        }
    )
    optimizer = OptimizerSettings.model_validate(
        {
            "split": {"profiling_size": 2, "evaluation_size": 4},
            "profiling": {"max_attempts": 4},
        }
    )
    experiment_dir = tmp_path / "staged"
    provider = MockProvider(
        input_tokens_per_evaluation=10, output_tokens_per_evaluation=1
    )
    profile, _ = await profile_experiment(
        query=query,
        settings={"executor": {"concurrency": 2}, "pricing": {}},
        optimizer=optimizer,
        provider=provider,
        experiment_dir=experiment_dir,
        research_layout=True,
    )
    assert profile.complete
    plan_experiment(
        query=query,
        optimizer=optimizer,
        experiment_dir=experiment_dir,
        provider_model=provider.model_identifier,
    )
    assert {item.name for item in (experiment_dir / "profiling").iterdir()} == {
        "decisions.jsonl",
        "statistics.json",
        "report.json",
        "manifest.json",
    }
    assert {item.name for item in (experiment_dir / "planning").iterdir()} == {
        "candidate_plans.json",
        "selected_plan.json",
        "report.json",
    }
    planning_report = json.loads(
        (experiment_dir / "planning" / "report.json").read_text()
    )
    assert planning_report["held_out_evaluation_outcomes_used"] is False
    assert planning_report["selected_is_minimum_objective"] is True
