from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from semanticflow.optimizer.models import OptimizerSettings
from semanticflow.optimizer.planner import predicate_hash
from semanticflow.optimizer.workflow import run_offline_experiment
from semanticflow.providers.mock import MockProvider
from semanticflow.query.models import load_query

QUERY_DIR = Path("configs/queries")
WORKLOADS = sorted(QUERY_DIR.glob("*.yaml"))
PRODUCTION_DATASET = "data/processed/reviews_1000.parquet"


def test_all_workload_configs_are_discovered() -> None:
    names = {path.stem for path in WORKLOADS}
    assert {"product_quality", "listing_mismatch", "positive_experience", "mixed_feedback"} <= names


@pytest.mark.parametrize("path", WORKLOADS, ids=lambda path: path.stem)
def test_workload_config_is_well_formed(path: Path) -> None:
    query = load_query(path)
    ids = [item.id for item in query.predicates]
    assert 2 <= len(ids) <= 6
    assert len(ids) == len(set(ids))
    assert str(query.dataset_path).replace("\\", "/") == PRODUCTION_DATASET
    assert query.limit == 1000
    if path.stem != "milestone_2_controlled":
        # Semantic workloads ask yes/no questions; the controlled config is mock-only.
        assert all(item.instructions.strip().endswith("?") for item in query.predicates)


def test_listing_mismatch_reuses_the_product_quality_dissatisfaction_definition() -> None:
    def hashes(name: str) -> dict[str, str]:
        query = load_query(QUERY_DIR / f"{name}.yaml")
        return {item.id: predicate_hash(item.id, item.instructions) for item in query.predicates}

    assert hashes("listing_mismatch")["dissatisfaction"] == hashes("product_quality")[
        "dissatisfaction"
    ]


@pytest.mark.parametrize("path", WORKLOADS, ids=lambda path: path.stem)
async def test_workload_runs_end_to_end_with_mock_provider(path: Path, tmp_path) -> None:
    dataset = tmp_path / "reviews.parquet"
    rows = [{"review_id": f"r{index:02}", "review_text": f"review {index}"} for index in range(24)]
    pq.write_table(pa.Table.from_pylist(rows), dataset)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config.update({"dataset_path": dataset.as_posix(), "limit": 24, "max_calls": 200})
    query_path = tmp_path / path.name
    query_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    width = len(config["predicates"])
    optimizer = OptimizerSettings.model_validate(
        {
            "split": {"seed": 2027, "profiling_size": 8, "evaluation_size": 16},
            "profiling": {"concurrency": 2, "max_attempts": 8 * width},
            "execution": {"max_attempts_per_run": 16 * width, "cache_enabled": False},
            "planner": {"objective": "estimated_cost_usd"},
        }
    )
    usage = {
        item["id"]: (50 + 25 * index, 1) for index, item in enumerate(config["predicates"])
    }
    experiment_dir, comparison = await run_offline_experiment(
        query_path=query_path,
        settings={
            "executor": {"concurrency": 2},
            "pricing": {"id": "synthetic", "input_per_million_usd": 1.0,
                        "output_per_million_usd": 0.0},
        },
        optimizer=optimizer,
        provider=MockProvider(usage_by_predicate=usage),
        output_root=tmp_path / "results",
        experiment_id=path.stem,
    )
    assert comparison["fixed"]["completion_status"] == "complete"
    assert comparison["optimized"]["completion_status"] == "complete"
    assert comparison["output_equivalence"]["equivalent"]
    assert (experiment_dir / "selected_plan.json").is_file()
