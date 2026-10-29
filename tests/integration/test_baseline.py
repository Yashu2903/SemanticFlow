import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from semanticflow.providers.mock import MockProvider
from semanticflow.runtime import run_query


@pytest.mark.asyncio
async def test_end_to_end_mock_artifacts(tmp_path: Path) -> None:
    dataset = tmp_path / "reviews.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"review_id": "a", "review_text": "bad"},
                {"review_id": "b", "review_text": "good"},
            ]
        ),
        dataset,
    )
    query = tmp_path / "query.yaml"
    query.write_text(
        f"query_id: e2e\nworkload_id: test\ndataset_path: {dataset.as_posix()}\n"
        "dataset_version: fixture-v1\nlimit: 2\nthreshold: 0.5\nmax_calls: 4\n"
        "predicates:\n  - id: negative\n    instructions: Is this negative?\n"
        "  - id: broken\n    instructions: Is it broken?\n",
        encoding="utf-8",
    )
    provider = MockProvider(
        {
            ("bad", "negative"): 0.9,
            ("good", "negative"): 0.1,
            ("bad", "broken"): 0.8,
        }
    )
    outcome = await run_query(
        query_path=query,
        settings={"executor": {"concurrency": 2}},
        provider=provider,
        output_dir=tmp_path / "results",
    )
    expected = {"matches.jsonl", "decisions.jsonl", "report.json", "manifest.json"}
    assert {item.name for item in outcome.run_dir.iterdir()} == expected
    matches = [
        json.loads(line)
        for line in (outcome.run_dir / "matches.jsonl").read_text().splitlines()
    ]
    decisions = [
        json.loads(line) for line in (outcome.run_dir / "decisions.jsonl").read_text().splitlines()
    ]
    report = json.loads((outcome.run_dir / "report.json").read_text())
    manifest = json.loads((outcome.run_dir / "manifest.json").read_text())
    assert [row["review_id"] for row in matches] == ["a"]
    assert len(decisions) == 3
    assert all(row["model_identifier"] and row["question_hash"] for row in decisions)
    assert report["logical_evaluations"] == 3
    assert report["actual_provider_calls"] == 3
    assert report["reviews_evaluated"] == 2
    assert report["predicate_decisions_evaluated"] == 3
    assert report["successful_api_calls"] == 3
    assert report["failed_api_calls"] == 0
    assert report["quality"]["status"] == "pending"
    assert manifest["dataset_fingerprint"].startswith("sha256:")
    assert manifest["completion_status"] == "complete"


@pytest.mark.asyncio
async def test_runtime_max_calls_override_marks_run_incomplete(tmp_path: Path) -> None:
    dataset = tmp_path / "reviews.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"review_id": str(index), "review_text": f"review {index}"} for index in range(3)]
        ),
        dataset,
    )
    query = tmp_path / "query.yaml"
    query.write_text(
        f"query_id: budget\ndataset_path: {dataset.as_posix()}\nlimit: 3\n"
        "threshold: 0.0\nmax_calls: 100\npredicates:\n"
        "  - id: first\n    instructions: First?\n"
        "  - id: second\n    instructions: Second?\n",
        encoding="utf-8",
    )
    outcome = await run_query(
        query_path=query,
        settings={"executor": {"concurrency": 1}},
        provider=MockProvider(),
        output_dir=tmp_path / "results",
        max_calls_override=3,
    )
    assert outcome.report["completion_status"] == "incomplete"
    assert outcome.report["request_attempts"] == 3
    assert outcome.report["final_record_count"] == 0
