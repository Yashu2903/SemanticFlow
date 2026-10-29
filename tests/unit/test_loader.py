from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from semanticflow.dataset import DatasetError, load_dataset


def write(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_parquet_loading_filters_empty_and_respects_limit(tmp_path: Path) -> None:
    path = tmp_path / "reviews.parquet"
    write(
        path,
        [
            {"review_id": "b", "review_text": "second", "rating": 2},
            {"review_id": "a", "review_text": "first", "rating": 5},
            {"review_id": "c", "review_text": "  ", "rating": 3},
            {"review_id": "d", "review_text": None, "rating": 3},
        ],
    )
    records = load_dataset(path, limit=1)
    assert [(row.review_id, row.review_text) for row in records] == [("a", "first")]
    assert records[0].metadata["rating"] == 5


def test_stable_constructed_ids_and_duplicate_suffixes(tmp_path: Path) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    rows = [{"review_text": "same", "rating": 1}, {"review_text": "other", "rating": 2}]
    write(first, rows)
    write(second, list(reversed(rows)))
    assert [row.review_id for row in load_dataset(first)] == [
        row.review_id for row in load_dataset(second)
    ]

    duplicated = tmp_path / "duplicated.parquet"
    write(
        duplicated,
        [{"review_id": "x", "review_text": "a"}, {"review_id": "x", "review_text": "b"}],
    )
    assert [row.review_id for row in load_dataset(duplicated)] == ["x", "x__dup1"]


def test_loader_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing.parquet")
    path = tmp_path / "bad.parquet"
    write(path, [{"id": "x", "body": "no known text column"}])
    with pytest.raises(DatasetError, match="review text column"):
        load_dataset(path)
    with pytest.raises(DatasetError):
        load_dataset(path, limit=0)
