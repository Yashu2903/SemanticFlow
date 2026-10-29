import csv
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.export_labels_template import export_template


def test_label_template_is_empty_and_stable(tmp_path: Path) -> None:
    dataset = tmp_path / "reviews.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"review_id": str(i), "review_text": f"review {i}"} for i in range(5)]
        ),
        dataset,
    )
    query = tmp_path / "query.yaml"
    query.write_text(
        f"query_id: q\ndataset_path: {dataset.as_posix()}\nlimit: 5\nthreshold: 0.5\n"
        "max_calls: 5\npredicates:\n  - id: p\n    instructions: Is it negative?\n",
        encoding="utf-8",
    )
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    export_template(query, first, 3, 7)
    export_template(query, second, 3, 7)
    with first.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with second.open(encoding="utf-8") as handle:
        other = list(csv.DictReader(handle))
    assert [row["review_id"] for row in rows] == [row["review_id"] for row in other]
    assert all(row["match"] == "" for row in rows)
    assert all(row["query_context"] == "Is it negative?" for row in rows)

