import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts import download_dataset as downloader
from semanticflow.dataset.preparation import (
    prepare_sample,
    sample_fingerprint,
    validate_prepared_dataset,
)


def write_source(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)
    path.with_suffix(path.suffix + ".source.json").write_text(
        json.dumps(
            {
                "source_repository": "https://example.test/source",
                "dataset_variant": "fixture-real-shape",
                "original_source_filename": path.name,
            }
        ),
        encoding="utf-8",
    )


def test_preparation_is_order_independent_and_validation_reproduces(tmp_path: Path) -> None:
    rows = [
        {
            "review_id": f"id-{index}",
            "review_text": f"Genuine-shaped fixture review {index}",
            "rating": float(index % 5 + 1),
            "parent_asin": f"product-{index % 3}",
        }
        for index in range(12)
    ]
    rows.extend(
        [
            {"review_id": "empty", "review_text": "", "rating": 1.0, "parent_asin": "p"},
            {"review_id": "id-0", "review_text": "different duplicate", "rating": 2.0},
        ]
    )
    first_source = tmp_path / "first.parquet"
    second_source = tmp_path / "second.parquet"
    write_source(first_source, rows)
    write_source(second_source, list(reversed(rows)))
    first = tmp_path / "first_sample.parquet"
    second = tmp_path / "second_sample.parquet"
    prepare_sample(first_source, first, sample_size=5, seed=77)
    prepare_sample(second_source, second, sample_size=5, seed=77)
    assert sample_fingerprint(first) == sample_fingerprint(second)

    manifest_path = tmp_path / "manifest.json"
    manifest = validate_prepared_dataset(
        first,
        source=first_source,
        expected_rows=5,
        manifest_path=manifest_path,
    )
    assert manifest["output_row_count"] == 5
    assert manifest["validation"]["reproduction_check_passed"] is True
    assert manifest_path.is_file()


def test_semceb_download_metadata(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "reviews_filtered.parquet"

    def fake_download(_url: str, destination: Path):
        destination.write_bytes(b"PAR1fixture")
        return "upstream-checksum", destination.stat().st_size

    monkeypatch.setattr(downloader, "_download", fake_download)
    metadata = downloader.download_dataset(output, source="semceb")
    assert metadata["is_semceb_filtered_dataset"] is True
    assert metadata["source_repository"].endswith("/SemCEB")
    assert output.with_suffix(".parquet.source.json").is_file()
