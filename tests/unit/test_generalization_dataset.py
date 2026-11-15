import json
from pathlib import Path

import pytest

from semanticflow.dataset.generalization import (
    IndependentRecord,
    hamming_distance,
    normalize_text,
    prepare_independent_dataset,
    simhash64,
    validate_independent_dataset,
)


def _records() -> list[IndependentRecord]:
    rows = [
        IndependentRecord(
            source_id=f"record-{index}",
            text=(
                f"Customer narrative {index} describes a distinct product or banking issue "
                f"with enough words for deterministic preparation and validation number {index}."
            ),
            primary_group=f"primary-{index}",
            secondary_group=f"secondary-{index}",
            metadata={"index": index},
        )
        for index in range(30)
    ]
    rows.append(
        IndependentRecord(
            source_id="duplicate-record",
            text=rows[0].text,
            primary_group="duplicate-primary",
            secondary_group="duplicate-secondary",
            metadata={},
        )
    )
    return rows


def _prepare(tmp_path: Path, name: str) -> dict[str, object]:
    raw = tmp_path / "raw.txt"
    raw.write_text("public fixture\n", encoding="utf-8")
    return prepare_independent_dataset(
        _records(),
        tmp_path / f"{name}.parquet",
        dataset_id="independent_fixture",
        source_path=raw,
        source_url="https://example.invalid/public-fixture",
        source_repository="https://example.invalid/repository",
        usage_rights="fixture license",
        profiling_size=3,
        evaluation_size=5,
        seed=46,
        candidate_multiplier=10,
    )


def test_normalization_and_simhash_are_stable() -> None:
    assert normalize_text("  A\r\n  B  ") == "a b"
    assert simhash64("same text") == simhash64("same text")
    assert hamming_distance(0b1010, 0b0011) == 2


def test_preparation_is_deterministic_group_disjoint_and_fingerprinted(tmp_path) -> None:
    first = _prepare(tmp_path, "first")
    second = _prepare(tmp_path, "second")
    assert first["dataset_fingerprint_sha256"] == second["dataset_fingerprint_sha256"]
    assert first["profiling_ids"] == second["profiling_ids"]
    assert first["evaluation_ids"] == second["evaluation_ids"]
    assert not set(first["profiling_ids"]) & set(first["evaluation_ids"])
    assert first["duplicate_detection"]["exact_duplicates_removed"] == 1
    validation = validate_independent_dataset(tmp_path / "first.parquet")
    assert validation == {
        "dataset_id": "independent_fixture",
        "rows": 8,
        "profiling_rows": 3,
        "evaluation_rows": 5,
        "dataset_fingerprint_sha256": first["dataset_fingerprint_sha256"],
        "output_sha256": first["output_sha256"],
        "near_duplicate_cross_partition_pairs": 0,
        "valid": True,
    }


def test_preparation_rejects_historical_overlap(tmp_path) -> None:
    raw = tmp_path / "raw.txt"
    raw.write_text("fixture", encoding="utf-8")
    with pytest.raises(ValueError, match="historical SemCEB"):
        prepare_independent_dataset(
            _records(),
            tmp_path / "overlap.parquet",
            dataset_id="overlap",
            source_path=raw,
            source_url="https://example.invalid",
            source_repository="https://example.invalid",
            usage_rights="fixture",
            profiling_size=3,
            evaluation_size=5,
            seed=46,
            candidate_multiplier=10,
            historical_texts=[row.text for row in _records()],
        )


def test_dataset_fingerprint_detects_tampering(tmp_path) -> None:
    _prepare(tmp_path, "tamper")
    manifest_path = tmp_path / "tamper.parquet.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        validate_independent_dataset(tmp_path / "tamper.parquet")
