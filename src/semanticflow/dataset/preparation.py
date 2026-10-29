from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from semanticflow.dataset.loader import ID_COLUMNS, TEXT_COLUMNS, DatasetError

SAMPLING_ALGORITHM = "sha256-seed-stable-id-v2"
SEMCEB_REPOSITORY = "https://github.com/utndatasystems/SemCEB"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _schema(connection: duckdb.DuckDBPyConnection, source: Path) -> list[str]:
    try:
        rows = connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{_sql_path(source)}')"
        ).fetchall()
    except duckdb.Error as exc:
        raise DatasetError(f"Unable to read source Parquet {source}: {exc}") from exc
    return [str(row[0]) for row in rows]


def _stable_id_expression(columns: list[str], existing_id: str | None) -> str:
    pieces = [
        f"COALESCE(CAST({_quote(column)} AS VARCHAR), '<NULL>')" for column in sorted(columns)
    ]
    generated = "'sf_' || sha256(concat_ws(chr(31), " + ", ".join(pieces) + "))"
    if existing_id:
        quoted = _quote(existing_id)
        return (
            f"CASE WHEN trim(CAST({quoted} AS VARCHAR)) <> '' "
            f"THEN CAST({quoted} AS VARCHAR) ELSE {generated} END"
        )
    return generated


def _row_hash_expression(columns: list[str]) -> str:
    pieces = [
        f"COALESCE(CAST({_quote(column)} AS VARCHAR), '<NULL>')" for column in sorted(columns)
    ]
    return "sha256(concat_ws(chr(31), " + ", ".join(pieces) + "))"


def _metadata_selects(columns: list[str], excluded: set[str]) -> list[str]:
    preferred = (
        "rating",
        "parent_asin",
        "asin",
        "product_id",
        "user_id",
        "review_title",
        "title",
        "timestamp_ms",
        "timestamp",
        "helpful_vote",
        "verified_purchase",
    )
    return [_quote(name) for name in preferred if name in columns and name not in excluded]


def prepare_sample(
    source: str | Path,
    output: str | Path,
    *,
    sample_size: int,
    seed: int,
    overwrite: bool = False,
    source_metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create an order-independent deterministic review sample with DuckDB."""

    source_path = Path(source)
    output_path = Path(output)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source dataset not found: {source_path}")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output_path}")

    connection = duckdb.connect(database=":memory:")
    try:
        columns = _schema(connection, source_path)
        text_column = next((name for name in TEXT_COLUMNS if name in columns), None)
        if text_column is None:
            raise DatasetError(
                "Source must contain a review text column; expected one of "
                + ", ".join(TEXT_COLUMNS)
            )
        existing_id = next((name for name in ID_COLUMNS if name in columns), None)
        review_id = _stable_id_expression(columns, existing_id)
        row_hash = _row_hash_expression(columns)
        text = _quote(text_column)
        metadata = _metadata_selects(columns, {text_column, existing_id or ""})
        selected = ", ".join([f"CAST({text} AS VARCHAR) AS review_text", *metadata])
        valid_count = connection.execute(
            f"SELECT count(*) FROM read_parquet('{_sql_path(source_path)}') "
            f"WHERE {text} IS NOT NULL AND length(trim(CAST({text} AS VARCHAR))) > 0"
        ).fetchone()[0]
        if valid_count < sample_size:
            raise DatasetError(
                f"Requested {sample_size} reviews but source contains only {valid_count} valid rows"
            )
        query = f"""
            WITH normalized AS (
                SELECT {review_id} AS review_id, {row_hash} AS _row_hash, {selected}
                FROM read_parquet('{_sql_path(source_path)}')
                WHERE {text} IS NOT NULL AND length(trim(CAST({text} AS VARCHAR))) > 0
            ), unique_reviews AS (
                SELECT * EXCLUDE (duplicate_number, _row_hash)
                FROM (
                    SELECT *, row_number() OVER (
                        PARTITION BY review_id ORDER BY _row_hash
                    ) AS duplicate_number
                    FROM normalized
                )
                WHERE duplicate_number = 1
            )
            SELECT * FROM unique_reviews
            ORDER BY sha256('{seed}:' || review_id), review_id
            LIMIT {sample_size}
        """
        unique_count = connection.execute(
            f"WITH normalized AS (SELECT {review_id} AS review_id FROM "
            f"read_parquet('{_sql_path(source_path)}') WHERE {text} IS NOT NULL "
            f"AND length(trim(CAST({text} AS VARCHAR))) > 0) "
            "SELECT count(DISTINCT review_id) FROM normalized"
        ).fetchone()[0]
        if unique_count < sample_size:
            raise DatasetError(
                f"Requested {sample_size} reviews but source has only {unique_count} unique IDs"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(output_path.name + ".part")
        if temporary.exists():
            temporary.unlink()
        connection.execute(
            f"COPY ({query}) TO '{_sql_path(temporary)}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        temporary.replace(output_path)
    finally:
        connection.close()

    source_metadata: dict[str, Any] = {}
    metadata_path = Path(source_metadata_path) if source_metadata_path else source_path.with_suffix(
        source_path.suffix + ".source.json"
    )
    if metadata_path.is_file():
        source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    prepared_at = datetime.now(UTC).isoformat()
    preparation = {
        "source_path": str(source_path),
        "source_file_checksum_sha256": file_sha256(source_path),
        "source_repository": source_metadata.get("source_repository", "unknown"),
        "dataset_variant": source_metadata.get("dataset_variant", "unknown"),
        "original_source_filename": source_metadata.get(
            "original_source_filename", source_path.name
        ),
        "sampling_seed": seed,
        "sampling_algorithm_version": SAMPLING_ALGORITHM,
        "input_valid_row_count": int(valid_count),
        "input_unique_id_count": int(unique_count),
        "output_row_count": sample_size,
        "output_checksum_sha256": file_sha256(output_path),
        "preparation_timestamp": prepared_at,
    }
    sidecar = output_path.with_suffix(output_path.suffix + ".preparation.json")
    sidecar.write_text(json.dumps(preparation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return preparation


def sample_fingerprint(path: str | Path) -> str:
    connection = duckdb.connect(database=":memory:")
    try:
        rows = connection.execute(
            f"SELECT review_id, review_text FROM read_parquet('{_sql_path(Path(path))}') "
            "ORDER BY review_id"
        ).fetchall()
    finally:
        connection.close()
    digest = hashlib.sha256()
    for review_id, review_text in rows:
        digest.update(str(review_id).encode())
        digest.update(b"\0")
        digest.update(str(review_text).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def validate_prepared_dataset(
    dataset: str | Path,
    *,
    source: str | Path,
    expected_rows: int = 1000,
    manifest_path: str | Path | None = None,
    verify_reproduction: bool = True,
) -> dict[str, Any]:
    dataset_path = Path(dataset)
    source_path = Path(source)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Prepared dataset not found: {dataset_path}")
    preparation_path = dataset_path.with_suffix(dataset_path.suffix + ".preparation.json")
    if not preparation_path.is_file():
        raise FileNotFoundError(f"Preparation metadata not found: {preparation_path}")
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))

    connection = duckdb.connect(database=":memory:")
    try:
        columns = _schema(connection, dataset_path)
        required = {"review_id", "review_text"}
        missing = required.difference(columns)
        if missing:
            raise DatasetError(f"Prepared dataset missing required columns: {sorted(missing)}")
        counts = connection.execute(
            f"SELECT count(*), count(*) FILTER (WHERE review_text IS NULL OR "
            f"length(trim(review_text)) = 0), count(*) - count(DISTINCT review_id) "
            f"FROM read_parquet('{_sql_path(dataset_path)}')"
        ).fetchone()
    finally:
        connection.close()
    row_count, invalid_text_count, duplicate_id_count = map(int, counts)
    if row_count != expected_rows:
        raise DatasetError(f"Expected {expected_rows} rows, found {row_count}")
    if invalid_text_count:
        raise DatasetError(f"Found {invalid_text_count} null or empty review texts")
    if duplicate_id_count:
        raise DatasetError(f"Found {duplicate_id_count} duplicate review IDs")

    reproducible = False
    if verify_reproduction:
        with tempfile.TemporaryDirectory(prefix="semanticflow-reproduce-") as directory:
            reproduced = Path(directory) / "reproduced.parquet"
            prepare_sample(
                source_path,
                reproduced,
                sample_size=expected_rows,
                seed=int(preparation["sampling_seed"]),
                overwrite=False,
                source_metadata_path=source_path.with_suffix(source_path.suffix + ".source.json"),
            )
            reproducible = sample_fingerprint(dataset_path) == sample_fingerprint(reproduced)
        if not reproducible:
            raise DatasetError("Reproduction check produced a different sample")

    manifest = {
        "source_repository": preparation["source_repository"],
        "dataset_variant": preparation["dataset_variant"],
        "original_source_filename": preparation["original_source_filename"],
        "source_path": str(source_path),
        "source_file_checksum_sha256": file_sha256(source_path),
        "sampling_seed": preparation["sampling_seed"],
        "sampling_algorithm_version": preparation["sampling_algorithm_version"],
        "input_valid_row_count": preparation["input_valid_row_count"],
        "input_unique_id_count": preparation["input_unique_id_count"],
        "output_row_count": row_count,
        "output_checksum_sha256": file_sha256(dataset_path),
        "sample_content_fingerprint_sha256": sample_fingerprint(dataset_path),
        "preparation_timestamp": preparation["preparation_timestamp"],
        "validation_timestamp": datetime.now(UTC).isoformat(),
        "validation": {
            "required_columns_present": True,
            "null_or_empty_review_text_count": invalid_text_count,
            "duplicate_review_id_count": duplicate_id_count,
            "duckdb_readable": True,
            "reproduction_check_passed": reproducible if verify_reproduction else None,
        },
    }
    target = Path(manifest_path) if manifest_path else dataset_path.parent / "dataset_manifest.json"
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
