from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb

TEXT_COLUMNS = ("review_text", "text", "review_body", "reviewText")
ID_COLUMNS = ("review_id", "reviewId", "id", "reviewer_id")


class DatasetError(ValueError):
    """Raised when a dataset cannot satisfy the internal review schema."""


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    review_id: str
    review_text: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return value.hex()
    return str(value)


def _stable_rows(
    rows: list[dict[str, Any]], text_column: str, id_column: str | None
) -> list[ReviewRecord]:
    prepared: list[tuple[str, str, dict[str, Any], str]] = []
    for row in rows:
        text = str(row[text_column]).strip()
        metadata = {
            key: _canonical(value)
            for key, value in row.items()
            if key not in {text_column, id_column} and value is not None
        }
        canonical = json.dumps(
            {key: _canonical(value) for key, value in row.items()},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if id_column and row.get(id_column) is not None and str(row[id_column]).strip():
            base_id = str(row[id_column]).strip()
        else:
            base_id = "sf_" + hashlib.sha256(canonical.encode()).hexdigest()[:24]
        prepared.append((base_id, text, metadata, canonical))

    # Sorting before duplicate suffixing makes IDs and output independent of Parquet scan order.
    prepared.sort(key=lambda item: (item[0], item[3]))
    seen: dict[str, int] = {}
    records: list[ReviewRecord] = []
    for base_id, text, metadata, _ in prepared:
        occurrence = seen.get(base_id, 0)
        seen[base_id] = occurrence + 1
        review_id = base_id if occurrence == 0 else f"{base_id}__dup{occurrence}"
        records.append(ReviewRecord(review_id=review_id, review_text=text, metadata=metadata))
    return records


def load_dataset(path: str | Path, limit: int | None = None) -> list[ReviewRecord]:
    """Load valid reviews from Parquet into a stable internal schema."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Dataset not found: {source}")
    if limit is not None and limit <= 0:
        raise DatasetError("limit must be positive")

    try:
        connection = duckdb.connect(database=":memory:")
        escaped = str(source.resolve()).replace("'", "''")
        description = connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{escaped}')"
        ).fetchall()
        columns = [row[0] for row in description]
        text_column = next((name for name in TEXT_COLUMNS if name in columns), None)
        if text_column is None:
            raise DatasetError(
                "Dataset must contain a review text column; expected one of "
                + ", ".join(TEXT_COLUMNS)
            )
        id_column = next((name for name in ID_COLUMNS if name in columns), None)
        quoted_text = '"' + text_column.replace('"', '""') + '"'
        result = connection.execute(
            f"SELECT * FROM read_parquet('{escaped}') "
            f"WHERE {quoted_text} IS NOT NULL AND length(trim(CAST({quoted_text} AS VARCHAR))) > 0"
        )
        names = [item[0] for item in result.description]
        rows = [dict(zip(names, values, strict=True)) for values in result.fetchall()]
    except duckdb.Error as exc:
        raise DatasetError(f"Unable to read Parquet dataset {source}: {exc}") from exc
    finally:
        if "connection" in locals():
            connection.close()

    records = _stable_rows(rows, text_column, id_column)
    return records[:limit] if limit is not None else records
