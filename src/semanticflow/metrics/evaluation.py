from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class LabelError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    labeled_records: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_labels(
    labels_path: str | Path,
    predicted_matches: set[str],
    valid_record_ids: set[str] | None = None,
) -> QualityMetrics:
    source = Path(labels_path)
    if not source.is_file():
        raise FileNotFoundError(f"Labels file not found: {source}")
    labels: dict[str, int] = {}
    seen_ids: set[str] = set()
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"review_id", "match"}.issubset(reader.fieldnames):
            raise LabelError("Labels CSV must contain review_id and match columns")
        for row_number, row in enumerate(reader, start=2):
            review_id = (row.get("review_id") or "").strip()
            raw_label = (row.get("match") or "").strip()
            if not review_id:
                raise LabelError(f"Missing review_id on row {row_number}")
            if review_id in seen_ids:
                raise LabelError(f"Duplicate label for review_id {review_id}")
            seen_ids.add(review_id)
            if not raw_label:
                continue
            if raw_label not in {"0", "1"}:
                raise LabelError(f"Invalid label {raw_label!r} for review_id {review_id}")
            if valid_record_ids is not None and review_id not in valid_record_ids:
                raise LabelError(f"Label references unknown review_id {review_id}")
            labels[review_id] = int(raw_label)

    tp = sum(label == 1 and record_id in predicted_matches for record_id, label in labels.items())
    fp = sum(label == 0 and record_id in predicted_matches for record_id, label in labels.items())
    fn = sum(
        label == 1 and record_id not in predicted_matches for record_id, label in labels.items()
    )
    tn = sum(
        label == 0 and record_id not in predicted_matches for record_id, label in labels.items()
    )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return QualityMetrics(len(labels), tp, fp, fn, tn, precision, recall, f1)
