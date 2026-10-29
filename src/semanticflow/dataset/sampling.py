from __future__ import annotations

import hashlib

from semanticflow.dataset.loader import ReviewRecord


def deterministic_sample(
    records: list[ReviewRecord], sample_size: int, seed: int
) -> list[ReviewRecord]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size > len(records):
        raise ValueError(f"sample_size {sample_size} exceeds available records {len(records)}")

    def rank(record: ReviewRecord) -> tuple[str, str]:
        digest = hashlib.sha256(f"{seed}:{record.review_id}".encode()).hexdigest()
        return digest, record.review_id

    return sorted(sorted(records, key=rank)[:sample_size], key=lambda record: record.review_id)

