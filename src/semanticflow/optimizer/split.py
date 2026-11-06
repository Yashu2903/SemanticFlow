from __future__ import annotations

import hashlib
from pathlib import Path

from semanticflow.dataset.loader import ReviewRecord
from semanticflow.optimizer.models import SplitManifest


def id_fingerprint(review_ids: list[str]) -> str:
    canonical = "\n".join(sorted(review_ids)).encode()
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def deterministic_split(
    records: list[ReviewRecord],
    *,
    profiling_size: int,
    evaluation_size: int,
    seed: int,
    dataset_path: str | Path,
    dataset_fingerprint: str,
    dataset_version: str,
) -> tuple[list[ReviewRecord], list[ReviewRecord], SplitManifest]:
    if profiling_size <= 0 or evaluation_size <= 0:
        raise ValueError("split sizes must be positive")
    if profiling_size + evaluation_size > len(records):
        raise ValueError("split sizes exceed available records")
    ids = [record.review_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("record IDs must be unique before splitting")

    def rank(record: ReviewRecord) -> tuple[str, str]:
        digest = hashlib.sha256(f"{seed}:{record.review_id}".encode()).hexdigest()
        return digest, record.review_id

    ranked = sorted(records, key=rank)
    profiling = sorted(ranked[:profiling_size], key=lambda item: item.review_id)
    evaluation = sorted(
        ranked[profiling_size : profiling_size + evaluation_size],
        key=lambda item: item.review_id,
    )
    profiling_ids = [record.review_id for record in profiling]
    evaluation_ids = [record.review_id for record in evaluation]
    manifest = SplitManifest(
        dataset_path=str(dataset_path),
        dataset_fingerprint=dataset_fingerprint,
        dataset_version=dataset_version,
        seed=seed,
        source_record_count=len(records),
        profiling_size=profiling_size,
        evaluation_size=evaluation_size,
        profiling_ids=profiling_ids,
        evaluation_ids=evaluation_ids,
        profiling_fingerprint=id_fingerprint(profiling_ids),
        evaluation_fingerprint=id_fingerprint(evaluation_ids),
    )
    return profiling, evaluation, manifest
