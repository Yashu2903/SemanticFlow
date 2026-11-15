from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from semanticflow.dataset.generalization import (
    IndependentRecord,
    _choose_partition,
    _deduplicate,
    _rank,
    hamming_distance,
    simhash64,
    stable_fingerprint,
    text_sha256,
)
from semanticflow.dataset.preparation import file_sha256

M5_DATASET_SCHEMA_VERSION = "m5-fusion-benchmark-v1"
M5_SPLIT_ALGORITHM = "sha256-ranked-group-aware-four-way-v1"


def iter_sms_spam(path: str | Path) -> Iterator[IndependentRecord]:
    """Normalize the UCI SMS Spam Collection tab-separated source."""
    with Path(path).open(encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            label, separator, text = line.rstrip("\r\n").partition("\t")
            text = text.strip()
            if not separator or not text:
                continue
            source_id = f"sms:{index}:{hashlib.sha256(text.encode()).hexdigest()[:16]}"
            yield IndependentRecord(
                source_id=source_id,
                text=text,
                primary_group=source_id,
                secondary_group=None,
                metadata={"source_label": label.strip()},
            )


def _historical_signature_index(
    texts: Sequence[str],
) -> tuple[set[str], dict[tuple[int, int], set[int]]]:
    exact = {text_sha256(text) for text in texts}
    bands: dict[tuple[int, int], set[int]] = defaultdict(set)
    for text in texts:
        signature = simhash64(text)
        for band in range(4):
            bands[(band, (signature >> (band * 16)) & 0xFFFF)].add(signature)
    return exact, bands


def _overlaps_historical(
    text: str,
    exact: set[str],
    bands: Mapping[tuple[int, int], set[int]],
    maximum_hamming: int,
) -> bool:
    if text_sha256(text) in exact:
        return True
    signature = simhash64(text)
    candidates: set[int] = set()
    for band in range(4):
        candidates.update(bands.get((band, (signature >> (band * 16)) & 0xFFFF), set()))
    return any(
        hamming_distance(signature, candidate) <= maximum_hamming
        for candidate in candidates
    )


def prepare_fusion_dataset(
    records: Iterable[IndependentRecord],
    output: str | Path,
    *,
    dataset_id: str,
    source_path: str | Path,
    source_url: str,
    source_repository: str,
    usage_rights: str,
    partition_sizes: Mapping[str, int],
    seed: int,
    historical_texts: Sequence[str],
    candidate_multiplier: int = 20,
    maximum_near_duplicate_hamming: int = 3,
    overwrite: bool = False,
) -> dict[str, Any]:
    required_partitions = ("development", "pilot", "selection", "evaluation")
    if tuple(partition_sizes) != required_partitions:
        raise ValueError(f"partition order must be {required_partitions}")
    if any(size <= 0 for size in partition_sizes.values()):
        raise ValueError("all partition sizes must be positive")
    destination = Path(output)
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    if (destination.exists() or manifest_path.exists()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite frozen dataset {destination}")
    historical_exact, historical_bands = _historical_signature_index(historical_texts)
    target = sum(partition_sizes.values())
    ranked = sorted(records, key=lambda row: (_rank(seed, row.source_id), row.source_id))
    source_valid_count = len(ranked)
    candidates: list[IndependentRecord] = []
    historical_excluded = 0
    for record in ranked:
        if _overlaps_historical(
            record.text,
            historical_exact,
            historical_bands,
            maximum_near_duplicate_hamming,
        ):
            historical_excluded += 1
            continue
        candidates.append(record)
        if len(candidates) >= target * candidate_multiplier:
            break
    unique, duplicate_report = _deduplicate(
        candidates, maximum_hamming_distance=maximum_near_duplicate_hamming
    )
    if len(unique) < target:
        raise ValueError(f"{dataset_id} has only {len(unique)} eligible unique candidates")
    grouped: dict[str, list[IndependentRecord]] = defaultdict(list)
    for record in unique:
        grouped[record.primary_group].append(record)
    remaining = sorted(grouped.items(), key=lambda item: (_rank(seed + 1, item[0]), item[0]))
    blocked_secondary: set[str] = set()
    partition_records: dict[str, list[IndependentRecord]] = {}
    partition_groups: dict[str, set[str]] = {}
    for partition, size in partition_sizes.items():
        selected, used_groups, used_secondary = _choose_partition(
            remaining,
            target=size,
            blocked_secondary=blocked_secondary,
        )
        partition_records[partition] = selected
        partition_groups[partition] = used_groups
        blocked_secondary.update(used_secondary)
        remaining = [item for item in remaining if item[0] not in used_groups]
    output_rows: list[dict[str, Any]] = []
    for partition, records_in_partition in partition_records.items():
        for record in records_in_partition:
            record_id = f"{dataset_id}:" + hashlib.sha256(
                record.source_id.encode()
            ).hexdigest()[:24]
            output_rows.append(
                {
                    "review_id": record_id,
                    "review_text": record.text,
                    "partition": partition,
                    "source_record_id": record.source_id,
                    "primary_group": record.primary_group,
                    "secondary_group": record.secondary_group,
                    "metadata_json": json.dumps(
                        record.metadata, sort_keys=True, ensure_ascii=False
                    ),
                    "text_sha256": text_sha256(record.text),
                    "simhash64": f"{simhash64(record.text):016x}",
                }
            )
    output_rows.sort(key=lambda row: (row["partition"], row["review_id"]))
    if len({row["review_id"] for row in output_rows}) != len(output_rows):
        raise ValueError("prepared record IDs are not unique")
    if len({row["text_sha256"] for row in output_rows}) != len(output_rows):
        raise ValueError("exact duplicates remain in prepared dataset")
    all_groups = [groups for groups in partition_groups.values()]
    if any(
        left & right
        for index, left in enumerate(all_groups)
        for right in all_groups[index + 1 :]
    ):
        raise ValueError("primary groups cross partition boundaries")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    pq.write_table(pa.Table.from_pylist(output_rows), temporary, compression="zstd")
    temporary.replace(destination)
    partition_manifest: dict[str, Any] = {}
    for partition, size in partition_sizes.items():
        rows = [
            (row["review_id"], row["review_text"])
            for row in output_rows
            if row["partition"] == partition
        ]
        partition_manifest[partition] = {
            "size": size,
            "ids": [record_id for record_id, _ in sorted(rows)],
            "fingerprint_sha256": stable_fingerprint(rows),
        }
    all_pairs = [(row["review_id"], row["review_text"]) for row in output_rows]
    manifest = {
        "schema_version": M5_DATASET_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "source_path": str(source_path),
        "source_url": source_url,
        "source_repository": source_repository,
        "usage_rights": usage_rights,
        "source_sha256": file_sha256(source_path),
        "source_valid_record_count": source_valid_count,
        "sampling_seed": seed,
        "split_algorithm": M5_SPLIT_ALGORITHM,
        "near_duplicate_hamming_threshold": maximum_near_duplicate_hamming,
        "historical_records_excluded": historical_excluded,
        "historical_sources": "Milestone 1-4.6 prepared text snapshots",
        "duplicate_detection": duplicate_report,
        "partitions": partition_manifest,
        "dataset_fingerprint_sha256": stable_fingerprint(all_pairs),
        "output_sha256": file_sha256(destination),
        "prepared_at": datetime.now(UTC).isoformat(),
        "live_jev_calls": 0,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def validate_fusion_dataset(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    manifest_path = source.with_suffix(source.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != M5_DATASET_SCHEMA_VERSION:
        raise ValueError("unexpected Milestone 5 dataset schema")
    if file_sha256(source) != manifest["output_sha256"]:
        raise ValueError("dataset output fingerprint differs from manifest")
    rows = pq.read_table(source).to_pylist()
    expected = sum(item["size"] for item in manifest["partitions"].values())
    if len(rows) != expected:
        raise ValueError("dataset row count differs from partition manifest")
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    groups_by_partition: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["review_id"] in seen_ids or row["text_sha256"] in seen_text:
            raise ValueError("ID or exact-text duplicate in frozen dataset")
        seen_ids.add(row["review_id"])
        seen_text.add(row["text_sha256"])
        groups_by_partition[row["partition"]].add(row["primary_group"])
    group_sets = list(groups_by_partition.values())
    if any(
        left & right
        for index, left in enumerate(group_sets)
        for right in group_sets[index + 1 :]
    ):
        raise ValueError("primary groups cross partitions")
    for partition, expected_info in manifest["partitions"].items():
        selected = [
            (row["review_id"], row["review_text"])
            for row in rows
            if row["partition"] == partition
        ]
        if len(selected) != expected_info["size"]:
            raise ValueError(f"partition size differs for {partition}")
        if stable_fingerprint(selected) != expected_info["fingerprint_sha256"]:
            raise ValueError(f"partition fingerprint differs for {partition}")
    return {
        "dataset_id": manifest["dataset_id"],
        "rows": len(rows),
        "partitions": {
            name: details["size"] for name, details in manifest["partitions"].items()
        },
        "dataset_fingerprint_sha256": manifest["dataset_fingerprint_sha256"],
        "output_sha256": manifest["output_sha256"],
        "valid": True,
    }
