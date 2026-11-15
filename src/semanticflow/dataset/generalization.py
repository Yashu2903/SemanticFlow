"""Independent-dataset preparation for the Milestone 4.6 generalization study.

The module is deliberately provider-free.  It normalizes public records, removes exact and
near duplicates, creates entity-aware profiling/evaluation partitions, and records enough
provenance to reproduce and audit the resulting Parquet files.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from semanticflow.dataset.preparation import file_sha256

M46_DATASET_SCHEMA_VERSION = "m46-independent-dataset-v1"
M46_SPLIT_ALGORITHM = "sha256-group-aware-near-deduplicated-v1"
_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True, slots=True)
class IndependentRecord:
    source_id: str
    text: str
    primary_group: str
    secondary_group: str | None
    metadata: Mapping[str, Any]


def normalize_text(text: str) -> str:
    return " ".join(_WORD.findall(text.casefold()))


def text_sha256(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode()).hexdigest()


def _shingles(text: str) -> list[str]:
    words = _WORD.findall(text.casefold())
    if len(words) < 3:
        return words
    return [" ".join(words[index : index + 3]) for index in range(len(words) - 2)]


def simhash64(text: str) -> int:
    """Stable 64-bit SimHash over word trigrams."""

    weights = [0] * 64
    for shingle in _shingles(text):
        value = int.from_bytes(hashlib.sha256(shingle.encode()).digest()[:8], "big")
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def stable_fingerprint(rows: Iterable[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for record_id, text in sorted(rows):
        digest.update(record_id.encode())
        digest.update(b"\0")
        digest.update(text.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def iter_amazon_reviews(path: str | Path) -> Iterator[IndependentRecord]:
    """Normalize Amazon Reviews'23 category JSONL (optionally gzip-compressed)."""

    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else source.open
    with opener(source, "rt", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = str(row.get("text") or "").strip()
            source_id = ":".join(
                str(row.get(name) or "")
                for name in ("user_id", "asin", "timestamp", "sort_timestamp")
            ).strip(":")
            if not text or not source_id:
                continue
            product = str(row.get("parent_asin") or row.get("asin") or source_id)
            user = str(row.get("user_id") or "").strip() or None
            yield IndependentRecord(
                source_id=source_id,
                text=text,
                primary_group=f"product:{product}",
                secondary_group=f"author:{user}" if user else None,
                metadata={
                    "asin": row.get("asin"),
                    "parent_asin": row.get("parent_asin"),
                    "rating": row.get("rating"),
                    "verified_purchase": row.get("verified_purchase"),
                    "timestamp": row.get("timestamp") or row.get("sort_timestamp"),
                    "title": row.get("title"),
                },
            )


def iter_cfpb_complaints(path: str | Path) -> Iterator[IndependentRecord]:
    """Normalize an official CFPB complaint CSV export with published narratives."""

    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            text = str(
                row.get("Consumer complaint narrative")
                or row.get("consumer_complaint_narrative")
                or row.get("complaint_what_happened")
                or ""
            ).strip()
            source_id = str(row.get("Complaint ID") or row.get("complaint_id") or "").strip()
            if not text or not source_id:
                continue
            yield IndependentRecord(
                source_id=source_id,
                text=text,
                primary_group=f"complaint:{source_id}",
                secondary_group=None,
                metadata={
                    "date_received": row.get("Date received") or row.get("date_received"),
                    "product": row.get("Product") or row.get("product"),
                    "sub_product": row.get("Sub-product") or row.get("sub_product"),
                    "issue": row.get("Issue") or row.get("issue"),
                    "sub_issue": row.get("Sub-issue") or row.get("sub_issue"),
                    "company": row.get("Company") or row.get("company"),
                },
            )


def iter_banking77(path: str | Path, *, split: str | None = None) -> Iterator[IndependentRecord]:
    """Normalize a pinned Banking77 train or test CSV."""

    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            text = str(row.get("text") or "").strip()
            category = str(row.get("category") or row.get("label") or "").strip()
            row_split = split or str(row.get("upstream_split") or "unspecified")
            if not text:
                continue
            source_id = (
                f"{row_split}:{index}:{hashlib.sha256(text.encode()).hexdigest()[:16]}"
            )
            yield IndependentRecord(
                source_id=source_id,
                text=text,
                primary_group=f"query:{source_id}",
                secondary_group=None,
                metadata={"upstream_split": row_split, "intent_category": category},
            )


def _rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _deduplicate(
    records: Sequence[IndependentRecord], *, maximum_hamming_distance: int
) -> tuple[list[IndependentRecord], dict[str, int]]:
    exact: set[str] = set()
    bands: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    kept: list[IndependentRecord] = []
    exact_removed = near_removed = malformed_removed = 0
    for record in records:
        normalized = normalize_text(record.text)
        if len(normalized) < 20:
            malformed_removed += 1
            continue
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        if digest in exact:
            exact_removed += 1
            continue
        signature = simhash64(normalized)
        candidates: set[int] = set()
        for band in range(4):
            candidates.update(
                index
                for index, _ in bands[(band, (signature >> (16 * band)) & 0xFFFF)]
            )
        duplicate = False
        for index in candidates:
            other = kept[index]
            length_ratio = len(normalized) / max(1, len(normalize_text(other.text)))
            if 0.8 <= length_ratio <= 1.25 and hamming_distance(
                signature, simhash64(other.text)
            ) <= maximum_hamming_distance:
                duplicate = True
                break
        if duplicate:
            near_removed += 1
            continue
        index = len(kept)
        kept.append(record)
        exact.add(digest)
        for band in range(4):
            bands[(band, (signature >> (16 * band)) & 0xFFFF)].append((index, signature))
    return kept, {
        "exact_duplicates_removed": exact_removed,
        "near_duplicates_removed": near_removed,
        "malformed_or_too_short_removed": malformed_removed,
    }


def _historical_overlap(
    records: Sequence[IndependentRecord], historical_texts: Sequence[str]
) -> dict[str, int]:
    historical_exact = {text_sha256(text) for text in historical_texts}
    historical_signatures = [simhash64(text) for text in historical_texts]
    exact = near = 0
    for record in records:
        signature = simhash64(record.text)
        if text_sha256(record.text) in historical_exact:
            exact += 1
        elif any(hamming_distance(signature, item) <= 3 for item in historical_signatures):
            near += 1
    return {"exact_text_overlap": exact, "near_text_overlap": near}


def _choose_partition(
    groups: Sequence[tuple[str, list[IndependentRecord]]],
    *,
    target: int,
    blocked_secondary: set[str],
) -> tuple[list[IndependentRecord], set[str], set[str]]:
    selected: list[IndependentRecord] = []
    used_primary: set[str] = set()
    used_secondary: set[str] = set()
    remaining = target
    for primary, rows in groups:
        secondary = {row.secondary_group for row in rows if row.secondary_group}
        if len(rows) > remaining or secondary & blocked_secondary:
            continue
        selected.extend(rows)
        used_primary.add(primary)
        used_secondary.update(secondary)
        remaining -= len(rows)
        if remaining == 0:
            break
    if remaining:
        raise ValueError(
            f"group-aware selection could not fill {target} records; {remaining} short"
        )
    return selected, used_primary, used_secondary


def prepare_independent_dataset(
    records: Iterable[IndependentRecord],
    output: str | Path,
    *,
    dataset_id: str,
    source_path: str | Path,
    source_url: str,
    source_repository: str,
    usage_rights: str,
    profiling_size: int = 200,
    evaluation_size: int = 800,
    seed: int = 4601,
    candidate_multiplier: int = 12,
    maximum_near_duplicate_hamming: int = 3,
    historical_texts: Sequence[str] = (),
    overwrite: bool = False,
) -> dict[str, Any]:
    """Prepare a deterministic, group-disjoint independent evaluation snapshot."""

    target = profiling_size + evaluation_size
    destination = Path(output)
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {destination}")
    ranked = sorted(records, key=lambda row: (_rank(seed, row.source_id), row.source_id))
    valid_source_count = len(ranked)
    candidates = ranked[: min(len(ranked), target * candidate_multiplier)]
    unique, duplicate_report = _deduplicate(
        candidates, maximum_hamming_distance=maximum_near_duplicate_hamming
    )
    if len(unique) < target:
        raise ValueError(f"{dataset_id} has only {len(unique)} usable unique candidates")
    grouped: dict[str, list[IndependentRecord]] = defaultdict(list)
    for record in unique:
        grouped[record.primary_group].append(record)
    groups = sorted(grouped.items(), key=lambda item: (_rank(seed + 1, item[0]), item[0]))
    profiling, profiling_groups, profiling_secondary = _choose_partition(
        groups, target=profiling_size, blocked_secondary=set()
    )
    remaining_groups = [item for item in groups if item[0] not in profiling_groups]
    evaluation, evaluation_groups, evaluation_secondary = _choose_partition(
        remaining_groups,
        target=evaluation_size,
        blocked_secondary=profiling_secondary,
    )
    selected = [("profiling", row) for row in profiling] + [
        ("evaluation", row) for row in evaluation
    ]
    overlap = _historical_overlap([row for _, row in selected], historical_texts)
    if overlap["exact_text_overlap"] or overlap["near_text_overlap"]:
        raise ValueError(f"{dataset_id} overlaps the historical SemCEB text sample: {overlap}")
    output_rows = []
    for partition, record in selected:
        record_id = f"{dataset_id}:" + hashlib.sha256(record.source_id.encode()).hexdigest()[:24]
        output_rows.append(
            {
                "review_id": record_id,
                "review_text": record.text,
                "partition": partition,
                "source_record_id": record.source_id,
                "primary_group": record.primary_group,
                "secondary_group": record.secondary_group,
                "metadata_json": json.dumps(record.metadata, sort_keys=True, ensure_ascii=False),
                "text_sha256": text_sha256(record.text),
                "simhash64": f"{simhash64(record.text):016x}",
            }
        )
    output_rows.sort(key=lambda row: (row["partition"], row["review_id"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    pq.write_table(pa.Table.from_pylist(output_rows), temporary, compression="zstd")
    temporary.replace(destination)
    profile_rows = [
        (row["review_id"], row["review_text"])
        for row in output_rows
        if row["partition"] == "profiling"
    ]
    evaluation_rows = [
        (row["review_id"], row["review_text"])
        for row in output_rows
        if row["partition"] == "evaluation"
    ]
    manifest = {
        "schema_version": M46_DATASET_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "source_path": str(Path(source_path)),
        "source_url": source_url,
        "source_repository": source_repository,
        "usage_rights": usage_rights,
        "source_sha256": file_sha256(source_path),
        "source_valid_record_count": valid_source_count,
        "candidate_count": len(candidates),
        "candidate_multiplier": candidate_multiplier,
        "sampling_seed": seed,
        "split_algorithm": M46_SPLIT_ALGORITHM,
        "near_duplicate_hamming_threshold": maximum_near_duplicate_hamming,
        "profiling_size": profiling_size,
        "evaluation_size": evaluation_size,
        "profiling_ids": [item[0] for item in sorted(profile_rows)],
        "evaluation_ids": [item[0] for item in sorted(evaluation_rows)],
        "profiling_fingerprint_sha256": stable_fingerprint(profile_rows),
        "evaluation_fingerprint_sha256": stable_fingerprint(evaluation_rows),
        "dataset_fingerprint_sha256": stable_fingerprint(profile_rows + evaluation_rows),
        "output_sha256": file_sha256(destination),
        "duplicate_detection": duplicate_report,
        "historical_semceb_overlap": overlap,
        "integrity": {
            "partition_id_overlap_count": 0,
            "primary_group_overlap_count": len(profiling_groups & evaluation_groups),
            "secondary_group_overlap_count": len(profiling_secondary & evaluation_secondary),
            "profiling_exact_text_duplicates": len(profile_rows)
            - len({text_sha256(text) for _, text in profile_rows}),
            "evaluation_exact_text_duplicates": len(evaluation_rows)
            - len({text_sha256(text) for _, text in evaluation_rows}),
        },
        "prepared_at": datetime.now(UTC).isoformat(),
        "live_jev_calls": 0,
    }
    if any(manifest["integrity"].values()):
        raise ValueError(f"partition integrity failure: {manifest['integrity']}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def validate_independent_dataset(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    manifest_path = source.with_suffix(source.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    table = pq.read_table(source)
    rows = table.to_pylist()
    profiling = [row for row in rows if row["partition"] == "profiling"]
    evaluation = [row for row in rows if row["partition"] == "evaluation"]
    if (
        len(profiling) != manifest["profiling_size"]
        or len(evaluation) != manifest["evaluation_size"]
    ):
        raise ValueError("partition sizes do not match the manifest")
    if file_sha256(source) != manifest["output_sha256"]:
        raise ValueError("dataset file fingerprint differs from its frozen manifest")
    left_ids = {row["review_id"] for row in profiling}
    right_ids = {row["review_id"] for row in evaluation}
    if left_ids & right_ids:
        raise ValueError("profiling and evaluation IDs overlap")
    if {row["primary_group"] for row in profiling} & {row["primary_group"] for row in evaluation}:
        raise ValueError("primary entities cross the partition boundary")
    left_secondary = {row["secondary_group"] for row in profiling if row["secondary_group"]}
    right_secondary = {row["secondary_group"] for row in evaluation if row["secondary_group"]}
    if left_secondary & right_secondary:
        raise ValueError("secondary entities cross the partition boundary")
    signatures: dict[tuple[int, int], list[tuple[str, int]]] = defaultdict(list)
    for row in profiling:
        signature = int(row["simhash64"], 16)
        for band in range(4):
            signatures[(band, (signature >> (16 * band)) & 0xFFFF)].append(
                (row["review_id"], signature)
            )
    near_overlap = set()
    for row in evaluation:
        signature = int(row["simhash64"], 16)
        candidates = set()
        for band in range(4):
            candidates.update(signatures[(band, (signature >> (16 * band)) & 0xFFFF)])
        for left_id, left_signature in candidates:
            if (
                hamming_distance(signature, left_signature)
                <= manifest["near_duplicate_hamming_threshold"]
            ):
                near_overlap.add((left_id, row["review_id"]))
    if near_overlap:
        raise ValueError(f"near-identical records cross partitions: {sorted(near_overlap)[:3]}")
    return {
        "dataset_id": manifest["dataset_id"],
        "rows": len(rows),
        "profiling_rows": len(profiling),
        "evaluation_rows": len(evaluation),
        "dataset_fingerprint_sha256": manifest["dataset_fingerprint_sha256"],
        "output_sha256": manifest["output_sha256"],
        "near_duplicate_cross_partition_pairs": 0,
        "valid": True,
    }


def token_proxy(text: str, instructions: str) -> int:
    """Conservative preregistration proxy, not provider-reported token usage."""

    return math.ceil((len(text) + len(instructions)) / 4) + 32
