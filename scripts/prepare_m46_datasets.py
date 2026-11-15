"""Retrieve and prepare the two frozen Milestone 4.6 public dataset snapshots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pyarrow.parquet as pq
import yaml

from semanticflow.dataset.generalization import (
    iter_amazon_reviews,
    iter_banking77,
    prepare_independent_dataset,
)

BANKING77_SHA256 = {
    "train": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
}


def _download(url: str, destination: Path, *, overwrite: bool) -> dict[str, Any]:
    if destination.exists() and not overwrite:
        return {
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "reused_existing_file": True,
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    digest = hashlib.sha256()
    size = 0
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=300.0) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": str(destination),
        "bytes": size,
        "sha256": digest.hexdigest(),
        "reused_existing_file": False,
    }


def _historical_texts(path: Path) -> list[str]:
    return [
        str(value)
        for value in pq.read_table(path, columns=["review_text"])["review_text"].to_pylist()
    ]


def _existing_manifest(path: Path) -> dict[str, Any] | None:
    manifest = path.with_suffix(path.suffix + ".manifest.json")
    if path.is_file() and manifest.is_file():
        return json.loads(manifest.read_text(encoding="utf-8"))
    return None


def _combine_banking77(train: Path, test: Path, output: Path) -> None:
    for name, path in (("train", train), ("test", test)):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != BANKING77_SHA256[name]:
            raise ValueError(f"Banking77 {name} checksum mismatch: {digest}")
    temporary = output.with_name(output.name + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target, fieldnames=["text", "category", "upstream_split"]
        )
        writer.writeheader()
        for name, path in (("train", train), ("test", test)):
            with path.open(encoding="utf-8-sig", newline="") as source:
                for row in csv.DictReader(source):
                    writer.writerow(
                        {
                            "text": row["text"],
                            "category": row["category"],
                            "upstream_split": name,
                        }
                    )
    temporary.replace(output)


def prepare_all(
    *, protocol_path: Path, overwrite: bool = False, skip_download: bool = False
) -> dict[str, Any]:
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    raw_root = Path("data/raw/milestone_4_6")
    output_root = Path("data/milestone_4_6")
    historical = _historical_texts(Path("data/processed/reviews_1000.parquet"))
    amazon_raw = raw_root / "All_Beauty.jsonl.gz"
    banking_train = raw_root / "banking77_train.csv"
    banking_test = raw_root / "banking77_test.csv"
    banking_raw = raw_root / "banking77_combined.csv"
    downloads = {}
    if not skip_download:
        downloads["amazon_all_beauty"] = _download(
            protocol["datasets"]["amazon_all_beauty"]["raw_url"],
            amazon_raw,
            overwrite=overwrite,
        )
        banking_spec = protocol["datasets"]["banking77_queries"]
        downloads["banking77_train"] = _download(
            banking_spec["train_url"],
            banking_train,
            overwrite=overwrite,
        )
        downloads["banking77_test"] = _download(
            banking_spec["test_url"],
            banking_test,
            overwrite=overwrite,
        )
        _combine_banking77(banking_train, banking_test, banking_raw)
    elif not banking_raw.is_file():
        _combine_banking77(banking_train, banking_test, banking_raw)
    for source in (amazon_raw, banking_raw):
        if not source.is_file():
            raise FileNotFoundError(f"missing raw source {source}")
    amazon_spec = protocol["datasets"]["amazon_all_beauty"]
    banking_spec = protocol["datasets"]["banking77_queries"]
    amazon_output = output_root / "amazon_all_beauty.parquet"
    banking_output = output_root / "banking77_queries.parquet"
    amazon_existing = _existing_manifest(amazon_output) if not overwrite else None
    banking_existing = _existing_manifest(banking_output) if not overwrite else None
    manifests = {
        "amazon_all_beauty": amazon_existing
        or prepare_independent_dataset(
            iter_amazon_reviews(amazon_raw),
            amazon_output,
            dataset_id="amazon_all_beauty",
            source_path=amazon_raw,
            source_url=amazon_spec["raw_url"],
            source_repository=amazon_spec["source_page"],
            usage_rights=amazon_spec["usage_rights"],
            profiling_size=int(amazon_spec["profiling_size"]),
            evaluation_size=int(amazon_spec["evaluation_size"]),
            seed=int(amazon_spec["sampling_seed"]),
            historical_texts=historical,
            overwrite=overwrite,
        ),
        "banking77_queries": banking_existing
        or prepare_independent_dataset(
            iter_banking77(banking_raw),
            banking_output,
            dataset_id="banking77_queries",
            source_path=banking_raw,
            source_url=f"{banking_spec['train_url']} + {banking_spec['test_url']}",
            source_repository=banking_spec["source_page"],
            usage_rights=banking_spec["usage_rights"],
            profiling_size=int(banking_spec["profiling_size"]),
            evaluation_size=int(banking_spec["evaluation_size"]),
            seed=int(banking_spec["sampling_seed"]),
            historical_texts=historical,
            overwrite=overwrite,
        ),
    }
    report = {
        "schema_version": "m46-dataset-preparation-run-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "protocol": protocol_path.as_posix(),
        "downloads": downloads,
        "datasets": manifests,
        "historical_overlap_source": "data/processed/reviews_1000.parquet",
        "paid_jev_calls": 0,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "preparation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/milestone_4_6/protocol.yaml")
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    result = prepare_all(
        protocol_path=args.protocol,
        overwrite=args.overwrite,
        skip_download=args.skip_download,
    )
    for dataset_id, manifest in result["datasets"].items():
        print(dataset_id, manifest["dataset_fingerprint_sha256"])


if __name__ == "__main__":
    main()
