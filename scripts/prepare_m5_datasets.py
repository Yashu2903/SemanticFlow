from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import yaml

from semanticflow.dataset.fusion_benchmark import iter_sms_spam, prepare_fusion_dataset
from semanticflow.dataset.generalization import iter_amazon_reviews, iter_banking77
from semanticflow.dataset.loader import load_dataset


def _download(url: str, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_name(destination.name + ".part")
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return {
        "url": url,
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": digest,
    }


def _reuse_or_download(
    url: str, destination: Path, reusable: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    if reusable is not None and reusable.is_file():
        digest = hashlib.sha256(reusable.read_bytes()).hexdigest()
        return reusable, {
            "url": url,
            "path": str(reusable),
            "bytes": reusable.stat().st_size,
            "sha256": digest,
            "reused_local_source": True,
        }
    return destination, _download(url, destination)


def _combine_banking(train: Path, test: Path, output: Path) -> None:
    if output.exists():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["text", "category", "upstream_split"])
        writer.writeheader()
        for split, source in (("train", train), ("test", test)):
            with source.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    writer.writerow(
                        {
                            "text": row["text"],
                            "category": row.get("category", ""),
                            "upstream_split": split,
                        }
                    )
    temporary.replace(output)


def _extract_sms(archive: Path, output: Path) -> None:
    if output.exists():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        member = next(name for name in bundle.namelist() if name.endswith("SMSSpamCollection"))
        temporary = output.with_name(output.name + ".part")
        with bundle.open(member) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
        temporary.replace(output)


def _historical_texts(paths: list[str]) -> list[str]:
    texts: dict[str, str] = {}
    for path in paths:
        for record in load_dataset(path):
            digest = hashlib.sha256(record.review_text.strip().casefold().encode()).hexdigest()
            texts[digest] = record.review_text
    return list(texts.values())


def prepare(config_path: str | Path, *, overwrite: bool = False) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    raw = Path(config["raw_directory"])
    output = Path(config["output_directory"])
    downloads: list[dict[str, Any]] = []

    reusable = Path("data/raw/milestone_4_6")
    amazon_raw, amazon_download = _reuse_or_download(
        config["datasets"]["amazon_all_beauty"]["raw_url"],
        raw / "All_Beauty.jsonl.gz",
        reusable / "All_Beauty.jsonl.gz",
    )
    downloads.append(amazon_download)
    train, train_download = _reuse_or_download(
        config["datasets"]["banking77_queries"]["train_url"],
        raw / "banking77_train.csv",
        reusable / "banking77_train.csv",
    )
    test, test_download = _reuse_or_download(
        config["datasets"]["banking77_queries"]["test_url"],
        raw / "banking77_test.csv",
        reusable / "banking77_test.csv",
    )
    downloads.extend((train_download, test_download))
    banking_raw = raw / "banking77_combined.csv"
    _combine_banking(train, test, banking_raw)
    sms_raw = raw / "SMSSpamCollection"
    sms_url = config["datasets"]["sms_spam"]["raw_url"]
    if sms_url.lower().endswith(".zip"):
        sms_zip = raw / "sms_spam_collection.zip"
        downloads.append(_download(sms_url, sms_zip))
        _extract_sms(sms_zip, sms_raw)
    else:
        downloads.append(_download(sms_url, sms_raw))

    historical = _historical_texts(config["historical_text_datasets"])
    partition_sizes = config["partition_sizes"]
    sources = {
        "amazon_all_beauty": (iter_amazon_reviews(amazon_raw), amazon_raw),
        "banking77_queries": (iter_banking77(banking_raw), banking_raw),
        "sms_spam": (iter_sms_spam(sms_raw), sms_raw),
    }
    manifests: dict[str, Any] = {}
    for dataset_id, (records, source_path) in sources.items():
        spec = config["datasets"][dataset_id]
        target = output / f"{dataset_id}.parquet"
        if target.exists() and not overwrite:
            manifests[dataset_id] = json.loads(
                target.with_suffix(target.suffix + ".manifest.json").read_text("utf-8")
            )
            continue
        manifests[dataset_id] = prepare_fusion_dataset(
            records,
            target,
            dataset_id=dataset_id,
            source_path=source_path,
            source_url=spec.get("raw_url") or spec.get("train_url"),
            source_repository=spec["source_page"],
            usage_rights=spec["usage_rights"],
            partition_sizes=partition_sizes,
            seed=spec["sampling_seed"],
            historical_texts=historical,
            overwrite=overwrite,
        )
    report = {
        "schema_version": "m5-dataset-preparation-run-v1",
        "config": str(config_path),
        "downloads": downloads,
        "historical_text_count": len(historical),
        "datasets": manifests,
        "live_jev_calls": 0,
    }
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "preparation_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare frozen Milestone 5 fusion datasets")
    parser.add_argument("--config", default="configs/milestone_5/datasets.yaml")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    result = prepare(arguments.config, overwrite=arguments.overwrite)
    for dataset_id, manifest in result["datasets"].items():
        print(dataset_id, manifest["dataset_fingerprint_sha256"])


if __name__ == "__main__":
    main()
