from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import httpx

from semanticflow.dataset.preparation import SEMCEB_REPOSITORY, file_sha256

SEMCEB_URL = (
    "https://azimmerer-semceb-datasets.s3.amazonaws.com/"
    "amazon-reviews/reviews_filtered.parquet"
)
MCAULEY_URL = (
    "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/"
    "review_categories/Arts_Crafts_and_Sewing.jsonl.gz"
)


def _download(url: str, destination: Path) -> tuple[str, int]:
    partial = destination.with_name(destination.name + ".part")
    if partial.exists():
        partial.unlink()
    digest = hashlib.sha256()
    total = 0
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    handle.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
        partial.replace(destination)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise
    return digest.hexdigest(), total


def _convert_mcauley(downloaded: Path, output: Path) -> None:
    connection = duckdb.connect(database=":memory:")
    try:
        source = str(downloaded.resolve()).replace("'", "''")
        target = str(output.resolve()).replace("'", "''")
        connection.execute(
            f"""
            COPY (
                SELECT
                    user_id,
                    asin,
                    parent_asin,
                    CAST(rating AS DOUBLE) AS rating,
                    title AS review_title,
                    text AS review_text,
                    CAST(timestamp AS BIGINT) AS timestamp_ms,
                    CAST(helpful_vote AS BIGINT) AS helpful_vote,
                    CAST(verified_purchase AS BOOLEAN) AS verified_purchase,
                    to_json(images) AS images_json
                FROM read_json_auto('{source}', format='newline_delimited')
            ) TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    finally:
        connection.close()


def download_dataset(
    output: str | Path,
    *,
    source: str = "auto",
    overwrite: bool = False,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    selected = source
    semceb_error: str | None = None
    if source in {"auto", "semceb"}:
        try:
            checksum, size = _download(SEMCEB_URL, destination)
            selected = "semceb"
        except (httpx.HTTPError, OSError) as exc:
            if source == "semceb":
                raise RuntimeError(f"SemCEB download failed: {exc}") from exc
            semceb_error = f"{type(exc).__name__}: {exc}"
            selected = "mcauley"

    if selected == "mcauley":
        downloaded = destination.parent / "Arts_Crafts_and_Sewing.jsonl.gz"
        original_checksum, size = _download(MCAULEY_URL, downloaded)
        temporary = destination.with_name(destination.name + ".part")
        _convert_mcauley(downloaded, temporary)
        temporary.replace(destination)
        checksum = file_sha256(destination)
        variant = "McAuley-Lab Amazon Reviews 2023 raw Arts_Crafts_and_Sewing (fallback)"
        repository = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023"
        original_filename = downloaded.name
        distribution_url = MCAULEY_URL
        upstream_checksum = original_checksum
        semceb_filtered = False
    else:
        variant = "SemCEB Arts_Crafts_and_Sewing raw_5core reviews_filtered"
        repository = SEMCEB_REPOSITORY
        original_filename = "reviews_filtered.parquet"
        distribution_url = SEMCEB_URL
        upstream_checksum = checksum
        semceb_filtered = True

    metadata = {
        "source_repository": repository,
        "dataset_variant": variant,
        "original_source_filename": original_filename,
        "distribution_url": distribution_url,
        "local_path": str(destination),
        "local_checksum_sha256": file_sha256(destination),
        "upstream_download_checksum_sha256": upstream_checksum,
        "downloaded_bytes": size,
        "download_timestamp": datetime.now(UTC).isoformat(),
        "is_semceb_filtered_dataset": semceb_filtered,
        "semceb_download_error": semceb_error,
    }
    sidecar = destination.with_suffix(destination.suffix + ".source.json")
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the official SemanticFlow review source")
    parser.add_argument(
        "--output", type=Path, default=Path("data/raw/reviews_filtered.parquet")
    )
    parser.add_argument("--source", choices=("auto", "semceb", "mcauley"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    metadata = download_dataset(args.output, source=args.source, overwrite=args.overwrite)
    print(
        f"Downloaded {metadata['dataset_variant']} to {args.output}; "
        f"sha256={metadata['local_checksum_sha256']}"
    )


if __name__ == "__main__":
    main()
