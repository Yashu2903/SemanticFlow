from __future__ import annotations

import argparse
import csv
from pathlib import Path

from semanticflow.dataset import deterministic_sample, load_dataset
from semanticflow.query import load_query


def export_template(
    query_path: Path,
    output: Path,
    sample_size: int,
    seed: int,
    overwrite: bool = False,
) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output}")
    query = load_query(query_path)
    records = deterministic_sample(load_dataset(query.dataset_path), sample_size, seed)
    context = " AND ".join(predicate.instructions for predicate in query.predicates)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["review_id", "review_text", "query_id", "query_context", "match"]
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "review_id": record.review_id,
                    "review_text": record.review_text,
                    "query_id": query.query_id,
                    "query_context": context,
                    "match": "",
                }
            )
    print(f"Wrote {len(records)} unlabeled rows to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an empty human-label CSV")
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    export_template(args.query, args.output, args.sample_size, args.seed, args.overwrite)


if __name__ == "__main__":
    main()

