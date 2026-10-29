from __future__ import annotations

import argparse
from pathlib import Path

from semanticflow.dataset.preparation import validate_prepared_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and reproduce a prepared dataset")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=1000)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/processed/dataset_manifest.json")
    )
    parser.add_argument(
        "--skip-reproduction-check",
        action="store_true",
        help="Skip only for diagnostics; acceptance validation must not use this option",
    )
    args = parser.parse_args()
    manifest = validate_prepared_dataset(
        args.dataset,
        source=args.source,
        expected_rows=args.expected_rows,
        manifest_path=args.manifest,
        verify_reproduction=not args.skip_reproduction_check,
    )
    print(
        f"Validated {manifest['output_row_count']} rows; reproduction="
        f"{manifest['validation']['reproduction_check_passed']}; manifest={args.manifest}"
    )


if __name__ == "__main__":
    main()

