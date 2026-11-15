from __future__ import annotations

import argparse

from prepare_m5_datasets import prepare


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare nonoverlapping M5.1 datasets")
    parser.add_argument("--config", default="configs/milestone_5_1/datasets.yaml")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    report = prepare(arguments.config, overwrite=arguments.overwrite)
    for dataset_id, manifest in report["datasets"].items():
        print(dataset_id, manifest["dataset_fingerprint_sha256"])


if __name__ == "__main__":
    main()
