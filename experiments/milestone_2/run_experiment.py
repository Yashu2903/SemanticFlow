from __future__ import annotations

import sys

from semanticflow.cli import main

if __name__ == "__main__":
    arguments = [
        "experiment",
        "--provider",
        "mock",
        "--query",
        "configs/queries/product_quality.yaml",
        "--optimizer-config",
        "configs/optimizer.yaml",
        "--output-dir",
        "results/milestone_2",
        *sys.argv[1:],
    ]
    raise SystemExit(main(arguments))
