from __future__ import annotations

import argparse
import json
from pathlib import Path

from semanticflow.optimizer.m52_offline import DOCS, OUTPUT, generate


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the offline-only M5.2 development replay")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--docs", type=Path, default=DOCS)
    args = parser.parse_args()
    print(json.dumps(generate(args.output, args.docs), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
