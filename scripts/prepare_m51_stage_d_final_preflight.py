from __future__ import annotations

import argparse
import json
from pathlib import Path

from semanticflow.optimizer.m51_stage_d_preflight import ROOT, generate


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the offline M5.1 Stage D final preflight"
    )
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    result = generate(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
