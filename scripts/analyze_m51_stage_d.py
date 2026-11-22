from __future__ import annotations

import argparse
import json
from pathlib import Path

from semanticflow.optimizer.m51_stage_d_analysis import analyze


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a completed frozen M5.1 Stage D session")
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.session, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
