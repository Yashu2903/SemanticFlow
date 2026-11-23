from __future__ import annotations

import argparse
import json
from pathlib import Path

from semanticflow.optimizer.m531_analysis import generate_analysis


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze complete M5.3.1 fresh Jev evidence")
    parser.add_argument(
        "--session",
        type=Path,
        default=Path("results/milestone_5_3_1/live_validation_session_01"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/milestone_5_3_1/live_validation_session_01/analysis"),
    )
    args = parser.parse_args()
    print(json.dumps(generate_analysis(args.session, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
