from __future__ import annotations

import argparse
from pathlib import Path

from semanticflow.optimizer.m51_selection_analysis import OUTPUT, analyze


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze the completed M5.1 selection stage")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = analyze(args.output)
    summary = result["summary"]
    print(
        f"{summary['session_status']}: {summary['physical_requests']} requests, "
        f"{summary['input_tokens']} input tokens, ${summary['cost_usd']:.9f}, "
        "0 Stage D calls"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
