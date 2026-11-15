from __future__ import annotations

import argparse
import json

from semanticflow.optimizer.m51_stage_c_analysis import generate_analysis


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline M5.1 Stage C detailed analysis")
    parser.add_argument(
        "--stage-b-root",
        default="results/milestone_5_1/live_batching_pilot_stage_b",
    )
    parser.add_argument(
        "--output",
        default="results/milestone_5_1/stage_c_detailed_analysis",
    )
    parser.add_argument(
        "--report",
        default="docs/milestone_5_1_stage_c_detailed_analysis.md",
    )
    args = parser.parse_args()
    result = generate_analysis(
        stage_b_root=args.stage_b_root,
        output_directory=args.output,
        report_path=args.report,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
