from __future__ import annotations

import argparse
import json
from pathlib import Path

from semanticflow.optimizer.m51_final_audit import FIGURES, REPORT, ROOT, generate


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the offline M5.1 final research audit")
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--figures", type=Path, default=FIGURES)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    args = parser.parse_args()
    result = generate(
        args.output,
        report_path=args.report,
        figures_path=args.figures,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
