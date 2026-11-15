from __future__ import annotations

import argparse
from pathlib import Path

from semanticflow.optimizer.m51_selection_preflight import ROOT, generate


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the offline M5.1 selection preflight")
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    summary = generate(args.output)
    print(
        f"{summary['status']}: {summary['logical_measurement_roles']} roles, "
        f"{summary['unique_physical_requests']} unique physical requests, 0 Jev calls"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
