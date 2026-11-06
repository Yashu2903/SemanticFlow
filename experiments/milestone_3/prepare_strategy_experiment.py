"""Copy split and profiling artifacts into a new experiment directory without modifying the source.

Lets another plan (or a repeated run) execute against an existing profile while historical
experiment directories stay untouched. Offline only: no provider is constructed.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from semanticflow.optimizer.evidence import file_sha256

REQUIRED = [
    "split_manifest.json",
    "profiling/report.json",
    "profiling/statistics.json",
    "profiling/decisions.jsonl",
    "profiling/manifest.json",
]
PLANNING = ["planning/candidate_plans.json", "planning/selected_plan.json", "planning/report.json"]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source", type=Path, required=True)
    result.add_argument("--destination", type=Path, required=True)
    result.add_argument(
        "--copy-planning",
        action="store_true",
        help="Also copy the source plan (for repeats); omit to re-plan with another config",
    )
    return result


def run(arguments: argparse.Namespace) -> int:
    if arguments.destination.exists():
        raise FileExistsError(f"Refusing to reuse existing directory {arguments.destination}")
    names = REQUIRED + (PLANNING if arguments.copy_planning else [])
    missing = [name for name in names if not (arguments.source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"source experiment lacks {missing}")
    copied = {}
    for name in names:
        target = arguments.destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(arguments.source / name, target)
        source_hash = file_sha256(arguments.source / name)
        if file_sha256(target) != source_hash:
            raise RuntimeError(f"copy of {name} does not match its source")
        copied[name] = source_hash
    provenance = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_experiment": arguments.source.as_posix(),
        "copied_files_sha256": copied,
        "planning_copied": arguments.copy_planning,
        "network_requests_made": 0,
    }
    (arguments.destination / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
