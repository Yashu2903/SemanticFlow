from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    root = Path("results/milestone_5_2/offline_development")
    summary = json.loads((root / "unsafe_selection_analysis.json").read_text("utf-8"))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
