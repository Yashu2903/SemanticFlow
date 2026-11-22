from __future__ import annotations

import json

from semanticflow.optimizer.m521_offline import generate


def main() -> int:
    print(json.dumps(generate(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
