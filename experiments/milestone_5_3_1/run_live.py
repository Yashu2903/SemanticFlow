from __future__ import annotations

import argparse
import json
from pathlib import Path

from semanticflow.optimizer.m531_live import run_live
from semanticflow.optimizer.m531_protocol import (
    DEFAULT_HARD_COST_USD,
    HARD_ATTEMPT_LIMIT,
    HARD_REQUEST_LIMIT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M5.3.1 fresh Jev evidence runner")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/milestone_5_3_1/live_validation_session_01"),
    )
    parser.add_argument("--authorize-paid", action="store_true")
    parser.add_argument("--hard-request-limit", type=int, default=HARD_REQUEST_LIMIT)
    parser.add_argument("--hard-attempt-limit", type=int, default=HARD_ATTEMPT_LIMIT)
    parser.add_argument("--hard-dollar-limit", type=float, default=DEFAULT_HARD_COST_USD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    provider_factory = None
    if args.authorize_paid:
        # Deliberately delayed: a dry run cannot initialize Jev or read credentials.
        from semanticflow.optimizer.m522_jev_adapter import PilotAJevProvider

        def provider_factory() -> PilotAJevProvider:
            return PilotAJevProvider.from_env(model="jev-1.14.0", max_retries=2)

    result = run_live(
        output=args.output,
        authorize_paid=args.authorize_paid,
        provider_factory=provider_factory,
        hard_request_limit=args.hard_request_limit,
        hard_attempt_limit=args.hard_attempt_limit,
        hard_cost_usd=args.hard_dollar_limit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
