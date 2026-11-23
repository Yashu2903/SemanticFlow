from __future__ import annotations

from experiments.milestone_5_2_2.run_pilot_a import main as run_existing_pilot
from semanticflow.optimizer.m523_preflight import generate
from semanticflow.optimizer.m523_protocol import build_protocol


def main() -> int:
    return run_existing_pilot(
        protocol_builder=build_protocol,
        preflight_generator=generate,
    )


if __name__ == "__main__":
    raise SystemExit(main())
