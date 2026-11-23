from __future__ import annotations

from semanticflow.optimizer.m522_protocol import PilotAProtocol
from semanticflow.optimizer.m522_protocol import build_protocol as build_base_protocol

EXPERIMENT_ID = "semanticflow-m5.2.3-live-pilot-a-instrumentation"
LIVE_OUTPUT_DIRECTORY = "results/milestone_5_2_3/live_pilot_a"


def build_protocol() -> PilotAProtocol:
    """Build the M5.2.3 scope without changing the frozen M5.2.2 defaults."""

    return build_base_protocol(
        experiment_id=EXPERIMENT_ID,
        output_directory=LIVE_OUTPUT_DIRECTORY,
    )
