from __future__ import annotations

import math


def observed_selectivity(passing: int, completed: int) -> float | None:
    if passing < 0 or completed < 0 or passing > completed:
        raise ValueError("invalid passing/completed counts")
    return passing / completed if completed else None


def wilson_interval(
    passing: int, completed: int, *, z: float = 1.959963984540054
) -> tuple[float, float] | None:
    if passing < 0 or completed < 0 or passing > completed:
        raise ValueError("invalid passing/completed counts")
    if z <= 0:
        raise ValueError("z must be positive")
    if completed == 0:
        return None
    proportion = passing / completed
    denominator = 1 + z**2 / completed
    center = (proportion + z**2 / (2 * completed)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / completed + z**2 / (4 * completed**2))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)
