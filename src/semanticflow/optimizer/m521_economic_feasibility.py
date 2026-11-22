from __future__ import annotations

import math
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class EconomicVerdict(StrEnum):
    JUSTIFIED = "JUSTIFIED"
    UNJUSTIFIED = "UNJUSTIFIED"
    UNKNOWN = "UNKNOWN"


class EconomicFeasibility(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reuse_horizon: int = Field(ge=1)
    execution_saving_per_run_usd: float
    remaining_profiling_cost_usd: float = Field(ge=0)
    sunk_profiling_cost_usd: float = Field(ge=0)
    optimization_cost_usd: float | None = Field(default=None, ge=0)
    qualification_probability: float | None = Field(default=None, ge=0, le=1)
    maximum_gross_benefit_usd: float
    expected_gross_benefit_usd: float | None
    net_benefit_excluding_unmetered_optimization_usd: float | None
    break_even_reuse_horizon: int | None
    verdict: EconomicVerdict
    assumptions: tuple[str, ...]


def evaluate_economic_feasibility(
    *,
    reuse_horizon: int,
    execution_saving_per_run_usd: float,
    remaining_profiling_cost_usd: float,
    sunk_profiling_cost_usd: float = 0,
    qualification_probability: float | None = None,
    optimization_cost_usd: float | None = None,
) -> EconomicFeasibility:
    maximum = reuse_horizon * max(0.0, execution_saving_per_run_usd)
    total_profile = remaining_profiling_cost_usd + sunk_profiling_cost_usd
    expected = None if qualification_probability is None else maximum * qualification_probability
    net = None if expected is None else expected - total_profile
    if execution_saving_per_run_usd <= 0:
        break_even = None
        verdict = EconomicVerdict.UNJUSTIFIED
    else:
        break_even = math.ceil(total_profile / execution_saving_per_run_usd)
        verdict = (
            EconomicVerdict.UNJUSTIFIED
            if maximum <= remaining_profiling_cost_usd
            or (net is not None and net <= 0)
            else EconomicVerdict.JUSTIFIED
        )
    return EconomicFeasibility(
        reuse_horizon=reuse_horizon,
        execution_saving_per_run_usd=execution_saving_per_run_usd,
        remaining_profiling_cost_usd=remaining_profiling_cost_usd,
        sunk_profiling_cost_usd=sunk_profiling_cost_usd,
        optimization_cost_usd=optimization_cost_usd,
        qualification_probability=qualification_probability,
        maximum_gross_benefit_usd=maximum,
        expected_gross_benefit_usd=expected,
        net_benefit_excluding_unmetered_optimization_usd=net,
        break_even_reuse_horizon=break_even,
        verdict=verdict,
        assumptions=(
            "semantic quality remains a hard constraint",
            "maximum benefit is an optimistic decision bound, not a proof",
            (
                "optimization CPU cost is unmetered"
                if optimization_cost_usd is None
                else "optimization cost supplied"
            ),
        ),
    )
