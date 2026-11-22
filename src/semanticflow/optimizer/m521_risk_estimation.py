from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class RiskPolicyType(StrEnum):
    FORMALLY_CERTIFIED = "FORMALLY_CERTIFIED"
    ESTIMATED_RISK_ONLY = "ESTIMATED_RISK_ONLY"
    UNQUALIFIED_FALLBACK = "UNQUALIFIED_FALLBACK"


class RiskEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_type: RiskPolicyType
    failures: int = Field(ge=0)
    trials: int = Field(ge=0)
    posterior_mean_failure_rate: float | None = Field(default=None, ge=0, le=1)
    prior: str | None = None
    formal_frequentist_guarantee: bool = False
    independent_validation: bool = False
    limitation: str


def exploratory_beta_risk(failures: int, trials: int) -> RiskEstimate:
    if failures < 0 or trials < 0 or failures > trials:
        raise ValueError("invalid risk counts")
    # Jeffreys' Beta(1/2, 1/2) prior is explicit and used only for exploration.
    posterior_mean = (failures + 0.5) / (trials + 1) if trials >= 0 else None
    return RiskEstimate(
        policy_type=RiskPolicyType.ESTIMATED_RISK_ONLY,
        failures=failures,
        trials=trials,
        posterior_mean_failure_rate=posterior_mean,
        prior="Jeffreys Beta(0.5, 0.5)",
        limitation=(
            "retrospective exploratory posterior; not a frequentist population guarantee "
            "and not independently validated"
        ),
    )
