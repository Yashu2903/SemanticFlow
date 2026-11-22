from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, model_validator


class ExperimentObjective(StrEnum):
    PRODUCTION_QUALIFICATION = "PRODUCTION_QUALIFICATION"
    RESEARCH_INSTRUMENTATION = "RESEARCH_INSTRUMENTATION"


class ObjectivePolicy(BaseModel):
    """Typed objective boundary; research mode is never a quality-check bypass."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "m5.2.2-experiment-objective-v1"
    objective: ExperimentObjective
    permits_diagnostic_acquisition: bool
    permits_formal_certification: bool
    permits_production_deployment: bool
    requires_live_authorization: bool = True

    @model_validator(mode="after")
    def enforce_boundary(self) -> ObjectivePolicy:
        if self.objective == ExperimentObjective.RESEARCH_INSTRUMENTATION:
            if self.permits_formal_certification or self.permits_production_deployment:
                raise ValueError("research instrumentation cannot certify or deploy a plan")
            if not self.permits_diagnostic_acquisition:
                raise ValueError("instrumentation must permit bounded diagnostic acquisition")
        elif not self.permits_formal_certification:
            raise ValueError("production qualification must retain formal qualification")
        return self


def research_instrumentation_policy() -> ObjectivePolicy:
    return ObjectivePolicy(
        objective=ExperimentObjective.RESEARCH_INSTRUMENTATION,
        permits_diagnostic_acquisition=True,
        permits_formal_certification=False,
        permits_production_deployment=False,
    )


def production_qualification_policy() -> ObjectivePolicy:
    return ObjectivePolicy(
        objective=ExperimentObjective.PRODUCTION_QUALIFICATION,
        permits_diagnostic_acquisition=True,
        permits_formal_certification=True,
        permits_production_deployment=True,
    )
