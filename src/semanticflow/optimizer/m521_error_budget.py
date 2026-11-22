from __future__ import annotations

import hashlib
import json
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class QualificationPolicyVersion(StrEnum):
    M52_LEGACY_STRICT = "m52_legacy_strict"
    M521_NONREDUNDANT = "m5.2.1_nonredundant_v1"
    M521_UNEQUAL = "m5.2.1_unequal_v1"


class QualificationRequirements(BaseModel):
    """Frozen statistical requirements for a qualification decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "m5.2.1-qualification-requirements-v1"
    policy_version: QualificationPolicyVersion
    minimum_records: int = Field(default=40, ge=1)
    minimum_positive_references: int = Field(default=3, ge=0)
    minimum_recall: float = Field(default=0.95, gt=0, lt=1)
    minimum_agreement: float = Field(default=0.98, gt=0, lt=1)
    maximum_fnr: float = Field(default=0.05, gt=0, lt=1)
    maximum_changed_rate: float = Field(default=0.02, gt=0, lt=1)
    family_alpha: float = Field(default=0.05, gt=0, lt=1)
    candidate_count: int = Field(default=4, ge=1)
    checkpoint_count: int = Field(default=8, ge=1)
    distinct_estimands: int = Field(default=4, ge=1)
    recall_alpha_fraction: float = Field(default=0.5, gt=0, lt=1)
    method: str = "one_sided_clopper_pearson_fixed_checkpoint_spending"
    representative_sampling_required: bool = True

    @model_validator(mode="after")
    def validate_complements(self) -> QualificationRequirements:
        if abs(self.minimum_recall + self.maximum_fnr - 1) > 1e-12:
            raise ValueError("recall and FNR thresholds must be complements")
        if abs(self.minimum_agreement + self.maximum_changed_rate - 1) > 1e-12:
            raise ValueError("agreement and change-rate thresholds must be complements")
        if self.policy_version == QualificationPolicyVersion.M52_LEGACY_STRICT:
            if self.distinct_estimands != 4:
                raise ValueError("the historical M5.2 policy allocates four reported bounds")
        elif self.distinct_estimands != 2:
            raise ValueError("prospective nonredundant policies control two estimands")
        return self

    @property
    def recall_alpha(self) -> float:
        base = self.family_alpha / (self.candidate_count * self.checkpoint_count)
        if self.policy_version == QualificationPolicyVersion.M52_LEGACY_STRICT:
            return base / 4
        if self.policy_version == QualificationPolicyVersion.M521_UNEQUAL:
            return base * self.recall_alpha_fraction
        return base / 2

    @property
    def agreement_alpha(self) -> float:
        base = self.family_alpha / (self.candidate_count * self.checkpoint_count)
        if self.policy_version == QualificationPolicyVersion.M52_LEGACY_STRICT:
            return base / 4
        if self.policy_version == QualificationPolicyVersion.M521_UNEQUAL:
            return base * (1 - self.recall_alpha_fraction)
        return base / 2

    @property
    def policy_fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def legacy_requirements() -> QualificationRequirements:
    return QualificationRequirements(
        policy_version=QualificationPolicyVersion.M52_LEGACY_STRICT,
        distinct_estimands=4,
    )


def nonredundant_requirements() -> QualificationRequirements:
    return QualificationRequirements(
        policy_version=QualificationPolicyVersion.M521_NONREDUNDANT,
        distinct_estimands=2,
    )


def unequal_requirements(recall_fraction: float = 0.6) -> QualificationRequirements:
    return QualificationRequirements(
        policy_version=QualificationPolicyVersion.M521_UNEQUAL,
        distinct_estimands=2,
        recall_alpha_fraction=recall_fraction,
    )


def redundancy_audit() -> dict[str, object]:
    return {
        "recall_fnr_same_denominator": True,
        "agreement_change_same_denominator": True,
        "fnr_is_one_minus_recall": True,
        "changed_is_one_minus_agreement": True,
        "exact_bounds_are_complements": True,
        "legacy_simultaneous_events_redundant": True,
        "historical_policy_preserved": True,
        "anytime_valid_confidence_sequence_implemented": False,
        "anytime_valid_decision": (
            "not required for the frozen finite checkpoint family; prospective work may "
            "compare a beta-binomial mixture confidence sequence before changing guarantees"
        ),
        "prospective_change": (
            "control recall and agreement once; report FNR and changed-decision bounds "
            "as mathematical complements"
        ),
    }
