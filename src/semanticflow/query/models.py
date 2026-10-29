from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PredicateSpec(BaseModel):
    """One semantic yes/no predicate. Order is significant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    instructions: str

    @field_validator("id", "instructions")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


class QuerySpec(BaseModel):
    """Validated, ordered Milestone 1 conjunction configuration."""

    model_config = ConfigDict(extra="forbid")

    query_id: str
    workload_id: str | None = None
    dataset_path: Path
    dataset_version: str = "unspecified"
    dataset_preparation_version: str = "m1-v1"
    sample_seed: int | None = None
    operation: Literal["and"] = "and"
    limit: int = Field(gt=0)
    threshold: float = Field(ge=0.0, le=1.0)
    max_calls: int = Field(gt=0)
    predicates: list[PredicateSpec] = Field(min_length=1, max_length=6)
    config_id: str = "m1-baseline-v1"

    @field_validator("query_id", "dataset_version", "config_id")
    @classmethod
    def required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @model_validator(mode="after")
    def unique_predicates(self) -> QuerySpec:
        ids = [predicate.id for predicate in self.predicates]
        if len(ids) != len(set(ids)):
            raise ValueError("predicate IDs must be unique")
        if self.workload_id is None:
            self.workload_id = self.query_id
        return self


def load_query(path: str | Path) -> QuerySpec:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Query configuration not found: {source}")
    with source.open(encoding="utf-8") as handle:
        payload: Any = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Query configuration must be a YAML mapping")
    return QuerySpec.model_validate(payload)

