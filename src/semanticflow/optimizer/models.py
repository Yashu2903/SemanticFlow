from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from semanticflow.metrics.collector import DecisionTrace

OptimizationMetric = Literal["estimated_cost_usd", "total_tokens", "input_tokens", "evaluations"]
CardinalityEstimatorName = Literal["independent", "correlation_aware"]
SelectionPolicyName = Literal["minimum_expected_objective", "confidence_aware"]


class SplitSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = 2027
    profiling_size: int = Field(default=200, gt=0)
    evaluation_size: int = Field(default=800, gt=0)


class ProfilingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    concurrency: int = Field(default=4, gt=0)
    max_attempts: int = Field(default=1800, gt=0)


class ExecutionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_attempts_per_run: int = Field(default=2640, gt=0)
    cache_enabled: bool = False


COST_MODEL_PREFIXES: dict[str, str] = {
    "independent": "independent-",
    "correlation_aware": "correlation-aware-",
}


class PlannerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    objective: OptimizationMetric = "total_tokens"
    tie_relative_tolerance: float = Field(default=1e-9, ge=0.0)
    cost_model_version: str = "independent-and-v1"
    optimizer_version: str = "exhaustive-order-v1"
    statistics_version: str = "unconditional-profile-v1"
    cardinality_estimator: CardinalityEstimatorName = "independent"
    selection_policy: SelectionPolicyName = "minimum_expected_objective"

    @model_validator(mode="after")
    def cost_model_matches_estimator(self) -> PlannerSettings:
        prefix = COST_MODEL_PREFIXES[self.cardinality_estimator]
        if not self.cost_model_version.startswith(prefix):
            raise ValueError(
                f"cost_model_version for the {self.cardinality_estimator} estimator "
                f"must start with {prefix!r}"
            )
        return self


class CorrelationSettings(BaseModel):
    """Sparse-observation policy for the correlation-aware estimator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    joint_statistics_version: str = "joint-profile-v1"
    # Beta pseudo-count. 1.0 gives the Jeffreys posterior mean for marginals.
    prior_strength: float = Field(default=1.0, ge=0.0)
    # Conditioning sets with fewer rows are labelled low confidence.
    minimum_conditional_support: int = Field(default=20, ge=1)
    conditional_usage: bool = True
    usage_prior_strength: float = Field(default=5.0, ge=0.0)
    allow_incomplete_matrix: bool = False


class EconomicsSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # None means unknown reuse; the conservative default horizon is one execution.
    expected_executions: int | None = Field(default=None, ge=0)
    unknown_reuse_default_executions: int = Field(default=1, ge=0)
    profiling_budget_usd: float | None = Field(default=None, ge=0.0)
    minimum_net_savings_usd: float = 0.0
    maximum_relative_uncertainty: float | None = Field(default=None, ge=0.0)
    minimum_savings_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    statistics_reusable: bool = True
    optimization_overhead_usd: float = Field(default=0.0, ge=0.0)

    def horizon(self) -> int:
        if self.expected_executions is None:
            return self.unknown_reuse_default_executions
        return self.expected_executions


class BootstrapSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    iterations: int = Field(default=1000, ge=1)
    seed: int = 3031
    interval_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    # A plan is called stable when it wins at least this share of resamples.
    stability_threshold: float = Field(default=0.95, gt=0.0, le=1.0)
    # Confidence-aware selection: required one-sided probability that the
    # cheaper plan beats the original, and the minimum relative saving that is
    # treated as practically distinguishable.
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    minimum_relative_saving: float = Field(default=0.0, ge=0.0)


class AdaptiveProfilingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoints: list[int] = Field(default_factory=lambda: [25, 50, 100, 150, 200])
    seed: int = 4049
    bootstrap_iterations: int = Field(default=1000, ge=1)
    stability_threshold: float = Field(default=0.95, gt=0.0, le=1.0)
    profiling_budget_usd: float | None = Field(default=None, ge=0.0)
    maximum_profiling_decisions: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def increasing_checkpoints(self) -> AdaptiveProfilingSettings:
        if not self.checkpoints:
            raise ValueError("adaptive profiling requires at least one checkpoint")
        if any(value <= 0 for value in self.checkpoints):
            raise ValueError("adaptive checkpoints must be positive")
        if any(b <= a for a, b in itertools.pairwise(self.checkpoints)):
            raise ValueError("adaptive checkpoints must be strictly increasing")
        return self


class MockUsageSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens_per_evaluation: int | None = Field(default=100, ge=0)
    output_tokens_per_evaluation: int | None = Field(default=1, ge=0)


class OptimizerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    split: SplitSettings = SplitSettings()
    profiling: ProfilingSettings = ProfilingSettings()
    execution: ExecutionSettings = ExecutionSettings()
    planner: PlannerSettings = PlannerSettings()
    mock_usage: MockUsageSettings = MockUsageSettings()
    correlation: CorrelationSettings = CorrelationSettings()
    economics: EconomicsSettings = EconomicsSettings()
    bootstrap: BootstrapSettings = BootstrapSettings()
    adaptive: AdaptiveProfilingSettings = AdaptiveProfilingSettings()


def load_optimizer_settings(path: str | Path) -> OptimizerSettings:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Optimizer configuration not found: {source}")
    with source.open(encoding="utf-8") as handle:
        payload: Any = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Optimizer configuration must be a YAML mapping")
    return OptimizerSettings.model_validate(payload)


class SplitManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "m2-split-v1"
    dataset_path: str
    dataset_fingerprint: str
    dataset_version: str
    seed: int
    source_record_count: int
    profiling_size: int
    evaluation_size: int
    profiling_ids: list[str]
    evaluation_ids: list[str]
    profiling_fingerprint: str
    evaluation_fingerprint: str
    split_algorithm: str = "sha256-seed-review-id-rank-v1"

    @model_validator(mode="after")
    def validate_membership(self) -> SplitManifest:
        profiling = set(self.profiling_ids)
        evaluation = set(self.evaluation_ids)
        if len(profiling) != self.profiling_size:
            raise ValueError("profiling IDs must be unique and match profiling_size")
        if len(evaluation) != self.evaluation_size:
            raise ValueError("evaluation IDs must be unique and match evaluation_size")
        if profiling & evaluation:
            raise ValueError("profiling and evaluation IDs must be disjoint")
        return self


class PredicateStatistics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    predicate_id: str
    question_hash: str
    sample_size: int = Field(ge=0)
    completed_decisions: int = Field(ge=0)
    passing_decisions: int = Field(ge=0)
    rejected_decisions: int = Field(ge=0)
    estimated_selectivity: float | None = Field(default=None, ge=0.0, le=1.0)
    estimated_rejection_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    wilson_lower: float | None = Field(default=None, ge=0.0, le=1.0)
    wilson_upper: float | None = Field(default=None, ge=0.0, le=1.0)
    average_input_tokens: float | None = Field(default=None, ge=0.0)
    average_output_tokens: float | None = Field(default=None, ge=0.0)
    average_total_tokens: float | None = Field(default=None, ge=0.0)
    average_inference_latency_seconds: float = Field(ge=0.0)
    average_estimated_cost_usd: float | None = Field(default=None, ge=0.0)
    provider_attempts: int = Field(ge=0)
    http_attempts: int = Field(ge=0)
    successful_api_calls: int = Field(ge=0)
    failed_http_attempts: int = Field(ge=0)
    retry_attempts: int = Field(ge=0)
    provider_errors: int = Field(ge=0)
    budget_blocked_evaluations: int = Field(ge=0)
    model_versions: list[str]
    estimate_kind: Literal["empirical_unconditional"] = "empirical_unconditional"

    @model_validator(mode="after")
    def validate_counts(self) -> PredicateStatistics:
        if self.passing_decisions + self.rejected_decisions != self.completed_decisions:
            raise ValueError("pass and rejection counts must equal completed decisions")
        if self.completed_decisions > self.sample_size:
            raise ValueError("completed decisions cannot exceed sample size")
        if self.completed_decisions == 0:
            if any(
                value is not None
                for value in (
                    self.estimated_selectivity,
                    self.estimated_rejection_rate,
                    self.wilson_lower,
                    self.wilson_upper,
                )
            ):
                raise ValueError("zero completed decisions require unknown selectivity")
        elif self.estimated_selectivity is None or self.estimated_rejection_rate is None:
            raise ValueError("completed decisions require selectivity estimates")
        elif not math.isclose(
            self.estimated_selectivity,
            self.passing_decisions / self.completed_decisions,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("selectivity must match completed passing decisions")
        elif not math.isclose(
            self.estimated_rejection_rate,
            1 - self.estimated_selectivity,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("rejection rate must complement selectivity")
        return self


class ProfilingResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    schema_version: str = "m2-profile-v1"
    complete: bool
    incomplete_reason: str | None = None
    query_id: str
    workload_id: str
    dataset_fingerprint: str
    profiling_fingerprint: str
    threshold: float
    requested_model: str
    resolved_model_versions: list[str]
    predicate_order: list[str]
    predicate_instructions: dict[str, str]
    predicate_hashes: dict[str, str]
    sample_size: int
    requested_evaluations: int
    completed_decisions: int
    budget_blocked_evaluations: int
    provider_attempts: int
    http_attempts: int
    successful_api_calls: int
    failed_http_attempts: int
    retry_attempts: int
    provider_errors: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    summed_inference_latency_seconds: float
    wall_time_seconds: float
    estimated_cost_usd: float | None
    pricing: dict[str, Any]
    statistics_version: str
    statistics: list[PredicateStatistics]
    traces: list[DecisionTrace] = Field(default_factory=list, exclude=True)
    errors: list[str] = Field(default_factory=list)
    budget_diagnostics: list[str] = Field(default_factory=list)

    def require_complete(self) -> None:
        if not self.complete:
            raise ValueError(
                f"Profiling is incomplete: {self.incomplete_reason or 'unknown reason'}"
            )
        expected = self.sample_size * len(self.predicate_order)
        if self.completed_decisions != expected:
            raise ValueError(
                f"Profiling requires {expected} decisions; found {self.completed_decisions}"
            )
        by_id = {item.predicate_id: item for item in self.statistics}
        if set(by_id) != set(self.predicate_order):
            raise ValueError("Profiling statistics do not cover every predicate")
        for predicate_id in self.predicate_order:
            item = by_id[predicate_id]
            if item.completed_decisions != self.sample_size:
                raise ValueError(f"Incomplete statistics for predicate {predicate_id}")
            if item.estimated_selectivity is None:
                raise ValueError(f"Unknown selectivity for predicate {predicate_id}")
        if len(self.resolved_model_versions) != 1:
            raise ValueError("Profiling must use exactly one resolved model version")

    def artifact_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"traces"})


class CandidatePlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(ge=1)
    predicate_order: list[str]
    estimated_evaluations: float = Field(ge=0.0)
    estimated_input_tokens: float | None = Field(default=None, ge=0.0)
    estimated_output_tokens: float | None = Field(default=None, ge=0.0)
    estimated_total_tokens: float | None = Field(default=None, ge=0.0)
    estimated_cost_usd: float | None = Field(default=None, ge=0.0)
    estimated_summed_latency_seconds: float = Field(ge=0.0)
    objective: OptimizationMetric
    objective_value: float = Field(ge=0.0)
    estimated_reduction_vs_original_percent: float | None = None
    is_original_order: bool
    selected: bool = False
    cardinality_estimator: CardinalityEstimatorName = "independent"
    estimated_stage_input_records: list[float] | None = None
    stage_estimates: list[dict[str, Any]] | None = None
    confidence: Literal["high", "low"] | None = None


class PlanSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "m2-plan-v1"
    query_id: str
    dataset_fingerprint: str
    profiling_fingerprint: str
    threshold: float
    requested_model: str
    predicate_hashes: dict[str, str]
    statistics_version: str
    cost_model_version: str
    optimizer_version: str
    objective: OptimizationMetric
    evaluation_record_count: int = Field(gt=0)
    original_predicate_order: list[str]
    selected_predicate_order: list[str]
    original_estimated_objective: float = Field(ge=0.0)
    selected_estimated_objective: float = Field(ge=0.0)
    estimated_reduction_percent: float | None = None
    reordered: bool
    candidate_count: int = Field(gt=0)
    candidates: list[CandidatePlan]
    cardinality_estimator: CardinalityEstimatorName = "independent"
    joint_statistics_version: str | None = None
    joint_matrix_fingerprint: str | None = None
    selection_policy: SelectionPolicyName = "minimum_expected_objective"
    minimum_objective_predicate_order: list[str] | None = None
    selection_explanation: dict[str, Any] | None = None

    def artifact_dict(self, *, include_candidates: bool = True) -> dict[str, Any]:
        exclude = None if include_candidates else {"candidates"}
        return self.model_dump(mode="json", exclude=exclude)
