from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from semanticflow.dataset.fusion_benchmark import validate_fusion_dataset
from semanticflow.dataset.loader import ReviewRecord, load_dataset
from semanticflow.executor.fused import FusedExecutor
from semanticflow.optimizer.fusion import (
    ProviderFusionCapabilities,
    build_fused_operator,
)
from semanticflow.optimizer.fusion_cost import GroupStatistics
from semanticflow.optimizer.fusion_planner import (
    enumerate_fusion_plans,
    full_fusion_plan,
    naive_adjacent_partial_plan,
    separate_plan,
)
from semanticflow.optimizer.fusion_profile import (
    FusionProfilingAction,
    FusionProfilingBudget,
    choose_fusion_profiling_actions,
)
from semanticflow.optimizer.fusion_quality import (
    FusionQualityGate,
    compare_binary_decisions,
    evaluate_quality_gate,
)
from semanticflow.providers.mock import MockProvider
from semanticflow.query.models import PredicateSpec
from semanticflow.storage.results import git_commit, git_worktree_dirty

M5_PROTOCOL_SCHEMA_VERSION = "m5-preregistered-protocol-v1"
M5_WORKLOAD_SCHEMA_VERSION = "m5-fusion-workloads-v1"
STAGE_B_AUTHORIZATION = "AUTHORIZE M5 STAGE B SMALL LIVE FUSION PILOT"
STAGE_D_AUTHORIZATION = "AUTHORIZE M5 STAGE D HELD-OUT FUSION EVALUATION"


class FusionDatasetDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    domain: str
    predicates: dict[str, str]


class FusionWorkloadDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    dataset: str
    predicate_ids: list[str] = Field(min_length=2, max_length=4)
    original_order: list[str] = Field(min_length=2, max_length=4)
    characteristics: list[str]

    @model_validator(mode="after")
    def validate_order(self) -> FusionWorkloadDefinition:
        if len(set(self.predicate_ids)) != len(self.predicate_ids):
            raise ValueError("workload predicates must be unique")
        if set(self.original_order) != set(self.predicate_ids):
            raise ValueError("original order must be a predicate permutation")
        return self


class FusionWorkloadSuite(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["m5-fusion-workloads-v1"]
    operation: Literal["and"]
    threshold: float = Field(ge=0.0, le=1.0)
    datasets: dict[str, FusionDatasetDefinition]
    workloads: list[FusionWorkloadDefinition]

    @model_validator(mode="after")
    def validate_references(self) -> FusionWorkloadSuite:
        workload_ids = [workload.id for workload in self.workloads]
        if len(workload_ids) != len(set(workload_ids)):
            raise ValueError("workload IDs must be unique")
        for workload in self.workloads:
            dataset = self.datasets.get(workload.dataset)
            if dataset is None:
                raise ValueError(f"unknown dataset {workload.dataset}")
            missing = set(workload.predicate_ids).difference(dataset.predicates)
            if missing:
                raise ValueError(f"unknown predicates for {workload.id}: {sorted(missing)}")
        return self


def load_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return payload


def load_fusion_suite(path: str | Path) -> FusionWorkloadSuite:
    return FusionWorkloadSuite.model_validate(load_yaml(path))


def load_fusion_protocol(path: str | Path) -> dict[str, Any]:
    protocol = load_yaml(path)
    if protocol.get("schema_version") != M5_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unsupported Milestone 5 protocol")
    return protocol


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_mapping(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def records_for_partition(path: str | Path, partition: str) -> list[ReviewRecord]:
    records = [
        record for record in load_dataset(path) if record.metadata.get("partition") == partition
    ]
    if not records:
        raise ValueError(f"no records in partition {partition}: {path}")
    return records


def suite_predicates(
    suite: FusionWorkloadSuite, dataset_id: str, predicate_ids: Sequence[str] | None = None
) -> list[PredicateSpec]:
    dataset = suite.datasets[dataset_id]
    ids = list(predicate_ids or dataset.predicates)
    return [
        PredicateSpec(id=predicate_id, instructions=dataset.predicates[predicate_id])
        for predicate_id in ids
    ]


def offline_audit(
    *,
    suite: FusionWorkloadSuite,
    protocol_path: str | Path,
    fusion_config_path: str | Path,
) -> dict[str, Any]:
    historical_paths = [
        "configs/optimizer.milestone_4_5_frozen.yaml",
        "results/milestone_4_6/frozen_plans_20261113_budget_corrected/plans.json",
        "results/milestone_4_6/stage_c_restart_execution_audit.json",
        "results/milestone_4_6/stage_c_analysis_20261113_restart.json",
    ]
    historical = {
        path: sha256_file(path) for path in historical_paths if Path(path).is_file()
    }
    dataset_checks = {
        dataset_id: validate_fusion_dataset(dataset.path)
        for dataset_id, dataset in suite.datasets.items()
    }
    source_paths = [
        "src/semanticflow/providers/base.py",
        "src/semanticflow/providers/jev.py",
        "src/semanticflow/executor/sequential.py",
        "src/semanticflow/optimizer/planner.py",
        "src/semanticflow/optimizer/adaptive.py",
        "src/semanticflow/storage/results.py",
        "src/semanticflow/optimizer/fusion.py",
        "src/semanticflow/optimizer/fusion_planner.py",
        "src/semanticflow/optimizer/fusion_cost.py",
        "src/semanticflow/optimizer/fusion_quality.py",
        "src/semanticflow/optimizer/fusion_profile.py",
        "src/semanticflow/optimizer/fusion_analysis.py",
        "src/semanticflow/optimizer/fusion_experiment.py",
        "src/semanticflow/executor/fused.py",
        "src/semanticflow/dataset/fusion_benchmark.py",
    ]
    return {
        "schema_version": "m5-offline-audit-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "git_worktree_dirty": git_worktree_dirty(),
        "network_requests_made": 0,
        "provider_initialized": False,
        "live_jev_calls": 0,
        "protocol_sha256": sha256_file(protocol_path),
        "fusion_config_sha256": sha256_file(fusion_config_path),
        "workload_count": len(suite.workloads),
        "dataset_checks": dataset_checks,
        "historical_artifact_sha256": historical,
        "current_source_sha256": {
            path: sha256_file(path) for path in source_paths if Path(path).is_file()
        },
        "m45_frozen_config_sha256": historical.get(
            "configs/optimizer.milestone_4_5_frozen.yaml"
        ),
        "m46_frozen_plans_sha256": historical.get(
            "results/milestone_4_6/frozen_plans_20261113_budget_corrected/plans.json"
        ),
    }


async def offline_preflight(
    suite: FusionWorkloadSuite,
    fusion_config: Mapping[str, Any],
) -> dict[str, Any]:
    capabilities = ProviderFusionCapabilities(
        provider="mock",
        conjunctive_binary=True,
        multi_decision=True,
        evidence="offline deterministic mock only",
    )
    config_fingerprint = fingerprint_mapping(fusion_config)
    counts: dict[str, int] = {}
    executions: dict[str, Any] = {}
    for workload in suite.workloads:
        predicates = suite_predicates(suite, workload.dataset, workload.predicate_ids)
        plans = enumerate_fusion_plans(
            predicates,
            workload_id=workload.id,
            logical_query_id=workload.id,
            requested_model="mock-deterministic-v1",
            threshold=suite.threshold,
            fusion_mode="conjunctive_binary",
            capabilities=capabilities,
            configuration_fingerprint=config_fingerprint,
            maximum_group_size=int(fusion_config["maximum_group_size"]),
        )
        counts[workload.id] = len(plans)
        expected = {2: 3, 3: 13, 4: 75}[len(predicates)]
        if len(plans) != expected:
            raise ValueError(f"wrong candidate count for {workload.id}")
        full = full_fusion_plan(plans)
        original = separate_plan(plans, workload.original_order)
        naive = naive_adjacent_partial_plan(plans, workload.original_order)
        records = records_for_partition(suite.datasets[workload.dataset].path, "development")[:5]
        provider = MockProvider(
            input_tokens_per_evaluation=120,
            output_tokens_per_evaluation=2,
        )
        result = await FusedExecutor(provider, concurrency=2).execute(
            records,
            full,
            run_id=f"mock-{workload.id}",
            maximum_http_attempts=len(records),
        )
        if not result.complete or result.provider_attempts != len(records):
            raise ValueError(f"mock fused execution failed for {workload.id}")
        executions[workload.id] = {
            "full_fusion_plan": full.plan_id,
            "separate_original_plan": original.plan_id,
            "naive_partial_plan": naive.plan_id,
            "mock_group_evaluations": result.logical_group_evaluations,
            "mock_only": True,
        }

    # Exercise quality, cost, and adaptive-profile plumbing without claiming evidence.
    quality = compare_binary_decisions(
        [True, True, False, False], [True, False, True, False]
    )
    gate = FusionQualityGate.model_validate(fusion_config["quality_gate"])
    gate_result = evaluate_quality_gate(
        quality,
        gate,
        evidence_scope="complete_plan",
        real_fused_measurement=False,
    )
    profiling = choose_fusion_profiling_actions(
        [
            FusionProfilingAction(
                group_id="mock-group",
                predicate_ids=("p1", "p2"),
                records=10,
                estimated_cost_per_request_usd=0.00001,
                optimistic_execution_saving_per_reuse_usd=0.0001,
                required_for_plan_ids=("mock-plan",),
            )
        ],
        budget=FusionProfilingBudget.model_validate(fusion_config["profiling_budget"]),
        reuse_horizon=int(fusion_config["reuse_horizon"]),
    )
    return {
        "schema_version": "m5-offline-preflight-v1",
        "live_jev_calls": 0,
        "provider_initialized": False,
        "mock_results_are_semantic_evidence": False,
        "candidate_counts": counts,
        "mock_executions": executions,
        "quality_gate_fails_nonreal_evidence": not gate_result.eligible,
        "profiling_controller_exercised": bool(profiling.selected_actions),
        "existing_adaptive_default_unchanged": True,
    }


def token_proxy(text: str, instructions: str) -> int:
    return math.ceil((len(text) + len(instructions)) / 4) + 32


def _m46_usage_calibration() -> dict[str, dict[str, float]]:
    """Load frozen separate-call token ratios without treating them as fused evidence."""
    path = Path(
        "results/milestone_4_6/frozen_plans_20261113_budget_corrected/plans.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    calibrated = payload["stage_b_usage_calibration"]["datasets"]
    output: dict[str, dict[str, float]] = {}
    all_mean: list[float] = []
    all_maximum: list[float] = []
    for dataset_id, dataset in calibrated.items():
        predicates = list(dataset["predicates"].values())
        mean_ratio = sum(item["mean_input_token_ratio"] for item in predicates) / len(
            predicates
        )
        maximum_ratio = max(item["maximum_input_token_ratio"] for item in predicates)
        output[dataset_id] = {
            "expected_ratio": mean_ratio,
            "conservative_ratio": maximum_ratio,
        }
        all_mean.append(mean_ratio)
        all_maximum.append(maximum_ratio)
    # SMS has no historical Jev measurements. Cross-domain transfer is an explicit
    # forecast assumption; the conservative proxy uses the largest observed ratio.
    output["sms_spam"] = {
        "expected_ratio": sum(all_mean) / len(all_mean),
        "conservative_ratio": max(all_maximum),
    }
    return output


def build_pilot_budget(
    suite: FusionWorkloadSuite,
    protocol: Mapping[str, Any],
    fusion_config: Mapping[str, Any],
    pilot_groups: Mapping[str, Any],
) -> dict[str, Any]:
    records_per_dataset = int(protocol["pilot"]["records_per_dataset"])
    expected_input_tokens = 0
    conservative_input_tokens = 0
    expected_output_tokens = 0
    schedule: list[dict[str, Any]] = []
    usage_calibration = _m46_usage_calibration()
    for dataset_id, dataset in suite.datasets.items():
        calibration = usage_calibration[dataset_id]
        records = records_for_partition(dataset.path, "pilot")[:records_per_dataset]
        for predicate_id, instructions in dataset.predicates.items():
            tokens = sum(token_proxy(record.review_text, instructions) for record in records)
            calibrated = math.ceil(tokens * calibration["expected_ratio"])
            conservative = math.ceil(tokens * calibration["conservative_ratio"])
            expected_input_tokens += calibrated
            conservative_input_tokens += conservative
            expected_output_tokens += 25 * len(records)
            schedule.append(
                {
                    "kind": "separate_reference",
                    "dataset": dataset_id,
                    "predicate_ids": [predicate_id],
                    "records": len(records),
                    "logical_requests": len(records),
                    "character_proxy_input_tokens": tokens,
                    "expected_input_tokens": calibrated,
                    "conservative_input_tokens": conservative,
                }
            )
        for ids in pilot_groups["conjunctive_binary"][dataset_id]:
            operator = build_fused_operator(
                suite_predicates(suite, dataset_id, ids),
                mode="conjunctive_binary",
                requested_model=fusion_config["provider"]["model"],
                threshold=suite.threshold,
            )
            tokens = sum(
                token_proxy(record.review_text, operator.generated_prompt) for record in records
            )
            calibrated = math.ceil(tokens * calibration["expected_ratio"])
            conservative = math.ceil(tokens * calibration["conservative_ratio"])
            expected_input_tokens += calibrated
            conservative_input_tokens += conservative
            expected_output_tokens += 25 * len(records)
            schedule.append(
                {
                    "kind": "conjunctive_binary",
                    "dataset": dataset_id,
                    "group_id": operator.group_id,
                    "predicate_ids": list(ids),
                    "fusion_prompt_hash": operator.fusion_prompt_hash,
                    "records": len(records),
                    "logical_requests": len(records),
                    "character_proxy_input_tokens": tokens,
                    "expected_input_tokens": calibrated,
                    "conservative_input_tokens": conservative,
                }
            )
        ids = pilot_groups["multi_decision_smoke"]["groups"][dataset_id]
        smoke_count = int(pilot_groups["multi_decision_smoke"]["records_per_group"])
        operator = build_fused_operator(
            suite_predicates(suite, dataset_id, ids),
            mode="multi_decision",
            requested_model=fusion_config["provider"]["model"],
            threshold=suite.threshold,
        )
        smoke = records[:smoke_count]
        instruction_text = "\n".join(operator.source_instructions)
        tokens = sum(token_proxy(record.review_text, instruction_text) for record in smoke)
        calibrated = math.ceil(tokens * calibration["expected_ratio"])
        conservative = math.ceil(tokens * calibration["conservative_ratio"])
        expected_input_tokens += calibrated
        conservative_input_tokens += conservative
        expected_output_tokens += 25 * len(ids) * len(smoke)
        schedule.append(
            {
                "kind": "multi_decision_smoke",
                "dataset": dataset_id,
                "group_id": operator.group_id,
                "predicate_ids": list(ids),
                "fusion_prompt_hash": operator.fusion_prompt_hash,
                "records": len(smoke),
                "logical_requests": len(smoke),
                "character_proxy_input_tokens": tokens,
                "expected_input_tokens": calibrated,
                "conservative_input_tokens": conservative,
            }
        )
    logical = sum(item["logical_requests"] for item in schedule)
    expected_cost = expected_input_tokens * float(
        fusion_config["provider"]["input_price_per_million_usd"]
    ) / 1_000_000
    conservative_cost = conservative_input_tokens * float(
        fusion_config["provider"]["input_price_per_million_usd"]
    ) / 1_000_000
    max_retries = int(fusion_config["provider"]["max_retries"])
    return {
        "schema_version": "m5-stage-b-pilot-budget-v1",
        "schedule": schedule,
        "unique_measurement_arms": len(schedule),
        "logical_requests": logical,
        "maximum_http_attempts": logical * (max_retries + 1),
        "expected_input_tokens": expected_input_tokens,
        "expected_output_tokens": expected_output_tokens,
        "expected_cost_usd": expected_cost,
        "conservative_input_tokens": conservative_input_tokens,
        "conservative_no_retry_cost_proxy_usd": conservative_cost,
        "retry_inclusive_cost_proxy_usd": conservative_cost * (max_retries + 1),
        "token_forecast_method": (
            "character proxy multiplied by frozen M4.6 Stage B mean per-domain ratios; "
            "SMS transfers the cross-domain mean"
        ),
        "conservative_proxy_method": (
            "character proxy multiplied by the per-domain maximum frozen M4.6 ratio; "
            "SMS uses the global maximum"
        ),
        "usage_calibration": usage_calibration,
        "monetary_values_are_empirical_proxies_not_hard_caps": True,
        "hard_limit": "HTTP attempts",
        "model": fusion_config["provider"]["model"],
        "authorization_phrase": STAGE_B_AUTHORIZATION,
        "live_jev_calls": 0,
    }


def build_authorization_manifest(
    *,
    budget: Mapping[str, Any],
    suite_path: str | Path,
    protocol_path: str | Path,
    fusion_config_path: str | Path,
    pilot_groups_path: str | Path,
) -> dict[str, Any]:
    dataset_hashes = {
        dataset_id: sha256_file(dataset.path)
        for dataset_id, dataset in load_fusion_suite(suite_path).datasets.items()
    }
    schedule_hash = fingerprint_mapping({"schedule": budget["schedule"]})
    return {
        "schema_version": "m5-stage-b-authorization-manifest-v1",
        "stage": "stage_b_pilot",
        "authorization_phrase": STAGE_B_AUTHORIZATION,
        "paid_calls_authorized": False,
        "model": budget["model"],
        "logical_request_ceiling": budget["logical_requests"],
        "http_attempt_ceiling": budget["maximum_http_attempts"],
        "schedule_sha256": schedule_hash,
        "budget_sha256": fingerprint_mapping(dict(budget)),
        "dataset_sha256": dataset_hashes,
        "configuration_sha256": {
            "workloads": sha256_file(suite_path),
            "protocol": sha256_file(protocol_path),
            "fusion": sha256_file(fusion_config_path),
            "pilot_groups": sha256_file(pilot_groups_path),
        },
        "runner_sha256": sha256_file("experiments/milestone_5/run_fusion.py"),
        "provider_sha256": sha256_file("src/semanticflow/providers/jev.py"),
        "fusion_source_sha256": {
            path: sha256_file(path)
            for path in (
                "src/semanticflow/optimizer/fusion.py",
                "src/semanticflow/executor/fused.py",
                "src/semanticflow/optimizer/fusion_planner.py",
                "src/semanticflow/optimizer/fusion_cost.py",
                "src/semanticflow/optimizer/fusion_quality.py",
                "src/semanticflow/optimizer/fusion_profile.py",
                "src/semanticflow/optimizer/fusion_analysis.py",
                "src/semanticflow/optimizer/fusion_experiment.py",
                "src/semanticflow/dataset/fusion_benchmark.py",
            )
        },
    }


def require_paid_authorization(
    *,
    stage: Literal["stage_b", "stage_d"],
    supplied: str | None,
) -> None:
    expected = STAGE_B_AUTHORIZATION if stage == "stage_b" else STAGE_D_AUTHORIZATION
    if supplied != expected:
        raise PermissionError(
            f"{stage} is not authorized; supply the separately approved exact phrase"
        )


def validate_stage_b_launch(
    *,
    manifest_path: str | Path,
    authorization: str | None,
    output: str | Path,
    suite_path: str | Path,
    protocol_path: str | Path,
    fusion_config_path: str | Path,
    pilot_groups_path: str | Path,
) -> dict[str, Any]:
    require_paid_authorization(stage="stage_b", supplied=authorization)
    output_path = Path(output)
    if output_path.exists():
        raise FileExistsError(f"pilot output already exists: {output_path}")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest["authorization_phrase"] != STAGE_B_AUTHORIZATION:
        raise ValueError("authorization manifest phrase mismatch")
    expected_configs = {
        "workloads": sha256_file(suite_path),
        "protocol": sha256_file(protocol_path),
        "fusion": sha256_file(fusion_config_path),
        "pilot_groups": sha256_file(pilot_groups_path),
    }
    if manifest["configuration_sha256"] != expected_configs:
        raise ValueError("pilot configuration fingerprint mismatch")
    if manifest["runner_sha256"] != sha256_file("experiments/milestone_5/run_fusion.py"):
        raise ValueError("pilot runner fingerprint mismatch")
    if manifest["provider_sha256"] != sha256_file("src/semanticflow/providers/jev.py"):
        raise ValueError("provider fingerprint mismatch")
    suite = load_fusion_suite(suite_path)
    budget = build_pilot_budget(
        suite,
        load_fusion_protocol(protocol_path),
        load_yaml(fusion_config_path),
        load_yaml(pilot_groups_path),
    )
    if manifest["budget_sha256"] != fingerprint_mapping(budget):
        raise ValueError("pilot budget fingerprint mismatch")
    if manifest["schedule_sha256"] != fingerprint_mapping({"schedule": budget["schedule"]}):
        raise ValueError("pilot schedule fingerprint mismatch")
    if manifest["logical_request_ceiling"] != budget["logical_requests"]:
        raise ValueError("pilot logical-request ceiling mismatch")
    if manifest["http_attempt_ceiling"] != budget["maximum_http_attempts"]:
        raise ValueError("pilot HTTP-attempt ceiling mismatch")
    actual_datasets = {
        dataset_id: sha256_file(dataset.path)
        for dataset_id, dataset in suite.datasets.items()
    }
    if manifest["dataset_sha256"] != actual_datasets:
        raise ValueError("pilot dataset fingerprint mismatch")
    if manifest["http_attempt_ceiling"] <= 0:
        raise ValueError("invalid HTTP attempt ceiling")
    return manifest


def separate_statistics_from_historical(
    group_id: str,
    predicate_ids: tuple[str, ...],
    *,
    selectivity: float,
    mean_input_tokens: float,
    mean_output_tokens: float,
    mean_latency_seconds: float,
    input_price_per_million: float,
) -> GroupStatistics:
    """Explicit model input for compatibility simulations; never fused evidence."""
    sample = 200
    passing = round(selectivity * sample)
    return GroupStatistics(
        group_id=group_id,
        predicate_ids=predicate_ids,
        fusion_mode="single",
        sample_count=sample,
        passing_count=passing,
        empirical_selectivity=passing / sample,
        smoothed_selectivity=(passing + 0.5) / (sample + 1),
        mean_input_tokens=mean_input_tokens,
        mean_output_tokens=mean_output_tokens,
        mean_cost_usd=mean_input_tokens * input_price_per_million / 1_000_000,
        mean_latency_seconds=mean_latency_seconds,
        provider_error_rate=0.0,
        evidence_source="measured_separate",
        evidence_artifact="Milestone 4.6 Stage B profiling matrix",
    )
