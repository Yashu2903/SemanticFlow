"""Milestone 4.6 independent-generalization protocol and offline harness helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import json
import math
import platform
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from semanticflow.dataset.generalization import token_proxy, validate_independent_dataset
from semanticflow.dataset.loader import ReviewRecord, load_dataset
from semanticflow.optimizer.m45_calibration import fingerprint_mapping
from semanticflow.optimizer.matrix import MatrixRow, ProfileMatrix
from semanticflow.optimizer.models import load_optimizer_settings
from semanticflow.optimizer.planner import predicate_hash
from semanticflow.optimizer.profiler import PredicateProfiler
from semanticflow.providers.mock import MockProvider
from semanticflow.query.models import PredicateSpec, QuerySpec
from semanticflow.storage.results import git_commit, git_worktree_dirty

M46_PROTOCOL_SCHEMA_VERSION = "m46-preregistered-protocol-v1"
M46_WORKLOAD_SCHEMA_VERSION = "m46-workload-suite-v1"
M46_FREEZE_SCHEMA_VERSION = "m46-strategy-freeze-v1"
M46_PREFLIGHT_SCHEMA_VERSION = "m46-offline-preflight-v1"
STAGE_B_AUTHORIZATION = "AUTHORIZE M4.6 STAGE B JEV PROFILING"
STAGE_C_AUTHORIZATION = "AUTHORIZE M4.6 STAGE C HELD-OUT JEV EVALUATION"
HISTORICAL_MILESTONES = (
    "milestone_1_full_baseline",
    "milestone_2",
    "milestone_3",
    "milestone_4",
    "milestone_4_5",
)
_TEXT_HASH_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".txt", ".yaml", ".yml"}


class PredicateDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    instructions: str = Field(min_length=1)


class DatasetWorkloadDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: Path
    provenance: str
    domain: str
    predicates: dict[str, PredicateDefinition]


class WorkloadDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    dataset: str
    predicate_ids: list[str] = Field(min_length=2, max_length=4)
    original_order: list[str] = Field(min_length=2, max_length=4)
    categories: list[str]
    rationale: str
    hypothesis: str

    @model_validator(mode="after")
    def order_is_permutation(self) -> WorkloadDefinition:
        if len(set(self.predicate_ids)) != len(self.predicate_ids):
            raise ValueError("predicate_ids must be unique")
        if set(self.original_order) != set(self.predicate_ids):
            raise ValueError("original_order must be a predicate_ids permutation")
        return self


class WorkloadSuite(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["m46-workload-suite-v1"]
    threshold: float = Field(ge=0.0, le=1.0)
    operation: Literal["and"]
    datasets: dict[str, DatasetWorkloadDefinition]
    workloads: list[WorkloadDefinition]

    @model_validator(mode="after")
    def references_are_valid(self) -> WorkloadSuite:
        ids = [workload.id for workload in self.workloads]
        if len(ids) != len(set(ids)):
            raise ValueError("workload IDs must be unique")
        for workload in self.workloads:
            if workload.dataset not in self.datasets:
                raise ValueError(f"unknown workload dataset {workload.dataset}")
            bank = self.datasets[workload.dataset].predicates
            missing = set(workload.predicate_ids).difference(bank)
            if missing:
                raise ValueError(f"{workload.id} has unknown predicates {sorted(missing)}")
        return self


def _load_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping in {path}")
    return payload


def load_workload_suite(path: str | Path) -> WorkloadSuite:
    return WorkloadSuite.model_validate(_load_yaml(path))


def load_protocol(path: str | Path) -> dict[str, Any]:
    payload = _load_yaml(path)
    if payload.get("schema_version") != M46_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unsupported Milestone 4.6 protocol schema")
    return payload


def sha256_file(path: str | Path, *, normalize_newlines: bool = False) -> str:
    data = Path(path).read_bytes()
    if normalize_newlines and Path(path).suffix.lower() in _TEXT_HASH_SUFFIXES:
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def _repository_root() -> Path:
    return Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            text=True,
        ).strip()
    )


def _git_tracked_files(*paths: str | Path) -> list[Path]:
    root = _repository_root()
    listed = subprocess.check_output(
        ["git", "ls-files", "-z", "--", *[str(Path(path)) for path in paths]],
        cwd=root,
    )
    files = []
    for raw in listed.split(b"\0"):
        if not raw:
            continue
        path = root / raw.decode()
        if path.is_file():
            files.append(path)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def _source_fingerprint(paths: Sequence[str | Path], parameters: Mapping[str, Any]) -> str:
    payload = {
        "files": {Path(path).as_posix(): sha256_file(path) for path in paths},
        "parameters": parameters,
    }
    return fingerprint_mapping(payload)


def verify_m45_fingerprint(
    config_path: str | Path, frozen_manifest_path: str | Path
) -> dict[str, Any]:
    optimizer = load_optimizer_settings(config_path)
    frozen = json.loads(Path(frozen_manifest_path).read_text(encoding="utf-8"))
    parameters = {
        "correlation.prior_strength": optimizer.correlation.prior_strength,
        "risk.minimum_saving_probability": optimizer.risk.minimum_saving_probability,
        "risk.minimum_relative_saving": optimizer.risk.minimum_relative_saving,
        "adaptive.checkpoints": optimizer.adaptive.checkpoints,
        "joint.voi_decision_rule": optimizer.joint.voi_decision_rule,
        "joint.uncertainty_contraction_exponent": (
            optimizer.joint.uncertainty_contraction_exponent
        ),
        "joint.bootstrap_iterations": optimizer.joint.bootstrap_iterations,
        "joint.voi_scenarios": optimizer.joint.voi_scenarios,
        "bootstrap.iterations": optimizer.bootstrap.iterations,
    }
    calculated = fingerprint_mapping(parameters)
    if calculated != frozen["config_fingerprint"]:
        raise ValueError("Milestone 4.5 parameter fingerprint mismatch")
    return {
        "declared": frozen["config_fingerprint"],
        "calculated": calculated,
        "full_config_sha256": sha256_file(config_path),
        "verified": True,
    }


def strategy_freeze_manifest(
    *,
    m4_config: str | Path,
    m45_config: str | Path,
    m45_frozen_manifest: str | Path,
) -> dict[str, Any]:
    m45 = verify_m45_fingerprint(m45_config, m45_frozen_manifest)
    planner_files = [
        "src/semanticflow/optimizer/planner.py",
        "src/semanticflow/optimizer/cardinality.py",
        "src/semanticflow/optimizer/cost_model.py",
        "src/semanticflow/optimizer/bootstrap.py",
        "src/semanticflow/optimizer/risk.py",
        "src/semanticflow/optimizer/adaptive.py",
        "src/semanticflow/optimizer/joint.py",
    ]
    definitions: dict[str, dict[str, Any]] = {
        "fixed": {"config": None, "overrides": {"policy": "original_order"}},
        "independent": {
            "config": str(m4_config),
            "overrides": {
                "cardinality_estimator": "independent",
                "selection_policy": "minimum_expected_objective",
            },
        },
        "correlation_aware": {
            "config": str(m4_config),
            "overrides": {
                "cardinality_estimator": "correlation_aware",
                "selection_policy": "minimum_expected_objective",
            },
        },
        "existing_adaptive": {
            "config": str(m4_config),
            "overrides": {"controller": "simulate_adaptive_profiling"},
        },
        "confidence_aware": {
            "config": str(m4_config),
            "overrides": {"selection_policy": "confidence_aware"},
        },
        "risk_aware": {
            "config": str(m4_config),
            "overrides": {"selection_policy": "risk_aware"},
        },
        "joint_adaptive_risk": {
            "config": str(m4_config),
            "overrides": {"controller": "simulate_joint_adaptive_risk"},
        },
        "milestone_4_5_frozen": {
            "config": str(m45_config),
            "overrides": {"controller": "simulate_joint_adaptive_risk"},
        },
    }
    strategies = {}
    for name, definition in definitions.items():
        files = list(planner_files)
        if definition["config"]:
            files.append(definition["config"])
        strategies[name] = {
            **definition,
            "source_files": {path: sha256_file(path) for path in files},
            "fingerprint_sha256": _source_fingerprint(files, definition["overrides"]),
        }
    return {
        "schema_version": M46_FREEZE_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "m45_parameter_fingerprint": m45,
        "strategies": strategies,
        "post_outcome_retuning_permitted": False,
    }


def workload_freeze_manifest(suite: WorkloadSuite) -> dict[str, Any]:
    datasets = {}
    for dataset_id, dataset in suite.datasets.items():
        predicates = {
            predicate_id: {
                "instructions": spec.instructions,
                "prompt_sha256": predicate_hash(predicate_id, spec.instructions),
            }
            for predicate_id, spec in dataset.predicates.items()
        }
        manifest_path = dataset.path.with_suffix(dataset.path.suffix + ".manifest.json")
        dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        datasets[dataset_id] = {
            "path": dataset.path.as_posix(),
            "dataset_fingerprint_sha256": dataset_manifest["dataset_fingerprint_sha256"],
            "profiling_fingerprint_sha256": dataset_manifest[
                "profiling_fingerprint_sha256"
            ],
            "evaluation_fingerprint_sha256": dataset_manifest[
                "evaluation_fingerprint_sha256"
            ],
            "predicates": predicates,
        }
    workloads = []
    for workload in suite.workloads:
        prompt_hashes = {
            predicate_id: datasets[workload.dataset]["predicates"][predicate_id][
                "prompt_sha256"
            ]
            for predicate_id in workload.predicate_ids
        }
        payload = {**workload.model_dump(mode="json"), "prompt_hashes": prompt_hashes}
        payload["workload_fingerprint_sha256"] = fingerprint_mapping(payload)
        workloads.append(payload)
    return {
        "schema_version": M46_WORKLOAD_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "threshold": suite.threshold,
        "operation": suite.operation,
        "datasets": datasets,
        "workloads": workloads,
        "held_out_outcomes_observed": False,
    }


def historical_artifact_manifest(root: str | Path = "results") -> dict[str, Any]:
    """Fingerprint git-tracked historical artifacts only.

    A working tree can contain gitignored local reruns. CI checkouts do not, so hashing
    every file on disk made this check pass locally and fail in GitHub Actions. Text
    files are hashed after newline normalization so Windows CRLF checkouts match Linux.
    """
    repo = _repository_root()
    base = Path(root)
    if not base.is_absolute():
        base = repo / base
    files = []
    for path in _git_tracked_files(*[base / milestone for milestone in HISTORICAL_MILESTONES]):
        files.append(
            {
                "path": path.relative_to(repo).as_posix(),
                "sha256": sha256_file(path, normalize_newlines=True),
                "bytes": path.stat().st_size,
            }
        )
    return {
        "schema_version": "m46-historical-integrity-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
        "aggregate_sha256": fingerprint_mapping({item["path"]: item["sha256"] for item in files}),
        "git_tracked_only": True,
        "newlines_normalized": True,
    }


def environment_manifest() -> dict[str, Any]:
    """Pin the executable and installed direct dependencies used for validation."""
    packages = (
        "semanticflow",
        "duckdb",
        "httpx",
        "pyarrow",
        "pydantic",
        "python-dotenv",
        "PyYAML",
        "pytest",
        "pytest-asyncio",
        "ruff",
    )
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    lock = Path("uv.lock")
    project = Path("pyproject.toml")
    return {
        "schema_version": "m46-environment-freeze-v1",
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
        "pyproject_sha256": sha256_file(project),
        "uv_lock_sha256": sha256_file(lock),
        "git_commit": git_commit(),
        "git_worktree_dirty": git_worktree_dirty(),
    }


def _partition_rows(path: Path, partition: str) -> list[dict[str, Any]]:
    table = pq.read_table(path, filters=[("partition", "=", partition)])
    return table.to_pylist()


def profiling_budget(
    suite: WorkloadSuite, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    max_retries = int(protocol["provider"]["max_retries"])
    input_price = float(protocol["provider"]["pricing"]["input_per_million_usd"])
    output_price = float(protocol["provider"]["pricing"]["output_per_million_usd"])
    by_dataset = []
    logical_total = input_tokens = output_tokens = 0
    for dataset_id, dataset in suite.datasets.items():
        rows = _partition_rows(dataset.path, "profiling")
        predicates = dataset.predicates
        decisions = len(rows) * len(predicates)
        estimated_input = sum(
            token_proxy(str(row["review_text"]), predicate.instructions)
            for row in rows
            for predicate in predicates.values()
        )
        estimated_output = decisions
        attempts = decisions * (max_retries + 1)
        by_dataset.append(
            {
                "dataset_id": dataset_id,
                "profiling_records": len(rows),
                "unique_predicates": len(predicates),
                "logical_decisions": decisions,
                "maximum_http_attempts": attempts,
                "estimated_input_tokens": estimated_input,
                "estimated_output_tokens": estimated_output,
                "estimated_cost_usd": (
                    estimated_input * input_price + estimated_output * output_price
                )
                / 1_000_000,
                "retry_inclusive_cost_proxy_usd": (
                    (estimated_input * (max_retries + 1)) * input_price
                    + (estimated_output * (max_retries + 1)) * output_price
                )
                / 1_000_000,
            }
        )
        logical_total += decisions
        input_tokens += estimated_input
        output_tokens += estimated_output
    return {
        "schema_version": "m46-stage-b-budget-v1",
        "basis": (
            "one shared full predicate matrix per dataset; token counts are preregistered "
            "character-based proxies, not Jev measurements"
        ),
        "model": protocol["provider"]["model"],
        "max_retries": max_retries,
        "datasets": by_dataset,
        "logical_decisions": logical_total,
        "maximum_http_attempts": logical_total * (max_retries + 1),
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "estimated_cost_usd": (
            input_tokens * input_price + output_tokens * output_price
        )
        / 1_000_000,
        "retry_inclusive_cost_proxy_usd": (
            input_tokens * (max_retries + 1) * input_price
            + output_tokens * (max_retries + 1) * output_price
        )
        / 1_000_000,
        "pricing": protocol["provider"]["pricing"],
        "paid_calls_executed": 0,
    }


def candidate_orders(workload: WorkloadDefinition) -> list[list[str]]:
    return [list(order) for order in itertools.permutations(workload.original_order)]


def build_abba_schedule(
    workload: WorkloadDefinition,
    selected_orders: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    original = tuple(workload.original_order)
    distinct = {tuple(order) for order in selected_orders.values()}
    distinct.add(original)
    comparator = next(
        (
            tuple(selected_orders[name])
            for name in ("existing_adaptive", "correlation_aware", "risk_aware")
            if name in selected_orders and tuple(selected_orders[name]) != original
        ),
        None,
    )
    runs: list[dict[str, Any]] = []
    replicated: set[tuple[str, ...]] = {original}
    if comparator is not None:
        replicated.add(comparator)
        for role, order in (
            ("abba_fixed_a", original),
            ("abba_comparator_b", comparator),
            ("abba_comparator_b", comparator),
            ("abba_fixed_a", original),
        ):
            runs.append({"role": role, "order": list(order)})
    else:
        runs.append({"role": "fixed_only", "order": list(original)})
    for order in sorted(distinct - replicated):
        runs.append({"role": "selected_order_coverage", "order": list(order)})
    for index, run in enumerate(runs, start=1):
        run["index"] = index
        run["run_id"] = f"{workload.id}-run-{index:02d}"
    return {
        "schema_version": "m46-abba-schedule-v1",
        "workload_id": workload.id,
        "design": (
            "ABBA fixed-versus-primary-comparator plus one execution of each other unique "
            "selected order"
        ),
        "selected_orders": {key: list(value) for key, value in selected_orders.items()},
        "unique_execution_orders": len(distinct),
        "runs": runs,
    }


def economics_table(
    *,
    fixed_cost: float,
    selected_cost: float,
    profiling_cost: float,
    optimization_cost: float,
    horizons: Sequence[int],
) -> list[dict[str, Any]]:
    saving = fixed_cost - selected_cost
    break_even = None
    if saving > 0:
        break_even = math.ceil((profiling_cost + optimization_cost) / saving)
    return [
        {
            "reuse_horizon": horizon,
            "execution_only_saving_usd": horizon * saving,
            "cold_start_net_saving_usd": (
                horizon * saving - profiling_cost - optimization_cost
            ),
            "warm_start_net_saving_usd": horizon * saving - optimization_cost,
            "break_even_reuse_count": break_even,
            "repetitions_are_simulated": True,
        }
        for horizon in horizons
    ]


def validate_preflight_inputs(
    suite: WorkloadSuite, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    dataset_checks = {
        dataset_id: validate_independent_dataset(dataset.path)
        for dataset_id, dataset in suite.datasets.items()
    }
    category_counts = Counter(
        category for workload in suite.workloads for category in workload.categories
    )
    order_counts = {workload.id: len(candidate_orders(workload)) for workload in suite.workloads}
    for workload in suite.workloads:
        expected = 1
        for value in range(2, len(workload.predicate_ids) + 1):
            expected *= value
        if order_counts[workload.id] != expected:
            raise ValueError(f"candidate enumeration failed for {workload.id}")
    return {
        "schema_version": M46_PREFLIGHT_SCHEMA_VERSION,
        "network_requests_made": 0,
        "provider_initialized": False,
        "live_jev_calls": 0,
        "dataset_checks": dataset_checks,
        "workload_count": len(suite.workloads),
        "dataset_count": len(suite.datasets),
        "category_counts": dict(sorted(category_counts.items())),
        "candidate_order_counts": order_counts,
        "adaptive_checkpoints": load_optimizer_settings(
            protocol["strategy_sources"]["milestone_4"]
        ).adaptive.checkpoints,
        "risk_policy_unchanged": True,
        "historical_optimizer_behavior_available": True,
    }


def records_for_partition(path: str | Path, partition: str) -> list[ReviewRecord]:
    """Load exactly one preregistered split from a prepared independent dataset."""
    if partition not in {"profiling", "evaluation"}:
        raise ValueError("partition must be profiling or evaluation")
    records = [
        record
        for record in load_dataset(path)
        if record.metadata.get("partition") == partition
    ]
    if not records:
        raise ValueError(f"dataset has no {partition} records: {path}")
    return records


def dataset_query(
    suite: WorkloadSuite,
    dataset_id: str,
    *,
    partition: str,
    predicate_order: Sequence[str] | None = None,
    max_calls: int | None = None,
) -> QuerySpec:
    """Construct a query without changing any frozen predicate text or threshold."""
    dataset = suite.datasets[dataset_id]
    predicate_ids = list(predicate_order or dataset.predicates)
    if set(predicate_ids).difference(dataset.predicates):
        raise ValueError(f"unknown predicate in query for {dataset_id}")
    records = records_for_partition(dataset.path, partition)
    manifest = json.loads(
        dataset.path.with_suffix(dataset.path.suffix + ".manifest.json").read_text("utf-8")
    )
    return QuerySpec(
        query_id=f"m46-{dataset_id}-{partition}",
        workload_id=f"m46-{dataset_id}-shared-matrix",
        dataset_path=dataset.path,
        dataset_version=manifest["dataset_fingerprint_sha256"],
        dataset_preparation_version=manifest["schema_version"],
        sample_seed=manifest["sampling_seed"],
        operation=suite.operation,
        limit=len(records),
        threshold=suite.threshold,
        max_calls=max_calls or len(records) * len(predicate_ids),
        predicates=[
            PredicateSpec(
                id=predicate_id,
                instructions=dataset.predicates[predicate_id].instructions,
            )
            for predicate_id in predicate_ids
        ],
        config_id=M46_WORKLOAD_SCHEMA_VERSION,
    )


def subset_matrix(matrix: ProfileMatrix, predicate_ids: Sequence[str]) -> ProfileMatrix:
    """Project a shared dataset matrix to one workload while preserving row identity."""
    indices = [matrix.index(predicate_id) for predicate_id in predicate_ids]
    rows = [
        MatrixRow(
            review_id=row.review_id,
            outcomes=tuple(row.outcomes[index] for index in indices),
            input_tokens=tuple(row.input_tokens[index] for index in indices),
            output_tokens=tuple(row.output_tokens[index] for index in indices),
            latency_seconds=tuple(row.latency_seconds[index] for index in indices),
        )
        for row in matrix.rows
    ]
    return ProfileMatrix(predicate_ids, rows, resampled=matrix.resampled)


def require_paid_authorization(stage: str, supplied: str | None) -> None:
    """Fail closed before a Jev provider can be imported or initialized."""
    expected = {
        "stage_b": STAGE_B_AUTHORIZATION,
        "stage_c": STAGE_C_AUTHORIZATION,
    }.get(stage)
    if expected is None:
        raise ValueError(f"unknown paid stage {stage}")
    if supplied != expected:
        raise PermissionError(
            f"{stage} is not authorized; supply the exact separately approved phrase: {expected}"
        )


def canonical_m45_reproduction(
    report_path: str | Path = "results/milestone_4_5/calibration_20261111/report.json",
) -> dict[str, Any]:
    """Verify the canonical stored findings without rerunning or changing the optimizer."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    baseline = report["baseline_metrics"]["joint_adaptive_risk"]
    cardinality = report["calibration_summaries"]["cardinality"]["1.0"]
    coverage = report["calibration_summaries"]["bootstrap"]["1000"]["mean_coverage"]
    observed = {
        "existing_adaptive_r10_net_usd": baseline["R10_existing_adaptive_net"],
        "joint_adaptive_risk_r10_net_usd": baseline["R10_joint_net"],
        "adaptive_risk_r10_net_usd": baseline["R10_existing_adaptive_risk_net"],
        "correlation_concordance": cardinality["mean_correlation_concordance"],
        "independence_concordance": cardinality["mean_independent_concordance"],
        "correlation_regret_tokens": cardinality["mean_correlation_regret_tokens"],
        "independence_regret_tokens": cardinality["mean_independent_regret_tokens"],
        "bootstrap_95_coverage": coverage["0.95"],
    }
    expected = {
        "existing_adaptive_r10_net_usd": 0.10338711600000036,
        "joint_adaptive_risk_r10_net_usd": 0.09694629000000032,
        "adaptive_risk_r10_net_usd": 0.09651255600000004,
        "correlation_concordance": 0.9208333333333334,
        "independence_concordance": 0.9041666666666668,
        "correlation_regret_tokens": 37.9375,
        "independence_regret_tokens": 84.75,
        "bootstrap_95_coverage": 0.7708333333333333,
    }
    if observed != expected or report["baseline_reproduction"] != "confirmed":
        raise ValueError("canonical Milestone 4.5 stored results do not reproduce")
    return {
        "verified": True,
        "source": Path(report_path).as_posix(),
        "source_sha256": sha256_file(report_path),
        "metrics": observed,
        "reran_paid_inference": False,
    }


async def mock_preflight(
    suite: WorkloadSuite,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Exercise shared profiling and leakage guards using deterministic mock decisions."""
    optimizer = load_optimizer_settings(protocol["strategy_sources"]["milestone_4"])
    datasets: dict[str, Any] = {}
    matrices: dict[str, ProfileMatrix] = {}
    for dataset_id, dataset in suite.datasets.items():
        profiling = records_for_partition(dataset.path, "profiling")
        evaluation = records_for_partition(dataset.path, "evaluation")
        query = dataset_query(suite, dataset_id, partition="profiling")
        mock = MockProvider(
            input_tokens_per_evaluation=(
                optimizer.mock_usage.input_tokens_per_evaluation or 100
            ),
            output_tokens_per_evaluation=(
                optimizer.mock_usage.output_tokens_per_evaluation or 1
            ),
        )
        mock_optimizer = optimizer.model_copy(
            update={
                "profiling": optimizer.profiling.model_copy(
                    update={"max_attempts": len(profiling) * len(query.predicates)}
                )
            }
        )
        manifest = json.loads(
            dataset.path.with_suffix(dataset.path.suffix + ".manifest.json").read_text("utf-8")
        )
        result = await PredicateProfiler(mock, mock_optimizer).profile(
            profiling,
            query,
            run_id=f"m46-preflight-{dataset_id}",
            dataset_fingerprint=manifest["dataset_fingerprint_sha256"],
            profiling_fingerprint=manifest["profiling_fingerprint_sha256"],
            pricing=protocol["provider"]["pricing"],
        )
        if not result.complete:
            raise RuntimeError(f"mock profiling failed for {dataset_id}")
        question_hashes = {
            predicate.id: predicate_hash(predicate.id, predicate.instructions)
            for predicate in query.predicates
        }
        matrix = ProfileMatrix.from_traces(
            [trace.to_dict() for trace in result.traces],
            [predicate.id for predicate in query.predicates],
            review_ids=[record.review_id for record in profiling],
            forbidden_review_ids=[record.review_id for record in evaluation],
            model_identifier=mock.model_identifier,
            question_hashes=question_hashes,
        )
        matrices[dataset_id] = matrix
        datasets[dataset_id] = {
            "complete": result.complete,
            "profiling_records": len(profiling),
            "held_out_records_not_read_as_outcomes": len(evaluation),
            "predicates": len(query.predicates),
            "completed_decisions": result.completed_decisions,
            "provider_attempts": result.provider_attempts,
            "http_attempts": result.http_attempts,
            "matrix_complete": matrix.complete,
            "matrix_fingerprint_sha256": matrix.fingerprint(),
            "synthetic_mock_only": True,
        }
    mock_plans = select_frozen_plans(suite, protocol, matrices)
    return {
        "schema_version": "m46-mock-preflight-v1",
        "datasets": datasets,
        "provider": "mock",
        "provider_initialized": True,
        "jev_provider_initialized": False,
        "network_requests_made": 0,
        "live_jev_calls": 0,
        "mock_plan_selection": mock_plans,
        "interpretation": "infrastructure verification only; not semantic performance",
    }


def select_frozen_plans(
    suite: WorkloadSuite,
    protocol: Mapping[str, Any],
    matrices: Mapping[str, ProfileMatrix],
) -> dict[str, Any]:
    """Select all frozen strategies from profiling evidence only."""
    from semanticflow.optimizer.cardinality import (
        build_estimator,
        predicate_statistics_from_matrix,
    )
    from semanticflow.optimizer.cost_model import estimate_plan
    from semanticflow.optimizer.m45_calibration import evaluate_policy_on_fold

    m4 = load_optimizer_settings(protocol["strategy_sources"]["milestone_4"])
    m45 = load_optimizer_settings(protocol["strategy_sources"]["milestone_4_5_frozen"])
    horizon = int(protocol["analysis"]["plan_selection_reuse_horizon"])
    input_price = float(protocol["provider"]["pricing"]["input_per_million_usd"])
    output_price = float(protocol["provider"]["pricing"]["output_per_million_usd"])
    policies = {
        "fixed": ("fixed", m4),
        "independent": ("independent_full", m4),
        "correlation_aware": ("correlation_full", m4),
        "existing_adaptive": ("adaptive", m4),
        "confidence_aware": ("confidence", m4),
        "risk_aware": ("risk_aware", m4),
        "joint_adaptive_risk": ("joint", m4),
        "milestone_4_5_frozen": ("joint", m45),
    }
    selected: dict[str, Any] = {}
    for workload in suite.workloads:
        train = subset_matrix(matrices[workload.dataset], workload.predicate_ids)
        multiplier = math.ceil(800 / train.size)
        validation_proxy = ProfileMatrix(
            train.predicate_ids,
            (list(train.rows) * multiplier)[:800],
            resampled=True,
        )
        dataset = suite.datasets[workload.dataset]
        question_hashes = {
            predicate_id: predicate_hash(
                predicate_id, dataset.predicates[predicate_id].instructions
            )
            for predicate_id in workload.predicate_ids
        }
        predictions = {}
        statistics = predicate_statistics_from_matrix(
            train,
            question_hashes=question_hashes,
            input_price_per_million=input_price,
            output_price_per_million=output_price,
        )
        for estimator in ("independent", "correlation_aware"):
            cardinality = build_estimator(
                estimator,
                train,
                question_hashes=question_hashes,
                correlation=m4.correlation,
                input_price_per_million=input_price,
                output_price_per_million=output_price,
            )
            candidates = [
                estimate_plan(
                    order,
                    statistics,
                    input_record_count=800,
                    objective="estimated_cost_usd",
                    original_order=workload.original_order,
                    estimator=cardinality,
                )
                for order in itertools.permutations(workload.original_order)
            ]
            predictions[estimator] = [
                {
                    "rank": rank,
                    "order": candidate.predicate_order,
                    "estimated_cost_usd": candidate.estimated_cost_usd,
                    "estimated_evaluations": candidate.estimated_evaluations,
                    "estimated_stage_input_records": (
                        candidate.estimated_stage_input_records
                    ),
                }
                for rank, candidate in enumerate(
                    sorted(
                        candidates,
                        key=lambda item: (item.objective_value, item.predicate_order),
                    ),
                    start=1,
                )
            ]
        strategies = {}
        for strategy, (policy, optimizer) in policies.items():
            result = evaluate_policy_on_fold(
                train,
                validation_proxy,
                original=workload.original_order,
                question_hashes=question_hashes,
                optimizer=optimizer,
                policy=policy,
                horizon=horizon,
                input_price=input_price,
                output_price=output_price,
            )
            strategies[strategy] = {
                "selected_order": result["selected_order"],
                "action": result["action"],
                "profiling_decisions": result["profiling_decisions"],
                "profiling_cost_usd": result["profiling_cost_usd"],
                "selection_input": "profiling_partition_only",
            }
        selected[workload.id] = {
            "dataset": workload.dataset,
            "original_order": workload.original_order,
            "candidate_predictions": predictions,
            "strategies": strategies,
        }
    return {
        "schema_version": "m46-frozen-plan-selection-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "reuse_horizon": horizon,
        "held_out_outcomes_observed": False,
        "post_selection_retuning_permitted": False,
        "workloads": selected,
    }


def heldout_budget(
    suite: WorkloadSuite,
    protocol: Mapping[str, Any],
    plans: Mapping[str, Any],
) -> dict[str, Any]:
    """Calculate the frozen Stage C schedule and conservative request ceiling."""
    max_retries = int(protocol["provider"]["max_retries"])
    input_price = float(protocol["provider"]["pricing"]["input_per_million_usd"])
    output_price = float(protocol["provider"]["pricing"]["output_per_million_usd"])
    workload_rows = []
    logical_total = input_total = output_total = 0
    conservative_input_total = conservative_output_total = 0
    calibration = plans.get("stage_b_usage_calibration")
    for workload in suite.workloads:
        plan = plans["workloads"][workload.id]
        selected_orders = {
            strategy: payload["selected_order"]
            for strategy, payload in plan["strategies"].items()
        }
        schedule = build_abba_schedule(workload, selected_orders)
        records = records_for_partition(suite.datasets[workload.dataset].path, "evaluation")
        instructions = suite.datasets[workload.dataset].predicates
        logical = sum(len(records) * len(run["order"]) for run in schedule["runs"])
        input_tokens_float = conservative_input_float = 0.0
        output_tokens_float = conservative_output_float = 0.0
        for run in schedule["runs"]:
            for predicate_id in run["order"]:
                proxies = sum(
                    token_proxy(record.review_text, instructions[predicate_id].instructions)
                    for record in records
                )
                if calibration:
                    usage = calibration["datasets"][workload.dataset]["predicates"][
                        predicate_id
                    ]
                    input_tokens_float += proxies * usage["mean_input_token_ratio"]
                    conservative_input_float += proxies * usage["maximum_input_token_ratio"]
                    output_tokens_float += len(records) * usage["mean_output_tokens"]
                    conservative_output_float += (
                        len(records) * usage["maximum_output_tokens"]
                    )
                else:
                    input_tokens_float += proxies
                    conservative_input_float += proxies
                    output_tokens_float += len(records)
                    conservative_output_float += len(records)
        input_tokens = math.ceil(input_tokens_float)
        output_tokens = math.ceil(output_tokens_float)
        conservative_input = math.ceil(conservative_input_float)
        conservative_output = math.ceil(conservative_output_float)
        workload_rows.append(
            {
                "workload_id": workload.id,
                "dataset": workload.dataset,
                "evaluation_records": len(records),
                "schedule": schedule,
                "logical_decision_ceiling": logical,
                "maximum_http_attempts": logical * (max_retries + 1),
                "estimated_input_token_ceiling": input_tokens,
                "estimated_output_token_ceiling": output_tokens,
                "conservative_input_token_ceiling": conservative_input,
                "conservative_output_token_ceiling": conservative_output,
                "estimated_cost_ceiling_usd": (
                    input_tokens * input_price + output_tokens * output_price
                )
                / 1_000_000,
                "conservative_no_retry_cost_proxy_usd": (
                    conservative_input * input_price
                    + conservative_output * output_price
                )
                / 1_000_000,
                "retry_inclusive_cost_proxy_usd": (
                    conservative_input * (max_retries + 1) * input_price
                    + conservative_output * (max_retries + 1) * output_price
                )
                / 1_000_000,
            }
        )
        logical_total += logical
        input_total += input_tokens
        output_total += output_tokens
        conservative_input_total += conservative_input
        conservative_output_total += conservative_output
    return {
        "schema_version": "m46-stage-c-budget-v1",
        "model": protocol["provider"]["model"],
        "max_retries": max_retries,
        "workloads": workload_rows,
        "logical_decision_ceiling": logical_total,
        "maximum_http_attempts": logical_total * (max_retries + 1),
        "estimated_input_token_ceiling": input_total,
        "estimated_output_token_ceiling": output_total,
        "conservative_input_token_ceiling": conservative_input_total,
        "conservative_output_token_ceiling": conservative_output_total,
        "estimated_cost_ceiling_usd": (
            input_total * input_price + output_total * output_price
        )
        / 1_000_000,
        "conservative_no_retry_cost_proxy_usd": (
            conservative_input_total * input_price
            + conservative_output_total * output_price
        )
        / 1_000_000,
        "retry_inclusive_cost_proxy_usd": (
            conservative_input_total * (max_retries + 1) * input_price
            + conservative_output_total * (max_retries + 1) * output_price
        )
        / 1_000_000,
        "token_estimation_basis": (
            "Stage B per-predicate observed mean and maximum provider-token-to-character-proxy "
            "ratios applied to the frozen evaluation texts"
            if calibration
            else "preregistered character proxy; no live usage calibration available"
        ),
        "pricing": protocol["provider"]["pricing"],
        "paid_calls_executed": 0,
    }
