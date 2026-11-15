from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from semanticflow.optimizer.generalization import (
    STAGE_C_AUTHORIZATION,
    heldout_budget,
    load_protocol,
    load_workload_suite,
)
from semanticflow.optimizer.stage_c_preflight import (
    build_authorization_manifest,
    load_stage_b_matrix,
    per_run_cost_forecasts,
    profiling_cost_reconciliation,
    support_and_strategy_diagnostics,
    validate_schedule,
    validate_stage_c_launch,
)
from semanticflow.providers.base import BudgetExceeded, RequestBudget
from semanticflow.storage.results import ResultStore

PROFILE_ROOT = Path("results/milestone_4_6/live_profile_20261112_stage_b_approved")
PLAN_PATH = Path(
    "results/milestone_4_6/frozen_plans_20261113_budget_corrected/plans.json"
)
BUDGET_PATH = PLAN_PATH.with_name("stage_c_budget.json")


@pytest.fixture(scope="module")
def context() -> dict[str, object]:
    suite = load_workload_suite("configs/milestone_4_6/workloads.yaml")
    protocol = load_protocol("configs/milestone_4_6/protocol.yaml")
    plans = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    matrices = {
        dataset_id: load_stage_b_matrix(suite, dataset_id, PROFILE_ROOT)
        for dataset_id in suite.datasets
    }
    return {
        "suite": suite,
        "protocol": protocol,
        "plans": plans,
        "matrices": matrices,
    }


@pytest.fixture()
def authorization_manifest(tmp_path: Path, context: dict[str, object]) -> Path:
    payload = build_authorization_manifest(
        suite=context["suite"],  # type: ignore[arg-type]
        protocol=context["protocol"],  # type: ignore[arg-type]
        plans_path=PLAN_PATH,
        budget_path=BUDGET_PATH,
    )
    path = tmp_path / "authorization.json"
    ResultStore.write_json(path, payload)
    return path


def test_stage_b_matrices_are_complete_and_model_filtered(
    context: dict[str, object],
) -> None:
    matrices = context["matrices"]
    assert sum(matrix.size * len(matrix.predicate_ids) for matrix in matrices.values()) == 2400
    assert all(matrix.complete for matrix in matrices.values())


def test_expected_forecast_uses_early_rejection_and_independence_comparison(
    context: dict[str, object],
) -> None:
    forecast = per_run_cost_forecasts(
        suite=context["suite"],  # type: ignore[arg-type]
        protocol=context["protocol"],  # type: ignore[arg-type]
        plans=context["plans"],  # type: ignore[arg-type]
        matrices=context["matrices"],  # type: ignore[arg-type]
        bootstrap_iterations=2,
    )
    assert len(forecast["runs"]) == 42
    assert forecast["aggregate"]["expected_logical_evaluations"] < 103200
    assert forecast["aggregate"]["maximum_logical_evaluations"] == 103200
    assert forecast["aggregate"]["allowed_http_attempts"] == 309600
    for row in forecast["runs"]:
        assert row["correlation_aware"]["expected_logical_evaluations"] <= row[
            "maximum_logical_evaluations"
        ]
        assert row["independence_comparison"]["stages"][0]["reach_probability"] == 1
        assert row["expected_estimated_cost_usd"] <= row[
            "conservative_no_retry_cost_proxy_usd"
        ]


def test_expected_forecast_is_deterministic(context: dict[str, object]) -> None:
    arguments = {
        "suite": context["suite"],
        "protocol": context["protocol"],
        "plans": context["plans"],
        "matrices": context["matrices"],
        "bootstrap_iterations": 2,
    }
    assert per_run_cost_forecasts(**arguments) == per_run_cost_forecasts(**arguments)  # type: ignore[arg-type]


def test_actual_and_counterfactual_profiling_costs_reconcile(
    context: dict[str, object],
) -> None:
    report = profiling_cost_reconciliation(
        suite=context["suite"],  # type: ignore[arg-type]
        protocol=context["protocol"],  # type: ignore[arg-type]
        plans=context["plans"],  # type: ignore[arg-type]
        profile_root=PROFILE_ROOT,
    )
    actual = report["actual_experiment_expenditure"]
    assert actual["unique_provider_requests"] == 2400
    assert actual["cost_usd"] == pytest.approx(0.032638536)
    for strategy, shared in report["shared_statistics_totals_by_strategy"].items():
        standalone = report["standalone_totals_by_strategy"][strategy]
        assert shared["decisions"] <= standalone["decisions"]
        assert shared["cost_usd"] <= standalone["cost_usd"]


def test_sparse_and_negative_replay_diagnostics(context: dict[str, object]) -> None:
    diagnostics = support_and_strategy_diagnostics(
        suite=context["suite"],  # type: ignore[arg-type]
        protocol=context["protocol"],  # type: ignore[arg-type]
        plans=context["plans"],  # type: ignore[arg-type]
        matrices=context["matrices"],  # type: ignore[arg-type]
    )
    prefixes = [
        prefix
        for workload in diagnostics["workloads"]
        for prefix in workload["prefix_diagnostics"]
    ]
    assert any(prefix["zero_support_prefix"] for prefix in prefixes)
    assert any(prefix["smoothing_dominated"] for prefix in prefixes)
    negative_workloads = {
        row["workload_id"]
        for row in diagnostics["negative_profiling_replay_reorderings"]
        if row["strategy"] == "existing_adaptive"
    }
    assert {
        "banking_unauthorized_card_refund",
        "banking_unauthorized_withdrawal_refund",
        "beauty_packaging_value",
    } <= negative_workloads


def test_frozen_schedule_reproduces_counts_and_comparator_coverage(
    context: dict[str, object],
) -> None:
    report = validate_schedule(
        suite=context["suite"],  # type: ignore[arg-type]
        protocol=context["protocol"],  # type: ignore[arg-type]
        plans=context["plans"],  # type: ignore[arg-type]
    )
    assert report["valid"]
    assert report["total_runs"] == 42
    assert report["unique_workload_order_combinations"] == 24
    assert report["required_comparators_missing"] == []
    assert set(report["strategies_mapped"]) == set(context["protocol"]["strategies"])  # type: ignore[index]


@pytest.mark.parametrize("phrase", [None, "", "AUTHORIZE STAGE C"])
def test_stage_c_launch_rejects_missing_or_wrong_authorization(
    phrase: str | None,
    tmp_path: Path,
    context: dict[str, object],
    authorization_manifest: Path,
) -> None:
    with pytest.raises(PermissionError):
        validate_stage_c_launch(
            suite=context["suite"],  # type: ignore[arg-type]
            protocol=context["protocol"],  # type: ignore[arg-type]
            plans_path=PLAN_PATH,
            budget_path=BUDGET_PATH,
            authorization_manifest_path=authorization_manifest,
            authorization_phrase=phrase,
            output_path=tmp_path / "new-output",
        )


def test_stage_c_launch_accepts_only_exact_frozen_scope(
    tmp_path: Path,
    context: dict[str, object],
    authorization_manifest: Path,
) -> None:
    result = validate_stage_c_launch(
        suite=context["suite"],  # type: ignore[arg-type]
        protocol=context["protocol"],  # type: ignore[arg-type]
        plans_path=PLAN_PATH,
        budget_path=BUDGET_PATH,
        authorization_manifest_path=authorization_manifest,
        authorization_phrase=STAGE_C_AUTHORIZATION,
        output_path=tmp_path / "new-output",
    )
    assert result["validated"]
    assert not result["provider_imported"]
    assert result["paid_calls_executed"] == 0


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("plans_sha256", "0" * 64),
        ("schedule_sha256", "1" * 64),
        ("prompt_hashes", {}),
        ("datasets", {}),
    ],
)
def test_stage_c_launch_rejects_plan_schedule_prompt_or_dataset_drift(
    field: str,
    replacement: object,
    tmp_path: Path,
    context: dict[str, object],
    authorization_manifest: Path,
) -> None:
    payload = json.loads(authorization_manifest.read_text(encoding="utf-8"))
    payload[field] = replacement
    ResultStore.write_json(authorization_manifest, payload)
    with pytest.raises(ValueError, match="authorization scope mismatch"):
        validate_stage_c_launch(
            suite=context["suite"],  # type: ignore[arg-type]
            protocol=context["protocol"],  # type: ignore[arg-type]
            plans_path=PLAN_PATH,
            budget_path=BUDGET_PATH,
            authorization_manifest_path=authorization_manifest,
            authorization_phrase=STAGE_C_AUTHORIZATION,
            output_path=tmp_path / "new-output",
        )


def test_stage_c_launch_rejects_invalid_request_ceiling(
    tmp_path: Path, context: dict[str, object]
) -> None:
    budget = json.loads(BUDGET_PATH.read_text(encoding="utf-8"))
    budget["maximum_http_attempts"] += 1
    altered_budget = tmp_path / "budget.json"
    ResultStore.write_json(altered_budget, budget)
    authorization = build_authorization_manifest(
        suite=context["suite"],  # type: ignore[arg-type]
        protocol=context["protocol"],  # type: ignore[arg-type]
        plans_path=PLAN_PATH,
        budget_path=altered_budget,
    )
    authorization_path = tmp_path / "authorization.json"
    ResultStore.write_json(authorization_path, authorization)
    with pytest.raises(ValueError, match="budget or frozen run schedule"):
        validate_stage_c_launch(
            suite=context["suite"],  # type: ignore[arg-type]
            protocol=context["protocol"],  # type: ignore[arg-type]
            plans_path=PLAN_PATH,
            budget_path=altered_budget,
            authorization_manifest_path=authorization_path,
            authorization_phrase=STAGE_C_AUTHORIZATION,
            output_path=tmp_path / "new-output",
        )


def test_stage_c_launch_rejects_conflicting_output(
    tmp_path: Path,
    context: dict[str, object],
    authorization_manifest: Path,
) -> None:
    conflict = tmp_path / "existing"
    conflict.mkdir()
    with pytest.raises(FileExistsError):
        validate_stage_c_launch(
            suite=context["suite"],  # type: ignore[arg-type]
            protocol=context["protocol"],  # type: ignore[arg-type]
            plans_path=PLAN_PATH,
            budget_path=BUDGET_PATH,
            authorization_manifest_path=authorization_manifest,
            authorization_phrase=STAGE_C_AUTHORIZATION,
            output_path=conflict,
        )


@pytest.mark.asyncio
async def test_request_budget_reservation_is_atomic_under_concurrency() -> None:
    budget = RequestBudget(17)

    async def reserve() -> bool:
        try:
            await budget.reserve()
        except BudgetExceeded:
            return False
        return True

    results = await asyncio.gather(*(reserve() for _ in range(100)))
    assert sum(results) == 17
    assert budget.used == 17


def test_atomic_result_writes_leave_no_partial_files(tmp_path: Path) -> None:
    json_path = tmp_path / "one.json"
    jsonl_path = tmp_path / "many.jsonl"
    ResultStore.write_json(json_path, {"value": 1})
    ResultStore.write_jsonl(jsonl_path, [{"value": 1}, {"value": 2}])
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"value": 1}
    assert len(jsonl_path.read_text(encoding="utf-8").splitlines()) == 2
    assert list(tmp_path.glob("*.part")) == []


def test_atomic_result_writes_support_long_stage_c_paths(tmp_path: Path) -> None:
    directory = tmp_path
    for segment in ("w" * 45, "r" * 45, "s" * 45):
        directory /= segment
    directory.mkdir(parents=True)
    target = directory / "decisions.jsonl"
    ResultStore.write_jsonl(target, [{"review_id": "record-1", "passed": True}])
    assert target.is_file()
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "review_id": "record-1",
        "passed": True,
    }
    assert not list(directory.glob(".sf-*.tmp"))


def test_frozen_budget_is_exact_recomputation(context: dict[str, object]) -> None:
    expected = heldout_budget(
        context["suite"],  # type: ignore[arg-type]
        context["protocol"],  # type: ignore[arg-type]
        context["plans"],  # type: ignore[arg-type]
    )
    assert expected == json.loads(BUDGET_PATH.read_text(encoding="utf-8"))
