import json
from pathlib import Path

import pytest

from semanticflow.optimizer.generalization import (
    STAGE_B_AUTHORIZATION,
    STAGE_C_AUTHORIZATION,
    build_abba_schedule,
    candidate_orders,
    canonical_m45_reproduction,
    economics_table,
    heldout_budget,
    historical_artifact_manifest,
    load_protocol,
    load_workload_suite,
    profiling_budget,
    require_paid_authorization,
    strategy_freeze_manifest,
    validate_preflight_inputs,
    workload_freeze_manifest,
)
from semanticflow.optimizer.generalization_analysis import analyze_live_validation

PROTOCOL = "configs/milestone_4_6/protocol.yaml"
WORKLOADS = "configs/milestone_4_6/workloads.yaml"
PREFLIGHT = Path("results/milestone_4_6/offline_preflight_20261112_complete")


def test_workloads_parse_and_enumerate_all_orders() -> None:
    suite = load_workload_suite(WORKLOADS)
    assert len(suite.workloads) == 10
    assert {len(workload.predicate_ids) for workload in suite.workloads} == {2, 3, 4}
    for workload in suite.workloads:
        orders = candidate_orders(workload)
        assert len(orders) in {2, 6, 24}
        assert len({tuple(order) for order in orders}) == len(orders)
        assert tuple(workload.original_order) in {tuple(order) for order in orders}


def test_frozen_strategy_and_workload_fingerprints_verify() -> None:
    suite = load_workload_suite(WORKLOADS)
    protocol = load_protocol(PROTOCOL)
    freeze = strategy_freeze_manifest(
        m4_config=protocol["strategy_sources"]["milestone_4"],
        m45_config=protocol["strategy_sources"]["milestone_4_5_frozen"],
        m45_frozen_manifest="results/milestone_4_5/calibration_20261111/frozen_config.json",
    )
    assert set(freeze["strategies"]) == set(protocol["strategies"])
    assert freeze["m45_parameter_fingerprint"]["declared"] == (
        protocol["strategy_sources"]["expected_m45_parameter_fingerprint"]
    )
    workloads = workload_freeze_manifest(suite)
    assert workloads["held_out_outcomes_observed"] is False
    assert all(item["prompt_hashes"] for item in workloads["workloads"])


def test_preflight_integrity_and_exact_retry_accounting() -> None:
    suite = load_workload_suite(WORKLOADS)
    protocol = load_protocol(PROTOCOL)
    validation = validate_preflight_inputs(suite, protocol)
    assert all(item["valid"] for item in validation["dataset_checks"].values())
    assert validation["adaptive_checkpoints"] == [25, 50, 100, 150, 200]
    budget = profiling_budget(suite, protocol)
    assert budget["logical_decisions"] == 2400
    assert budget["maximum_http_attempts"] == 7200
    assert all(
        item["maximum_http_attempts"] == 3 * item["logical_decisions"]
        for item in budget["datasets"]
    )
    assert budget["paid_calls_executed"] == 0
    assert "proxies" in budget["basis"]


def test_heldout_budget_uses_live_usage_calibration_conservatively() -> None:
    suite = load_workload_suite(WORKLOADS)
    protocol = load_protocol(PROTOCOL)
    mock = json.loads((PREFLIGHT / "mock_preflight.json").read_text(encoding="utf-8"))
    plans = mock["mock_plan_selection"]
    uncalibrated = heldout_budget(suite, protocol, plans)
    plans["stage_b_usage_calibration"] = {
        "datasets": {
            dataset_id: {
                "predicates": {
                    predicate_id: {
                        "mean_input_token_ratio": 2.0,
                        "maximum_input_token_ratio": 3.0,
                        "mean_output_tokens": 4.0,
                        "maximum_output_tokens": 5,
                    }
                    for predicate_id in dataset.predicates
                }
            }
            for dataset_id, dataset in suite.datasets.items()
        }
    }
    calibrated = heldout_budget(suite, protocol, plans)
    assert calibrated["estimated_input_token_ceiling"] == (
        2 * uncalibrated["estimated_input_token_ceiling"]
    )
    assert calibrated["conservative_input_token_ceiling"] == (
        3 * uncalibrated["estimated_input_token_ceiling"]
    )
    assert calibrated["retry_inclusive_cost_proxy_usd"] == pytest.approx(
        3 * calibrated["conservative_no_retry_cost_proxy_usd"]
    )
    assert "Stage B" in calibrated["token_estimation_basis"]


def test_paid_authorization_is_fail_closed() -> None:
    for supplied in (None, "--authorize-paid", "yes"):
        with pytest.raises(PermissionError, match="not authorized"):
            require_paid_authorization("stage_b", supplied)
    require_paid_authorization("stage_b", STAGE_B_AUTHORIZATION)
    with pytest.raises(PermissionError, match="not authorized"):
        require_paid_authorization("stage_c", None)
    require_paid_authorization("stage_c", STAGE_C_AUTHORIZATION)
    with pytest.raises(ValueError, match="unknown paid stage"):
        require_paid_authorization("stage_e", "anything")


def test_abba_deduplicates_identical_strategy_orders() -> None:
    workload = load_workload_suite(WORKLOADS).workloads[0]
    original = workload.original_order
    alternative = list(reversed(original))
    schedule = build_abba_schedule(
        workload,
        {
            "fixed": original,
            "existing_adaptive": alternative,
            "correlation_aware": alternative,
            "risk_aware": original,
        },
    )
    assert [run["order"] for run in schedule["runs"][:4]] == [
        original,
        alternative,
        alternative,
        original,
    ]
    assert schedule["unique_execution_orders"] == 2


def test_cold_and_warm_economics_remain_distinct() -> None:
    rows = economics_table(
        fixed_cost=10.0,
        selected_cost=8.0,
        profiling_cost=5.0,
        optimization_cost=1.0,
        horizons=[1, 5],
    )
    assert rows[0]["cold_start_net_saving_usd"] == -4.0
    assert rows[0]["warm_start_net_saving_usd"] == 1.0
    assert rows[0]["break_even_reuse_count"] == 3
    assert rows[1]["repetitions_are_simulated"] is True


def test_mock_preflight_covers_all_policies_without_jev() -> None:
    report = json.loads((PREFLIGHT / "mock_preflight.json").read_text(encoding="utf-8"))
    assert report["jev_provider_initialized"] is False
    assert report["network_requests_made"] == report["live_jev_calls"] == 0
    assert all(item["matrix_complete"] for item in report["datasets"].values())
    expected = set(load_protocol(PROTOCOL)["strategies"])
    for workload in report["mock_plan_selection"]["workloads"].values():
        assert set(workload["strategies"]) == expected
        original = set(workload["original_order"])
        assert all(
            set(strategy["selected_order"]) == original
            for strategy in workload["strategies"].values()
        )


def test_preflight_freezes_executable_and_dependencies() -> None:
    environment = json.loads((PREFLIGHT / "environment.json").read_text(encoding="utf-8"))
    assert environment["python_version"].startswith("3.11")
    assert environment["packages"]["semanticflow"] == "0.1.0"
    assert all(environment["packages"].values())
    assert len(environment["uv_lock_sha256"]) == 64


def test_historical_results_are_unchanged_and_m45_reproduces() -> None:
    stored = json.loads((PREFLIGHT / "historical_integrity.json").read_text(encoding="utf-8"))
    current = historical_artifact_manifest()
    assert stored["git_tracked_only"] is True
    assert stored["aggregate_sha256"] == current["aggregate_sha256"]
    assert current["file_count"] == stored["file_count"]
    assert all(not Path(item["path"]).is_absolute() for item in current["files"])
    reproduction = canonical_m45_reproduction()
    assert reproduction["verified"] is True
    assert reproduction["reran_paid_inference"] is False


def test_incomplete_live_analysis_never_substitutes_estimates(tmp_path) -> None:
    suite = load_workload_suite(WORKLOADS)
    protocol = load_protocol(PROTOCOL)
    session = tmp_path / "session"
    session.mkdir()
    (session / "authorized_budget.json").write_text(
        json.dumps({"workloads": [{"schedule": {"runs": [{"run_id": "missing"}]}}]}),
        encoding="utf-8",
    )
    report = analyze_live_validation(
        suite=suite,
        protocol=protocol,
        plans={"workloads": {}},
        profile_root=tmp_path / "profiles",
        session_root=session,
    )
    assert report["status"] == "incomplete"
    assert report["performance_claims_permitted"] is False
    assert report["expected_runs"] == 1
    assert report["observed_reports"] == 0


def test_budget_correction_preserves_every_frozen_selection() -> None:
    original = json.loads(
        Path("results/milestone_4_6/frozen_plans_20261113/plans.json").read_text(
            encoding="utf-8"
        )
    )
    corrected = json.loads(
        Path(
            "results/milestone_4_6/frozen_plans_20261113_budget_corrected/plans.json"
        ).read_text(encoding="utf-8")
    )
    assert corrected["workloads"] == original["workloads"]
    assert corrected["held_out_outcomes_observed"] is False
    assert corrected["stage_b_usage_calibration"][
        "changes_optimizer_or_selected_plans"
    ] is False
