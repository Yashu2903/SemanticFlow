from semanticflow.optimizer.benchmark import (
    build_schedule,
    permutation_labels,
    plan_strategies,
    schedule_preflight,
)
from semanticflow.optimizer.benchmark_stats import compare_workloads, sign_test
from semanticflow.optimizer.models import OptimizerSettings
from semanticflow.storage.results import ResultStore
from tests.m3_support import matrix_from_patterns, profile_for, query, split_for, trace

IDS = ["A", "B", "C"]


def _experiment(tmp_path, patterns):
    matrix = matrix_from_patterns(IDS, patterns, input_tokens={item: 10 for item in IDS})
    split = split_for(matrix, evaluation_size=40)
    profile = profile_for(matrix, input_price=1.0, output_price=0.0)
    payload = profile.model_dump(mode="json")
    payload["estimated_cost_usd"] = matrix.size * 3 * 10 / 1_000_000
    payload["pricing"] = {
        "input_per_million_usd": 1.0,
        "output_per_million_usd": 0.0,
    }
    experiment = tmp_path / "profile"
    (experiment / "profiling").mkdir(parents=True)
    ResultStore.write_json(experiment / "profiling" / "report.json", payload)
    ResultStore.write_json(experiment / "split_manifest.json", split.model_dump(mode="json"))
    ResultStore.write_jsonl(
        experiment / "profiling" / "decisions.jsonl",
        [
            trace(row.review_id, predicate_id, bool(row.outcomes[index]), input_tokens=10)
            for row in matrix.rows
            for index, predicate_id in enumerate(IDS)
        ],
    )
    optimizer = OptimizerSettings.model_validate(
        {
            "planner": {
                "objective": "estimated_cost_usd",
                "cardinality_estimator": "correlation_aware",
                "cost_model_version": "correlation-aware-and-v1",
            },
            "bootstrap": {"iterations": 20, "seed": 5},
            "adaptive": {
                "checkpoints": [10, 20, matrix.size],
                "bootstrap_iterations": 12,
            },
            "execution": {"max_attempts_per_run": 200, "cache_enabled": False},
        }
    )
    return experiment, optimizer, query(IDS)


def test_sign_test_is_exact_and_excludes_empty_trials() -> None:
    assert sign_test(0, 0) is None
    assert sign_test(8, 0) == 1 / 128
    assert sign_test(3, 3) == 1.0
    assert sign_test(5, 1) == sign_test(1, 5)


def test_abba_schedule_pairs_fixed_with_a_distinct_comparator(tmp_path) -> None:
    # A and B always agree; C always passes. Independence underestimates later stages.
    patterns = [(index % 2 == 0, index % 2 == 0, True) for index in range(40)]
    experiment, optimizer, spec = _experiment(tmp_path, patterns)
    strategies = plan_strategies(
        query=spec,
        optimizer=optimizer,
        profile_dir=experiment,
        prior_cost_per_decision=1e-5,
        prior_cost_source="unit-test prior",
    )
    assert strategies["held_out_outcomes_used"] is False
    assert strategies["strategies"]["fixed"]["profiling_decisions"] == 0
    assert strategies["strategies"]["independent"]["profiling_decisions"] == 120
    assert strategies["strategies"]["economics_gated"]["by_horizon"]["1"]["selected_order"] == IDS
    schedule = build_schedule(strategies, orders="selected", abba_blocks=1, seed=1)
    assert schedule["fixed_order"] == IDS
    assert schedule["comparator_order"] != IDS
    roles = [run["role"] for run in schedule["runs"]]
    assert roles[:4] == ["abba_fixed", "abba_comparator", "abba_comparator", "abba_fixed"]
    assert {run["order_key"] for run in schedule["runs"][:4]} == {
        " -> ".join(IDS),
        " -> ".join(schedule["comparator_order"]),
    }
    preflight = schedule_preflight(
        strategies, schedule, optimizer, cost_per_decision=1e-5
    )
    assert preflight["network_requests_made"] == 0
    assert preflight["run_count"] == 4
    assert preflight["max_http_attempts"] == 800


def test_all_orders_schedule_covers_every_permutation(tmp_path) -> None:
    patterns = [(True, False, True) if index % 2 else (False, True, True) for index in range(40)]
    experiment, optimizer, spec = _experiment(tmp_path, patterns)
    strategies = plan_strategies(
        query=spec,
        optimizer=optimizer,
        profile_dir=experiment,
        prior_cost_per_decision=1e-5,
        prior_cost_source="unit-test prior",
    )
    schedule = build_schedule(strategies, orders="all", abba_blocks=2, seed=9)
    assert len(schedule["runs"]) == 8 + 4
    labels = permutation_labels(IDS)
    assert {run["order_key"] for run in schedule["runs"]} == set(labels)
    second = [run["role"] for run in schedule["runs"][4:8]]
    assert second == ["abba_comparator", "abba_fixed", "abba_fixed", "abba_comparator"]


def test_compare_workloads_pools_saved_draws() -> None:
    def fake(name: str, independent_cheaper: bool) -> dict:
        savings = [0.002 if independent_cheaper else -0.001] * 10
        return {
            "workload": name,
            "strategies": {
                "fixed": {
                    "executed": True,
                    "selected_order": ["A", "B"],
                    "execution_cost_usd": 0.02,
                    "savings_vs_fixed_per_execution_usd": 0.0,
                    "savings_vs_fixed_percent": 0.0,
                    "regret_usd": 0.002 if independent_cheaper else 0.0,
                    "relative_regret_percent": 10.0 if independent_cheaper else 0.0,
                    "selected_is_observed_cheapest": not independent_cheaper,
                    "bootstrap_probability_selected_cheapest": 0.1,
                    "profiling_cost_usd": 0.0,
                    "break_even_executions": None,
                },
                "independent": {
                    "executed": True,
                    "selected_order": ["B", "A"],
                    "execution_cost_usd": 0.018 if independent_cheaper else 0.021,
                    "savings_vs_fixed_per_execution_usd": 0.002 if independent_cheaper else -0.001,
                    "savings_vs_fixed_percent": 10.0 if independent_cheaper else -5.0,
                    "regret_usd": 0.0 if independent_cheaper else 0.001,
                    "relative_regret_percent": 0.0 if independent_cheaper else 5.0,
                    "selected_is_observed_cheapest": independent_cheaper,
                    "bootstrap_probability_selected_cheapest": 0.9,
                    "profiling_cost_usd": 0.01,
                    "break_even_executions": 5,
                },
            },
            "strategy_draws": {
                "fixed": {
                    "savings_vs_fixed": [0.0] * 10,
                    "regret": [0.002] * 10,
                    "cost": [0.02] * 10,
                },
                "independent": {
                    "savings_vs_fixed": savings,
                    "regret": [0.0] * 10,
                    "cost": [0.018 if independent_cheaper else 0.021] * 10,
                },
            },
            "cardinality": {
                "independent": {
                    "stage_rows": [
                        {
                            "error": -2.0,
                            "absolute_percent_error": 20.0,
                            "order": "A -> B",
                            "stage": 2,
                            "predicted": 8,
                            "observed": 10,
                        }
                    ],
                    "cost_rows": [
                        {
                            "error": -0.001,
                            "absolute_percent_error": 5.0,
                            "order": "A -> B",
                            "predicted": 0.019,
                            "observed": 0.02,
                        }
                    ],
                    "pairwise_order_concordance": {"concordant": 1, "discordant": 0, "rate": 1.0},
                },
                "correlation_aware": {
                    "stage_rows": [
                        {
                            "error": -0.5,
                            "absolute_percent_error": 5.0,
                            "order": "A -> B",
                            "stage": 2,
                            "predicted": 9.5,
                            "observed": 10,
                        }
                    ],
                    "cost_rows": [
                        {
                            "error": 0.0002,
                            "absolute_percent_error": 1.0,
                            "order": "A -> B",
                            "predicted": 0.0202,
                            "observed": 0.02,
                        }
                    ],
                    "pairwise_order_concordance": {"concordant": 1, "discordant": 0, "rate": 1.0},
                },
            },
            "cardinality_unit_errors": {
                "independent": {"A -> B|2": -2.0},
                "correlation_aware": {"A -> B|2": -0.5},
            },
        }

    pooled = compare_workloads([fake("one", True), fake("two", True)])
    independent = pooled["strategies"]["independent"]
    assert independent["workloads_evaluated"] == 2
    assert independent["selection_accuracy"] == 1.0
    assert independent["total_execution_savings_vs_fixed_usd"] == 0.004
    assert independent["probability_total_savings_positive"] == 1.0
    paired = pooled["cardinality"]["paired_absolute_error"]
    assert paired["correlation_aware_smaller"] == 2
    assert paired["sign_test_p_value"] == 0.5
