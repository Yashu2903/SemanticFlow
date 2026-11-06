"""Statistics for live strategy benchmark sessions.

Each scheduled run executes one predicate order on the same held-out reviews, so per-review
costs are paired across orders. The bootstrap resamples reviews jointly for every order and,
for orders executed more than once, draws one replicate run per iteration; intervals therefore
include response variability between repeated runs whenever replicates exist.
"""

from __future__ import annotations

import itertools
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from semanticflow.optimizer.bootstrap import quantile
from semanticflow.optimizer.economics import break_even_executions, break_even_range
from semanticflow.optimizer.evidence import read_jsonl

STATISTICS_SCHEMA_VERSION = "m3-strategy-statistics-v1"
EVIDENCE = {
    "kind": "live_counterbalanced_executions",
    "causal_measurement": True,
    "scope": (
        "differences between orders executed in one session on identical held-out reviews; "
        "response variability is included only through repeated runs of the same order"
    ),
}
COMPARED_ESTIMATORS = ("independent", "correlation_aware")


@dataclass
class RunData:
    session: str
    index: int
    label: str
    role: str
    order_key: str
    started_at: str
    review_costs: list[float]
    total_cost: float
    decisions: int
    stage_inputs: list[int]
    final_matches: set[str]
    outcomes: dict[tuple[str, str], tuple[float, bool]] = field(repr=False)


def _rate(pricing: dict[str, Any], key: str) -> float:
    value = pricing.get(key)
    if value is None:
        raise ValueError(f"pricing lacks {key}")
    return float(value)


def load_session_runs(
    session_dir: Path, *, evaluation_ids: Sequence[str], pricing: dict[str, Any]
) -> tuple[dict[str, Any], list[RunData]]:
    manifest = json.loads((session_dir / "session_manifest.json").read_text(encoding="utf-8"))
    input_rate = _rate(pricing, "input_per_million_usd")
    output_rate = _rate(pricing, "output_per_million_usd")
    position = {review_id: index for index, review_id in enumerate(evaluation_ids)}
    runs = []
    for entry in manifest["runs"]:
        if entry.get("completion_status") != "complete":
            continue
        run_dir = session_dir / entry["run_dir"]
        run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if run_manifest.get("evaluation_record_ids") != list(evaluation_ids):
            raise ValueError(f"{run_dir} used different held-out review IDs")
        if run_manifest.get("executed_predicate_order") != entry["order"]:
            raise ValueError(f"{run_dir} executed a different order than scheduled")
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        costs = [0.0] * len(evaluation_ids)
        outcomes: dict[tuple[str, str], tuple[float, bool]] = {}
        decisions = 0
        for row in read_jsonl(run_dir / "decisions.jsonl"):
            if row.get("input_tokens") is None or row.get("output_tokens") is None:
                raise ValueError(f"{run_dir} has a decision without token usage")
            costs[position[row["review_id"]]] += (
                row["input_tokens"] * input_rate + row["output_tokens"] * output_rate
            ) / 1_000_000
            outcomes[(row["review_id"], row["predicate_id"])] = (
                float(row["probability"]),
                bool(row["passed"]),
            )
            decisions += 1
        stages = sorted(report["predicates"], key=lambda item: item["stage"])
        runs.append(
            RunData(
                session=session_dir.as_posix(),
                index=entry["index"],
                label=entry["label"],
                role=entry["role"],
                order_key=entry["order_key"],
                started_at=entry["started_at"],
                review_costs=costs,
                total_cost=sum(costs),
                decisions=decisions,
                stage_inputs=[int(item["input_record_count"]) for item in stages],
                final_matches={row["review_id"] for row in read_jsonl(run_dir / "matches.jsonl")},
                outcomes=outcomes,
            )
        )
    return manifest, runs


def _interval(values: Sequence[float], level: float) -> dict[str, float]:
    tail = (1 - level) / 2
    return {
        "mean": sum(values) / len(values),
        "lower": quantile(values, tail),
        "upper": quantile(values, 1 - tail),
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _errors(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "mae": None, "mape_percent": None, "mean_signed_error": None}
    relative = [
        row["absolute_percent_error"]
        for row in rows
        if row["absolute_percent_error"] is not None
    ]
    return {
        "count": len(rows),
        "mae": _mean([abs(row["error"]) for row in rows]),
        "mape_percent": _mean(relative) if relative else None,
        "mean_signed_error": _mean([row["error"] for row in rows]),
    }


def sign_test(wins: int, losses: int) -> float | None:
    """Exact two-sided binomial sign test with p = 0.5; ties must be excluded beforehand."""
    trials = wins + losses
    if trials == 0:
        return None
    tail = sum(math.comb(trials, k) for k in range(min(wins, losses) + 1)) / 2**trials
    return min(1.0, 2 * tail)


def _replicate_variability(replicates: Sequence[RunData]) -> dict[str, Any]:
    pairs = []
    for first, second in itertools.combinations(replicates, 2):
        shared = set(first.outcomes) & set(second.outcomes)
        agree = sum(first.outcomes[key][1] == second.outcomes[key][1] for key in shared)
        pairs.append(
            {
                "runs": [first.index, second.index],
                "cost_difference_usd": second.total_cost - first.total_cost,
                "decision_difference": second.decisions - first.decisions,
                "shared_decision_pairs": len(shared),
                "boolean_agreement": agree / len(shared) if shared else None,
                "threshold_crossings": len(shared) - agree,
                "final_match_symmetric_difference": len(
                    first.final_matches ^ second.final_matches
                ),
            }
        )
    totals = [item.total_cost for item in replicates]
    spread = None
    if len(totals) > 1:
        mean = _mean(totals)
        spread = math.sqrt(sum((value - mean) ** 2 for value in totals) / (len(totals) - 1))
    return {
        "replicate_costs_usd": totals,
        "replicate_standard_deviation_usd": spread,
        "pairs": pairs,
    }


def analyze_workload(
    name: str,
    strategies: dict[str, Any],
    session_dirs: Sequence[Path],
    *,
    evaluation_ids: Sequence[str],
    iterations: int = 2000,
    seed: int = 6067,
    interval_level: float = 0.95,
) -> dict[str, Any]:
    """Per-workload statistics, including bootstrap draws for later cross-workload pooling."""
    pricing = strategies["pricing"]
    sessions = []
    runs: list[RunData] = []
    for path in session_dirs:
        manifest, loaded = load_session_runs(path, evaluation_ids=evaluation_ids, pricing=pricing)
        sessions.append(
            {
                "session_dir": path.as_posix(),
                "status": manifest["status"],
                "provider": manifest["provider"],
                "requested_model": manifest["requested_model"],
                "git_commit": manifest["git_commit"],
                "git_worktree_dirty": manifest["git_worktree_dirty"],
                "started_at": manifest["started_at"],
                "finished_at": manifest["finished_at"],
                "completed_runs": len(loaded),
                "scheduled_runs": len(manifest["schedule"]["runs"]),
            }
        )
        runs.extend(loaded)
    if not runs:
        raise ValueError(f"{name}: no completed runs")
    by_order: dict[str, list[RunData]] = {}
    for run in runs:
        by_order.setdefault(run.order_key, []).append(run)
    executed = sorted(by_order)
    all_orders = list(strategies["permutation_labels"])
    fixed = " -> ".join(strategies["original_order"])
    if fixed not in by_order:
        raise ValueError(f"{name}: the fixed order was not executed")
    mean_cost = {key: _mean([item.total_cost for item in by_order[key]]) for key in executed}
    minimum = min(mean_cost.values())

    rng = random.Random(seed)
    size = len(evaluation_ids)
    draws: dict[str, list[float]] = {key: [] for key in executed}
    for _ in range(iterations):
        counts = [0] * size
        for _ in range(size):
            counts[rng.randrange(size)] += 1
        for key in executed:
            replicate = by_order[key][rng.randrange(len(by_order[key]))]
            draws[key].append(
                sum(c * v for c, v in zip(counts, replicate.review_costs, strict=True) if c)
            )
    draw_minimum = [min(draws[key][b] for key in executed) for b in range(iterations)]
    cheapest_frequency = {
        key: sum(draws[key][b] == draw_minimum[b] for b in range(iterations)) / iterations
        for key in executed
    }

    orders = {}
    for key in executed:
        replicates = by_order[key]
        orders[key] = {
            "label": strategies["permutation_labels"][key],
            "replicates": len(replicates),
            "run_indices": [item.index for item in replicates],
            "mean_cost_usd": mean_cost[key],
            "cost_interval_usd": _interval(draws[key], interval_level),
            "mean_decisions": _mean([item.decisions for item in replicates]),
            "mean_stage_inputs": [
                _mean(values)
                for values in zip(
                    *(item.stage_inputs for item in replicates),
                    strict=True,
                )
            ],
            "bootstrap_probability_cheapest": cheapest_frequency[key],
            "final_match_counts": [len(item.final_matches) for item in replicates],
            "variability": _replicate_variability(replicates),
        }
    ranking = sorted(executed, key=lambda key: (mean_cost[key], key))
    fixed_matches = by_order[fixed][0].final_matches
    match_comparison = {
        key: {
            "symmetric_difference_vs_first_fixed_run": len(
                by_order[key][0].final_matches ^ fixed_matches
            ),
            "identical_to_first_fixed_run": by_order[key][0].final_matches == fixed_matches,
        }
        for key in executed
    }

    def evaluate(selected: list[str], profiling_cost: float, horizons: Sequence[int]) -> dict:
        key = " -> ".join(selected)
        if key not in by_order:
            return {"selected_order": selected, "executed": False}
        savings = [draws[fixed][b] - draws[key][b] for b in range(iterations)]
        regret = [draws[key][b] - draw_minimum[b] for b in range(iterations)]
        saving = mean_cost[fixed] - mean_cost[key]
        saving_ci = _interval(savings, interval_level)
        return {
            "selected_order": selected,
            "executed": True,
            "execution_cost_usd": mean_cost[key],
            "savings_vs_fixed_per_execution_usd": saving,
            "savings_vs_fixed_percent": 100 * saving / mean_cost[fixed],
            "savings_interval_usd": saving_ci,
            "probability_cheaper_than_fixed": sum(value > 0 for value in savings) / iterations,
            "regret_usd": mean_cost[key] - minimum,
            "regret_interval_usd": _interval(regret, interval_level),
            "relative_regret_percent": 100 * (mean_cost[key] - minimum) / mean_cost[fixed],
            "selected_is_observed_cheapest": mean_cost[key] == minimum,
            "observed_rank": ranking.index(key) + 1,
            "bootstrap_probability_selected_cheapest": cheapest_frequency[key],
            "profiling_cost_usd": profiling_cost,
            "break_even_executions": break_even_executions(profiling_cost, saving),
            "break_even_range_over_savings_interval": break_even_range(
                profiling_cost, saving_ci["lower"], saving_ci["upper"]
            ),
            "net_savings_by_horizon_usd": {
                str(h): {
                    "point": h * saving - profiling_cost,
                    "lower": h * saving_ci["lower"] - profiling_cost,
                    "upper": h * saving_ci["upper"] - profiling_cost,
                }
                for h in horizons
            },
            "_savings_draws": savings,
            "_regret_draws": regret,
            "_cost_draws": draws[key],
        }

    horizons = strategies["horizons"]
    evaluated: dict[str, dict[str, Any]] = {}
    for strategy, item in strategies["strategies"].items():
        if item["horizon_dependent"]:
            for horizon, entry in item["by_horizon"].items():
                evaluated[f"{strategy}@{horizon}"] = evaluate(
                    entry["selected_order"], entry["profiling_cost_usd"], [int(horizon)]
                ) | {"profiling_decisions": entry["profiling_decisions"]}
        else:
            evaluated[strategy] = evaluate(
                item["selected_order"], item["profiling_cost_usd"], horizons
            ) | {"profiling_decisions": item["profiling_decisions"]}

    cardinality: dict[str, Any] = {}
    unit_rows: dict[str, dict[tuple[str, int], float]] = {}
    for estimator in COMPARED_ESTIMATORS:
        candidates = {
            " -> ".join(item["predicate_order"]): item
            for item in strategies["plans"][estimator]["candidates"]
        }
        stage_rows, cost_rows = [], []
        for key in executed:
            predicted = candidates[key]["estimated_stage_input_records"]
            observed = orders[key]["mean_stage_inputs"]
            for stage, (p, o) in enumerate(zip(predicted, observed, strict=True), start=1):
                if stage == 1:
                    continue
                stage_rows.append(
                    {
                        "order": key,
                        "stage": stage,
                        "predicted": p,
                        "observed": o,
                        "error": p - o,
                        "absolute_percent_error": 100 * abs(p - o) / o if o else None,
                    }
                )
            p_cost = candidates[key]["estimated_cost_usd"]
            cost_rows.append(
                {
                    "order": key,
                    "predicted": p_cost,
                    "observed": mean_cost[key],
                    "error": p_cost - mean_cost[key],
                    "absolute_percent_error": 100 * abs(p_cost - mean_cost[key]) / mean_cost[key],
                }
            )
        concordant = discordant = 0
        for first, second in itertools.combinations(executed, 2):
            observed_sign = mean_cost[first] - mean_cost[second]
            predicted_sign = (
                candidates[first]["estimated_cost_usd"] - candidates[second]["estimated_cost_usd"]
            )
            if observed_sign == 0 or predicted_sign == 0:
                continue
            if (observed_sign > 0) == (predicted_sign > 0):
                concordant += 1
            else:
                discordant += 1
        unit_rows[estimator] = {(row["order"], row["stage"]): row["error"] for row in stage_rows}
        cardinality[estimator] = {
            "stage_input_error": _errors(stage_rows),
            "cost_prediction_error": _errors(cost_rows),
            "pairwise_order_concordance": {
                "concordant": concordant,
                "discordant": discordant,
                "rate": concordant / (concordant + discordant)
                if concordant + discordant
                else None,
            },
            "stage_rows": stage_rows,
            "cost_rows": cost_rows,
        }

    abba = None
    roles = {run.role for run in runs}
    if {"abba_fixed", "abba_comparator"} <= roles:
        fixed_runs = sorted((r for r in runs if r.role == "abba_fixed"), key=lambda r: r.index)
        comp_runs = sorted(
            (r for r in runs if r.role == "abba_comparator"), key=lambda r: r.index
        )
        abba = {
            "comparator_order": comp_runs[0].order_key,
            "fixed_run_costs_usd": [r.total_cost for r in fixed_runs],
            "comparator_run_costs_usd": [r.total_cost for r in comp_runs],
            "abba_savings_estimate_usd": _mean([r.total_cost for r in fixed_runs])
            - _mean([r.total_cost for r in comp_runs]),
            "within_order_drift_usd": {
                "fixed_last_minus_first": fixed_runs[-1].total_cost - fixed_runs[0].total_cost,
                "comparator_last_minus_first": (
                    comp_runs[-1].total_cost - comp_runs[0].total_cost
                ),
            },
            "note": (
                "the symmetric ABBA mean cancels a linear time trend in cost; replicate "
                "differences of the same order measure response variability"
            ),
        }

    result = {
        "schema_version": STATISTICS_SCHEMA_VERSION,
        "workload": name,
        "evidence": EVIDENCE,
        "sessions": sessions,
        "bootstrap": {
            "iterations": iterations,
            "seed": seed,
            "interval_level": interval_level,
            "unit": "held-out review, resampled jointly across orders",
            "replicate_handling": "one replicate run per order drawn uniformly per iteration",
        },
        "orders_executed": len(executed),
        "orders_total": len(all_orders),
        "all_orders_executed": len(executed) == len(all_orders),
        "regret_reference": (
            "all orders" if len(executed) == len(all_orders) else "executed orders only"
        ),
        "fixed_order": fixed,
        "observed_cheapest_order": ranking[0],
        "observed_ranking": ranking,
        "orders": orders,
        "final_match_comparison": match_comparison,
        "abba": abba,
        "strategies": {
            key: {k: v for k, v in value.items() if not k.startswith("_")}
            for key, value in evaluated.items()
        },
        "cardinality": cardinality,
        "cardinality_unit_errors": {
            estimator: {
                f"{order}|{stage}": error for (order, stage), error in units.items()
            }
            for estimator, units in unit_rows.items()
        },
        "strategy_draws": {
            key: {
                "savings_vs_fixed": item["_savings_draws"],
                "regret": item["_regret_draws"],
                "cost": item["_cost_draws"],
            }
            for key, item in evaluated.items()
            if item.get("executed")
        },
        "profiling": {
            "profiling_decisions": strategies["profiling_decisions"],
            "profiling_cost_usd": strategies["profiling_cost_usd"],
            "fixed_execution_cost_usd": mean_cost[fixed],
            "profiling_cost_relative_to_fixed_execution": strategies["profiling_cost_usd"]
            / mean_cost[fixed],
        },
    }
    return result


def compare_workloads(
    analyses: Sequence[Mapping[str, Any]],
    *,
    interval_level: float = 0.95,
) -> dict[str, Any]:
    """Pool strategy outcomes across workloads; bootstrap draws are independent per workload."""
    if not analyses:
        raise ValueError("no workloads to compare")
    names = [result["workload"] for result in analyses]
    strategy_keys = sorted(set.intersection(*(set(result["strategies"]) for result in analyses)))
    per_strategy = {}
    for key in strategy_keys:
        entries = [
            (result["workload"], result, result["strategies"][key]) for result in analyses
        ]
        usable = [
            (workload, result, item)
            for workload, result, item in entries
            if item["executed"]
        ]
        if not usable:
            per_strategy[key] = {"workloads_evaluated": 0}
            continue
        iterations = len(usable[0][1]["strategy_draws"][key]["savings_vs_fixed"])
        total_savings = [
            sum(result["strategy_draws"][key]["savings_vs_fixed"][b] for _, result, _ in usable)
            for b in range(iterations)
        ]
        best = sum(item["selected_is_observed_cheapest"] for _, _, item in usable)
        per_strategy[key] = {
            "workloads_evaluated": len(usable),
            "workloads_without_executed_selection": [
                w for w, _, i in entries if not i["executed"]
            ],
            "selected_observed_cheapest": best,
            "selection_accuracy": best / len(usable),
            "mean_bootstrap_probability_selected_cheapest": _mean(
                [item["bootstrap_probability_selected_cheapest"] for _, _, item in usable]
            ),
            "mean_regret_usd": _mean([item["regret_usd"] for _, _, item in usable]),
            "mean_relative_regret_percent": _mean(
                [item["relative_regret_percent"] for _, _, item in usable]
            ),
            "total_execution_savings_vs_fixed_usd": sum(
                item["savings_vs_fixed_per_execution_usd"] for _, _, item in usable
            ),
            "total_execution_savings_interval_usd": _interval(total_savings, interval_level),
            "probability_total_savings_positive": sum(v > 0 for v in total_savings)
            / iterations,
            "mean_savings_vs_fixed_percent": _mean(
                [item["savings_vs_fixed_percent"] for _, _, item in usable]
            ),
            "workloads_cheaper_than_fixed": sum(
                item["savings_vs_fixed_per_execution_usd"] > 0 for _, _, item in usable
            ),
            "total_profiling_cost_usd": sum(item["profiling_cost_usd"] for _, _, item in usable),
            "per_workload": {
                workload: {
                    "selected_order": item["selected_order"],
                    "savings_vs_fixed_percent": item["savings_vs_fixed_percent"],
                    "regret_usd": item["regret_usd"],
                    "selected_is_observed_cheapest": item["selected_is_observed_cheapest"],
                    "break_even_executions": item["break_even_executions"],
                }
                for workload, _, item in usable
            },
        }

    pairwise = {}
    for first, second in itertools.combinations(
        ["fixed", "independent", "correlation_aware", "correlation_aware_confidence"], 2
    ):
        if first not in strategy_keys or second not in strategy_keys:
            continue
        pairs = [
            result
            for result in analyses
            if result["strategies"][first]["executed"]
            and result["strategies"][second]["executed"]
        ]
        if not pairs:
            continue
        iterations = len(pairs[0]["strategy_draws"][first]["cost"])
        differences = [
            sum(
                result["strategy_draws"][first]["cost"][b]
                - result["strategy_draws"][second]["cost"][b]
                for result in pairs
            )
            for b in range(iterations)
        ]
        point = sum(
            result["strategies"][first]["execution_cost_usd"]
            - result["strategies"][second]["execution_cost_usd"]
            for result in pairs
        )
        wins = sum(
            result["strategies"][first]["execution_cost_usd"]
            < result["strategies"][second]["execution_cost_usd"]
            for result in pairs
        )
        losses = sum(
            result["strategies"][first]["execution_cost_usd"]
            > result["strategies"][second]["execution_cost_usd"]
            for result in pairs
        )
        pairwise[f"{first}_minus_{second}"] = {
            "workloads": len(pairs),
            "total_execution_cost_difference_usd": point,
            "difference_interval_usd": _interval(differences, interval_level),
            "probability_first_cheaper": sum(v < 0 for v in differences) / iterations,
            "workloads_first_cheaper": wins,
            "workloads_second_cheaper": losses,
            "workloads_tied_same_order": len(pairs) - wins - losses,
            "sign_test_p_value": sign_test(wins, losses),
        }

    pooled = {}
    for estimator in COMPARED_ESTIMATORS:
        rows = [
            row
            for result in analyses
            for row in result["cardinality"][estimator]["stage_rows"]
        ]
        cost_rows = [
            row
            for result in analyses
            for row in result["cardinality"][estimator]["cost_rows"]
        ]
        concordant = sum(
            result["cardinality"][estimator]["pairwise_order_concordance"]["concordant"]
            for result in analyses
        )
        discordant = sum(
            result["cardinality"][estimator]["pairwise_order_concordance"]["discordant"]
            for result in analyses
        )
        pooled[estimator] = {
            "stage_input_error": _errors(rows),
            "cost_prediction_error": _errors(cost_rows),
            "pairwise_order_concordance": {
                "concordant": concordant,
                "discordant": discordant,
                "rate": concordant / (concordant + discordant)
                if concordant + discordant
                else None,
            },
        }
    smaller = larger = 0
    differences = []
    for result in analyses:
        independent = result["cardinality_unit_errors"]["independent"]
        correlated = result["cardinality_unit_errors"]["correlation_aware"]
        for unit in sorted(set(independent) & set(correlated)):
            a, b = abs(correlated[unit]), abs(independent[unit])
            differences.append(a - b)
            smaller += a < b
            larger += a > b
    paired_error = {
        "units": len(differences),
        "unit": "(workload, executed order, stage >= 2) observed mean stage input",
        "mean_absolute_error_difference_correlation_minus_independent": (
            _mean(differences) if differences else None
        ),
        "correlation_aware_smaller": smaller,
        "independent_smaller": larger,
        "sign_test_p_value": sign_test(smaller, larger),
        "caveat": (
            "stages within one order and orders within one workload share predicates, so units "
            "are not independent; the sign test is descriptive"
        ),
    }
    return {
        "schema_version": STATISTICS_SCHEMA_VERSION,
        "evidence": EVIDENCE,
        "workloads": names,
        "workload_count": len(names),
        "caveat": (
            "with few workloads, cross-workload tests have low power; intervals pool only "
            "within-workload review sampling and replicate variability, not workload sampling"
        ),
        "strategies": per_strategy,
        "pairwise_strategy_comparisons": pairwise,
        "cardinality": {"pooled": pooled, "paired_absolute_error": paired_error},
    }
