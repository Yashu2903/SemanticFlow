"""Offline-only Stage C preflight accounting and experimental-validity checks.

This module never imports a paid provider.  It consumes only frozen configuration,
dataset metadata, and the recorded Stage B profiling matrix.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from semanticflow.dataset.generalization import token_proxy
from semanticflow.optimizer.adaptive import sampling_order
from semanticflow.optimizer.bootstrap import quantile
from semanticflow.optimizer.cardinality import CorrelationAwareEstimator, build_estimator
from semanticflow.optimizer.generalization import (
    STAGE_C_AUTHORIZATION,
    WorkloadSuite,
    candidate_orders,
    heldout_budget,
    records_for_partition,
    sha256_file,
    strategy_freeze_manifest,
    subset_matrix,
)
from semanticflow.optimizer.matrix import ProfileMatrix
from semanticflow.optimizer.models import load_optimizer_settings
from semanticflow.optimizer.planner import predicate_hash

STAGE_C_PREFLIGHT_SCHEMA = "m46-stage-c-preflight-hardening-v1"
STAGE_C_RESTART_AUTHORIZATION = "AUTHORIZE M4.6 STAGE C FULL RESTART AFTER PATH FIX"
DEFAULT_EXPECTED_PLANS_CANONICAL_SHA256 = (
    "69daad220c1cec5b87bb4539d4c26c77cd16fc265756652127d66bb8ec71c2e2"
)


def _json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def _canonical_sha(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _price(input_tokens: float, output_tokens: float, protocol: Mapping[str, Any]) -> float:
    pricing = protocol["provider"]["pricing"]
    return (
        input_tokens * float(pricing["input_per_million_usd"])
        + output_tokens * float(pricing["output_per_million_usd"])
    ) / 1_000_000


def load_stage_b_matrix(
    suite: WorkloadSuite, dataset_id: str, profile_root: str | Path
) -> ProfileMatrix:
    """Load one complete profiling matrix and reject held-out leakage or hash drift."""
    root = Path(profile_root)
    dataset = suite.datasets[dataset_id]
    report = _json(root / dataset_id / "report.json")
    live_manifest = _json(root / dataset_id / "manifest.json")
    dataset_manifest = _json(dataset.path.with_suffix(dataset.path.suffix + ".manifest.json"))
    if not report.get("complete") or report.get("completed_decisions") != 1200:
        raise ValueError(f"Stage B matrix is not complete for {dataset_id}")
    if report["dataset_fingerprint"] != dataset_manifest["dataset_fingerprint_sha256"]:
        raise ValueError(f"dataset fingerprint mismatch for {dataset_id}")
    if report["profiling_fingerprint"] != dataset_manifest["profiling_fingerprint_sha256"]:
        raise ValueError(f"profiling fingerprint mismatch for {dataset_id}")
    decisions_path = root / dataset_id / "decisions.jsonl"
    if sha256_file(decisions_path) != live_manifest["decisions_sha256"]:
        raise ValueError(f"decision artifact hash mismatch for {dataset_id}")
    resolved = report["resolved_model_versions"]
    if report["requested_model"] != "jev-1.14.0" or resolved != ["jev-1.14.0"]:
        raise ValueError(f"model mismatch for {dataset_id}")
    predicate_ids = list(dataset.predicates)
    hashes = {
        predicate_id: predicate_hash(predicate_id, predicate.instructions)
        for predicate_id, predicate in dataset.predicates.items()
    }
    if report["predicate_hashes"] != hashes:
        raise ValueError(f"prompt hash mismatch for {dataset_id}")
    profiling = records_for_partition(dataset.path, "profiling")
    evaluation = records_for_partition(dataset.path, "evaluation")
    matrix = ProfileMatrix.from_traces(
        _jsonl(decisions_path),
        predicate_ids,
        review_ids=[record.review_id for record in profiling],
        forbidden_review_ids=[record.review_id for record in evaluation],
        model_identifier=resolved[0],
        question_hashes=hashes,
    )
    if not matrix.complete or matrix.size != 200 or len(matrix.predicate_ids) != 6:
        raise ValueError(f"unexpected Stage B matrix shape for {dataset_id}")
    return matrix


def _question_hashes(suite: WorkloadSuite, workload: Any) -> dict[str, str]:
    predicates = suite.datasets[workload.dataset].predicates
    return {
        predicate_id: predicate_hash(predicate_id, predicates[predicate_id].instructions)
        for predicate_id in workload.predicate_ids
    }


def _prefix_rows(matrix: ProfileMatrix, prefix: Sequence[str]) -> list[int]:
    indices = [matrix.index(predicate_id) for predicate_id in prefix]
    return [
        row_index
        for row_index, row in enumerate(matrix.rows)
        if all(bool(row.outcomes[index]) for index in indices)
    ]


def _stage_usage_forecast(
    matrix: ProfileMatrix,
    order: Sequence[str],
    stage: int,
    profile_records: Mapping[str, str],
    evaluation_records: Sequence[Any],
    instructions: str,
    smoothed_input: float | None,
    smoothed_output: float | None,
) -> dict[str, Any]:
    """Estimate per-call usage with reach-conditioned Stage B usage where supported."""
    predicate_id = order[stage]
    predicate_index = matrix.index(predicate_id)
    prefix = order[:stage]
    selected = _prefix_rows(matrix, prefix)
    fallback = len(selected) < 5
    used = list(range(matrix.size)) if fallback else selected
    observed_inputs = [matrix.rows[index].input_tokens[predicate_index] for index in used]
    observed_outputs = [matrix.rows[index].output_tokens[predicate_index] for index in used]
    if any(value is None for value in observed_inputs + observed_outputs):
        raise ValueError("Stage B token usage is incomplete")
    profile_proxies = [
        token_proxy(profile_records[matrix.rows[index].review_id], instructions)
        for index in used
    ]
    input_ratio = sum(float(value) for value in observed_inputs) / sum(profile_proxies)
    evaluation_proxy_mean = sum(
        token_proxy(record.review_text, instructions) for record in evaluation_records
    ) / len(evaluation_records)
    conditional_input = evaluation_proxy_mean * input_ratio
    conditional_output = sum(float(value) for value in observed_outputs) / len(used)
    return {
        "input_tokens_per_evaluation": conditional_input,
        "output_tokens_per_evaluation": conditional_output,
        "profiling_rows_supporting_usage": len(selected),
        "usage_rows_used": len(used),
        "reach_conditioned": not fallback and stage > 0,
        "usage_fallback": "unconditional_per_predicate" if fallback else None,
        "smoothed_estimator_input_tokens": smoothed_input,
        "smoothed_estimator_output_tokens": smoothed_output,
        "evaluation_mean_character_token_proxy": evaluation_proxy_mean,
        "observed_input_tokens_per_character_proxy": input_ratio,
    }


def forecast_order(
    *,
    suite: WorkloadSuite,
    protocol: Mapping[str, Any],
    workload: Any,
    matrix: ProfileMatrix,
    order: Sequence[str],
    estimator_name: str,
) -> dict[str, Any]:
    """Forecast one run with early rejection from profiling evidence only."""
    optimizer = load_optimizer_settings(protocol["strategy_sources"]["milestone_4"])
    question_hashes = _question_hashes(suite, workload)
    estimator = build_estimator(
        estimator_name,
        matrix,
        question_hashes=question_hashes,
        correlation=optimizer.correlation,
        input_price_per_million=float(
            protocol["provider"]["pricing"]["input_per_million_usd"]
        ),
        output_price_per_million=float(
            protocol["provider"]["pricing"]["output_per_million_usd"]
        ),
    )
    stages = estimator.stage_estimates(order)
    evaluation = records_for_partition(suite.datasets[workload.dataset].path, "evaluation")
    profiling = records_for_partition(suite.datasets[workload.dataset].path, "profiling")
    profile_text = {record.review_id: record.review_text for record in profiling}
    details = []
    expected_calls = expected_input = expected_output = 0.0
    for index, estimate in enumerate(stages):
        predicate = suite.datasets[workload.dataset].predicates[estimate.predicate_id]
        usage = _stage_usage_forecast(
            matrix,
            order,
            index,
            profile_text,
            evaluation,
            predicate.instructions,
            estimate.average_input_tokens,
            estimate.average_output_tokens,
        )
        calls = len(evaluation) * estimate.reach_probability
        input_tokens = calls * usage["input_tokens_per_evaluation"]
        output_tokens = calls * usage["output_tokens_per_evaluation"]
        expected_calls += calls
        expected_input += input_tokens
        expected_output += output_tokens
        details.append(
            {
                **estimate.to_dict(),
                **usage,
                "expected_evaluations": calls,
                "expected_input_tokens": input_tokens,
                "expected_output_tokens": output_tokens,
            }
        )
    return {
        "estimator": estimator_name,
        "order": list(order),
        "records": len(evaluation),
        "expected_logical_evaluations": expected_calls,
        "expected_input_tokens": expected_input,
        "expected_output_tokens": expected_output,
        "expected_cost_usd": _price(expected_input, expected_output, protocol),
        "maximum_logical_evaluations": len(evaluation) * len(order),
        "stages": details,
        "held_out_outcomes_used": False,
    }


def _bootstrap_forecast(
    *,
    suite: WorkloadSuite,
    protocol: Mapping[str, Any],
    workload: Any,
    matrix: ProfileMatrix,
    orders: Sequence[Sequence[str]],
    iterations: int,
    seed: int,
) -> dict[tuple[str, ...], dict[str, Any]]:
    """Paired-row sensitivity for scheduled-order expected calls and costs."""
    rng = random.Random(seed)
    draws: dict[tuple[str, ...], dict[str, list[float]]] = {
        tuple(order): {"calls": [], "cost": []} for order in orders
    }
    optimizer = load_optimizer_settings(protocol["strategy_sources"]["milestone_4"])
    question_hashes = _question_hashes(suite, workload)
    dataset = suite.datasets[workload.dataset]
    profiling = records_for_partition(dataset.path, "profiling")
    evaluation = records_for_partition(dataset.path, "evaluation")
    profiling_text = {record.review_id: record.review_text for record in profiling}
    usage_scale = {}
    for predicate_id in workload.predicate_ids:
        instructions = dataset.predicates[predicate_id].instructions
        profile_mean = sum(
            token_proxy(profiling_text[row.review_id], instructions) for row in matrix.rows
        ) / matrix.size
        evaluation_mean = sum(
            token_proxy(record.review_text, instructions) for record in evaluation
        ) / len(evaluation)
        usage_scale[predicate_id] = evaluation_mean / profile_mean
    for _ in range(iterations):
        sampled = matrix.take([rng.randrange(matrix.size) for _ in range(matrix.size)])
        estimator = build_estimator(
            "correlation_aware",
            sampled,
            question_hashes=question_hashes,
            correlation=optimizer.correlation,
            input_price_per_million=float(
                protocol["provider"]["pricing"]["input_per_million_usd"]
            ),
            output_price_per_million=float(
                protocol["provider"]["pricing"]["output_per_million_usd"]
            ),
        )
        for order in orders:
            expected_calls = input_tokens = output_tokens = 0.0
            for stage in estimator.stage_estimates(order):
                calls = len(evaluation) * stage.reach_probability
                expected_calls += calls
                input_tokens += calls * float(stage.average_input_tokens or 0) * usage_scale[
                    stage.predicate_id
                ]
                output_tokens += calls * float(stage.average_output_tokens or 0)
            key = tuple(order)
            draws[key]["calls"].append(expected_calls)
            draws[key]["cost"].append(_price(input_tokens, output_tokens, protocol))
    output = {}
    for order, values in draws.items():
        output[order] = {
            "iterations": iterations,
            "interval_level": 0.90,
            "expected_calls_lower": quantile(values["calls"], 0.05),
            "expected_calls_upper": quantile(values["calls"], 0.95),
            "expected_cost_usd_lower": quantile(values["cost"], 0.05),
            "expected_cost_usd_upper": quantile(values["cost"], 0.95),
            "interpretation": "profiling-sample sensitivity, not a guaranteed bound",
        }
    return output


def per_run_cost_forecasts(
    *,
    suite: WorkloadSuite,
    protocol: Mapping[str, Any],
    plans: Mapping[str, Any],
    matrices: Mapping[str, ProfileMatrix],
    bootstrap_iterations: int = 2000,
) -> dict[str, Any]:
    """Build the 42-row expected, conservative, and retry-inclusive budget report."""
    legacy = heldout_budget(suite, protocol, plans)
    legacy_by_workload = {item["workload_id"]: item for item in legacy["workloads"]}
    max_retries = int(protocol["provider"]["max_retries"])
    rows = []
    for workload_index, workload in enumerate(suite.workloads):
        schedule = legacy_by_workload[workload.id]["schedule"]
        matrix = subset_matrix(matrices[workload.dataset], workload.predicate_ids)
        unique_orders = list(
            dict.fromkeys(tuple(run["order"]) for run in schedule["runs"])
        )
        sensitivity = _bootstrap_forecast(
            suite=suite,
            protocol=protocol,
            workload=workload,
            matrix=matrix,
            orders=unique_orders,
            iterations=bootstrap_iterations,
            seed=int(protocol["analysis"]["bootstrap_seed"]) + workload_index,
        )
        calibration = plans["stage_b_usage_calibration"]["datasets"][workload.dataset]
        eval_records = records_for_partition(
            suite.datasets[workload.dataset].path, "evaluation"
        )
        for run in schedule["runs"]:
            correlation = forecast_order(
                suite=suite,
                protocol=protocol,
                workload=workload,
                matrix=matrix,
                order=run["order"],
                estimator_name="correlation_aware",
            )
            independent = forecast_order(
                suite=suite,
                protocol=protocol,
                workload=workload,
                matrix=matrix,
                order=run["order"],
                estimator_name="independent",
            )
            conservative_input = conservative_output = 0.0
            for predicate_id in run["order"]:
                predicate = suite.datasets[workload.dataset].predicates[predicate_id]
                proxies = sum(
                    token_proxy(record.review_text, predicate.instructions)
                    for record in eval_records
                )
                usage = calibration["predicates"][predicate_id]
                conservative_input += proxies * usage["maximum_input_token_ratio"]
                conservative_output += (
                    len(eval_records) * usage["maximum_output_tokens"]
                )
            max_logical = len(eval_records) * len(run["order"])
            no_retry_proxy = _price(conservative_input, conservative_output, protocol)
            block = workload.id if len(schedule["runs"]) >= 4 else None
            rows.append(
                {
                    "run_id": run["run_id"],
                    "dataset": workload.dataset,
                    "workload_id": workload.id,
                    "role": run["role"],
                    "abba_block_id": block,
                    "predicate_order": run["order"],
                    "predicate_count": len(run["order"]),
                    "held_out_records": len(eval_records),
                    "correlation_aware": correlation,
                    "independence_comparison": independent,
                    "bootstrap_sensitivity": sensitivity[tuple(run["order"])],
                    "maximum_logical_evaluations": max_logical,
                    "expected_input_tokens": math.ceil(correlation["expected_input_tokens"]),
                    "expected_output_tokens": math.ceil(correlation["expected_output_tokens"]),
                    "expected_estimated_cost_usd": correlation["expected_cost_usd"],
                    "conservative_input_token_proxy": math.ceil(conservative_input),
                    "conservative_output_token_proxy": math.ceil(conservative_output),
                    "conservative_no_retry_cost_proxy_usd": no_retry_proxy,
                    "retry_inclusive_cost_proxy_usd": no_retry_proxy * (max_retries + 1),
                    "allowed_http_attempts": max_logical * (max_retries + 1),
                }
            )
    dataset_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for row in rows:
        totals = dataset_totals[row["dataset"]]
        for key in (
            "maximum_logical_evaluations",
            "expected_input_tokens",
            "expected_output_tokens",
            "expected_estimated_cost_usd",
            "conservative_no_retry_cost_proxy_usd",
            "retry_inclusive_cost_proxy_usd",
            "allowed_http_attempts",
        ):
            totals[key] += row[key]
        totals["expected_logical_evaluations"] += row["correlation_aware"][
            "expected_logical_evaluations"
        ]
    aggregate = defaultdict(float)
    for totals in dataset_totals.values():
        for key, value in totals.items():
            aggregate[key] += value
    return {
        "schema_version": "m46-stage-c-per-run-cost-forecast-v1",
        "forecast_basis": (
            "Stage B correlation-aware prefix reach, reach-conditioned token usage when "
            "supported, and evaluation-partition text-length proxies; no evaluation outcomes"
        ),
        "monetary_limit_note": (
            "Only logical decisions and HTTP attempts have hard guards. Token and cost "
            "figures are empirical estimates, not guaranteed monetary ceilings."
        ),
        "bootstrap_iterations": bootstrap_iterations,
        "runs": rows,
        "dataset_totals": {key: dict(value) for key, value in dataset_totals.items()},
        "aggregate": dict(aggregate),
        "legacy_no_early_rejection_expected_cost_usd": legacy[
            "estimated_cost_ceiling_usd"
        ],
        "legacy_conservative_no_retry_cost_proxy_usd": legacy[
            "conservative_no_retry_cost_proxy_usd"
        ],
        "legacy_retry_inclusive_cost_proxy_usd": legacy[
            "retry_inclusive_cost_proxy_usd"
        ],
        "paid_calls_executed": 0,
    }


def decision_identity(
    *,
    dataset_fingerprint: str,
    trace: Mapping[str, Any],
    threshold: float,
) -> str:
    """Stable compatibility key for one reusable profiling decision."""
    payload = {
        "dataset_fingerprint": dataset_fingerprint,
        "record_id": trace["review_id"],
        "predicate_id": trace["predicate_id"],
        "prompt_hash": trace["question_hash"],
        "model": trace["model_identifier"],
        "threshold": threshold,
        "question_type": trace["question_type"],
        "cache_status": trace.get("cache_status"),
    }
    return _canonical_sha(payload)


def profiling_cost_reconciliation(
    *,
    suite: WorkloadSuite,
    protocol: Mapping[str, Any],
    plans: Mapping[str, Any],
    profile_root: str | Path,
) -> dict[str, Any]:
    """Reconcile physical, standalone, and shared-statistics profiling costs."""
    root = Path(profile_root)
    ledgers: dict[str, dict[str, dict[str, Any]]] = {}
    actual_dataset = {}
    for dataset_id in suite.datasets:
        report = _json(root / dataset_id / "report.json")
        traces = _jsonl(root / dataset_id / "decisions.jsonl")
        identities: dict[str, dict[str, Any]] = {}
        for trace in traces:
            identity = decision_identity(
                dataset_fingerprint=report["dataset_fingerprint"],
                trace=trace,
                threshold=suite.threshold,
            )
            if identity in identities:
                raise ValueError(f"duplicate physical profiling request {identity}")
            cost = _price(trace["input_tokens"], trace["output_tokens"], protocol)
            identities[identity] = {**trace, "decision_identity": identity, "cost_usd": cost}
        if len(identities) != report["completed_decisions"]:
            raise ValueError(f"unaccounted Stage B requests for {dataset_id}")
        calculated = sum(item["cost_usd"] for item in identities.values())
        if not math.isclose(calculated, report["estimated_cost_usd"], abs_tol=1e-12):
            raise ValueError(f"Stage B cost does not reconcile for {dataset_id}")
        ledgers[dataset_id] = identities
        actual_dataset[dataset_id] = {
            "unique_provider_requests": len(identities),
            "recorded_cost_usd": report["estimated_cost_usd"],
            "reconciled_cost_usd": calculated,
            "reconciled": True,
        }

    optimizers = {
        "m4": load_optimizer_settings(protocol["strategy_sources"]["milestone_4"]),
        "m45": load_optimizer_settings(
            protocol["strategy_sources"]["milestone_4_5_frozen"]
        ),
    }
    standalone: dict[str, dict[str, Any]] = {}
    shared_paid: dict[str, set[str]] = {strategy: set() for strategy in protocol["strategies"]}
    shared_totals: dict[str, dict[str, float]] = {
        strategy: {"decisions": 0.0, "cost_usd": 0.0}
        for strategy in protocol["strategies"]
    }
    for workload in suite.workloads:
        entry = plans["workloads"][workload.id]
        workload_rows = {}
        dataset_traces = list(ledgers[workload.dataset].values())
        by_cell = {
            (trace["review_id"], trace["predicate_id"]): trace
            for trace in dataset_traces
        }
        all_review_ids = sorted({trace["review_id"] for trace in dataset_traces})
        for strategy, selection in entry["strategies"].items():
            decisions = int(selection["profiling_decisions"])
            width = len(workload.predicate_ids)
            if decisions % width:
                raise ValueError(f"partial profiling row for {workload.id}/{strategy}")
            reviews_needed = decisions // width
            optimizer = optimizers["m45" if strategy == "milestone_4_5_frozen" else "m4"]
            ordered = sampling_order(all_review_ids, optimizer.adaptive.seed)
            chosen = ordered[:reviews_needed]
            identities = [
                by_cell[(review_id, predicate_id)]
                for review_id in chosen
                for predicate_id in workload.predicate_ids
            ]
            standalone_cost = sum(item["cost_usd"] for item in identities)
            incremental = [
                item
                for item in identities
                if item["decision_identity"] not in shared_paid[strategy]
            ]
            shared_paid[strategy].update(item["decision_identity"] for item in incremental)
            shared_totals[strategy]["decisions"] += len(incremental)
            shared_totals[strategy]["cost_usd"] += sum(
                item["cost_usd"] for item in incremental
            )
            workload_rows[strategy] = {
                "counterfactual_decisions": decisions,
                "profiled_reviews": reviews_needed,
                "standalone_cost_usd": standalone_cost,
                "shared_incremental_decisions": len(incremental),
                "shared_incremental_cost_usd": sum(
                    item["cost_usd"] for item in incremental
                ),
                "not_physical_provider_requests": True,
                "no_future_checkpoint_exposed": True,
            }
        standalone[workload.id] = workload_rows
    standalone_totals = {
        strategy: {
            "decisions": sum(
                standalone[workload.id][strategy]["counterfactual_decisions"]
                for workload in suite.workloads
            ),
            "cost_usd": sum(
                standalone[workload.id][strategy]["standalone_cost_usd"]
                for workload in suite.workloads
            ),
        }
        for strategy in protocol["strategies"]
    }
    if any(
        shared_totals[strategy]["cost_usd"]
        > standalone_totals[strategy]["cost_usd"] + 1e-15
        for strategy in protocol["strategies"]
    ):
        raise AssertionError("shared profiling exceeded standalone profiling")
    actual_total = sum(row["reconciled_cost_usd"] for row in actual_dataset.values())
    if not math.isclose(actual_total, 0.032638536, abs_tol=1e-12):
        raise ValueError("actual Stage B spend does not reconcile to $0.032638536")
    return {
        "schema_version": "m46-profiling-cost-reconciliation-v1",
        "actual_experiment_expenditure": {
            "datasets": actual_dataset,
            "unique_provider_requests": sum(
                row["unique_provider_requests"] for row in actual_dataset.values()
            ),
            "cost_usd": actual_total,
            "each_provider_request_charged_once": True,
        },
        "standalone_workload_economics": standalone,
        "standalone_totals_by_strategy": standalone_totals,
        "shared_statistics_totals_by_strategy": shared_totals,
        "shared_never_exceeds_standalone": True,
        "physical_and_counterfactual_costs_separated": True,
        "paid_calls_executed": 0,
    }


def _exact_replay_cost(
    matrix: ProfileMatrix,
    order: Sequence[str],
    protocol: Mapping[str, Any],
    *,
    record_multiplier: float = 1.0,
) -> float:
    total = 0.0
    for row in matrix.rows:
        for predicate_id in order:
            index = matrix.index(predicate_id)
            input_tokens = row.input_tokens[index]
            output_tokens = row.output_tokens[index]
            if input_tokens is None or output_tokens is None:
                raise ValueError("exact replay requires token usage")
            total += _price(input_tokens, output_tokens, protocol)
            if not row.outcomes[index]:
                break
    return total * record_multiplier


def support_and_strategy_diagnostics(
    *,
    suite: WorkloadSuite,
    protocol: Mapping[str, Any],
    plans: Mapping[str, Any],
    matrices: Mapping[str, ProfileMatrix],
) -> dict[str, Any]:
    """Describe sparse conditional evidence and profiling-replay policy disagreements."""
    optimizer = load_optimizer_settings(protocol["strategy_sources"]["milestone_4"])
    workloads = []
    strategy_rows = []
    for workload in suite.workloads:
        matrix = subset_matrix(matrices[workload.dataset], workload.predicate_ids)
        estimator = CorrelationAwareEstimator(
            matrix,
            optimizer.correlation,
            input_price_per_million=float(
                protocol["provider"]["pricing"]["input_per_million_usd"]
            ),
            output_price_per_million=float(
                protocol["provider"]["pricing"]["output_per_million_usd"]
            ),
        )
        prefix_rows = []
        order_rows = []
        for order in candidate_orders(workload):
            stages = estimator.stage_estimates(order)
            sparse = zero = fallback = False
            for stage_index, _stage in enumerate(stages):
                prefix = order[:stage_index]
                predicate_index = matrix.index(order[stage_index])
                prefix_indices = [matrix.index(item) for item in prefix]
                smoothed, conditional_passing, support, method = estimator.conditional(
                    predicate_index, prefix_indices
                )
                empirical = conditional_passing / support if support else None
                smoothing_weight = (
                    optimizer.correlation.prior_strength
                    / (support + optimizer.correlation.prior_strength)
                    if support + optimizer.correlation.prior_strength
                    else 1.0
                )
                item = {
                    "order": order,
                    "stage": stage_index + 1,
                    "prefix": prefix,
                    "predicate_id": order[stage_index],
                    "profiling_records": matrix.size,
                    "prefix_passing_count": support,
                    "conditional_passing_count": conditional_passing,
                    "empirical_conditional_selectivity": empirical,
                    "smoothed_conditional_selectivity": smoothed,
                    "effective_sample_support": support,
                    "fallback_used": method == "backoff_unseen_prefix",
                    "estimator_method": method,
                    "smoothing_weight": smoothing_weight,
                    "smoothing_dominated": smoothing_weight >= 0.5,
                    "zero_support_prefix": support == 0,
                    "low_support": support < optimizer.correlation.minimum_conditional_support,
                }
                prefix_rows.append(item)
                sparse |= item["low_support"]
                zero |= item["zero_support_prefix"]
                fallback |= item["fallback_used"]
            correlation_prediction = next(
                row
                for row in plans["workloads"][workload.id]["candidate_predictions"][
                    "correlation_aware"
                ]
                if row["order"] == order
            )
            order_rows.append(
                {
                    "order": order,
                    "predicted_cost_usd": correlation_prediction["estimated_cost_usd"],
                    "profiling_replay_cost_usd_800_record_scale": _exact_replay_cost(
                        matrix, order, protocol, record_multiplier=4.0
                    ),
                    "has_low_support_prefix": sparse,
                    "has_zero_support_prefix": zero,
                    "uses_estimator_fallback": fallback,
                }
            )
        ranked = sorted(order_rows, key=lambda row: (row["predicted_cost_usd"], row["order"]))
        gap = ranked[1]["predicted_cost_usd"] - ranked[0]["predicted_cost_usd"]
        relative_gap = (
            gap / ranked[0]["predicted_cost_usd"]
            if ranked[0]["predicted_cost_usd"]
            else 0
        )
        support_category = (
            "fallback_sensitive"
            if any(row["uses_estimator_fallback"] for row in order_rows)
            else "sparse_conditional_evidence"
            if any(row["has_low_support_prefix"] for row in order_rows)
            else "moderate_or_strong_support"
        )
        workloads.append(
            {
                "workload_id": workload.id,
                "dataset": workload.dataset,
                "support_category": support_category,
                "best_vs_second_predicted_cost_gap_usd": gap,
                "best_vs_second_relative_gap": relative_gap,
                "near_tied_prediction": relative_gap < 0.01,
                "prefix_diagnostics": prefix_rows,
                "candidate_ranking_diagnostics": order_rows,
                "classification_is_descriptive_not_calibrated_probability": True,
            }
        )
        original_cost = _exact_replay_cost(
            matrix, workload.original_order, protocol, record_multiplier=4.0
        )
        plan = plans["workloads"][workload.id]
        for strategy, selection in plan["strategies"].items():
            selected = selection["selected_order"]
            replay = _exact_replay_cost(matrix, selected, protocol, record_multiplier=4.0)
            estimator_name = (
                "independent" if strategy == "independent" else "correlation_aware"
            )
            predicted_candidates = plan["candidate_predictions"][estimator_name]
            original_prediction = next(
                row for row in predicted_candidates if row["order"] == workload.original_order
            )["estimated_cost_usd"]
            selected_prediction = next(
                row for row in predicted_candidates if row["order"] == selected
            )["estimated_cost_usd"]
            reason = {
                "fixed": "preregistered original-order comparator",
                "independent": "minimum independence-estimated objective after full profiling",
                "correlation_aware": "minimum correlation-aware objective after full profiling",
                "existing_adaptive": "frozen adaptive checkpoint decision",
                "confidence_aware": "frozen bootstrap confidence gate",
                "risk_aware": "frozen downside-risk and minimum-saving gates",
                "joint_adaptive_risk": "frozen joint adaptive-risk checkpoint decision",
                "milestone_4_5_frozen": "frozen Milestone 4.5 joint policy checkpoint decision",
            }[strategy]
            strategy_rows.append(
                {
                    "workload_id": workload.id,
                    "dataset": workload.dataset,
                    "strategy": strategy,
                    "original_order": workload.original_order,
                    "selected_order": selected,
                    "chosen_action": selection["action"],
                    "profiling_decisions": selection["profiling_decisions"],
                    "profiling_depth_records": selection["profiling_decisions"]
                    // len(workload.predicate_ids),
                    "predicted_saving_vs_original_usd": (
                        original_prediction - selected_prediction
                    ),
                    "profiling_replay_saving_vs_original_usd": original_cost - replay,
                    "profiling_replay_negative_reordering": (
                        selected != workload.original_order and replay > original_cost
                    ),
                    "reason": reason,
                    "profiling_replay_is_not_held_out_evidence": True,
                }
            )
    by_workload = defaultdict(list)
    for row in strategy_rows:
        by_workload[row["workload_id"]].append(row)
    disagreements = []
    for workload in suite.workloads:
        rows = by_workload[workload.id]
        distinct = {tuple(row["selected_order"]) for row in rows}
        adaptive = next(row for row in rows if row["strategy"] == "existing_adaptive")
        risk = next(row for row in rows if row["strategy"] == "risk_aware")
        disagreements.append(
            {
                "workload_id": workload.id,
                "distinct_selected_orders": len(distinct),
                "strategy_disagreement": len(distinct) > 1,
                "adaptive_vs_risk_disagreement": (
                    adaptive["selected_order"] != risk["selected_order"]
                ),
                "adaptive_order": adaptive["selected_order"],
                "risk_aware_order": risk["selected_order"],
            }
        )
    return {
        "schema_version": "m46-stage-c-support-diagnostics-v1",
        "workloads": workloads,
        "strategy_comparison_matrix": strategy_rows,
        "strategy_disagreements": disagreements,
        "negative_profiling_replay_reorderings": [
            row for row in strategy_rows if row["profiling_replay_negative_reordering"]
        ],
        "held_out_outcomes_used": False,
        "paid_calls_executed": 0,
    }


def validate_schedule(
    *, suite: WorkloadSuite, protocol: Mapping[str, Any], plans: Mapping[str, Any]
) -> dict[str, Any]:
    """Reconstruct and validate all frozen Stage C comparisons."""
    budget = heldout_budget(suite, protocol, plans)
    runs = []
    unique_pairs = set()
    blocks = []
    strategies_mapped = Counter()
    for workload in suite.workloads:
        entry = next(row for row in budget["workloads"] if row["workload_id"] == workload.id)
        schedule = entry["schedule"]
        selected = schedule["selected_orders"]
        allowed = {tuple(workload.original_order), *(tuple(value) for value in selected.values())}
        for strategy in selected:
            strategies_mapped[strategy] += 1
        seen_run_ids = set()
        for run in schedule["runs"]:
            if tuple(run["order"]) not in allowed:
                raise ValueError(f"unfrozen order scheduled in {run['run_id']}")
            if run["run_id"] in seen_run_ids:
                raise ValueError(f"duplicate run ID {run['run_id']}")
            seen_run_ids.add(run["run_id"])
            runs.append({"workload_id": workload.id, "dataset": workload.dataset, **run})
            unique_pairs.add((workload.id, tuple(run["order"])))
        if len(schedule["runs"]) >= 4:
            first = schedule["runs"][:4]
            roles = [row["role"] for row in first]
            if roles != [
                "abba_fixed_a",
                "abba_comparator_b",
                "abba_comparator_b",
                "abba_fixed_a",
            ]:
                raise ValueError(f"invalid ABBA positions for {workload.id}")
            blocks.append(
                {
                    "block_id": workload.id,
                    "workload_id": workload.id,
                    "dataset": workload.dataset,
                    "a_order": first[0]["order"],
                    "b_order": first[1]["order"],
                    "run_positions": [1, 2, 3, 4],
                    "run_ids": [row["run_id"] for row in first],
                    "comparison_strength": "repeated_controlled_abba",
                }
            )
        else:
            blocks.append(
                {
                    "block_id": workload.id,
                    "workload_id": workload.id,
                    "dataset": workload.dataset,
                    "a_order": workload.original_order,
                    "b_order": workload.original_order,
                    "run_positions": [1],
                    "run_ids": [schedule["runs"][0]["run_id"]],
                    "comparison_strength": "identical_orders_single_fixed_execution",
                    "redundant_paid_abba_runs_avoided": True,
                }
            )
    if len(runs) != 42 or len(unique_pairs) != 24:
        raise ValueError("frozen Stage C schedule no longer reproduces 42 runs / 24 orders")
    if len({row["run_id"] for row in runs}) != 42:
        raise ValueError("Stage C run identifiers are not globally unique")
    if set(strategies_mapped) != set(protocol["strategies"]):
        raise ValueError("not every strategy is represented in the analysis mapping")
    exploratory = [row for row in runs if row["role"] == "selected_order_coverage"]
    return {
        "schema_version": "m46-stage-c-schedule-validation-v1",
        "valid": True,
        "total_runs": len(runs),
        "unique_workload_order_combinations": len(unique_pairs),
        "workloads": len(suite.workloads),
        "strategies_mapped": dict(strategies_mapped),
        "abba_blocks": blocks,
        "single_run_exploratory_comparisons": exploratory,
        "single_run_exploratory_count": len(exploratory),
        "identical_strategy_orders_require_no_duplicate_execution": True,
        "required_comparators_missing": [],
        "unnecessary_duplicate_executions": [],
        "schedule_amendment_recommended": False,
        "evidential_strength_note": (
            "ABBA comparisons are repeated controlled measurements; selected-order coverage "
            "runs are exploratory and must not be reported at equal evidential strength."
        ),
        "runs": runs,
    }


def build_authorization_manifest(
    *,
    suite: WorkloadSuite,
    protocol: Mapping[str, Any],
    plans_path: str | Path,
    budget_path: str | Path,
) -> dict[str, Any]:
    plans = _json(plans_path)
    budget = _json(budget_path)
    schedule = validate_schedule(suite=suite, protocol=protocol, plans=plans)
    datasets = {}
    prompts = {}
    for dataset_id, dataset in suite.datasets.items():
        manifest_path = dataset.path.with_suffix(dataset.path.suffix + ".manifest.json")
        manifest = _json(manifest_path)
        datasets[dataset_id] = {
            "path": dataset.path.as_posix(),
            "file_sha256": sha256_file(dataset.path),
            "manifest_sha256": sha256_file(manifest_path),
            "dataset_fingerprint_sha256": manifest["dataset_fingerprint_sha256"],
            "evaluation_fingerprint_sha256": manifest["evaluation_fingerprint_sha256"],
        }
        prompts[dataset_id] = {
            predicate_id: predicate_hash(predicate_id, item.instructions)
            for predicate_id, item in dataset.predicates.items()
        }
    execution_source_paths = (
        "experiments/milestone_4_6/run_generalization.py",
        "src/semanticflow/optimizer/stage_c_preflight.py",
        "src/semanticflow/optimizer/generalization.py",
        "src/semanticflow/runtime.py",
        "src/semanticflow/executor/sequential.py",
        "src/semanticflow/providers/base.py",
        "src/semanticflow/providers/jev.py",
        "src/semanticflow/storage/results.py",
    )
    return {
        "schema_version": "m46-stage-c-exact-authorization-manifest-v1",
        "required_phrase": STAGE_C_AUTHORIZATION,
        "plans_sha256": sha256_file(plans_path),
        "budget_sha256": sha256_file(budget_path),
        "selected_workloads_sha256": _canonical_sha(plans["workloads"]),
        "schedule_sha256": _canonical_sha(schedule["runs"]),
        "workloads_config_sha256": sha256_file("configs/milestone_4_6/workloads.yaml"),
        "protocol_config_sha256": sha256_file("configs/milestone_4_6/protocol.yaml"),
        "datasets": datasets,
        "prompt_hashes": prompts,
        "execution_source_sha256": {
            path: sha256_file(path) for path in execution_source_paths
        },
        "model": protocol["provider"]["model"],
        "cache_enabled": protocol["provider"]["cache_enabled"],
        "concurrency": protocol["provider"]["concurrency"],
        "max_retries": protocol["provider"]["max_retries"],
        "schedule_runs": schedule["total_runs"],
        "unique_workload_orders": schedule["unique_workload_order_combinations"],
        "maximum_logical_decisions": budget["logical_decision_ceiling"],
        "maximum_http_attempts": budget["maximum_http_attempts"],
        "paid_calls_authorized_by_manifest_alone": False,
        "live_jev_calls": 0,
    }


def validate_stage_c_launch(
    *,
    suite: WorkloadSuite,
    protocol: Mapping[str, Any],
    plans_path: str | Path,
    budget_path: str | Path,
    authorization_manifest_path: str | Path,
    authorization_phrase: str | None,
    output_path: str | Path,
) -> dict[str, Any]:
    """Fail-closed, provider-free validation of an exact Stage C launch."""
    if Path(output_path).exists():
        raise FileExistsError(f"refusing conflicting Stage C output {output_path}")
    authorized = _json(authorization_manifest_path)
    required_phrase = authorized.get("required_phrase")
    if required_phrase not in {STAGE_C_AUTHORIZATION, STAGE_C_RESTART_AUTHORIZATION}:
        raise PermissionError("authorization manifest contains an unsupported phrase")
    if authorization_phrase != required_phrase:
        raise PermissionError("exact Stage C authorization phrase is required")
    expected = build_authorization_manifest(
        suite=suite,
        protocol=protocol,
        plans_path=plans_path,
        budget_path=budget_path,
    )
    expected["required_phrase"] = required_phrase
    if authorized != expected:
        changed = sorted(
            key
            for key in set(authorized) | set(expected)
            if authorized.get(key) != expected.get(key)
        )
        raise ValueError(f"Stage C authorization scope mismatch: {changed}")
    plans = _json(plans_path)
    budget = _json(budget_path)
    recomputed_budget = heldout_budget(suite, protocol, plans)
    if budget != recomputed_budget:
        raise ValueError("Stage C budget or frozen run schedule differs from recomputation")
    if plans.get("held_out_outcomes_observed") is not False:
        raise ValueError("plans do not prove held-out concealment")
    if _canonical_sha(plans) != DEFAULT_EXPECTED_PLANS_CANONICAL_SHA256:
        raise ValueError("frozen corrected plan artifact changed")
    if expected["model"] != "jev-1.14.0" or expected["cache_enabled"] is not False:
        raise ValueError("provider model/cache controls changed")
    if expected["maximum_http_attempts"] != 309600:
        raise ValueError("invalid Stage C request ceiling")
    return {
        "validated": True,
        "provider_imported": False,
        "provider_initialized": False,
        "paid_calls_executed": 0,
        "authorization_scope_sha256": _canonical_sha(expected),
        "plans_sha256": expected["plans_sha256"],
        "schedule_sha256": expected["schedule_sha256"],
    }


def baseline_integrity_audit(
    *,
    suite: WorkloadSuite,
    protocol: Mapping[str, Any],
    plans_path: str | Path,
    budget_path: str | Path,
    profile_root: str | Path,
) -> dict[str, Any]:
    """Re-verify baseline inputs without reading evaluation outcomes."""
    plans = _json(plans_path)
    budget = _json(budget_path)
    matrices = {
        dataset_id: load_stage_b_matrix(suite, dataset_id, profile_root)
        for dataset_id in suite.datasets
    }
    freeze = strategy_freeze_manifest(
        m4_config=protocol["strategy_sources"]["milestone_4"],
        m45_config=protocol["strategy_sources"]["milestone_4_5_frozen"],
        m45_frozen_manifest="results/milestone_4_5/calibration_20261111/frozen_config.json",
    )
    expected_fingerprints = {
        key: value["fingerprint_sha256"]
        for key, value in plans["strategy_freeze"]["strategies"].items()
    }
    actual_fingerprints = {
        key: value["fingerprint_sha256"] for key, value in freeze["strategies"].items()
    }
    if actual_fingerprints != expected_fingerprints:
        raise ValueError("frozen strategy source/config fingerprint drift")
    schedule = validate_schedule(suite=suite, protocol=protocol, plans=plans)
    return {
        "schema_version": "m46-stage-c-baseline-verification-v1",
        "plans_sha256": sha256_file(plans_path),
        "budget_sha256": sha256_file(budget_path),
        "plans_hash_expected": (
            _canonical_sha(plans) == DEFAULT_EXPECTED_PLANS_CANONICAL_SHA256
        ),
        "strategy_fingerprints_verified": True,
        "matrix_fingerprints": {
            key: matrix.fingerprint() for key, matrix in matrices.items()
        },
        "total_matrix_decisions": sum(
            matrix.size * len(matrix.predicate_ids) for matrix in matrices.values()
        ),
        "schedule_runs": schedule["total_runs"],
        "unique_workload_orders": schedule["unique_workload_order_combinations"],
        "logical_decision_ceiling": budget["logical_decision_ceiling"],
        "maximum_http_attempts": budget["maximum_http_attempts"],
        "held_out_outcomes_observed": False,
        "provider_imported": False,
        "paid_calls_executed": 0,
    }
