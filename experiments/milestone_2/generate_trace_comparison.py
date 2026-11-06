from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def reduction(original: float, optimized: float) -> float | None:
    if original == 0:
        return 0.0 if optimized == 0 else None
    return (original - optimized) / original * 100


def prediction_error(predicted: float, observed: float) -> dict[str, float]:
    absolute = abs(predicted - observed)
    return {
        "predicted": predicted,
        "observed": observed,
        "absolute_error": absolute,
        "relative_error_percent": absolute / observed * 100 if observed else 0.0,
    }


def decision_stability(
    fixed: list[dict[str, Any]], optimized: list[dict[str, Any]]
) -> dict[str, Any]:
    fixed_by_key = {(row["review_id"], row["predicate_id"]): row for row in fixed}
    optimized_by_key = {
        (row["review_id"], row["predicate_id"]): row for row in optimized
    }
    keys = sorted(fixed_by_key.keys() & optimized_by_key.keys())

    def summarize(selected: list[tuple[str, str]]) -> dict[str, Any]:
        differences = [
            abs(
                float(optimized_by_key[key]["probability"])
                - float(fixed_by_key[key]["probability"])
            )
            for key in selected
        ]
        agreements = sum(
            bool(fixed_by_key[key]["passed"])
            == bool(optimized_by_key[key]["passed"])
            for key in selected
        )
        crossings = [
            {
                "review_id": key[0],
                "predicate_id": key[1],
                "fixed_probability": fixed_by_key[key]["probability"],
                "optimized_probability": optimized_by_key[key]["probability"],
                "fixed_passed": fixed_by_key[key]["passed"],
                "optimized_passed": optimized_by_key[key]["passed"],
            }
            for key in selected
            if bool(fixed_by_key[key]["passed"])
            != bool(optimized_by_key[key]["passed"])
        ]
        return {
            "shared_pairs": len(selected),
            "boolean_agreements": agreements,
            "boolean_agreement_percent": agreements / len(selected) * 100
            if selected
            else None,
            "threshold_crossings": len(crossings),
            "mean_absolute_probability_difference": mean(differences)
            if differences
            else None,
            "median_absolute_probability_difference": median(differences)
            if differences
            else None,
            "maximum_absolute_probability_difference": max(differences)
            if differences
            else None,
            "crossing_details": crossings,
        }

    by_predicate: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in keys:
        by_predicate[key[1]].append(key)
    result = summarize(keys)
    result["by_predicate"] = {
        predicate_id: summarize(predicate_keys)
        for predicate_id, predicate_keys in sorted(by_predicate.items())
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--historical-baseline-dir", type=Path, required=True)
    args = parser.parse_args()

    experiment = args.experiment_dir
    baseline = args.historical_baseline_dir
    optimized_dir = experiment / "optimized"
    optimized_report = load_json(optimized_dir / "report.json")
    optimized_manifest = load_json(optimized_dir / "manifest.json")
    optimized_decisions = load_jsonl(optimized_dir / "decisions.jsonl")
    optimized_matches = {
        row["review_id"] for row in load_jsonl(optimized_dir / "matches.jsonl")
    }

    evaluation_ids = set(optimized_manifest["evaluation_record_ids"])
    baseline_decisions = [
        row
        for row in load_jsonl(baseline / "decisions.jsonl")
        if row["review_id"] in evaluation_ids
    ]
    baseline_matches = {
        row["review_id"]
        for row in load_jsonl(baseline / "matches.jsonl")
        if row["review_id"] in evaluation_ids
    }

    input_rate = float(
        optimized_manifest["pricing_configuration"]["input_per_million_usd"]
    )
    output_rate = float(
        optimized_manifest["pricing_configuration"]["output_per_million_usd"]
    )
    fixed_input_tokens = sum(int(row["input_tokens"]) for row in baseline_decisions)
    fixed_output_tokens = sum(int(row["output_tokens"]) for row in baseline_decisions)
    fixed_cost = (
        fixed_input_tokens * input_rate + fixed_output_tokens * output_rate
    ) / 1_000_000

    fixed_predicates = []
    for predicate_id in ["poor_quality", "product_defect", "dissatisfaction"]:
        rows = [row for row in baseline_decisions if row["predicate_id"] == predicate_id]
        fixed_predicates.append(
            {
                "predicate_id": predicate_id,
                "input_record_count": len(rows),
                "completed_decisions": len(rows),
                "passing_record_count": sum(bool(row["passed"]) for row in rows),
                "rejected_record_count": sum(not bool(row["passed"]) for row in rows),
                "selectivity": sum(bool(row["passed"]) for row in rows) / len(rows)
                if rows
                else None,
                "input_tokens": sum(int(row["input_tokens"]) for row in rows),
                "output_tokens": sum(int(row["output_tokens"]) for row in rows),
                "summed_inference_latency_seconds": sum(
                    float(row["request_latency_seconds"]) for row in rows
                ),
            }
        )

    fixed = {
        "source": "historical_milestone_1_trace_subset",
        "total_input_reviews": len(evaluation_ids),
        "total_completed_reviews": len(evaluation_ids),
        "completed_decisions": len(baseline_decisions),
        "http_attempts": sum(
            int(row.get("request_attempt_count", 1)) for row in baseline_decisions
        ),
        "successful_api_calls": len(baseline_decisions),
        "failed_http_attempts": 0,
        "retry_attempts": 0,
        "input_tokens": fixed_input_tokens,
        "output_tokens": fixed_output_tokens,
        "total_tokens": fixed_input_tokens + fixed_output_tokens,
        "estimated_cost_usd": fixed_cost,
        "summed_inference_latency_seconds": sum(
            float(row["request_latency_seconds"]) for row in baseline_decisions
        ),
        "wall_time_seconds": None,
        "final_matching_reviews": len(baseline_matches),
        "final_rejected_reviews": len(evaluation_ids) - len(baseline_matches),
        "completion_status": "complete",
        "predicates": fixed_predicates,
    }

    optimized = {
        key: optimized_report[key]
        for key in [
            "total_input_reviews",
            "total_completed_reviews",
            "completed_decisions",
            "http_attempts",
            "successful_api_calls",
            "failed_http_attempts",
            "retry_attempts",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "estimated_cost_usd",
            "summed_inference_latency_seconds",
            "wall_time_seconds",
            "final_matching_reviews",
            "final_rejected_reviews",
            "completion_status",
        ]
    }
    optimized["source"] = "fresh_live_jev_optimized_execution"
    optimized["predicates"] = optimized_report["predicates"]

    profile = load_json(experiment / "profiling" / "report.json")
    plan = load_json(experiment / "planning" / "selected_plan.json")
    candidate_payload = load_json(experiment / "planning" / "candidate_plans.json")
    selected_candidate = next(
        candidate for candidate in candidate_payload["candidates"] if candidate["selected"]
    )
    failed_transport = load_json(
        experiment
        / "optimized_failed_transport_original_20261102T174727Z"
        / "report.json"
    )

    fixed_only = sorted(baseline_matches - optimized_matches)
    optimized_only = sorted(optimized_matches - baseline_matches)
    gross_savings = fixed_cost - float(optimized_report["estimated_cost_usd"])
    profiling_cost = float(profile["estimated_cost_usd"])
    net_savings = gross_savings - profiling_cost
    break_even = math.ceil(profiling_cost / gross_savings) if gross_savings > 0 else None
    authorized_attempts = int(failed_transport["http_attempts"]) + int(
        optimized_report["http_attempts"]
    )

    comparison = {
        "schema_version": "m2-historical-trace-comparison-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "comparison_status": "complete",
        "comparison_method": "historical_fixed_trace_subset_vs_fresh_live_optimized",
        "dataset_fingerprint": optimized_manifest["dataset_fingerprint"],
        "evaluation_fingerprint": optimized_manifest["evaluation_fingerprint"],
        "evaluation_record_count": len(evaluation_ids),
        "identical_evaluation_record_ids": True,
        "requested_model": optimized_manifest["requested_model"],
        "resolved_model_versions": optimized_report["model_identifier"],
        "threshold": optimized_manifest["threshold"],
        "original_predicate_order": plan["original_predicate_order"],
        "selected_predicate_order": plan["selected_predicate_order"],
        "fixed": fixed,
        "optimized": optimized,
        "reductions": {
            "completed_decisions_percent": reduction(
                fixed["completed_decisions"], optimized["completed_decisions"]
            ),
            "http_attempts_percent": reduction(
                fixed["http_attempts"], optimized["http_attempts"]
            ),
            "input_tokens_percent": reduction(
                fixed["input_tokens"], optimized["input_tokens"]
            ),
            "total_tokens_percent": reduction(
                fixed["total_tokens"], optimized["total_tokens"]
            ),
            "estimated_cost_percent": reduction(
                fixed["estimated_cost_usd"], optimized["estimated_cost_usd"]
            ),
            "summed_inference_latency_percent": reduction(
                fixed["summed_inference_latency_seconds"],
                optimized["summed_inference_latency_seconds"],
            ),
            "wall_time_percent": None,
        },
        "wall_time_comparison": {
            "status": "not_calculated",
            "reason": "historical subset has no directly measured comparable wall time",
            "optimized_measured_wall_time_seconds": optimized["wall_time_seconds"],
        },
        "prediction_error": {
            "original_cost": prediction_error(
                float(plan["original_estimated_objective"]), fixed_cost
            ),
            "selected_cost": prediction_error(
                float(plan["selected_estimated_objective"]),
                float(optimized["estimated_cost_usd"]),
            ),
            "selected_evaluations": prediction_error(
                float(selected_candidate["estimated_evaluations"]),
                float(optimized["completed_decisions"]),
            ),
            "selected_total_tokens": prediction_error(
                float(selected_candidate["estimated_total_tokens"]),
                float(optimized["total_tokens"]),
            ),
            "selected_summed_latency_seconds": prediction_error(
                float(selected_candidate["estimated_summed_latency_seconds"]),
                float(optimized["summed_inference_latency_seconds"]),
            ),
        },
        "profiling_overhead": {
            key: profile[key]
            for key in [
                "completed_decisions",
                "http_attempts",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "estimated_cost_usd",
                "wall_time_seconds",
            ]
        },
        "savings_after_profiling": {
            "gross_execution_savings_usd": gross_savings,
            "net_savings_for_one_query_usd": net_savings,
            "profiling_plus_optimized_cost_usd": profiling_cost
            + float(optimized["estimated_cost_usd"]),
            "break_even_execution_count": break_even,
        },
        "output_equivalence": {
            "equivalent": not fixed_only and not optimized_only,
            "fixed_final_review_ids": sorted(baseline_matches),
            "optimized_final_review_ids": sorted(optimized_matches),
            "fixed_only_review_ids": fixed_only,
            "optimized_only_review_ids": optimized_only,
        },
        "shared_decision_stability": decision_stability(
            baseline_decisions, optimized_decisions
        ),
        "authorized_attempt_accounting": {
            "absolute_limit": 2640,
            "preserved_initial_transport_failures": failed_transport["http_attempts"],
            "successful_run_http_attempts": optimized_report["http_attempts"],
            "total_counted_attempts": authorized_attempts,
            "remaining_authorized_attempts": 2640 - authorized_attempts,
            "initial_transport_failure_cost_usd": 0.0,
        },
        "sources": {
            "historical_baseline_directory": str(baseline),
            "optimized_directory": str(optimized_dir),
            "profiling_directory": str(experiment / "profiling"),
            "failed_transport_artifact_directory": str(
                experiment / "optimized_failed_transport_original_20261102T174727Z"
            ),
        },
        "limitations": [
            "The fixed-order comparator is trace-derived, not a fresh concurrent run.",
            "No wall-clock speedup is calculated from the historical subset.",
            "Independent live model calls can vary even with the same resolved model.",
            "The planner uses unconditional selectivity and does not model predicate correlation.",
        ],
    }

    output_path = experiment / "comparison.json"
    output_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    ids = "\n".join(f"- `{review_id}`" for review_id in sorted(optimized_matches))
    predicate_rows = "\n".join(
        f"| Historical fixed | {row['predicate_id']} | {row['input_record_count']} | "
        f"{row['passing_record_count']} | {row['rejected_record_count']} | "
        f"{row['selectivity']:.6f} |"
        for row in fixed_predicates
    )
    predicate_rows += "\n" + "\n".join(
        f"| Fresh optimized | {row['predicate_id']} | {row['input_record_count']} | "
        f"{row['passing_record_count']} | {row['rejected_record_count']} | "
        f"{row['selectivity']:.6f} |"
        for row in optimized_report["predicates"]
    )
    fixed_only_text = "\n".join(f"- `{review_id}`" for review_id in fixed_only) or "- None"
    optimized_only_text = (
        "\n".join(f"- `{review_id}`" for review_id in optimized_only) or "- None"
    )
    crossing_text = "\n".join(
        f"- `{row['review_id']}` / `{row['predicate_id']}`: "
        f"{row['fixed_probability']:.2f} ({'pass' if row['fixed_passed'] else 'fail'}) "
        f"to {row['optimized_probability']:.2f} "
        f"({'pass' if row['optimized_passed'] else 'fail'})"
        for row in comparison["shared_decision_stability"]["crossing_details"]
    ) or "- None"
    token_summary = (
        f"{optimized['input_tokens']} / {optimized['output_tokens']} / "
        f"{optimized['total_tokens']}"
    )
    selected_relative_error = comparison["prediction_error"]["selected_cost"][
        "relative_error_percent"
    ]
    shared_agreement = comparison["shared_decision_stability"][
        "boolean_agreement_percent"
    ]
    transport_note = (
        "The failed transport attempt produced no model decisions, tokens, or cost and "
        "is preserved separately. No new fixed-order run was performed."
    )
    summary = f"""# Milestone 2 Jev Optimized Evaluation

## Execution

- Selected order: `{' -> '.join(plan['selected_predicate_order'])}`
- Held-out SemCEB reviews: {len(evaluation_ids)}
- Completed decisions / successful Jev calls: {optimized['completed_decisions']}
- Input / output / total tokens: {token_summary}
- Estimated execution cost: ${optimized['estimated_cost_usd']:.9f}
- Summed inference latency: {optimized['summed_inference_latency_seconds']:.6f} seconds
- Measured optimized wall time: {optimized['wall_time_seconds']:.6f} seconds
- Final matches: {optimized['final_matching_reviews']}
- Completion: `{optimized['completion_status']}`

## Per-predicate flow

| Source | Predicate | Input | Passed | Rejected | Observed selectivity |
|---|---|---:|---:|---:|---:|
{predicate_rows}

## Historical fixed-order trace comparison

- Fixed decisions: {fixed['completed_decisions']}
- Optimized decisions: {optimized['completed_decisions']}
- Decision reduction: {comparison['reductions']['completed_decisions_percent']:.6f}%
- Fixed total tokens: {fixed['total_tokens']}
- Optimized total tokens: {optimized['total_tokens']}
- Token reduction: {comparison['reductions']['total_tokens_percent']:.6f}%
- Fixed estimated cost: ${fixed_cost:.9f}
- Optimized estimated cost: ${optimized['estimated_cost_usd']:.9f}
- Cost reduction: {comparison['reductions']['estimated_cost_percent']:.6f}%
- Fixed summed inference latency: {fixed['summed_inference_latency_seconds']:.6f} seconds
- Optimized summed inference latency: {optimized['summed_inference_latency_seconds']:.6f} seconds
- Summed-latency reduction: {comparison['reductions']['summed_inference_latency_percent']:.6f}%
- Wall-clock speedup: not calculated; the historical subset has no comparable measured wall time.

## Prediction error

- Predicted optimized evaluations: {selected_candidate['estimated_evaluations']}
- Observed optimized evaluations: {optimized['completed_decisions']}
- Predicted optimized cost: ${plan['selected_estimated_objective']:.9f}
- Observed optimized cost: ${optimized['estimated_cost_usd']:.9f}
- Absolute cost error: ${comparison['prediction_error']['selected_cost']['absolute_error']:.9f}
- Relative cost error: {selected_relative_error:.6f}%

## Profiling overhead

- Profiling calls: {profile['completed_decisions']}
- Profiling tokens: {profile['total_tokens']}
- Profiling cost: ${profiling_cost:.9f}
- Gross execution savings: ${gross_savings:.9f}
- Net savings for one query after profiling: ${net_savings:.9f}
- Break-even executions: {break_even}

## Output equivalence

- Exact final match-set equality: `{not fixed_only and not optimized_only}`
- Fixed-only IDs: {len(fixed_only)}
- Optimized-only IDs: {len(optimized_only)}
- Shared decision-pair Boolean agreement: {shared_agreement:.6f}%
- Shared-pair threshold crossings: {comparison['shared_decision_stability']['threshold_crossings']}

Fixed-only final IDs:

{fixed_only_text}

Optimized-only final IDs:

{optimized_only_text}

Shared-pair threshold crossings:

{crossing_text}

## Final optimized match IDs

{ids}

## Authorization accounting

- Absolute authorized attempt limit: 2640
- Preserved pre-run transport failures: {failed_transport['http_attempts']}
- Successful run attempts: {optimized['http_attempts']}
- Total counted attempts: {authorized_attempts}
- Remaining authorization: {2640 - authorized_attempts}

{transport_note}
"""
    (experiment / "summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
