"""Offline preflight for proposed Milestone 3 Jev experiments.

Reads configs and stored Milestone 2 reports only. Makes no network requests, does not read
credentials, and runs nothing paid; it prints the commands that would need authorization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from semanticflow.dataset.loader import load_dataset
from semanticflow.optimizer.models import load_optimizer_settings
from semanticflow.optimizer.workflow import prepare_split
from semanticflow.query.models import load_query
from semanticflow.runtime import load_settings

M2_EXPERIMENT = Path("results/milestone_2/jev_product_quality_profile_20261102")
WORKLOADS = {
    "listing_mismatch": Path("configs/queries/listing_mismatch.yaml"),
    "positive_experience": Path("configs/queries/positive_experience.yaml"),
    "mixed_feedback": Path("configs/queries/mixed_feedback.yaml"),
}
STRATEGY_CONFIGS = {
    "independent": Path("configs/optimizer.milestone_2_jev.yaml"),
    "correlation_aware": Path("configs/optimizer.milestone_3.yaml"),
    "correlation_aware_confidence": Path("configs/optimizer.milestone_3_confidence.yaml"),
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--settings", type=Path, default=Path("configs/settings.milestone_2_jev.yaml")
    )
    result.add_argument("--output", type=Path, help="Optional new JSON file for the estimates")
    return result


def _usage_per_decision() -> dict[str, float]:
    profile = json.loads((M2_EXPERIMENT / "profiling" / "report.json").read_text("utf-8"))
    held_out = json.loads((M2_EXPERIMENT / "optimized" / "report.json").read_text("utf-8"))
    return {
        "profiling_input": profile["input_tokens"] / profile["completed_decisions"],
        "profiling_output": profile["output_tokens"] / profile["completed_decisions"],
        "held_out_input": held_out["input_tokens"] / held_out["completed_decisions"],
        "held_out_output": held_out["output_tokens"] / held_out["completed_decisions"],
    }


def _cost(decisions: float, input_per: float, output_per: float, pricing: dict) -> float:
    return (
        decisions * input_per * float(pricing["input_per_million_usd"])
        + decisions * output_per * float(pricing["output_per_million_usd"])
    ) / 1_000_000


def run(arguments: argparse.Namespace) -> int:
    settings = load_settings(arguments.settings)
    pricing = settings["pricing"]
    retries = int(settings["providers"]["jev"]["max_retries"])
    usage = _usage_per_decision()
    profiling_unit = _cost(1, usage["profiling_input"], usage["profiling_output"], pricing)
    held_out_unit = _cost(1, usage["held_out_input"], usage["held_out_output"], pricing)
    plans = []
    for name, query_path in WORKLOADS.items():
        query = load_query(query_path)
        optimizer = load_optimizer_settings(STRATEGY_CONFIGS["correlation_aware"])
        records = load_dataset(query.dataset_path, limit=query.limit)
        _, _, split = prepare_split(query, optimizer)
        width = len(query.predicates)
        profiling_decisions = split.profiling_size * width
        minimum_run = split.evaluation_size
        maximum_run = split.evaluation_size * width
        # Fixed order plus at most four distinct selected orders, plus one repeat of the
        # fixed order as a response-variability control.
        maximum_runs = 6
        experiment_root = f"results/milestone_3/jev_{name}"
        commands = [
            (
                ".\\.venv\\Scripts\\python.exe -m semanticflow.cli profile --provider jev "
                f"--authorize-paid --query {query_path.as_posix()} "
                f"--settings {arguments.settings.as_posix()} "
                f"--optimizer-config {STRATEGY_CONFIGS['correlation_aware'].as_posix()} "
                f"--experiment-dir {experiment_root}/profile"
            )
        ]
        plans.append(
            {
                "workload": name,
                "query": query_path.as_posix(),
                "records": len(records),
                "dataset_fingerprint": split.dataset_fingerprint,
                "profiling_fingerprint": split.profiling_fingerprint,
                "evaluation_fingerprint": split.evaluation_fingerprint,
                "predicates": [item.id for item in query.predicates],
                "profiling": {
                    "decisions": profiling_decisions,
                    "max_http_attempts": optimizer.profiling.max_attempts,
                    "required_attempt_cap": profiling_decisions * (retries + 1),
                    "estimated_cost_usd": profiling_decisions * profiling_unit,
                    "cost_if_every_allowed_attempt_were_billed_usd": (
                        optimizer.profiling.max_attempts * profiling_unit
                    ),
                },
                "held_out_run": {
                    "minimum_decisions": minimum_run,
                    "maximum_decisions": maximum_run,
                    "max_http_attempts": optimizer.execution.max_attempts_per_run,
                    "estimated_cost_usd_range": [
                        minimum_run * held_out_unit,
                        maximum_run * held_out_unit,
                    ],
                },
                "maximum_held_out_runs": maximum_runs,
                "maximum_total_estimated_cost_usd": (
                    profiling_decisions * profiling_unit
                    + maximum_runs * maximum_run * held_out_unit
                ),
                "maximum_total_http_attempts": (
                    optimizer.profiling.max_attempts
                    + maximum_runs * optimizer.execution.max_attempts_per_run
                ),
                "first_paid_command": commands[0],
            }
        )
    repeat_decisions = 958
    repeat_cap = load_optimizer_settings(STRATEGY_CONFIGS["independent"]).execution
    repeat = {
        "experiment": "product_quality fixed-order fresh repeat (response-variability control)",
        "expected_decisions": repeat_decisions,
        "expected_decisions_source": "Milestone 1 fixed trace replayed on the 800 held-out reviews",
        "max_http_attempts": repeat_cap.max_attempts_per_run,
        "estimated_cost_usd": repeat_decisions * held_out_unit,
        "cost_if_every_allowed_attempt_were_billed_usd": (
            repeat_cap.max_attempts_per_run * held_out_unit
        ),
        "preparation_command": (
            ".\\.venv\\Scripts\\python.exe experiments/milestone_3/prepare_strategy_experiment.py "
            f"--source {M2_EXPERIMENT.as_posix()} "
            "--destination results/milestone_3/jev_product_quality_fixed_repeat --copy-planning"
        ),
        "paid_command": (
            ".\\.venv\\Scripts\\python.exe -m semanticflow.cli run --provider jev --authorize-paid "
            "--strategy fixed --query configs/queries/product_quality.yaml "
            f"--settings {arguments.settings.as_posix()} "
            f"--optimizer-config {STRATEGY_CONFIGS['independent'].as_posix()} "
            "--experiment-dir results/milestone_3/jev_product_quality_fixed_repeat"
        ),
    }
    report = {
        "network_requests_made": 0,
        "credentials_read": False,
        "pricing_id": pricing["id"],
        "usage_per_decision_from_stored_milestone_2_reports": usage,
        "profiling_cost_per_decision_usd": profiling_unit,
        "held_out_cost_per_decision_usd": held_out_unit,
        "retry_ceiling_per_decision": retries + 1,
        "workloads": plans,
        "variability_repeat": repeat,
        "note": (
            "Token usage of new predicates is assumed to match the stored product_quality "
            "averages; instruction length differences make this approximate."
        ),
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if arguments.output is not None:
        if arguments.output.exists():
            raise FileExistsError(f"Refusing to overwrite {arguments.output}")
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text + "\n", encoding="utf-8")
    for item in plans:
        if item["profiling"]["max_http_attempts"] < item["profiling"]["required_attempt_cap"]:
            raise RuntimeError(f"{item['workload']}: profiling attempt cap is below the ceiling")
        if item["held_out_run"]["max_http_attempts"] < item["held_out_run"]["maximum_decisions"]:
            raise RuntimeError(f"{item['workload']}: execution cap below worst-case decisions")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
