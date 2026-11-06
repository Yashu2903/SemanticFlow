from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from semanticflow.dataset.loader import load_dataset
from semanticflow.optimizer.models import load_optimizer_settings
from semanticflow.optimizer.workflow import prepare_split
from semanticflow.query.models import load_query
from semanticflow.runtime import load_settings

EXPECTED_DATASET_FINGERPRINT = (
    "sha256:21cd6032e72379828f334422e29365688c5b5063c07ca23f08f8f025c2d55a99"
)
BASELINE_CALLS = 1207
BASELINE_INPUT_TOKENS = 384225
BASELINE_OUTPUT_TOKENS = 26732


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Read-only Jev profiling preflight")
    result.add_argument(
        "--query", type=Path, default=Path("configs/queries/product_quality.yaml")
    )
    result.add_argument(
        "--settings",
        type=Path,
        default=Path("configs/settings.milestone_2_jev.yaml"),
    )
    result.add_argument(
        "--optimizer-config",
        type=Path,
        default=Path("configs/optimizer.milestone_2_jev.yaml"),
    )
    result.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("results/milestone_2/jev_product_quality_profile_20261102"),
    )
    return result


def run(arguments: argparse.Namespace) -> int:
    query = load_query(arguments.query)
    settings = load_settings(arguments.settings)
    optimizer = load_optimizer_settings(arguments.optimizer_config)
    records = load_dataset(query.dataset_path, limit=query.limit)
    profiling, evaluation, split = prepare_split(query, optimizer)
    load_dotenv(override=False)
    credentials_available = bool(os.getenv("TYPESAFE_API_KEY", "").strip())
    provider = settings["providers"]["jev"]
    pricing = settings["pricing"]

    if split.dataset_fingerprint != EXPECTED_DATASET_FINGERPRINT:
        raise RuntimeError("dataset fingerprint differs from the approved baseline")
    if len(records) != 1000 or len({record.review_id for record in records}) != 1000:
        raise RuntimeError("dataset must contain exactly 1,000 unique usable reviews")
    if len(profiling) != 200 or len(evaluation) != 800:
        raise RuntimeError("unexpected profiling/evaluation split sizes")
    if set(split.profiling_ids) & set(split.evaluation_ids):
        raise RuntimeError("profiling and evaluation sets overlap")
    if provider["model"] != "jev-1.14.0":
        raise RuntimeError("Jev model must be pinned to jev-1.14.0")
    if not credentials_available:
        raise RuntimeError("TYPESAFE_API_KEY is unavailable")
    required_decisions = len(profiling) * len(query.predicates)
    retry_inclusive_maximum = required_decisions * (int(provider["max_retries"]) + 1)
    if optimizer.profiling.max_attempts != retry_inclusive_maximum:
        raise RuntimeError("profiling attempt cap must cover exactly the configured retry ceiling")
    if optimizer.execution.cache_enabled:
        raise RuntimeError("cache must be disabled")

    estimated_input = BASELINE_INPUT_TOKENS / BASELINE_CALLS * required_decisions
    estimated_output = BASELINE_OUTPUT_TOKENS / BASELINE_CALLS * required_decisions
    expected_cost = (
        estimated_input * float(pricing["input_per_million_usd"])
        + estimated_output * float(pricing["output_per_million_usd"])
    ) / 1_000_000
    conservative_retry_proxy = expected_cost * (int(provider["max_retries"]) + 1)
    command = (
        ".\\.venv\\Scripts\\python.exe -m semanticflow.cli profile "
        "--provider jev --authorize-paid "
        f"--query {arguments.query} --settings {arguments.settings} "
        f"--optimizer-config {arguments.optimizer_config} "
        f"--experiment-dir {arguments.experiment_dir}"
    )

    print(f"dataset_fingerprint={split.dataset_fingerprint}")
    print(f"source_records={split.source_record_count}")
    print(f"profiling_records={split.profiling_size}")
    print(f"evaluation_records={split.evaluation_size}")
    print("split_overlap=0")
    print(f"split_seed={split.seed}")
    print(f"split_algorithm={split.split_algorithm}")
    print(f"profiling_fingerprint={split.profiling_fingerprint}")
    print(f"evaluation_fingerprint={split.evaluation_fingerprint}")
    print(f"requested_model={provider['model']}")
    print(f"credentials_available={credentials_available}")
    print(f"planned_decisions={required_decisions}")
    print(f"max_http_attempts={optimizer.profiling.max_attempts}")
    print(f"estimated_input_tokens={estimated_input:.0f}")
    print(f"estimated_output_tokens={estimated_output:.0f}")
    print(f"estimated_list_cost_usd={expected_cost:.8f}")
    print(f"conservative_retry_proxy_usd={conservative_retry_proxy:.8f}")
    print(f"pricing_id={pricing['id']}")
    print(f"pricing_source={pricing['source']}")
    print(f"pricing_verified_date={pricing['verified_date']}")
    print(f"command={command}")
    print("network_requests_made=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
