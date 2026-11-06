from __future__ import annotations

import argparse
import asyncio
import hashlib
from pathlib import Path

from semanticflow.dataset.loader import load_dataset
from semanticflow.optimizer.models import load_optimizer_settings
from semanticflow.optimizer.workflow import run_offline_experiment
from semanticflow.providers.mock import MockProvider
from semanticflow.query.models import load_query
from semanticflow.runtime import load_settings

PASS_PERCENT = {
    "expensive_broad": 80,
    "moderate_filter": 40,
    "cheap_selective": 10,
}

SYNTHETIC_TOKEN_USAGE = {
    "expensive_broad": (400, 20),
    "moderate_filter": (120, 10),
    "cheap_selective": (30, 5),
}


def deterministic_probabilities(
    review_texts: list[str],
) -> dict[tuple[str, str], float]:
    probabilities: dict[tuple[str, str], float] = {}
    for text in review_texts:
        for predicate_id, pass_percent in PASS_PERCENT.items():
            digest = hashlib.sha256(f"{predicate_id}\0{text}".encode()).digest()
            percentile = int.from_bytes(digest[:8], "big") % 100
            probabilities[(text, predicate_id)] = 0.9 if percentile < pass_percent else 0.1
    return probabilities


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Run the controlled Milestone 2 mock reordering demonstration"
    )
    result.add_argument(
        "--query",
        type=Path,
        default=Path("configs/queries/milestone_2_controlled.yaml"),
    )
    result.add_argument("--settings", type=Path, default=Path("configs/settings.yaml"))
    result.add_argument(
        "--optimizer-config", type=Path, default=Path("configs/optimizer.yaml")
    )
    result.add_argument("--output-dir", type=Path, default=Path("results/milestone_2"))
    result.add_argument(
        "--experiment-id", default="offline_controlled_reordering_semceb_v1"
    )
    return result


async def run(arguments: argparse.Namespace) -> int:
    query = load_query(arguments.query)
    records = load_dataset(query.dataset_path, limit=query.limit)
    settings = load_settings(arguments.settings)
    settings["pricing"] = {
        "id": "controlled-mock-synthetic-v1",
        "status": "synthetic_infrastructure_only",
        "input_per_million_usd": 2.0,
        "output_per_million_usd": 4.0,
    }
    optimizer = load_optimizer_settings(arguments.optimizer_config)
    optimizer = optimizer.model_copy(
        update={
            "planner": optimizer.planner.model_copy(
                update={"objective": "estimated_cost_usd"}
            )
        }
    )
    provider = MockProvider(
        deterministic_probabilities([record.review_text for record in records]),
        model="mock-controlled-reordering-v1",
        usage_by_predicate=SYNTHETIC_TOKEN_USAGE,
    )
    try:
        experiment_dir, comparison = await run_offline_experiment(
            query_path=arguments.query,
            settings=settings,
            optimizer=optimizer,
            provider=provider,
            output_root=arguments.output_dir,
            experiment_id=arguments.experiment_id,
        )
    finally:
        await provider.aclose()

    original = comparison["original_predicate_order"]
    selected = comparison["selected_predicate_order"]
    fixed = comparison["fixed"]
    optimized = comparison["optimized"]
    if selected == original:
        raise RuntimeError("controlled experiment did not reorder predicates")
    if optimized["completed_decisions"] >= fixed["completed_decisions"]:
        raise RuntimeError("controlled optimized execution did not reduce evaluations")
    if not comparison["output_equivalence"]["equivalent"]:
        raise RuntimeError("controlled fixed and optimized matches differ")

    print(f"Original order: {' -> '.join(original)}")
    print(f"Selected order: {' -> '.join(selected)}")
    print(
        "Completed decisions: "
        f"{fixed['completed_decisions']} fixed, "
        f"{optimized['completed_decisions']} optimized"
    )
    print(
        "Estimated execution cost: "
        f"${fixed['estimated_cost_usd']:.8f} fixed, "
        f"${optimized['estimated_cost_usd']:.8f} optimized"
    )
    print(f"Equivalent final IDs: {comparison['output_equivalence']['equivalent']}")
    print(f"Artifacts: {experiment_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parser().parse_args())))
