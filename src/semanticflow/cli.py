from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from semanticflow.providers.jev import JevProvider
from semanticflow.providers.mock import MockProvider
from semanticflow.runtime import load_settings, run_query


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a SemanticFlow baseline semantic query")
    parser.add_argument(
        "--query", type=Path, default=Path("configs/queries/product_quality.yaml")
    )
    parser.add_argument("--settings", type=Path, default=Path("configs/settings.yaml"))
    parser.add_argument("--provider", choices=("mock", "jev"), default="mock")
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--cache", choices=("on", "off"), default="off")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--max-calls",
        type=int,
        help="Override the retry-inclusive provider request-attempt budget",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    settings = load_settings(args.settings)
    provider_config = settings.get("providers", {}).get(args.provider, {})
    if args.provider == "mock":
        provider = MockProvider(model=provider_config.get("model", "mock-deterministic-v1"))
    else:
        provider = JevProvider.from_env(
            model=provider_config.get("model", "jev-latest"),
            api_url=provider_config.get("api_url", "https://api.typesafe.ai/v1/systemone"),
            timeout_seconds=float(provider_config.get("timeout_seconds", 30)),
            max_retries=int(provider_config.get("max_retries", 2)),
            backoff_seconds=float(provider_config.get("backoff_seconds", 0.5)),
        )
    print(f"Provider: {provider.name}; workload: {args.query}")
    try:
        outcome = await run_query(
            query_path=args.query,
            settings=settings,
            provider=provider,
            output_dir=args.output_dir,
            cache_enabled=args.cache == "on",
            labels_path=args.labels,
            limit_override=args.limit,
            max_calls_override=args.max_calls,
        )
    finally:
        await provider.aclose()
    report = outcome.report
    for stage in report["predicates"]:
        print(
            f"Predicate {stage['predicate_id']}: {stage['input_record_count']} input, "
            f"{stage['passing_record_count']} surviving"
        )
    print(f"Actual provider calls: {report['actual_provider_calls']}")
    print(f"Final output count: {report['final_record_count']}")
    print(f"Total execution time: {report['wall_time_seconds']:.3f}s")
    print(f"Results directory: {outcome.run_dir}")
    return 0 if report["completion_status"] == "complete" else 2


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        return asyncio.run(_run(build_parser().parse_args(argv)))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
