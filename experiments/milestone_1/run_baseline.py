from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from semanticflow.providers.jev import JevProvider
from semanticflow.providers.mock import MockProvider
from semanticflow.runtime import load_settings, run_query

WORKLOADS = (
    "product_quality.yaml",
    "product_defects.yaml",
    "customer_experience.yaml",
    "refund_requests.yaml",
)


async def run_all(args: argparse.Namespace) -> int:
    settings = load_settings(args.settings)
    provider_settings = settings.get("providers", {}).get(args.provider, {})
    if args.provider == "mock":
        provider = MockProvider(model=provider_settings.get("model", "mock-deterministic-v1"))
    else:
        provider = JevProvider.from_env(
            model=provider_settings.get("model", "jev-latest"),
            api_url=provider_settings.get("api_url", "https://api.typesafe.ai/v1/systemone"),
            timeout_seconds=float(provider_settings.get("timeout_seconds", 30)),
            max_retries=int(provider_settings.get("max_retries", 2)),
            backoff_seconds=float(provider_settings.get("backoff_seconds", 0.5)),
        )
    exit_code = 0
    try:
        for filename in WORKLOADS:
            query = args.queries_dir / filename
            print(f"Running {query} with {args.provider}")
            outcome = await run_query(
                query_path=query,
                settings=settings,
                provider=provider,
                output_dir=args.output_dir,
                cache_enabled=args.cache == "on",
                limit_override=args.limit,
                max_calls_override=args.max_calls,
            )
            print(f"  {outcome.report['final_record_count']} matches -> {outcome.run_dir}")
            if outcome.report["completion_status"] != "complete":
                exit_code = 2
    finally:
        await provider.aclose()
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all Milestone 1 reference workloads")
    parser.add_argument("--provider", required=True, choices=("mock", "jev"))
    parser.add_argument("--settings", type=Path, default=Path("configs/settings.yaml"))
    parser.add_argument("--queries-dir", type=Path, default=Path("configs/queries"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--cache", choices=("on", "off"), default="off")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-calls", type=int)
    return asyncio.run(run_all(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
