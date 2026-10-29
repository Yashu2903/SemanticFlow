from __future__ import annotations

import argparse
import asyncio
import json

from semanticflow.providers.base import RequestBudget
from semanticflow.providers.jev import JevProvider
from semanticflow.query import PredicateSpec


async def smoke_test(model: str) -> None:
    provider = JevProvider.from_env(model, max_retries=0, timeout_seconds=30)
    try:
        decision = await provider.decide(
            "The product broke after one day and I want a replacement.",
            PredicateSpec(
                id="product_problem",
                instructions="Does the customer report a product problem?",
            ),
            RequestBudget(1),
        )
    finally:
        await provider.aclose()
    print(
        json.dumps(
            {
                "status": "success",
                "probability": decision.probability,
                "model": decision.model_identifier,
                "input_tokens": decision.input_tokens,
                "output_tokens": decision.output_tokens,
                "latency_seconds": round(decision.latency_seconds, 3),
                "request_attempts": decision.request_attempts,
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Make one live Jev API smoke-test request")
    parser.add_argument("--model", default="jev-latest")
    args = parser.parse_args()
    asyncio.run(smoke_test(args.model))


if __name__ == "__main__":
    main()
