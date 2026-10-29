from __future__ import annotations

import argparse
from pathlib import Path

from semanticflow.dataset.preparation import prepare_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a deterministic SemanticFlow sample")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = prepare_sample(
        args.source,
        args.output,
        sample_size=args.sample_size,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(
        f"Wrote {result['output_row_count']} deterministic real reviews to {args.output} "
        f"using seed {args.seed}"
    )


if __name__ == "__main__":
    main()

