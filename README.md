# SemanticFlow

SemanticFlow is an academic research harness for measuring the cost, latency, and quality of
natural-language predicates over tabular data. Milestone 1 remains the fixed-order reference
engine. Milestone 2 adds an offline cost-based planner that profiles unconditional predicate
selectivity and measured inference cost, exhaustively ranks orders, and sends the selected order
through the same sequential AND executor. Milestone 3 adds an optional correlation-aware
cardinality estimator, offline counterfactual replay of stored decisions, profiling-cost
economics, paired-bootstrap plan stability, and an offline adaptive-profiling simulation. The
independence planner remains the default. SemanticFlow does not fuse predicates or adapt during
execution.

## Architecture and requirements

YAML is validated into an ordered query, DuckDB reads Parquet into stable `review_id` /
`review_text` records, and the asynchronous executor evaluates one predicate stage at a time.
Within a stage it uses bounded concurrency; failures short-circuit records between stages. A
provider-independent interface supports a deterministic mock and TypeSafe AI Jev. Each run writes
`matches.jsonl`, `decisions.jsonl`, `report.json`, and `manifest.json` to a unique directory. See
[architecture](docs/architecture.md) and the [experiment protocol](docs/experiment_protocol.md).

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest -q
ruff check .
```

## Dataset acquisition, preparation, and validation

SemanticFlow uses the official SemCEB `Arts_Crafts_and_Sewing__raw_5core` prepared review artifact:
936,216 filtered reviews in `reviews_filtered.parquet`. It downloads reviews only—no images or
embeddings—and never downloads data during normal application startup.

```powershell
python scripts/download_dataset.py --source semceb `
  --output data/raw/reviews_filtered.parquet
```

`--source auto` tries SemCEB first and falls back only when that download is inaccessible. The
fallback is the official McAuley-Lab Amazon Reviews 2023 raw Arts, Crafts & Sewing review file; its
provenance explicitly says it is **not** SemCEB's filtered 5-core dataset. Downloads are streamed to
a `.part` file, SHA-256 hashed, atomically renamed, and accompanied by a `.source.json` sidecar.

Create the common deterministic 1,000-review sample:

```powershell
python scripts/prepare_sample.py --source data/raw/reviews_filtered.parquet `
  --output data/processed/reviews_1000.parquet --sample-size 1000 --seed 2026
```

Then validate row count, required columns, text, unique IDs, DuckDB readability, and a complete
second preparation pass from the same source and seed:

```powershell
python scripts/validate_dataset.py --source data/raw/reviews_filtered.parquet `
  --dataset data/processed/reviews_1000.parquet `
  --manifest data/processed/dataset_manifest.json
```

Use `--overwrite` only when intentionally replacing an existing sample. Source IDs are preserved
when available; otherwise IDs are SHA-256 hashes of canonical source rows. Sampling ranks
`sha256(seed:review_id)`, making selection independent of Parquet row order. The sample retains
rating, product IDs, user ID, title, timestamp, helpful votes, and verified-purchase status. The
manifest records source/output checksums, provenance, seed, algorithm version, row counts, and
timestamps. Query YAML files declare fixed predicate order, threshold, record limit, and budget.

## Running queries

The safe default is the mock provider. It is deterministic infrastructure, not a semantic model,
and its synthetic token counts and decisions must never be reported as research accuracy.

```powershell
semanticflow --provider mock --query configs/queries/product_quality.yaml
semanticflow --provider mock --query configs/queries/product_quality.yaml --cache on --limit 50
python experiments/milestone_1/run_baseline.py --provider mock
```

## Milestone 2 optimizer

The optimizer deterministically splits the 1,000-review SemCEB sample into 200 profiling and 800
held-out evaluation reviews. Profiling evaluates every predicate on every profiling review without
early rejection. Planning enumerates every permutation for 2–6 predicates and minimizes the
configured objective in `configs/optimizer.yaml`. Fixed and optimized execution use the same held-
out IDs and the existing sequential executor.

Run the complete credential-free workflow:

```powershell
semanticflow experiment --provider mock `
  --query configs/queries/product_quality.yaml `
  --optimizer-config configs/optimizer.yaml `
  --output-dir results/milestone_2
```

The individual phases are also available:

```powershell
semanticflow profile --provider mock --experiment-dir results/milestone_2/example
semanticflow plan --experiment-dir results/milestone_2/example
semanticflow run --provider mock --strategy fixed --experiment-dir results/milestone_2/example
semanticflow run --provider mock --strategy optimized --experiment-dir results/milestone_2/example
semanticflow compare --experiment-dir results/milestone_2/example
```

Mock token usage is explicitly synthetic and validates only plumbing. Jev commands additionally
require `--authorize-paid`, but that flag is only a CLI safety acknowledgement: review the exact
dataset, model, request caps, estimated usage, and cost and obtain experiment-specific approval
before using it. See the [Milestone 2 architecture](docs/milestone_2_architecture.md) and
[experiment protocol](docs/milestone_2_experiment_protocol.md).

## Milestone 3 offline analysis

Select the estimator in the optimizer config:

- `cardinality_estimator: correlation_aware` with `cost_model_version: correlation-aware-and-v1`
  uses chain-rule conditionals from the joint profiling matrix, with Beta shrinkage.
- `selection_policy: confidence_aware` keeps the original order unless a paired bootstrap shows
  the cheaper order is credibly cheaper.

See `configs/optimizer.milestone_3.yaml` and `configs/optimizer.milestone_3_confidence.yaml`.
Two new commands work only on stored decisions and never construct a provider:

```powershell
semanticflow replay --decisions "m1=<run>/decisions.jsonl" --split-manifest <split_manifest.json> `
  --split-part evaluation --model jev-1.14.0 --order poor_quality,product_defect,dissatisfaction
semanticflow analyze --experiment-dir <milestone 2 experiment> `
  --historical-baseline-dir <milestone 1 run> --output-dir results/milestone_3/<new directory>
```

On the stored `product_quality` Jev traces:

- Offline replay estimates that 4–5 of the 6 calls saved in Milestone 2 come from predicate
  ordering. This is an estimate under stated assumptions, not a causal measurement. The
  remainder is consistent with Jev response variability.
- The correlation-aware estimator halves stage-cardinality error under cross-validation.
  Its 90% predictive intervals cover every held-out count, but its held-out point predictions
  are biased high, and it is not more accurate on held-out total evaluations.
- The selected order wins only 76.9% of bootstrap resamples, and its savings interval includes
  zero.
- A fresh profile's break-even depends on the savings basis:
  - 62–63 executions on predicted savings;
  - 113 on the observed, confounded difference;
  - 151–224 on replay-estimated savings;
  - 18 executions to never across the bootstrap interval.

  The economics gate can skip adaptive profiling entirely when even the bounded saving
  cannot cover the first checkpoint.

A later live Jev evaluation ran contemporaneous ABBA sessions on `product_quality` and the
three new workloads (19,407 HTTP attempts, $0.264). Independence beat the original order on
all four; correlation-aware agreed on two, kept the original on `positive_experience`, and
selected the cheapest order on `mixed_feedback` (45% execution-cost reduction). Details are in
the [live evaluation](docs/milestone_3_live_evaluation.md).

Full offline results are in the [research report](docs/milestone_3_research_report.md). The design is in
the [Milestone 3 architecture](docs/milestone_3_architecture.md), and the live protocol is in
the [benchmark protocol](docs/milestone_3_benchmark_protocol.md). The three additional
workloads `listing_mismatch`, `positive_experience`, and `mixed_feedback` were executed on
Jev in the live evaluation.

The executed Milestone 1 real-data mock run (`product_quality`, seed 2026) loaded 1,000 reviews and
produced stage counts `1000 -> 476 -> 234 -> 117`, 1,710 decisions/provider calls, and a complete
four-artifact result directory. These deterministic mock outcomes validate plumbing, not semantic
quality.

Baseline caching is off by default. Disabled caching performs no lookups, so both cache counters
remain zero. When enabled, it is memory-only and scoped to one run. Its key
includes provider/model, review text, predicate ID and instructions, question type, and inference
parameters. Concurrent duplicates use one request. Reports keep logical evaluations, provider
calls, attempts, and hits/misses distinct.

For real Jev inference, put the API key in the ignored local `.env` file or set it in the process
environment; never place it in YAML or results. An existing process variable takes precedence over
`.env`.
The adapter follows TypeSafe's `POST /v1/systemone` contract with exactly one named `noul` question
per request.

```powershell
# .env: TYPESAFE_API_KEY=your-api-key
semanticflow --provider jev --query configs/queries/product_quality.yaml --cache off --limit 50
python experiments/milestone_1/run_baseline.py --provider jev --limit 50
```

For a deliberately incomplete, maximum-10-attempt real-data smoke run (retries consume the same
budget), use:

```powershell
semanticflow --provider jev --query configs/queries/product_quality.yaml `
  --limit 10 --max-calls 10 --cache off
```

If any row survives the first predicate, the next stage encounters the exhausted budget and the run
is retained with `completion_status: incomplete`, `incomplete_reason: request_budget_exhausted`,
empty completed matches, and separate budget diagnostics.

The runner requires an explicit provider and each workload enforces `max_calls`. Retry attempts
count against that budget. Authentication and validation failures are not retried; timeouts, 429,
and 5xx responses use bounded exponential backoff. An exhausted/failed run is marked incomplete
and does not present partial records as completed matches.

## Human labels and quality

Create a stable held-out annotation sheet without auto-filled labels:

```powershell
python scripts/export_labels_template.py --query configs/queries/product_quality.yaml `
  --output data/labels/product_quality_eval.csv --sample-size 200 --seed 8801
```

Annotators put `1` or `0` in `match`. Keep this held-out set fixed and do not use it to edit prompts
or tune thresholds. Evaluate during a run:

```powershell
semanticflow --provider jev --query configs/queries/product_quality.yaml `
  --labels data/labels/product_quality_eval.csv
```

The report includes TP, FP, FN, TN, precision, recall, and F1 over valid labeled rows only. Without
labels, quality is `pending`, never inferred from ratings or model output.

## Results and cost accounting

`decisions.jsonl` contains only completed predicate decisions, including probability, threshold,
cache status, model, usage, latency, attempt count, timestamp, and a question hash. The manifest
captures configuration, dataset SHA-256, Git commit, requested/resolved model, concurrency,
caching, and completion. Token use is measured from provider responses. `estimated_cost_usd` is
`null` until both verified input/output prices are configured in `configs/settings.yaml`; unknown
cost is not zero. Report schema 1.1 distinguishes requested evaluations, completed decisions,
budget blocks, HTTP attempts, provider errors, and actual cache lookups; see the
[reporting schema](docs/reporting_schema.md).

## Troubleshooting

- Missing dataset: run download, preparation, and validation commands above.
- Jev 401/403: verify `TYPESAFE_API_KEY`; authorization errors are intentionally not retried.
- Incomplete run: inspect `report.json` errors and budget/attempt metrics.
- No matches: low-selectivity conjunctions are valid; do not weaken predicates to manufacture hits.
- Existing output: run directories are unique; preparation/label scripts require `--overwrite`.

## Roadmap

Milestone 2 supplies the first cost-based ordering baseline. Milestone 3 adds correlation-aware
estimation, confidence-aware selection, profiling economics, and offline adaptive-profiling
simulation. Still open:

- live sequential profiling;
- live evaluation of the new workloads;
- predicate fusion;
- quality constraints from human labels;
- adaptive execution.

Background systems are summarized in [research notes](docs/research_notes.md).
