# SemanticFlow

SemanticFlow is an academic research harness for measuring the cost, latency, and quality of
natural-language predicates over tabular data. Milestone 1 is deliberately a reference engine: it
runs predicates in configured order as an AND filter and records what actually happened. It does
not reorder, fuse, learn, route, or adapt queries.

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

The executed Milestone 1 real-data mock run (`product_quality`, seed 2026) loaded 1,000 reviews and
produced stage counts `1000 -> 476 -> 234 -> 117`, 1,710 decisions/provider calls, and a complete
four-artifact result directory. These deterministic mock outcomes validate plumbing, not semantic
quality.

Baseline caching is off by default. When enabled, it is memory-only and scoped to one run. Its key
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
is retained with `completion_status: incomplete`, empty completed matches, and diagnostic traces.

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

`decisions.jsonl` contains only predicates actually evaluated, including probability, threshold,
cache status, model, usage, latency, attempt count, timestamp, and a question hash. The manifest
captures configuration, dataset SHA-256, Git commit, requested/resolved model, concurrency,
caching, and completion. Token use is measured from provider responses. `estimated_cost_usd` is
`null` until both verified input/output prices are configured in `configs/settings.yaml`; unknown
cost is not zero.

## Troubleshooting

- Missing dataset: run download, preparation, and validation commands above.
- Jev 401/403: verify `TYPESAFE_API_KEY`; authorization errors are intentionally not retried.
- Incomplete run: inspect `report.json` errors and budget/attempt metrics.
- No matches: low-selectivity conjunctions are valid; do not weaken predicates to manufacture hits.
- Existing output: run directories are unique; preparation/label scripts require `--overwrite`.

## Roadmap

Milestone 1 supplies measured stage selectivity, latency, usage, cache behavior, and record-level
decisions. Later research may consume those interfaces for cost-based ordering, fusion,
conditional selectivity, quality constraints, uncertainty, and adaptive execution. None of those
optimizations are implemented here. Background systems are summarized in
[research notes](docs/research_notes.md).
