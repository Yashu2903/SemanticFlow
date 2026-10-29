# Milestone 1 experiment protocol

## Data and splits

Acquire the prepared SemCEB artifact with `download_dataset.py`; preserve its provenance sidecar
and checksum. Produce samples with `prepare_sample.py`, storing
the seed, preparation version, and resulting SHA-256. Stable source IDs are preferred; content
hashes are the fallback. Establish separate profiling/development, future calibration, and
held-out evaluation samples. The recommended initial sample seed is 2026. A held-out label export
uses a separately recorded seed (the README example uses 8801). Never inspect final labels to tune
instructions or thresholds.

Run `validate_dataset.py` before experiments. Acceptance requires exactly 1,000 rows, non-empty
text, unique IDs, DuckDB readability, required columns, and an independently reproduced sample
with the same content fingerprint. Preserve `dataset_manifest.json` beside the prepared sample.

## Configuration and repeated runs

Commit workload YAML and settings before a run. Record the Git commit, query/config IDs, exact
predicate order/instructions, threshold, limit, maximum submissions, provider, requested model,
resolved model, timeout/retries/concurrency, caching state, and pricing ID. Keep caching off for the
uncached reference baseline. Repeat live nondeterministic runs into new directories; never replace
old artifacts. Saved decisions are the durable evidence even when a provider alias later changes.

## Measurements

Report initial/final records; stage inputs, passes, rejects, and selectivity; logical evaluations;
provider calls; actual HTTP attempts; successful/failed attempts; retries; cache hits/misses; input,
output, and total tokens; wall time; summed uncached inference latency; and completion status. Wall
time and summed latency are different under concurrency. Calculate cost only from verified rates
for the exact run configuration; otherwise store JSON `null`.

## Quality

Human annotators assign complete-query labels of 1 or 0. Blank labels remain excluded. Validate
IDs, duplicates, and values, and evaluate the same held-out IDs for every compared method. Report
the labeled count and confusion matrix plus precision, recall, and F1. Mock decisions and unlabeled
records are not ground truth.

## Reproducibility checklist

- Preserve all four run artifacts and source query/settings.
- Retain dataset fingerprint, sample seed, preparation version, and upstream provenance.
- Retain question hashes and requested/resolved model identifiers.
- Record failures and incomplete status rather than deleting failed runs.
- Confirm `pytest -q` and `ruff check .` offline before experiments.
- Do not mix cache comparisons with future predicate-order comparisons.
