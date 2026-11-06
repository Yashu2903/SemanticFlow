# Milestone 2 experiment protocol

## Objective and safeguards

The experiment asks whether a small unconditional profile can select a cheaper sequential order
for an equivalent AND query. Unit, mock, and offline real-data validation precede any paid run.
Milestone 1 artifacts are immutable. A Jev run requires new, phase-specific authorization and must
never be inferred from earlier approval.

## Dataset and held-out split

Use `data/processed/reviews_1000.parquet`, verify its SHA-256 fingerprint and 1,000 unique nonempty
reviews, and retain original IDs. With the configured seed, deterministically select 200 profiling
and 800 evaluation IDs. Save both ID lists and fingerprints. Do not resample based on model output,
Parquet order, ratings, or labels.

## Profiling

On the 200 profiling reviews, evaluate every predicate independently: three predicates require 600
completed decisions before retries. Do not use early rejection. Save raw decisions and compute
unconditional selectivity, Wilson intervals, mean token usage, mean inference latency, and mean
cost when pricing is verified. Enforce the separate profiling attempt cap, including retries. If
any required statistic is missing, stop before planning and retain partial artifacts.

## Planning

Validate query, model, threshold, prompt hashes, dataset/profile fingerprints, and statistics
version. Enumerate all candidate orders locally, calculate expected cost using the independent
AND model, rank by the configured objective, and save every estimate. Tie-breaking must preserve
the original order when effectively equal. Evaluation-set outcomes are unavailable at this point.

## Held-out execution

Run the original and selected orders over the exact same 800 held-out IDs with identical model,
prompts, threshold, concurrency, retry policy, cache state, and per-run request cap. Each predicate
remains a separate provider request. The existing executor performs early rejection. Do not
automatically repeat an interrupted paid run.

For deterministic mocks or recorded-decision replay, final review-ID sets must be identical. For
separate live stochastic calls, compare sets and investigate every difference rather than assuming
equivalence.

## Measurements

For profiling and both executions, retain completed decisions, attempts, retries, failures, token
usage, summed inference latency, wall time, estimated cost, and completion status. Execution-only
reductions are:

```text
reduction = (fixed - optimized) / fixed * 100
```

Zero denominators are handled explicitly. Wall-time reduction is reported only for fresh runs made
under comparable conditions; reconstructed traces cannot supply wall time. Prediction error
includes absolute and relative error for the original and selected orders.

Report profiling overhead independently. Net objective savings equal execution savings minus the
profile objective. When per-execution savings are positive, the break-even reuse count is the
ceiling of profiling overhead divided by per-execution savings. Never claim positive net savings
when the result is zero or negative.

## Reproducibility and artifacts

The experiment manifest includes the dataset and split fingerprints, exact IDs, seed, query and
prompt hashes, threshold, model, budgets, cache and concurrency settings, statistics/cost-model/
optimizer versions, objective, orders, timestamps, Git commit, and dirty state. Raw profiling and
normal execution decisions allow independent accounting checks.

Validate that both runs are complete, each accounts for all 800 reviews, decision files match
reported counts, request attempts remain under caps, tokens reconcile, and evaluation IDs match.
Then run `pytest -q`, `ruff check .`, and `git diff --check`.

## Live authorization checkpoint

Before Jev profiling, display the dataset, 600 required profiling decisions, profiling HTTP-attempt
cap, model, verified price source/date, token/cost estimate, and exact command. Obtain explicit
approval. Fixed and optimized evaluation phases each have separately identifiable caps; state
whether authorization covers one or both. If credentials, version, fingerprint, or pricing differ
from preflight, stop. Offline mock completion is not live-run authorization.

The Stage 2 Jev configuration pins `jev-1.14.0`, uses the public list price recorded in
`configs/settings.milestone_2_jev.yaml`, and writes staged artifacts under `profiling/`. Planning
then writes `planning/candidate_plans.json`, `planning/selected_plan.json`, and
`planning/report.json` without making provider requests. The evaluation directories must not be
created until the separately authorized held-out executions actually occur.

## Threats to validity

- Unconditional estimates ignore predicate correlation and may rank orders incorrectly.
- A 200-review profile has sampling uncertainty; Wilson intervals are diagnostic only.
- Provider stochasticity or version drift can change decisions and usage between runs.
- Concurrency, network conditions, rate limits, and run order confound wall-time comparisons.
- Profiling may cost more than it saves for a one-off query.
- Synthetic mock decisions and usage validate infrastructure, not semantic quality or Jev price.
- Product-quality already begins with a highly selective predicate, so a no-reorder result is valid.

This protocol is an experimental cost-based baseline informed by research context such as Sema,
Larch, and Abacus; it does not claim to reproduce their full algorithms.
