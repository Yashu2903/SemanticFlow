# Milestone 4.6 independent-generalization protocol

Status: preregistered offline on 2026-11-12. No paid Jev requests have been made.

## Research question and freeze

This study asks whether SemanticFlow's frozen adaptive optimizer reduces total semantic-query
cost on previously unseen datasets and workloads, and whether correlation-aware estimation
improves decisions under distribution shift. The study may measure existing methods only. It
does not authorize predicate fusion, a new optimization algorithm, Milestone 5 work, or tuning
after any new outcome is observed.

The machine-readable protocol is
`configs/milestone_4_6/protocol.yaml`; exact predicate text and workload order are in
`configs/milestone_4_6/workloads.yaml`. The workload freeze records SHA-256 prompt hashes,
partition fingerprints, and workload fingerprints. The strategy freeze records every source
and configuration hash for `fixed`, `independent`, `correlation_aware`,
`existing_adaptive`, `confidence_aware`, `risk_aware`, `joint_adaptive_risk`, and
`milestone_4_5_frozen`.

The frozen Milestone 4.5 parameter fingerprint is
`c00fadea786bd2075739576501460f51f0997172caccb5e133a0d2f0db3447c7`; it is reproduced
from the configuration parameters. The full frozen YAML SHA-256 is
`9470bdde8d8cd6b70e881a2ffc9a8fc5d1bf84f166301f9f6edad6e493f145e8`.

## Independent datasets and partitions

Two real public corpora were selected before semantic outcomes were obtained.

1. Amazon Reviews'23 `All_Beauty` is a previously unused Amazon category with 700,808 valid
   source reviews. The source is the McAuley Lab public research release. The prepared snapshot
   contains 200 profiling and 800 evaluation reviews. Product IDs are primary split groups and
   author IDs are secondary split groups.
2. Banking77 contains 13,083 real online-banking support queries across 77 intents. The pinned
   PolyAI source commit is `9d081458ff52e53cf7e848f414e6e9344e4e6696`, and the dataset is
   distributed under CC BY 4.0. The prepared snapshot contains 200 profiling and 800 evaluation
   queries. Source row identity is retained while exact and near duplicates are excluded.

The alternative CFPB complaint corpus was investigated but rejected: the tested current export
did not contain the public narrative field needed for semantic predicates. It was not silently
replaced with fabricated text.

Preparation normalizes text, assigns content-derived stable IDs, removes malformed records,
detects exact duplicates and 64-bit SimHash near duplicates, ranks records deterministically,
performs group-aware splitting, and writes a non-overwriting Parquet snapshot plus manifest.
Both partitions are checked against the original 1,000 SemCEB review texts. Exact IDs, primary
groups, secondary groups, exact text, and near-identical text cannot cross the partition
boundary.

| Dataset | Source valid rows | Prepared split | Dataset fingerprint |
|---|---:|---:|---|
| Amazon All Beauty | 700,808 | 200 / 800 | `dcb6df57fb239160a111c9d34b2f0e02b9c252eadc071e1dc1b259f721b189a1` |
| Banking77 | 13,083 | 200 / 800 | `2afaeb6717ba4fe205d9c6ac90c52e3349596af58e3874f382c2b4c8a56a849f` |

## Preregistered workloads

The ten conjunctions below were written before any live profiling or held-out inference. The
listed characteristics are hypotheses, not measured facts.

| Workload | Predicates in original order | Intended test |
|---|---|---|
| `beauty_failure_usability_return` | malfunction, hard to use, return/refund | strong correlation; potentially unfavorable reorder |
| `beauty_packaging_value` | packaging failure, poor value | weak correlation; near tie |
| `beauty_reaction_refund_packaging` | adverse reaction, return/refund, packaging failure | rare event; plausible original advantage |
| `beauty_usability_packaging_return` | hard to use, packaging failure, return/refund | weak correlation; near tie |
| `beauty_reaction_failure_value_return` | adverse reaction, malfunction, poor value, return/refund | rare event; four predicates; real prompt-cost differences |
| `banking_unauthorized_card_refund` | unauthorized activity, card related, refund/reversal | strong correlation |
| `banking_withdrawal_access` | cash withdrawal, access/verification | weak correlation; rare event; near tie |
| `banking_unauthorized_withdrawal_refund` | unauthorized activity, cash withdrawal, refund/reversal | rare conjunction; potentially unfavorable reorder |
| `banking_card_access` | card related, access/verification | positive correlation; near tie |
| `banking_transfer_unauthorized_refund_access` | transfer, unauthorized activity, refund/reversal, access/verification | four predicates; correlation and cost differences |

All use conjunction, threshold 0.5, the exact stored instructions, and two to four predicates.
Candidate generation must yield 2, 6, or 24 permutations as appropriate.

## Hypotheses and estimands

The primary hypothesis is that adaptive profiling improves cold-start net cost relative to the
original fixed order at a sufficiently large reuse horizon. The primary estimand is the
cost-weighted aggregate cold-start net saving at each preregistered horizon: 1, 5, 10, 25, 50,
100, 250, and 1,000 executions.

Secondary hypotheses are:

- H2: correlation-aware estimation lowers stage-cardinality and cost-prediction error;
- H3: correlation-aware estimation lowers regret and improves ranking concordance;
- H4: risk-aware selection accepts fewer harmful reorderings, possibly rejecting benefits;
- H5: the frozen Milestone 4.5 joint policy is economically competitive with existing adaptive;
- H6: 50%, 80%, 90%, and 95% bootstrap interval coverage can be measured without recalibration.

Reported estimands include workload wins, losses, ties, mean and median saving, maximum downside,
cost-weighted totals, dataset aggregates, leave-one-workload-out sensitivity, profiling cost,
break-even reuse count, selection regret against executed orders only, cheapest-executed-order
frequency, cardinality MAE/MAPE/bias, cost error, ranking concordance, interval coverage, and risk
policy error counts. A single large saving cannot establish success: the median may not be
negative, no workload may supply more than 75% of positive aggregate saving, and direction must
survive leave-one-workload-out analysis.

Paired workload bootstrap intervals, an exact sign test over non-ties, and a dataset/workload
cluster bootstrap are preregistered. Records are paired within workloads. Individual predicate
requests are not treated as independent experiments.

## Profiling and plan selection

Stage B profiles each of six unique predicates once on all 200 profiling records in its dataset.
This produces two shared 200-by-6 matrices, avoiding redundant paid profiling across workloads.
Each request records the individual decision, resolved model, input/output tokens, latency,
attempt count, failure/retry accounting, prompt hash, and record ID.

Every strategy selects its plan from profiling evidence only at the preregistered reuse horizon
R=10. The plan artifact contains the selected order, action, profiling decisions and cost,
independence and correlation-aware candidate predictions, stage-cardinality predictions,
matrix fingerprints, and source/configuration freeze. The artifact is written before Stage C.
An incomplete profile cannot produce a plan.

## Held-out evaluation and replication

Stage C is separately authorized only after real plans are frozen. Within each workload it
deduplicates identical selected orders. The original order and the primary differing comparator
use an ABBA block; every other unique selected order is executed once. Every run uses the same
800 evaluation records, threshold, prompt text, pinned model, concurrency, timeout, retry policy,
and disabled cache. Each run has its own retry-inclusive cap, and the session has a global cap.

The analyzer averages true ABBA repetitions but labels reuse-horizon economics as simulated. It
compares final matching-record ID sets, not merely counts. Match agreement does not establish
semantic accuracy or recall without human labels. The cheapest plan means the cheapest executed
order unless every permutation was measured.

Stage D repetitions require a third authorization and a new budget. No silent replication or
increase in run count is allowed.

## Cost accounting

For reuse horizon R:

`NetSaving(R) = R * (C_fixed - C_selected) - C_profiling - C_optimization`.

Cold start charges new profiling; warm start treats valid historical profiling as sunk. Provider
token charges, local optimization cost, summed inference latency, and wall time are separate.
The preregistration token estimate is a character-based proxy and can never replace missing
provider-reported cost.

Stage B requests authorization for exactly 2,400 logical decisions and at most 7,200 HTTP
attempts (two retries). The proxy is 241,374 input and 2,400 output tokens: $0.010137708 at the
frozen rate, or $0.030413124 if every attempt consumes the full proxy. Model: `jev-1.14.0`.

Exact Stage B command after separate approval:

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv run python experiments/milestone_4_6/run_generalization.py profile `
  --output results/milestone_4_6/live_profile_<approved-run-id> `
  --authorization "AUTHORIZE M4.6 STAGE B JEV PROFILING"
```

This document does not grant that authorization. Stage C remains unbudgeted until the real plans
are frozen; the mock schedule is only a non-authoritative infrastructure upper-bound example.

## Stopping and incomplete-run rules

- Stop on any provider error, incomplete dataset profile, or request-cap exhaustion.
- Preserve partial diagnostics but exclude partial match sets from performance claims.
- Do not select from an incomplete matrix or substitute proxy cost for absent live cost.
- Do not alter prompts, thresholds, policies, intervals, or success criteria after outcomes.
- Do not claim a global optimum from only the selected-order comparison.
- A model-version mismatch or malformed response stops the stage.

## Reproduction interfaces

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv run python scripts/prepare_m46_datasets.py --skip-download
uv run python experiments/milestone_4_6/run_generalization.py preflight
uv run pytest -q
uv run ruff check src tests configs
git diff --check
```

After Stage B, `plan` freezes real selections and creates the exact Stage C budget. After a
separate Stage C approval, `heldout` executes the frozen schedule. `analyze` creates the structured
generalization, uncertainty, risk, consistency, and economics report.
