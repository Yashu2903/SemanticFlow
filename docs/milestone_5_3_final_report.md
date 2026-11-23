# Milestone 5.3 — Efficient Semantic Evidence Acquisition

## Result

M5.3 completed as an offline historical replay. No provider was initialized and no new Jev requests were made. The primary evaluation uses 1,200 fully paired M5.1 Stage D evaluation records (400 per exact Pilot A workload). M5.2.3 discovery records with missing follow-ups remain unlabelled and are excluded from policy scoring.

## Equal 100-request end-to-end comparison

The table pools three distinct workload strata for an exploratory aggregate; workload-specific files remain authoritative.

| Policy | Records confirmed | Predicate mismatches | Query mismatches | Positives | Screens | Mean cost |
|---|---:|---:|---:|---:|---:|---:|
| random | 27.16 | 0.240 | 0.040 | 0.800 | 0.00 | $0.001404 |
| uncertainty_continuous | 12.48 | 0.220 | 0.100 | 0.800 | 65.24 | $0.001386 |
| positive_oriented | 12.62 | 0.260 | 0.180 | 1.920 | 65.66 | $0.001387 |
| hybrid_weighted | 12.68 | 0.320 | 0.200 | 1.680 | 65.98 | $0.001384 |

## Findings

- **H1 — inconclusive.** Continuous uncertainty found 0.22 predicate mismatches on average at 100 end-to-end requests versus 0.24 for random. Its paired difference crossed zero. Conditional uncertainty did concentrate three predicate mismatches in 37 confirmations, but none was a complete-query disagreement.
- **H2 — supported on this fixed replay, not population-confirmed.** Positive-oriented sampling found 1.92 positives on average versus 0.80 for random. The paired seed difference was 1.12 (95% ordering-bootstrap interval 0.64 to 1.60).
- **H3 — no universal hybrid win.** The weighted hybrid found 1.68 positives and 0.20 query disagreements, versus 0.80 and 0.04 for random. Its predicate-mismatch difference was 0.08 with interval -0.12 to 0.28, which crosses zero. Component ablations sometimes outperformed the full hybrid, so the fixed weighting is not justified as optimal.
- **H4 — screening is economically material.** At 100 requests, the hybrid spent 65.98 requests and $0.000900 on screens, leaving 12.68 confirmed records; direct random confirmed 27.16. Screening still improved positive and query-disagreement yield on the pooled fixed replay, but not predicate-mismatch yield.
- Separate Jev is a comparison reference, not human truth. These results measure execution disagreement and reference-positive collection, not semantic accuracy.

## Workload heterogeneity

- Banking77: hybrid found 0.06 query disagreements and 0.08 positives, versus random 0.08 and 0.18.
- All Beauty: hybrid found 0.44 query disagreements and 3.64 positives, versus random 0.16 and 1.80.
- SMS Spam: neither policy found a complete-query disagreement; hybrid found 1.56 positives versus random 0.80.

The pooled apparent advantage is driven mainly by Beauty and SMS positive enrichment; Banking does not reproduce it. Policy choice must therefore remain workload-aware.

## Ablations

Uncertainty margins were not monotonic: at 100 end-to-end requests, margins 0.05, 0.10, 0.15, and 0.20 found mean predicate-mismatch counts 0.05, 0.05, 0.16, and 0.00. Removing uncertainty increased positive yield from 1.68 to 2.35; removing cost awareness increased query-disagreement yield from 0.20 to 0.30 in the 20-seed ablation. These sparse differences do not support the full hybrid formulation.

## Statistical limits

Sparse-event Wilson intervals and paired seed-bootstrap intervals are reported in the machine-readable analysis. Seed variation measures policy ordering variation over this fixed historical pool; it does not create new independent provider observations. The Stage D data has already been inspected in earlier milestones, so this is exploratory historical evaluation, not a new confirmatory holdout.

## M5.4 readiness

The infrastructure and cost-accounted replay are ready to inform M5.4 feature design, but the disagreement evidence is not sufficient to select a universal production acquisition policy. A useful next evidence increment is an independently sampled, fully paired set stratified toward near-threshold screens while retaining a representative stratum. A 300-record design would require about 1,100 physical requests (300 screens plus 800 remaining separate/batch requests for the current workload mix) and approximately $0.0155 at the historical mean request cost. This proposal is not authorized or executed here.

## Status

`M53_EXPERIMENT_COMPLETE`
