# Milestone 3 live evaluation (5 November 2026)

This is a live Jev evaluation, not an offline replay. Fixed and comparator orders ran in
one ABBA session per workload, cache off, model `jev-1.14.0`, threshold 0.5, 200/800 split
with seed 2027. No historical trace was reused as a held-out execution.

Session manifests record `git_commit` `213c3a5`. The worktree was dirty only because
`.gitignore` gained exceptions for the new `jev_*` directories during the first session.

**Totals.** 19,407 HTTP attempts, estimated **$0.263921**. Zero retries. Every profile and
session completed.

| Job | HTTP attempts | Estimated cost | Notes |
|---|---:|---:|---|
| `product_quality` ABBA session (existing profile) | 3,817 | $0.051058 | 4 runs |
| `listing_mismatch` profile | 600 | $0.008095 | 200 reviews × 3 |
| `listing_mismatch` ABBA session | 3,331 | $0.045879 | 4 runs |
| `positive_experience` profile | 600 | $0.007977 | 200 reviews × 3 |
| `positive_experience` ABBA session | 4,490 | $0.061212 | 4 runs |
| `mixed_feedback` profile | 600 | $0.008053 | 200 reviews × 3 |
| `mixed_feedback` session (ABBA + independence order) | 5,969 | $0.081648 | 5 runs |
| **Total** | **19,407** | **$0.263921** | |

## Design

Each session used `orders=selected` and one ABBA block of the query-file order versus the
correlation-aware selection. If those two orders were the same, the comparator was the
independence selection. Remaining distinct strategy orders ran after the block.
That keeps the primary pair close in time and still executes every order a strategy actually
selects.

Artifacts:

- `results/milestone_3/jev_product_quality_benchmark/`
- `results/milestone_3/jev_listing_mismatch/`
- `results/milestone_3/jev_positive_experience/`
- `results/milestone_3/jev_mixed_feedback/`
- pooled file `results/milestone_3/strategy_comparison.json`

## Per-workload execution savings

Savings are ABBA means of two contemporaneous runs of each order, except `mixed_feedback`
independence, which has one coverage run. Percents are versus the fixed-order mean.

| Workload | Fixed order mean | Selected order | Execution saving | P(cheaper than fixed) | Break-even vs full profile |
|---|---:|---|---:|---:|---:|
| `product_quality` | $0.012787 | `pd -> pq -> d` (both estimators) | 0.36% ($4.56e-5) | 0.65 | 174 |
| `listing_mismatch` | $0.011791 | `value_complaint` first (both, including confidence-aware) | 5.44% ($6.42e-4) | 1.00 | 13 |
| `positive_experience` | $0.015362 | independence: `easy_to_use -> would_recommend -> meets_expectations`; correlation-aware kept original | 0.77% for independence; 0% for correlation-aware | 0.80 / 0.00 | 68 / never |
| `mixed_feedback` | $0.022321 | correlation-aware `size_or_quantity_issue -> praises_product -> reports_problem`; independence swapped the last two | 45.02% / 44.17% | 1.00 / 1.00 | 1 / 1 |

Match counts moved by a few IDs between repeats, as in Milestone 2. That is response
variability, not proof that AND-order changes the match set.

## Strategy comparison across the four workloads

Pooled review-level bootstrap, 2,000 iterations. Cross-workload sign tests have four
workloads and little power.

| Strategy | Selection accuracy (observed cheapest) | Mean execution saving vs fixed | Workloads cheaper than fixed | Total profiling cost charged |
|---|---:|---:|---:|---:|
| Fixed | 0 / 4 | 0% | 0 | $0 |
| Independence | 3 / 4 | 12.68% | 4 / 4 | $0.032034 |
| Correlation-aware | 3 / 4 | 12.71% | 3 / 4 | $0.032034 |
| Correlation-aware + confidence gate | 2 / 4 | 12.62% | 2 / 4 | $0.032034 |

Independence missed `mixed_feedback`'s cheapest order (rank 2, 0.85% more expensive than
correlation-aware). Correlation-aware missed `positive_experience` (kept the original). They
tied on the other two. P(independence cheaper than correlation-aware in the pooled bootstrap)
is 0.31; the four-workload sign test is 1–1 with two ties.

The 12.7% mean saving is almost entirely `mixed_feedback`. Without it, independence still
beats fixed on every remaining workload, but the average drops to about 2%.

Adaptive profiling, charged only for the checkpoint it would have used, selected the same
orders as correlation-aware and spent $0.00398–$0.0110 of profiling depending on horizon
instead of $0.032. That is an offline costing of a live-selected order, not a live sequential
profile.

## Cardinality error on executed orders

| Estimator | Stage MAE (18 units) | Cost MAPE (9 executed orders) | Pairwise ranking concordance |
|---|---:|---:|---:|
| Correlation-aware | 9.26 | 2.21% | 5 / 6 |
| Independence | 22.22 | 4.28% | 5 / 6 |

On 18 (workload, order, stage ≥ 2) units, correlation-aware had the smaller absolute error
14 times, independence 4 times (sign-test p = 0.031). Those units share predicates, so the
p-value is descriptive. On `positive_experience` independence ranked the two executed orders
correctly and correlation-aware reversed them, which is why it kept the original.

## What this changes relative to the offline report

- The ~6-call `product_quality` gap survives a contemporaneous ABBA pair (958/955 then
  949/955 decisions). It is still a fraction of a percent in cost, and P(cheaper) = 0.65, so
  it is not a reliable saving.
- Different predicate correlations do produce larger opportunities. `mixed_feedback` saved
  about 45% of execution cost; `listing_mismatch` saved 5.4% with P = 1.00 in this session.
- Correlation-aware cardinality is more accurate on these live orders, but it is not a
  uniformly better planner than independence. Each estimator won one of the two workloads
  where they disagreed.
- A fresh full profile still needs substantial reuse except on `mixed_feedback`, where
  break-even is one execution.

## Limitations

- Four workloads, three predicates each, one dataset.
- Only strategy-selected orders were executed, not all six permutations, except the extra
  independence order on `mixed_feedback`.
- One ABBA block per workload. Time drift inside a session was small relative to
  `mixed_feedback` and `listing_mismatch` savings, and comparable to the
  `product_quality` gap.
- Adaptive and economics-gated numbers reuse the live order costs with offline profiling
  charges; they are not live sequential-profiling experiments.
- Match-ID differences are not treated as quality ground truth.
