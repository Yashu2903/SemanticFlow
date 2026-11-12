# Milestone 4 research report

**Uncertainty-aware plan selection for semantic query optimization**

Follow-up: the joint controller's VOI calibration, sensitivity analysis, and finite-scenario
lower-bound correction are documented in `docs/milestone_4_voi_validation.md`. The original
canonical artifacts and conclusions below are preserved unchanged.

Exploratory retrospective evaluation. No new Jev requests. Bootstrap frequencies are
resampling diagnostics over the 200-review profiling split, not calibrated production
probabilities.

Canonical artifacts: `results/milestone_4/offline_benchmark_20261106/`
(`schema_version: m4-offline-benchmark-v1`, 2,000 paired bootstrap iterations, seed 3031).

## A. Research question and motivation

Milestone 3 showed that better cardinality estimates do not necessarily produce better plans.

On four live SemCEB / Jev 1.14.0 workloads, correlation-aware estimation reduced stage-input
MAE from 22.22 to 9.26 records, yet independence and correlation-aware each selected the
cheaper executed order on one of the two workloads where they disagreed. Several candidate
pairs differed by less than 1% in execution cost. Profiling sometimes cost more than the
saving from a single optimized execution. A confidence-aware policy already avoided the
uncertain `product_quality` reorder while keeping the large `mixed_feedback` gain.

The motivating example is `mixed_feedback`. Unconditional selectivities
(`reports_problem` 29.5%, `praises_product` 87%) reverse after `size_or_quantity_issue`
passes (`P(reports_problem | size_issue) ≈ 91.7%`). Correlation-aware planning used that
conditional structure and selected a cheaper order than independence. In contrast,
`positive_experience` had more accurate correlation-aware cost predictions and a worse
ranking: the two remaining predicates were almost perfectly dependent after `easy_to_use`
passed, so prefix cardinality was right while the cheaper swap was a near-tie.

**Question.** Can uncertainty-aware plan selection improve reliability by considering
estimation uncertainty, expected regret, model disagreement, and profiling cost?

This report does not assume that the new policy outperforms existing baselines. The
four-workload comparison is labeled exploratory: the same traces already informed Milestone 3
development, so they are not an untouched confirmatory test.

## B. Method

### Uncertainty model

Let \(P_1,\ldots,P_k\) be the modeled permutations of the query-file order \(P_{\text{original}}\).
A paired bootstrap of \(B=2000\) draws resamples complete profiling reviews with replacement.
Predicate columns are never resampled independently. Token and latency observations travel
with the row.

On draw \(b\), every candidate is re-estimated from the resampled matrix, yielding
\(\hat C_b(P)\). Reported summaries are the mean, median, sample standard deviation, 5th and
95th percentiles, and a percentile interval at the configured level (default 95%). Expected
saving versus the original order is

\[
\hat S(P)=\frac{1}{B}\sum_b \bigl(\hat C_b(P_{\text{original}})-\hat C_b(P)\bigr).
\]

\(P_{\text{bootstrap}}(P\text{ cheaper})=B^{-1}|\{b:\hat C_b(P)<\hat C_b(P_{\text{original}})\}|\)
is a resampling diagnostic. It is not \(P(\text{production cost}(P)<\text{production cost}(P_{\text{original}}))\).

Four uncertainty sources are distinguished and are not treated as independent additive noise:

1. Sampling across reviews (the bootstrap).
2. Statistical uncertainty of selectivity estimates (each resample refits the estimator).
3. Variability in model responses (sensitivity analysis on repeated live traces only).
4. Uncertainty in future per-request cost and latency (usage resampled with the row).

Repeated Jev traces exist for these workloads, but threshold crossings are few (1, 6, 7, and
14 recorded crossings). No response-flip distribution is fitted from that evidence.

### Expected regret and downside risk

Regret of \(P\) relative to the modeled set on draw \(b\):

\[
\operatorname{Regret}_b(P)=\hat C_b(P)-\min_j \hat C_b(P_j).
\]

This is not regret against the unknown true optimum or against unexecuted permutations.

Downside loss versus the original order:

\[
L_b(P)=\hat C_b(P)-\hat C_b(P_{\text{original}}).
\]

CVaR at level \(\alpha=0.95\) is the mean of \(L\) values at or above the \(\alpha\) quantile
(upper tail). Hard CVaR and expected-regret gates default to disabled (`null`) so that those
metrics are reported without becoming workload-specific thresholds.

### Risk-aware policy

`risk_aware` (primary estimator: correlation-aware) adopts a reorder only when all of the
following hold:

- expected saving \(\ge\) `minimum_expected_saving` (default 0);
- mean relative saving \(\ge\) 1%;
- \(P_{\text{bootstrap}}(\text{cheaper}) \ge 0.95\);
- optional CVaR / expected-regret caps, if configured;
- optional forward-looking economics, if `apply_profiling_economics` is true.

Otherwise the original order is retained, with a machine-readable
`Decision: KEEP_ORIGINAL` / `Reason: ...` record.

`risk_aware_robust` adds an estimator-disagreement rule: if independence and correlation-aware
select different orders, keep the original unless both select some reorder (both agree the
original is worse). Agreement is not treated as proof of correctness. Weighting parameters,
if any, would be fit on profiling data only; this implementation does not blend estimators.

### Profiling economics

Net benefit for \(R\) executions is

\[
\text{NetBenefit}=R\bigl(C_{\text{original}}-C_{\text{candidate}}\bigr)
-C_{\text{new profiling}}-C_{\text{optimization}}.
\]

Sunk profiling is reported and not subtracted from forward-looking net benefit. Unknown reuse
defaults to \(R=1\). A structural upper bound is never presented as an expected saving. The
economic skip happens before new paid profiling (`statistics_state: none` / adaptive start
gate). Default `apply_profiling_economics` is false for plan selection on an already-collected
profile.

## C. Experimental methodology

| Item | Value |
|---|---|
| Dataset | SemCEB Arts, Crafts & Sewing 5-core sample, `reviews_1000.parquet` |
| Fingerprint | `sha256:21cd6032e72379828f334422e29365688c5b5063c07ca23f08f8f025c2d55a99` |
| Split | sha256(seed:review_id), seed 2027, 200 profiling / 800 held-out |
| Model | `jev-1.14.0`, threshold 0.5 |
| Pricing | $0.042 / M input tokens, $0 output |
| Live evidence | Milestone 3 ABBA sessions (19,407 HTTP attempts, $0.263921); not re-run |
| Plan selection | Profiling split only |
| Held-out outcomes | Retrospective scoring only |
| Candidate plans | All 6 permutations of each 3-predicate query |
| Observed costs | Mean live execution cost of an order, only if that order was executed |
| Unexecuted orders | Marked `observed_result: unavailable`; no estimate substituted |
| Reproducibility | Bootstrap seed 3031, 2,000 iterations, output directory refused if it exists |
| Label | `exploratory_retrospective_benchmark` |

Workloads: `product_quality`, `listing_mismatch`, `positive_experience`, `mixed_feedback`.
Independence, correlation-aware, and confidence-aware selections reproduced the stored
Milestone 3 `strategies.json` orders on every workload.

## D. Results

Observed execution-cost saving versus the contemporaneous fixed-order baseline. Only executed
orders contribute measurements. No strategy selected a plan that was more expensive than
fixed (`workloads_worse = 0` throughout).

| Strategy | product_quality | listing_mismatch | positive_experience | mixed_feedback | mean % | median % | keep original | selection accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0/4 |
| Independence | 0.36 | 5.44 | 0.77 | 44.17 | 12.68 | 3.10 | 0 | 3/4 |
| Correlation-aware | 0.36 | 5.44 | 0 | 45.02 | 12.71 | 2.90 | 1 | 3/4 |
| Confidence-aware | 0 | 5.44 | 0 | 45.02 | 12.62 | 2.72 | 2 | 2/4 |
| Risk-aware | 0 | 5.44 | 0 | 45.02 | 12.62 | 2.72 | 2 | 2/4 |

Selection accuracy is the share of workloads whose selected order was the cheapest *executed*
order, not the unknown true optimum. Unweighted means are dominated by `mixed_feedback`.
Leave-one-workload-out means for risk-aware percent savings: 16.82 excluding PQ or PE,
15.01 excluding listing_mismatch, and **1.81 excluding mixed_feedback**.

| Strategy | Total observed saving (USD) | Mean regret (USD) | Max regret (USD) | Profiling cost (USD) |
|---|---:|---:|---:|---:|
| Independence | 0.010665 | 4.74e-5 | 1.90e-4 | 0.032034 |
| Correlation-aware | 0.010737 | 2.94e-5 | 1.18e-4 | 0.032034 |
| Confidence-aware | 0.010691 | 4.08e-5 | 1.18e-4 | 0.032034 |
| Risk-aware | 0.010691 | 4.08e-5 | 1.18e-4 | 0.032034 |

Net execution saving minus full-profile cost, summed over the four workloads:

| Horizon R | Risk-aware net USD (sunk charged) | Forward net, sunk excluded |
|---:|---:|---:|
| 1 | −0.0213 | 0.0107 |
| 10 | 0.0749 | 0.1069 |
| 100 | 1.037 | 1.069 |
| 1,000 | 10.66 | 10.69 |

A single execution does not repay a new four-workload profile. The mixed_feedback saving alone
covers one comparable profile; listing_mismatch's 5.44% does not (break-even 13 executions).
Those are diagnostics, not the default decision: existing statistics are sunk.

### Critical test cases

**product_quality.** Both estimators select `product_defect → poor_quality → dissatisfaction`
(observed saving 0.36%, historical bootstrap \(P \approx 0.77\)). Risk-aware keeps the original.
Reasons: expected relative saving below 1% and bootstrap saving probability below 0.95. This
matches confidence-aware. The missed 0.36% is smaller than profiling cost at any modest reuse.

**listing_mismatch.** Reordered `value_complaint → description_mismatch → dissatisfaction`
saves 5.44%, \(P_{\text{bootstrap}}=0.9985\), relative saving 5.2%. Risk-aware adopts it.
CVaR of loss versus original is negative (the upper tail is still a saving).

**positive_experience.** Independence selects the cheaper executed order (0.77%).
Correlation-aware ranks the original cheapest, so ungated correlation-aware, confidence-aware,
and `risk_aware` all keep the original. `risk_aware_robust` keeps the original because the
estimators disagree on whether the original is worse. The policy was not tuned to force
independence's historical winner. The 0.77% gap is below the 1% materiality gate in any case.

**mixed_feedback.** Correlation-aware selects `size_or_quantity_issue → praises_product → reports_problem`
(45.02%, cheapest executed order). Independence selects the other size-first order (44.17%).
Both agree the original is much worse. Risk-aware and risk-aware-robust adopt the
correlation-aware order. \(P_{\text{bootstrap}}=1.0\), relative saving 45%.

Predicted-versus-observed saving error on adopted plans is on the order of \(10^{-5}\) USD
(listing_mismatch: predicted 0.000623 vs observed 0.000642). Plans that keep the original
report predicted saving 0 by construction; that is not a forecast of the missed reorder.

## E. Ablations

| Ablation | product_quality | listing_mismatch | positive_experience | mixed_feedback | mean % |
|---|---:|---:|---:|---:|---:|
| Independence, no gate | 0.36 | 5.44 | 0.77 | 44.17 | 12.68 |
| Correlation-aware, no gate | 0.36 | 5.44 | 0 | 45.02 | 12.71 |
| Correlation-aware + confidence gate | 0 | 5.44 | 0 | 45.02 | 12.62 |
| Risk-aware + correlation-aware | 0 | 5.44 | 0 | 45.02 | 12.62 |
| Risk-aware + both estimators | 0 | 5.44 | 0 | 45.02 | 12.62 |
| Risk-aware + economics on existing stats | 0 | 5.44 | 0 | 45.02 | 12.62 |
| Risk-aware + adaptive prefix (25 reviews) | 0 | **0** | 0 | 45.02 | 11.26 |

What actually changed decisions:

1. The existing confidence gate already rejected `product_quality` and kept `listing_mismatch`
   and `mixed_feedback`. Default `risk_aware` selected the **same four plans**.
2. Robust disagreement changed the *reason* on `positive_experience`, not the plan.
3. CVaR and expected regret were reported; with `maximum_cvar_loss: null` they did not gate.
4. Economics on existing statistics did not reverse any adopt: forward net benefit excludes
   sunk profiling, and listing/mixed already passed the risk gates.
5. Adaptive profiling stopped at 25 reviews on every workload. Applying `risk_aware` to that
   prefix **rejected listing_mismatch** (`bootstrap_saving_probability_below_minimum`) even
   though the adaptive controller had already selected the historically cheaper order. The
   25-row sample was too small for the 0.95 diagnostic. Mixed_feedback still passed. Adaptive
   profiling cost fell from $0.0320 to $0.0040, but observed mean saving fell to 11.26% and
   selection accuracy to 1/4.

On this exploratory set, the decision-relevant pieces are the 0.95 resampling probability and
the 1% materiality gate—the same ingredients as confidence-aware plus an explicit near-tie
rule that did not bind beyond what probability already rejected. Additional CVaR, regret, and
disagreement machinery did not change selected orders. Stacking risk-aware on a tiny adaptive
prefix made one previously useful reorder fail.

## F. Failure analysis

**Near-tied orders.** `product_quality` (0.36%) and `positive_experience` (0.77%) sit near or
below the 1% gate. Keeping the original is conservative and, on this set, matches the
confidence-aware policy. It also forgoes two measured savings that independence captured.

**Sampling uncertainty.** `product_quality`'s cheaper order wins well under 0.95 of 200-review
resamples. That is why both confidence-aware and risk-aware abstain. The frequency is a
profile diagnostic, not a claim about other review distributions.

**Model-response variability.** Repeated ABBA runs recorded 1–14 threshold crossings per
workload. Replicate cost standard deviations are much smaller than the mixed_feedback gap and
comparable to the product_quality gap. They are reported as sensitivity, not folded into the
bootstrap as independent noise.

**Estimator disagreement.** Independence and correlation-aware disagree on
`positive_experience` and on which mixed_feedback reorder is cheapest. Robust selection keeps
the original only in the first case, because in the second both estimators agree the original
is worse. Disagreement is a useful flag, not a correctness test: independence was right on
`positive_experience` and wrong on `mixed_feedback`.

**Profiling cost versus savings.** At \(R=1\), full profiling ($0.008 per workload) exceeds
listing_mismatch's $0.00064 execution saving and product_quality's $0.000046. Mixed_feedback's
$0.010 saving covers one profile. Default risk-aware does not charge sunk profiles against
reorders; the skip-profiling path exists for the case where statistics are still absent.

**Unobserved candidates.** Each workload executed two or three of six permutations. Risk-aware
selected only executed orders on this run. The implementation marks any other selection
`observed_result: unavailable` and does not substitute a replay estimate for a measurement.

## G. Limitations

- Only four live workloads, all 3-predicate conjunctions over one SemCEB category.
- Workloads share a dataset and likely share residual semantic correlation.
- Incomplete coverage of predicate permutations.
- Retrospective evaluation on traces that already informed Milestone 3.
- Uncalibrated bootstrap probabilities.
- Profiling \(n=200\) is small for prefix conditionals; the adaptive 25-row prefix is smaller.
- No independent human quality labels; cheaper is not known to be equally correct.
- Repeated analysis of the same benchmark risks overfitting policy defaults.
- Default CVaR/regret gates were left disabled; a different configuration could look better
  or worse, and fitting those gates on these four workloads would not be confirmatory.

## H. Future experiments

Do not execute these as part of Milestone 4.

1. An independent confirmatory set: new predicates, a different product category, and a
   pre-registered policy configuration.
2. Workloads with four or more predicates, where unexecuted permutations are the typical case.
3. A calibration study that compares \(P_{\text{bootstrap}}\) with realized sign of savings on
   many held-out workloads.
4. Live sequential profiling with the pre-profiling economics gate, not an offline prefix
   replay.
5. Repeated identical-order Jev runs designed to measure response variability, rather than
   opportunistic ABBA duplicates.
6. Human labels on a frozen held-out sheet, so cost-optimal orders can be checked for quality.

## I. Answers to the research questions

1. **Did uncertainty-aware optimization select better plans?** Not uniformly. On this
   exploratory set it selected exactly the same plans as the existing confidence-aware policy:
   abstain on `product_quality` and `positive_experience`, adopt listing_mismatch and
   mixed_feedback. Ungated correlation-aware had a slightly higher unweighted mean (12.71% vs
   12.62%) because it took the 0.36% product_quality saving.

2. **Did it reduce the probability of selecting a more expensive plan?** No selected plan was
   more expensive than the fixed order. Versus the cheapest executed order, risk-aware's mean
   regret ($4.08 \times 10^{-5}$) is worse than ungated correlation-aware ($2.94 \times 10^{-5}$)
   and better than independence ($4.74 \times 10^{-5}$). Four workloads cannot establish a
   probability.

3. **Did it improve expected or observed regret?** Observed mean regret matched
   confidence-aware and did not beat ungated correlation-aware. Expected bootstrap regret is a
   profile diagnostic; on adopted plans it was near zero.

4. **Did it retain the large savings on mixed_feedback?** Yes, 45.02%, including under the
   robust disagreement rule, because both estimators agree the original is worse.

5. **Did it appropriately reject low-confidence optimizations?** It rejected product_quality
   (0.36%, unstable bootstrap). It also rejected positive_experience's 0.77% independence
   winner, which *was* cheaper among executed orders. "Appropriate" depends on whether a
   sub-1% gap is worth acting on; the pre-declared 1% / 0.95 gates say no.

6. **Did additional complexity provide measurable value?** On these four workloads, no
   additional adopted or rejected plan relative to confidence-aware. CVaR, regret, and
   disagreement reports are useful diagnostics. The adaptive+risk-aware stack reduced
   profiling cost and lost the listing_mismatch saving. That is evidence against naive
   stacking, not for it.

7. **What cannot be established from four workloads?** Generalization to other datasets,
   models, predicate arities, or quality-constrained optimization; calibration of bootstrap
   frequencies; statistical significance of strategy differences; whether a different risk
   configuration would help; and whether the policy overfits this benchmark.

The success criterion for the software is met: SemanticFlow can adopt a reorder or keep the
original using estimated savings, regret, downside risk, and profiling economics, with
reproducible offline artifacts and no new Jev calls. The research outcome is that, on the
existing four-workload live evidence, the new policy did not outperform the simpler
confidence-aware gate.

## J. Joint adaptive-risk follow-up

The disconnected adaptive-stop/final-risk failure was addressed in the follow-up
[joint adaptive profiling and risk-aware plan selection report](milestone_4_joint_adaptive_risk_report.md).
The joint controller uses checkpoint-local sparse posterior-predictive uncertainty and an expected
value of additional profiling approximation before choosing `PROFILE_MORE`.

In the aligned 2,000-draw offline replay, it retains the existing R=1 economics, is worse at R=5,
and recovers `listing_mismatch` at 100 reviews for R>=10. At R=10 its four-workload cold-start net
is $0.096946 versus $0.096513 for the existing adaptive+risk stack. This is an exploratory
$0.000434 improvement, offset by unproductive extra profiling on `product_quality`; it does not
change the original Milestone 4 conclusion about four-workload generalization.
