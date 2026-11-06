# Milestone 3 research report: correlation-aware, profiling-aware ordering

Every number below comes from stored Jev traces (Milestone 1 fixed run, Milestone 2 profiling
and optimized runs, and the Milestone 1 smoke run) or from deterministic offline computation over
them. **No new Jev request was made for Milestone 3.** Replay, bootstrap, calibration, and
adaptive results are offline analyses, not new live evaluations. Replay-derived savings are
estimates, not causal measurements. The generated artifacts are in
`results/milestone_3/offline_analysis_20261105_corrected/`, reproducible with:

```powershell
semanticflow analyze --query configs/queries/product_quality.yaml `
  --optimizer-config configs/optimizer.milestone_3.yaml `
  --experiment-dir results/milestone_2/jev_product_quality_profile_20261102 `
  --historical-baseline-dir results/milestone_1_full_baseline/product_quality_20261102T163100.862806Z_d23f73f2cbd1 `
  --extra-decisions "m1_smoke_jev=results/product_quality_20261029T022635.415784Z_96caeebde714/decisions.jsonl" `
  --output-dir results/milestone_3/<new-directory>
```

Two runs produced byte-identical artifacts apart from the manifest timestamp. The first
Milestone 3 output, `results/milestone_3/offline_analysis_20261105/`, is preserved unchanged
but superseded; the corrections are listed below. Workload: `product_quality`
(`poor_quality AND product_defect AND dissatisfaction`), model `jev-1.14.0`, threshold 0.5,
200 profiling and 800 held-out reviews, list price $0.042 per million input tokens and $0 per
million output tokens.

### Corrections in this revision

1. **Break-even is separated by evidence type.**
   - The first version summarized break-even as "roughly 63–113 executions". That range
     mixed a predicted saving (63) with an observed saving confounded by response
     variability (113).
   - It also omitted the break-even under the independence prediction, which made the
     Milestone 2 choice (62).
   - Section D now reports observed, replay-estimated, and predicted break-even separately,
     plus the range implied by the bootstrap interval of the prediction (18 executions to
     never).
   - Break-even now charges optimization overhead exactly as the net-savings totals do. With
     the default overhead of $0 the numbers are unchanged.
2. **The old "savings upper bound" was not a bound.**
   - The old value, $0.002793527, subtracted from a *predicted* original-order cost.
   - It is replaced by a structural bound that holds for any selectivities: $0.021103824.
   - Under the new bound, the "new workload, upper bound" scenario recommends collecting
     statistics from 1 expected execution onward, instead of from 10.
3. **Profiling economics can now skip adaptive profiling entirely.** Previously the
   simulation always bought at least the first checkpoint. Section F now applies an
   economics gate first.
4. **Replay outputs are labelled as estimates.**
   - Every replay artifact carries `evidence_kind: offline_replay_estimate`,
     `causal_measurement: false`, and the assumptions it rests on.
   - Keys that read like measurements were renamed:
     - `ordering_effect_*` became `replay_estimated_ordering_effect_*`.
     - `variability_residual_*` became `residual_not_explained_by_replay_*`.
     - `replay_measured_ordering_savings_usd` became `replay_estimated_ordering_savings_usd`.
     - `difference_is_exact` became `difference_fully_determined_by_stored_decisions`.
5. **Calibration of the correlation-aware estimator is investigated** in section B.1.

## A. Existing Milestone 2 baseline (unchanged)

Reproduced from `results/milestone_2/jev_product_quality_profile_20261102/summary.md`:

| Metric | Historical fixed (`pq -> pd -> d`) | Fresh optimized (`pd -> pq -> d`) |
|---|---:|---:|
| Completed decisions | 958 | 952 |
| Total tokens | 326,210 | 324,409 |
| Estimated cost | $0.012810000 | $0.012739986 |
| Summed inference latency | 110.987545 s | 95.992702 s |
| Final matches | 66 | 66 |

The recorded match sets are **not** identical. One ID is fixed-only and one is optimized-only, and
both align with the two recorded threshold crossings. Profiling cost $0.007909734 (600 calls), the
recorded net result for one query was −$0.007839720, and the recorded break-even was 113
executions. Independence predicted 914 optimized evaluations and $0.012069965, a 5.26% cost error.

The Milestone 2 break-even of 113 divides the profiling cost by the *observed* difference
between a historical fixed run and a fresh optimized run. That difference is confounded by
response variability (section A.1), so it is not a forecast; see section D.

### A.1 Counterfactual replay of the 958-versus-952 difference

Replay re-executes sequential early rejection over stored decisions and reports bounds whenever a
needed decision is missing. It first reproduces both stored runs exactly from their own traces:
958 evaluations, 305,000 input tokens, $0.01281 (fixed); 952 evaluations, 303,333 input tokens,
$0.012739986 (optimized). Both reproductions also match the recorded match IDs.

The two runs disagree on decisions they share, so the difference is split under explicit
resolution policies:

| Resolution policy | Fixed order (replay) | Optimized order (replay) | Replay-estimated ordering effect | Residual not explained by replay | Replay-estimated ordering effect, cost |
|---|---:|---:|---:|---:|---:|
| Historical decision first | 958 | 953 | 5 calls (fully determined) | 1 call | $5.271e-5 |
| Fresh decision first | 956 | 952 | 4 calls (fully determined) | 2 calls | $3.536e-5 |
| Conflicts treated as unknown | 956–958 | 952–953 | 4–5 calls (bounded) | 1–2 calls | $3.536e-5 – $5.271e-5 |

**These are offline replay estimates, not causal measurements.** "Fully determined" means the
stored decisions fix the replayed count without bounds or imputation under that policy. It
does not make the difference a measured effect. Each estimate rests on three assumptions:

1. A stored decision is what Jev would return for that review and predicate at any time and
   in any evaluation order.
2. The resolution policy is an arbitrary but explicit choice between conflicting repeats.
3. Missing decisions are bounded, not observed.

Neither order was ever run in a randomized or contemporaneous paired execution.

Under those assumptions, replay estimates that ordering accounts for 4–5 of the 6 saved calls,
and for $3.54e-5 to $5.27e-5 (51%–75%) of the observed $7.00e-5 saving. The residual is
consistent with Jev response variability between the two runs. It would also absorb any
violation of the replay assumptions, so it is not a measured variability effect either.

Under any single consistent decision source, both orders return the same 66 match IDs,
because an AND conjunction does not depend on order. The observed one-ID swap therefore
coincides with recorded threshold crossings rather than with ordering. With conflicts treated
as unknown, matches are bounded at 65–67.

Replay cannot attribute retries, HTTP attempts, or latency to an unexecuted order; it reports them
as `null`. The 13.5% summed-latency reduction is therefore not attributed to ordering. Six fewer
calls is a 0.63% reduction in the observed (confounded) comparison.

**Repeated-decision variability.** Shared (review, predicate, model, prompt-hash) pairs:

| Sources | Shared pairs | Threshold crossings | Mean abs. probability change | Max change | Input-token mismatches |
|---|---:|---:|---:|---:|---:|
| M1 fixed vs M2 profiling | 249 | 1 (0.40%) | 0.0041 | 0.05 | 0 |
| M1 fixed vs M2 optimized | 243 | 2 (0.82%) | 0.0097 | 0.10 | 0 |
| M1 fixed vs M1 smoke | 10 | 0 | 0.0030 | 0.03 | 0 |

21 of 1,207 Milestone 1 decisions and 19 of 952 Milestone 2 optimized decisions lie within ±0.05
of the threshold.

**Other held-out orders** (historical decision first; bounds are hard limits, and the estimate
column is labelled imputation from the smoothed profiling outcome-pattern distribution):

| Order | Observability | Unresolved reviews | Evaluation bounds | Imputation estimate |
|---|---|---:|---:|---:|
| `pq -> pd -> d` | fully observable | 0 | 958 | 958 |
| `pd -> pq -> d` | fully observable | 0 | 953 | 953 |
| `pd -> d -> pq` | partially | 19 | 952–971 | 968.6 |
| `pq -> d -> pd` | partially | 24 | 957–981 | 979.3 |
| `d -> pd -> pq` | partially | 733 | 932–1,684 | 1,054.5 |
| `d -> pq -> pd` | partially | 733 | 932–1,689 | 1,060.2 |

`pd -> d -> pq` cannot be excluded as the cheapest held-out order: its lower bound is 952.

## B. Cardinality estimation comparison

Both estimators were fitted only on the 200 profiling reviews. Held-out outcomes were never used
for fitting (`held_out_outcomes_used_for_fitting: false`).

**Held-out, total logical evaluations (the two orders actually executed):**

| Metric | Independence estimator | Correlation-aware estimator |
|---|---:|---:|
| Fixed order: predicted evaluations | 926.0 | 999.98 |
| Fixed order: observed evaluations | 958 | 958 |
| Fixed order: absolute error | 32.0 | 41.98 |
| Fixed order: relative error | 3.34% | 4.38% |
| Optimized order: predicted evaluations | 914.0 | 987.92 |
| Optimized order: observed evaluations | 952 | 952 |
| Optimized order: absolute error | 38.0 | 35.92 |
| Optimized order: relative error | 3.99% | 3.77% |
| Cost relative error (fixed / optimized) | −4.77% / −5.26% | +4.05% / +3.63% |

**Stage input cardinalities** (stages 2 and 3, where the estimators differ):

| Evaluation | Independence MAE / MAPE / mean signed error | Correlation-aware MAE / MAPE / mean signed error |
|---|---|---|
| Held-out, 2 orders × 2 stages | 35.0 / 49.3% / −17.5 | 19.5 / 25.7% / +19.5 |
| Profiling 5-fold cross-validation | 2.75 / 51.2% / −1.97 | 1.33 / 24.7% / −0.001 |
| Profiling in-sample (by construction, not evidence) | 9.84 / 39.5% / −9.84 | 0.34 / 1.3% / −0.006 |

For the fixed order, observed held-out stage inputs were 800, 91, 67. Independence predicted 800,
112, 14; correlation-aware predicted 800, 113.4, 86.5.

Interpretation:

- Independence predicts 14 records reaching stage 3 where 67 and 66 were observed. Its error in
  total evaluations is small only because a stage-2 overestimate cancels a large stage-3
  underestimate.
- The correlation-aware model gets the conditional structure right. It predicts 76.3% of
  stage-2 survivors pass `product_defect` (observed 73.6%); independence predicts 12.5%.
- The correlation-aware errors come from the marginal: profiling saw 14.0% `poor_quality`
  passes, but the held-out set had 11.4%.
- The correlation-aware estimator is clearly more accurate per stage and under
  cross-validation. On held-out **total** evaluations it is not better: MAE 38.95 vs 35.0.
- Neither estimator falls inside the replay bounds of the four tightly bounded held-out orders.
  Independence underestimates all of them (914–938 against 952–981); correlation-aware
  overestimates them (988–1,024).

### B.1 Calibration of the correlation-aware estimator

**Method.** Point errors say nothing about whether the estimator's uncertainty is honest. So
each estimator gets predictive intervals for stage counts:

1. Refit the estimator on 1,000 paired bootstrap resamples of the 200 profiling rows (seed
   3031). This captures parameter uncertainty.
2. In each resample, draw sequential binomial stage survivors on the target record count.
   This captures sampling of the target set.
3. Score each observed count by interval coverage and by mid-PIT, P(draw < observed) +
   0.5·P(draw = observed). Calibrated predictions give PIT values spread uniformly around
   0.5.

Fitting uses profiling rows only. Held-out counts are only scored.

The quantities are stage-2 inputs, stage-3 inputs, and final matches:
- **Held-out:** the two executed orders (6 checks).
- **Profiling 5-fold cross-validation:** 5 folds × 6 orders × 3 quantities (90 checks).

| Evaluation | Estimator | Checks | 50% coverage | 90% coverage | Mean PIT | Mean signed error |
|---|---|---:|---:|---:|---:|---:|
| Held-out | correlation-aware | 6 | 1/6 | 6/6 | 0.166 | +19.8 |
| Held-out | independence | 6 | 1/6 | 2/6 | 0.735 | −32.6 |
| Profiling CV | correlation-aware | 90 | 0.767 | 1.000 | 0.530 | −0.04 |
| Profiling CV | independence | 90 | 0.244 | 0.400 | 0.834 | −2.72 |

Held-out detail for the correlation-aware estimator (point prediction, 90% interval, observed,
PIT):

| Order | Stage-2 inputs | Stage-3 inputs | Final matches |
|---|---|---|---|
| `pq -> pd -> d` | 113.4, [78, 151], 91, 0.145 | 86.5, [55, 121], 67, 0.147 | 86.4, [55, 121], 66, 0.142 |
| `pd -> pq -> d` | 101.5, [68, 139], 86, 0.247 | 86.4, [55, 120], 66, 0.157 | 86.3, [55, 120], 66, 0.158 |

For the fixed order, independence predicts 14.0 stage-3 inputs (interval [6, 25]) and 3.3
final matches ([0, 7]). Observed values were 67 and 66, so both PITs are 1.000.

**Conditional pass rates against held-out Wilson 95% intervals.** Correlation-aware lands
inside 3 of 6; independence lands inside 1 of 6.

| Order | Stage conditional | Correlation-aware | Independence | Observed held-out (Wilson 95%) |
|---|---|---:|---:|---|
| `pq -> pd -> d` | P(pq) | 0.142 | 0.140 | 0.114 [0.094, 0.138] |
| `pq -> pd -> d` | P(pd \| pq) | 0.763 | 0.125 | 0.736 [0.637, 0.816] |
| `pq -> pd -> d` | P(d \| pq, pd) | 0.9988 | 0.235 | 0.985 [0.920, 0.997] |
| `pd -> pq -> d` | P(pd) | 0.127 | 0.125 | 0.1075 [0.088, 0.131] |
| `pd -> pq -> d` | P(pq \| pd) | 0.852 | 0.140 | 0.767 [0.668, 0.844] |
| `pd -> pq -> d` | P(d \| pd, pq) | 0.9988 | 0.235 | 1.000 [0.945, 1.000] |

**First-stage marginal shift** (pooled two-proportion z test):

- `poor_quality`: 14.0% in profiling vs 11.4% on held-out (z = 1.03, p = 0.31).
- `product_defect`: 12.5% vs 10.75% (z = 0.70, p = 0.48).

Both shifts run in the same direction, and neither is individually significant.

**Smoothing sensitivity.** This uses profiling cross-validation only; held-out data was not
used, and the configuration was not changed. Stage MAE by prior strength α:

| α | Stage MAE |
|---:|---:|
| 0 | 1.317 |
| 0.5 | 1.324 |
| 1 (configured) | 1.331 |
| 2 | 1.345 |
| 5 | 1.382 |
| 10 | 1.502 |

For α ≤ 2 the differences are under 0.03 records, so smoothing strength is not a material
error source. α = 1 is kept rather than tuned.

**Findings.**

- **Independence is miscalibrated wherever correlation matters.** Its stage-3 and
  final-match intervals cover 20% and 0% of cross-validation checks, with PIT 0.98 and 1.00,
  and miss both held-out orders. Stage-2 counts depend only on a marginal, and there it is
  calibrated: 90% coverage 1.0, PIT 0.52.
- **Correlation-aware intervals are not too narrow.**
  - All six held-out quantities fall inside their 90% intervals.
  - In cross-validation, mean PIT is 0.53 and no PIT falls outside [0.05, 0.95].
  - Cross-validation 50% intervals cover 77% instead of 50%, so the intervals are, if
    anything, too wide. A plausible but untested contributor is discreteness: 40-review folds
    give stage counts of about 4–6, where inclusive integer quantile bounds over-cover.
- **On held-out data the correlation-aware point predictions are biased high.**
  - All six PITs lie between 0.14 and 0.25.
  - The mean signed error is +19.8 records.
- **The bias traces mainly to the first-stage marginals.** Both were higher in profiling
  than on held-out data, and every later stage inherits that.
  - The shifts are consistent with sampling variability of a 200-review profile.
  - Two executed orders cannot distinguish that from a real distribution difference.
  - The six held-out checks share those two marginals, so they amount to roughly two
    independent pieces of evidence, not six.
- **Two conditionals fall just outside their held-out intervals.**
  - P(pq | pd): 0.852 predicted against 0.767 observed.
  - P(d | pq, pd): 0.9988 predicted against 0.985 observed. This one comes from 22 of 22
    profiling reviews with a backoff target near 1.0, so smoothing barely moves it. Its effect
    on counts is about one review.
- **Verdict.** The correlation-aware estimator is markedly better calibrated than
  independence, and its intervals are usable. Its held-out point predictions are not
  unbiased, so plans and costs should be read with the intervals rather than the point
  values. This is weak evidence: one workload, two executed orders, and correlated checks.

## C. Plan selection

Every candidate, predicted for 800 held-out records, alongside the exact profiling replay. All
correlation-aware candidates have `confidence: high` (minimum conditional support ≥ 20).

| Order | Correlation-aware cost | Correlation-aware evals | Independence cost | Independence evals | Profiling exact evals / cost (200 reviews) |
|---|---:|---:|---:|---:|---:|
| `pd -> pq -> d` | **$0.013202018** | 987.92 | **$0.012069965** | 914.0 | 247 / $0.0033091 |
| `pq -> pd -> d` (original) | $0.013328639 | 999.98 | $0.012199096 | 926.0 | 250 / $0.0033400 |
| `pd -> d -> pq` | $0.013379331 | 1,000.00 | $0.012195070 | 923.5 | 250 / $0.0033537 |
| `pq -> d -> pd` | $0.013658301 | 1,023.88 | $0.012357738 | 938.32 | 256 / $0.0034226 |
| `d -> pd -> pq` | $0.014604904 | 1,088.02 | $0.013328228 | 1,011.5 | 272 / $0.0036603 |
| `d -> pq -> pd` | $0.014754840 | 1,099.90 | $0.013358574 | 1,014.32 | 275 / $0.0036976 |

- **Independence (Milestone 2 baseline, kept executable):** selects `pd -> pq -> d`, a predicted
  reduction of 1.06%. It reproduces every stored Milestone 2 candidate exactly (maximum
  difference 0.0).
- **Correlation-aware:** selects `pd -> pq -> d`, a predicted reduction of 0.95%. Both
  estimators choose the same order. The correlation model changes the predicted magnitudes and
  the ranks of `pd -> d -> pq` and `pq -> pd -> d`, not the winner.
- **Correlation-aware with the confidence-aware policy:** retains the original
  `pq -> pd -> d`, because P(minimum order cheaper) = 0.769 < 0.95
  (`savings_probability_below_confidence_level`).

## D. Profiling economics

All amounts are per execution, meaning one run over the 800 held-out reviews.

**Profiling cost.**

- **Sunk.** The recorded Milestone 2 profile cost $0.007909734 for 600 decisions.
- **Hypothetical cached profile.** 249 of those 600 decisions are already stored in
  Milestone 1 traces with the same review, prompt hash, and model. Reusing them would leave
  $0.004627194 of new spend.

Optimization overhead is $0. The net result of one execution including the sunk profile is
−$0.007839720, using the observed saving.

**Break-even by evidence type.** Break-even is the smallest whole number of executions whose
cumulative saving covers profiling plus overhead. "Never" means that savings basis is not
positive.

| Savings basis | What it is | Saving per execution | Break-even, sunk profile | Break-even, cached profile |
|---|---|---:|---:|---:|
| **Observed** | Difference between the historical fixed run and the fresh optimized run; confounded by response variability and time | $0.000070014 | 113 | 67 |
| **Replay estimate**, historical decision first | Offline replay, not a causal measurement | $0.000052710 | 151 | 88 |
| **Replay estimate**, fresh decision first | Offline replay, not a causal measurement | $0.000035364 | 224 | 131 |
| **Replay estimate**, conflicts unknown | Offline replay, bounded | $0.000035364 – $0.000052710 | 151 – 224 | 88 – 131 |
| **Predicted**, correlation-aware | Planner prediction from profiling statistics | $0.000126621 | 63 | 37 |
| **Predicted**, independence (the Milestone 2 selection model) | Planner prediction from profiling statistics | $0.000129131 | 62 | 36 |
| **Predicted**, correlation-aware 95% bootstrap interval | Sampling uncertainty of the prediction | −$0.000216290 – $0.000456166 | 18 – never | 11 – never |

How to read these rows:

- **The observed and replay rows are retrospective.** They describe this workload's single
  pair of executions and are not forecasts. Neither is a causal measurement of the ordering
  effect:
  - The observed row mixes ordering with response variability.
  - The replay rows depend on the replay assumptions in section A.1.
- **Only the predicted rows are available to a planner before execution.** The prediction
  is 2.4–3.6 times the replay-estimated ordering saving and 1.8 times the observed
  difference, so on this workload the predicted break-even of 62–63 is optimistic relative to
  the retrospective evidence.
- **The prediction's own uncertainty spans everything from paying back in 18 executions to
  never paying back,** because the bootstrap interval includes non-positive savings.

**Savings bounds for a workload that has not been profiled.**

- **Structural upper bound: $0.021103824.** It equals N × (sum of per-evaluation costs minus
  the cheapest). The original order evaluates at most every predicate on every record, and
  every order evaluates all N records at stage 1, so this holds under any selectivities. It
  is about 167 times the predicted saving, which makes it valid but uninformative.
- **Transferred prediction: $0.000126621.** This assumes a comparable workload saves what
  this one is predicted to save. It is an assumption, not a bound.

**Decisions with conservative settings**, from `economics.json`. Each decision records its
`savings_basis`.

- "Existing" means the sunk profile is reused.
- "Confidence-gated" additionally requires P(saving) ≥ 0.95.
- The last two columns describe a new workload that has not been profiled, and assume a
  full 200-review profile.

| Expected executions | Existing statistics | Existing + confidence-gated | New workload, full profile, transferred prediction | New workload, full profile, structural bound |
|---:|---|---|---|---|
| 0 | execute original (no executions) | execute original | execute original, no profiling | execute original, no profiling |
| 1 | reuse plan | decline (P = 0.769) | do not profile (cannot break even) | collect statistics |
| 10 | reuse plan | decline | do not profile | collect statistics |
| 100 | reuse plan | decline | collect statistics | collect statistics |
| 113 | reuse plan | decline | collect statistics | collect statistics |
| 1,000 | reuse plan | decline | collect statistics | collect statistics |

What the scenarios show:

- **Once profiling is sunk,** reusing the plan is forward-profitable at any positive horizon
  by its point estimate. That estimate is not credible at the 95% level, so the
  confidence-gated policy declines.
- **For a new comparable workload,** a full profile pays back on the predicted basis only
  from 63 expected executions onward. On the retrospective bases it would take 113
  (observed) or 151–224 (replay estimate). The bootstrap interval leaves open that it never
  pays back.
- **The structural bound** clears every positive horizon, because it is too loose to rule
  anything out.

## E. Statistical robustness

Paired bootstrap, 1,000 resamples of whole profiling rows (seed 3031). This is a diagnostic of
sampling uncertainty, not evidence of generalization.

| Estimator | `pd -> pq -> d` | `pq -> pd -> d` | Other orders | Stable at 0.95 |
|---|---:|---:|---:|---|
| Correlation-aware | 76.9% | 23.1% | 0% | no |
| Independence | 78.6% | 21.4% | 0% | no |

Saving of `pd -> pq -> d` over the original, per 800-record execution (correlation-aware):

- Mean $1.195e-4, with a 95% percentile interval of −$2.163e-4 to $4.562e-4.
- Relative saving: mean 0.89%, interval −1.68% to 3.41%.
- P(cheaper than original) is 0.769.
- Mean regret is $2.43e-5, and the regret is positive in 23.1% of resamples.

The independence estimator gives an interval of −$1.872e-4 to $4.387e-4 and P = 0.786. The
selected plan is **not stable**: about one resample in four prefers the original order, and every
savings interval includes zero.

## F. Adaptive profiling (offline simulation)

The simulation reveals rows of the complete 200-review matrix one checkpoint at a time, in a
fixed sha256-ranked order. Checkpoints are at 25, 50, 100, 150, and 200 reviews, and each runs
a 1,000-resample bootstrap. Savings are offline estimates.

**Economics gate before the first checkpoint.** Adaptive profiling commits to at least the
first checkpoint: 75 decisions, an estimated $0.000988717 at the recorded average of
$1.318289e-5 per decision. Before buying it, `decide_adaptive_profiling_start` skips
profiling entirely in any of these cases:

- no execution is expected;
- reusable statistics already exist;
- the first checkpoint would exceed the decision cap or a budget;
- cost or savings bound is unknown;
- horizon × savings bound minus the first-checkpoint cost does not exceed the minimum net
  saving.

When profiling is skipped, the original order runs, no decisions are consumed, and the
artifact still reports the full-profile plan, so the forgone predicted saving stays visible.
The gate is evaluated under the two savings bases from section D.

| Gate basis | Expected executions | Gate | Decisions used | Profiling cost avoided vs full profile (offline estimate) | Stop reason | Selected order |
|---|---:|---|---:|---:|---|---|
| Either | 0 | **skip** (no expected executions) | 0 | $0.007910 | — | original |
| Transferred prediction | 1 | **skip** (best-case net −$0.000862; break-even 8) | 0 | $0.007910 | — | original |
| Transferred prediction | 10 | start | 75 | $0.006927 | value of information < next increment | `pd -> pq -> d` |
| Transferred prediction | 113 | start | 150 | $0.005959 | value of information < next increment | `pd -> pq -> d` |
| Transferred prediction | 1,000 | start | 600 | $0 | final checkpoint | `pd -> pq -> d` |
| Structural bound | 1 | start | 75 | $0.006927 | value of information < next increment | `pd -> pq -> d` |
| Structural bound | 10 | start | 75 | $0.006927 | value of information < next increment | `pd -> pq -> d` |
| Structural bound | 113 | start | 150 | $0.005959 | value of information < next increment | `pd -> pq -> d` |
| Structural bound | 1,000 | start | 600 | $0 | final checkpoint | `pd -> pq -> d` |

Skipping at one execution forgoes a predicted $0.000127, against the $0.000989 the first
checkpoint would have cost.

The two bases differ in how they can fail:

- **Structural bound.** It does not skip profiling that could pay off, unless the
  per-evaluation cost estimates are wrong. But it is so loose that it skips only at zero
  reuse or when a cap applies.
- **Transferred prediction.** It is a point assumption used where the gate expects a bound,
  so it can skip profiling that would have paid off if the new workload saves more than this
  one.

Once profiling starts, the gate does not change the controller: the trajectory is identical
to an ungated run. A unit test checks this.

Every started run selected `pd -> pq -> d` at every checkpoint. Bootstrap selection frequency by checkpoint was 0.645,
0.889, 0.773, 0.746, 0.787, which never reached the 0.95 stability threshold and did not rise
monotonically with sample size. All stops came from the value-of-information rule. That bound is
itself noisy: at the 113-execution horizon it fell below the increment cost at 50 reviews
($0.00036 vs $0.00195), but it would have exceeded it at 100 reviews ($0.00271 vs $0.00197).
Agreement with the full profile is agreement with another estimate, not with a known optimum.

## G. Limitations

- **Jev response variability.** Repeated decisions disagree on 0.4%–0.8% of shared pairs, and
  probabilities move by up to 0.10. Replay leaves one or two of the six saved calls, and the
  match-set difference, unexplained by ordering; this is consistent with variability but is not
  a measurement of it.
- **Replay is an estimate, not a causal measurement.** It assumes stored decisions are
  stationary and independent of evaluation order, and its result depends on the resolution
  policy. No randomized or contemporaneous paired execution of both orders exists.
- **Calibration evidence is thin.** Held-out calibration rests on 6 checks from 2 executed
  orders that share their first-stage marginals. On held-out data the correlation-aware point
  predictions are biased high (PIT 0.14–0.25), although their 90% intervals cover.
- **Break-even depends on the savings basis.** Predicted (62–63), observed (113), and
  replay-estimated (151–224) break-even differ by up to 3.6×. The bootstrap interval of the
  prediction includes never.
- **Strong predicate correlation.** Phi is 0.81 for `pq`–`pd`, 0.73 for `pq`–`d`, and 0.68 for
  `pd`–`d`. `dissatisfaction` holds on every profiled review where `poor_quality` or
  `product_defect` holds. Independence is badly misspecified here; the correlation model uses one
  sample's joint structure.
- **Limited profiling sample.** 200 reviews contain 28, 25, and 22 passes, and the marginal
  difference that decides the order is 3 reviews. The held-out `poor_quality` rate (11.4%)
  differs from profiling (14.0%).
- **Limited semantic workloads.** Only `product_quality` has Jev outcomes. Three new workload
  configurations are verified only with mocks; their model quality and optimization gains are
  unmeasured.
- **Missing held-out counterfactual decisions.** Four of six orders are only bounded on held-out
  data; orders starting with `dissatisfaction` leave 733 reviews unresolved.
- **No human-annotated quality ground truth.** Match counts measure agreement between runs, not
  correctness.
- **Historical versus contemporaneous comparison.** The fixed run (Milestone 1) and the optimized
  run (Milestone 2) were executed at different times, so time-dependent variability and latency
  are confounded with ordering.
- **Uncertain future reuse.** Profitability depends on the expected number of executions, which
  is unknown. The default assumes one execution, under which a fresh profile never pays off.
