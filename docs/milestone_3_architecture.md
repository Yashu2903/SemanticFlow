# Milestone 3 architecture

Milestone 3 extends the Milestone 2 planner; it does not replace it. The sequential executor,
providers, request budgets, split, profiler, tracing, and result schemas are unchanged. The
independence estimator remains the default (`cardinality_estimator: independent`,
`cost_model_version: independent-and-v1`) and reproduces every stored Milestone 2 candidate
exactly. All new analysis runs offline over stored decisions.

## Module map (`src/semanticflow/optimizer/`)

| Module | Responsibility |
|---|---|
| `matrix.py` | `ProfileMatrix`: review-by-predicate outcomes and usage from profiling traces; joint statistics (marginals, pairwise phi and lift, conditionals with support, all conjunctions, outcome patterns) and a matrix fingerprint. |
| `cardinality.py` | `IndependentEstimator` (Milestone 2 model) and `CorrelationAwareEstimator`; both return per-stage `StageEstimate`s consumed by the cost model. |
| `cost_model.py` | `estimate_plan(..., estimator=None)`: unchanged validation and arithmetic, now fed by stage estimates; records stage cardinalities and confidence on each candidate. |
| `planner.py` | Selects the estimator, validates the matrix against the profile and split, applies the Milestone 2 tie rule, and optionally the confidence-aware policy. |
| `evidence.py` | `DecisionEvidence`: stored decisions indexed by (review, predicate, model version, question hash), with repeated observations kept separately. |
| `replay.py` | Counterfactual early-rejection replay with exact results or bounds, resolution policies, joint order comparison, and labelled imputation estimates. |
| `bootstrap.py` | Paired row bootstrap over the profiling matrix and the confidence-aware choice. |
| `economics.py` | Break-even (point and interval), sunk-versus-new profiling accounting, forward-looking decision actions with machine-readable reasons and a recorded savings basis, and the adaptive-profiling start gate. |
| `calibration.py` | Predictive intervals for stage cardinalities (bootstrap refits plus sequential binomial draws), coverage and PIT scoring, cross-validated calibration, and a two-proportion marginal-shift test. |
| `adaptive.py` | Offline sequential-profiling controller simulation, gated by profiling economics before the first checkpoint. |
| `analysis.py` | `run_offline_analysis`: orchestrates the phases above over stored artifacts into a new directory. |

CLI additions (`semanticflow replay`, `semanticflow analyze`) are synchronous, take no provider
argument, and never construct a provider. Existing commands are unchanged.

## Correlation-aware cardinality

For a candidate order `P1, ..., Pk` over `N` held-out records, the expected input to stage `k` is

```text
N_k = N * P(P1 AND ... AND P_{k-1})
    = N * prod_{j<k} P(P_j | P_1 ... P_{j-1})
```

Each conditional uses the full prefix, counted on the complete profiling matrix:
`x` = rows passing the prefix and `P_j`, `m` = rows passing the prefix. It is shrunk toward a
backoff target `t` with a Beta prior of strength `alpha` (`correlation.prior_strength`):

```text
P_hat = (x + alpha * t) / (m + alpha)
```

Backoff targets:

- marginal (empty prefix): the Jeffreys mean 0.5;
- one conditioning predicate: the smoothed marginal of `P_j`;
- longer prefixes: the support-weighted mean of the smoothed pairwise conditionals
  `P(P_j | P_i)` over prefix members, so information from shorter conditioning sets is used
  when the full prefix is sparse.

If `m = 0`, the estimate is the backoff target, labelled `backoff_unseen_prefix`. With
`alpha = 0`, the estimator equals exact early-rejection replay over the profiling matrix; tests
check this for all 24 orders of a 4-predicate workload.

**Why smooth.** Sparse prefixes (for example the 22 reviews passing two predicates) make raw
ratios jump between 0 and 1. `alpha = 1` adds one pseudo-observation, which is negligible at
`m >= 20` and prevents zero or one estimates from tiny supports. This is a regularizer, not a
claim about the true prior. Because the backoff target is itself a conditional, a conditional
that is exactly 1 moves only slightly. For example, `P(d | pq AND pd) = 22/22` becomes
`(22 + 0.972) / 23 ≈ 0.999`.

**Confidence.** Each stage records the minimum conditional support along its chain. A stage is
`confidence: high` when that support is at least `minimum_conditional_support` (20), the usual
normal-approximation rule of thumb. A plan is `low` if any stage is low. Confidence is reported
and never silently changes the plan.

**Conditional usage.** With `conditional_usage: true`, per-stage token and latency means are taken
over profiling rows that pass the prefix, shrunk toward the unconditional mean with strength
`usage_prior_strength` (5). Review length correlates with complaints, so reached records cost
more than average.

**Assumptions.** Profiling and held-out reviews are exchangeable samples of one population; the
joint structure measured on 200 reviews transfers; Jev outputs are stationary over the run window.
The matrix must be complete unless `allow_incomplete_matrix` is set; complete-case analysis then
assumes missingness is unrelated to outcomes.

**Calibration.** `calibration.py` checks whether the estimator's uncertainty is honest, not
just its point error:

1. Refit the estimator on paired bootstrap resamples of profiling rows.
2. Draw stage survivors sequentially as `Binomial(survivors, P(P_j | prefix))` on the target
   record count.
3. Score observed counts by 50% and 90% interval coverage and by mid-PIT.

Products of the chain conditionals reproduce each estimator's reach probabilities, and a
test checks this. Held-out counts are scored but never fitted. Cross-validated calibration
uses only the profiling matrix. Smoothing sensitivity is reported from cross-validation only,
and it does not change the configuration.

**Leakage safeguards.** `ProfileMatrix.from_traces` raises on held-out IDs, IDs outside the
profiling split, duplicates, or overlapping splits. The planner checks that matrix rows equal the
profiling IDs, that the size matches, and that matrix marginals equal the stored statistics.
Selected plans record `cardinality_estimator`, `joint_statistics_version`, and
`joint_matrix_fingerprint`.

## Counterfactual replay

Each review's cell for a predicate is resolved from stored observations that match the exact model
version and question hash. A threshold different from the stored one re-thresholds the recorded
probability. Resolution policies list the allowed sources in priority order:

- `prefer_priority` takes the highest-priority source when repeated decisions disagree;
- `treat_as_unknown` turns a disagreement into an unknown cell.

The replay then enumerates every assignment of unknown cells for each review. A review is resolved
when all assignments give the same early-rejection path. Per-review minima and maxima give bounds
for evaluations, stage counts, matches, tokens, and cost. The report labels each order fully,
partially, or not observable.

`compare_orders` bounds differences per assignment, so they are joint rather than differences of
marginal bounds. The optional estimate weights assignments by a Dirichlet(0.5)-smoothed
profiling outcome-pattern distribution and is labelled `profile_pattern_conditional_imputation`.
Counterfactual HTTP attempts, retries, and wall time are reported as `null`, because stored
decisions do not determine them.

**Evidence label.** Every replay result carries `REPLAY_EVIDENCE`, which states
`evidence_kind: offline_replay_estimate`, `causal_measurement: false`, and the assumptions:

- stored decisions are stationary and independent of evaluation order;
- the resolution policy is a choice between conflicting repeats;
- missing decisions are bounded, not observed.

"Exact" bounds and `difference_fully_determined_by_stored_decisions` mean the stored
decisions fix the count without imputation. They do not make a replayed difference a measured
or causal effect. The analysis names the parts of the observed difference accordingly:

- the ordering part is `replay_estimated_ordering_effect_*`;
- the remainder is `residual_not_explained_by_replay_*`.

## Profiling economics

Decisions are forward-looking: money already spent on profiling never justifies or blocks a
decision. Retrospective totals including sunk cost are reported separately.
`break_even = ceil((profiling_cost + overhead) / saving_per_execution)`, charging the same
overhead as the matching net-savings total, with float noise near integers tolerated. It is
`null` when the saving is not positive. `break_even_range` maps a savings interval to the
fewest and the most executions.

Every saving carries a basis from `SAVINGS_BASES`, and break-even is reported separately for
each:

- `observed_confounded`: two recorded executions;
- `replay_estimate`;
- `predicted`;
- `bootstrap_interval`;
- `structural_upper_bound`;
- `transferred_prediction`.

Only `predicted`, `structural_upper_bound`, and `transferred_prediction` are available to a
planner before execution. None of the bases is a causal measurement.

Actions:

| Action | When it is chosen |
|---|---|
| `execute_original_without_profiling` | Pricing unknown, budget exceeded, no executions, or profiling cannot break even. |
| `collect_additional_statistics` | No statistics yet and the best-case saving could repay profiling. |
| `optimize_with_existing_statistics` | Forward net saving above the minimum and all gates pass. |
| `reuse_existing_plan` | The same, when an existing plan is reused. |
| `decline_optimization_execute_original` | Statistics exist but the saving is non-positive, too small, too uncertain, or too improbable. |

Conservative defaults: unknown reuse means one execution, with no budget, uncertainty, or
probability gates unless configured.

The structural savings upper bound for a new workload is
`N * (sum of per-evaluation costs - cheapest per-evaluation cost)`. It holds under any
selectivities, because:

- the original order evaluates at most every predicate on every record;
- every order evaluates all `N` records at stage 1.

An earlier version subtracted from the *predicted* original-order cost, which is not a bound.

**Adaptive start gate.** `decide_adaptive_profiling_start` runs before any profiling
decision. It returns one of:

- `skip_profiling_use_existing_statistics`, when reusable statistics exist;
- `skip_profiling_execute_original`, when no execution is expected, the first checkpoint
  exceeds the decision cap or a budget, cost or bound is unknown, or
  `horizon * bound - first_checkpoint_cost - overhead` does not exceed the minimum net
  saving;
- `start_adaptive_profiling`, otherwise.

Without economics settings, only the caps and budgets apply.

## Paired bootstrap and confidence-aware selection

Whole profiling rows are resampled with replacement, keeping every predicate's outcome and usage
together. Statistics and all candidate costs are recomputed, and the planner tie rule picks a
winner. Only `random.Random(seed).random()` is used, because it is Python's cross-version
deterministic generator output.

Thresholds:

| Setting | Default | Meaning |
|---|---:|---|
| `bootstrap.iterations` | 1,000 | Number of resamples. |
| `bootstrap.seed` | 3031 | Resampling seed. |
| `bootstrap.interval_level` | 0.95 | Two-sided percentile intervals. |
| `bootstrap.stability_threshold` | 0.95 | A plan is "stable" when it wins at least this share of resamples. |
| `bootstrap.confidence_level` | 0.95 | The confidence-aware policy adopts the cheaper order only if it beats the original in at least this share of resamples (a one-sided percentile criterion). |
| `bootstrap.minimum_relative_saving` | 0.0 | Additionally require at least this full-sample relative saving. |

Otherwise the policy keeps the original order and records the reasons.

## Adaptive profiling (offline)

Design: a uniform sample without replacement of complete rows, in `sha256(seed:review_id)` order
fixed before any outcome is seen. Every predicate is evaluated for every sampled review, so each
checkpoint is a simple random sample and there is no outcome-dependent selection bias. The stopping
time is data-dependent, so estimates at the stop are not guaranteed unbiased.
Predicate-selective (outcome-dependent) profiling is not implemented; it would require
inverse-probability weighting.

Before the first checkpoint, the start gate above may skip profiling entirely. The original
order then runs, no decisions are consumed, the trajectory is empty, and the artifact still
reports the full-profile plan as a diagnostic. A passed gate does not alter the controller.
The first-checkpoint cost is `estimated_cost_per_decision × decisions` when given. Otherwise
it is the recorded usage of those rows, and the source is labelled.

At each checkpoint the controller re-plans, bootstraps, and stops when any of these holds:

- the plan is stable (selection frequency ≥ `adaptive.stability_threshold`);
- the expected value of information (expected executions × mean bootstrap regret) is at most
  the cost of the next increment;
- the next increment exceeds the profiling budget or decision cap;
- the final checkpoint is reached.

Checkpoints after the stop are recorded as `after_stop_diagnostic`. The simulation reports zero
live provider calls.
