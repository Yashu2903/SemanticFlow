# Milestone 4 improvement: joint adaptive profiling and risk-aware plan selection

Exploratory retrospective evaluation over stored Milestone 3 Jev traces. No new Jev request was
made. The canonical artifacts are in
`results/milestone_4/joint_adaptive_risk_20261106_2000/` (2,000 uncertainty draws, seed 5051,
32 VOI scenarios). The earlier `joint_adaptive_risk_20261106/` run is retained as a 500-draw
sensitivity artifact; it is not the source of the tables below.

The four workloads already informed Milestones 3 and 4. These results are exploratory, not a
confirmatory test, and no statistical significance is claimed.

## 1. Motivating failure and reproduced baseline

The canonical Milestone 4 benchmark was reproduced before this change in
`results/milestone_4/offline_benchmark_joint_baseline_reproduction_20261106/`. Its four
per-workload JSON files are identical to the original; `benchmark.json` differs only in its
creation timestamp.

The exact `listing_mismatch` failure path was:

1. The independent adaptive controller stopped after 25 reviews (75 predicate decisions,
   $0.001005816 in recorded profiling usage).
2. It selected `value_complaint -> description_mismatch -> dissatisfaction`.
3. The separate risk gate applied an empirical paired bootstrap to that prefix.
4. The candidate failed `minimum_saving_probability: 0.95` and the planner executed the original
   order.
5. The historical live benchmark's $0.000641823 (5.44%) per-execution saving was missed.

The full 200-review risk-aware policy adopted the reorder with empirical bootstrap frequency
0.9985. Existing full-profile and adaptive results were reproduced:

| Metric | Full-profile risk-aware | Adaptive + final risk gate |
|---|---:|---:|
| Profiling cost | $0.032034 | $0.003980 |
| Observed execution saving per four-workload bundle | $0.010691 | $0.010049 |
| Mean observed execution saving | 12.62% | 11.26% |
| Cold-start net at R=1 | -$0.021343 | +$0.006069 |
| Cold-start net at R=10 | +$0.074877 | +$0.096513 |

These profiling costs are offline charges from already recorded Jev outcomes, not measured savings
from a live sequential-profiling experiment.

## 2. Joint controller design

The controller makes one state transition before profiling and one after each nested checkpoint.

```text
NO STATISTICS
  |-- reusable evidence ------------------------------> REUSE_EXISTING_PLAN
  |-- cannot recover first checkpoint / R=0 / budget -> SKIP_PROFILING
  `-- start ------------------------------------------> CHECKPOINT

CHECKPOINT (revealed prefix only)
  |-- candidate passes configured risk policy --------> ADOPT_REORDER
  |-- positive net value of next checkpoint ----------> PROFILE_MORE -> CHECKPOINT
  `-- no valuable/budget-feasible next checkpoint ----> KEEP_ORIGINAL
```

`ADOPT_REORDER`, `PROFILE_MORE`, `KEEP_ORIGINAL`, `SKIP_PROFILING`, and
`REUSE_EXISTING_PLAN` are machine-readable actions. Every transition records its reasons.

The implementation extends the existing architecture:

- `joint.py` owns sequential evidence state, sparse uncertainty, VOI, budget checks, and traces.
- `risk.py` remains the adoption policy; regret, CVaR, confidence, and economics are not copied.
- `bootstrap.py` retains the empirical paired bootstrap and adds one sparse posterior-predictive
  alternative.
- `joint_benchmark.py` separates controller evidence from held-out scoring and produces the
  horizon/ablation artifacts.

## 3. Sparse-data uncertainty

### Problem

An empirical bootstrap cannot produce an unseen positive outcome. At 0/25 passes it treats zero as
structurally fixed in every resample. The existing point estimator smooths conditionals, but that
does not make the empirical bootstrap distribution non-degenerate for the unseen event.

### Treatment

For a checkpoint with binary outcome pattern (z \in \{0,1\}^k), the controller uses a
Dirichlet posterior-predictive approximation:

\[
\alpha_z = n_z + \lambda q_z,
\]

where (n_z) is the observed complete-pattern count, (lambda=1), and (q_z) is the product
distribution formed from Jeffreys-smoothed marginal means. Thus every possible pattern has positive
prior mass, while observed joint patterns dominate as the profile grows.

Each draw:

1. samples pattern probabilities from the Dirichlet distribution;
2. samples a complete outcome pattern for every synthetic row;
3. reuses usage from an observed row with the same pattern when available, otherwise backing off to
   any observed row;
4. refits the existing estimator and costs all candidate orders.

The draws are statistical simulations, not Jev responses and not fabricated observations in the
profile. Tests cover 0/25, 1/25, 2/25, sparse conditional support, zero conjunction support, and
deterministic replay.

### Uncertainty scope

| Source | Representation |
|---|---|
| Predicate selectivity | Smoothed posterior-predictive pattern draws |
| Predicate correlations | Joint outcome-pattern distribution and its finite-sample uncertainty |
| Estimated execution cost | Existing estimator and cost model refit on every draw |
| Jev response variability | Not included as an independent noise term; repeated live traces remain sensitivity evidence only |

This is not a complete Bayesian response model. It assumes exchangeable reviews, uses a simple
pattern prior, backs unseen-pattern usage off to all rows, and does not model distribution shift.

## 4. Value of additional profiling

Let (d_n) be the current risk-aware decision at (n) reviews and let (C(P,\theta)) be the
candidate cost in a current posterior-predictive scenario (	heta). Decision loss is regret against
the cheapest modeled permutation in that scenario:

\[
L(d,\theta)=C(d,\theta)-\min_P C(P,\theta).
\]

For the next checkpoint (m>n), a projected evidence center is updated toward a predictive draw,
weighted by (n) and (m-n). Dispersion contracts by ((n/m)^{1/2}). The existing risk gate is
then rerun on the projected distribution, producing (d_m). The approximation is:

\[
\widehat{EVSI}_{n\rightarrow m}
= E[L(d_n,\theta)-L(d_m,\theta)],
\]

\[
VOI = R\widehat{EVSI}_{n\rightarrow m} - C_{n\rightarrow m}^{profile}.
\]

The controller continues only when `VOI > minimum_voi_usd` (default 0), a next checkpoint exists,
and cost/decision budgets permit it.

The projection uses only current-prefix draws. It never reads stored outcomes from reviews
(n+1,\ldots,m). The incremental cost estimate uses mean observed profiling cost through the
current checkpoint; actual future-row usage is charged only after the controller chooses
`PROFILE_MORE` and those rows are revealed.

This is an EVSI approximation, not an exact pre-posterior calculation. Root-n contraction may be
poor for rare, highly correlated predicates; model-response variability and future distribution
shift are omitted.

## 5. No-lookahead and replay methodology

- Workloads: `product_quality`, `listing_mismatch`, `positive_experience`, `mixed_feedback`.
- Checkpoints: 25, 50, 100, 150, 200 reviews.
- Sampling: deterministic SHA-256 rank with seed 4049; checkpoints are nested.
- Controller inputs: revealed profiling prefix, query definition, configured prices/budgets, and
  reuse horizon.
- Evaluation inputs: complete stored profile and historical live order costs, joined only after a
  terminal action.
- Uncertainty: 2,000 draws at every checkpoint, seed `5051 + checkpoint`; 32 deterministic VOI
  scenarios.
- Candidate set: all six permutations of each three-predicate conjunction.

Tests perturb every stored row after the first checkpoint while holding the prefix fixed and assert
that the first trace entry is byte-for-byte identical. The checkpoint API has no future-matrix or
held-out-evidence parameter.

## 6. `listing_mismatch` checkpoint trace

At 25 reviews the revealed pass counts are 2 `description_mismatch`, 1 `value_complaint`, and 10
`dissatisfaction`; no row passes both rare predicates. The controller nevertheless identifies the
historically cheaper candidate at every checkpoint.

| Reviews | Candidate bootstrap frequency cheaper | Expected relative saving | EVSI reduction / execution | Next checkpoint cost | Risk action |
|---:|---:|---:|---:|---:|---|
| 25 | 0.8000 | 7.08% | $0.000778705 | $0.001005816 | keep; insufficient confidence |
| 50 | 0.8845 | 5.56% | $0.000389202 | $0.001996512 | keep; insufficient confidence |
| 100 | 0.9910 | 6.06% | n/a | n/a | adopt reorder |
| 150 | 0.9890 | 5.90% | n/a | n/a | adopt reorder |
| 200 | 0.9920 | 5.44% | n/a | n/a | adopt reorder |

Horizon-dependent path:

- **R=1:** 25-review VOI is negative after cost; keep the original at 25.
- **R=5:** continue at 25, but 50-review VOI for reaching 100 is negative; keep at 50.
- **R>=10:** continue at 25 and 50; at 100 the candidate passes the existing risk gate and is
  adopted.

At R=10 the joint controller consumes 100 reviews / 300 decisions and costs $0.004032714. Relative
to the previous 25-review stack, it buys 75 additional reviews / 225 decisions for $0.003026898.
The historical live saving is $0.000641823 per execution, so ten executions save $0.006418230 and
leave +$0.002385516 after the complete 100-review profile. The previous adaptive+risk stack had
-$0.001005816 net at the same horizon.

The controller therefore recovers the reorder at R=10, not at R=1 or R=5. It is not hardcoded to
this workload; the same policy over-profiles `product_quality` at some horizons and still rejects
its reorder.

## 7. Per-workload comparison at R=10

All savings and observed regrets use historical live costs. "Cheapest" means cheapest executed
order, not the unknown globally cheapest permutation.

| Workload / strategy | Reviews | Calls | Profile cost | Selected order | Saving | Expected regret | Observed regret | Net R=10 |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| product_quality, existing adaptive+risk | 25 | 75 | $0.000983 | original | 0% | $0.000387 | $0.0000456 | -$0.000983 |
| product_quality, joint | 100 | 300 | $0.003940 | original | 0% | $0.000212 | $0.0000456 | -$0.003940 |
| listing_mismatch, existing adaptive+risk | 25 | 75 | $0.001006 | original | 0% | $0.000943 | $0.000642 | -$0.001006 |
| listing_mismatch, joint | 100 | 300 | $0.004033 | value_complaint first | 5.44% | $0.00000112 | $0 | +$0.002386 |
| positive_experience, existing adaptive+risk | 25 | 75 | $0.000991 | original | 0% | $0.0000189 | $0.000118 | -$0.000991 |
| positive_experience, joint | 25 | 75 | $0.000991 | original | 0% | $0.0000475 | $0.000118 | -$0.000991 |
| mixed_feedback, existing adaptive+risk | 25 | 75 | $0.001001 | size issue first | 45.02% | $0.0000278 | $0 | +$0.099492 |
| mixed_feedback, joint | 25 | 75 | $0.001001 | size issue first | 45.02% | $0.0000464 | $0 | +$0.099492 |

At R=10 the joint aggregate has mean/median execution savings 12.62% / 2.72%, mean/max observed
regret $0.0000408 / $0.0001177, two beneficial reorders accepted, two beneficial executed
reorders rejected, zero harmful reorders accepted, 187.5 average profiling decisions, and
$0.009964710 total profiling cost.

Leave-one-workload-out mean observed savings are 16.82% excluding `product_quality` or
`positive_experience`, 15.01% excluding `listing_mismatch`, and 1.81% excluding `mixed_feedback`.
The last result shows that the aggregate remains dominated by `mixed_feedback`.

## 8. Cold-start economic comparison

Net saving sums the four workloads and charges each strategy's new profiling. Optimization
overhead is configured to zero.

| R | Fixed | Full correlation | Full confidence | Existing adaptive | Adaptive+risk | Joint |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | $0 | -$0.021297 | -$0.021343 | +$0.006757 | +$0.006069 | +$0.006069 |
| 5 | $0 | +$0.021650 | +$0.021422 | +$0.049703 | +$0.046266 | +$0.045275 |
| 10 | $0 | +$0.075334 | +$0.074877 | +$0.103387 | +$0.096513 | +$0.096946 |
| 25 | $0 | +$0.236385 | +$0.235244 | +$0.264438 | +$0.247252 | +$0.257313 |
| 50 | $0 | +$0.504803 | +$0.502521 | +$0.530880 | +$0.496507 | +$0.522589 |
| 100 | $0 | +$1.041640 | +$1.037076 | +$1.062678 | +$0.993933 | +$1.057144 |
| 250 | $0 | +$2.652150 | +$2.640741 | +$2.664207 | +$2.492343 | +$2.660809 |
| 1,000 | $0 | +$10.704699 | +$10.659066 | +$10.708761 | +$10.663128 | +$10.679134 |

Relative to adaptive+risk, joint is tied at R=1, worse by $0.000991 at R=5, better by $0.000434
at R=10, and better by $0.010061 at R=25. At R=1, the new controller preserves the previous
adaptive economics. At R=5 it buys an unproductive second checkpoint for `listing_mismatch`. At
R>=10 it recovers that workload, while extra `product_quality` profiling offsets part of the gain.

## 9. Warm-start economics

Warm-start rows exclude historical profiling expenditure as sunk and assume the evidence/plan
produced by that horizon's policy is reusable. No strategy is charged for collecting the same
statistics again.

| R | Full correlation | Full confidence | Existing adaptive | Adaptive+risk | Joint |
|---:|---:|---:|---:|---:|---:|
| 1 | $0.010737 | $0.010691 | $0.010737 | $0.010049 | $0.010049 |
| 5 | $0.053684 | $0.053456 | $0.053684 | $0.050246 | $0.050246 |
| 10 | $0.107367 | $0.106911 | $0.107367 | $0.100493 | $0.106911 |
| 25 | $0.268418 | $0.267278 | $0.268418 | $0.251232 | $0.267278 |
| 50 | $0.536837 | $0.534555 | $0.536837 | $0.502464 | $0.534555 |
| 100 | $1.073673 | $1.069110 | $1.073673 | $1.004928 | $1.069110 |
| 250 | $2.684183 | $2.672775 | $2.684183 | $2.512319 | $2.672775 |
| 1,000 | $10.736733 | $10.691100 | $10.736733 | $10.691100 | $10.691100 |

Warm-start values are forward-looking benefits of the selected plan, not recovery of historical
profiling spend.

## 10. Controlled ablations at R=10

| Policy | Reviews by PQ / LM / PE / MF | Mean saving | Total cold net | Finding |
|---|---|---:|---:|---|
| Joint | 100 / 100 / 25 / 25 | 12.62% | $0.096946 | reference |
| Independent stop, then risk | 25 / 25 / 25 / 25 | 11.26% | $0.096513 | misses listing_mismatch |
| Empirical sparse-data ablation | 100 / 100 / 25 / 25 | 12.62% | $0.096946 | no terminal change at R=10 |
| VOI disabled | 25 / 25 / 25 / 25 | 11.26% | $0.096513 | same failure as independent stacking |
| Active expected-regret cap ($0.0001) | 100 / 100 / 25 / 25 | 12.62% | $0.096946 | no terminal change |
| Active CVaR loss cap ($0) | 25 / 100 / 25 / 25 | 12.62% | $0.099904 | avoids unproductive PQ profiling |
| Robust estimator disagreement | 100 / 100 / 25 / 25 | 12.62% | $0.096946 | no terminal change |
| Profiling cost ignored in VOI | 150 / 100 / 25 / 25 | 12.62% | $0.094945 | over-profiles PQ; economics matters |
| 75-decision budget | 25 / 25 / 25 / 25 | 11.26% | $0.096513 | cannot recover listing_mismatch |
| 150-decision budget | 50 / 50 / 25 / 25 | 11.26% | $0.094554 | pays more and still cannot recover it |
| 300- or 600-decision budget | 100 / 100 / 25 / 25 | 12.62% | $0.096946 | sufficient for the reference path |

Only joint stopping/VOI and sufficient budget change the `listing_mismatch` decision. Sparse
posterior prediction changes uncertainty diagnostics (for example 0.8845 at 50 reviews versus an
empirical estimate below the 0.95 gate) but does not change the R=10 terminal decisions. Active
CVaR happens to improve net cost by stopping `product_quality` early; this four-workload result is
not evidence that the threshold generalizes and the cap remains disabled by default.

## 11. Failure cases and whether complexity was justified

- **R=5 regression.** Joint pays for 50 `listing_mismatch` reviews and still keeps the original,
  making the four-workload net $0.000991 worse than adaptive+risk.
- **Unproductive `product_quality` profiling.** Joint consumes 100 reviews at R=10 and 150 at
  R>=50, but the reorder never passes the policy. At R=10 that loses $0.002958 relative to stopping
  at 25.
- **Horizon sensitivity.** The benefit depends on a reuse estimate that may be wrong. The
  controller does not learn future reuse.
- **Monte Carlo sensitivity.** A retained 500-draw run changed two routes. The canonical run was
  raised to 2,000 draws to match the existing risk gate; deterministic seeds make this auditable,
  but simulation error remains.
- **Approximate EVSI.** Root-n contraction is deliberately simple. It can over- or under-value
  another checkpoint for rare correlated predicates.

Was the added complexity justified? **Conditionally.** At R=10 the improvement over adaptive+risk
is only $0.000434 across four workloads; at R=5 it is worse. At R=25–250 the recovered
`listing_mismatch` saving outweighs extra profiling and the advantage becomes material. The
diagnostic value is clear—the controller resolves the original logical conflict—but four related
workloads do not justify replacing the simpler policy universally.

## 12. Limitations and future experiments

- Four related three-predicate workloads on one dataset; results are exploratory.
- Only executed live orders can be scored. The globally cheapest permutation is unknown.
- Historical live costs are retrospective and model responses varied across repeats.
- Sparse posterior draws do not model Jev response variability or distribution shift.
- The pattern prior and root-n projection were not calibrated on an independent workload set.
- Reuse horizons are assumed, not observed.
- The active CVaR result is post-development exploratory evidence, not a tuned default.
- A live sequential experiment is still needed to measure actual cost avoidance.

Recommended next experiment: pre-register the 2,000-draw policy and VOI approximation on new
predicates/category data, then run live sequential profiling with exact checkpoint authorization
and repeated identical-order executions for response-variability calibration.

## 13. Reproduction

```powershell
semanticflow joint-risk-benchmark `
  --optimizer-config configs/optimizer.milestone_4.yaml `
  --output-dir results/milestone_4/<new-directory>
```

The command has no provider argument, refuses to overwrite its output directory, and records
`live_jev_calls: 0` and `paid_api_calls: 0`.
