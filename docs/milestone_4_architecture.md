# Milestone 4 architecture

Milestone 4 extends the Milestone 3 planner with uncertainty-aware plan selection. It does not
replace the independence estimator, the correlation-aware estimator, confidence-aware selection,
counterfactual replay, profiling economics, or adaptive profiling. Those components remain
independently selectable. Offline commands never construct a provider and must not be used to
issue Jev requests.

## Module map (`src/semanticflow/optimizer/`)

| Module | Responsibility |
|---|---|
| `bootstrap.py` | Paired review-row bootstrap. `collect_paired_bootstrap` resamples complete rows, refits every candidate order on the same draw, and records cost vectors. `paired_bootstrap` remains the Milestone 3 summary over those draws. |
| `regret.py` | Expected regret versus the cheapest modeled candidate on each resample, and upper-tail CVaR of loss versus the original order. |
| `uncertainty.py` | Cost-distribution summaries and labeled uncertainty sources. Model-response variability is a sensitivity analysis over repeated traces, not a generative noise model. |
| `disagreement.py` | Independence versus correlation-aware comparison on aligned bootstrap draws. |
| `risk.py` | `select_risk_aware_plan`: configurable `risk_aware` / `risk_aware_robust` policy with machine-readable `KEEP_ORIGINAL`, `ADOPT_REORDER`, `SKIP_PROFILING`, and `REUSE_EXISTING_PLAN` decisions. |
| `risk_benchmark.py` | Retrospective four-workload benchmark and ablations over stored Milestone 3 live traces. |
| `planner.py` | Unchanged exhaustive ranking, plus the two new selection policies. |
| `economics.py` | Unchanged forward-looking net-benefit and adaptive-start gates, invoked before new paid profiling. |
| `adaptive.py` | Historical independent sequential-profiling controller, retained for compatibility and ablation. |
| `joint.py` | Joint checkpoint controller: sparse posterior-predictive uncertainty, existing risk-policy evaluation, approximate value of additional profiling, budget enforcement, and reproducible traces. Its checkpoint API receives only the revealed prefix. |
| `joint_benchmark.py` | Strict four-workload sequential replay, reuse-horizon economics, controlled ablations, and held-out scoring after the controller terminates. |
| `voi_validation.py` | Offline next-checkpoint calibration, structured VOI diagnostics, sensitivity sweeps, reuse-horizon traces, and canonical-artifact integrity checks. |
| `profiling_splits.py` | Milestone 4.5 deterministic review-level profiling folds shared across workloads. |
| `m45_calibration.py` | Milestone 4.5 profiling-only calibration, grid tuning, config freeze, and retrospective scoring. |

Existing Milestone 1–3 modules are not duplicated. Milestone 4.5 does not replace these modules;
see [Milestone 4.5 architecture](milestone_4_5_architecture.md).

## Joint adaptive-risk extension

The follow-up controller coordinates profiling and selection instead of applying the risk gate only
after the adaptive controller has stopped. Its actions are `ADOPT_REORDER`, `PROFILE_MORE`,
`KEEP_ORIGINAL`, `SKIP_PROFILING`, and `REUSE_EXISTING_PLAN`.

At a checkpoint it:

1. receives only the nested profiling prefix revealed so far;
2. recalculates marginal, joint, conditional, usage, and candidate-cost evidence;
3. calls the existing risk-aware selector;
4. if adoption fails, projects the next evidence state from current posterior-predictive draws;
5. continues only when reuse-weighted expected decision-loss reduction exceeds incremental
   profiling cost and budgets allow it.

The offline harness owns the complete stored matrix but reveals another prefix only after
`PROFILE_MORE`. Held-out live costs are joined after the terminal action. A no-lookahead test
changes every future row and requires the current checkpoint trace to remain identical.

### Sparse posterior-predictive approximation

The empirical paired bootstrap remains available and unchanged. The joint controller adds a
Dirichlet distribution over complete binary outcome patterns, with prior mass centered on the
product of Jeffreys-smoothed marginal means. This prevents 0/25 from being represented as certainly
zero. Predictive draws are explicitly labeled simulations, not Jev observations.

The approximation represents selectivity, observed joint-pattern/correlation uncertainty, and
candidate cost uncertainty. It does not identify Jev response variability or future distribution
shift. Usage for an unseen pattern backs off to all currently observed rows.

### Value of additional profiling

For current decision `d_n`, modeled loss is its cost minus the cheapest modeled candidate on a
predictive scenario. The projected next-checkpoint distribution updates its center toward a current
predictive draw and contracts dispersion by `(n / n_next) ** 0.5`. The existing risk gate is rerun.

```text
VOI = expected_executions
      * E[current decision loss - projected next decision loss]
      - incremental profiling cost
```

This is an auditable EVSI approximation using current evidence only, not an exact Bayesian
pre-posterior calculation.

The validation follow-up adds an optional finite-scenario decision rule. `point_estimate`
reproduces the original controller. `lower_confidence_bound` subtracts a one-sided normal Monte
Carlo margin from per-execution EVSI before applying reuse and checkpoint cost. The bound addresses
only error from the finite set of predictive scenarios; it does not claim coverage for posterior
misspecification, contraction error, or Jev response drift. Both values are recorded.

## Uncertainty model

Sampling unit: a complete profiling review. All predicate outcomes, token counts, and latencies
travel with the row. Predicate columns are never resampled independently. Default iterations are
2,000 with seed 3031 (`configs/optimizer.milestone_4.yaml`). Unit tests use fewer iterations.

For each resample \(b = 1, \ldots, B\) and each candidate order \(P\):

1. Recalculate marginals, joints, and prefix conditionals on the resampled matrix.
2. Re-estimate every candidate with the chosen cardinality estimator.
3. Record \(\hat C_b(P)\) and the selected order under the Milestone 2 tie rule.

Because every plan is scored on the same resample, cost differences preserve dependence between
candidates. Per-predicate usage variation is part of the same draw; an average token cost is not
treated as certain.

Reported summaries for each plan: mean, median, sample standard deviation, 5th and 95th
percentiles, a configurable percentile interval, expected saving versus the original order,
\(P_{\text{bootstrap}}(\hat C(P) < \hat C(P_{\text{original}}))\), expected regret, and CVaR of
downside loss.

**Interpretation.** \(P_{\text{bootstrap}}(\text{plan cheaper}) = 0.95\) is a resampling diagnostic
over the 200-review profile. It is not a calibrated 95% probability that the plan will be cheaper
in production.

### Uncertainty sources

| Source | Representation |
|---|---|
| Sampling across reviews | Paired bootstrap of complete rows |
| Selectivity estimation | Each resample refits the estimator; this is profile sampling variability, not a posterior over a super-population |
| Per-request cost and latency | Token and latency observations resampled with the row |
| Model-response variability | Sensitivity analysis from repeated live traces only. No flip distribution is fitted from a handful of threshold crossings. Sources are not assumed independent. |

## Expected regret and downside risk

Regret of a selected plan \(P\) relative to the modeled candidate set on resample \(b\):

\[
\operatorname{Regret}_b(P)=\hat C_b(P)-\min_j \hat C_b(P_j)
\]

This is not regret against the unknown true optimum or against unexecuted permutations.

Downside loss versus the original order:

\[
L_b(P)=\hat C_b(P)-\hat C_b(P_{\text{original}})
\]

Positive \(L\) means the candidate is more expensive. CVaR at level \(\alpha\) (default 0.95) is
the mean of draws at or above the \(\alpha\) quantile of \(L\) (upper tail). The metric is always
reported. Hard gates `maximum_cvar_loss` and `maximum_expected_regret` default to `null`
(disabled) so that CVaR is not silently converted into a workload-specific threshold.

## Risk-aware selection

`selection_policy: risk_aware` considers, in order:

1. Optional pre-profiling economics (`statistics_state: none`): skip new profiling when even a
   structural savings bound cannot cover the first adaptive checkpoint.
2. Optional estimator disagreement (`risk_aware_robust` with
   `disagreement_policy: agree_original_is_worse`): keep the original unless both estimators
   select some reorder, i.e. both agree the original is worse. Agreement is not treated as proof
   of correctness.
3. Hard gates on each reorder: expected saving, relative saving (default 1%), bootstrap saving
   probability (default 0.95), optional CVaR and expected-regret caps.
4. Among plans that pass, select the minimum expected cost with the Milestone 2 tie rule
   (original order on ties).
5. Optional `apply_profiling_economics` on existing statistics: forward-looking net benefit
   excludes sunk profiling.

Default `apply_profiling_economics: false` because a completed profile is a sunk cost. Charging
it again would reject useful reorders whose per-execution saving is smaller than one profile,
which is a different question (whether to profile at all).

Machine-readable explanation:

```text
Decision: KEEP_ORIGINAL
Reason: Expected saving is smaller than the configured risk tolerance.
```

## Configuration

`configs/optimizer.milestone_4.yaml` documents every setting. Defaults:

| Setting | Default | Role |
|---|---|---|
| `planner.selection_policy` | `risk_aware` | Or `risk_aware_robust` |
| `planner.cardinality_estimator` | `correlation_aware` | Primary estimator; `independent` remains available |
| `bootstrap.iterations` | 2000 | Paired resamples |
| `bootstrap.seed` | 3031 | Deterministic `random.Random` |
| `risk.minimum_relative_saving` | 0.01 | Near-tie fallback, not a workload-specific constant |
| `risk.minimum_saving_probability` | 0.95 | Resampling diagnostic threshold |
| `risk.cvar_level` | 0.95 | Upper-tail CVaR of loss versus original |
| `risk.maximum_cvar_loss` | `null` | Disabled hard gate |
| `risk.apply_profiling_economics` | `false` | Do not re-charge existing statistics |
| `risk.disagreement_policy` | `agree_original_is_worse` | Used by `risk_aware_robust` |
| `economics.expected_executions` | `null` | Unknown reuse defaults to one execution |

Joint extension settings:

| Setting | Default | Role |
|---|---:|---|
| `joint.sparse_posterior_predictive` | `true` | Permit unseen outcome patterns in uncertainty draws |
| `joint.predictive_prior_strength` | `1.0` | Total Dirichlet base-measure mass |
| `joint.bootstrap_iterations` | `2000` | Match the existing risk gate's Monte Carlo resolution |
| `joint.seed` | `5051` | Deterministic predictive draws |
| `joint.voi_scenarios` | `32` | Pre-posterior decision-loss scenarios |
| `joint.minimum_voi_usd` | `0` | Continue only for strictly positive net VOI |
| `joint.uncertainty_contraction_exponent` | `0.5` | Root-n projection to the next checkpoint |
| `joint.voi_decision_rule` | `point_estimate` | `lower_confidence_bound` remains an experimental ablation |
| `joint.voi_confidence_level` | `0.95` | One-sided Monte Carlo lower-bound level |

## CLI

Both commands are synchronous, take no `--provider`, and never instantiate Jev:

```powershell
semanticflow risk-plan --experiment-dir <profile dir> `
  --optimizer-config configs/optimizer.milestone_4.yaml --output <new json>

semanticflow risk-plan --statistics-state none `
  --first-checkpoint-cost 0.01 --savings-upper-bound 0.0001

semanticflow risk-benchmark --output-dir results/milestone_4/<new directory>

semanticflow joint-risk-benchmark `
  --optimizer-config configs/optimizer.milestone_4.yaml `
  --output-dir results/milestone_4/<new directory>

semanticflow voi-validation `
  --optimizer-config configs/optimizer.milestone_4.yaml `
  --canonical-dir results/milestone_4/joint_adaptive_risk_20261106_2000 `
  --output-dir results/milestone_4/<new validation directory>

semanticflow calibrate-m45 `
  --optimizer-config configs/optimizer.milestone_4.yaml `
  --output-dir results/milestone_4_5/<new directory>
```

`risk-benchmark` reads stored Milestone 3 profiles and `analysis.json` files. It refuses to
overwrite an existing output directory. Observed costs are taken only from executed live orders.
`joint-risk-benchmark` has the same offline/overwrite guarantees and additionally records
checkpoint evidence, horizon-dependent routes, cold/warm economics, and controlled ablations.
`calibrate-m45` selects hyperparameters from profiling folds only, freezes a candidate config
before any held-out retrospective, and never constructs a provider.

## Data contracts

- Plan artifacts keep `m2-plan-v1` and add `selection_policy` values `risk_aware` /
  `risk_aware_robust` with `selection_explanation`.
- Bootstrap summaries remain `m3-paired-bootstrap-v1` with additional distribution fields.
- Risk explanations use `m4-risk-aware-v1`.
- Benchmark reports use `m4-offline-benchmark-v1`.
- Joint traces use `m4-joint-adaptive-risk-v1`.
- Joint benchmark reports use `m4-joint-adaptive-risk-benchmark-v1`.
- Historical Milestone 1–3 result directories are not rewritten.
