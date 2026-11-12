# Milestone 4.5 — Statistical Calibration and Optimizer Tuning

Exploratory offline study. No live or paid Jev calls. Artifacts:
`results/milestone_4_5/calibration_20261111/`. Frozen candidate:
`configs/optimizer.milestone_4_5_frozen.yaml`.

The four workloads previously informed Milestone 2–4 development. The 800-review live benchmark is
used only after configuration freeze and remains **exploratory, not confirmatory**.

## A. Motivation

Milestone 4 showed that predicate reordering can reduce inference cost, but more sophisticated
controllers did not reliably beat simpler adaptive profiling. At \(R=10\) (canonical joint
benchmark fingerprints, reproduced here):

| Strategy | Cold-start net |
|---|---:|
| Full correlation-aware (prior offline) | \$0.075334 |
| Full confidence-aware (prior offline) | \$0.074877 |
| Existing adaptive profiling | \$0.103387 |
| Adaptive + risk-aware | \$0.096513 |
| Joint adaptive-risk | \$0.096946 |
| Experimental VOI LCB joint | \$0.099904 (unstable under scenario count) |

Better cardinality estimates and richer uncertainty models often improved *predictions* without
improving *decisions*: risk gates rejected useful near-ties, VOI sometimes paid for profiling that
did not change the plan, and small profiling samples left ranking uncertainty large. Milestone 4.5
asks whether calibration and constrained tuning can improve accuracy, stability, and net economics
without forcing the complex strategy to win.

## B. Calibration methodology

### Data

Each of `product_quality`, `listing_mismatch`, `positive_experience`, and `mixed_feedback` has a
200-review profiling matrix with complete predicate outcomes and an 800-review held-out live
partition. All four share the same profiling review IDs.

### Profiling-only protocol

1. Deterministic review-level K-fold splits (`fold_seed=7073`, `folds=4` for calibration metrics;
   expensive joint grid uses `folds=2`).
2. Train fold fits estimators / controllers; validation fold scores exact matrix replay costs for
   all permutations (profiling validation ≠ live evaluation).
3. Nested boundary: hyperparameters chosen only from profiling-validation cold-start net at
   \(R=10\); held-out live costs are joined only after freeze.
4. Cross-workload partitions are asserted identical to prevent review-level leakage.

### Selection rule

Maximize mean profiling-validation cold-start net; tie-break by leave-one-workload-out floor, then
lower complexity. Adopt tuned joint as **recommended default** only if margin over existing
adaptive exceeds \$5e-5 and leave-one-out mean stays competitive. Otherwise keep
`existing_adaptive`.

## C. Cardinality estimation

On 4×4 workload-fold cells, correlation-aware estimation beat independence on both ranking
concordance and selection regret for every tested prior:

| Prior | Corr. concordance | Ind. concordance | Corr. regret (tokens) | Ind. regret (tokens) | Corr. wins (regret) |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.921 | 0.904 | 37.9 | 84.8 | 16/16 |
| 1.0 | 0.921 | 0.904 | 37.9 | 84.8 | 16/16 |
| 2.0 | 0.917 | 0.904 | 64.4 | 84.8 | 16/16 |

**Findings.** Correlation-aware estimation improves plan ranking and reduces selection regret on
profiling validation. Stronger smoothing (`prior=2.0`) worsens regret without helping concordance.
Priors 0.5 and 1.0 are indistinguishable on these folds; **prior 1.0 is retained** (simpler
continuity with Milestone 3–4 defaults). Absolute cost MAE improvements do not by themselves
justify full-matrix profiling at \(R=10\): charging all 200×predicates makes full estimators
strongly negative on cold-start net despite low validation regret.

## D. Bootstrap calibration

Paired review-row bootstrap coverage of exact validation-matrix savings (profiling-only):

| Iterations | Emp. 50% | Emp. 80% | Emp. 90% | Emp. 95% | Mean modal frequency |
|---:|---:|---:|---:|---:|---:|
| 500 | 0.385 | 0.615 | 0.740 | 0.771 | 0.760 |
| 1000 | 0.396 | 0.625 | 0.729 | 0.771 | 0.768 |

Intervals are **systematically too narrow**, especially at 90–95%. Raising iterations from 500 to
1000 barely moves coverage. Modal selection frequency ~0.76 indicates frequent near-ties.
Interpretation unchanged: bootstrap frequencies remain **resampling diagnostics**, not calibrated
production probabilities. Frozen evaluation keeps 2000 iterations for continuity with Milestone 4.

## E. VOI calibration

Milestone 4 VOI validation (cited, not retuned on held-out outcomes) found:

- Root-n contraction is reasonable for some aggregate selectivity/cost stds, weaker for
  plan-ranking uncertainty and expected decision benefit.
- Finite-scenario LCB reduced profiling in one configuration but was sensitive to scenario count,
  checkpoints, and risk thresholds.
- Additional profiling often reduces uncertainty **without** changing the adopted plan; under an
  execution-cost objective that information has zero immediate decision value.

Milestone 4.5 therefore **did not promote** `lower_confidence_bound` or alternate contraction
exponents into the default. The frozen candidate keeps `voi_decision_rule=point_estimate` and
`uncertainty_contraction_exponent=0.5`. Grid search preferred shorter checkpoints `(25,50,100)`,
which limits how far VOI can push profiling before a stop.

## F. Adaptive profiling

On profiling-validation folds (\(R=10\)), adaptive was the best cold-start policy among the
baselines scored in-calibration:

| Policy | Mean cold net | Mean profiling \$ | Mean val. regret \$ |
|---|---:|---:|---:|
| fixed | 0.000000 | 0.000000 | 0.000170 |
| adaptive | **0.000688** | 0.000980 | 0.000003 |
| joint (baseline settings) | 0.000581 | 0.000980 | 0.000014 |
| correlation_full | −0.004322 | 0.006006 | 0.000002 |
| independent_full | −0.004341 | 0.006006 | 0.000003 |
| confidence / risk_aware (full profile) | −0.004334 | 0.006006 | 0.000003 |

Adaptive stops early enough that profiling cost is ~6× lower than full profiling while still
capturing nearly all available validation savings (tiny regret). Median adaptive net is slightly
negative: gains concentrate on some folds/workloads (`mixed_feedback` leave-one-out hurts).

## G. Risk-aware selection

Full-profile confidence and risk-aware policies match each other on these folds and are dominated
by adaptive on cold-start net because they pay for complete profiling. Risk gates do reduce
adoption relative to unconstrained correlation ranking on the live retrospective
(`product_quality` / `positive_experience` stay original under joint/risk), which prevents some
weak reorderings but also forgoes adaptive’s cheap partial-profile wins. Hard CVaR/regret caps
were not introduced: profiling-only evidence did not justify workload-specific downside thresholds.

## H. Economic performance (frozen retrospective)

Held-out live scoring after freeze (exploratory). Cold-start nets:

| R | Existing adaptive | Adaptive+risk | Joint | Tuned joint |
|---:|---:|---:|---:|---:|
| 1 | 0.006757 | 0.006069 | 0.006069 | 0.006069 |
| 5 | 0.049703 | 0.046266 | 0.045275 | 0.045275 |
| 10 | **0.103387** | 0.096513 | 0.096946 | 0.096946 |
| 25 | 0.264438 | 0.247252 | 0.257313 | 0.257313 |
| 50 | 0.530880 | 0.496507 | 0.522589 | 0.524590 |
| 100 | 1.062678 | 0.993933 | 1.057144 | 1.059145 |
| 250 | 2.664207 | 2.492343 | 2.660809 | 2.662810 |
| 1000 | 10.708761 | 10.663128 | 10.679134 | 10.681135 |

At \(R=10\), existing adaptive remains best. Tuned joint matches baseline joint at low \(R\) and
slightly reduces profiling cost versus joint at high \(R\) (shorter schedule), but never overtakes
adaptive on aggregate cold-start net. Warm-start economics (sunk profiling) favor whichever plan
was selected; adaptive’s earlier stop still leaves competitive execution savings.

Actions at \(R=10\): adaptive reorders all four; joint/tuned keep original on `product_quality`
and `positive_experience`, adopt on `listing_mismatch` and `mixed_feedback`.

## I. Ablations

Profiling-validation ablations (components as policies):

1. **Baseline statistics + fixed order** — zero profiling, highest validation regret.
2. **Calibrated cardinality (full profile independent / correlation)** — lowest regret, worst
   cold-start net (profiling dominates at \(R=10\)).
3. **Bootstrap / confidence / risk on full profile** — same economic failure mode as (2).
4. **Adaptive stopping only** — best mean cold-start net; recommended default.
5. **Joint VOI + risk** — slightly worse than adaptive on profiling folds; recovers
   `listing_mismatch` on live retrospective but over-profiles relative to adaptive.
6. **Tuned joint** — best among joint grid on profiling folds, still below adaptive
   (\(\Delta \approx -\$2.4\times 10^{-5}\)); not adopted as default.

Combining better cardinality with risk gates without an early-stop controller is actively harmful
under cold-start accounting at modest reuse.

## J. Failure analysis

- **`product_quality`**: joint/tuned keep original; adaptive still reorders from a cheap prefix.
  Risk/VOI spend or caution does not pay at \(R=10\).
- **`listing_mismatch`**: joint correctly continues/adopts the valuable reorder; adaptive also
  finds it with less process complexity.
- **`positive_experience`**: near-tied rankings; risk keeps original; adaptive’s small saving is
  fragile across folds.
- **`mixed_feedback`**: strong conditionals make correlation valuable, but leave-one-out means
  without this workload turn cold-start nets negative for adaptive/joint — aggregate success is
  workload-concentrated.
- **VOI sensitivity**: scenario-count fragility of LCB (Milestone 4) argues against promoting
  that rule.
- **Small samples**: bootstrap undercoverage and ~0.76 modal frequency show residual ranking
  uncertainty after short prefixes.

## K. Limitations

- Only four related workloads on one SemCEB review sample.
- Shared underlying reviews; cross-workload dependence remains.
- Repeated exploratory analysis of the same benchmark (not an untouched confirmatory test).
- Limited statistical power; no formal multiple-testing control.
- Missing live executions for some candidate orders (retrospective marks those unavailable;
  estimates are never substituted for measured costs).
- Model-response variability is not calibrated from generative noise.
- No independent human ground-truth labels for semantic predicates.
- Tuning grid was intentionally small to avoid overfit and keep compute tractable.

## L. Next experiments (not executed)

1. Independent workloads / domains with fresh profiling and live traces.
2. Prospective live evaluation of frozen vs adaptive defaults at pre-registered horizons.
3. Richer cost-interval calibration (e.g., conformal or variance inflation) without changing the
   planner architecture.
4. Human-labeled predicate audits for a subset of reviews.
5. Larger scenario-count stability study for VOI **before** any LCB default change.

Do not start Milestone 5 from these exploratory numbers alone.

## Configuration freeze

| Parameter | Frozen value | Justification |
|---|---|---|
| `correlation.prior_strength` | 1.0 | Best/tied ranking; continuity |
| `risk.minimum_saving_probability` | 0.95 | Grid-competitive; Milestone 4 default |
| `risk.minimum_relative_saving` | 0.01 | Near-tie guard |
| `adaptive.checkpoints` | 25 → 50 → 100 | Best joint-grid schedule; less over-profiling |
| `joint.voi_decision_rule` | `point_estimate` | LCB not justified |
| `joint.uncertainty_contraction_exponent` | 0.5 | Retained root-n |
| `joint.voi_scenarios` / bootstrap iterations | 32 / 2000 | Restored full fidelity after cheap tuning |
| **Recommended default** | `existing_adaptive` | Profiling-validation margin not cleared |
| **Experimental strategy** | `tuned_joint_adaptive_risk` | Frozen for retrospective comparison |

Config fingerprint:
`c00fadea786bd2075739576501460f51f0997172caccb5e133a0d2f0db3447c7`.

## Research answers

1. **Prediction accuracy** — Yes for correlation-aware vs independent ranking/regret; prior 2.0
   hurts.
2. **Plan-ranking accuracy** — Yes for correlation-aware (16/16 regret wins).
3. **VOI stability** — Not improved enough to change defaults; LCB remains experimental.
4. **Unnecessary profiling** — Tuned schedule shortens joint profiling vs the long schedule; adaptive
   remains leanest among useful policies.
5. **Risk-aware regret** — Low validation regret when a full profile is paid for, but cold-start net
   suffers; risk alone does not beat adaptive.
6. **Net economics** — Profiling-validation and held-out aggregates still favor existing adaptive.
7. **Robustness across splits** — Adaptive leads on mean but median fold net is negative; gains are
   uneven.
8. **Complexity justified?** — Not for replacing the default. Tuned joint is retained as an
   experimental artifact only.
9. **Replace default?** — **No.** Keep existing adaptive as recommended default.
10. **Unproven** — Generalization beyond these four workloads; calibrated production probabilities;
    prospective live confirmation; VOI contraction as a decision-quality model.
