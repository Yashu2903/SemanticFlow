# Milestone 4 VOI validation and calibration

Exploratory retrospective validation of the joint adaptive-risk controller. No Jev requests were
made. The four workloads previously informed development, so this is not a confirmatory test and
does not establish statistical significance.

## Evidence labels

| Label | Meaning |
|---|---|
| Observed Jev data | Stored profiling outcomes or costs of orders that were historically executed |
| Offline replay | A deterministic prefix of the stored profiling matrix, revealed only after `PROFILE_MORE` |
| Statistical simulation | Sparse posterior-predictive or bootstrap draws; never treated as Jev responses |
| Estimated cost | Optimizer prediction for a candidate order |
| Measured cost | Mean historical live cost, available only for an executed order |

Canonical point-estimate controller:
`results/milestone_4/joint_adaptive_risk_20261106_2000/`.

Behavior-neutral validation:
`results/milestone_4/voi_validation_current_20261106_v2/`.

Experimental lower-confidence-bound controller:
`results/milestone_4/joint_adaptive_risk_voi_lcb_20261106_2000/` and
`results/milestone_4/voi_validation_corrected_20261106/`.
Its exact settings are preserved in `configs/optimizer.milestone_4_voi_lcb.yaml`; the benchmark
directory includes `config_manifest.json` explaining why the original run's recorded config path
names the main file that was subsequently restored to the point-estimate default.

## Baseline reproduction

Before implementation changes, the canonical benchmark was rerun into
`results/milestone_4/voi_validation_baseline_reproduction_20261106/`. After removing only
`created_at`, the reproduced `benchmark.json` was equal to the canonical JSON. Both report
`live_jev_calls: 0` and `paid_api_calls: 0`.

The canonical aggregate cold-start results were reproduced exactly:

| R | Existing adaptive+risk | Point-estimate joint |
|---:|---:|---:|
| 1 | $0.006069 | $0.006069 |
| 5 | $0.046266 | $0.045275 |
| 10 | $0.096513 | $0.096946 |
| 25 | $0.247252 | $0.257313 |
| 50 | $0.496507 | $0.522589 |
| 100 | $0.993933 | $1.057144 |
| 250 | $2.492343 | $2.660809 |
| 1,000 | $10.663128 | $10.679134 |

## Controller objective and diagnostic instrumentation

The current action is selected by the configured risk policy. For predictive future evidence
(D'), the controller estimates policy-constrained sample information value as

\[
\widehat{EVSI} = E_{D'}[L(a_D)-L(a_{D,D'})],
\]

where (L) is expected cost regret against the cheapest modeled candidate on the same posterior
draw. The economic continuation value is

\[
\widehat{NetVOI}=R\widehat{EVSI}-C_{profile,next}.
\]

This is not unconstrained expected-cost EVSI: the future action is still constrained by the
confidence, expected-saving, regret, and CVaR policy. The implementation now records that
distinction explicitly.

Every checkpoint now emits structured fields for candidate cost distributions, plan-selection
frequencies, expected regret, downside risk, probability of an action change, benefit conditional
on a change, benefit without a change, incremental cost, gross VOI, net VOI, Monte Carlo error,
the final action, and reason codes. Under this execution-cost objective, information that changes
confidence but not the selected order has zero immediate execution-cost value. That quantity is
reported separately instead of being silently credited.

## Predicted versus realized uncertainty contraction

The next stored profiling block was revealed only after each current-prefix prediction. Root-n is
reasonable in aggregate for scalar marginal, conditional, and cost standard deviations, but not
uniformly reliable for individual sparse quantities.

| Quantity | Comparisons | Median effective exponent | Mean absolute next-SD error | Mean absolute relative error |
|---|---:|---:|---:|---:|
| Marginal selectivity | 48 | 0.526 | 0.002019 | 6.74% |
| Pairwise conditional selectivity | 96 | 0.471 | 0.013165 | 12.08% |
| Candidate cost | 96 | 0.513 | $0.0000703 | 8.40% |

Individual effective exponents ranged from -0.222 to 0.968 for marginals, -0.299 to 1.934 for
conditionals, and -0.670 to 1.009 for cost. Sparse conditional uncertainty therefore does not
obey a universal root-n law even though the pooled medians are near 0.5.

Plan-ranking uncertainty was substantially less well predicted: mean absolute error was 0.1661
and maximum error was 0.3310 across ten projected transitions. Projected selected-plan regret
standard deviation had mean absolute error $0.0000640 and maximum error $0.0002506. The next-block
profiling-cost estimate was comparatively accurate (mean absolute error $0.0000251; maximum
$0.0000397).

For the ten transitions where both compared orders had historical live costs, the one-step
decision-benefit prediction had mean absolute error $0.0001541 per execution, median $0.0000278,
and maximum $0.0007787. A single stored future path is not the expectation over all possible
future samples, so these are calibration diagnostics rather than unbiased EVSI error estimates.

## Product-quality overprofiling

At R=10 the point-estimate controller followed this route:

| Reviews | Candidate | P(candidate cheaper) | Point EVSI/run | Point net VOI | Result after reveal |
|---:|---|---:|---:|---:|---|
| 25 | `product_defect, poor_quality, dissatisfaction` | 0.5305 | $0.000164562 | +$0.000662902 | action unchanged |
| 50 | same | 0.7590 | $0.000289232 | +$0.000942004 | action unchanged |
| 100 | same | 0.7035 | $0.000055605 | -$0.001414107 | stop, keep original |

The controller expected a 18.75% chance of an order change after reviews 26-50 and 40.625% after
reviews 51-100. Neither change occurred. Confidence first increased, then fell; the additional
75 reviews cost $0.002957598 and produced no observable action or execution-cost benefit.

This was not primarily a prior-strength failure: strengths 0.25, 0.5, 1, 2, and 5 all produced
the same 100-review R=10 route in the controlled 500-draw sweep. Nor does the pooled contraction
audit justify replacing root-n with a workload-favorable exponent. Exponents 0.25 and 0.375 stop
at 25, while 0.5 through 0.75 profile to 100, but the cross-workload profiling-only calibration is
centered near 0.5.

The general problem was decision-making on a noisy 32-scenario point estimate. The point EVSI was
positive, but its finite-scenario uncertainty was large. One-sided 95% lower-bound R=10 net VOI
was -$0.000374748 at 25 reviews and -$0.000264789 at 50 reviews.

## Listing-mismatch recovery

| Reviews | Candidate | P(candidate cheaper) | Point EVSI/run | 95% lower EVSI/run | Lower-bound net VOI at R=10 | Action |
|---:|---|---:|---:|---:|---:|---|
| 25 | `value_complaint, description_mismatch, dissatisfaction` | 0.8000 | $0.000778705 | $0.000400158 | +$0.002995765 | profile to 50 |
| 50 | same | 0.8845 | $0.000389202 | $0.000232934 | +$0.000332829 | profile to 100 |
| 100 | same | 0.9910 | n/a | n/a | n/a | adopt reorder |

The correction therefore preserves the recovery without using future rows. It spends $0.003026898
more than the old 25-review stop and observes $0.000641823 saving per execution (5.44%) on the
historically executed reordered plan. The R=10 cold-start net is +$0.002385516 versus
-$0.001005816 for the previous adaptive+risk route.

This does not prove that 100 reviews is universally optimal. It shows that, under the stated
posterior and risk policy, the continuation decision remains positive after accounting for
finite-scenario Monte Carlo error.

## Evaluated correction and adoption decision

The controller retains the original point estimate and adds an optional continuation rule:

\[
NetVOI_{LCB}=R(\widehat{EVSI}-z_{0.95}SE(\widehat{EVSI}))-C_{profile,next}.
\]

The experimental policy continues only when this one-sided 95% lower bound exceeds
`minimum_voi_usd`. The bound covers finite predictive-scenario Monte Carlo error only. It does not
cover posterior misspecification, contraction error, Jev response drift, or live-cost variability.

This is a narrow option rather than a new optimizer. Sensitivity analysis did not justify adopting
it as the default: at 128 scenarios `product_quality` again profiles to 100, while at 8 and 16
scenarios `listing_mismatch` stops at 50 and loses the observed reorder. The repository therefore
retains `point_estimate` as the configured default and exposes the lower bound for explicit
ablation. New settings are:

```yaml
joint:
  voi_decision_rule: lower_confidence_bound
  voi_confidence_level: 0.95
```

The default offline validation command is:

```powershell
semanticflow voi-validation `
  --optimizer-config configs/optimizer.milestone_4.yaml `
  --canonical-dir results/milestone_4/joint_adaptive_risk_20261106_2000 `
  --output-dir results/milestone_4/<fresh-directory>
```

Use `configs/optimizer.milestone_4_voi_lcb.yaml` only to reproduce the experimental lower-bound
ablation.

## Reuse-horizon behavior

Experimental lower-bound routes at the canonical 32 scenarios (`reviews/action`):

| Workload | R=1 | R=5 | R=10 | R=25 | R=50 | R=100 | R=250 | R=1,000 |
|---|---|---|---|---|---|---|---|---|
| product_quality | 25/keep | 25/keep | 25/keep | 100/keep | 100/keep | 100/keep | 100/keep | 100/keep |
| listing_mismatch | 25/keep | 50/keep | 100/adopt | 100/adopt | 100/adopt | 100/adopt | 100/adopt | 100/adopt |
| positive_experience | 25/keep | 25/keep | 25/keep | 25/keep | 25/keep | 25/keep | 25/keep | 25/keep |
| mixed_feedback | 25/adopt | 25/adopt | 25/adopt | 25/adopt | 25/adopt | 25/adopt | 25/adopt | 25/adopt |

The sharp reduction in joint advantage at R=1,000 is not an arithmetic error. The independent
adaptive+risk baseline profiles `listing_mismatch` to 50 reviews at R=250 and rejects it, but at
R=1,000 profiles to 100 and adopts the same reorder as the joint controller. The execution-order
advantage disappears; only profiling-depth differences remain. A regression test now covers this
horizon-dependent adaptive depth.

## Experimental lower-bound economic results

### Cold start

| R | Existing adaptive+risk | Point joint | Lower-bound joint | Lower bound minus point |
|---:|---:|---:|---:|---:|
| 1 | $0.006069 | $0.006069 | $0.006069 | $0.000000 |
| 5 | $0.046266 | $0.045275 | $0.045275 | $0.000000 |
| 10 | $0.096513 | $0.096946 | $0.099904 | +$0.002958 |
| 25 | $0.247252 | $0.257313 | $0.257313 | $0.000000 |
| 50 | $0.496507 | $0.522589 | $0.524590 | +$0.002001 |
| 100 | $0.993933 | $1.057144 | $1.059145 | +$0.002001 |
| 250 | $2.492343 | $2.660809 | $2.662810 | +$0.002001 |
| 1,000 | $10.663128 | $10.679134 | $10.681135 | +$0.002001 |

At R=10, execution-only selections are unchanged from the point controller: `listing_mismatch`
and `mixed_feedback` are adopted; `product_quality` and `positive_experience` keep the original.
Mean/median execution saving remains 12.62%/2.72%; mean/maximum observed regret remains
$0.0000408/$0.0001177; two beneficial reorders are accepted, two are rejected, and no harmful
reorder is accepted. Profiling falls from 187.5 to 131.25 decisions per workload and from
$0.009964710 to $0.007007112 total.

At R=10 per workload:

| Workload | Reviews | Profiling cost | Saving/run | Observed regret | Cold net |
|---|---:|---:|---:|---:|---:|
| product_quality | 25 | $0.000982716 | $0 | $0.000045633 | -$0.000982716 |
| listing_mismatch | 100 | $0.004032714 | $0.000641823 | $0 | +$0.002385516 |
| positive_experience | 25 | $0.000991116 | $0 | $0.000117684 | -$0.000991116 |
| mixed_feedback | 25 | $0.001000566 | $0.010049277 | $0 | +$0.099492204 |

### Warm start

Historical profiling is sunk. At R=10 the corrected route's warm-start aggregate saving is
$0.106911, equal to full-profile confidence-aware selection because the same two reorderings are
executed. No historical profile is charged again. The adaptive costs above remain offline
estimates derived from stored calls, not measurements from a new live adaptive run.

The lower-bound variant does not maximize observed savings on this retrospective set. Existing
adaptive without the final risk gate reaches $0.103387 cold net at R=10, and full correlation-aware
or ordinary adaptive can be better at high reuse by accepting small beneficial reorders. Those
comparisons trade greater observed saving here for a less conservative risk policy.

## Sensitivity and ablations

For the point controller at R=10:

- Prior strengths 0.25-5 do not change any terminal route.
- `listing_mismatch` adopts at 100 for contraction exponents 0.25-0.75 and 8-128 VOI scenarios.
- `product_quality` stops at 25 for exponents 0.25/0.375 but profiles to 100 for 0.5-0.75.
- `product_quality` stops at 25 with 8/16 scenarios but profiles to 100 with 32/64/128, showing
  finite-scenario route instability below 32 rather than evidence for using fewer scenarios.
- Raising the confidence gate to 0.99 avoids `product_quality` profiling but loses the
  `listing_mismatch` recovery at 50 reviews.
- A zero CVaR-loss cap avoids the `product_quality` continuation and preserves
  `listing_mismatch`, but four related workloads do not justify turning that diagnostic into a
  universal hard gate.
- A $0.001 minimum net-VOI margin also separates the two cases, but is scale-dependent and was
  not adopted.
- A 150-decision budget stops `listing_mismatch` at 50; 300 decisions are sufficient for recovery.

For the experimental lower-bound rule at R=10:

- `product_quality` stops at 25 for contraction exponents 0.25-0.75, but scenario-count
  sensitivity is not stable: it profiles to 100 at 128 scenarios and stops at 25 for 8-64.
- `listing_mismatch` is recovered at 100 for 32, 64, and 128 scenarios, but is lost at 50 for 8
  and 16 scenarios.
- `listing_mismatch` is recovered for contraction exponents 0.375-0.75 but lost at 0.25.
- In the fixed 500-draw prior sweep, `listing_mismatch` is recovered for strengths 0.25-5;
  `product_quality` profiles to 100 for strengths 0.25-2 and stops at 25 only at strength 5.
  These routes must not be compared directly with the 2,000-draw canonical route because the
  prior family deliberately uses a common smaller diagnostic draw budget.
- A minimum net-VOI margin of $0.0005 loses `listing_mismatch`; a 0.99 saving-probability gate
  also loses it. A 300-decision budget remains the minimum tested budget that permits recovery.
- The zero-CVaR-loss ablation loses `listing_mismatch` under the lower-bound rule, illustrating
  that stacking conservative gates can be counterproductive.

The lower-bound variant therefore improves the canonical 32-scenario arithmetic but is not robust
to reasonable changes in scenario count and contraction. It is retained as an experimental option,
not selected as the validated default.

## Limitations

1. Four related workloads are insufficient to calibrate production probabilities or establish
   statistical significance.
2. The same workloads informed development; all comparisons are exploratory.
3. Only executed orders have observed live costs. The globally cheapest permutation is unknown.
4. One realized next block cannot identify the expectation over all possible future samples.
5. Jeffreys beta summaries are diagnostic; the controller uses a smoothed joint outcome-pattern
   posterior. Neither model represents future distribution shift.
6. Sparse posterior draws do not model Jev response variability. Stored repeats remain a
   sensitivity source, not a fitted response-noise model.
7. The lower confidence bound covers Monte Carlo scenario error, not structural uncertainty.
8. Runtime increases with the number of predictive scenarios because the actual risk gate is
   rerun for each projected evidence state.

## Verification

- Final point-policy replay: all aggregate metrics, cold/warm economic totals, and every
  workload/horizon route equal the canonical benchmark.
- Canonical integrity: all six SHA-256 hashes match the pre-change values.
- Full test suite: 226 passed in 14.22 seconds.
- Ruff: `ruff check .` passed.
- Patch whitespace: `git diff --check` passed; Git emitted only the repository's existing Windows
  LF-to-CRLF notices.
- Determinism: repeated synthetic controller tests pass, and the final 2,000-draw replay matches
  the canonical route and aggregate structures exactly.
- Paid/live Jev calls: zero in every new benchmark and validation artifact.

## Research conclusion

The point VOI approximation is directionally useful but not reliable as an unqualified scalar.
Root-n uncertainty contraction is adequate in pooled scalar summaries, while plan-ranking and
one-step decision-benefit predictions are materially noisier. At the canonical 32 scenarios, the
finite-scenario lower-bound variant removes the observed `product_quality` overprofiling at R=10,
preserves `listing_mismatch`, and improves total cold-start net saving without changing execution
orders or observed regret. Those gains are not robust across scenario count and contraction.

The additional diagnostic complexity is justified: the controller already simulates scenarios,
so reporting their standard error and decomposing value by action change is auditable. The evidence
does not justify the extra lower-bound gate as a default behavior change, so the point policy is
retained. A future confirmatory protocol should preregister policy settings, use new workload
families, execute all policy-selected orders, record repeated Jev responses, and compare predicted
EVSI with many independently sampled future profiling blocks.
