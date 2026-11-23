# Milestone 5.3.1 — Independent Live Jev Validation

## Completion status

Implementation, dry-run validation, live provider collection, reconciliation, and independent policy replay are complete. The frozen configuration and acquisition policies were not modified after authorization.

## Provider accounting and integrity

| Measure | Result |
|---|---:|
| Fully paired independent records | 300 |
| Physical Jev requests | 1,100 |
| HTTP attempts | 1,100 |
| Retries / ambiguous attempts | 0 / 0 |
| Input tokens | 384,086 |
| Output tokens | 45,600 |
| Token-priced cost | $0.016131612 |
| Raw durable provider responses | 1,100 |
| Unique response identities | 1,100 |

The actual cost was 4.07% above the rounded $0.0155 planning estimate and 2.15% above the $0.015792 summed action forecast, while remaining below the frozen $0.025 limit. Banking77 cost $0.005662272, All Beauty cost $0.004786404, and SMS Spam cost $0.005682936. Screens cost $0.004266570, remaining separate predicates $0.006897072, and complete multi-decision requests $0.004967970.

Independent reconciliation verified all 4,403 hash-linked events, all authorized action identities, raw responses before validation, token and cost sums, exact predicate groups, absence of duplicate scientific measurements, complete record pairing, and nine generated session artifact hashes. The terminal session state is `COMPLETE`.

## Observed evidence

| Pool | Records | Predicate disagreements | Query disagreements | Batch false negatives | Separate-query positives |
|---|---:|---:|---:|---:|---:|
| Banking77 | 100 | 5 | 2 | 0 | 2 |
| All Beauty | 100 | 2 | 0 | 0 | 10 |
| SMS Spam | 100 | 1 | 0 | 0 | 9 |
| Representative-random stratum | 150 | 4 | 0 | 0 | 6 |
| Lexical-enrichment stratum | 150 | 4 | 2 | 0 | 15 |

There were eight records with one named-predicate disagreement each. Both complete-query disagreements occurred in enriched Banking77 records and were batching false positives relative to separate Jev. No batching false negative was observed. The zero false-negative count has a 95% Wilson upper bound of approximately 1.26% over 300 records and does not establish a true zero rate. Separate Jev remains a comparison reference, not human ground truth; all eight cases are preserved in the human-review sheet.

## Equal 100-request end-to-end comparison

These pooled values describe the fixed 150/150 mixture only. They are not an unweighted population estimate.

| Policy | Confirmed records | Screens | Cost | Predicate disagreements | Query disagreements | Positives | Positives/$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | 27.08 | 0.00 | $0.001454494 | 0.58 | 0.20 | 1.92 | 1,320 |
| Uncertainty | 12.80 | 66.02 | $0.001448643 | 1.46 | 0.36 | 1.00 | 690 |
| Positive-oriented | 12.26 | 65.06 | $0.001436866 | 0.06 | 0.06 | 3.58 | 2,492 |
| Weighted hybrid | 12.96 | 67.00 | $0.001447732 | 1.40 | 0.32 | 2.50 | 1,727 |
| Quota hybrid | 12.80 | 66.02 | $0.001448643 | 1.46 | 0.36 | 1.00 | 690 |

Screening consumed approximately 64.5%–65.9% of active policies' end-to-end cost and reduced confirmations from about 27 records under direct random acquisition to 12–13. Despite that overhead, objective-specific gains survived total cost accounting.

### Paired differences from random at 100 requests

Intervals below are paired policy-seed bootstrap intervals over the same provider evidence. They describe policy-selection variability, not independent provider-response variability.

| Policy | Predicate disagreement difference | Query disagreement difference | Positive difference |
|---|---:|---:|---:|
| Uncertainty | +0.88 [0.54, 1.24] | +0.16 [-0.02, 0.32] | -0.92 [-1.44, -0.42] |
| Positive-oriented | -0.52 [-0.76, -0.30] | -0.14 [-0.28, -0.02] | +1.66 [1.04, 2.22] |
| Weighted hybrid | +0.82 [0.46, 1.18] | +0.12 [-0.04, 0.28] | +0.58 [-0.04, 1.18] |
| Quota hybrid | +0.88 [0.54, 1.24] | +0.16 [-0.02, 0.32] | -0.92 [-1.44, -0.42] |

At the matched $0.0015 end-to-end budget, conclusions were materially unchanged: positive-oriented gained 1.76 positives [1.14, 2.32], uncertainty gained 0.88 predicate disagreements [0.54, 1.24], and weighted hybrid gained 0.82 [0.46, 1.20]. Weighted-hybrid positive gain remained inconclusive at +0.60 [-0.02, 1.20].

## RQ1 — Positive acquisition

**Result: supported for positive-oriented sampling in both strata and for All Beauty and SMS; unsupported as a workload-universal claim.**

At 100 end-to-end requests, positive-oriented minus random found:

| Pool | Paired positive difference [95% interval] | Verdict |
|---|---:|---|
| Representative-random | +0.92 [0.44, 1.34] | Supported |
| Lexical enrichment | +2.00 [1.32, 2.64] | Supported |
| Banking77 | -0.12 [-0.38, 0.12] | Inconclusive |
| All Beauty | +3.38 [2.72, 4.02] | Supported |
| SMS Spam | +2.86 [2.14, 3.52] | Supported |

Positive-oriented achieved 1,651 positives per dollar in the representative stratum and 3,120 in enrichment, versus random's 971 and 1,741 respectively. Thus the positive-discovery improvement survives paid screening, but it does not generalize to the sparse-positive Banking77 workload.

## RQ2 — Disagreement acquisition

**Result: supported for predicate-disagreement discovery in Banking77, All Beauty, and both strata; contradicted for SMS. Complete-query evidence remains sparse, and false-negative discovery is inconclusive.**

At 100 end-to-end requests, uncertainty minus random found:

| Pool | Predicate difference [95% interval] | Query difference [95% interval] |
|---|---:|---:|
| Representative-random | +0.56 [0.22, 0.90] | 0.00 [0.00, 0.00] |
| Lexical enrichment | +1.02 [0.66, 1.38] | +0.48 [0.14, 0.78] |
| Banking77 | +1.44 [1.02, 1.88] | +0.46 [0.20, 0.74] |
| All Beauty | +0.80 [0.44, 1.14] | 0.00 [0.00, 0.00] |
| SMS Spam | -0.14 [-0.26, -0.04] | 0.00 [0.00, 0.00] |

Weighted hybrid also improved predicate discovery in the representative stratum (+0.58 [0.24, 0.90]) and enrichment (+0.84 [0.52, 1.18]). Its query-disagreement advantage in enrichment touched zero (+0.30 [0.00, 0.58]). Quota hybrid produced the same end-to-end order and results as uncertainty under this frozen replay, although their conditional-follow-up orders differed.

No policy can be credited with false-negative discovery because the complete evidence pool contained no batching false negatives. The only two query disagreements were false positives relative to separate Jev.

## RQ3 — Full acquisition economics

**Result: objective-specific gains survive total acquisition cost, but no policy dominates all objectives.**

- Positive-oriented more than doubled pooled positives per dollar relative to random while performing substantially worse for disagreements.
- Uncertainty and quota hybrid increased predicate disagreements per dollar from approximately 399 to 1,008 at the pooled 100-request checkpoint, while positive yield per dollar fell.
- Weighted hybrid increased predicate disagreements per dollar to approximately 967 and positives per dollar to 1,727, but its positive advantage over random was not statistically supported in the fixed pooled mixture.
- Conditional follow-up results were much stronger because screens were free: positive-oriented found 16 positives and weighted hybrid found six predicate disagreements at a 100-request follow-up budget. Those conditional results do not establish end-to-end economic superiority.

Across all eleven matched request and dollar checkpoints, uncertainty improved predicate discovery at nine checkpoints in enrichment but was worse for positives at ten. Positive-oriented improved positives at eight checkpoints in All Beauty and six in SMS, while it had no supported positive gain in Banking77. These differences demonstrate workload heterogeneity rather than one generally superior policy.

## Hybrid, uncertainty, and workload conclusions

- **Weighted hybrid versus random:** better predicate-disagreement discovery in both strata; better positive discovery in the representative stratum and in Beauty/SMS; pooled positive superiority remained inconclusive.
- **Quota hybrid versus random:** better disagreement discovery where uncertainty worked, but worse positive discovery. It did not provide the intended balanced end-to-end outcome in this experiment.
- **Uncertainty versus random:** clearly better for disagreement discovery in Banking77 and Beauty, but worse in SMS. It is not a workload-general strategy.
- **Positive-oriented versus random:** clearly better for positives in Beauty and SMS and both strata, but did not improve Banking77.

## Statistical limitations and recommendation

Only eight predicate disagreements, two query disagreements, zero false negatives, and 21 separate-query positives were independently observed. Fifty replay seeds reorder one fixed set of provider responses and must not be treated as 50 provider replications. The representative and enriched strata remain separate; pooled tables describe only the frozen mixture. Provider nondeterminism was not measured, and human classification of disagreement cases remains pending.

The results do not justify selecting one universal policy for production or proceeding directly to quality-aware M5.4 plan selection as though disagreement risk were calibrated. A narrowly scoped M5.3.2 is recommended first: human-review the eight cases, measure provider-repeatability on a predeclared subset, and acquire targeted independent complete-query disagreement/false-negative evidence. This is a research recommendation only; M5.3.2 and M5.4 were not started.
