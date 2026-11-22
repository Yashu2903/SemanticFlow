# SemanticFlow Milestone 5.2 Statistical Protocol

## Hypothesis and estimands

The hypothesis is that uncertainty-aware active qualification can reduce unsafe selection and wasted profiling while preserving economic opportunities. Primary quality estimands are representative-distribution recall, FNR, agreement, and changed-decision rate relative to separate Jev inference. They are consistency estimands, not human accuracy.

## Bounds and sequential validity

Wilson intervals are descriptive. Certification uses one-sided Clopper-Pearson exact bounds. Family alpha 0.050 is allocated in advance across candidates, checkpoints, and four simultaneous bounds by Bonferroni. The per-bound allocation is 0.00039063. This finite-family construction remains valid under outcome-dependent stopping because every possible inspection is included before replay.

## Qualification

At least 40 representative records and three representative positives remain necessary. Certification additionally requires lower recall and agreement bounds and upper FNR and change-rate bounds to cross the original thresholds. Zero positives leave recall undefined. Evidence may be QUALIFIED, REJECTED, INSUFFICIENT_SUPPORT, UNCERTAIN, UNSUPPORTED, or PROVIDER_FAILURE.

## Active sampling bias

Positive, threshold, disagreement, and cost-focused discovery samples are diagnostic only. Weighted certification is deliberately not implemented: the historical replay does not provide a sufficiently justified sequential unequal-probability design. Representative certification is isolated by record identity.

## Dependence and interpretation

Workloads share records, predicates, and provider measurements; offline policy rows are not independent confirmations. M5.1 Stage D is retrospective M5.2 development evidence. A future evaluation must freeze all rules and use new records before outcomes are available.
