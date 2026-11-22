# SemanticFlow Milestone 5.2.1 Statistical Feasibility

## Exact construction

For all successes, the one-sided exact lower bound is `alpha^(1/n)`, giving `ceil(log(alpha)/log(q))`. For nonzero errors, M5.2.1 evaluates the exact binomial tail at the threshold. Qualification is attainable exactly when that tail is at most the allocated alpha. Log-space binomial calculations avoid small-alpha underflow.

| Policy | Recall alpha | Agreement alpha | Positive minimum | Record minimum |
|---|---:|---:|---:|---:|
| m52_legacy_strict | 0.000390625 | 0.000390625 | 153 | 389 |
| m5.2.1_nonredundant_v1 | 0.000781250 | 0.000781250 | 140 | 355 |
| m5.2.1_unequal_v1 | 0.000937500 | 0.000625000 | 136 | 366 |
| m52_legacy_strict | 0.012500000 | 0.012500000 | 86 | 217 |
| m52_legacy_strict | 0.003125000 | 0.003125000 | 113 | 286 |
| m52_legacy_strict | 0.001562500 | 0.001562500 | 126 | 320 |
| m52_legacy_strict | 0.003125000 | 0.003125000 | 113 | 286 |
| m52_legacy_strict | 0.000781250 | 0.000781250 | 140 | 355 |
| m52_legacy_strict | 0.001562500 | 0.001562500 | 126 | 320 |
| m52_legacy_strict | 0.000390625 | 0.000390625 | 153 | 389 |
| m52_legacy_strict | 0.000195313 | 0.000195313 | 167 | 423 |
| m5.2.1_nonredundant_v1 | 0.025000000 | 0.025000000 | 72 | 183 |
| m5.2.1_nonredundant_v1 | 0.006250000 | 0.006250000 | 99 | 252 |
| m5.2.1_nonredundant_v1 | 0.003125000 | 0.003125000 | 113 | 286 |
| m5.2.1_nonredundant_v1 | 0.006250000 | 0.006250000 | 99 | 252 |
| m5.2.1_nonredundant_v1 | 0.001562500 | 0.001562500 | 126 | 320 |
| m5.2.1_nonredundant_v1 | 0.003125000 | 0.003125000 | 113 | 286 |
| m5.2.1_nonredundant_v1 | 0.000781250 | 0.000781250 | 140 | 355 |
| m5.2.1_nonredundant_v1 | 0.000390625 | 0.000390625 | 153 | 389 |

## Redundancy

The implementation audit verified that recall and FNR share the positive denominator and are exact complements; agreement and changed-decision rate likewise share the complete-query denominator. Legacy M5.2 still allocates four bounds. The prospective nonredundant policy controls recall and agreement once and reports their complements, while retaining candidate and checkpoint multiplicity.

## Power and prevalence

Positive support is calculated exactly under a stated binomial prevalence. Wilson intervals show prevalence uncertainty, including nonzero upper bounds after zero observed positives. Certification power is exact conditional on a fixed denominator. Joint qualification uses a labeled dependence approximation rather than claiming independence.

### Legacy recall power at the best-case 153-positive denominator

| True recall | Certification probability | Maximum errors |
|---:|---:|---:|
| 0.950 | 0.000391 | 0 |
| 0.970 | 0.009464 | 0 |
| 0.980 | 0.045456 | 0 |
| 0.990 | 0.214874 | 0 |
| 0.995 | 0.464442 | 0 |
| 1.000 | 1.000000 | 0 |

## Validity limitations

Fixed checkpoint spending covers every predeclared inspection. Representative records must be sampled independently of outcomes. Active discovery is diagnostic only. Stage D is retrospective development evidence, and separate Jev inference is not human truth.
