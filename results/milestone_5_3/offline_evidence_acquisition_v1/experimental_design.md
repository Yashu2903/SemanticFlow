# M5.3 Experimental Design

## Questions and hypotheses

M5.3 compares random, uncertainty, positive-oriented, and cost-aware hybrid evidence acquisition for (A) separate-versus-batch disagreement discovery and (B) complete-query positive-reference discovery. H1–H4 are evaluated separately; a negative result is valid.

## Frozen evidence and eligibility

The primary pool is the M5.1 Stage D historical evaluation partition for the exact Pilot A predicate groups. A record is eligible only when all required single-predicate and exact multi-decision responses, request identities, tokens, and costs are present. The M5.1 selection partition is development-only. M5.2.3 representative evidence is a separate external historical check. Its discovery track is inventory-only because 29 of 30 optional follow-ups were never acquired.

## Visibility and policies

Deployable policy objects receive only record identity, workload, the acquired screen probability, and follow-up cost. Hidden separate/batch labels are revealed only after selection. Random sampling does not consume a free screen. End-to-end active replay screens candidates in hidden-outcome-independent windows of 5 and pays for every screen; conditional replay treats the same screens as sunk evidence.

The weighted hybrid is explicitly heuristic: disagreement data is too sparse for calibration. Its frozen score combines equal uncertainty and positive-likelihood weights and divides by observed compatible follow-up cost. The quota hybrid alternates uncertainty and positive rankings. No outcome labels tune either policy.

## Budgets and statistics

Request budgets are [5, 10, 20, 30, 50, 100]; dollar budgets are [0.0001, 0.00025, 0.0005, 0.001, 0.0015]. Results preserve 50 seed-level replicates. Comparisons use paired seed bootstrap intervals. Sparse historical event prevalence uses 95% Wilson intervals with exact numerators and denominators. Workloads are analyzed separately before the exploratory stratified aggregate.

## Limits

Separate Jev is not human ground truth. Previously inspected historical evaluation data is not an untouched holdout. Different physical groups are never substituted for the exact frozen group. Missing outcomes remain missing, and no provider code is used.
