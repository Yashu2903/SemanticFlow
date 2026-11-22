# SemanticFlow Milestone 5.2 Offline Results

## Status

All requested subsystems are implemented and exercised against stored real Jev evidence. No provider calls occurred. Results are retrospective development diagnostics.

## Policy comparison

| Policy | Unsafe | Missed opportunities | Selected | Profiling requests | Profiling cost | Net benefit |
|---|---:|---:|---:|---:|---:|---:|
| existing_adaptive | 0 | 1 | 0 | 0 | $0.000000000 | $0.000000000 |
| m51_fixed_sample | 1 | 1 | 1 | 1885 | $0.028590996 | $0.535091004 |
| random_original_gate | 0 | 1 | 0 | 1512 | $0.022920786 | $-0.022920786 |
| random_uncertainty | 0 | 1 | 0 | 1512 | $0.022920786 | $-0.022920786 |
| active_original_gate | 0 | 1 | 0 | 1885 | $0.028590996 | $-0.028590996 |
| active_uncertainty | 0 | 1 | 0 | 1885 | $0.028590996 | $-0.028590996 |
| cost_aware_uncertainty | 0 | 1 | 0 | 473 | $0.007150290 | $-0.007150290 |
| risk_budget | 0 | 1 | 0 | 1885 | $0.028590996 | $-0.028590996 |
| cost_only | 1 | 0 | 2 | 0 | $0.000000000 | $0.843654000 |
| full_m52 | 0 | 1 | 0 | 211 | $0.003208002 | $-0.003208002 |


## Banking77 replay

The historical selection evidence contained 3 positives and 0 false negatives. The old point gate passed: True. The strict state is **UNCERTAIN** because 3/3 cannot establish 95% population recall under the frozen family-wise sequential rule. The controller therefore cannot certify the known failed batch; depending on reuse horizon it profiles further or falls back when information cannot repay its cost.

## Beauty replay

Selection evidence had 8 positives and 1 false negatives. Both the old point gate and strict rule fail on permitted selection evidence. Active discovery can expose mechanisms and prioritize records, but cannot turn biased enrichment into representative certification. The later Stage D pass is not leaked back into the decision.

## Sparse positives

Four full-batch workloads have zero held-out complete-query positives. Recall remains undefined, strict qualification returns insufficient support, and the controller safely falls back or stops when further evidence is uneconomic.

## Main finding

The strict procedure removes unsafe historical certification but often abstains. This is statistically defensible and economically negative at short horizons. The offline evidence therefore does not establish that M5.2 improves deployment utility; it establishes a leakage-safe implementation and identifies the positive-support burden for a fresh pilot.

## Limitations

Stage D is known retrospective evidence, no human labels exist, weighted active certification is not justified, and the measured operator catalog bounds replay coverage. Optimization CPU cost is unmetered.
