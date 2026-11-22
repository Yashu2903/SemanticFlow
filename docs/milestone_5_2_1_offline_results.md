# SemanticFlow Milestone 5.2.1 Offline Results

## Central result

The legacy M5.2 rule needs 153 positive references and 389 representative records even with zero errors. The proposed 60-record pilot is mathematically incapable of certification. The nonredundant prospective rule lowers the limits but remains infeasible at 60 records.

Feasibility-aware stopping avoids 211 historical M5.2 profiling requests and $0.003208002 before acquisition. This is cost containment relative to M5.2 profiling, not a deployment improvement over immediate zero-cost fallback.

## Banking77

At the retrospective 0.75% positive rate, 153 positives require about 20,400 representative records in expectation and about $0.698917. The 60-record proposal cannot qualify it. Its known quality failure remains unchanged.

## Beauty

At the retrospective 5.5% rate, 153 positives require about 2,782 records in expectation and about $0.094631. That may be mathematically possible, but is not supported as an economical prospective selection-time decision.

## Active sampling

Historical active and random policies used different evidence mixes, so raw request reductions cannot identify an active-sampling effect. Under identical feasibility stopping both acquire zero records. Active discovery therefore shows no incremental information-efficiency benefit once impossibility is known; this is a negative finding.

| Controlled policy | Requests | Profiling cost | Representative | Discovery |
|---|---:|---:|---:|---:|
| random_fixed_budget | 1512 | $0.022920786 | 720 | 0 |
| active_fixed_budget | 1512 | $0.022920786 | 540 | 180 |
| random_identical_stopping | 211 | $0.003216906 | 93 | 0 |
| active_identical_stopping | 212 | $0.003212160 | 69 | 24 |
| random_feasibility_stop | 0 | $0.000000000 | 0 | 0 |
| active_feasibility_stop | 0 | $0.000000000 | 0 | 0 |

## Economics

All policies are recomputed at R=1,5,10,25,50,100,250,1000. Economic-only benefits from the unsafe Banking77 batch are separated from quality-admissible benefit. Immediate `existing_adaptive` fallback remains net zero; any policy that profiles and selects nothing is economically worse.

| Policy | R | Profiling cost | Economic-only net | Quality-admissible net | Unsafe |
|---|---:|---:|---:|---:|---:|
| existing_adaptive | 10 | $0.000000000 | $0.000000000 | $0.000000000 | 0 |
| existing_adaptive | 1000 | $0.000000000 | $0.000000000 | $0.000000000 | 0 |
| full_m52 | 10 | $0.003208002 | $-0.003208002 | $-0.003208002 | 0 |
| full_m52 | 1000 | $0.003208002 | $-0.003208002 | $-0.003208002 | 0 |
| m51_fixed_sample | 10 | $0.028590996 | $-0.022954176 | $-0.028590996 | 1 |
| m51_fixed_sample | 1000 | $0.028590996 | $0.535091004 | $-0.028590996 | 1 |

## Conclusion

M5.2.1 solves stop-before-spend feasibility detection. It does not solve rare-positive formal certification. The central obstacle is representative positive support at a cost that execution savings can amortize.
