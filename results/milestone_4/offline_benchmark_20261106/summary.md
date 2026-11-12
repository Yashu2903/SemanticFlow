# Milestone 4 offline benchmark

Exploratory retrospective evaluation on stored Milestone 3 live traces.
No new Jev requests. Bootstrap probabilities are resampling diagnostics.

Bootstrap iterations: 2000
Seed: 3031

## Major strategies (observed execution-cost saving percent)

| Strategy | PQ | LM | PE | MF | mean | median | worse | keep original |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 4 |
| independent | 0.36 | 5.44 | 0.77 | 44.17 | 12.68 | 3.10 | 0 | 0 |
| correlation_aware | 0.36 | 5.44 | 0.00 | 45.02 | 12.71 | 2.90 | 0 | 1 |
| correlation_aware_confidence | 0.00 | 5.44 | 0.00 | 45.02 | 12.62 | 2.72 | 0 | 2 |
| risk_aware | 0.00 | 5.44 | 0.00 | 45.02 | 12.62 | 2.72 | 0 | 2 |
