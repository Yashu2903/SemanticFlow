# Milestone 4.5 calibration summary

Baseline reproduction: confirmed
Recommended default: existing_adaptive
Adopt tuned joint as default: False
Margin vs adaptive (profiling validation): -2.3887499999999083e-05

Held-out live outcomes were not used for selection.
Bootstrap probabilities remain resampling diagnostics.

## Retrospective cold-start net at R=10

| Strategy | Net USD | Mean saving % | Profiling USD |
|---|---:|---:|---:|
| existing_adaptive | 0.103387 | 12.71 | 0.003980 |
| existing_adaptive_risk | 0.096513 | 11.26 | 0.003980 |
| joint | 0.096946 | 12.62 | 0.009965 |
| tuned_joint | 0.096946 | 12.62 | 0.009965 |
