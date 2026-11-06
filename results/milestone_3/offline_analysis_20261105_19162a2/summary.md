# Milestone 3 offline analysis

Generated from stored traces only. No provider was contacted.

## Counterfactual replay of the Milestone 2 execution difference

- Observed calls: fixed 958, optimized 952 (difference 6).
- Ordering effects below are offline replay estimates, not causal measurements.
- Policy `historical_first`: replay-estimated ordering effect 5..5 calls; residual not explained by replay 1..1 calls.
- Policy `fresh_first`: replay-estimated ordering effect 4..4 calls; residual not explained by replay 2..2 calls.
- Policy `conflicts_unknown`: replay-estimated ordering effect 4..5 calls; residual not explained by replay 1..2 calls.

## Cardinality prediction (stage inputs after stage 1)

| Evaluation | Estimator | MAE | MAPE % | Mean signed error |
|---|---|---:|---:|---:|
| Profiling in-sample | independent | 9.841 | 39.52 | -9.841 |
| Profiling in-sample | correlation_aware | 0.338 | 1.25 | -0.006 |
| Profiling 5-fold CV | independent | 2.751 | 51.21 | -1.967 |
| Profiling 5-fold CV | correlation_aware | 1.331 | 24.70 | -0.001 |
| Held-out observed orders | independent | 35.000 | 49.31 | -17.500 |
| Held-out observed orders | correlation_aware | 19.476 | 25.70 | 19.476 |

## Plans

- `independent`: selected `product_defect -> poor_quality -> dissatisfaction`; minimum `product_defect -> poor_quality -> dissatisfaction`
- `correlation_aware`: selected `product_defect -> poor_quality -> dissatisfaction`; minimum `product_defect -> poor_quality -> dissatisfaction`
- `correlation_aware_confidence`: selected `poor_quality -> product_defect -> dissatisfaction`; minimum `product_defect -> poor_quality -> dissatisfaction`

## Bootstrap (sampling-uncertainty diagnostic)

- Current plan selection frequency: 0.769
- P(current plan cheaper than original): 0.769
- Stable at threshold 0.95: False

## Calibration of predictive cardinality intervals

| Evaluation | Estimator | Checks | 50% coverage | 90% coverage | Mean PIT |
|---|---|---:|---:|---:|---:|
| Held-out observed orders | independent | 6 | 0.167 | 0.333 | 0.735 |
| Held-out observed orders | correlation_aware | 6 | 0.167 | 1.000 | 0.166 |
| Profiling 5-fold CV | independent | 90 | 0.244 | 0.400 | 0.834 |
| Profiling 5-fold CV | correlation_aware | 90 | 0.767 | 1.000 | 0.530 |

## Break-even executions (sunk full profile)

- Observed, confounded: 113
- Replay estimate `historical_first`: 151..151
- Replay estimate `fresh_first`: 224..224
- Replay estimate `conflicts_unknown`: 151..224
- Predicted, correlation-aware: 63
- Predicted, independence: 62
- Predicted, bootstrap interval: 18..None (None means never)

## Adaptive profiling (offline simulation)

- `structural_upper_bound`, horizon 0: profiling skipped (['no_expected_executions'])
- `structural_upper_bound`, horizon 1: stop after 25 reviews (75 decisions); agrees with full profile: True; reasons ['expected_value_of_information_below_increment_cost']
- `structural_upper_bound`, horizon 10: stop after 25 reviews (75 decisions); agrees with full profile: True; reasons ['expected_value_of_information_below_increment_cost']
- `structural_upper_bound`, horizon 113: stop after 50 reviews (150 decisions); agrees with full profile: True; reasons ['expected_value_of_information_below_increment_cost']
- `structural_upper_bound`, horizon 1000: stop after 200 reviews (600 decisions); agrees with full profile: True; reasons ['maximum_checkpoint_reached']
- `transferred_prediction`, horizon 0: profiling skipped (['no_expected_executions'])
- `transferred_prediction`, horizon 1: profiling skipped (['profiling_cannot_break_even_within_horizon'])
- `transferred_prediction`, horizon 10: stop after 25 reviews (75 decisions); agrees with full profile: True; reasons ['expected_value_of_information_below_increment_cost']
- `transferred_prediction`, horizon 113: stop after 50 reviews (150 decisions); agrees with full profile: True; reasons ['expected_value_of_information_below_increment_cost']
- `transferred_prediction`, horizon 1000: stop after 200 reviews (600 decisions); agrees with full profile: True; reasons ['maximum_checkpoint_reached']
