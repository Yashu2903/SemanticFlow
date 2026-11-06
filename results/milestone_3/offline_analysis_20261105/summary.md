# Milestone 3 offline analysis

Generated from stored traces only. No provider was contacted.

## Counterfactual replay of the Milestone 2 execution difference

- Observed calls: fixed 958, optimized 952 (difference 6).
- Policy `historical_first`: ordering effect 5..5 calls; variability residual 1..1 calls.
- Policy `fresh_first`: ordering effect 4..4 calls; variability residual 2..2 calls.
- Policy `conflicts_unknown`: ordering effect 4..5 calls; variability residual 1..2 calls.

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

## Adaptive profiling (offline simulation)

- Horizon 1: stop after 25 reviews (75 decisions); agrees with full profile: True; reasons ['expected_value_of_information_below_increment_cost']
- Horizon 113: stop after 50 reviews (150 decisions); agrees with full profile: True; reasons ['expected_value_of_information_below_increment_cost']
- Horizon 1000: stop after 200 reviews (600 decisions); agrees with full profile: True; reasons ['maximum_checkpoint_reached']
