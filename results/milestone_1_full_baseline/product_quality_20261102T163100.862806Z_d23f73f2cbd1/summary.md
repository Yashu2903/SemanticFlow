# SemanticFlow Milestone 1 Full Jev Baseline

## Objective

Establish the fixed-order, sequential AND baseline for the `product_quality` workload over all
1,000 reviews in the deterministic SemCEB sample. This run is the comparison baseline for future
optimization work; it performs no predicate reordering, fusion, routing, or adaptation.

The user explicitly authorized the paid live experiment before execution on November 2, 2026.

## Dataset

- Path: `data/processed/reviews_1000.parquet`
- Source: SemCEB `Arts_Crafts_and_Sewing` filtered review data
- Reviews: 1,000 usable records with 1,000 unique preserved IDs
- Sampling seed: 2026
- Preparation version: `m1-v1`
- SHA-256: `21cd6032e72379828f334422e29365688c5b5063c07ca23f08f8f025c2d55a99`
- Dataset reproduction check: passed

## Execution configuration

- Provider: Jev
- Requested and resolved model: `jev-1.14.0`
- Query: `product_quality`
- Operation: fixed-order sequential AND
- Predicate order: `poor_quality`, `product_defect`, `dissatisfaction`
- Probability threshold: 0.5 (`probability >= 0.5` passes)
- Early rejection: enabled
- Concurrency: 4
- Timeout: 30 seconds
- Maximum retries: 2
- Maximum HTTP attempts: 3,300, including retries
- Cache: disabled
- Reporting schema: 1.1.0
- SemanticFlow version: 0.1.0
- Source commit: `bd960cb359e9432bc97634b5c2907eb28b556c00`

## Results

| Predicate | Input | Passed | Rejected | Observed selectivity | HTTP attempts | Input tokens | Output tokens | Summed latency | Average latency | Estimated cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `poor_quality` | 1,000 | 118 | 882 | 11.8000% | 1,000 | 314,034 | 22,000 | 112.779 s | 0.1128 s | $0.013189 |
| `product_defect` | 118 | 89 | 29 | 75.4237% | 118 | 40,388 | 2,596 | 15.663 s | 0.1327 s | $0.001696 |
| `dissatisfaction` | 89 | 88 | 1 | 98.8764% | 89 | 29,803 | 2,136 | 11.232 s | 0.1262 s | $0.001252 |

All 1,000 input reviews completed. The final result contains 88 matching reviews and 912
definitively rejected reviews. The complete-query match rate is 8.8%.

## Request, token, cost, and timing totals

- Requested evaluations: 1,207
- Completed decisions: 1,207
- HTTP attempts: 1,207
- Successful API calls: 1,207
- Failed HTTP attempts: 0
- Retry attempts: 0
- Provider errors: 0
- Budget-blocked evaluations: 0
- Input tokens: 384,225
- Output tokens: 26,732
- Total tokens: 410,957
- Summed inference latency: 139.675 seconds
- Average inference latency: 0.1157 seconds per successful provider call
- End-to-end wall time: 54.583 seconds
- Estimated cost: $0.01613745

Cost uses TypeSafe's public list price verified on November 2, 2026: $0.042 per million input
tokens and free output tokens. It is a list-price estimate, not confirmation of the account's
actual billed charge.

## Early-rejection effectiveness

A naive evaluation of all three predicates for all 1,000 reviews would require 3,000 successful
record-predicate decisions. Sequential short-circuiting required 1,207:

- Calls avoided: 1,793
- Call reduction: 59.7667%
- Stage-two calls avoided after `poor_quality`: 882
- Stage-three calls avoided relative to evaluating every review: 911

Actual HTTP attempts equal completed decisions because the run had no failed attempts or retries.
This call-savings comparison does not make a claim about semantic quality.

## Consistency validation

All 23 post-run consistency checks passed:

- Completed reviews equal 1,000.
- Rejected reviews plus final matches equal 1,000.
- Each stage input equals the preceding stage's pass count.
- Final matches equal the third predicate's pass count.
- `decisions.jsonl` contains exactly 1,207 rows.
- Query, predicate, and trace token totals agree.
- `matches.jsonl` contains exactly 88 rows.
- HTTP attempts remain below the 3,300-attempt limit.
- Cache hits and misses are both zero.
- Every decision resolved to `jev-1.14.0`.
- The recorded dataset fingerprint matches the expected fingerprint.

## Observations and limitations

- `poor_quality` consumed the most tokens and summed inference time because every review reached
  it. It was also the most selective predicate in this configured order.
- `dissatisfaction` rejected only one of the 89 reviews that reached it in this workload.
- No human labels were supplied, so this experiment measures execution behavior, not precision,
  recall, or preservation of semantic quality.
- This is one live baseline run. Provider latency and model probabilities may vary across runs.
- A preceding sandbox-restricted attempt was preserved separately as an incomplete run. It could
  not connect to the provider, produced zero decisions and zero tokens, and is not part of these
  baseline results.
