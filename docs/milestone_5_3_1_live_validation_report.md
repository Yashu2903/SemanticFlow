# Milestone 5.3.1 — Live Validation Status

## Current status

| Stage | Status |
|---|---|
| Implementation | Complete |
| M5.3 matched-seed statistical correction | Complete |
| Independent record selection | Complete |
| Dry-run plan validation | Complete |
| Live provider execution | Not executed |
| Independent policy evaluation | Pending live evidence |

No paid Jev requests were made during implementation.

## Dry-run result

The source-derived frozen configuration selected 300 records: 100 per workload and 50 per stratum within each workload. Independence checks use stable record IDs and content SHA-256 values. Eligible pools contained 589 Banking77, 586 All Beauty, and 588 SMS Spam records after exclusions.

The request plan contains 300 screens, 500 remaining separate-predicate requests, and 300 complete multi-decision requests. The 1,100 physical requests comprise 400 Banking77, 300 All Beauty, and 400 SMS Spam requests. Maximum exposure is 3,300 HTTP attempts under the two-retry policy. The rounded planning estimate is $0.0155 and the sum of the per-action historical-mean forecasts is $0.015792; neither is a billing guarantee. The frozen execution cap is $0.025 token-priced cost.

The dry run generated no execution ledger because no provider dispatch occurred. Provider evidence coverage is therefore 0 of 300 fully paired records, and all research questions remain unanswered.

## M5.3 correction

The original ablation prose compared a 50-seed hybrid aggregate against 20-seed ablation aggregates. The corrected analysis pairs the exact same seed, pool, setting, budget kind, and budget value. Surplus baseline seeds are explicitly excluded. At the pooled 100-request end-to-end checkpoint, the matched 20-seed hybrid had means of 0.30 predicate disagreements, 0.30 query disagreements, and 2.25 positives. The correction does not support the former suggestion that removing cost awareness increased disagreement yield; both means were 0.30. Removing the positive component materially reduced positive discovery, while rare events and intervals touching zero limit other component conclusions.

## M5.1 integrity reconciliation

Historical artifacts remain byte-identical. The previously failing integrity assertion reflected the committed M5.2.3 `jev.py` response-shape repair. A versioned evolution ledger now reconciles the exact old hash, exact new hash, and repair commit while retaining `all_unchanged = false`. Unknown source changes still fail reconciliation.

## Pending research questions

RQ1–RQ3 have no live result yet. The report will not label the experiment or independent evaluation complete until all compatible responses exist and the offline analysis command succeeds. Low disagreement prevalence, enriched-stratum selection effects, single-execution provider variability, and actual-cost forecast error remain principal risks.
