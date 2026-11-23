# Milestone 5.3.1 — Independent Live Validation

## Completion status

Implementation complete; dry-run validated; live provider execution complete; independent evaluation complete.

The session acquired 1100 physical requests across 300 fully paired independent records, used 1100 HTTP attempts, and incurred $0.01613161 in token-priced cost.

| Workload | Records | Predicate disagreements | Query disagreements | Positives |
|---|---:|---:|---:|---:|
| m51_banking_card_unauthorized_refund | 100 | 5 | 2 | 2 |
| m51_beauty_failure_usability | 100 | 2 | 0 | 10 |
| m51_sms_prize_urgency_contact | 100 | 1 | 0 | 9 |

## RQ1 — Positive acquisition

At 100 end-to-end requests, positive-oriented minus random had a paired mean positive difference of 1.660 (policy-seed 95% bootstrap interval 1.040 to 2.220). Verdict: **supported**.

## RQ2 — Disagreement acquisition

Weighted hybrid minus random had a paired mean predicate-disagreement difference of 0.820 (0.460 to 1.180). Verdict for execution-disagreement discovery: **supported**. Human review is required before calling any case a genuine semantic error.

## RQ3 — Workload generalization

`workload_specific_results.csv` contains matched workload comparisons. A strategy generalizes only if its direction is consistent across all three workload pools; pooled ordering alone is insufficient.

## Limitations

Representative and enriched strata have different selection mechanisms. Repeated policy seeds measure selection variability over one fixed provider outcome per record, not provider-response variability or additional independent evidence. Rare disagreement counts can make disagreement-policy verdicts inconclusive.
