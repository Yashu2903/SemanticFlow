# Milestone 5.2.3 Cost Reconciliation

## Preserved failed session

The frozen full-pilot forecast was 96,550 input tokens, 12,200 output tokens, and a $0.004055100
input-token cost proxy. The session dispatched one request whose preflight estimate was 325 input
tokens, 35 output tokens, and $0.00001365.

The persistent ledger records one HTTP attempt, zero attributable tokens, and a $0.00 attributable
proxy. This is incomplete accounting: the provider response passed HTTP and provider parsing before
the Pilot A adapter rejected its probability shape, but the failure path did not persist usage or
the raw response. Actual provider billing exposure is therefore unknown for one attempt. No retry
exposure occurred.

## Repaired fresh session

The full 280-action forecast was 96,550 input tokens, 12,200 output tokens, and $0.004055100 under
the frozen $0.042-per-million-input-token model. The conservative forecast was $0.006708384375.
Adaptive execution requested 251 actions; their action-level forecasts totaled 85,348 input tokens,
10,419 output tokens, and $0.003584616.

Actual usage was 84,890 input tokens and 10,239 output tokens, for a measured cost proxy of
**$0.003565380**. Compared with the actually executed action forecasts, input usage and cost were
458 tokens and $0.000019236 lower, both a 0.54% overforecast; output usage was 180 tokens lower, a
1.73% overforecast. Compared with the full schedule, adaptive execution used 11,660 fewer input
tokens (12.08%), 1,961 fewer output tokens (16.07%), and $0.000489720 less proxy cost (12.08%).

Discovery screening itself used 9,936 input tokens and 1,050 output tokens across 30 requests,
costing $0.000417312, or 11.70% of measured pilot cost. The single diagnostic follow-up used 395
input tokens and 69 output tokens and cost $0.000016590. With one informative screen, screening cost
per informative result was $0.000417312.

Per-workload measured accounting was:

| Workload | Requests | Input tokens | Output tokens | Cost proxy |
|---|---:|---:|---:|---:|
| Banking77 | 91 | 30,073 | 3,909 | $0.001263066 |
| Amazon All Beauty | 70 | 26,043 | 2,570 | $0.001093806 |
| SMS Spam | 90 | 28,774 | 3,760 | $0.001208508 |

The largest request-type percentage underforecast was All Beauty representative singles at 11.43%
for input tokens. The largest overforecast was SMS candidate batches at 8.52%. These findings are
retrospective diagnostics only; the frozen estimator was not retuned from this session. No retries
created additional exposure, and provider-billing adjustments were not independently reconciled.
