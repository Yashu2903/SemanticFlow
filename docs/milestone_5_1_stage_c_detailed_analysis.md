# Milestone 5.1 Stage C detailed analysis

## 1. Executive summary

The authorized Stage B run is internally complete: **810 physical requests**, **810 HTTP attempts**, **1,380 named-question evaluations**, 299,406 input tokens, 37,690 output tokens, and **$0.012575052** under the frozen $0.042/M-input, $0-output experimental pricing assumption. All 33 operators completed, all traces returned HTTP 200, and no retries or provider errors were recorded.

Multi-decision batching clearly reduced requests and input tokens relative to evaluating every question separately. On the representative sample it was cheaper than all-questions separate execution at every batch size. The realistic early-reject comparison is mixed: full batching won on 1/6 fully measurable declared workloads, while a measured selective-first hybrid beat the original-order separate replay on 2/4 workloads with a compatible hybrid.

Semantic consistency is less conclusive. Across the representative multi-decision comparisons there were 807/810 matching named decisions (99.63%), 0 false positives and 3 false negatives relative to separate Jev calls. These are consistency errors, not errors against human truth. Only 11/12 representative batch/conjunction groups preserved the complete AND-query result set exactly. No batching or hybrid candidate passes the preregistered quality gate because every representative group has 30 records, below the minimum 40, and three of nine declared workloads lack compatible complete batch measurements.

The appropriate conclusion is **READY_FOR_SELECTION_STAGE_AUTHORIZATION**. Stage D is not ready. The protocol reserves a 100-record selection partition that has not received inference. A bounded 3,300-request selection proposal is documented, but no request is authorized or executed here.

## 2. Experimental setup

The run used `jev-1.14.0`, threshold 0.5, cache disabled, concurrency four, and at most two retries. The frozen schedule selected 30 pilot records per dataset for four single predicates, nested two/three/four-question batches, and a four-condition conjunctive comparator. Ten development records per dataset were selected by deterministic lexical enrichment for two singles plus one two-question batch. The datasets were Amazon All Beauty, Banking77, and UCI SMS Spam.

The executed composition reconciles exactly: 360 representative single requests, 270 representative multi-decision requests, 90 representative conjunctive requests, and 90 enriched diagnostic requests. No selection or evaluation ID appears in a trace. Partition checks used frozen manifest IDs; the analysis did not read evaluation text or outcomes.

An important coverage limitation is structural: the nested schedule does not equal all nine declared workloads. The Beauty two- and three-predicate workloads and the Banking three-predicate workload include predicates that were not measured as Stage B singles/batches. Those workloads are reported as not evaluable rather than assigned modeled zeroes.

## 3. Provider usage and cost reconciliation

The 810 JSONL rows reconcile to 810 unique `(dataset, record, operator, sample)` keys and to `session.json`. Their input tokens sum to 299,406; multiplying by $0.042/1,000,000 gives exactly $0.012575052. Output tokens sum to 37,690 and are reported separately. The frozen experimental rate assigns them $0, but the artifacts contain no invoice or billing contract; therefore this report calls $0.012575052 a token-priced experimental cost, not a verified provider bill.

The persisted operator metadata requests `jev-1.14.0`; every trace resolves to `jev-1.14.0`. Prompt hashes rebuild exactly from the frozen questions, and each trace hash equals its operator hash. Stage B did not persist the `FusionPhysicalPlan` object or its plan hash, so physical plan hashes cannot be independently verified. This is a provenance defect, not a missing request result.

Summed request latency is available; end-to-end wall time per operator is not. Concurrency makes timestamp-span reconstruction unsuitable as a substitute for persisted wall time.

## 4. Multi-decision batching performance

| Batch size | Records | Batch requests | All-separate requests | Batch cost | All-separate cost | Saving vs all | Early-separate cost | Saving vs early | Batch latency (sum s) | Early latency (sum s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 90 | 90 | 180 | $0.001361010 | $0.002452380 | $0.001091370 | $0.001447488 | $0.000086478 | 9.431 | 12.182 |
| 3 | 90 | 90 | 270 | $0.001487010 | $0.003669750 | $0.002182740 | $0.001488480 | $0.000001470 | 9.838 | 12.431 |
| 4 | 90 | 90 | 360 | $0.001629390 | $0.004903500 | $0.003274110 | $0.001302042 | $-0.000327348 | 9.295 | 10.804 |

Per-dataset paired results show the heterogeneity hidden by the size aggregate:

| Dataset | Size | Batch cost | All-separate cost | Early cost | Batch saving vs early |
|---|---|---|---|---|---|
| amazon_all_beauty | 4 | $0.000568470 | $0.001722000 | $0.000431130 | $-0.000137340 |
| amazon_all_beauty | 3 | $0.000521850 | $0.001290870 | $0.000583926 | $0.000062076 |
| amazon_all_beauty | 2 | $0.000477750 | $0.000862260 | $0.000568806 | $0.000091056 |
| banking77_queries | 4 | $0.000542808 | $0.001573992 | $0.000464940 | $-0.000077868 |
| banking77_queries | 2 | $0.000442008 | $0.000785736 | $0.000433104 | $-0.000008904 |
| banking77_queries | 3 | $0.000489888 | $0.001177344 | $0.000459228 | $-0.000030660 |
| sms_spam | 4 | $0.000518112 | $0.001607508 | $0.000405972 | $-0.000112140 |
| sms_spam | 2 | $0.000441252 | $0.000804384 | $0.000445578 | $0.000004326 |
| sms_spam | 3 | $0.000475272 | $0.001201536 | $0.000445326 | $-0.000029946 |

Scenario A is fully paired: the same predicates, records, and model are compared against all corresponding single calls. Scenario B replays the same single traces in the batch predicate order and stops after the first rejection. It is a saved-trace replay, not an additional call. Batching's request savings are large by construction, but dollar savings follow only when the shared record/prompt overhead outweighs lost early rejection.

The stored data support association, not provider-internal causality. Input savings are consistent with sharing record context and request framing, but provider-side tokenization components are not separately observable.

## 5. Effects of batch size

| Batch size | Named decisions | Agreement | 95% CI | FP | FN | FNR |
|---|---|---|---|---|---|---|
| 2 | 180 | 99.44% | [96.92%, 99.90%] | 0 | 1 | 0.0370 |
| 3 | 270 | 100.00% | [98.60%, 100.00%] | 0 | 0 | 0.0000 |
| 4 | 360 | 99.44% | [98.00%, 99.85%] | 0 | 2 | 0.0541 |

Complete-query consistency for the representative multi-decision groups was:

| Dataset | Size | Reference + | Agreement | FP | FN | Query Jaccard | Exact set |
|---|---|---|---|---|---|---|---|
| amazon_all_beauty | 4 | 0 | 100.00% | 0 | 0 | 1.000 | True |
| amazon_all_beauty | 3 | 0 | 100.00% | 0 | 0 | 1.000 | True |
| amazon_all_beauty | 2 | 0 | 100.00% | 0 | 0 | 1.000 | True |
| banking77_queries | 4 | 0 | 100.00% | 0 | 0 | 1.000 | True |
| banking77_queries | 2 | 2 | 96.67% | 0 | 1 | 0.500 | False |
| banking77_queries | 3 | 0 | 100.00% | 0 | 0 | 1.000 | True |
| sms_spam | 4 | 0 | 100.00% | 0 | 0 | 1.000 | True |
| sms_spam | 2 | 2 | 100.00% | 0 | 0 | 1.000 | True |
| sms_spam | 3 | 0 | 100.00% | 0 | 0 | 1.000 | True |

Average input tokens per request rise with batch size, while input tokens per named question fall. The marginal input-token field in `batch_size_economics.csv` compares adjacent aggregate sizes; it is descriptive because the added predicates differ by dataset. Larger-batch disagreement rates also differ, but batch size is confounded with predicate set and workload. The pilot cannot establish that size itself caused the differences.

## 6. Hybrid execution performance

| Workload | Status | Original separate | Full batch | Full-batch delta | Best measured hybrid | Hybrid delta |
|---|---|---|---|---|---|---|
| m51_beauty_failure_usability | not_evaluable | not evaluable | not evaluable | not evaluable | not evaluable | not evaluable |
| m51_beauty_reaction_packaging_refund | not_evaluable | not evaluable | not evaluable | not evaluable | not evaluable | not evaluable |
| m51_beauty_reaction_failure_value_return | fully_measured_plan_matrix | $0.000431130 | $0.000568470 | $-0.000137340 | $0.000431130 | $0.000000000 |
| m51_banking_unauthorized_refund | fully_measured_plan_matrix | $0.000433104 | $0.000442008 | $-0.000008904 | not evaluable | not evaluable |
| m51_banking_card_unauthorized_refund | not_evaluable | not evaluable | not evaluable | not evaluable | not evaluable | not evaluable |
| m51_banking_withdrawal_unauthorized_refund_access | fully_measured_plan_matrix | $0.000464940 | $0.000542808 | $-0.000077868 | $0.000436506 | $0.000028434 |
| m51_sms_prize_contact | fully_measured_plan_matrix | $0.000445578 | $0.000441252 | $0.000004326 | not evaluable | not evaluable |
| m51_sms_prize_urgency_contact | fully_measured_plan_matrix | $0.000445326 | $0.000475272 | $-0.000029946 | $0.000425292 | $0.000020034 |
| m51_sms_security_prize_urgency_contact | fully_measured_plan_matrix | $0.000405972 | $0.000518112 | $-0.000112140 | $0.000405972 | $0.000000000 |

Every listed cost is exact replay from saved traces on the same 30 pilot records. Plans are enumerated only when every stage group was actually measured. A record enters the next stage only when all decisions in the current group pass. `hybrid_plan_costs.csv` records stage entrants, conditional selectivity, avoided downstream calls, tokens, and cost.

The best measured hybrid is not automatically an admissible optimizer choice: its search and evaluation use the same 30 records, its complete-plan evidence remains below the quality minimum, and unmeasured groupings are excluded. Zero reach never certifies an unmeasured group.

## 7. Semantic disagreement analysis

The representative multi-decision matrix contains 810 comparable named decisions and 3 threshold crossings. All 3 disagreements had at least one probability within 0.10 of the 0.5 threshold, and 0 had both probabilities at least 0.25 away. The observed changes are therefore consistent with threshold proximity, although three cases cannot establish a general mechanism.

| Dataset | Size | Predicate | Separate p | Batch p | Record |
|---|---|---|---|---|---|
| banking77_queries | 4 | banking_unauthorized_activity | 0.50 | 0.44 | banking77_queries:2dc19c5462032e74d31047d2 |
| banking77_queries | 4 | banking_refund_or_reversal | 0.50 | 0.47 | banking77_queries:bbaff54f366663ff19cacb6c |
| banking77_queries | 2 | banking_unauthorized_activity | 0.50 | 0.49 | banking77_queries:2dc19c5462032e74d31047d2 |

Per-predicate probability deltas, crossing records, false positives, and false negatives are in `predicate_agreement.csv`.

Complete-query comparisons use the AND of separate decisions as reference. They report match-set equality and Jaccard overlap in `query_level_agreement.csv`. Conjunctive-binary traces expose only one group probability and are never represented as constituent predicate decisions.

Separate Jev calls are a reproducible LLM reference, not human ground truth. This analysis establishes batched/separate consistency only. With no identical-request repetitions it cannot isolate inference nondeterminism, and with no human labels it cannot establish semantic accuracy.

## 8. Representative versus enriched samples

Representative results use the first 30 sorted pilot IDs per dataset from the frozen 40-record pilot partition. Enriched development records were chosen by deterministic lexical counts (`fail/broke/return/refund/leak`, banking refund/unauthorized terms, or SMS prize/contact terms) with record ID as tie-breaker. They are diagnostic, not a probability sample.

The report never pools enriched and representative records for population agreement. Enriched rows are separate in every CSV. Sparse reference positives and undefined recall denominators are preserved in the quality artifacts; no zero-positive group is interpreted as perfect recall.

| Dataset | Records | Reference + | Agreement | FP | FN | Population estimate |
|---|---|---|---|---|---|---|
| amazon_all_beauty | 10 | 0 | 100.00% | 0 | 0 | no |
| banking77_queries | 10 | 3 | 90.00% | 0 | 1 | no |
| sms_spam | 10 | 1 | 100.00% | 0 | 0 | no |

## 9. Statistical uncertainty

Cost intervals use a paired record bootstrap with 2,000 draws. Proportion intervals use 95% Wilson intervals and always retain their numerator and denominator. Pooled batch-size intervals are descriptive because decisions share records and physical requests. There is no repeated identical-request evidence for provider stability.

Heterogeneity is reported per group, dataset, and workload. Aggregate savings do not override harmful workload-level cases. Unpaired or missing groups are marked not evaluable.

## 10. Cost-estimation accuracy

| Workload | Plans | Cost MAE | MAPE | Rank concordance | Regret | Selected best |
|---|---|---|---|---|---|---|
| m51_beauty_reaction_failure_value_return | 34 | $0.000002910 | 0.55% | 0.988 | $0.000000000 | True |
| m51_banking_unauthorized_refund | 3 | $0.000000518 | 0.11% | 1.000 | $0.000000000 | True |
| m51_banking_withdrawal_unauthorized_refund_access | 34 | $0.000010055 | 2.16% | 0.807 | $0.000028182 | False |
| m51_sms_prize_contact | 3 | $0.000001252 | 0.27% | 1.000 | $0.000000000 | True |
| m51_sms_prize_urgency_contact | 9 | $0.000008090 | 1.71% | 0.889 | $0.000002436 | False |
| m51_sms_security_prize_urgency_contact | 34 | $0.000004615 | 0.98% | 0.938 | $0.000000000 | True |

These are in-sample diagnostics of the current reach-and-group-mean cost model, not saved preregistered predictions. The model uses unconditional group selectivity recursively, whereas observed replay uses correlated record-level outcomes. Stage-cardinality and request-count errors expose the resulting correlation bias. The only true ex-ante Stage A forecast was aggregate: 308,763 input tokens and $0.012968030 versus 299,406 and $0.012575052 observed (an input/cost overprediction of 3.13%).

## 11. Plan-selection quality

Within the fully measured candidate subset, `plan_ranking_analysis.json` reports pairwise rank concordance, predicted-best versus observed-best, and regret. It does not claim global optimality: most of the 4/20/138 enumerated theoretical plans lack measured groups. Three workloads have no compatible complete matrix.

No actual post-pilot plan selection was persisted before this analysis. “Cost-only selection” here means an explicitly labeled in-sample diagnostic. It must not be read as a frozen Stage D plan.

## 12. Quality-constrained planning

The gate remains unchanged: at least 40 records, at least three separate-reference positives, agreement at least 0.98, recall at least 0.95, FNR at most 0.05, changed rate at most 0.02, and real complete-plan evidence. Of 50 direct or replayed candidate evaluations, 0 qualify. Quality-constrained and profiling-aware strategies therefore fall back to the existing adaptive baseline. Enriched development evidence cannot repair representative coverage without a valid weighting design.

## 13. Profiling-aware economics

| Workload | Exploratory strategy | Saving/run | Standalone profile | Break-even | Quality admissible |
|---|---|---|---|---|---|
| m51_beauty_reaction_failure_value_return | selective_first_hybrid | $0.000000000 | $0.003290070 | None | False |
| m51_banking_unauthorized_refund | full_multi_decision_batch | $-0.000008904 | $0.001227744 | None | False |
| m51_banking_withdrawal_unauthorized_refund_access | selective_first_hybrid | $0.000028434 | $0.003048696 | 108 | False |
| m51_sms_prize_contact | full_multi_decision_batch | $0.000004326 | $0.001245636 | 288 | False |
| m51_sms_prize_urgency_contact | selective_first_hybrid | $0.000020034 | $0.002118060 | 106 | False |
| m51_sms_security_prize_urgency_contact | selective_first_hybrid | $0.000000000 | $0.003042144 | None | False |

The actual Stage B expenditure is charged once: $0.012575052. Standalone economics charge every compatible single and batch profile needed by one workload. Shared economics allocate an already-paid group once per dataset in declared workload order. Warm-start economics charge zero incremental profiling. Horizons 1, 5, 10, 25, 50, 100, 250, and 1,000 are in `profiling_economics.csv`.

All positive batching/hybrid economics remain exploratory because quality eligibility is zero. Local optimization overhead is recorded as $0 API cost; CPU time was not monetized. A leakage-free adaptive profiling trajectory is not reconstructable because all 30 traces are saved as a completed matrix without checkpointed controller state.

## 14. Negative and surprising results

* Request reduction did not guarantee a win against early rejection. Workloads with selective early predicates can avoid enough downstream single calls to erase batching's all-questions advantage.
* Three declared workloads are absent from the compatible matrix even though the aggregate schedule completed. Execution completeness is therefore not workload completeness.
* The quality gate rejects every candidate before Stage D. Strong aggregate agreement cannot substitute for the 40-record minimum, sufficient positives, or complete representative evidence.
* All three named-decision disagreements were near the threshold, but the sample is too small to infer that larger batches are generally stable away from the threshold.
* The cost predictor is in-sample and still shows ranking/cardinality error where stage outcomes are correlated. Independent confirmation is necessary.
* Physical operator hashes are strong, but physical plan IDs were not persisted. Future runners should persist the complete plan object and configuration fingerprint.

## 15. Research implications

Multi-decision batching is a credible economic mechanism for all-questions evaluation: it shares context and reduces tokens as well as requests. SemanticFlow's realistic problem is conditional execution, so selective-first hybrids are the more relevant candidate when an inexpensive, rejecting predicate exists. The pilot supports continued research on both, but not production adoption or a claim of general improvement.

The previous M5 conjunction result remains negative. Nothing here reinterprets its 21.77% cost increase or quality-gate failures. Multi-decision batching is a different operator because it preserves named outputs, but it has its own observed consistency changes.

## 16. Remaining limitations

There are no human labels, no identical-request replicates, no saved wall times, no persisted Stage B physical plan IDs, only 30 representative records per group, sparse positives, three unmeasured declared workloads, and no selection-partition inference. Plan search is limited to measured compatible groups. Profiling break-even assumes repeated workloads resemble the pilot and does not monetize local compute.

## 17. Next-stage recommendation

| Strategy | Supported workloads | Aggregate cost | Quality-admissible batching | Recommendation |
|---|---|---|---|---|
| fixed_separate | 6 | $0.002626050 | 0 | Retain as measured reference and fallback. |
| existing_adaptive | 0 | not evaluable | 0 | Retain as production baseline; Stage B did not persist a new adaptive selection. |
| full_conjunctive_fusion | 3 | $0.001854930 | 0 | Retain only as a negative-control experimental candidate. |
| full_multi_decision_batch | 6 | $0.002987922 | 0 | Retain as an experimental candidate pending representative selection evidence. |
| selective_first_hybrid | 4 | $0.001698900 | 0 | Retain as an experimental candidate for measured compatible workloads. |
| cost_only_hybrid | 6 | $0.002595264 | 0 | Retain as an ablation; in-sample selection is not quality admissible. |
| quality_constrained_hybrid | 6 | $0.002626050 | 0 | Retain; currently falls back because no candidate passes the gate. |
| profiling_aware_hybrid | 6 | $0.002626050 | 0 | Retain; currently falls back and adaptive trajectory is not evaluable from saved traces. |

Aggregate strategy costs above are shown only with their support counts. They are directly comparable only when support sets match: fixed separate, full batch, cost-only, and the two fallback strategies cover the same six workloads; conjunction covers three and measured hybrids cover four. The table is not a cross-support global ranking.

Do **not** authorize Stage D from this evidence. The exact next stage is a separately authorized, versioned selection-stage measurement on the untouched 100-record selection partition. The proposed ceiling is 3,300 logical requests and 9,900 retry-inclusive attempts: 1,800 single references for all 18 dataset predicates, 900 full batches for all nine workloads, and 600 targeted remainder batches for one selective-first candidate on each three- or four-predicate workload. The schedule, prompts, hashes, and budget must be frozen before authorization. No paid call was made by this analysis.
