# SemanticFlow Milestone 5.1 Selection-Stage Results

## 1. Executive summary

The frozen selection experiment completed successfully. It acquired every one of the 3,000 unique physical requests, resolved all 300 aliases, and populated all 3,300 logical evidence roles. The successful session used 3,000 HTTP attempts with no retries or provider failures. A failed pre-response smoke session consumed three additional authorized attempts but no tokens; aggregate authorization usage was therefore 3,003 of 9,000 attempts.

The main finding is narrow. Full multi-decision batching was cheaper than the unchanged `existing_adaptive` baseline in two of nine workloads, but only the three-predicate Banking77 workload was both cheaper and admissible under the preregistered quality gate. Its full batch cost $0.001647786 per 100 records, versus $0.001778280 for adaptive separate execution, a saving of $0.000130494 or 7.34%. The other eight quality-constrained selections fell back to separate execution.

Full batching was harmful in seven workloads and in all three dataset aggregates. Four-question batches cost 21.24% to 28.24% more than adaptive separate execution. Selective-first hybrids produced exploratory execution savings in four workloads, but none supplied a second quality-admissible economic selection. This does not reverse the negative Milestone 5 fusion result: batching is not a generally cheaper replacement for early-rejecting separate execution.

Across the twelve measured batch groups, 3,490 of 3,500 named decisions agreed with their separate Jev references (99.71%; Wilson 95% CI 99.47%–99.84%). There were five relative false positives and five relative false negatives, and every disagreement crossed the 0.5 threshold from a near-threshold probability. These figures measure consistency with separate Jev execution, not human-validated semantic accuracy.

The frozen estimator selected the cheapest measured catalog plan in seven of nine workloads. Mean pairwise rank concordance was 91.89%; aggregate selection regret was $0.00002247, with two harmful selections. These are selection-sample diagnostics over the 111-plan bounded catalog, not evidence of global optimality or independent generalization.

The Stage D recommendation is to evaluate the frozen quality-constrained/profiling-aware policy independently. That policy batches only the Banking77 three-predicate workload and uses the existing adaptive separate plan elsewhere. The prepared broad paired schedule contains 10,800 unique held-out requests and has not been executed.

## 2. Authorization and experiment verification

The runner accepted the exact authorization phrase `AUTHORIZE M5.1 SELECTION STAGE FROZEN MEASUREMENT MATRIX`. The signed manifest bound execution to:

- model `jev-1.14.0`, threshold 0.5, cache disabled, concurrency 4;
- schedule fingerprint `2083c9321d4ed99ab1a48c527f71fd378d125d66868502751ab55e9e858a394d`;
- workload fingerprint `548852e317048aece4e9139113c62ea99ded1eae2b2960e3078b132d16e330f9`;
- protocol fingerprint `4b9dedcd2f889cdfcfdaf9852e1196e94434a0096a63b040781a4365a01ede35`;
- existing optimizer fingerprint `ac2f9a5966a379cace8b3322cdd13685035e8eb5918b5f1231aab9b2c3818a47`.

The initial network-restricted smoke session failed before receiving an HTTP response. Its three transport attempts were preserved in an incomplete session with zero completed requests and zero tokens. A versioned authorization amendment charged those attempts against the original 9,000-attempt aggregate ceiling and restricted the successful session to 8,997 further attempts. It did not expand the schedule or budget.

| Accounting item | Observed |
|---|---:|
| Unique physical requests | 3,000 |
| Logical evidence roles | 3,300 |
| Reusable aliases | 300 |
| Successful-session HTTP attempts | 3,000 |
| Failed pre-response attempts | 3 |
| Aggregate authorized attempts | 3,003 / 9,000 |
| Retries in successful session | 0 |
| Provider failures in successful session | 0 |
| Named-question evaluations | 5,300 |
| Input tokens | 1,064,824 |
| Output tokens | 142,700 |
| Input-token-priced cost | $0.044722608 |
| Resolved model | `jev-1.14.0` |
| Held-out calls | 0 |

The cost is an input-token-priced experimental amount at $0.042 per million input tokens. Output tokens were recorded but priced at $0 under the frozen experiment assumption; the value must not be represented as a complete provider bill under a different contract.

### Dataset reconciliation

| Dataset | Requests | Named decisions | Input tokens | Output tokens | Cost |
|---|---:|---:|---:|---:|---:|
| Amazon All Beauty | 1,100 | 2,000 | 421,959 | 53,200 | $0.017722278 |
| Banking77 | 1,000 | 1,800 | 344,230 | 49,700 | $0.014457660 |
| SMS Spam | 900 | 1,500 | 298,635 | 39,800 | $0.012542670 |
| **Total** | **3,000** | **5,300** | **1,064,824** | **142,700** | **$0.044722608** |

The no-retry forecast was 1,066,263 input tokens and $0.044783060. Actual use was 1,439 tokens and $0.000060452 lower. The earlier apparent price-rounding difference came from the forecast retaining fractional per-request token estimates before display rounding.

## 3. Measurement coverage

All nine workloads have complete separate-reference, full-batch, and scheduled remainder-group evidence. The 111-plan catalog contains three plans for each two-predicate workload, eight for each three-predicate workload, and 26 for each four-predicate workload. This is a bounded measured candidate set, not all theoretical physical plans.

| Dataset | Workload | Predicates | Reference positives / 100 | Coverage |
|---|---|---:|---:|---|
| Beauty | failure + usability | 2 | 8 | Complete |
| Beauty | reaction + packaging + refund | 3 | 0 | Complete; recall unsupported |
| Beauty | reaction + failure + value + return | 4 | 0 | Complete; recall unsupported |
| Banking77 | unauthorized + refund | 2 | 3 | Complete |
| Banking77 | card + unauthorized + refund | 3 | 3 | Complete |
| Banking77 | withdrawal + unauthorized + refund + access | 4 | 0 | Complete; recall unsupported |
| SMS | prize + contact | 2 | 3 | Complete |
| SMS | prize + urgency + contact | 3 | 2 | Complete; below positive minimum |
| SMS | security + prize + urgency + contact | 4 | 0 | Complete; recall unsupported |

No missing request keys, unexpected request keys, unresolved aliases, or incompatible model versions were found.

## 4. Existing adaptive baseline

The historical controller was replayed unchanged from `configs/optimizer.milestone_4.yaml`, using the frozen SHA-256 record order and checkpoints at 25, 50, and 100 records. With the frozen expected-execution horizon of ten, its initial economic gate declined profiling in every workload and retained the original predicate order. Consequently, every adaptive trajectory contains zero revealed profiling rows, zero profiling decisions, and $0 profiling expenditure. This is the controller's actual behavior, not an approximation.

| Workload | Selected adaptive order | Execution cost |
|---|---|---:|
| Beauty 2 | malfunction → hard-to-use | $0.001769040 |
| Beauty 3 | adverse-reaction → packaging-failure → return/refund | $0.001542072 |
| Beauty 4 | adverse-reaction → malfunction → poor-value → return/refund | $0.001606374 |
| Banking 2 | unauthorized-activity → refund/reversal | $0.001391838 |
| Banking 3 | card-related → unauthorized-activity → refund/reversal | $0.001778280 |
| Banking 4 | cash-withdrawal → unauthorized-activity → refund/reversal → access/verification | $0.001420146 |
| SMS 2 | prize/reward → contact-action | $0.001362774 |
| SMS 3 | prize/reward → urgency → contact-action | $0.001392300 |
| SMS 4 | account-security → prize/reward → urgency → contact-action | $0.001323630 |

The complete matrix remained unavailable to the controller at its decision point. Its zero-length trajectories therefore do not represent leakage; they record a pre-profile stop.

## 5. Full batching economics

| Workload | Adaptive cost | Full-batch cost | Saving vs adaptive | Quality status |
|---|---:|---:|---:|---|
| Beauty 2 | $0.001769040 | $0.001632498 | +7.72% | Failed: 1 FN, recall 0.875 |
| Beauty 3 | $0.001542072 | $0.001796298 | -16.49% | Insufficient positives (0) |
| Beauty 4 | $0.001606374 | $0.001947498 | -21.24% | Insufficient positives (0) |
| Banking 2 | $0.001391838 | $0.001475586 | -6.02% | Passed |
| Banking 3 | $0.001778280 | $0.001647786 | **+7.34%** | **Passed** |
| Banking 4 | $0.001420146 | $0.001811586 | -27.56% | Insufficient positives (0) |
| SMS 2 | $0.001362774 | $0.001441230 | -5.76% | Passed |
| SMS 3 | $0.001392300 | $0.001554630 | -11.66% | Insufficient positives (2) |
| SMS 4 | $0.001323630 | $0.001697430 | -28.24% | Insufficient positives (0) |

Only two full batches reduced execution cost. The Beauty 2 saving is economically positive but semantically inadmissible. The Banking 3 saving is both positive and admissible, though its paired 95% bootstrap interval for absolute savings spans zero (-$0.000012474 to $0.000284424). Thus the point estimate is promising, not conclusive.

Dataset aggregates all favored adaptive separate execution:

| Dataset | Adaptive total | Full-batch total | Full-batch difference |
|---|---:|---:|---:|
| Beauty | $0.004917486 | $0.005376294 | -$0.000458808 |
| Banking77 | $0.004590264 | $0.004934958 | -$0.000344694 |
| SMS | $0.004078704 | $0.004693290 | -$0.000614586 |

Request reduction was substantial but not sufficient for token savings. A full batch uses 100 requests per workload. Adaptive early rejection used between 100 and 135 requests, yet seven batches still consumed more input tokens. The results are consistent with shared record text helping small batches only when avoided prompt duplication exceeds the additional multi-question prompt and output structure.

## 6. Hybrid economics

The best supported batching-containing plan by measured execution cost was:

| Workload | Best batch/hybrid kind | Cost | Saving vs adaptive | 95% paired saving interval | Admissibility |
|---|---|---:|---:|---|---|
| Beauty 2 | Full batch | $0.001632498 | +7.72% | [$0.000034608, $0.000252882] | Failed |
| Beauty 3 | Selective-first | $0.001546608 | -0.29% | [-$0.000010584, $0] | Insufficient positives |
| Beauty 4 | Selective-first | $0.001551144 | +3.44% | [-$0.000005964, $0.000144438] | Insufficient positives |
| Banking 2 | Full batch | $0.001475586 | -6.02% | [-$0.000137928, -$0.000015372] | Passed but harmful |
| Banking 3 | Full batch | $0.001647786 | +7.34% | [-$0.000012474, $0.000284424] | Passed |
| Banking 4 | Selective-first | $0.001406832 | +0.94% | [-$0.000093870, $0.000120078] | Insufficient positives |
| SMS 2 | Full batch | $0.001441230 | -5.76% | [-$0.000121800, -$0.000023268] | Passed but harmful |
| SMS 3 | Selective-first | $0.001369914 | +1.61% | [-$0.000047250, $0.000091560] | Insufficient positives |
| SMS 4 | Selective-first | $0.001323630 | 0.00% | [$0, $0] | Insufficient positives |

Hybrid grouping rescued much of the large-batch downside in the four-predicate workloads, but positive support was absent. None of those apparent savings can be promoted into an admissible policy. A zero-reach later group was never treated as empirical certification of an unmeasured group; every replayed plan maps to all required measured operators.

## 7. Semantic consistency

Across all measured full and remainder batches, agreement with separate Jev references was 3,490/3,500 (99.71%). The ten threshold crossings comprised five relative false positives and five relative false negatives. Precision and recall were both 98.24% over 284 separate-reference-positive named decisions. The 95% Wilson interval for agreement was 99.47%–99.84%; for recall it was 95.95%–99.25%.

All ten disagreements were near the 0.5 threshold. Absolute probability changes ranged from 0.03 to 0.11. This concentration supports threshold sensitivity as an observed association, but does not establish that larger batches caused the shifts.

At complete-query level, the full-batch outcomes were:

| Workload | Reference positives | TP | TN | FP | FN | Agreement | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| Beauty 2 | 8 | 7 | 92 | 0 | 1 | 99% | Failed |
| Beauty 3 | 0 | 0 | 100 | 0 | 0 | 100% | Insufficient positives |
| Beauty 4 | 0 | 0 | 100 | 0 | 0 | 100% | Insufficient positives |
| Banking 2 | 3 | 3 | 97 | 0 | 0 | 100% | Passed |
| Banking 3 | 3 | 3 | 97 | 0 | 0 | 100% | Passed |
| Banking 4 | 0 | 0 | 100 | 0 | 0 | 100% | Insufficient positives |
| SMS 2 | 3 | 3 | 96 | 1 | 0 | 99% | Passed |
| SMS 3 | 2 | 2 | 98 | 0 | 0 | 100% | Insufficient positives |
| SMS 4 | 0 | 0 | 100 | 0 | 0 | 100% | Insufficient positives |

Perfect negative agreement in zero-positive workloads does not establish recall. With exactly three positives, the 0.95 recall threshold effectively permits no false negatives. Separate Jev execution remains an LLM-generated consistency reference; human-ground-truth semantic accuracy is unverified.

## 8. Quality gate

The gate was preserved exactly: at least 40 records, at least three positive references, agreement ≥0.98, recall ≥0.95, false-negative rate ≤0.05, changed-decision rate ≤0.02, and measured complete-plan evidence.

Four batching-containing plans passed:

- Banking 2 full batch: semantically passed but 6.02% more expensive.
- Banking 3 full batch: semantically passed and 7.34% cheaper.
- Banking 3 selective-first hybrid: semantically passed but was not the cheapest admissible candidate.
- SMS 2 full batch: semantically passed but 5.76% more expensive.

One batching plan failed with adequate evidence: Beauty 2 full batch, due to recall 7/8 and false-negative rate 1/8. Ten batching/hybrid catalog plans had insufficient positive support. Separate plans are reference plans and are marked separately rather than treated as newly validated semantic candidates.

## 9. Cost-estimator evaluation

The frozen `m5.1-existing-unconditional-group-reach-v1` estimator was evaluated without post-selection tuning.

| Metric | Result |
|---|---:|
| Measured catalog plans | 111 |
| MAE | $0.0000196835 |
| RMSE | $0.0000300540 |
| MAPE | 1.280% |
| Signed bias | -$0.0000176472 |
| Mean pairwise ranking concordance | 91.89% |
| Correct cheapest selections | 7 / 9 |
| Harmful selections | 2 / 9 |
| Total measured-set regret | $0.000022470 |

The negative signed bias indicates underprediction on average. The two ranking misses occurred for the Beauty four-predicate and SMS three-predicate workloads. SMS 3 contributed almost all regret ($0.000022386). The optimizer's rankings are useful but not reliable enough to substitute for measured selection evidence.

## 10. Frozen selection behavior

Cost-only selection chose the Beauty 2 full batch despite its quality failure; it selected separate permutations for the other workloads. This confirms why the economic ablation is not an admissible production policy.

Quality-constrained and profiling-aware selection both chose the Banking 3 full batch and fell back to `existing_adaptive` for the other eight workloads. The selected Banking plan is:

- plan ID `m51selplan_4e05511d49f2861f7dfb3361`;
- fingerprint `4e05511d49f2861f7dfb336183e349ed2c5b03c93e05a01cca2943431d91e1da`;
- group `[banking_card_related, banking_unauthorized_activity, banking_refund_or_reversal]`;
- selection cost $0.001647786 per 100 records;
- complete-query agreement 100/100 with three positive references.

The profiling-aware replay exposes evidence only at checkpoints 25, 50, and 100 in the frozen order. Because the preflight did not freeze a numerical early-stop value model for batching evidence, the conservative replay reaches the full checkpoint before certifying a non-reference candidate. This avoids using the complete matrix at checkpoint zero. Actual full-matrix research expenditure is reported separately from each counterfactual profiling scenario.

## 11. Profiling economics

For the selected Banking 3 batch, execution saving is $0.000130494 per 100-record workload execution.

| Scenario | Incremental profiling | Break-even executions | Net at R=10 | Net at R=100 |
|---|---:|---:|---:|---:|
| Standalone | $0.007067130 | 55 | -$0.005762190 | +$0.005982270 |
| Shared statistics | $0.002967972 | 23 | -$0.001663032 | +$0.010081428 |
| Warm start | $0 | immediate | +$0.001304940 | +$0.013049400 |

These values exclude optimization CPU cost because no separately metered optimization cost exists. They are deployment counterfactuals. The actual research acquisition cost remains $0.044722608 and is not reallocated as though every strategy independently paid for the matrix.

## 12. Negative and unexpected findings

1. Four-question batching was uniformly expensive: 21.24% to 28.24% worse than adaptive execution. Request consolidation did not offset its token overhead.
2. The largest clean economic win, Beauty 2 at 7.72%, failed quality due to one false negative among eight positives. Cheap and admissible were not interchangeable.
3. Banking 2 and SMS 2 passed the quality gate but cost 6.02% and 5.76% more. Semantic stability alone did not justify batching.
4. Four workloads had no positive complete-query references and a fifth had only two. The 100-record selection partition therefore remained unable to evaluate recall for most larger conjunctions.
5. The historical adaptive controller rationally declined profiling at a ten-run horizon. Complete-matrix acquisition was useful for research selection but would not have been privately economic for that controller.
6. The estimator's aggregate errors were small, yet one SMS ranking miss consumed essentially all selection regret. Aggregate MAE concealed workload-specific ranking risk.
7. All named-decision disagreements were near threshold, but one Beauty query-level false negative was enough to disqualify an otherwise profitable plan.

## 13. Frozen plan catalog

The canonical plan catalog persists full objects, group modes, prompts, prompt hashes, model, threshold, estimator and planner versions, profiling references, and deterministic fingerprints. The quality-constrained mapping freezes one full batch and eight separate fallbacks. Multiple strategies selecting the same physical configuration share the same canonical plan identity.

The complete mappings are in `results/milestone_5_1/selection_stage_results/frozen_strategy_selections.json` and the 111 canonical objects are in `frozen_physical_plan_catalog.json`.

## 14. Statistical limitations

Selection performance is in-sample: the same 100 records supplied costs, quality evidence, and plan choice. Workloads share records and predicates, so neither 111 plan rows nor 3,500 named comparisons are fully independent experimental units. The paired bootstrap resamples records but does not turn workload-level selection into independent validation. The single admissible win has only three positive complete-query references and its cost interval crosses zero. Provider repeatability was not measured through deliberate repetitions. No human labels establish semantic accuracy.

## 15. Stage D recommendation

Proceed to an independently authorized held-out evaluation of the frozen policy, with the primary comparison between unchanged adaptive separate execution and the selected quality-constrained/profiling-aware mapping. The scientific question is deliberately modest: does the Banking 3 batching advantage replicate on 400 untouched records while the fallback policy preserves the baseline elsewhere?

The prepared broad schedule contains 10,800 unique physical requests, expects 3,887,444 input tokens and 528,800 output tokens, and has an input-token-priced expected cost of $0.163272648. Its maximum is 32,400 attempts. These are proposals, not authorization. Stage D was not executed and held-out outcomes were not inspected.

**Readiness status: `READY_FOR_M5_1_STAGE_D_AUTHORIZATION`.**
