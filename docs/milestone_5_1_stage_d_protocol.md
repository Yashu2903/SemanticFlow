# SemanticFlow Milestone 5.1 Stage D Protocol

## 1. Purpose

Stage D is an independent held-out evaluation of physical execution policies selected on the Milestone 5.1 selection partition. It must test generalization, not revise the selected policy. No Stage D calls have been made.

The primary hypothesis is that the frozen quality-constrained/profiling-aware policy reduces Jev input-token cost relative to the unchanged `existing_adaptive` policy without violating the preregistered consistency gate. The selected policy differs from adaptive execution only for `m51_banking_card_unauthorized_refund`, where it uses one three-question batch.

Secondary analyses evaluate fixed separate execution, full batching, frozen selective-first hybrids, and cost-only selection. Conjunctive-binary fusion remains a historical negative control and is included only when its exact protocol supplies compatible evidence; this protocol does not silently add new fusion requests.

## 2. Partition and leakage boundary

Stage D uses the frozen 400-record held-out partition for each of Amazon All Beauty, Banking77, and SMS Spam. Only record identities and fingerprints may be used to construct the schedule before authorization. Inference outcomes must remain unavailable to planners, selectors, and quality gates until live acquisition completes.

Development and Stage B data were used for implementation and preflight. The 100-record selection partition was used to select and freeze plans. It must not be described as independent validation. Stage D may estimate the performance of frozen plans but may not be used to retune thresholds, estimator parameters, group boundaries, or tie-breaking rules.

## 3. Frozen inputs

| Item | Fingerprint |
|---|---|
| Selection schedule | `2083c9321d4ed99ab1a48c527f71fd378d125d66868502751ab55e9e858a394d` |
| Workloads | `548852e317048aece4e9139113c62ea99ded1eae2b2960e3078b132d16e330f9` |
| Selection protocol | `4b9dedcd2f889cdfcfdaf9852e1196e94434a0096a63b040781a4365a01ede35` |
| Existing optimizer | `ac2f9a5966a379cace8b3322cdd13685035e8eb5918b5f1231aab9b2c3818a47` |
| Stage D schedule | `6d6fc867d221c9ce1ac29011d0410558b939df842563a5d3116816401f1139f6` |

Provider settings remain model `jev-1.14.0`, threshold 0.5, cache disabled, concurrency 4, and at most two retries per physical request. Any change requires a versioned manifest and new explicit authorization.

## 4. Frozen strategy mapping

The quality-constrained and profiling-aware strategies share the following canonical mapping:

| Workload | Frozen execution | Plan ID | Fingerprint prefix |
|---|---|---|---|
| Beauty 2 | Separate original/adaptive | `m51selplan_d1572e4176aa19745ecec3ab` | `d1572e4176aa` |
| Beauty 3 | Separate original/adaptive | `m51selplan_c7c4f8bf8e099559ff909280` | `c7c4f8bf8e0` |
| Beauty 4 | Separate original/adaptive | `m51selplan_c9b259bbcab3297242646a30` | `c9b259bbcab` |
| Banking 2 | Separate original/adaptive | `m51selplan_330710722b4d97b2a056da04` | `330710722b4` |
| Banking 3 | Full 3-question batch | `m51selplan_4e05511d49f2861f7dfb3361` | `4e05511d49f` |
| Banking 4 | Separate original/adaptive | `m51selplan_9b4d88e47ca502dcd88f5f1e` | `9b4d88e47ca` |
| SMS 2 | Separate original/adaptive | `m51selplan_f361b7a31d31b68b86b5adb5` | `f361b7a31d3` |
| SMS 3 | Separate original/adaptive | `m51selplan_26acb5c1f363ff25040d456e` | `26acb5c1f36` |
| SMS 4 | Separate original/adaptive | `m51selplan_cc77e94c0413047269ff3b87` | `cc77e94c041` |

The Banking 3 batch fingerprint is `4e05511d49f2861f7dfb336183e349ed2c5b03c93e05a01cca2943431d91e1da`. The plan asks the named questions for card-related content, unauthorized activity, and refund/reversal in one physical request.

The cost-only and secondary frozen mappings are persisted in `results/milestone_5_1/selection_stage_results/frozen_strategy_selections.json`; Stage D must load those objects rather than reconstructing plans from prose.

## 5. Execution schedule and deduplication

The generated schedule contains 10,800 unique physical requests over 400 held-out records per dataset. It is the union of 27 distinct workload/operator arms selected for the primary and preregistered secondary comparisons. Compatible strategy references point to the same physical request; strategy aliases must never trigger duplicate paid calls.

For each request the schedule freezes request ID, dataset, record ID, workload references, plan references, operator identity, predicate IDs, question names, prompt hashes, model, threshold, response schema, and inference settings.

Before provider initialization the future runner must verify:

- all 10,800 request hashes and the schedule fingerprint;
- all held-out partition fingerprints without inspecting outcomes;
- every selected plan fingerprint;
- prompt, model, response-schema, and configuration compatibility;
- nonexistence of a conflicting result directory or validity of a resumable session;
- a 10,800 unique-request ceiling and a 32,400 aggregate-attempt ceiling;
- concurrent atomic attempt reservation and alias deduplication.

Smoke requests must be members of the frozen schedule and count toward both ceilings.

## 6. Authorization

The proposal requires the exact phrase:

`AUTHORIZE M5.1 STAGE D INDEPENDENT HELD-OUT EVALUATION`

The generated authorization manifest has `paid_calls_authorized: false`. The phrase alone is insufficient if any schedule, source, model, prompt, partition, or plan fingerprint differs. The provider must not initialize before all checks pass. This document does not authorize execution.

## 7. Budget

| Item | Proposal |
|---|---:|
| Unique physical requests | 10,800 |
| Maximum HTTP attempts | 32,400 |
| Expected input tokens | 3,887,444 |
| Expected output tokens | 528,800 |
| Expected input-token-priced cost | $0.163272648 |
| Conservative no-retry proxy | $0.204090810 |
| Retry-inclusive proxy | $0.612272430 |

The monetary figures are proxies, not runtime dollar caps. The experiment uses $0.042 per million input tokens and $0 per output token for continuity with prior accounting. Actual tokens and cost must be reconciled separately from Stage B and selection-stage expenditure.

## 8. Primary paired comparison

For every held-out record and workload, reconstruct the request path of:

1. unchanged adaptive separate execution, including its frozen profiling policy and order selection;
2. the frozen quality-constrained/profiling-aware policy.

The two policies must use compatible record identities and individual reference measurements. The primary economic estimand is paired per-record input-token cost difference. Report request count, named decisions, input/output tokens, latency, stage reach, early rejection, absolute cost difference, and percentage difference.

For the eight fallback workloads the two policies are physically identical. Their requests must be shared, and their paired difference is structurally zero rather than remeasured through duplicate calls. The only nontrivial primary economic comparison is Banking 3.

## 9. Semantic evaluation

Compare each batched named decision with the corresponding separate Jev decision for the same dataset, record, predicate, prompt, model, threshold, and settings. Report agreement, TP, TN, relative FP and FN, precision, recall, F1, MCC, changed-decision rate, probability difference, and threshold crossing.

Reconstruct complete AND-query outcomes for each measured physical plan and report exact match-set equality, Jaccard agreement, query-level FP and FN, and positive-reference support.

Apply the unchanged gate:

- at least 40 representative records;
- at least three positive separate-reference queries;
- agreement at least 0.98;
- recall at least 0.95;
- false-negative rate at most 0.05;
- changed-decision rate at most 0.02;
- actual complete-plan evidence.

Classify passed, failed, insufficient positive support, insufficient observations, undefined metric, missing evidence, and provider failure distinctly. Separate Jev decisions are not human ground truth.

## 10. Statistical analysis

Use paired record bootstrap intervals for cost differences, with a frozen random seed and at least 2,000 resamples. Report workload and dataset results rather than relying only on aggregates. For proportions use Wilson 95% intervals with numerator and denominator.

Workloads sharing records and predicates are dependent. Do not treat predicate decisions from the same batch as independent observations in workload-level inference. Report mean and median savings, beneficial/harmful workload counts, worst downside, and dataset heterogeneity.

The primary confirmatory interpretation concerns the frozen policy. Secondary plan comparisons and measured-set regret are descriptive. Do not claim global optimality over unmeasured plans.

## 11. Profiling and reuse economics

Reconstruct the unchanged adaptive policy with checkpoint visibility exactly as frozen. For profiling-aware hybrid execution, expose only measurements acquired through each recorded checkpoint. Report actual Stage D matrix expenditure separately from standalone, shared-statistics, and warm-start counterfactual profiling costs.

Evaluate net benefit at reuse horizons 1, 5, 10, 25, 50, 100, 250, and 1,000:

\[
NetBenefit(R)=R(C_{adaptive}-C_{candidate})-C_{profiling}-C_{optimization}.
\]

Do not double-charge shared requests. If optimization CPU cost remains unmetered, report it as unavailable rather than zero-cost evidence.

## 12. Failure, resume, and reconciliation

Persist atomic budget reservations, request traces, original provider responses under the established storage contract, session status, failures, and wall time. A partial experiment must remain explicitly incomplete. Resume is allowed only after verifying the same authorization, schedule, source, configuration, model, prompts, and prior request identities; completed requests must not be reissued.

Post-execution reconciliation must account for every physical request, alias, HTTP attempt, retry, provider failure, named question, input token, output token, latency, operator, and plan reference. Produce a discrepancy report even when no discrepancy exists.

## 13. Freeze discipline and interpretation

Stage D outcomes may confirm or reject the selected policy but may not change it. If the Banking 3 saving fails to replicate, that is a valid negative result. If quality fails, the policy is rejected under the preregistered gate. If positive support remains insufficient, the correct outcome is insufficient evidence, not a relaxed threshold or a replacement plan selected from Stage D.

No production default changes, follow-on replication, or new paid experiment is authorized by this protocol.

## Versioned final-preflight addendum v1

This addendum records the offline final audit. It does not change the frozen schedule, selected plans, quality gate, provider settings, or authorization phrase.

The 10,800-request schedule is composed of 6,000 single-reference requests, 3,600 full-batch requests, and 1,200 requests for three additional unique remainder-batch arms. The earlier shorthand of 7,200 singles plus 3,600 batches is not the actual manifest composition. Fifteen of the eighteen configured dataset predicates occur in the nine workloads, and three remainder groups add unique arms while three other remainder groups alias full batches.

The schedule has 27 operator arms, 29 unique selected plan identities, 26,400 unique plan/group/record roles, and 60,400 maximum strategy/group/record roles. Exact reuse reduces these to 10,800 paid physical acquisitions; no deliberate repetition is scheduled.

The full schedule remains the recommended broad benchmark. A separately fingerprinted 1,600-request Banking77-only alternative is documented in `results/milestone_5_1/stage_d_final_preflight/reduced_scope_option.json`. It is not a replacement, has no registered authorization phrase, and requires a separate protocol and approval.

Execution is now bound to the fail-closed runner `experiments/milestone_5_1/run_stage_d.py`. Interrupted sessions are not implicitly resumed. Any ambiguous provider receipt is charged to the attempt ledger, and restart requires reconciliation plus versioned reauthorization. The post-live analyzer is frozen at `src/semanticflow/optimizer/m51_stage_d_analysis.py`.
