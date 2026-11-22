# SemanticFlow Milestone 5.1 Stage D Final Preflight

## 1. Executive summary

The original full Stage D proposal is internally consistent and ready for separate paid authorization. Its frozen schedule contains 10,800 unique physical requests, 27 unique operator arms, and a maximum of 32,400 HTTP attempts. The schedule fingerprint recomputes to `6d6fc867d221c9ce1ac29011d0410558b939df842563a5d3116816401f1139f6`.

The audit found an important correction to the informal explanation of the schedule. It is not 7,200 individual references plus 3,600 full batches. The real schedule contains:

- 6,000 individual-reference requests for the 15 predicate definitions used by the nine workloads;
- 3,600 full-batch requests, one workload batch for each of nine workloads and 400 records;
- 1,200 requests for three additional unique remainder-batch operators used by frozen selective-first controls.

Three other remainder groups are exact aliases of scheduled full batches. No physical request is deliberately repeated. The schedule supports 26,400 unique plan/group/record roles through 15,600 aliases and as many as 60,400 strategy-specific group/record roles through 49,600 aliases.

Only 1,600 requests, 14.81% of the full schedule, are necessary for the primary Banking77 confirmatory matrix: three complete individual-reference arms and the selected three-question batch over the same 400 records. A separately versioned reduced option is therefore documented. It would cost an expected $0.022366176 instead of $0.163272648 but would sacrifice cross-workload fallback validation, broad negative controls, batch-size generalization, three remainder controls, and held-out ranking/regret evaluation. It has not replaced the original schedule and has no authorization phrase.

The full design is scientifically useful because its research objective includes testing whether the optimizer correctly avoided batching in the other eight workloads. If the objective were limited to confirming the Banking77 result, the reduced design would be sufficient and substantially cheaper. This preflight recommends preserving the existing full frozen schedule because the user requested the complete Stage D design and because its secondary negative-result evidence is explicitly preregistered. Separate authorization remains mandatory.

No Jev request was made. Only partition identities and fingerprints were read; no held-out text, provider outcomes, selectivities, or labels were inspected.

## 2. Research questions

The confirmatory question is whether the frozen Banking77 three-question multi-decision plan is cheaper than the unchanged adaptive separate plan on 400 paired held-out records while passing the unchanged semantic-consistency gate.

The primary cost effect is:

\[
\Delta C=C_{existing\_adaptive}-C_{batch}.
\]

A positive value favors batching. The primary analysis reports absolute and percentage saving, input/output-token differences, request difference, per-record paired differences, and a paired record-bootstrap 95% interval using 10,000 resamples and seed 51103.

The result classifications are frozen in advance:

- Confirmed economic and quality success: the cost-saving interval lower bound is above zero and the unchanged gate passes.
- Economic failure: the interval upper bound is at or below zero.
- Inconclusive cost: the interval includes zero.
- Quality failure: any gate requirement fails, regardless of cost.
- Insufficient quality support: fewer than three separate-reference-positive queries.

Secondary questions evaluate all nine full batches, the three unique remainder controls, two/three/four-question economics, fallback correctness, estimator error within the measured union, and profiling-inclusive net benefit. They remain descriptive and do not establish global optimality.

## 3. Frozen optimizer selections

Canonical serialization and plan hashes were recomputed from dataset, workload, predicates, group order, execution modes, prompts, model, threshold, response schema, execution settings, planner version, and configuration fingerprint. All nine matched their frozen artifacts.

| Workload | Frozen plan | Mode | Full plan hash |
|---|---|---|---|
| Beauty failure/usability | `m51selplan_d1572e4176aa19745ecec3ab` | Adaptive separate fallback | `d1572e4176aa19745ecec3ab4235888d06f7adb9478041d39d2531ef365c35f8` |
| Beauty reaction/packaging/refund | `m51selplan_c7c4f8bf8e099559ff909280` | Adaptive separate fallback | `c7c4f8bf8e099559ff9092806acdc9a74b6f8492b3f19bf809eb0639c667b616` |
| Beauty reaction/failure/value/return | `m51selplan_c9b259bbcab3297242646a30` | Adaptive separate fallback | `c9b259bbcab3297242646a3004d52fb96b22bb7abfd20e7027b53d414a19f700` |
| Banking unauthorized/refund | `m51selplan_330710722b4d97b2a056da04` | Adaptive separate fallback | `330710722b4d97b2a056da04aa9536133adca29b950568767b5a1b80d8eb1f3c` |
| Banking card/unauthorized/refund | `m51selplan_4e05511d49f2861f7dfb3361` | Full multi-decision batch | `4e05511d49f2861f7dfb336183e349ed2c5b03c93e05a01cca2943431d91e1da` |
| Banking withdrawal/unauthorized/refund/access | `m51selplan_9b4d88e47ca502dcd88f5f1e` | Adaptive separate fallback | `9b4d88e47ca502dcd88f5f1ece02a2aa282a7024754054cabe08f98c7ac1875b` |
| SMS prize/contact | `m51selplan_f361b7a31d31b68b86b5adb5` | Adaptive separate fallback | `f361b7a31d31b68b86b5adb514eb89e49d68cc8f275ca38f622736778ff6ad92` |
| SMS prize/urgency/contact | `m51selplan_26acb5c1f363ff25040d456e` | Adaptive separate fallback | `26acb5c1f363ff25040d456ecfba999d3f0f9a94d21721b86be72f6c8c2f4364` |
| SMS security/prize/urgency/contact | `m51selplan_cc77e94c0413047269ff3b87` | Adaptive separate fallback | `cc77e94c0413047269ff3b87e15e7499a2ccf8f1d8416fbb0cb3297923f96ee3` |

The eight fallbacks exactly match the unchanged `existing_adaptive` plans. No batching group or new profiling action was introduced into those selected policies.

## 4. Selected Banking77 candidate

The selected plan is one physical multi-decision request with three provider-visible question names. It is not a conjunctive-binary prompt, three HTTP requests, or a synthetic composition of separately measured answers. Response validation requires one named probability for each of:

- `banking_card_related`;
- `banking_unauthorized_activity`;
- `banking_refund_or_reversal`.

The final query result is the AND of the three thresholded named decisions at threshold 0.5.

On the selection partition, adaptive cost was $0.001778280 and batch cost was $0.001647786 per 100 records, a $0.000130494 or 7.34% point saving. Complete-query agreement was 100/100, with only three positive references. The paired cost interval was -$0.000012474 to +$0.000284424, so the population saving was not established. Passing the selection gate also does not establish a population semantic guarantee.

## 5. Full experiment schedule

### Actual composition

| Dataset | Records | Singles | Batches | Operator arms | Requests | Max attempts |
|---|---:|---:|---:|---:|---:|---:|
| Amazon All Beauty | 400 | 2,400 | 2,000 | 11 | 4,400 | 13,200 |
| Banking77 | 400 | 2,000 | 1,600 | 9 | 3,600 | 10,800 |
| SMS Spam | 400 | 1,600 | 1,200 | 7 | 2,800 | 8,400 |
| **Total** | — | **6,000** | **4,800** | **27** | **10,800** | **32,400** |

The 4,800 batch requests consist of nine full-batch arms (3,600 requests) and three additional unique remainder arms (1,200). The full schedule includes no deliberate stochastic repetitions.

### Exclusive purpose allocation

| Purpose | Arms | Requests | Expected input | Expected cost |
|---|---:|---:|---:|---:|
| Banking77 primary confirmatory matrix | 4 | 1,600 | 532,528 | $0.022366176 |
| Other separate-reference evidence | 12 | 4,800 | 1,598,560 | $0.067139520 |
| Eight other full-batch controls | 8 | 3,200 | 1,272,072 | $0.053427024 |
| Three additional remainder controls | 3 | 1,200 | 484,284 | $0.020339928 |
| **Total** | **27** | **10,800** | **3,887,444** | **$0.163272648** |

Individual operators can have multiple scientific uses, but each physical request appears in exactly one budget row above.

### Per-workload coverage

The following counts are overlapping coverage counts and must not be summed because predicates and remainder batches are shared across workloads.

| Workload | Unique arms supporting declared comparisons | Physical rows covered |
|---|---:|---:|
| Beauty 2 | 3 | 1,200 |
| Beauty 3 | 5 | 2,000 |
| Beauty 4 | 6 | 2,400 |
| Banking 2 | 3 | 1,200 |
| Banking 3 | 5 | 2,000 |
| Banking 4 | 6 | 2,400 |
| SMS 2 | 3 | 1,200 |
| SMS 3 | 5 | 2,000 |
| SMS 4 | 6 | 2,400 |

Banking 3 has five arms in the full schedule because its fixed selective-first control also uses the Banking 2 batch as a remainder. Only four arms are required for the primary comparison.

## 6. Reduced-scope alternative

The reduced option contains 1,600 requests:

- 400 card-related separate references;
- 400 unauthorized-activity separate references;
- 400 refund/reversal separate references;
- 400 selected three-question batches.

| Metric | Full option | Reduced option |
|---|---:|---:|
| Unique requests | 10,800 | 1,600 |
| Maximum attempts | 32,400 | 4,800 |
| Expected input tokens | 3,887,444 | 532,528 |
| Expected output tokens | 528,800 | 69,600 |
| Expected cost | $0.163272648 | $0.022366176 |
| Conservative no-retry | $0.204090810 | $0.027957720 |
| Retry-inclusive proxy | $0.612272430 | $0.083873160 |

The reduction is 9,200 requests, or 85.19%.

All three complete single-predicate arms are retained. A deployment replay may skip later predicates through early rejection, but the records reaching those stages depend on unseen outcomes. A complete single matrix provides paired individual quality evidence and permits exact counterfactual replay without outcome-dependent schedule construction.

The reduced option loses independent evidence about all eight fallback decisions, broad batch-size and dataset effects, the three additional remainder controls, and ranking/regret across the frozen union. It is a new protocol with fingerprint `678cc11f2ffe4a841d47577f4f2c0963862da00ee9c73a26a7cd8ca8a33be9e5`; it has no preregistered authorization phrase and is not automatically recommended or authorized.

## 7. Measurement completeness

The full schedule supports:

- paired adaptive-versus-full-batch execution and query comparison for all nine workloads;
- complete individual named-decision comparison for every full batch;
- the selected quality-constrained policy and eight fallbacks;
- six declared fixed-hybrid plans, where three remainder groups reuse full-batch measurements and three are additional arms;
- two-, three-, and four-question batch economics;
- measured-union estimator error, ranking, regret, and fallback correctness;
- standalone, shared-statistics, warm-start, and actual-research-spend accounting.

It does not support arbitrary unselected catalog plans, new conjunctive-binary fusion calls, or provider-repeatability claims. There are no deliberate repeated physical calls. The 29 unique plan IDs in the schedule are the measured union, not the full 111-plan catalog or all theoretical plans.

## 8. Held-out and quality integrity

Each dataset manifest contains exactly 400 evaluation IDs. The evaluation sets have zero identity overlap with development, pilot, or selection partitions, and no identity is duplicated across frozen partitions. The schedule IDs equal the manifest evaluation IDs exactly for all datasets.

Evaluation fingerprints are:

| Dataset | Evaluation fingerprint |
|---|---|
| Amazon All Beauty | `637cfadffb2f19afec775a97afef6411f8d885fed10995718fe303cf78bf78bd` |
| Banking77 | `0c8139bb3c39019ed27e600ae5b0cdbd24bc60c52541fe49d54598570c63160e` |
| SMS Spam | `afe8465451332a9fac2b06ed54df5fe83ab37ce66e2950d220ccc41d3c178617` |

Only manifest identities and fingerprints were read. Held-out record text and inference outcomes were not loaded by the preflight.

The gate remains: at least 40 observations, at least three positive references, agreement ≥0.98, recall ≥0.95, false-negative rate ≤0.05, changed-decision rate ≤0.02, and measured complete-plan evidence. Results distinguish passed, failed, insufficient positives, insufficient observations, undefined metrics, missing evidence, and provider failure. Separate Jev execution is a consistency reference rather than human truth.

With only three selection positives, the Banking candidate allowed no false negatives while meeting 0.95 recall. Stage D's larger sample may improve positive support, but the preflight did not inspect outcomes and makes no positive-count promise.

## 9. Profiling economics

After authorized Stage D execution, measured held-out savings replace the selection forecast in:

\[
NetBenefit(R)=R(C_{existing\_adaptive}-C_{candidate})-C_{profiling}-C_{optimization}.
\]

Reuse horizons are 1, 5, 10, 25, 50, 100, 250, and 1,000. Scenarios remain separate:

| Scenario | Selection-based profile cost | Forecast break-even |
|---|---:|---:|
| Standalone cold start | $0.007067130 | 55 executions |
| Shared-statistics cold start | $0.002967972 | 23 executions |
| Warm start | $0 | Immediate |
| Actual research expenditure | Charged once by physical acquisition | Not a deployment break-even |

The existing adaptive baseline is charged $0 profiling because it declined profiling at the frozen ten-execution horizon. Unmetered optimization CPU cost remains unavailable rather than being asserted to be economically zero.

Historical M5, M5.1 pilot, selection, failed-transport, and future Stage D spending remain separate. The three failed selection smoke attempts stay in the attempt ledger; their provider-side billing status is unknown.

## 10. Budget and cost proxies

Forecasts use per-operator input/output means from the completed selection matrix. No held-out text was used to refine them.

| Dataset | Requests | Input tokens | Output tokens | Expected cost | Conservative | Retry proxy |
|---|---:|---:|---:|---:|---:|---:|
| Beauty | 4,400 | 1,687,836 | 212,800 | $0.070889112 | $0.088611390 | $0.265834170 |
| Banking77 | 3,600 | 1,253,588 | 185,200 | $0.052650696 | $0.065813370 | $0.197440110 |
| SMS | 2,800 | 946,020 | 130,800 | $0.039732840 | $0.049666050 | $0.148998150 |
| **Total** | **10,800** | **3,887,444** | **528,800** | **$0.163272648** | **$0.204090810** | **$0.612272430** |

The recomputed totals match the stored proposal exactly. Cost uses $0.042 per million input tokens and $0 per output token. These are experimental proxies, not invoices or enforceable dollar caps. The technical limits are 10,800 unique requests, two retries per request, and 32,400 attempts.

## 11. Statistical analysis

The primary comparison is paired by held-out record. A record-cluster bootstrap preserves all stage costs and decisions from the same record. Wilson intervals report quality proportions with numerators and denominators.

Secondary workload and dataset summaries are not treated as nine or three independent replications. Workloads share predicates and records; named decisions within a batch are also dependent. The analysis reports workload-level heterogeneity, but claims are limited by three datasets and nine workloads.

The post-live analyzer was implemented before outcomes were available. It produces request and spending reconciliation, dataset/workload cost results, batch-size economics, individual and query agreement, the unchanged gate, estimator error, measured-set regret, fallback correctness, profiling economics, uncertainty, and negative-result analysis. Missing measurements are reported unavailable, never filled with zero.

## 12. Execution safety

The new Stage D runner imports and initializes the paid provider only after validating authorization, scope fingerprint, source files, dataset/partition fingerprints, schedule, all scheduled plan hashes, model, retry policy, output directory, and complete alias coverage.

It enforces:

- model `jev-1.14.0`, threshold 0.5, cache disabled, concurrency 4;
- at most two retries and 32,400 aggregate attempts;
- atomic worst-case reservation before each operator arm;
- four smoke requests drawn from the frozen schedule, not added to it;
- strict question-name and predicate-probability validation;
- one physical call count per request regardless of named decision count;
- atomic session updates and completed-request identities;
- rejection of conflicting outputs and implicit resumes.

An incomplete session is fail-closed with `resume_permitted: false`. Ambiguous provider receipt is charged as an attempt and is not silently reissued. Restart requires reconciliation and a versioned authorization amendment; the runner cannot automatically consume a fresh full ceiling.

Negative tests cover absent/wrong authorization, unsigned manifests, prompt or schedule changes, plan changes, dataset-fingerprint changes, attempt-policy changes, alias conflicts, existing outputs, unsupported operators, and partial named responses. Tests do not initialize the Jev provider.

## 13. Negative-result interpretation

If batching is costlier, the measured loss and interval are reported without plan substitution. If the batch changes decisions or fails the gate, the frozen evaluation remains a quality failure; thresholds and records are not altered. If the cost interval crosses zero, the economic conclusion is inconclusive even with a positive point estimate. If positive references are sparse, recall is insufficiently supported. Net-negative profiling is reported by scenario and reuse horizon.

For the eight fallbacks, full-batch controls distinguish correct avoidance from excessive conservatism. A cheaper, independently quality-admissible full batch is a forgone opportunity; it is reported but cannot replace the frozen policy using Stage D outcomes.

## 14. Authorization proposal

The recommended scope is the original full frozen schedule:

- schedule: `results/milestone_5_1/selection_stage_results/stage_d_execution_schedule.json`;
- proposed manifest: `results/milestone_5_1/stage_d_final_preflight/stage_d_authorization_manifest.json`;
- future countersigned manifest: `results/milestone_5_1/stage_d_final_preflight/stage_d_authorization_manifest_authorized.json`;
- output: `results/milestone_5_1/stage_d_live`;
- physical ceiling: 10,800;
- attempt ceiling: 32,400.

The current proposal has `paid_calls_authorized: false`. After separate explicit authorization, a versioned countersigned copy must preserve the same scope fingerprint and set authorization true. The validation and execution commands are:

```powershell
.\.venv\Scripts\python.exe -m experiments.milestone_5_1.run_stage_d validate --manifest results/milestone_5_1/stage_d_final_preflight/stage_d_authorization_manifest_authorized.json --schedule results/milestone_5_1/selection_stage_results/stage_d_execution_schedule.json --output results/milestone_5_1/stage_d_live --authorization "AUTHORIZE M5.1 STAGE D INDEPENDENT HELD-OUT EVALUATION"
```

```powershell
.\.venv\Scripts\python.exe -m experiments.milestone_5_1.run_stage_d execute --manifest results/milestone_5_1/stage_d_final_preflight/stage_d_authorization_manifest_authorized.json --schedule results/milestone_5_1/selection_stage_results/stage_d_execution_schedule.json --output results/milestone_5_1/stage_d_live --authorization "AUTHORIZE M5.1 STAGE D INDEPENDENT HELD-OUT EVALUATION"
```

The phrase above comes from the frozen protocol and authorization artifact; it was not invented by this preflight. Merely documenting it does not authorize execution.

## 15. Final readiness decision

The plan hashes, schedule, partitions, budget, analyzer, and fail-closed runner validate successfully. The full schedule is larger than the primary comparison alone requires, but every request has a preregistered primary or secondary purpose. The cost interval and sparse positives remain scientific limitations to be tested, not execution defects.

**READY_FOR_M5_1_STAGE_D_AUTHORIZATION**
