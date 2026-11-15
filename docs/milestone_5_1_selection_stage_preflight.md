# Milestone 5.1 selection-stage preflight

## 1. Executive summary

The offline selection-stage design is complete and internally consistent, but it is not an
authorization and the execution command remains fail-closed unless a countersigned manifest and
exact phrase validate. The frozen evidence
design covers all nine workloads, all 18 dataset-specific predicates, every full batch, and one
historically selected remainder batch for each three- or four-predicate workload. All 111 plans in
the bounded candidate catalog have complete scheduled group coverage on exactly the same 100
selection IDs per dataset.

The proposal contains 3,300 logical evidence roles: 1,800 individual references, 900 full batches,
and 600 remainder-batch roles. Exact deduplication reveals a useful correction to the Stage C
proposal. Three remainder arms are identical to already scheduled full-batch arms, so the design
requires 3,000 unique provider requests, not 3,300. The deduplicated retry-inclusive ceiling is
9,000 attempts; the original 9,900 ceiling remains an outer authorization bound and is not a target.

The expected no-retry usage is 1,066,263 input tokens and 135,500 output tokens. At the frozen
input-token rate, the experimental cost proxy is $0.044783060. The conservative no-retry proxy is
$0.062062875; the three-attempt deduplicated proxy is $0.186188625. These are forecasts, not runtime
monetary limits.

The preregistered quality gate is unchanged. Historical evidence predicts serious positive-support
risk: four measured conjunctions have zero positives in 30 records and a Jeffreys posterior
predictive probability of 77.55% of obtaining fewer than three positives in 100 records. Three
declared workloads have no compatible historical conjunction measurement at all. The selection
stage may therefore end with no batching plan eligible even if observed agreement is high.

**Readiness: READY_WITH_DOCUMENTED_LIMITATIONS.** The schedule and provenance are ready for review,
but no paid call is authorized and the launch manifest is explicitly unsigned. Provider import and
initialization occur only after every authorization and fingerprint check.

## 2. Why another experiment is necessary

Stage B completed 810 physical requests with 1,380 named evaluations, 299,406 input tokens, 37,690
output tokens, no retries, and no provider failures. Stage C found that two-question batching saved
5.97% against early-rejecting separate execution in aggregate, three-question batching tied, and
four-question batching cost 25.14% more. Only one of six fully measurable workloads improved under
full batching. Exploratory hybrids saved 6.12% on the Banking four-predicate workload and 4.50% on
the SMS three-predicate workload relative to original-order separate execution.

Those results cannot freeze a held-out plan. Only 30 representative records were measured per group;
three workloads lacked compatible matrices; no batching plan passed the gate; and the established
`existing_adaptive` baseline was not independently selected and evaluated. The untouched selection
partition exists specifically to resolve these deficiencies before Stage D.

## 3. Workload coverage

| Dataset | Workload | Predicates | Frozen measured plans | Remainder first | Remainder group | Coverage |
|---|---|---:|---:|---|---|---|
| Beauty | `m51_beauty_failure_usability` | 2 | 3 | — | — | all singles + full batch |
| Beauty | `m51_beauty_reaction_packaging_refund` | 3 | 8 | adverse reaction | packaging, return | complete |
| Beauty | `m51_beauty_reaction_failure_value_return` | 4 | 26 | adverse reaction | malfunction, poor value, return | complete |
| Banking | `m51_banking_unauthorized_refund` | 2 | 3 | — | — | all singles + full batch |
| Banking | `m51_banking_card_unauthorized_refund` | 3 | 8 | card related | unauthorized, refund | complete; remainder reuses size-2 full batch |
| Banking | `m51_banking_withdrawal_unauthorized_refund_access` | 4 | 26 | access/verification | unauthorized, refund, withdrawal | complete |
| SMS | `m51_sms_prize_contact` | 2 | 3 | — | — | all singles + full batch |
| SMS | `m51_sms_prize_urgency_contact` | 3 | 8 | urgency | prize, contact | complete; remainder reuses size-2 full batch |
| SMS | `m51_sms_security_prize_urgency_contact` | 4 | 26 | security | prize, urgency, contact | complete; remainder reuses size-3 full batch |

The catalog contains every separate-predicate permutation, one full multi-decision batch, and the
one frozen selective-first hybrid for each eligible workload. It deliberately excludes arbitrary
remainder groups, arbitrary multi-group partitions, and conjunctive-binary calls. Those plans are
unsupported by this selection experiment and cannot later be called measured. Zero observed reach
never waives a required group.

The remainder choices use Stage B only. Banking-four and SMS-three use the cheapest fully measured
historical selective-first candidates. Other eligible workloads retain a deterministic original-first
fallback or a previously measured tied candidate. No selection outcome was read.

## 4. Existing adaptive baseline

The baseline is the existing implementation in
`semanticflow.optimizer.adaptive.simulate_adaptive_profiling`, invoked through the established
Milestone 5 wrapper and the unchanged `configs/optimizer.milestone_4.yaml`. Its sampling order is
fixed by `sha256(4049:record_id)` before any outcome is visible. The usable checkpoints are 25, 50,
and 100 complete rows. At checkpoint *k*, the controller receives only the first *k* rows and all
predicate results on those rows. It cannot inspect later rows.

The 1,800 shared singles provide every measurement the baseline can require. Profiling expenditure is
the exact input-token-priced cost of the single calls revealed through the stopping checkpoint. The
selected order is replayed from the complete matrix and labelled selection evidence. The complete
matrix remains an experiment expenditure even when a counterfactual controller would stop early.
No change was made to the controller, thresholds, cost calculations, selectivity estimators, stopping
rules, or production default.

## 5. Physical-plan provenance

Each catalog entry records dataset, workload, logical predicate set, exact ordered groups, group
execution mode, predicate and provider-visible prompt hashes, question names, model, threshold,
response schema, inference settings, planner version, configuration fingerprint, strategy
eligibility, pending cost/quality state, and fallback. Canonical JSON serialization produces a
SHA-256 plan fingerprint. Strategy labels are deliberately excluded from physical identity: two
strategies selecting the same physical execution share the plan and its measurements. Changing a
group, prompt, mode, model, threshold, or execution configuration changes the fingerprint.

Every plan maps explicitly to its required operator measurements for all 100 IDs. Validation found
111/111 catalog plans fully scheduled and no unsupported catalog plan masquerading as complete.

## 6. Quality feasibility

| Workload | Historical n | Reference positives | Expected positives/100 | P(fewer than 3) | Interpretation |
|---|---:|---:|---:|---:|---|
| Beauty failure/usability | 0 | unavailable | unavailable | unavailable | missing compatible historical group |
| Beauty reaction/packaging/refund | 0 | unavailable | unavailable | unavailable | missing compatible historical group |
| Beauty reaction/failure/value/return | 30 | 0 | 1.61 | 77.55% | high sparse-support risk |
| Banking unauthorized/refund | 30 | 2 | 8.06 | 13.75% | potentially evaluable |
| Banking card/unauthorized/refund | 0 | unavailable | unavailable | unavailable | missing compatible historical group |
| Banking withdrawal/unauthorized/refund/access | 30 | 0 | 1.61 | 77.55% | high sparse-support risk |
| SMS prize/contact | 30 | 2 | 8.06 | 13.75% | potentially evaluable |
| SMS prize/urgency/contact | 30 | 0 | 1.61 | 77.55% | high sparse-support risk |
| SMS security/prize/urgency/contact | 30 | 0 | 1.61 | 77.55% | high sparse-support risk |

The predictive calculation uses a Jeffreys Beta(0.5, 0.5) posterior and a beta-binomial predictive
distribution. It is a sparse-sample planning model, not a population fact. Zero historical positives
does not mean zero prevalence.

The gate remains: at least 40 records, at least three positives, agreement at least 0.98, recall at
least 0.95, FNR at most 0.05, changed-decision rate at most 0.02, and real complete-plan evidence.
With 3–19 positives, one false negative already violates both recall and FNR, so zero false negatives
are required. At 100 total records, at most two final decisions may change. A zero-positive result
has undefined recall and is classified as insufficient support, never as validated.

## 7. Cost estimator

Stage C's existing unconditional group-reach model was audited on the six historically measurable
workloads. The in-sample diagnostics were:

| Metric | Result |
|---|---:|
| Mean workload MAE | $0.000004573 |
| Workload RMSE aggregate | $0.000008399 |
| Mean pairwise ranking concordance | 93.70% |
| Mean signed bias | -$0.000003704 |
| Total selection regret | $0.000030618 |
| Harmful predicted selections | 2 of 6 |

The intended correction estimates each stage using joint prefix pass probability and conditional
group cost:

`N * sum_i P(previous groups pass) * E[cost(G_i) | previous groups pass]`.

It is versioned as `m5.1-correlation-aware-prefix-v1`, but is not adopted for selection. The saved
30-record pilot diagnostics do not provide a non-tautological checkpointed validation of conditional
costs; fitting exact prefixes and scoring them on the same records would guarantee an artificial
improvement. The existing model is retained with uncertainty labels. Where joint evidence is absent,
the documented fallback multiplies marginal predicate pass rates; missing evidence is never assigned
zero cost.

## 8. Leakage-free adaptive profiling

Every controller state must persist checkpoint, available measurement IDs, candidate plans, cost
estimates, uncertainty, chosen profiling action, value of more profiling, expenditure, stop decision,
and selected plan. Availability is monotone and checkpoint-bounded. A complete stored matrix may be
used only by revealing rows in the frozen order.

Accounting reports four distinct quantities:

1. Actual full-matrix expenditure: every unique provider call exactly once.
2. Standalone workload expenditure: compatible calls needed by that workload.
3. Shared-statistics expenditure: each shared call once across workloads.
4. Warm-start incremental expenditure: only new compatible calls.

Cross-strategy sharing is off unless the declared scenario permits it. The controller is charged only
for actions its replay policy would acquire, while research accounting still reports the full matrix.

## 9. Expected Jev expenditure

| Dataset | Singles | Unique full/remainder batches | Unique requests | Expected input | Expected output | Expected cost |
|---|---:|---:|---:|---:|---:|---:|
| Amazon All Beauty | 600 | 500 | 1,100 | 411,933 | 50,700 | $0.017301200 |
| Banking77 | 600 | 400 | 1,000 | 346,550 | 47,150 | $0.014555100 |
| SMS Spam | 600 | 300 | 900 | 307,780 | 37,650 | $0.012926760 |
| **Total** | **1,800** | **1,200** | **3,000** | **1,066,263** | **135,500** | **$0.044783060** |

Forecasts use actual Stage B means by dataset, physical mode, and question count. Conservative tokens
use the greater of the observed maximum and 1.25 times the mean for each matching stratum.

| Budget view | Input-token cost proxy |
|---|---:|
| Expected no retry | $0.044783060 |
| Conservative no retry | $0.062062875 |
| Conservative deduplicated three-attempt proxy | $0.186188625 |
| Original 9,900-attempt outer proxy | $0.204807488 |

Output tokens are forecast and reported but carry a frozen rate of $0. Monetary values are not hard
limits; request and attempt ceilings are. The historical $0.012575052 Stage B expenditure is excluded
from the incremental selection forecast.

## 10. Experiment integrity

The frozen schedule fingerprint is
`2083c9321d4ed99ab1a48c527f71fd378d125d66868502751ab55e9e858a394d`.
Dataset fingerprints are:

| Dataset | Dataset fingerprint | Selection fingerprint |
|---|---|---|
| Beauty | `988170c284a8f32f38e171f8bcf9e4aeb870a37149c09ae5a7c7db4ba43a9a51` | `1d83d9e6272a6f97811bad78c2ecbc93140dca0fb65653a16ebcca23258f5270` |
| Banking | `e138176cf7da4e7f54ec25b6a4ec50bf8e74c1cc55360c700083d1a39475cf9e` | `c94684f67e41ea01d7ea88f4ad38a187fba49de625cd8d6bf922d4d79b2bd7f1` |
| SMS | `0aa8516d1d05e68646c7743dedccd97e7ab77654150bc5ec6af5bc7ca7a25463` | `582b301a949d61e0b8296f64d453746c2a9b61bec76ce9387534fe51c44dbaa1` |

The unsigned authorization manifest binds the schedule, model `jev-1.14.0`, prompt/operator
identities, datasets, 3,300-role ceiling, 3,000-request ceiling, and 9,900 outer attempt ceiling. It
requires the exact phrase `AUTHORIZE M5.1 SELECTION STAGE FROZEN MEASUREMENT MATRIX`, rejects output
conflicts and fingerprint changes, forbids resume, and requires provider initialization to occur only
after validation. Four predetermined physical request IDs cover one-, two-, three-, and four-question
provider contracts; they are included in the schedule and are not extra calls.

## 11. Limitations

* Three workloads lack compatible historical prevalence estimates.
* Most measured conjunctions have zero pilot positives, so selection may not make recall evaluable.
* The estimator diagnostics are in-sample and do not justify adopting the prefix correction.
* Only the bounded 111-plan catalog is measurable; most theoretical four-predicate partitions remain
  unsupported.
* Selection data may select and quality-gate plans, but cannot estimate chosen-plan generalization
  without selection bias.
* Separate Jev decisions are a consistency reference, not human ground truth.
* The launch tool is implemented but cannot initialize the provider while the generated manifest
  remains unsigned; execution requires a separately reviewed, countersigned manifest.

## 12. Authorization readiness

The scientifically frozen measurement proposal is ready for authorization review. It resolves the
Stage B coverage defect, supports the actual adaptive baseline, records complete plan provenance,
preserves the gate, and provides bounded usage forecasts. It does not authorize or execute anything.
Because sparse positives remain likely and the execution path is intentionally fail-closed, the
status is **READY_WITH_DOCUMENTED_LIMITATIONS**, not Stage D readiness.
