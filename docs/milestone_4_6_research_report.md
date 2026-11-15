# Milestone 4.6 research report

> **Stage C completion addendum (2026-11-13):** Stage B and the separately authorized Stage C
> full restart have completed. The restart followed the frozen 42-run schedule and used 36,796
> requests with zero retries or failures. The prior path-length incident is retained as an
> incomplete, excluded session. See
> [Milestone 4.6 Stage C Preflight Hardening Report](milestone_4_6_stage_c_preflight_report.md).
> Neither the incident correction nor the restart altered the preregistered hypotheses,
> policies, prompts, partitions, selected plans, or schedule. Optional replication was not run.

Status: offline preparation, Stage B live profiling, the Stage C held-out restart, and the
preregistered offline analysis completed on 2026-11-13. No Stage D replication is authorized.

## 1. Research motivation

Milestones 1–4.5 developed and calibrated SemanticFlow on four repeatedly used Amazon Arts,
Crafts & Sewing workloads. Milestone 4.6 freezes those methods and moves to two new domains so
that the evaluation can measure generalization rather than another round of optimizer tuning.

## 2. Historical findings

Milestone 3 observed correlation-aware execution savings of 0.36%, 5.44%, 0%, and 45.02% across
the four historical workloads. Milestone 4.5 reported correlation and independence ranking
concordance of 0.921 and 0.904; selection regret of 37.9 and 84.8 tokens; and nominal 95%
bootstrap coverage of about 77%. At R=10, retrospective net savings were $0.103387 for existing
adaptive, $0.096946 for the tuned joint controller, and $0.096513 for adaptive plus risk-aware
selection. The stored report and those exact figures reproduce without new inference.

## 3. Independent dataset selection

The prepared datasets are Amazon Reviews'23 `All_Beauty` and Banking77. Their source corpora
contain 700,808 valid reviews and 13,083 banking queries. Each frozen snapshot contains 200
profiling and 800 evaluation records. The respective dataset fingerprints are
`dcb6df57fb239160a111c9d34b2f0e02b9c252eadc071e1dc1b259f721b189a1` and
`2afaeb6717ba4fe205d9c6ac90c52e3349596af58e3874f382c2b4c8a56a849f`.

Preparation removed 123 exact candidate duplicates from All Beauty and 52 exact plus two near
duplicates from Banking77. Both snapshots have zero ID, group, exact-text, or near-duplicate
overlap across partitions and zero exact or near-text overlap with the original SemCEB sample.
The All Beauty and Banking77 output SHA-256 values are respectively
`758b3ebabf0c6d3d3e033e7b6c6e197cad51aff4d5a898ab75272b1b7cf7b2ce` and
`35f7ea9a2bcf2d371bb095dbd62b551be39634097535f23ec392a5522d71e655`.

## 4. New workload definitions

Ten workloads were frozen: five beauty-review conjunctions and five banking-support
conjunctions. They cover strong and weak hypothesized correlation, sparse events, near ties,
two/three/four-predicate planning, genuine prompt-cost differences, and plausible harmful
reorders. Full exact instructions, rationales, original orders, and prompt hashes are in the
workload freeze. None of these characteristics has yet been measured by Jev; they remain design
hypotheses.

## 5. Frozen optimizer configurations

All eight required strategies remain unchanged. Their source/config fingerprints are stored in
`strategy_freeze.json`. The Milestone 4.5 parameter fingerprint exactly matches the historical
manifest, and its full YAML hash is verified. No historical configuration or result was edited.
The 293 historical files (135,725,946 bytes) have aggregate integrity SHA-256
`f8c37822ec3121fa048bd563dd406e1570c454edcd9fb327ab4491b1c0b2b752`.

## 6. Hypotheses and success criteria

The primary and six secondary hypotheses, exact estimands, tests, interval levels, reuse
horizons, cluster units, stopping rules, and multi-criterion success definition are frozen in
the protocol. Success is not an unweighted mean percentage: the analysis includes median,
workload counts, dataset stability, cost weighting, maximum downside, and leave-one-out
sensitivity.

## 7. Experimental methodology

Stage B creates one full 200-by-6 profiling matrix per dataset. All plans are selected with only
those matrices and frozen at R=10. Stage C deduplicates strategy orders, uses ABBA for the fixed
versus primary differing comparator, measures other unique orders once, and evaluates identical
held-out records with cache disabled. Stage D is optional and separately authorized.

Paired records support cost differences; workload and dataset are the inferential units.
Provider variability is represented only by genuine repeated blocks. The analysis does not
treat thousands of predicate requests as independent experiments.

## 8. Offline infrastructure verification

The independent data pipeline, workload parser, permutation generator, shared profiler, matrix
leakage guard, all eight policy paths, authorization guards, ABBA scheduler, retry-inclusive
budgeting, cold/warm economics, incomplete-run handling, and structured analyzer have been
implemented. The deterministic mock preflight executed 2,400 synthetic decisions, produced two
complete matrices, and exercised every frozen strategy on all ten workloads. It initialized no
Jev provider and made zero network or paid calls. Mock outcomes are not semantic evidence.
The environment artifact pins Python, direct dependency versions, `pyproject.toml`, `uv.lock`,
the Git commit, platform, and executable path.

## 9. Profiling and live execution protocol

Jev remains pinned to `jev-1.14.0`; concurrency is four, timeout 30 seconds, maximum retries two,
and cache disabled. Each request records decisions, attempts, failures, tokens, latency, model,
and hashes. Stage B and Stage C use different exact authorization phrases. The provider is
imported and initialized only after the corresponding guard succeeds.

Authorized Stage B completed both shared matrices on `jev-1.14.0`. All 2,400 logical decisions
completed in 2,400 HTTP attempts with zero retries, failures, or provider errors. All Beauty used
404,034 input and 31,200 output tokens for $0.016969428; Banking77 used 373,074 input and 30,400
output tokens for $0.015669108. Total measured profiling cost was $0.032638536. No evaluation
record was used in profiling or plan selection.

Observed profiling selectivities were:

| Dataset | Predicate selectivities |
|---|---|
| All Beauty | malfunction 0.190; hard-to-use 0.095; adverse reaction 0.030; packaging failure 0.045; return/refund 0.020; poor value 0.045 |
| Banking77 | unauthorized activity 0.080; card-related 0.330; transfer-related 0.180; cash withdrawal 0.085; refund/reversal 0.155; access/verification 0.110 |

Frozen plan selection then completed from profiling evidence only. Existing adaptive reordered
nine of ten workloads, correlation-aware reordered six, risk-aware reordered five, and the frozen
Milestone 4.5 joint policy reordered six. These are plan-selection outcomes, not evidence that the
reorders improve held-out execution.

## 10. Statistical analysis plan

The analyzer reports stage-cardinality MAE/MAPE/bias, total-evaluation and monetary-cost error,
ranking concordance over executed orders, executed-order selection regret, cheapest-executed
frequency, and 50/80/90/95% interval coverage. It also counts harmful risk reorders and rejected
beneficial reorders, estimates downside avoided and conservative fallback regret, compares
matching-record ID sets, and generates dataset and leave-one-workload-out aggregates.

## 11. Cost and budget accounting

Stage B was authorized for exactly 2,400 logical decisions and at most 7,200 attempts. It used
2,400 attempts and cost $0.032638536 from provider-reported tokens. The preregistration proxy
underestimated measured cost because actual prompt accounting was higher; the run remained below
the authorized attempt ceiling and its conservative retry-inclusive proxy was not an expenditure
cap.

The frozen real-plan Stage C schedule contains 42 executions across 24 workload-specific unique
orders. Its all-predicates-reached ceiling is 103,200 logical decisions and 309,600 HTTP attempts.
The hardened early-rejection forecast predicted 36,759.43 evaluations and $0.515815590 while
preserving the $1.828022490 conservative no-retry and $5.484067470 retry-inclusive proxies.
Those monetary figures were empirical planning proxies, not hard spending caps.

The successful restart used 36,796 logical evaluations and attempts, 12,103,700 input tokens,
954,161 output tokens, and $0.508355400. All Beauty accounted for 20,248 evaluations and
$0.291362820; Banking77 accounted for 16,548 and $0.216992580. There were no retries or failures.

The first authorized attempt failed closed on a Windows atomic-temporary-path limit after 7,742
attempts. It has $0.099605058 of fully persisted cost plus 822 attempts whose exact cost is not
recoverable; no value is imputed. The successful restart is analyzed independently. Known paid
cost across Stage B, the persisted portion of the failed attempt, and the restart is $0.640598994,
plus the unknown cost of those 822 attempts.

## 12. Available results

All preparation, Stage B, and Stage C artifacts are now available. The main held-out execution
results, expressed as the total saving for one execution of each of the ten workloads, are:

| Strategy | Improved / tied / worse | Execution saving | Selection regret vs measured orders |
|---|---:|---:|---:|
| Existing adaptive | 8 / 1 / 1 | $0.006129249 | $0.000345492 |
| Independent full profiling | 8 / 2 / 0 | $0.006035211 | $0.000439530 |
| Correlation-aware full profiling | 6 / 4 / 0 | $0.006339984 | $0.000134757 |
| Confidence-aware | 5 / 5 / 0 | $0.006200691 | $0.000274050 |
| Risk-aware | 5 / 5 / 0 | $0.006200691 | $0.000274050 |
| Joint adaptive-risk | 7 / 3 / 0 | $0.006322050 | $0.000152691 |
| Milestone 4.5 frozen joint policy | 6 / 4 / 0 | $0.005389104 | $0.001085637 |

Existing adaptive's one harmful held-out reorder was
`banking_unauthorized_withdrawal_refund`, costing $0.000221550 relative to fixed. The risk-aware
policy avoided that loss, but rejected three beneficial reorderings and incurred $0.000193998
of conservative-fallback regret on those cases. This supports a real downside-protection effect
in one workload, with a measurable opportunity cost; ten workloads are too few for a broad
probability claim.

Correlation-aware estimation had lower mean stage-cardinality MAE (3.913 versus 5.443), mean
stage-cardinality MAPE (0.171 versus 0.401), mean total-evaluation MAE (8.803 versus 10.663), and
mean cost-prediction MAE ($0.000179582 versus $0.000241863) than independence. Its mean ranking
concordance over the nine comparable workloads was lower (0.778 versus 0.941), yet its measured
selection regret was also lower. Thus cardinality and decision-regret results favor correlation,
while rank concordance does not.

Observed bootstrap coverage over 24 executed-order checks was 87.5% at 50%, 95.8% at both 80%
and 90%, and 100% at 95%. This is materially better than the prior nominal-95% result, but it is
a small, dependent set and does not establish general calibration.

For existing adaptive, standalone profiling cost is $0.011375952 and shared incremental cost is
$0.005658366. The aggregate cold-start net is -$0.005246703 at R=1 and +$0.019270293 at R=5;
the standalone break-even reuse count is two. Shared compatible statistics make R=1 positive by
$0.000470883. These reuse results are projections; only the scheduled ABBA repetitions were
actually executed.

Sixty-one of 77 run-pair match sets were exactly equal. Four workloads had at least one unequal
pair. Across 29,522 overlapping record-predicate evaluations, 91 threshold decisions differed
(99.69% agreement). This measures provider/execution stability, not semantic accuracy or recall.

## 13. Limitations

Two datasets and ten workloads still provide limited workload-level power. Banking77 queries are
short and intent-oriented, while reviews are longer and entity-clustered. Prompt-defined
predicates have no new human accuracy labels. Match-set consistency therefore tests execution
stability, not truth. Four workloads showed order/run-dependent match-set differences. The 24
coverage checks share workloads and records, and single-run exploratory orders have weaker
evidence than the ABBA comparisons. The failed first session also incurred an unrecoverable exact
cost for 822 attempts, though none of its outcomes enter the performance analysis. Frozen
bootstrap intervals were evaluated, not repaired.

## 14. Research implications

Adaptive execution savings were positive in aggregate on both independent datasets, and eight
of ten existing-adaptive selections improved on fixed order. The single harmful adaptive reorder
shows that generalization is useful but not uniformly safe. Correlation improved cardinality and
cost errors and reduced selection regret, but did not improve ranking concordance. Risk-aware
selection avoided the observed harmful adaptive decision while forgoing smaller benefits.
Profiling is economically justified only after sufficient reuse unless compatible shared
statistics already exist.

## 15. Readiness for Milestone 5

Milestone 4.6's independent live-validation evidence is complete. It supports the claim that the
frozen optimizer generalizes economically at adequate reuse horizons, but it also documents a
harmful adaptive decision, mixed correlation-ranking evidence, sparse-prefix uncertainty, and
semantic instability without reference labels. Those limitations should be reviewed before an
explicit Milestone 5 decision. This work does not begin Milestone 5.

Optional Stage D replication remains a separate, unauthorized experiment. No further Jev calls
were made after the successful Stage C restart.
