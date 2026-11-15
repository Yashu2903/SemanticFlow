# Milestone 5.1 frozen selection-stage protocol

## Research hypotheses

H1: multi-decision batching can reduce token-priced inference cost versus all-question separate
execution. H2: the same batch can lose to early-rejecting separate execution. H3: a selective-first
hybrid can outperform the unchanged `existing_adaptive` baseline when first-stage inference cost and
conditional rejection jointly justify the remainder batch. H4: these economic gains are admissible
only when named and complete-query decisions meet the preregistered consistency gate. H5: a
checkpointed profiling policy can select admissible plans with less evidence than a full matrix.

Failure to qualify any batch is a valid negative result. The fallback remains `existing_adaptive`.

## Frozen partitions and workloads

Development and prior pilot data may be used for implementation, diagnostics, group selection, and
estimator development. The 100-record selection partition per dataset may be used for profiling,
selection, cost comparison, and quality gating. The 400-record evaluation partition per dataset is
reserved for a separately authorized Stage D and must not be inspected before plans are frozen.

The nine workloads and exact predicate instructions are frozen in
`configs/milestone_5_1/workloads.yaml`. Model `jev-1.14.0`, threshold 0.5, cache disabled, concurrency
four, and two retries are frozen. Predicate and group prompt hashes are persisted per request.

## Candidate strategies and plans

The measurable catalog contains every separate permutation, the full multi-decision batch, and one
selected single-plus-remainder hybrid for each three- or four-predicate workload. Strategies are:

* Fixed separate: original-order singleton groups.
* Existing adaptive: unchanged Milestone 4 controller over singleton measurements.
* Full batch: one multi-decision group.
* Fixed hybrid: the Stage-B-only selected singleton followed by its remainder batch.
* Cost-only hybrid: cheapest complete measured catalog plan.
* Quality-constrained hybrid: cheapest complete measured plan passing the frozen gate.
* Profiling-aware hybrid: net-benefit selection using only checkpoint-visible evidence.
* Conjunctive fusion: negative control only; not scheduled and therefore unsupported in selection.

Plans outside the catalog are unsupported. The selection analyzer must never infer an unmeasured
later group from zero profiling reach.

## Measurement schedule and pairing

The 3,300 logical roles are 1,800 singles (six per dataset-record), 900 full batches (nine
workload-record pairs), and 600 selected remainder roles (six workload-record pairs). Exact compatible
roles share acquisition. Three 100-record remainder arms reuse full-batch calls, yielding 3,000 unique
requests. Comparisons require identical dataset, record ID, predicates, instructions, model,
threshold, response schema, and inference settings.

Every request has a deterministic identity, operator identity, role list, selection partition,
question names, hashes, expected schema, and cost-calibration metadata. No inconvenient record may be
replaced. No enriched sample is part of the schedule.

## Complete-plan replay

Sequential and hybrid plans replay exact per-record measurements. A record enters stage *i* only if
all preceding group decisions pass. Execution cost is the sum of actual measured request costs for
entered stages. All-question comparisons sum every corresponding singleton. Missing groups make the
plan unsupported. Complete-query candidates are the AND of exposed named decisions; conjunctive-only
responses cannot be decomposed into predicate outcomes.

## Quality gate

The gate is unchanged: minimum 40 records, minimum three positive separate references, agreement at
least 0.98, recall at least 0.95, FNR at most 0.05, changed-decision rate at most 0.02, and actual
complete-plan evidence. Outcomes are classified as passed, observed failure, insufficient positives,
insufficient observations, undefined metric, missing group, missing complete plan, or provider
failure. Zero positives cannot establish recall. Provider consistency is not human semantic accuracy.

## Cost model

The retained estimator uses the existing physical-group mean costs and reach model. Joint prefix
reach is the desired versioned correction, but it is not adopted without non-tautological historical
validation. Compatible joint decisions should estimate prefix reach when available; otherwise the
independence fallback is labelled modeled and uncertain. Unavailable cost is never zero.

Expected cost is:

`N * sum_i P(G1 and ... and G(i-1)) * E[cost(Gi) | previous groups pass]`.

Tie rules are deterministic: prefer sufficient quality, then lower predicted cost; costs within a
relative tolerance of 1e-9 retain the baseline; then prefer fewer groups, fewer multi-predicate
groups, original predicate order, and finally lexicographically smaller plan fingerprint. Unsupported
quality or missing cost cannot win a tie.

## Adaptive profiling

The existing adaptive controller uses its frozen SHA-256 row order and checkpoints 25, 50, and 100.
At checkpoint *k*, only measurements revealed through *k* are available. Each state persists its
available IDs, candidates, estimates, uncertainty, action, value of information, expenditure, stop
reason, and selected plan. Full-matrix replay reveals records incrementally. Costs are reported for
actual acquisition, standalone workloads, shared statistics, and warm starts without double charging.

## Selection and freeze procedure

After separately authorized measurement:

1. Validate session completeness, identities, model, prompts, schemas, attempts, and tokens.
2. Validate every required group for all 100 IDs.
3. Reconstruct every supported catalog plan and the unchanged adaptive baseline.
4. Calculate observed and estimated execution costs with evidence labels.
5. Compute individual and complete-query consistency and classify the quality gate.
6. Reject unsupported or quality-ineligible candidates for constrained strategies.
7. Account for incremental profiling and optimization expenditure.
8. Apply frozen objective and tie rules; otherwise fall back to `existing_adaptive`.
9. Persist selected complete plan objects, hashes, quality evidence, and profiling trajectory.
10. Freeze a Stage D authorization proposal without accessing evaluation outcomes.

Thresholds, candidate groups, and tie rules may not change after selection outcomes are visible.
Selection performance is selection evidence, not an unbiased generalization estimate.

## Reuse economics

For reuse horizon *R*:

`NetBenefit(R) = R * (baseline cost - candidate cost) - incremental profiling - optimization`.

Report horizons 1, 5, 10, 25, 50, 100, 250, and 1,000. A nonpositive per-run saving has no finite
break-even. Actual research expenditure charges all 3,000 unique calls once. Scenario accounting may
charge fewer calls but must never relabel those savings as actual experiment expenditure.

## Statistical analysis

Use paired record bootstrap intervals for cost differences and Wilson or exact small-sample intervals
for proportions. Report numerators and denominators. Analyze datasets and workloads separately.
Question outcomes from the same physical request are dependent and cannot be treated as independent
requests. Unpaired evidence is descriptive, not causal. Stage D is the only independent test of
selection generalization.

## Authorization and failure safety

No provider may initialize until the exact phrase and countersigned frozen manifest validate.
Validation binds schedule, source/configuration, datasets, prompts/operators, model, logical roles,
unique requests, retries, and output destination. The output directory must not exist. Resume is off;
conflicting partial output is a hard failure. Atomic JSON persistence and exact attempt/token
accounting are mandatory. The four-request contract checkpoint is included in the total and verifies
one- through four-question mapping, model identity, usage, and persistence before continuation.

The current manifest has `paid_calls_authorized: false`; the runner therefore stops at the
authorization checkpoint before provider import. No selection or Stage D call is part of this
protocol-generation task.

## Stage D template

After selection, the primary independent comparison is the unchanged adaptive separate strategy
versus the frozen quality-constrained or profiling-aware hybrid. Identical selected physical plans may
share compatible provider measurements. Stage D reports cost, requests, named evaluations, tokens,
latency, reach, individual and final-query consistency, gate failures, regret, reuse economics,
dataset heterogeneity, and confidence intervals. It requires a separate forecast, attempt ceiling,
frozen-plan manifest, and explicit authorization.
