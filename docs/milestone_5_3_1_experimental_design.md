# Milestone 5.3.1 — Fresh Jev Validation Design

## Objective

M5.3.1 prospectively tests whether the frozen M5.3 acquisition policies reproduce their historical evidence-discovery behavior on new Jev executions. It does not tune a policy or change production plan selection.

## Frozen scope

- Provider model: `jev-1.14.0`.
- Threshold: 0.50.
- Workloads: Banking77 card/unauthorized/refund; Amazon All Beauty failure/usability; SMS Spam prize/urgency/contact.
- Records: 300 total, 100 per workload.
- Strata per workload: 50 deterministic representative-random records and 50 deterministic lexical-enrichment records.
- Required evidence: the first single predicate (also the screen), every remaining single predicate, and one exact multi-decision request for the complete predicate group.
- Requests: 400 Banking77, 300 All Beauty, and 400 SMS Spam; 1,100 total.
- Retry policy: at most two retries per physical request; at most 3,300 HTTP attempts.
- Planning cost: approximately $0.0155 token-priced cost ($0.015792 summed action forecast); frozen default hard cap: $0.025.

The enrichment score is computed before provider execution from workload-specific lexical cues and text length. It is not called a calibrated uncertainty score. Representative and enriched results remain separate in reporting.

## Independence

Selection excludes every stable record ID and content hash recoverable from the immutable M5.1 and M5.2.3 provider-evidence trees. The current source pool has 589 independently eligible Banking77 records, 586 All Beauty records, and 588 SMS Spam records. Selection also rejects duplicate IDs and duplicate content hashes within the new sample. Independence refers to provider evaluation history, not to the source dataset having never been inspected for dataset construction.

## Frozen policies

The evaluated source definitions are the unmodified M5.3 implementations:

- uniform random;
- continuous uncertainty by absolute distance from 0.50;
- positive-oriented descending screen probability;
- weighted hybrid with 0.5 uncertainty, 0.5 positive likelihood, divided by observed follow-up cost;
- quota hybrid alternating uncertainty and positive rankings.

The configuration persists the policy-source SHA-256, declarative parameters, configuration fingerprint, source dataset hashes, workload configuration hash, selection seed, replay seeds, and action-universe fingerprint. No score or weight may be changed after live outcomes are observed.

## Replay and metrics

Request budgets are 5, 10, 20, 30, 50, and 100. Dollar budgets are $0.0001, $0.00025, $0.0005, $0.001, and $0.0015. Conditional replay treats persisted screens as already acquired and budgets only follow-ups. End-to-end replay charges screens and confirmations. Results are generated overall, by workload, and by sampling stratum.

Event definitions are unchanged from M5.3: named-predicate decision disagreement, complete-query membership disagreement, batching false negative relative to separate Jev, and complete-query positive under separate Jev. Separate Jev is a comparison reference, not human truth.

Policy comparisons use identical candidate pools, seeds, budgets, and settings. Seed repetition estimates policy-selection variability; it does not create additional independent provider observations. Sparse event rates receive numerator/denominator reporting and Wilson intervals. A human review sheet separates execution disagreement from verified semantic error.

## Safety and stopping

Dry-run is the default and neither imports the Jev adapter nor reads credentials. Live mode requires `--authorize-paid`, validates the frozen request/attempt/cost limits, durably records raw response and usage before semantic validation, and resumes only completed or safely recoverable actions. An ambiguous dispatch or validation/integrity error fails closed. A failed-closed session cannot execute further.

No paid calls were made while implementing or validating this milestone.
