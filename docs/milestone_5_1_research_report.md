# Milestone 5.1 research report

## Executive summary

Milestone 5.1 offline remediation is complete. It does not overturn the Milestone 5 negative
fusion result. Instead, it adds genuine multi-decision batching, mixed hybrid plans, explicit
quality-evidence states, cost-aware profiling, and a fresh three-domain experiment. No new Jev
call was made and no Milestone 5 Stage D held-out result was inspected.

Status: **READY_FOR_M5_1_BATCHING_PILOT**.

## Verified historical evidence

The frozen Stage B session contains 1,230 decisions and 1,230 attempts with no retries or
failures, 450,942 input tokens, 36,000 output tokens, and $0.018939564 token-priced cost.
Artifact hashes for the session, original protocol/configuration, and corrected Stage C plan and
proposal are recorded in the preflight artifact. The corrected zero-reach rule is retained and
covered by regression tests.

Full conjunction fusion was costlier on every pilot workload. Across the twelve directly
measured groups, it cost $0.008799336 versus $0.007396368 for early-rejecting separate replay.
The stored eight mismatches are categorized as observations, not proven semantic errors. The
only multi-decision context mismatch confirms 59/60 agreement but is too small to certify
batching quality.

## Engineering delivered

The hybrid planner enumerates exact provider-visible plans across single, conjunction, and
multi-decision modes. The cost model refuses to complete an unmeasured later group at zero reach.
Quality-constrained selection treats unknown evidence as ineligible and falls back to the frozen
`existing_adaptive` baseline. The profiling controller shares group evidence across plans and
charges physical requests once.

Jev responses must map exactly to requested names. Hybrid execution preserves named decisions,
short-circuits only between groups, and records explicit fallback expenditure. Three alternative
conjunction templates are versioned but remain unmeasured.

## Offline verification

Mocks verified 2/3/4-question batches, selective-first rejection, multiple physical modes,
explicit fallback, token accounting, quality classifications, profiling budgets, strict response
mapping, and 4/20/138 candidate enumeration. Fresh datasets validate against their manifests and
contain disjoint four-way splits. The authorization path fails before provider import for absent
or wrong authorization, altered manifests, and conflicting output directories.

## Pending live evidence

Batch cost scaling beyond two questions, representative individual-decision agreement, latency,
and hybrid net economics are unknown. Human semantic accuracy is also unknown. Enriched records
will improve diagnostic positive coverage but cannot estimate population rates. The fixed pilot
is designed to resolve these uncertainties without exhaustively paying for 138 plans.

## Limitations

- Historical token anchors include only thirty two-question batch requests.
- The pilot dollar budget is an empirical forecast, not a guaranteed spending cap.
- Dataset domains overlap with prior work even though records are fresh.
- Separate Jev output remains an imperfect reference until human labels exist.
- Full plan quality cannot be inferred by multiplying group agreements.
- The current paid runner fails closed rather than resuming an uncertain partial session.

## Recommendation

Authorize only the bounded Stage B batching pilot if the stated $0.019452045 conservative
no-retry and $0.058356134 retry-inclusive proxies are acceptable. Do not authorize Stage D,
replication, prompt-template exploration, or automatic plan freeze under this approval.
