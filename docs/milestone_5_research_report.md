# Milestone 5 Stage A research report

## Executive summary

Stage A implements genuine one-request group execution, exhaustive joint grouping/order planning,
fusion-specific costing and evidence, quality gating, budgeted profiling, fresh benchmark splits,
authorization guards, and an offline harness. No Jev request was made. Status:
`READY_FOR_FUSION_PILOT_AUTHORIZATION`, subject to a separately authorized small pilot.

## Historical basis

M4.6 completed 42 held-out runs and 36,796 Jev evaluations at $0.508355400 in the successful
session. Existing adaptive remained the economic default; correlation-aware estimates improved
several metrics; fusion was not measured. The earlier failed-session cost limitation is preserved.
M4.6 outputs and frozen configs were not rewritten.

## Implemented functionality

- Deterministic `FusedOperator` and provider capability contract.
- One-request Jev conjunctive and multi-decision adapters, plus strict response validation.
- Early-rejecting `FusedExecutor` with attempts, retries, tokens, latency, errors, cache state,
  prompt hashes, and model tracing.
- `FusionPhysicalPlan` and exhaustive ordered-set-partition enumeration: 3, 13, and 75 plans.
- Measured/modeled/unknown group evidence, conditional reach cost estimation, and reuse economics.
- Complete binary-quality metrics with undefined-value handling and a fixed fail-closed gate.
- Cost-only, quality-constrained, and profiling-aware selectors with unchanged adaptive fallback.
- Deterministic prospective fusion profiling under request, attempt, record, group, and expenditure
  budgets.
- Four-way independent benchmark preparation and a stage-specific experiment runner.
- Measured pilot replay and plan-freeze path that cannot synthesize fused outcomes.

## Provider capability finding

The [official API contract](https://api.typesafe.ai/redoc) supports one or more named questions in
one request, name-matched answers, resolved model, and usage. It therefore supports a genuine
single composite conjunction question and documented multi-question batching. This is contract
evidence, not live M5 validation. The pilot specifically tests both paths.

## Benchmark readiness

Amazon All Beauty and Banking77 use fresh records; SMS adds a new domain. Each has 590 records
split 50/40/100/400 into development/pilot/selection/evaluation. Text overlap with historical
SemanticFlow evaluation records was excluded. The nine fixed workloads span all supported widths,
rare/selective events, hypothesized strong and weak correlations, and varied prompt complexity.

## Offline verification

The mock preflight parses all workloads, validates all datasets, obtains 3/13/75 candidates,
executes genuine group-level mock control flow, rejects non-real quality evidence, and exercises
the profiling controller. Mock decisions are not reported as semantic evidence. The complete
suite passed 293 tests in 56.79 seconds; Ruff and `git diff --check` passed. Tests cover one-request
Jev payloads through a mock HTTP transport, group widths through four, early rejection, attempt
budgets, missing evidence, quality fallback, enumeration, economics, and authorization binding.

## Pilot budget

| Measure | Proposal |
|---|---:|
| Unique measurement arms | 33 |
| Logical requests | 1,230 |
| Maximum HTTP attempts | 3,690 |
| Expected input tokens | 621,774 |
| Expected output tokens | 31,500 |
| Expected token-priced cost | $0.026114508 |
| Conservative no-retry proxy | $0.031997238 |
| Retry-inclusive proxy | $0.095991714 |

Expected input/cost by dataset is Beauty 212,387 tokens / $0.008920254, Banking77 217,302 /
$0.009126684, and SMS 192,085 / $0.008067570. Output is priced at zero under the frozen price.
Only the request/attempt limits are hard. Token and dollar values are proxies because no fused Jev
usage exists yet.

## Stage B addendum (2026-11-15)

The authorized pilot completed all 1,230 requests with zero retries or failures, 450,942 input
tokens, 36,000 output tokens, and $0.018939564 token-priced cost. All responses resolved to
`jev-1.14.0`. Across conjunctive arms, separate-reference agreement was 472/480 (98.33%), but
sparse positives and material errors in several arms caused every fused workload candidate to fail
the preregistered quality gate. Full fusion cost 21.77% more than fixed separate pilot replay in
aggregate and was costlier in all nine workloads. See
`docs/milestone_5_stage_b_pilot_report.md` for the full reconciliation and the documented
post-pilot analyzer correction. No held-out Jev calls have been made.

## Limitations and negative-result policy

The quality reference is an LLM conjunction, no human truth labels exist, the quality thresholds
are exploratory, multi-decision batching is not the same physical semantics as one conjunction,
SMS token ratios transfer from other domains, and full plan quality cannot be inferred from group
accuracy. Fallback and error requests may erase fusion savings and will be charged. A pilot that
shows no plausible economic advantage, unstable behavior, or unacceptable disagreement is a valid
negative result and will block fused held-out plans.

## Readiness

The architecture does not replace existing adaptive execution and does not begin Milestone 6.
Stage B is complete. Corrected offline plans and an exploratory Stage D budget are prepared, but
Stage D remains locked behind a new exact authorization. Pilot evidence does not currently support
adopting fusion as a quality-constrained strategy.
