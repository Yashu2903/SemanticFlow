# Architecture

## Execution flow

1. `query.models` parses YAML into an ordered, validated conjunction.
2. `dataset.loader` asks DuckDB to read Parquet, removes null/blank text, maps source fields to
   `ReviewRecord`, deterministically assigns IDs, sorts by ID, and applies the record limit.
3. `SequentialExecutor` keeps an active record list. For each configured predicate, it creates at
   most `concurrency` tasks, calls the provider/cache, applies `probability >= threshold`, and sends
   only survivors to the next stage. Predicate order is immutable.
4. `InferenceProvider` normalizes model output as `Decision`. `MockProvider` is deterministic and
   offline. `JevProvider` owns TypeSafe HTTP schema, timeout, retry, response validation, and client
   cleanup. A shared atomic `RequestBudget` is reserved before every submission attempt.
5. `RequestCache` is a per-run, optional in-memory single-flight cache. Failed decisions are never
   cached.
6. Every successful evaluated pair becomes a `DecisionTrace`. Stage metrics distinguish requested
   evaluations, completed decisions, budget blocks, provider calls, HTTP attempts, retries, usage,
   latency, and cache behavior. Observed selectivity divides passes by completed decisions and is
   unknown (`null`) when a stage has no completed decisions.
7. `runtime.run_query` optionally evaluates human labels, builds report/manifest data, and asks
   `ResultStore` to atomically scope artifacts to a never-reused run directory.

## Failure semantics

Transient Jev transport, HTTP 429, and 5xx failures are retried within configured and budget
limits. Authentication, other 4xx, malformed answers, missing answers, and invalid probabilities
are permanent. Budget exhaustion is expected execution control and is reported separately from
provider errors. Either condition marks the run incomplete; prior diagnostics remain, while
`matches.jsonl` is empty so partial output cannot be mistaken for a completed query.

The canonical metrics and compatibility aliases are defined in [reporting schema](reporting_schema.md).

## Research boundary

The executor exposes stage metrics and traces but contains no planner. This separation is
intentional: a planner supplies a different ordered `QuerySpec` while the same executor,
provider, tracing, and reporting layers remain the measurement substrate.

`semanticflow.optimizer` is that planner layer:

- Milestone 2 added the split, unconditional profiling, and exhaustive independence-based
  ordering ([Milestone 2 architecture](milestone_2_architecture.md)).
- Milestone 3 adds pluggable cardinality estimators that use the joint profiling matrix,
  replay of stored decisions, profiling economics, bootstrap stability, and adaptive-profiling
  simulation ([Milestone 3 architecture](milestone_3_architecture.md)).
- Milestone 4 adds cost-distribution estimates, expected regret, CVaR, a `risk_aware`
  selection policy, estimator-disagreement analysis, and an offline retrospective benchmark
  ([Milestone 4 architecture](milestone_4_architecture.md)).
- The Milestone 4 follow-up coordinates adaptive profiling with the same risk policy using sparse
  uncertainty and value of additional profiling; its strict no-lookahead replay is described in the
  [joint adaptive-risk report](milestone_4_joint_adaptive_risk_report.md).
- VOI validation adds checkpoint-local calibration traces and an optional finite-scenario lower
  confidence bound without changing the planner/executor boundary
  ([validation report](milestone_4_voi_validation.md)).
- Milestone 4.5 adds profiling-only statistical calibration and constrained optimizer tuning,
  freezing an experimental joint candidate while retaining existing adaptive as the recommended
  default when the profiling-validation margin is insufficient
  ([Milestone 4.5 architecture](milestone_4_5_architecture.md),
  [calibration report](milestone_4_5_calibration_report.md)).
- Milestone 4.6 adds independent public-data preparation, preregistered workload and strategy
  freezes, shared full-matrix profiling, separately authorized live stages, ABBA scheduling,
  generalization analysis, and fail-closed paid-provider guards. It reuses the frozen optimizer
  and executor rather than adding an algorithm. Stage C hardening adds provider-free
  early-rejection forecasting, compatible-decision cost ledgers, sparse-prefix diagnostics,
  schedule-strength classification, and exact-scope launch validation before provider import
  ([protocol](milestone_4_6_generalization_protocol.md),
  [research report](milestone_4_6_research_report.md),
  [Stage C preflight report](milestone_4_6_stage_c_preflight_report.md)).

None of these layers changes executor semantics, and the offline analyses never call a provider.
