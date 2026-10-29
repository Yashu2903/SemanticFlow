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
6. Every successful evaluated pair becomes a `DecisionTrace`. Stage metrics distinguish logical
   evaluations, provider calls, request attempts, retries, usage, latency, and cache behavior.
7. `runtime.run_query` optionally evaluates human labels, builds report/manifest data, and asks
   `ResultStore` to atomically scope artifacts to a never-reused run directory.

## Failure semantics

Transient Jev transport, HTTP 429, and 5xx failures are retried within configured and budget
limits. Authentication, other 4xx, malformed answers, missing answers, and invalid probabilities
are permanent. Any stage error marks the run incomplete; prior diagnostics remain, while
`matches.jsonl` is empty so partial output cannot be mistaken for a completed query.

## Research boundary

The executor exposes stage metrics and traces but contains no planner. This separation is
intentional: a future planner can supply a different ordered `QuerySpec` while the same executor,
provider, tracing, and reporting layers remain the measurement substrate.

