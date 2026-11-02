# Reporting schema

`report.json` uses schema version `1.1.0`. The schema separates work requested by the executor,
valid predicate decisions, HTTP transport activity, provider failures, budget enforcement, and
cache lookups.

## Canonical metrics

- `requested_evaluations`: record-predicate evaluations submitted to the execution layer. This
  includes evaluations stopped by the request-budget guard.
- `completed_decisions`: evaluations that returned a valid probability and produced a decision
  trace. Cache hits count because they return a valid prior decision.
- `budget_blocked_evaluations`: requested evaluations that could not finish because the shared
  retry-inclusive request budget was exhausted. They do not produce decision traces.
- `actual_provider_calls`: uncached evaluations that reserved at least one provider attempt.
- `provider_attempts`: provider submission attempts. For Jev these are HTTP attempts; mock-provider
  attempts remain visible here without being mislabeled as HTTP.
- `http_attempts`: Jev HTTP requests actually made, including retry attempts.
- `successful_api_calls`: HTTP requests that returned a valid model response.
- `failed_http_attempts`: transport failures and unsuccessful HTTP status responses. A malformed
  successful-status response is instead represented by `provider_errors`.
- `provider_errors`: evaluations that ended in a provider, response-validation, or execution
  failure. Budget enforcement is excluded.
- `retry_attempts`: HTTP retry requests actually made. A retry stopped by the budget guard is not
  an attempt.
- `cache_hits` and `cache_misses`: actual enabled-cache lookups. Both are zero when caching is
  disabled, and `caching_enabled` records that state explicitly.
- `stages_reached`: predicate stages that received one or more input records.
- `stages_with_completed_decisions`: stages that produced at least one valid decision.
- `total_completed_reviews`: records definitively rejected at a completed predicate plus final
  matches. A complete run must equal `total_input_reviews`.

Predicate entries include derived `total_tokens`, `summed_inference_latency_seconds`,
`average_inference_latency_seconds`, and `estimated_cost_usd` fields. Reports and manifests retain
the pricing identifier, source, verification date, billing basis, and rates used for estimates.

Observed predicate `selectivity` is `passing_record_count / completed_decisions`. It is JSON
`null` when there are no completed decisions, including a reached stage whose requests were all
budget-blocked and a later stage that received no records. Unevaluated records are never treated
as rejections. Every configured predicate has a predicate entry; a stage skipped after filtering or
an incomplete earlier stage is marked `reached: false` with zero requested/completed decisions.

`completion_status` is `incomplete` whenever required evaluations fail to finish. Budget-limited
runs use `incomplete_reason: "request_budget_exhausted"`; their details appear in
`budget_diagnostics`, not in the operational `errors` list. Incomplete runs have an empty
`matches.jsonl` so partial survivors cannot be mistaken for a complete result.

## Compatibility aliases

The original Milestone 1 names remain in version 1.1.0 for compatibility:

| Deprecated field | Canonical field |
| --- | --- |
| `logical_evaluations` | `requested_evaluations` |
| `predicate_decisions_evaluated` | `completed_decisions` |
| `predicate_stages_evaluated` | `stages_with_completed_decisions` |
| `request_attempts` | `provider_attempts` |
| `failed_api_calls` / `failed_attempts` | `failed_http_attempts` |
| `retries` | `retry_attempts` |

Predicate entries also retain `logical_evaluations`, `request_attempts`, `successful_requests`,
`failed_attempts`, `retries`, and `errors` aliases. New consumers should use the canonical names.

`decisions.jsonl` remains one row per completed decision. Schema 1.1.0 does not add synthetic rows
for budget-blocked or failed evaluations. The manifest records the report schema version, request
budget, cache configuration, sampling configuration, completion status, and incomplete reason in
addition to the existing reproducibility fields.
