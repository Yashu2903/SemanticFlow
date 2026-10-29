# Research notes

SemanticFlow studies how a database-like engine can reduce cost and latency for expensive,
uncertain language-model predicates while protecting answer quality. Relevant systems include
LOTUS (semantic operators in dataframe workflows), PALIMPZEST (cost/quality-aware AI dataflow
optimization), Sema (semantic data systems), Larch (LLM-aware query processing), Abacus (cost-aware
LLM pipelines), PLOP (language-model program optimization), and JEVDB/TypeSafe work on typed
semantic decisions. These names define a literature-review agenda; this document does not claim
feature parity or reproduce their results.

Milestone 1 is not a novel optimizer. It fixes configured predicate order and provides a trustworthy
measurement baseline: actual stage selectivity, logical versus physical requests, retries, token
usage, latency, cache behavior, model metadata, and record-level decisions. Later milestones can
use the same traces to study correlations and conditional selectivity, then compare cost-based
ordering, fusion, uncertainty-aware quality constraints, profiling budgets, and adaptive execution
against this unchanged reference path.

