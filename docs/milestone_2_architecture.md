# Milestone 2 architecture

## Boundaries

Milestone 2 adds a planner in `semanticflow.optimizer`; it does not add a second executor. The
existing sequential executor remains responsible for fixed-order AND filtering, thresholding,
early rejection, concurrency, retries, budgets, caching, and decision traces. A fixed run passes
the configured order to that executor. An optimized run passes the planner's selected order.

## Deterministic split

Records are loaded into SemanticFlow's stable review schema. Split membership is determined by
sorting on `sha256("<seed>:<review_id>")`, with review ID as a tie breaker, rather than Parquet row
position. The first configured IDs form the profiling set and the next IDs form the evaluation
set. The manifest stores every ID plus SHA-256 fingerprints of the two sorted ID sets. Duplicate
IDs, overlap, missing IDs, and insufficient records are rejected.

The default configuration uses 200 profiling and 800 evaluation reviews from the real
`reviews_1000.parquet` dataset. Evaluation observations never enter plan selection.

## Unconditional profiling

The profiler evaluates each predicate separately on every profiling record. It deliberately does
not perform sequential early rejection. Thus each estimate is empirical unconditional
selectivity, not selectivity conditional on earlier predicates. Every completed decision retains
the probability, thresholded Boolean result, tokens, model, latency, attempts, prompt hash, and
timestamp. Retry-inclusive request budgets and bounded concurrency use the existing provider
interfaces.

For predicate `i`, with `n_i` completed decisions and `p_i` passing decisions:

```text
s_i = p_i / n_i
```

Zero decisions means unknown selectivity, never zero selectivity. Planning refuses incomplete or
unknown statistics. A 95% Wilson binomial interval is recorded as a diagnostic, but confidence is
not used to change the plan.

Average input tokens, output tokens, total tokens, request latency, and—only with configured
verified prices—estimated dollars are computed from measured provider responses. Unknown token or
price information remains `null`. Mock runs may configure synthetic token counts solely to test
the token-optimization pipeline; their reports identify this limitation.

## Cost model

For input cardinality `N`, predicate evaluation cost `c_i`, unconditional selectivity `s_i`, and a
candidate order `pi`, the independent sequential-AND model is:

```text
ExpectedCost(pi) = N * sum_k(
    c_pi[k] * product_(j < k)(s_pi[j])
)
```

The same reach-probability calculation predicts evaluation count, input/output/total tokens,
dollars, and summed inference latency. Summed request latency is reported separately from wall
time because concurrency means they are not interchangeable.

The configured objective is explicit and stable for a run. The default is total tokens. Monetary
optimization is valid only when every predicate has measured usage and verified configured rates.

## Candidate generation and selection

The planner supports 2–6 unique AND predicates and exhaustively enumerates permutations—at most
720 candidates. Enumeration is local and makes no provider calls. Candidates are ranked by the
configured objective. If candidates are equal within the configured numeric tolerance, the
original order is preferred; otherwise lexicographic predicate IDs make selection deterministic.
Every candidate and its predicted dimensions is saved.

Compatibility checks bind statistics to query ID, dataset and profiling fingerprints, threshold,
model, predicate IDs, exact prompt hashes, and statistics version. Incompatible or incomplete
profiles cannot silently produce a plan.

## Reporting

The experiment root stores split membership, raw profiling decisions, statistics, all candidates,
the selected plan, a reproducibility manifest, comparison metrics, and a summary. Each held-out
run retains the normal Milestone 1 four-artifact schema under `fixed/` or `optimized/`. Reports and
manifests record both original and selected orders, strategy, objective, estimate, split
fingerprints, versions, Git commit, and dirty status.

Profiling overhead is separate from execution-only cost. Comparison reports gross fixed-versus-
optimized savings, net savings after profiling, and the reuse count needed to break even when
savings are positive. Output equivalence compares exact final review-ID sets.

## Deliberate limitation: independence

Semantic predicates are often correlated. In general, `P(B passes | A passes)` differs from
`P(B passes)`. This milestone intentionally substitutes unconditional profiling estimates into an
independence model. Consequently predicted cardinalities, candidate ranks, and savings may be
wrong even when every individual estimate is accurate. Conditional-selectivity and adaptive or
correlation-aware planning are later research questions, not hidden behavior in this baseline.
Milestone 3 adds them as opt-in components while keeping this model as the default; see
[Milestone 3 architecture](milestone_3_architecture.md).

SemanticFlow's design is informed by research systems including Sema, Larch, and Abacus, but this
milestone does not claim to reproduce their complete algorithms. It is a deliberately small,
auditable exhaustive-ordering baseline for studying cost estimates and predicate correlation.
