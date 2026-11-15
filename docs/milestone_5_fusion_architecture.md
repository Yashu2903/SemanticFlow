# Milestone 5 fusion architecture

## Scope and positioning

Milestone 5 adds physical predicate grouping without changing logical AND semantics or the
validated `existing_adaptive` default. Fusion itself is established prior work: [Sema]
(https://arxiv.org/abs/2603.11622) describes adaptive reordering and fusion, while [Abacus]
(https://arxiv.org/abs/2505.14661) searches physical semantic-operator implementations under
cost, quality, and latency objectives. SemanticFlow's experimental question is narrower:
cost-and-quality-aware grouping under a limited profiling budget, with a safe fallback.

## Execution architecture

```text
QuerySpec (logical predicates)
          |
          v
ordered-set-partition enumerator -- 3 / 13 / 75 candidates for 2 / 3 / 4 predicates
          |
          v
FusionPhysicalPlan: [FusedOperator, ...]
          |
          v
FusedExecutor -- group request --> provider -- validated FusedDecision
      |                                       |
      +-- reject record immediately ----------+
      +-- trace attempts, tokens, latency, prompt/model hashes
```

`FusedOperator` retains predicate IDs and exact instructions, a deterministic provider-visible
prompt, template and result-schema versions, model and threshold, execution settings, and
predicate/prompt hashes. `FusionPhysicalPlan` validates that every logical predicate occurs once.
Plan IDs hash the canonical grouping, group modes and prompt hashes, planner version, and frozen
configuration fingerprint.

## Provider contract

The [TypeSafe System One API](https://api.typesafe.ai/redoc) accepts one or more named questions
about one `state`, returns name-matched answers, and reports the resolved model and token usage.
The adapter exposes:

- `conjunctive_binary`: one named `noul` question containing a deterministic explicit
  conjunction; the result is one group-level probability.
- `multi_decision`: several named `noul` questions in one HTTP request; every named answer must
  validate. This is a provider batching capability, not the primary conjunction implementation.

Both modes are supported by the documented API shape and mock-tested. Neither mode has been
live-tested by this milestone's offline stage. The capability metadata therefore keeps
`live_validated=false` until an authorized pilot succeeds. No hidden per-predicate requests are
made by the fused adapter.

## Prompt construction

`m5-conjunction-v1` preserves each instruction verbatim and labels it with its stable predicate
ID. It says that a positive answer requires every condition and asks whether all conditions are
satisfied. Prompt creation is deterministic and never calls an LLM. Template changes create new
hashes and require new evidence.

## Lifecycle, failures, and fallback

The executor evaluates groups in order, validates the provider capability and response, persists
group-level traces, and short-circuits on rejection. Invalid/missing answers, timeout exhaustion,
rate limits, context errors, attempt-budget exhaustion, and other provider errors make the run
incomplete; they are never converted to negative semantic decisions. Partial results cannot be
reported as complete. Fallback calls, if a later live protocol permits them, must be separately
traced and charged.

The safe planning default is `existing_adaptive`. A fused candidate is rejected when its cost is
unknown, required real measurements are absent, its complete-plan quality evidence is missing,
or its preregistered gate fails. Cost-only selection exists only as an ablation.

## Cost and selectivity

For groups G1...Gk, expected cost is the sum of each group's measured per-request cost multiplied
by the probability that all preceding groups pass. The estimator records expected requests,
stage cardinalities, input/output tokens, latency, and evidence source. A real fused measurement
is distinct from a conservative model and from the logical reference obtained by AND-ing separate
decisions. Unmeasured groups remain unknown and cannot be quality-certified.

## Quality

Measured complete-plan outputs are compared with the conjunction of independently evaluated
predicate outputs on identical records. Metrics include agreement, confusion counts, precision,
recall, F1, MCC, false-positive/negative rates, Jaccard, and changed decisions; undefined metrics
remain `null`. This is agreement with an LLM reference, not semantic truth. Optional human labels
use a separate reference kind. Quality does not compose by multiplying group accuracies.

## Adaptive fusion profiling

The deterministic `m5-deterministic-net-value-v1` controller ranks prospective group
measurements by optimistic reuse-horizon benefit minus profiling cost. It enforces unique-group,
records-per-group, logical-request, retry-inclusive HTTP-attempt, and estimated-expenditure
budgets. Compatible measurements have identities incorporating dataset, record, predicates,
prompt hash, model, threshold, and execution settings. It never sees future pilot or evaluation
records. The expenditure limit is an estimate; the attempt counter is the hard concurrent guard.

## Persistence and limitations

Writes use temporary files followed by atomic replacement, new run directories refuse overwrite,
and authorization manifests bind datasets, configs, schedule, budget, runner, provider, and fusion
source. Fused semantics can differ from separate semantics; Jev response stability and context
limits are unknown before the pilot; the frozen quality thresholds are exploratory; and the
fresh-domain split is independent of M4.6 records but not entirely independent of its domains.

