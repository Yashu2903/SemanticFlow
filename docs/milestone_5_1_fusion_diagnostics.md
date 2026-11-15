# Milestone 5.1 fusion diagnostics

## Scope

This analysis replays only the frozen Milestone 5 Stage B traces. It made no provider
requests and did not inspect a Milestone 5 Stage D held-out outcome. The machine-readable
source is `results/milestone_5_1/stage_a_preflight/preflight_v2.json`.

## Economic diagnosis

Stage B established that full conjunctive fusion cost $0.006741042 versus $0.005536104
for fixed separate execution across the nine workload replays, a 21.77% increase. The
twelve directly measured fused groups cost $0.008799336 in total. Replaying their source
predicates with early rejection cost $0.007396368. Every directly measured group was more
expensive than its early-rejecting reference.

The trace decomposition identifies three observed mechanisms:

1. A fused prompt repeats every source instruction and adds conjunction framing. The added
   provider-visible text ranged beyond the longest individual instruction, increasing input
   tokens.
2. A fused request evaluates the whole conjunction even when the first separate predicate
   would reject. This was most visible in four-predicate beauty groups: forty fused requests
   replaced forty early-rejecting reference requests, but the fused prompts were much longer,
   adding $0.000288960 in one group.
3. Fused outputs were generally longer than one binary output, while the provider's stored
   usage does not expose a separable fixed HTTP-overhead charge. Request reduction therefore
   cannot be translated directly into token-price savings.

Changed decisions also change downstream reach, so outcome effects cannot be cleanly
separated from prompt and early-rejection effects. The artifact reports observed token deltas
and labels this causal limitation; it does not invent a provider-overhead component.

## Semantic disagreements

Eight fused/reference mismatches are preserved with record text, exact questions,
probabilities, and group output:

- Three false negatives for beauty malfunction plus hard-to-use.
- Two false negatives on one Banking77 record, once in the two-predicate unauthorized/refund
  group and once in the three-predicate unauthorized/card/refund group.
- Three SMS false positives: two in prize/fee/contact and one in prize/contact.

The diagnostic taxonomy marks only plausible cues. Negation tokens, compound conditions, and
multi-situation text are flagged where observable, but `linguistic_cause_established` is false
for every mismatch. One case remains `unknown_cause`. This is deliberate: a trace alone cannot
prove why the model changed its answer.

## Reference reliability

The separate decisions are a reproducible LLM reference, not human truth. Stage B did not
repeat identical requests, so it cannot isolate ordinary response instability. The sole
multi-decision disagreement (59/60 agreement) is a context change, not an identical-request
replicate. No mismatch is therefore called an independently validated semantic error.

## Consequence

The original preregistered gate remains unchanged. No measured fused candidate passed it.
Milestone 5.1 treats full fusion as an experimental comparator, gives unknown quality a failing
state, and tests multi-decision batching without claiming that it inherits individual-call
quality.
