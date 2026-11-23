# Milestone 5.2.3 Adaptive Behavior Analysis

## Preserved failed session

The controller made one deterministic initial selection using the empty-evidence fingerprint. It
selected a required Banking77 representative reference inside the frozen universe. The response
was rejected before evidence acceptance, so the controller never reached a response-dependent
checkpoint.

Real adaptive branching was therefore not exercised. Discovery screens, informative screens,
follow-ups, skipped follow-ups, near-threshold cases, and disagreements were all zero observed—not
estimated population rates. The remaining 279 frozen actions were not requested because the
session failed closed, not because adaptive profiling avoided them.

## Repaired fresh session

The repaired session completed all 220 required representative actions before running 30 paid
discovery screens. Each screen became visible to the next controller checkpoint only after its raw
response, usage, validated receipt, and evidence-acceptance event had been durably recorded.
Replay of every evidence fingerprint and prerequisite set found no visibility or ordering error.

Twenty-nine screen probabilities lay outside the frozen `0.50 ± 0.15` informative interval and
their follow-ups were skipped. Banking77 record
`banking77_queries:b5824008afa61a119f8ade02` returned a card-related screen probability of 0.37,
which lies inside that interval, so the controller acquired the preauthorized three-predicate
follow-up. Its probabilities were 0.39 for card-related, 0.08 for unauthorized activity, and 0.41
for refund/reversal. All were below threshold, and the repeated card decision agreed with the
screen.

The observed branch counts were:

- 30 response-dependent screen decisions;
- 1 informative screen and 1 executed follow-up;
- 29 skipped follow-ups;
- 29 actions avoided relative to the full 280-action universe;
- 109 requests of unused capacity below the 360 hard ceiling;
- no early, budget-driven, or failure-driven termination.

The informative-screen and follow-up rates were both 1/30 (3.33%). The one follow-up did not expose
a screen-versus-batch decision disagreement, so diagnostic disagreement yield was 0/1. It still
provided the two additional predicate probabilities for that discovery record. This is evidence
that real responses changed the acquisition sequence, not evidence that the acquisition rule is
population-optimal.
