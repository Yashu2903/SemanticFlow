# Milestone 5.2.3 Live Pilot Results

The authorized Pilot A session failed closed after its first physical request. The selected action
was an authorized Banking77 representative single-predicate reference. One HTTP attempt was
recorded, no retry occurred, and no measurement was accepted.

The failure was `ResponseValidationError: single response has invalid probability shape`. The
session ended as `FAILED_CLOSED` with no pending or ambiguous action. The ledger contains one
attempt, zero named evaluations, and zero attributable input or output tokens. Because the parsed
provider usage was discarded by post-response validation, provider billing exposure for the one
attempt is unknown; the recorded zero-dollar proxy is not evidence that the provider charged zero.

No discovery screen, diagnostic follow-up, semantic disagreement, or real adaptive branch was
observed. This run does not establish semantic quality, batching agreement, optimizer superiority,
or correct real-response adaptive profiling.

## Post-incident implementation status

The response-shape defect has since been repaired and covered by targeted regression tests. Raw
provider responses and usage are now persisted before Pilot A semantic validation, and
`FAILED_CLOSED` sessions are terminal at both the runner and reconciliation gates. These repairs
do not alter, resume, or retroactively complete this failed session. A later live attempt still
requires a fresh, exact-scope authorization and must preserve this incident evidence.
