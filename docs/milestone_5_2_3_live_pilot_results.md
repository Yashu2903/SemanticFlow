# Milestone 5.2.3 Live Pilot Results

## Preserved failed session

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

## Repaired fresh session

The separately authorized repaired session
`semanticflow-m5.2.3-live-pilot-a-repaired-session-01` completed with status
`INSTRUMENTATION_COMPLETE`. It used protocol fingerprint
`a9d5218c478d217a007ed590e95c072d58e5bd319e02673781260c09e87f96f8`, action-universe
fingerprint `33bcbf43db2cd6d926818616cf57171f06c8627170a88c9d4d39347775ccb2a7`, policy
fingerprint `2d190356e5883f91342d31aeed02b816a7bf84a78b0a339364ab349f92dd1a7f`, and
authorization-scope fingerprint
`bc9145b834b326269f1fc4cc797c54d8a33606762d32910b92e9385b9109baeb`.

The run completed 251 unique physical requests and 251 HTTP attempts with no retry, ambiguous
receipt, provider failure, malformed response, or duplicate measurement. Jev returned 353 named
decisions using 84,890 input tokens and 10,239 output tokens. The frozen input-token pricing model
produces a measured cost proxy of **$0.003565380**. Provider invoice adjustments remain unknown.

All 220 representative actions completed: 160 separate predicate references and 60 batched
candidates over 60 records. Thirty discovery screens completed. One Banking77 screen was within
the frozen 0.15 acquisition margin and triggered its authorized follow-up; the other 29 follow-ups
were skipped. The controller therefore executed a real response-dependent branch and avoided 29
requests relative to the 280-action universe.

Representative query decisions agreed on all 60 records. Banking77 and All Beauty each had one
separate-query positive and one matching batch positive; SMS Spam had no separate-query positive,
so its precision, recall, false-negative rate, and MCC are undefined where their denominators are
zero. There were no query-level or predicate-threshold disagreements. These observations describe
this small Jev sample only and do not constitute semantic-quality certification.

Independent reconciliation found no scope, prerequisite, evidence-visibility, ledger, or artifact
integrity error. All 251 raw provider payloads and usage records were persisted before scientific
receipt acceptance. The complete artifact set is under
`results/milestone_5_2_3/live_pilot_a_repaired_session_01/`.
