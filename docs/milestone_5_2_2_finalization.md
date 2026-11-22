# Milestone 5.2.2 Finalization

## Initial repository state

- Branch: `milestone-4.5-calibration`
- HEAD: `71b2d8a8bc32353dc61cb208d4907474abe62287`
- Upstream: `origin/milestone-4.5-calibration`
- Staged changes: none
- Tracked modifications: four M4.6/M5.1 clean-checkout integrity and artifact-skip changes
- Untracked work: M5.1 final audit, M5.2, M5.2.1, and M5.2.2 source, tests,
  runners, and documents
- Ignored local evidence: M5.1 through M5.2.2 result directories, prepared raw data,
  virtual environments, caches, and `.env`
- Repository instructions: no `AGENTS.md` was present
- Secret review: `.env` remained ignored; no high-confidence private key, bearer token, or API
  credential pattern was found in the commit candidates. Dataset URLs containing
  `task-specific-datasets` were reviewed as false positives.

No file was reset, discarded, stashed, or overwritten. Historical paid result directories were
read and hashed but not regenerated.

## Files and boundaries reviewed

The review covered all `m522_*` optimizer modules, the Pilot A runner, all four M5.2.2 test
modules, the four generated M5.2.2 documents, and the direct M5.1/M5.2 dependency chain. The
following boundaries were inspected explicitly:

- `RESEARCH_INSTRUMENTATION` versus `PRODUCTION_QUALIFICATION`
- frozen action identity and universe validation
- response-dependent selection and evidence visibility
- authorization validation before live-provider import/initialization
- unique-request and retry-inclusive attempt reservation
- named-question, model, usage, probability, and retry-metadata validation
- representative/discovery evidence isolation
- append-only session persistence and hash-chain validation
- restart, interruption, receipt recovery, ambiguity, and duplicate handling
- cost and attempt accounting
- runner preflight, simulate, validate, status, reconcile, analyze, and execute paths

The production qualification code path and its statistical thresholds were not changed. Pilot A
continues to set formal certification and production deployment permissions to false.

## Defects discovered and fixed

1. **Post-response retry under-accounting.** If a provider returned a response after multiple
   attempts and the live adapter then rejected its model, usage, shape, or named decisions, the
   controller could charge one attempt instead of the response's reported attempt count. Adapter
   validation errors now retain attempt/retry metadata, and a regression proves all three attempts
   are charged before failing closed.
2. **Mock/live restart-class mismatch.** Persistent state did not expose the creation event's
   `mock_only` and objective fields to restart validation. A copied mock log could therefore match
   the protocol fingerprints. Objective and mock/live class are now typed persistent state and are
   bound into controller and live-runner restart checks.
3. **Mixed reconciliation status reopening.** A restart containing both a recoverable receipt and
   an unresolved dispatch could append `AUTHORIZED` after marking the unresolved action ambiguous.
   Reconciliation now reauthorizes only when no new ambiguity exists. A mixed-case regression
   proves subsequent execution remains blocked.
4. **Budget-pressure runner mismatch.** The CLI's `budget_pressure` mock scenario used the normal
   360-request protocol and therefore did not exercise exhaustion. The mock-only fixture now uses
   the documented 250-request/750-attempt reduced budget. The frozen live protocol remains
   unchanged at 360 requests and 1,080 attempts.
5. **Incomplete clean-checkout artifact guards.** Two historical test modules declared only a
   subset of the ignored evidence they load. A source-only checkout therefore errored instead of
   reporting optional historical-artifact skips. Their markers now enumerate the Stage D schedule
   and authorization proposal and the selection preflight/live inputs they consume. All 26 tests
   still run and pass when the local corpus is installed.

No statistical methodology, quality threshold, research conclusion, live request ceiling, or live
attempt ceiling changed.

## Test results

Baseline validation before repairs:

- Dedicated M5.2.2 suite: **23 passed, 0 failed, 0 skipped, 0 errors** in 76.50 seconds.
- Full repository suite: **449 passed, 0 failed, 0 skipped, 0 errors** in 191.73 seconds.

Final validation after repairs:

- Dedicated M5.2.2 suite: **33 passed, 0 failed, 0 skipped, 0 errors** in 60.85 seconds
  before commit; the committed isolated checkout repeated this result in 66.41 seconds.
- Full repository suite: **459 passed, 0 failed, 0 skipped, 0 errors** in 179.58 seconds
  after all source and test-guard repairs.
- Isolated committed checkout: **375 passed, 0 failed, 84 skipped, 0 errors** in 114.00
  seconds. Every skip was an explicit `research_artifacts` skip for an ignored historical corpus;
  the dedicated M5.2.2 suite had no skips.

The full suite includes M5.1, M5.2, M5.2.1, M5.2.2, adaptive optimization, physical execution,
provider integration, statistical qualification, economics, provenance, authorization, and replay
tests. No expected result was weakened to obtain a pass.

## Static analysis and hygiene

- `ruff check .`: passed.
- M5.2.2 `ruff format --check`: 17 files formatted, passed.
- Repository-wide formatting is not a CI requirement; CI runs `pytest -q` and `ruff check .`.
  A diagnostic repository-wide format check found 113 legacy files that Ruff would reformat. They
  were intentionally left untouched to avoid unrelated churn.
- Mypy is neither configured nor installed by the project's development dependencies, so no mypy
  claim is made.
- `git diff --check`: passed before staging; staged whitespace is checked again before each commit.
- No unused imports or lint violations remain.
- No hard-coded machine paths or credential values were introduced.

## Pilot A objective, workload, and fingerprint verification

The frozen protocol objective is `RESEARCH_INSTRUMENTATION`. It permits bounded diagnostics and
explicitly forbids formal certification and production deployment. The threshold is `0.5`, model
is `jev-1.14.0`, cache is disabled, and live concurrency is one.

| Workload | Dataset fingerprint | Development partition fingerprint | Predicate hashes |
|---|---|---|---|
| Banking77 Card / Unauthorized / Refund | `e138176cf7da4e7f54ec25b6a4ec50bf8e74c1cc55360c700083d1a39475cf9e` | `2bb824d8c9dc72e2d6a0d4ef5605f6adff590ef722872b1ca6f4998163abdc06` | card `a08f581b...4757b`, unauthorized `2dbd9f21...6ad4b`, refund `d7497a77...617bb` |
| Amazon All Beauty Failure / Usability | `988170c284a8f32f38e171f8bcf9e4aeb870a37149c09ae5a7c7db4ba43a9a51` | `ccfde2350f93108a14793680120825bd8c3dd9c63028ab31b5b74afad27b9bbc` | failure `68b6d9e9...e1a0f`, usability `40ab833d...9523f` |
| SMS Spam Prize / Urgency / Contact | `0aa8516d1d05e68646c7743dedccd97e7ab77654150bc5ec6af5bc7ca7a25463` | `7d3ff4a9805ca1342134ec4511691ffeb0ed5bf3360a45af0d361021c48630ee` | prize `5bee4790...8535`, urgency `1312c611...cbd5`, contact `d4a79afe...e79a` |

The exact predicate instructions, selected record text hashes, threshold, model, and operator shape
are included in the authorized action universe, whose fingerprint is
`56d868716503ec95f2689ef0e421fdba8cd4865be9b97042a7ab5999243a4174`.
The policy fingerprint is
`2d190356e5883f91342d31aeed02b816a7bf84a78b0a339364ab349f92dd1a7f`.

## Request budget and cost reconciliation

- Frozen physical actions: 280
- Named-question evaluations if all actions execute: 430
- Maximum unique physical requests: 360
- Maximum HTTP attempts: 1,080
- Maximum retries per request: two
- Attempt capacity reserved for all frozen actions: 840
- Unallocated contingency: 80 request identities and 240 attempts; it does not authorize actions
  outside the frozen universe
- Validated input-token forecast: 96,550
- Validated output-token forecast: 12,200
- Validated input-only cost proxy: `$0.004055100`
- Prior M5.2.1 estimate retained for comparison: `$0.004174105833333334`
- Conservative proxy: `$0.006708384375`

The difference from the prior M5.2.1 estimate is intentional: M5.2.2 recomputes the frozen action
universe using rounded M5.1 Stage D request means by named-question count. Neither value is an
invoice or enforceable dollar cap.

## Mock adaptive branch verification

The generated preflight and direct runner simulations use the real controller, adapter, budget
ledger, and persistent event log with `DeterministicMockProvider` only.

| Scenario | Physical / attempts | Screens | Follow-ups | Skipped | Stop reason |
|---|---:|---:|---:|---:|---|
| Informative | 280 / 280 | 30 | 30 | 0 | instrumentation objectives complete |
| Uninformative | 250 / 250 | 30 | 0 | 30 | instrumentation objectives complete |
| Budget-pressure fixture | 250 / 250 | 15 | 15 | 0 | unique physical request ceiling exhausted |

The informative and uninformative runs share the same frozen Pilot A protocol and action universe,
but produce different valid acquisition sequences solely from acquired screen responses. The
budget-pressure run is an explicitly reduced mock fault fixture (250 requests, 750 attempts), not a
change to the live authorization scope. All three runs report formal certification and production
deployment as false, finish with no pending or ambiguous action, and persist request identities and
budget state.

## Fault injection and recovery

The dedicated suite covers pre-dispatch interruption, timeout after possible dispatch, response
without a durable receipt, crash after receipt persistence, duplicate worker reservation,
malformed JSON, missing decisions, duplicate named decisions, invalid probabilities, missing
usage, budget exhaustion, definitive retry exhaustion, matching-scope restart,
mismatched-authorization restart, mock-to-live restart, corrupted hash chains, ambiguous receipts,
mixed recovered/ambiguous sessions, and post-response attempt accounting.

Key safety results:

- ambiguous transport failure is attempted once and is never automatically retried;
- persisted receipts recover without another provider request;
- undispatched reservations are released only when the durable log proves dispatch did not start;
- duplicate action identity reservation is atomic;
- all successful, failed, and ambiguous attempt counts consume reserved capacity;
- unresolved ambiguity blocks subsequent controller steps;
- protocol, authorization, policy, action universe, objective, and mock/live class must match on
  restart;
- the request and attempt ceilings cannot be over-allocated.

## Artifact integrity and reproducibility

The M5.2.2 preflight was regenerated through the mock-only runner after the repairs. It produced 24
artifacts and four documents, reported `new_jev_requests: 0`, and retained scope fingerprint
`a3027e6bc4827e8995f5faef0aab55fdf384ec7ceec67e986fd5b9f63a8ae32c`.

- M5.1 final audit: all 22 audit artifacts, 11 figures, and the final report match their manifest.
- M5.2 offline results: all 21 artifacts and four documents match their manifest.
- M5.2.1 offline results: all 21 artifacts and four documents match their manifest.
- M5.2.2 preflight: all 23 subordinate artifacts and four documents match its manifest.
- M5.2.2 integrity-manifest SHA-256:
  `acb4d36fe35509026aa7d27ee81fc9758de1ea88775d1597c276151ed141ab60`.
- Unsigned authorization-proposal SHA-256:
  `7a4079a5f6748dd3317776814e457600bcee27c815bdc92b50d83f8f1c9e5ebc`.
- No tracked M1-M4.6 result artifact changed.

The M5.2.2 artifact generator's fault result now identifies itself as a regression contract whose
PASS values require corroboration by the dedicated pytest run; it is not presented as independent
runtime attestation.

## Authorization safety

The proposed manifest remains unsigned: `paid_calls_authorized` is false and `countersigned_by` is
null. Historical authorization reuse is false. The output directory
`results/milestone_5_2_2/pilot_a_live` does not exist. No live Jev result file exists, no executable
live session has started, and no Jev request was made during finalization.

The paid `execute` path validates the exact manifest scope and exact authorization contract before
the Pilot A provider module is imported or `from_env` is called. Preflight and simulation construct
only the deterministic mock provider. An unsigned manifest fails closed.

## Git dependency analysis

M5.2.2 directly imports the M5.2 evidence-track type and the M5.1 final-audit hash utility. The M5.2
offline layer also imports M5.1 selection/final-audit modules, and M5.2.1 imports the M5.2 controller,
evidence, replay, cost, and uncertainty layers. Therefore M5.2.2 cannot be committed reproducibly
as an isolated directory.

The commit order is:

1. `573c72e44516ae84f7adb77d313f0074a79b93c0` — **Finalize Milestone 5.1
   research audit dependencies.** Thirty-four M5.1 audit, runner, report, figure, test, and minimal
   frozen-plan dependency files.
2. `43567faa3a0bb29abf83bdcab0910ce5da883114` — **Add Milestone 5.2 offline
   adaptive profiling.** Twenty M5.2 source, runner, test, document, and integrity-manifest files.
3. `24400fd96db33a347e9b5d194a3276a1cfc97e5c` — **Add Milestone 5.2.1
   feasibility analysis.** Fifteen M5.2.1 source, runner, test, document, and integrity-manifest
   files.
4. `bafffcb14e38a0664123621deaf481df743f7b57` — **Finalize Milestone 5.2.2 Pilot A
   preflight.** Twenty-one M5.2.2 controller, safety, provider, analysis, runner, test, and document
   files.
5. `5d4ef6636446a0ad5f47ef2c4e9eb635dcf463a6` — **Harden clean-checkout artifact
   test guards.** Two historical test modules, seven marker lines only.
6. This finalization report is committed last; its self-referential commit hash is recorded in the
   external final response and Git history rather than embedded in its own content.

No remote push, merge, branch creation, or history rewrite was performed.

## Clean-checkout verification

An LF-preserving detached worktree at commit
`5d4ef6636446a0ad5f47ef2c4e9eb635dcf463a6` was validated using the existing locked local
environment and that checkout's own `src` tree:

- the complete source-only suite passed with 375 passes and 84 explicit optional-artifact skips;
- all 33 dedicated M5.2.2 tests passed without skips;
- the preflight regenerated 24 artifacts and four documents, reported zero new Jev requests, and
  reproduced scope fingerprint
  `a3027e6bc4827e8995f5faef0aab55fdf384ec7ceec67e986fd5b9f63a8ae32c`;
- content hashes of the four generated tracked documents remained identical to the index;
- an `execute` invocation using the unsigned proposal exited 1 with `PermissionError: a
  countersigned live manifest is required`, and the live directory was absent both before and
  after the command.

The first isolated diagnostic run also exposed the incomplete artifact markers fixed in
`5d4ef66`. A fresh checkout with preserved LF content then produced the passing results above.
No network or provider request was made.

## Remaining limitations

- Pilot A is not authorized or executed; provider behavior is validated with mocked HTTP responses,
  not a live Jev request.
- Twenty representative records per workload cannot meet the preserved formal qualification
  requirements. No semantic accuracy or batching-quality certification is claimed.
- Token costs are forecasts/proxies and the protocol has no hard token-denominated cap.
- The durable writer and budget lock are process-local; distributed multi-process execution remains
  unsupported for Pilot A.
- Large ignored historical paid evidence is not duplicated into new commits. Clean checkouts can
  run source-only tests, while tests that require the optional historical corpus may skip unless
  that corpus is installed. The minimal frozen fingerprints required by M5.2.2 preflight are
  committed with their dependency layers.

## Readiness verdict

`M522_FINALIZED_READY_FOR_M523`

M5.2.2 is code-, artifact-, mock-, safety-, and commit-ready for the separately authorized M5.2.3
phase. This verdict does not authorize M5.2.3, Pilot A, or any paid request; the current proposal
remains unsigned and live execution remains fail-closed.
