# Milestone 5 Stage B live fusion pilot report

## Status

Stage B completed on 2026-11-15 under the exact authorization phrase
`AUTHORIZE M5 STAGE B SMALL LIVE FUSION PILOT`. The run used `jev-1.14.0` and the frozen
fingerprint-bound schedule. No held-out evaluation records were read or executed.

## Provider and budget reconciliation

| Measure | Authorized/forecast | Observed |
|---|---:|---:|
| Logical requests | 1,230 | 1,230 |
| Maximum HTTP attempts | 3,690 | 1,230 |
| Retries | up to 2/request | 0 |
| Failures | — | 0 |
| Input tokens | 621,774 expected | 450,942 |
| Output tokens | 31,500 expected | 36,000 |
| Token-priced cost | $0.026114508 expected | $0.018939564 |

Observed input and token-priced cost were 27.47% below the preregistered expected proxy. Every
response resolved to `jev-1.14.0`. All 720 separate-reference calls, 480 conjunctive-binary calls,
and 30 multi-decision calls completed.

| Dataset | Requests | Input tokens | Output tokens | Cost |
|---|---:|---:|---:|---:|
| All Beauty | 410 | 154,635 | 12,480 | $0.006494670 |
| Banking77 | 410 | 147,808 | 12,060 | $0.006207936 |
| SMS | 410 | 148,499 | 11,460 | $0.006236958 |

This establishes that Jev's real endpoint accepts both implemented one-request contracts and
returns usable per-request accounting. It does not establish semantic accuracy.

## Semantic agreement

Across the twelve 40-record conjunctive arms, 472/480 group decisions agreed with the conjunction
of separate Jev decisions (98.33%). Five arms had at least one disagreement. Important examples:

- `beauty_malfunction AND beauty_hard_to_use`: 92.5% agreement, one true positive and three false
  negatives relative to the separate reference; recall 0.25.
- `banking_unauthorized_activity AND banking_refund_or_reversal`: 97.5% agreement and one false
  negative; recall 0.
- The corresponding three-predicate banking group also had one false negative.
- `sms_prize_or_reward AND sms_payment_or_fee AND sms_contact_action`: 95% agreement and two false
  positives.
- `sms_prize_or_reward AND sms_contact_action`: 97.5% agreement and one false positive.

Many other arms had zero or fewer than three positive reference examples, so apparently perfect
agreement is sparse evidence. None of the nine workload-level measured fused candidates passed the
preregistered quality gate. Quality-constrained and profiling-aware strategies therefore fell back
to unchanged separate adaptive execution for every workload.

The three 10-record multi-decision smoke requests produced 60 named predicate decisions. 59/60
agreed with separately issued questions. This validates response parsing and one-request batching,
not equivalence or production quality.

## Pilot economics

On pilot replay, full fusion was more expensive than fixed separate execution for all nine
workloads. Aggregate token-priced replay cost was $0.006741042 for full fusion versus $0.005536104
for fixed separate execution, a 21.77% increase. The cost-only joint selector chose only separate
predicate orders; its $0.005232402 aggregate was 5.49% below fixed, so the observed benefit came
from reordering rather than fusion.

These are pilot-sample replays, not held-out performance claims.

## Post-pilot analysis correction

The first offline freeze exposed an analysis defect: when earlier groups rejected every pilot
record, an unmeasured later fused group could be treated as unnecessary and the plan marked replay
complete. That is insufficient evidence for records which may reach the group later.

The original artifact is preserved under `results/milestone_5/frozen_plans_20261115_stage_c/`.
The analyzer now requires direct evidence for every fused group, even when that group was not
reached during the candidate replay, and a regression test covers the zero-reach case. The
correction changed one cost-only selection:

- `beauty_reaction_refund_packaging` changed from an unsupported
  `[adverse_reaction] -> [return_or_refund, packaging_failure]` plan to the measured all-separate
  `[adverse_reaction] -> [packaging_failure] -> [return_or_refund]` plan.

Quality-constrained and profiling-aware selections were unchanged and still fall back on all nine
workloads. Live prompts, calls, decisions, and costs were not modified. Corrected artifacts are in
`results/milestone_5/frozen_plans_20261115_stage_c_corrected_v2/`; this final version also binds
the analysis source and freeze-runner fingerprints.

## Stage D readiness

The corrected exploratory proposal contains 30 unique workload/plan executions, a no-early-
rejection ceiling of 27,600 logical requests, and a retry-inclusive ceiling of 82,800 HTTP
attempts. Pilot-scaled expected cost is $0.157931340. These values are not authorization.

Because no fused plan passed the gate and full fusion was costlier in every pilot workload, Stage D
would primarily measure negative-result generalization and the preregistered full/naive/cost-only
ablations. A separate decision is required before authorizing that expenditure.

## Conclusion

Stage B successfully validated genuine provider-compatible fusion mechanics, but the pilot does
not support adopting fusion: measured full fusion was economically unfavorable and no fused
candidate met the fixed quality requirement. Status: `READY_WITH_DOCUMENTED_LIMITATIONS` for a
separately considered Stage D benchmark; no Stage D calls are authorized.
