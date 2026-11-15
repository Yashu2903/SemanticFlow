# Milestone 5 preregistered experimental protocol

## Research question and hypotheses

The primary question is whether joint grouping and ordering reduces total Jev cost relative to
separate `existing_adaptive` execution while satisfying a fixed quality gate and recovering its
profiling expense. H1-H7 in the project brief are retained as hypotheses. Request count, input
tokens, dollars, latency, semantic disagreement, and cold-start economics are reported separately.

## Frozen inputs

The machine-readable protocol is `configs/milestone_5/protocol.yaml`; physical settings and the
quality gate are in `configs/milestone_5/fusion.yaml`; workload definitions are in
`configs/milestone_5/workloads.yaml`; pilot arms are in
`configs/milestone_5/pilot_groups.yaml`. Every live launch is fingerprint-bound.

Datasets and four-way partitions:

| Dataset | Domain | Development | Pilot | Selection | Evaluation | Dataset fingerprint |
|---|---|---:|---:|---:|---:|---|
| Amazon Reviews'23 All Beauty | reviews | 50 | 40 | 100 | 400 | `cfcd273412279d9bfa43f6eff103855fe89b3facec7d048ccc0e151d36328362` |
| Banking77 | support queries | 50 | 40 | 100 | 400 | `871adc1b21087fdf1a088ac996f2cecc26d30ca5bb641d3acb8e411be272e6e5` |
| UCI SMS Spam Collection | messages | 50 | 40 | 100 | 400 | `bd8b1ef522148d328def1f0722deaddb51a302898d613ba41da42ced9566a7f4` |

The 1,770 prepared records exclude texts exposed by M1-4.6. Exact duplicates and normalized
near-duplicates cannot cross partitions; deterministic group-aware splitting is used when a group
exists. All sources are public and provenance is stored in each manifest. M4.6 traces are used
only for separate-call token calibration and engineering compatibility, never as M5 fusion
outcomes.

## Workloads

Nine frozen AND workloads cover 2, 3, and 4 predicates in each domain:

- Beauty: `beauty_failure_usability`, `beauty_reaction_refund_packaging`,
  `beauty_reaction_failure_value_return`.
- Banking77: `banking_withdrawal_access`, `banking_unauthorized_card_refund`,
  `banking_transfer_unauthorized_refund_access`.
- SMS: `sms_urgency_security`, `sms_prize_fee_contact`,
  `sms_offer_prize_urgency_contact`.

The declared correlation/selectivity characteristics are design hypotheses until measured.

## Strategies

The comparisons are fixed separate order, unchanged existing adaptive separate execution, full
fusion, deterministic adjacent-first partial fusion, cost-only joint optimization, quality-
constrained joint optimization, and profiling-aware joint optimization. Identical physical plans
will share compatible measurements.

## Quality gate

Before held-out results, the gate requires 40 records, at least 3 positive separate-reference
examples, agreement >= 0.98, recall >= 0.95, false-negative rate <= 0.05, changed-decision rate
<= 0.02, real fused evidence, and complete-plan evidence. Sparse or undefined evidence fails
closed to `existing_adaptive`. No human labels are currently available, so semantic accuracy is
unverified even when the gate passes.

## Authorization stages

1. Stage A (this stage): implementation, fresh partitions, mocks, tests, docs; zero paid calls.
2. Stage B: small live fusion pilot; separately authorized exact frozen schedule.
3. Stage C: offline analysis and plan freeze from pilot and selection evidence only.
4. Stage D: separately authorized held-out execution over the 400-record evaluation partitions.
5. Stage E: separately authorized replication.

The Stage B pilot executes complete separate references (720 requests), twelve conjunctive group
arms (480), and three documented multi-decision capability smokes (30), totaling 1,230 logical
requests. Maximum retries are two, so the hard attempt ceiling is 3,690. Model `jev-1.14.0`,
concurrency 4, cache disabled, fixed timeout/retry behavior, exact prompts, and complete traces are
required.

Proposed command after explicit approval:

```powershell
.\.venv\Scripts\python.exe experiments/milestone_5/run_fusion.py pilot-profile `
  --authorization-manifest results/milestone_5/stage_a_preflight/stage_b_authorization_proposal_release_v2.json `
  --authorization "AUTHORIZE M5 STAGE B SMALL LIVE FUSION PILOT" `
  --output results/milestone_5/live_fusion_pilot_<approved_timestamp>
```

The output path must not already exist. The authorization phrase is scoped only to this bound
Stage B schedule; it does not authorize plan-freeze data collection, held-out evaluation, or
replication.

Input-token forecasts use the M4.6 observed provider-token/character-proxy mean ratios for Beauty
and Banking, and the cross-domain mean for SMS. The conservative proxy uses the per-domain
observed maximum, with SMS using the global maximum. These ratios are empirical, not guaranteed
bounds for fused prompts.

## Stage C and D analysis

Pilot analysis constructs complete physical-plan replays only where every group has a genuine
measurement on the same record. It never manufactures a fused decision. The unchanged adaptive
controller supplies the separate fallback. Plans and all hashes freeze before evaluation.

Stage D will use paired/counterbalanced measurements, deduplicate identical plans, and distinguish
replicated primary comparisons from single-run exploratory comparisons. Cost, ranking/regret
among measured valid candidates, quality, instability, dataset heterogeneity, and profiling
economics are reported at workload level. Reuse horizons are 1, 5, 10, 25, 50, 100, 250, and
1,000. Actual spend, standalone profiling, shared profiling, cold start, and warm start remain
separate ledgers. Calls are not treated as independent workload-level experiments.

## Interpretation

Savings require both lower total cost and gate compliance. No plan is called globally optimal
unless every necessary alternative has been measured. Mock outcomes are infrastructure-only.
Separate-predicate agreement is not human accuracy. Negative, harmful, and inconclusive outcomes
remain in the report.
