# Milestone 4.6 Stage C Preflight Hardening Report

> **Stage C execution outcome addendum (2026-11-13):** The separately authorized full restart
> after the Windows temporary-path fix completed all 42 frozen runs. It used 36,796 logical
> evaluations and HTTP attempts, 12,103,700 input tokens, 954,161 output tokens, zero retries,
> zero failures, and $0.508355400. The successful session is
> `results/milestone_4_6/live_heldout_20261113_stage_c_restart_approved/`.
>
> The first authorized session remains an incomplete, non-resumable incident artifact. It made
> 7,742 attempts; 6,920 attempts and $0.099605058 were fully persisted, while the exact cost of
> the remaining 822 attempts is unrecoverable and is not imputed. That session is excluded from
> the analysis. Frozen plans, prompts, partitions, model, and schedule remained unchanged. No
> optional replication was run and no further Jev calls are authorized.

## 1. Executive summary

Pre-execution recommendation: **READY_WITH_DOCUMENTED_LIMITATIONS**. The authorized Stage C
restart subsequently completed successfully with the documented limitations preserved.

The frozen 42-run Stage C experiment was internally consistent at authorization and completed
under its exact scope. No optimizer, selected order, prompt, partition, or schedule was changed.
The corrected frozen plan file remains SHA-256
`06bbc304af7da38d2d9f717af68246e895262c32c1d77bb5c094d42300a2c26b`.

The expected forecast now accounts for early rejection and uses Stage B joint outcomes plus
provider-reported token usage. It predicts 36,759.43 logical evaluations and $0.515815590 for
the 42 runs. The request guards remain unchanged at 103,200 logical decisions and 309,600 HTTP
attempts. The earlier $1.828022490 no-retry and $5.484067470 retry-inclusive figures remain the
authorization proxies. They are not hard monetary ceilings; only decisions and attempts have
hard guards.

## 2. Real Stage B verification

| Dataset | Matrix | Decisions / attempts | Input / output tokens | Cost |
|---|---:|---:|---:|---:|
| Amazon All Beauty | 200 × 6 | 1,200 / 1,200 | 404,034 / 31,200 | $0.016969428 |
| Banking77 | 200 × 6 | 1,200 / 1,200 | 373,074 / 30,400 | $0.015669108 |
| Total | two complete matrices | 2,400 / 2,400 | 777,108 / 61,600 | $0.032638536 |

There were zero retries and failures. Requested and resolved model IDs are `jev-1.14.0`.
Dataset, partition, decision-file, and prompt hashes match their frozen manifests. The
pre-change audit is in `results/milestone_4_6/stage_c_preflight_hardening/baseline_audit.json`.

The canonical selected-workload payload hash is
`6597277ac7bcdb98d8bade4824d7bb4db938c888bcf84ad1670414426e34bdd0` in both the original and
budget-corrected plan artifacts. The correction changed no selected order.

## 3. Revised cost forecasts

The point forecast uses the frozen correlation-aware estimator for prefix reach. Token usage is
reach-conditioned where at least five profiling records support a stage; otherwise it falls back
to unconditional predicate usage. Stage B provider token observations are adjusted for the
frozen evaluation text lengths. Independence forecasts accompany every run. A 2,000-draw paired
row bootstrap supplies 90% profiling-sample sensitivity ranges; these are not guaranteed bounds.

| Dataset | Expected evaluations | Expected input tokens | Expected cost | Logical ceiling | Attempt ceiling |
|---|---:|---:|---:|---:|---:|
| Amazon All Beauty | 20,358.85 | 7,137,629 | $0.299779983 | 60,000 | 180,000 |
| Banking77 | 16,400.58 | 5,143,717 | $0.216035607 | 43,200 | 129,600 |
| Total | 36,759.43 | 12,281,346 | $0.515815590 | 103,200 | 309,600 |

The earlier $1.529269224 estimate assumed no early rejection. The revised $0.515815590 estimate
models sequential rejection from Stage B evidence. Authorization limits were not lowered.

### All 42 scheduled runs

The conservative columns are empirical per-run proxies. Their raw sum differs from the preserved
workload-rounded proposal by less than $0.000001.

| Run | Dataset | Expected calls | Max calls | Expected cost | Conservative | Retry-inclusive |
|---|---|---:|---:|---:|---:|---:|
| beauty_failure_usability_return-run-01 | amazon_all_beauty | 1,016.5 | 2400 | $0.014750 | $0.047785 | $0.143354 |
| beauty_failure_usability_return-run-02 | amazon_all_beauty | 821.8 | 2400 | $0.012159 | $0.047785 | $0.143354 |
| beauty_failure_usability_return-run-03 | amazon_all_beauty | 821.8 | 2400 | $0.012159 | $0.047785 | $0.143354 |
| beauty_failure_usability_return-run-04 | amazon_all_beauty | 1,016.5 | 2400 | $0.014750 | $0.047785 | $0.143354 |
| beauty_packaging_value-run-01 | amazon_all_beauty | 837.8 | 1600 | $0.012520 | $0.032318 | $0.096954 |
| beauty_packaging_value-run-02 | amazon_all_beauty | 837.8 | 1600 | $0.012270 | $0.032318 | $0.096954 |
| beauty_packaging_value-run-03 | amazon_all_beauty | 837.8 | 1600 | $0.012270 | $0.032318 | $0.096954 |
| beauty_packaging_value-run-04 | amazon_all_beauty | 837.8 | 1600 | $0.012520 | $0.032318 | $0.096954 |
| beauty_reaction_refund_packaging-run-01 | amazon_all_beauty | 826.0 | 2400 | $0.012186 | $0.048096 | $0.144287 |
| beauty_reaction_refund_packaging-run-02 | amazon_all_beauty | 818.0 | 2400 | $0.012106 | $0.048096 | $0.144287 |
| beauty_reaction_refund_packaging-run-03 | amazon_all_beauty | 818.0 | 2400 | $0.012106 | $0.048096 | $0.144287 |
| beauty_reaction_refund_packaging-run-04 | amazon_all_beauty | 826.0 | 2400 | $0.012186 | $0.048096 | $0.144287 |
| beauty_usability_packaging_return-run-01 | amazon_all_beauty | 885.6 | 2400 | $0.012880 | $0.047679 | $0.143038 |
| beauty_usability_packaging_return-run-02 | amazon_all_beauty | 821.8 | 2400 | $0.012159 | $0.047679 | $0.143038 |
| beauty_usability_packaging_return-run-03 | amazon_all_beauty | 821.8 | 2400 | $0.012159 | $0.047679 | $0.143038 |
| beauty_usability_packaging_return-run-04 | amazon_all_beauty | 885.6 | 2400 | $0.012880 | $0.047679 | $0.143038 |
| beauty_usability_packaging_return-run-05 | amazon_all_beauty | 821.7 | 2400 | $0.012159 | $0.047679 | $0.143038 |
| beauty_reaction_failure_value_return-run-01 | amazon_all_beauty | 834.3 | 3200 | $0.012311 | $0.064513 | $0.193539 |
| beauty_reaction_failure_value_return-run-02 | amazon_all_beauty | 837.9 | 3200 | $0.012271 | $0.064513 | $0.193539 |
| beauty_reaction_failure_value_return-run-03 | amazon_all_beauty | 837.9 | 3200 | $0.012271 | $0.064513 | $0.193539 |
| beauty_reaction_failure_value_return-run-04 | amazon_all_beauty | 834.3 | 3200 | $0.012311 | $0.064513 | $0.193539 |
| beauty_reaction_failure_value_return-run-05 | amazon_all_beauty | 826.0 | 3200 | $0.012186 | $0.064513 | $0.193539 |
| beauty_reaction_failure_value_return-run-06 | amazon_all_beauty | 818.0 | 3200 | $0.012106 | $0.064513 | $0.193539 |
| beauty_reaction_failure_value_return-run-07 | amazon_all_beauty | 818.1 | 3200 | $0.012105 | $0.064513 | $0.193539 |
| banking_unauthorized_card_refund-run-01 | banking77_queries | 894.0 | 2400 | $0.011726 | $0.034812 | $0.104436 |
| banking_unauthorized_card_refund-run-02 | banking77_queries | 897.2 | 2400 | $0.011774 | $0.034812 | $0.104436 |
| banking_unauthorized_card_refund-run-03 | banking77_queries | 897.2 | 2400 | $0.011774 | $0.034812 | $0.104436 |
| banking_unauthorized_card_refund-run-04 | banking77_queries | 894.0 | 2400 | $0.011726 | $0.034812 | $0.104436 |
| banking_withdrawal_access-run-01 | banking77_queries | 869.7 | 1600 | $0.011575 | $0.023271 | $0.069813 |
| banking_unauthorized_withdrawal_refund-run-01 | banking77_queries | 877.6 | 2400 | $0.011551 | $0.034950 | $0.104849 |
| banking_unauthorized_withdrawal_refund-run-02 | banking77_queries | 881.6 | 2400 | $0.011737 | $0.034950 | $0.104849 |
| banking_unauthorized_withdrawal_refund-run-03 | banking77_queries | 881.6 | 2400 | $0.011737 | $0.034950 | $0.104849 |
| banking_unauthorized_withdrawal_refund-run-04 | banking77_queries | 877.6 | 2400 | $0.011551 | $0.034950 | $0.104849 |
| banking_card_access-run-01 | banking77_queries | 1,064.7 | 1600 | $0.014111 | $0.023133 | $0.069400 |
| banking_card_access-run-02 | banking77_queries | 889.6 | 1600 | $0.011716 | $0.023133 | $0.069400 |
| banking_card_access-run-03 | banking77_queries | 889.6 | 1600 | $0.011716 | $0.023133 | $0.069400 |
| banking_card_access-run-04 | banking77_queries | 1,064.7 | 1600 | $0.014111 | $0.023133 | $0.069400 |
| banking_transfer_unauthorized_refund_access-run-01 | banking77_queries | 945.7 | 3200 | $0.012341 | $0.046078 | $0.138234 |
| banking_transfer_unauthorized_refund_access-run-02 | banking77_queries | 866.5 | 3200 | $0.011388 | $0.046078 | $0.138234 |
| banking_transfer_unauthorized_refund_access-run-03 | banking77_queries | 866.5 | 3200 | $0.011388 | $0.046078 | $0.138234 |
| banking_transfer_unauthorized_refund_access-run-04 | banking77_queries | 945.7 | 3200 | $0.012341 | $0.046078 | $0.138234 |
| banking_transfer_unauthorized_refund_access-run-05 | banking77_queries | 897.5 | 3200 | $0.011775 | $0.046078 | $0.138234 |

## 4. Shared profiling economics

The physical ledger identity includes dataset fingerprint, record, predicate, prompt hash, model,
threshold, question type, and cache state. Exactly 2,400 unique requests reconcile to
$0.032638536 without duplication. Standalone economics charge each workload independently;
shared economics charge only compatible identities not already paid for.

| Strategy | Standalone decisions / cost | Shared decisions / cost |
|---|---:|---:|
| fixed | 0 / $0 | 0 / $0 |
| independent | 5,800 / $0.079018422 | 2,400 / $0.032638536 |
| correlation_aware | 5,800 / $0.079018422 | 2,400 / $0.032638536 |
| existing_adaptive | 850 / $0.011375952 | 425 / $0.005658366 |
| confidence_aware | 5,800 / $0.079018422 | 2,400 / $0.032638536 |
| risk_aware | 5,800 / $0.079018422 | 2,400 / $0.032638536 |
| joint_adaptive_risk | 1,775 / $0.023579010 | 1,250 / $0.016559214 |
| milestone_4_5_frozen | 1,375 / $0.018394866 | 850 / $0.011375070 |

Physical expenditure and the two counterfactual accounting modes remain explicitly separate.

## 5. Sparse-data analysis

Nine workloads have at least one candidate prefix below support 20. Unseen-prefix fallback occurs
for `beauty_reaction_refund_packaging`, `beauty_reaction_failure_value_return`, and
`banking_transfer_unauthorized_refund_access`. Six workloads have less than a 1% predicted gap:
`beauty_packaging_value`, `beauty_reaction_refund_packaging`,
`beauty_usability_packaging_return`, `beauty_reaction_failure_value_return`,
`banking_unauthorized_card_refund`, and `banking_transfer_unauthorized_refund_access`.
Only `banking_card_access` has moderate-or-strong support across candidate prefixes. These are
descriptive categories, not calibrated risk probabilities, and selected plans were not changed.

## 6. Potentially harmful reorderings

Profiling replay finds seven negative strategy/workload reorder decisions. Existing adaptive has:

| Workload | Replay saving vs original | Profiling depth |
|---|---:|---:|
| `beauty_packaging_value` | -$0.000031584 | 25 records |
| `banking_unauthorized_card_refund` | -$0.000044856 | 25 records |
| `banking_unauthorized_withdrawal_refund` | -$0.000133728 | 50 records |

Independence also selects the negative-replay order for the first two; both joint policies select
it for `beauty_packaging_value`. These are in-sample diagnostics, not held-out evidence.

## 7. Schedule verification

The schedule reproduces 42 unique run IDs and 24 workload/order combinations. All ten workloads
and eight strategies are mapped. Nine workloads use repeated ABBA blocks. The comparator equals
the original for `banking_withdrawal_access`, so one fixed run avoids redundant paid execution.
Five other selected orders are single-run exploratory comparisons. No comparator is missing and
no amendment is recommended. The final analysis distinguishes ABBA and exploratory evidence.

## 8. Statistical limitations

- Prefix estimates and bootstrap ranges come from 200 profiling records per dataset.
- Several conjunctions have zero support, making smoothing and backoff material.
- Evaluation-stage reach is unobserved until Stage C; forecast conditioning is approximate.
- Bootstrap ranges omit future Jev response variability and provider drift.
- Single-run exploratory orders have weaker evidence than repeated ABBA comparisons.
- Shared records induce dependence; predicate requests are not independent experiments.
- Match agreement cannot establish semantic accuracy or recall without labels.

## 9. Safety verification

Before Jev can be imported, the launch path checks the exact phrase, plan and budget SHA, schedule
SHA, dataset files and partition fingerprints, prompt hashes, model, cache setting, retry policy,
concurrency, output path, and request ceiling. Existing outputs fail closed. Paid-call resume is
intentionally unsupported because an interrupted request cannot always be proven unbilled.

Attempt reservation is asynchronous and lock-protected. JSON and JSONL writes use atomic replace.
Incomplete runs cannot be analyzed as complete. Provider-free negative tests cover absent or bad
authorization, plan/schedule/prompt/dataset drift, invalid ceilings, conflicting output,
concurrent budget enforcement, and atomic artifact generation.

## 10. Final outcome

**STAGE_C_COMPLETE_WITH_DOCUMENTED_LIMITATIONS**

The separately authorized restart executed exactly the frozen 42-run schedule and stayed below
both hard request guards: 36,796 of 103,200 logical evaluations and 36,796 of 309,600 permitted
HTTP attempts. It completed with no retries or failures. Actual cost was $0.508355400, compared
with the $0.515815590 early-rejection forecast, $1.828022490 conservative no-retry proxy, and
$5.484067470 retry-inclusive proxy.

The machine-readable execution audit is
`results/milestone_4_6/stage_c_restart_execution_audit.json`; the preregistered analysis is
`results/milestone_4_6/stage_c_analysis_20261113_restart.json`. The failed partial session is
retained for audit but excluded. Optional Stage D replication remains unauthorized.
