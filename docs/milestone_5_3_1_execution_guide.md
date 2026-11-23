# Milestone 5.3.1 Execution Guide

## Dry run

```powershell
.venv\Scripts\python.exe -m experiments.milestone_5_3_1.run_live --output results/milestone_5_3_1/live_validation_session_01
```

This is the default zero-call mode. Review `frozen_experiment_config.json`, `fresh_record_selection_manifest.csv`, `request_plan.csv`, and `dry_run_report.json` before authorization.

## Authorized live execution

The exact frozen command is:

```powershell
.venv\Scripts\python.exe -m experiments.milestone_5_3_1.run_live --authorize-paid --output results/milestone_5_3_1/live_validation_session_01 --hard-request-limit 1100 --hard-attempt-limit 3300 --hard-dollar-limit 0.025
```

The `--authorize-paid` flag is the explicit spending gate. It authorizes only the source-derived M5.3.1 fingerprint, 1,100-request universe, 3,300-attempt ceiling, and $0.025 token-priced cap. It must not be supplied until the user separately authorizes paid execution.

The runner appends a hash-chained event before dispatch and persists the provider response, raw payload, and token usage before semantic validation. Completed requests are skipped on resume. A pending dispatch without a durable response is ambiguous and is not resent. Do not delete or edit a failed session; use reconciliation or a separately authorized fresh session.

## Analysis after complete collection

```powershell
.venv\Scripts\python.exe -m experiments.milestone_5_3_1.run_analysis --session results/milestone_5_3_1/live_validation_session_01 --output results/milestone_5_3_1/live_validation_session_01/analysis
```

Analysis refuses partial or predicate-incompatible evidence. It generates seed-level results, aggregate comparisons, paired policy differences, workload-specific results, sparse-event intervals, a disagreement review sheet, and an integrity manifest.

## Interpretation

The 150 representative-random records support holdout-style descriptive statements. The 150 enrichment records measure deliberately targeted behavior and must not be pooled into an unweighted population estimate. If disagreement counts remain small, RQ2 is inconclusive regardless of policy mean ordering. Human review is required before calling an execution disagreement a semantic error.
