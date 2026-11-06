# Milestone 3 offline experiments

Everything here runs offline over stored Jev traces or mocks. Nothing contacts a provider.

Regenerate the consolidated analysis into a **new** directory; existing directories are refused:

```powershell
.\.venv\Scripts\semanticflow.exe analyze --query configs/queries/product_quality.yaml `
  --optimizer-config configs/optimizer.milestone_3.yaml `
  --experiment-dir results/milestone_2/jev_product_quality_profile_20261102 `
  --historical-baseline-dir results/milestone_1_full_baseline/product_quality_20261102T163100.862806Z_d23f73f2cbd1 `
  --extra-decisions "m1_smoke_jev=results/product_quality_20261029T022635.415784Z_96caeebde714/decisions.jsonl" `
  --output-dir results/milestone_3/offline_analysis_<date>
```

Replay one order, or compare two, over any stored `decisions.jsonl` files:

```powershell
.\.venv\Scripts\semanticflow.exe replay --query configs/queries/product_quality.yaml `
  --decisions "m1=results/milestone_1_full_baseline/product_quality_20261102T163100.862806Z_d23f73f2cbd1/decisions.jsonl" `
  --split-manifest results/milestone_2/jev_product_quality_profile_20261102/split_manifest.json `
  --split-part evaluation --model jev-1.14.0 `
  --order poor_quality,product_defect,dissatisfaction `
  --order product_defect,poor_quality,dissatisfaction `
  --settings configs/settings.milestone_2_jev.yaml
```

Preflight the proposed paid experiments. This reads no credentials and makes no requests:

```powershell
.\.venv\Scripts\python.exe experiments/milestone_3/preflight_workloads.py
```

`prepare_strategy_experiment.py` copies split and profiling artifacts, with hash verification,
into a new directory. Another plan or a repeated run can then execute without writing into a
historical experiment.

The commands that preflight prints are paid. They must not be run without explicit,
experiment-specific authorization. See `docs/milestone_3_benchmark_protocol.md` and
`docs/milestone_3_research_report.md`.

Contemporaneous strategy comparison. Plan and schedule are offline; `benchmark-session` is
paid. Copy a stored profile first so historical experiment directories stay untouched:

```powershell
.\.venv\Scripts\python.exe experiments/milestone_3/prepare_strategy_experiment.py `
  --source results/milestone_2/jev_product_quality_profile_20261102 `
  --destination results/milestone_3/jev_product_quality_benchmark
.\.venv\Scripts\semanticflow.exe benchmark-plan --query configs/queries/product_quality.yaml `
  --optimizer-config configs/optimizer.milestone_3.yaml `
  --experiment-dir results/milestone_3/jev_product_quality_benchmark `
  --prior-cost-per-decision 0.00001318289 `
  --prior-cost-source "Milestone 2 stored profiling average" `
  --output results/milestone_3/jev_product_quality_benchmark/strategies.json
.\.venv\Scripts\semanticflow.exe benchmark-schedule `
  --strategies results/milestone_3/jev_product_quality_benchmark/strategies.json `
  --orders selected --abba-blocks 1 `
  --output results/milestone_3/jev_product_quality_benchmark/schedule.json
.\.venv\Scripts\semanticflow.exe benchmark-preflight `
  --strategies results/milestone_3/jev_product_quality_benchmark/strategies.json `
  --schedule results/milestone_3/jev_product_quality_benchmark/schedule.json `
  --cost-per-decision 0.00001338234
```
