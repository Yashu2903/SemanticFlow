# Milestone 4.5 architecture

Milestone 4.5 calibrates and tunes the existing Milestone 3–4 optimizer components. It does not
add semantic operators, predicate fusion, operator placement, providers, or a new planner. Offline
commands never construct a provider and must not be used to issue Jev requests.

## Goals

1. Fit and select statistical parameters from **profiling-only** evidence.
2. Keep the 800-review live traces for a **frozen retrospective report** only.
3. Prefer the simplest policy that is competitive on profiling-validation economics.
4. Leave historical Milestone 1–4 artifacts unmodified.

## Module map

| Module | Responsibility |
|---|---|
| `profiling_splits.py` | Deterministic review-level K-fold partitions shared across workloads (`m45-profiling-split-v1`). |
| `m45_calibration.py` | Baseline fingerprinting, cardinality/bootstrap fold calibration, policy scoring, predefined grid tuning, config freeze, retrospective scoring. |
| Existing `cardinality.py`, `bootstrap.py`, `adaptive.py`, `risk.py`, `joint.py`, `voi_validation.py` | Unchanged semantics; consumed as libraries by the calibration pipeline. |

## Leakage boundary

```text
profiling matrix (200 reviews / workload)
  ├── train fold  → fit estimators, bootstrap, choose plans / hyperparameters
  └── validation fold → exact matrix-cost ranking, regret, cold-start net

held-out live traces (800 reviews)
  └── used only after freeze, labeled exploratory / not confirmatory
```

When the same SemCEB review IDs appear in multiple workloads, fold assignments are identical so a
review cannot train one workload and validate another under a different partition.

## Selection objective

Maximize mean profiling-validation **cold-start net** at \(R=10\):

\[
\text{net} = R\cdot(\text{validation original cost} - \text{validation selected cost})
- \text{profiling cost}.
\]

Tie-break by leave-one-workload-out floor, then lower complexity. Adopt the tuned joint policy as
the recommended default only if its mean margin over existing adaptive exceeds \$5e-5 and the
leave-one-out mean stays competitive. Otherwise retain `existing_adaptive`.

Held-out live outcomes are never part of this objective.

## Tunable parameters investigated

| Family | Grid / sweep |
|---|---|
| Correlation prior strength | \{0.5, 1.0, 2.0\} |
| Minimum saving probability | \{0.9, 0.95\} |
| Minimum relative saving | \{0.01\} |
| Adaptive checkpoints | (25,50,100,150,200), (25,50,100), (50,100,200) |
| Bootstrap iterations (diagnostic) | 500, 1000 (coverage); frozen eval restores 2000 |
| VOI rule / contraction | point estimate; exponent 0.5 (LCB remains experimental) |

## CLI

```powershell
semanticflow calibrate-m45 `
  --optimizer-config configs/optimizer.milestone_4.yaml `
  --output-dir results/milestone_4_5/<new directory> `
  --folds 4 --fold-seed 7073
```

The command refuses to overwrite an existing output directory, records `live_jev_calls: 0`, writes
`baseline/`, `profiling_calibration.json`, `tuning.json`, `frozen_config.json`,
`optimizer.milestone_4_5_frozen.yaml`, optional `retrospective/`, `report.json`, and `summary.md`.

Frozen candidate configuration for this milestone:
`configs/optimizer.milestone_4_5_frozen.yaml` (experimental strategy; recommended default remains
existing adaptive unless a future profiling-only margin clears the adoption threshold).

## Data contracts

- Profiling splits: `m45-profiling-split-v1`
- Calibration / retrospective reports: `m45-calibration-v1`
- Frozen config manifest: `m45-frozen-config-v1`

Historical Milestone 1–4 result directories are not rewritten.
