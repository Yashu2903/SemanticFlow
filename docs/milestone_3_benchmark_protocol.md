# Milestone 3 benchmark protocol (not executed)

This protocol is prepared but **not run**. Every Jev step needs separate, explicit authorization.
`--authorize-paid` is only a CLI acknowledgement and is not that authorization.

## Workloads

| Workload | Predicates | Expected regime | Status |
|---|---|---|---|
| `product_quality` (`configs/queries/product_quality.yaml`) | poor_quality, product_defect, dissatisfaction | sparse, strongly positively correlated | Jev outcomes stored |
| `listing_mismatch` | description_mismatch, value_complaint, dissatisfaction | sparse complaints, positively correlated | mock-verified only |
| `positive_experience` | easy_to_use, meets_expectations, would_recommend | high selectivity, positively correlated | mock-verified only |
| `mixed_feedback` | praises_product, reports_problem, size_or_quantity_issue | praise and problems negatively correlated | mock-verified only |

All configurations read the real `reviews_1000.parquet` and share the Milestone 2 split: seed
2027, 200 profiling and 800 held-out reviews, with profiling fingerprint `sha256:ca43925d…` and
evaluation fingerprint `sha256:5a1df48e…`. Available structured fields are rating, product IDs,
user ID, title, timestamp, helpful votes, and verified purchase. They are not used as labels.
`listing_mismatch` reuses the exact `dissatisfaction` definition, so stored Jev decisions with
the same question hash remain reusable. Model quality and optimization gains for the three new
workloads are **unmeasured**.

## Strategies compared

1. **Original fixed order:** the query file order.
2. **Independence optimization:** `configs/optimizer.milestone_2_jev.yaml`.
3. **Correlation-aware optimization:** `configs/optimizer.milestone_3.yaml`.
4. **Correlation-aware with profiling costs:** strategy 3 gated by `economics`, with the
   expected-execution horizon declared before any data are seen. If the decision is to decline,
   the strategy executes the original order, and its cost includes profiling. The
   confidence-aware variant (`configs/optimizer.milestone_3_confidence.yaml`) is reported
   alongside it.
5. **Adaptive profiling:** run the offline controller on the full profile to find its stopping
   checkpoint. Charge only the profiling decisions it would have consumed, and execute its plan.
   Live sequential profiling is not implemented, so this strategy is evaluated by simulation over
   the full profile.

Profiling is shared across strategies. For each workload, run held-out execution once for the
original order and once per distinct selected order. Reuse a run whenever two strategies select
the same order.

## Procedure per workload

1. Run the offline preflight. It makes no network requests and reads no credentials:
   `.\.venv\Scripts\python.exe experiments/milestone_3/preflight_workloads.py --output <new file>`.
2. **Paid, needs authorization.** Profile 200 reviews × 3 predicates (600 decisions, at most
   1,800 HTTP attempts):
   `semanticflow profile --provider jev --authorize-paid --query <workload> --settings configs/settings.milestone_2_jev.yaml --optimizer-config configs/optimizer.milestone_3.yaml --experiment-dir results/milestone_3/jev_<workload>/profile`.
3. Offline: for each strategy config, copy the profile with
   `experiments/milestone_3/prepare_strategy_experiment.py --source <profile dir> --destination <strategy dir>`,
   then run `semanticflow plan --optimizer-config <strategy config> --experiment-dir <strategy dir>`.
4. **Paid, needs authorization.** For each distinct order, run
   `semanticflow run --provider jev --authorize-paid --strategy {fixed|optimized} ...` on the 800
   held-out reviews, with at most 2,640 HTTP attempts per run.
5. **Paid, needs authorization.** Run a fresh repeat of the fixed order as a variability control.
6. Offline: `semanticflow analyze` and `semanticflow compare` on each directory.

## Metrics and controls

- **Execution:** logical evaluations, completed decisions, HTTP attempts, retries, input and
  output tokens, cost, summed latency, wall time, and stage counts.
- **Quality equivalence:** exact match-ID set equality; shared decision-pair agreement; and
  threshold crossings with their review IDs. Replay under the three resolution policies gives
  offline estimates of how much of a difference ordering accounts for; these are not causal
  measurements. Only contemporaneous, alternated executions measure the strategies directly.
  Exact equivalence is never claimed when IDs differ.
- **Model-variability controls:** a pinned `jev-1.14.0`; cache disabled; the repeated fixed run;
  near-threshold counts (±0.05); and run order alternated between strategies within one session.
- **Uncertainty:**
  - paired-bootstrap selection frequencies, savings intervals, and regret;
  - estimator MAE, MAPE, and signed error per stage;
  - predictive-interval coverage and PIT per estimator;
  - replay bounds for unexecuted orders;
  - break-even reported separately on observed, replay-estimated, and predicted savings.
- **Leakage:** profiling and held-out IDs disjoint by construction; the planner rejects any
  matrix containing held-out IDs.

## Preflight estimates

Unit costs come from the stored Milestone 2 reports: 313.9 input and 22.7 output tokens per
profiling decision, and 318.6 input and 22.1 output tokens per held-out decision. At the list
price this is $1.3183e-5 and $1.3383e-5 per decision. New predicates are assumed to cost the
same, which is approximate.

| Item | Decisions | Max HTTP attempts | Estimated cost | Cost if every allowed attempt were billed |
|---|---:|---:|---:|---:|
| Profiling, per workload | 600 | 1,800 | $0.00791 | $0.02373 |
| Held-out run, per order | 800–2,400 | 2,640 | $0.01071 – $0.03212 | $0.03533 |
| Per workload, worst case (profile + 6 runs) | ≤ 15,000 | 17,640 | ≤ $0.2006 | — |
| `product_quality` fixed-order repeat | ≈958 | 2,640 | $0.01282 | $0.03533 |
