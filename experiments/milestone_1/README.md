# Milestone 1 baseline

Prepare `data/processed/reviews_1000.parquet`, then run all four workloads with an explicit provider:

```powershell
python experiments/milestone_1/run_baseline.py --provider mock
$env:TYPESAFE_API_KEY = "..."
python experiments/milestone_1/run_baseline.py --provider jev --limit 50
```

Caching defaults to off. The live limit is intentionally explicit in the example to avoid
accidental paid batches. Each workload uses the same sequential executor and gets a separate result
directory. The runner performs no predicate optimization.
