# Milestone 2 offline experiment

From the repository root, run:

```powershell
.\.venv\Scripts\python.exe experiments/milestone_2/run_experiment.py
```

This uses 200 real SemCEB reviews for unconditional profiling and 800 disjoint reviews for both
fixed and optimized deterministic-mock executions. It makes no network or paid API requests. Mock
token counts are synthetic and are not Jev measurements.

The output is a new directory under `results/milestone_2`. To choose a stable experiment ID, append
`--experiment-id <id>`. The wrapper defaults to the mock provider; use the main `semanticflow`
commands only after completing the documented Jev preflight and obtaining explicit authorization.

To demonstrate a genuine optimizer reorder with deterministic, predicate-specific synthetic costs
on the genuine SemCEB records, run:

```powershell
.\.venv\Scripts\python.exe experiments/milestone_2/run_controlled_reordering.py `
  --experiment-id offline_controlled_reordering_semceb_v1
```

This controlled workload fails if the selected order is unchanged, optimized execution does not
reduce completed decisions, or the final match-ID sets differ. Its probabilities, tokens, prices,
and costs are synthetic infrastructure fixtures, not Jev research results.

The read-only Jev profiling preflight is:

```powershell
.\.venv\Scripts\python.exe experiments/milestone_2/preflight_jev_profile.py
```

It validates the dataset, deterministic split, pinned model, credential availability, retry-
inclusive budget, pricing metadata, and projected usage without contacting Jev. The paid command
printed by the preflight must not be executed without experiment-specific authorization.
