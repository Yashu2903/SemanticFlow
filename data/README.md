# Data

Dataset files are intentionally not committed. Download only the official SemCEB Amazon Arts,
Crafts & Sewing `raw_5core` prepared reviews (not images or embeddings):

```powershell
python scripts/download_dataset.py --source semceb `
  --output data/raw/reviews_filtered.parquet
```

The official distribution is documented at https://github.com/utndatasystems/SemCEB. If it becomes
inaccessible, `--source auto` falls back to McAuley-Lab Amazon Reviews 2023 and records that the raw
fallback does not reproduce SemCEB filtering.

Create and validate the versioned 1,000-review sample:

```powershell
python scripts/prepare_sample.py --source data/raw/reviews_filtered.parquet `
  --output data/processed/reviews_1000.parquet --sample-size 1000 --seed 2026
python scripts/validate_dataset.py --source data/raw/reviews_filtered.parquet `
  --dataset data/processed/reviews_1000.parquet `
  --manifest data/processed/dataset_manifest.json
```

Pass `--overwrite` only when intentionally replacing a prior snapshot. The sampler ranks stable
review IDs with SHA-256 of `seed:review_id`; it is deterministic and independent of input row order.
Generated IDs are content hashes. Duplicate source IDs select a canonical row deterministically.
Validation reruns preparation and compares stable ID/text content fingerprints.

Keep profiling/development, future calibration, and held-out evaluation samples separate. Human
labels belong under `data/labels/` and are ignored by Git by default because review text may be
subject to dataset licensing constraints.
