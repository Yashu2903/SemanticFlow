from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from semanticflow.dataset.loader import load_dataset
from semanticflow.executor.cache import RequestCache
from semanticflow.executor.sequential import SequentialExecutor
from semanticflow.metrics.evaluation import evaluate_labels
from semanticflow.metrics.reporter import build_report
from semanticflow.providers.base import InferenceProvider
from semanticflow.query.models import load_query
from semanticflow.storage.results import (
    ResultStore,
    build_manifest,
    dataset_fingerprint,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RunOutcome:
    run_dir: Path
    report: dict[str, Any]


def load_settings(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Settings file not found: {source}")
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Settings configuration must be a YAML mapping")
    return payload


async def run_query(
    *,
    query_path: str | Path,
    settings: dict[str, Any],
    provider: InferenceProvider,
    output_dir: str | Path,
    cache_enabled: bool = False,
    labels_path: str | Path | None = None,
    limit_override: int | None = None,
    max_calls_override: int | None = None,
) -> RunOutcome:
    query = load_query(query_path)
    if limit_override is not None:
        if limit_override <= 0:
            raise ValueError("record limit override must be positive")
        query.limit = limit_override
    if max_calls_override is not None:
        if max_calls_override <= 0:
            raise ValueError("max calls override must be positive")
        query.max_calls = max_calls_override
    records = load_dataset(query.dataset_path, limit=query.limit)
    executor_settings = settings.get("executor", {})
    provider_settings = settings.get("providers", {}).get(provider.name, {})
    concurrency = int(executor_settings.get("concurrency", 4))
    timeout_seconds = float(provider_settings.get("timeout_seconds", 30.0))
    max_retries = int(provider_settings.get("max_retries", 2))
    store = ResultStore(output_dir)
    run_id, run_dir, timestamp = store.create_run(query.query_id)
    LOGGER.info("Loaded %d records for workload %s", len(records), query.workload_id)

    execution = await SequentialExecutor(
        provider, concurrency=concurrency, cache=RequestCache(cache_enabled)
    ).execute(records, query, run_id)
    quality = None
    if labels_path is not None and execution.complete:
        quality = evaluate_labels(
            labels_path,
            {record.review_id for record in execution.matching_records},
            {record.review_id for record in records},
        )
    pricing = settings.get("pricing", {})
    input_price = pricing.get("input_per_million_usd")
    output_price = pricing.get("output_per_million_usd")
    report = build_report(
        execution,
        query,
        provider=provider.name,
        model=provider.model_identifier,
        input_price_per_million=float(input_price) if input_price is not None else None,
        output_price_per_million=float(output_price) if output_price is not None else None,
        quality=quality,
    )
    manifest = build_manifest(
        run_id=run_id,
        timestamp=timestamp,
        query=query,
        provider=provider.name,
        requested_model=provider.model_identifier,
        resolved_models=[trace.model_identifier for trace in execution.traces],
        dataset_hash=dataset_fingerprint(query.dataset_path),
        concurrency=concurrency,
        cache_enabled=cache_enabled,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        completion_status=report["completion_status"],
        pricing_id=pricing.get("id"),
    )
    store.write_artifacts(
        run_dir, execution.matching_records, execution.traces, report, manifest
    )
    return RunOutcome(run_dir=run_dir, report=report)
