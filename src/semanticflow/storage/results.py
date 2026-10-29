from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from semanticflow.dataset.loader import ReviewRecord
from semanticflow.metrics.collector import DecisionTrace
from semanticflow.query.models import QuerySpec


def dataset_fingerprint(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def git_commit() -> str | None:
    try:
        workspace = str(Path.cwd().resolve()).replace("\\", "/")
        return subprocess.run(
            ["git", "-c", f"safe.directory={workspace}", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


class ResultStore:
    def __init__(self, output_root: str | Path) -> None:
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def create_run(self, query_id: str) -> tuple[str, Path, str]:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        run_id = uuid.uuid4().hex[:12]
        run_dir = self.output_root / f"{query_id}_{timestamp}_{run_id}"
        run_dir.mkdir(parents=False, exist_ok=False)
        return run_id, run_dir, timestamp

    @staticmethod
    def write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    def write_artifacts(
        self,
        run_dir: Path,
        matches: list[ReviewRecord],
        traces: list[DecisionTrace],
        report: dict[str, Any],
        manifest: dict[str, Any],
    ) -> None:
        self.write_jsonl(run_dir / "matches.jsonl", [record.to_dict() for record in matches])
        self.write_jsonl(run_dir / "decisions.jsonl", [trace.to_dict() for trace in traces])
        self.write_json(run_dir / "report.json", report)
        self.write_json(run_dir / "manifest.json", manifest)


def build_manifest(
    *,
    run_id: str,
    timestamp: str,
    query: QuerySpec,
    provider: str,
    requested_model: str,
    resolved_models: list[str],
    dataset_hash: str,
    concurrency: int,
    cache_enabled: bool,
    timeout_seconds: float,
    max_retries: int,
    completion_status: str,
    pricing_id: str | None,
) -> dict[str, Any]:
    question_hashes = {
        predicate.id: hashlib.sha256(
            f"noul\0{predicate.id}\0{predicate.instructions}".encode()
        ).hexdigest()
        for predicate in query.predicates
    }
    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "semanticflow_version": "0.1.0",
        "git_commit": git_commit(),
        "query_id": query.query_id,
        "query_definition": query.model_dump(mode="json"),
        "workload_id": query.workload_id,
        "predicate_order": [predicate.id for predicate in query.predicates],
        "dataset_path": str(query.dataset_path),
        "dataset_fingerprint": dataset_hash,
        "dataset_version": query.dataset_version,
        "dataset_preparation_version": query.dataset_preparation_version,
        "sample_seed": query.sample_seed,
        "subset_selection_method": "stable-review-id-order-then-record-limit",
        "record_limit": query.limit,
        "provider": provider,
        "requested_model": requested_model,
        "resolved_model_versions": sorted(set(resolved_models)),
        "threshold": query.threshold,
        "max_calls": query.max_calls,
        "concurrency": concurrency,
        "timeout_seconds": timeout_seconds,
        "max_retries": max_retries,
        "caching_enabled": cache_enabled,
        "question_type": "noul",
        "question_hashes": question_hashes,
        "pricing_configuration_id": pricing_id,
        "completion_status": completion_status,
    }
