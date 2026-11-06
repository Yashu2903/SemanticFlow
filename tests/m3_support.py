"""Shared fixtures for Milestone 3 tests: synthetic matrices and stored-decision rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from semanticflow.optimizer.cardinality import predicate_statistics_from_matrix
from semanticflow.optimizer.matrix import MatrixRow, ProfileMatrix
from semanticflow.optimizer.models import ProfilingResult, SplitManifest
from semanticflow.optimizer.planner import predicate_hash
from semanticflow.query.models import QuerySpec

MODEL = "mock-model-1"


def hashes(predicate_ids: Sequence[str]) -> dict[str, str]:
    return {item: predicate_hash(item, f"{item}?") for item in predicate_ids}


def matrix_from_patterns(
    predicate_ids: Sequence[str],
    patterns: Sequence[Sequence[bool]],
    *,
    input_tokens: Mapping[str, int] | None = None,
    output_tokens: Mapping[str, int] | None = None,
    prefix: str = "p",
) -> ProfileMatrix:
    input_tokens = input_tokens or {}
    output_tokens = output_tokens or {}
    rows = [
        MatrixRow(
            review_id=f"{prefix}{index:03}",
            outcomes=tuple(bool(value) for value in pattern),
            input_tokens=tuple(input_tokens.get(item, 10) for item in predicate_ids),
            output_tokens=tuple(output_tokens.get(item, 0) for item in predicate_ids),
            latency_seconds=tuple(0.1 for _ in predicate_ids),
        )
        for index, pattern in enumerate(patterns)
    ]
    return ProfileMatrix(predicate_ids, rows)


def trace(
    review_id: str,
    predicate_id: str,
    passed: bool,
    *,
    probability: float | None = None,
    model: str = MODEL,
    question_hash: str | None = None,
    input_tokens: int = 10,
    output_tokens: int = 0,
    threshold: float = 0.5,
    attempts: int = 1,
    latency: float = 0.1,
) -> dict:
    return {
        "run_id": "run",
        "review_id": review_id,
        "predicate_id": predicate_id,
        "probability": probability if probability is not None else (0.9 if passed else 0.1),
        "threshold": threshold,
        "passed": passed,
        "model_identifier": model,
        "question_hash": question_hash or predicate_hash(predicate_id, f"{predicate_id}?"),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "request_latency_seconds": latency,
        "request_attempt_count": attempts,
        "timestamp": "2026-11-05T00:00:00+00:00",
    }


def query(predicate_ids: Sequence[str]) -> QuerySpec:
    return QuerySpec.model_validate(
        {
            "query_id": "q",
            "dataset_path": "unused.parquet",
            "limit": 10,
            "threshold": 0.5,
            "max_calls": 100,
            "predicates": [{"id": item, "instructions": f"{item}?"} for item in predicate_ids],
        }
    )


def split_for(matrix: ProfileMatrix, evaluation_size: int = 100) -> SplitManifest:
    return SplitManifest(
        dataset_path="unused.parquet",
        dataset_fingerprint="dataset",
        dataset_version="v",
        seed=1,
        source_record_count=matrix.size + evaluation_size,
        profiling_size=matrix.size,
        evaluation_size=evaluation_size,
        profiling_ids=[row.review_id for row in matrix.rows],
        evaluation_ids=[f"e{index:03}" for index in range(evaluation_size)],
        profiling_fingerprint="profile",
        evaluation_fingerprint="evaluation",
    )


def profile_for(
    matrix: ProfileMatrix,
    *,
    input_price: float | None = None,
    output_price: float | None = None,
) -> ProfilingResult:
    ids = list(matrix.predicate_ids)
    statistics = predicate_statistics_from_matrix(
        matrix,
        question_hashes=hashes(ids),
        input_price_per_million=input_price,
        output_price_per_million=output_price,
        model_versions=[MODEL],
    )
    input_total = sum(sum(value or 0 for value in row.input_tokens) for row in matrix.rows)
    output_total = sum(sum(value or 0 for value in row.output_tokens) for row in matrix.rows)
    pricing = (
        {"input_per_million_usd": input_price, "output_per_million_usd": output_price}
        if input_price is not None
        else {}
    )
    decisions = matrix.size * len(ids)
    return ProfilingResult(
        complete=True,
        query_id="q",
        workload_id="q",
        dataset_fingerprint="dataset",
        profiling_fingerprint="profile",
        threshold=0.5,
        requested_model=MODEL,
        resolved_model_versions=[MODEL],
        predicate_order=ids,
        predicate_instructions={item: f"{item}?" for item in ids},
        predicate_hashes=hashes(ids),
        sample_size=matrix.size,
        requested_evaluations=decisions,
        completed_decisions=decisions,
        budget_blocked_evaluations=0,
        provider_attempts=decisions,
        http_attempts=0,
        successful_api_calls=0,
        failed_http_attempts=0,
        retry_attempts=0,
        provider_errors=0,
        input_tokens=input_total,
        output_tokens=output_total,
        total_tokens=input_total + output_total,
        summed_inference_latency_seconds=0,
        wall_time_seconds=0,
        estimated_cost_usd=None,
        pricing=pricing,
        statistics_version="unconditional-profile-v1",
        statistics=[statistics[item] for item in ids],
    )
