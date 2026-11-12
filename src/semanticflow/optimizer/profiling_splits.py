"""Profiling-only train/validation partitions shared across workloads.

All four Milestone 3 workloads share the same 200 profiling review IDs.  Partitioning by
review ID therefore yields consistent folds and prevents cross-workload leakage of the same
reviews into both training and validation roles.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

from semanticflow.optimizer.matrix import ProfileMatrix

PROFILING_SPLIT_SCHEMA_VERSION = "m45-profiling-split-v1"


def ranked_review_ids(review_ids: Sequence[str], seed: int) -> list[str]:
    return sorted(
        review_ids,
        key=lambda review_id: (
            hashlib.sha256(f"{seed}:{review_id}".encode()).hexdigest(),
            review_id,
        ),
    )


def fold_assignments(
    review_ids: Sequence[str],
    *,
    folds: int,
    seed: int,
) -> dict[str, int]:
    if folds < 2:
        raise ValueError("profiling cross-validation requires at least two folds")
    ranked = ranked_review_ids(review_ids, seed)
    return {review_id: index % folds for index, review_id in enumerate(ranked)}


def train_validation_ids(
    review_ids: Sequence[str],
    *,
    folds: int,
    fold: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    if not 0 <= fold < folds:
        raise ValueError(f"fold must lie in [0, {folds})")
    assignments = fold_assignments(review_ids, folds=folds, seed=seed)
    train = [review_id for review_id in review_ids if assignments[review_id] != fold]
    validation = [review_id for review_id in review_ids if assignments[review_id] == fold]
    if not train or not validation:
        raise ValueError("train and validation partitions must both be non-empty")
    return train, validation


def split_matrix(
    matrix: ProfileMatrix,
    *,
    folds: int,
    fold: int,
    seed: int,
) -> tuple[ProfileMatrix, ProfileMatrix, dict[str, Any]]:
    """Return (train, validation, metadata) with no shared review IDs."""
    train_ids, validation_ids = train_validation_ids(
        [row.review_id for row in matrix.rows],
        folds=folds,
        fold=fold,
        seed=seed,
    )
    train = matrix.select_reviews(train_ids)
    validation = matrix.select_reviews(validation_ids)
    overlap = {row.review_id for row in train.rows} & {row.review_id for row in validation.rows}
    if overlap:
        raise ValueError("train/validation leak detected")
    metadata = {
        "schema_version": PROFILING_SPLIT_SCHEMA_VERSION,
        "folds": folds,
        "fold": fold,
        "seed": seed,
        "train_size": train.size,
        "validation_size": validation.size,
        "assignment": f"sha256({seed}:review_id) rank modulo {folds}",
        "held_out_live_reviews_used": False,
    }
    return train, validation, metadata


def assert_consistent_partitions(
    matrices: Sequence[ProfileMatrix],
    *,
    folds: int,
    seed: int,
) -> dict[str, Any]:
    """Verify every matrix uses the same review IDs and fold assignments."""
    if not matrices:
        raise ValueError("at least one matrix is required")
    reference = sorted(row.review_id for row in matrices[0].rows)
    reference_assignments = fold_assignments(reference, folds=folds, seed=seed)
    for matrix in matrices[1:]:
        ids = sorted(row.review_id for row in matrix.rows)
        if ids != reference:
            raise ValueError("profiling matrices do not share the same review IDs")
        if fold_assignments(ids, folds=folds, seed=seed) != reference_assignments:
            raise ValueError("fold assignments disagree across workloads")
    return {
        "schema_version": PROFILING_SPLIT_SCHEMA_VERSION,
        "review_count": len(reference),
        "folds": folds,
        "seed": seed,
        "consistent_across_matrices": True,
    }
