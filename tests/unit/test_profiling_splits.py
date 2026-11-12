import pytest

from semanticflow.optimizer.profiling_splits import (
    assert_consistent_partitions,
    fold_assignments,
    split_matrix,
    train_validation_ids,
)
from tests.m3_support import matrix_from_patterns


def test_fold_assignments_are_deterministic_and_balanced() -> None:
    ids = [f"r{index:03}" for index in range(20)]
    first = fold_assignments(ids, folds=4, seed=9)
    second = fold_assignments(ids, folds=4, seed=9)
    assert first == second
    counts = [sum(value == fold for value in first.values()) for fold in range(4)]
    assert counts == [5, 5, 5, 5]


def test_train_validation_do_not_leak() -> None:
    ids = [f"r{index:03}" for index in range(12)]
    train, validation = train_validation_ids(ids, folds=3, fold=1, seed=3)
    assert set(train).isdisjoint(validation)
    assert set(train) | set(validation) == set(ids)
    with pytest.raises(ValueError):
        train_validation_ids(ids, folds=3, fold=3, seed=3)


def test_split_matrix_rejects_inconsistent_workloads() -> None:
    patterns = [(True, False), (False, True), (True, True), (False, False)] * 3
    left = matrix_from_patterns(["A", "B"], patterns, prefix="p")
    right = matrix_from_patterns(["A", "B"], patterns, prefix="q")
    with pytest.raises(ValueError, match="same review IDs"):
        assert_consistent_partitions([left, right], folds=3, seed=1)
    train, validation, meta = split_matrix(left, folds=3, fold=0, seed=1)
    assert meta["train_size"] + meta["validation_size"] == left.size
    assert {row.review_id for row in train.rows}.isdisjoint(
        {row.review_id for row in validation.rows}
    )
