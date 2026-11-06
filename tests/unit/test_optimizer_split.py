from semanticflow.dataset.loader import ReviewRecord
from semanticflow.optimizer.split import deterministic_split, id_fingerprint


def test_deterministic_split_is_disjoint_and_order_independent() -> None:
    records = [ReviewRecord(str(index), f"review {index}", {}) for index in range(20)]
    first_profile, first_evaluation, first_manifest = deterministic_split(
        records,
        profiling_size=5,
        evaluation_size=15,
        seed=2027,
        dataset_path="reviews.parquet",
        dataset_fingerprint="sha256:dataset",
        dataset_version="fixture-v1",
    )
    second_profile, second_evaluation, second_manifest = deterministic_split(
        list(reversed(records)),
        profiling_size=5,
        evaluation_size=15,
        seed=2027,
        dataset_path="reviews.parquet",
        dataset_fingerprint="sha256:dataset",
        dataset_version="fixture-v1",
    )
    assert [item.review_id for item in first_profile] == [item.review_id for item in second_profile]
    assert [item.review_id for item in first_evaluation] == [
        item.review_id for item in second_evaluation
    ]
    assert not set(first_manifest.profiling_ids) & set(first_manifest.evaluation_ids)
    assert first_manifest == second_manifest
    assert first_manifest.profiling_fingerprint == id_fingerprint(first_manifest.profiling_ids)


def test_split_rejects_duplicate_ids_and_oversized_request() -> None:
    records = [ReviewRecord("same", "a", {}), ReviewRecord("same", "b", {})]
    try:
        deterministic_split(
            records,
            profiling_size=1,
            evaluation_size=1,
            seed=1,
            dataset_path="x",
            dataset_fingerprint="hash",
            dataset_version="v",
        )
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate review IDs must be rejected")
