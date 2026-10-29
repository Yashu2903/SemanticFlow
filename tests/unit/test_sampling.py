import pytest

from semanticflow.dataset import ReviewRecord, deterministic_sample


def test_sampling_is_deterministic_and_order_independent() -> None:
    records = [ReviewRecord(str(i), f"text {i}", {}) for i in range(20)]
    assert deterministic_sample(records, 5, 42) == deterministic_sample(
        list(reversed(records)), 5, 42
    )
    assert deterministic_sample(records, 5, 42) != deterministic_sample(records, 5, 43)


def test_sampling_validates_size() -> None:
    records = [ReviewRecord("1", "x", {})]
    with pytest.raises(ValueError):
        deterministic_sample(records, 0, 1)
    with pytest.raises(ValueError):
        deterministic_sample(records, 2, 1)
