import pytest

from semanticflow.optimizer.selectivity import observed_selectivity, wilson_interval


def test_observed_selectivity_and_unknown_value() -> None:
    assert observed_selectivity(0, 0) is None
    assert observed_selectivity(0, 5) == 0.0
    assert observed_selectivity(5, 5) == 1.0
    assert observed_selectivity(1, 3) == pytest.approx(1 / 3)


def test_wilson_interval_is_bounded() -> None:
    assert wilson_interval(0, 0) is None
    lower, upper = wilson_interval(40, 200) or (None, None)
    assert lower is not None and upper is not None
    assert 0 <= lower < 0.2 < upper <= 1


@pytest.mark.parametrize("passing,completed", [(-1, 1), (2, 1)])
def test_selectivity_rejects_invalid_counts(passing: int, completed: int) -> None:
    with pytest.raises(ValueError):
        observed_selectivity(passing, completed)
