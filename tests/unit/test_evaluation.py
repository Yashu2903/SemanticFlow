from pathlib import Path

import pytest

from semanticflow.metrics.evaluation import LabelError, evaluate_labels


def labels(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "labels.csv"
    path.write_text("review_id,match\n" + body, encoding="utf-8")
    return path


def test_precision_recall_f1_and_labeled_subset(tmp_path: Path) -> None:
    path = labels(tmp_path, "a,1\nb,1\nc,0\nd,0\ne,\n")
    result = evaluate_labels(path, {"a", "c", "outside"}, {"a", "b", "c", "d", "e"})
    assert (result.true_positives, result.false_positives) == (1, 1)
    assert (result.false_negatives, result.true_negatives) == (1, 1)
    assert result.precision == result.recall == result.f1 == 0.5
    assert result.labeled_records == 4


@pytest.mark.parametrize("body", ["a,1\na,0\n", ",1\n", "a,maybe\n"])
def test_invalid_labels(tmp_path: Path, body: str) -> None:
    with pytest.raises(LabelError):
        evaluate_labels(labels(tmp_path, body), set())


def test_unknown_id_and_zero_positive_case(tmp_path: Path) -> None:
    with pytest.raises(LabelError, match="unknown"):
        evaluate_labels(labels(tmp_path, "x,1\n"), set(), {"a"})
    result = evaluate_labels(labels(tmp_path, "a,0\nb,0\n"), set())
    assert result.precision == result.recall == result.f1 == 0
    assert result.true_negatives == 2

