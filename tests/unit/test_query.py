from pathlib import Path

import pytest
from pydantic import ValidationError

from semanticflow.query import QuerySpec, load_query


def valid(**changes):
    payload = {
        "query_id": "q1",
        "dataset_path": "reviews.parquet",
        "limit": 10,
        "threshold": 0.5,
        "max_calls": 20,
        "predicates": [{"id": "a", "instructions": "Is this negative?"}],
    }
    payload.update(changes)
    return payload


def test_valid_query_preserves_predicate_order() -> None:
    query = QuerySpec.model_validate(
        valid(predicates=[{"id": "b", "instructions": "B?"}, {"id": "a", "instructions": "A?"}])
    )
    assert [item.id for item in query.predicates] == ["b", "a"]
    assert query.workload_id == "q1"


@pytest.mark.parametrize(
    ("field", "value"),
    [("threshold", -0.1), ("threshold", 1.1), ("limit", 0), ("max_calls", 0)],
)
def test_invalid_numeric_fields(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        QuerySpec.model_validate(valid(**{field: value}))


def test_rejects_duplicate_and_empty_predicates() -> None:
    with pytest.raises(ValidationError, match="unique"):
        QuerySpec.model_validate(
            valid(predicates=[{"id": "a", "instructions": "A"}, {"id": "a", "instructions": "B"}])
        )
    with pytest.raises(ValidationError):
        QuerySpec.model_validate(valid(predicates=[]))
    with pytest.raises(ValidationError):
        QuerySpec.model_validate(valid(predicates=[{"id": "a", "instructions": "  "}]))


def test_rejects_unsupported_operation_and_too_many_predicates() -> None:
    with pytest.raises(ValidationError):
        QuerySpec.model_validate(valid(operation="or"))
    with pytest.raises(ValidationError):
        QuerySpec.model_validate(
            valid(predicates=[{"id": str(i), "instructions": "x"} for i in range(7)])
        )


def test_load_query(tmp_path: Path) -> None:
    path = tmp_path / "query.yaml"
    path.write_text(
        "query_id: q\ndataset_path: x.parquet\nlimit: 1\nthreshold: 0.5\n"
        "max_calls: 1\npredicates:\n  - id: p\n    instructions: Question?\n",
        encoding="utf-8",
    )
    assert load_query(path).query_id == "q"
    with pytest.raises(FileNotFoundError):
        load_query(tmp_path / "missing.yaml")

