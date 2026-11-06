import itertools

import pytest

from semanticflow.dataset.loader import ReviewRecord
from semanticflow.executor.sequential import SequentialExecutor
from semanticflow.providers.mock import MockProvider
from semanticflow.query.models import QuerySpec


@pytest.mark.asyncio
async def test_every_and_order_has_same_matches_but_may_use_different_evaluations() -> None:
    records = [ReviewRecord(str(index), f"text {index}", {}) for index in range(6)]
    predicate_ids = ["A", "B", "C"]
    probabilities = {}
    for index, record in enumerate(records):
        probabilities[(record.review_text, "A")] = 0.9 if index != 0 else 0.1
        probabilities[(record.review_text, "B")] = 0.9 if index < 3 else 0.1
        probabilities[(record.review_text, "C")] = 0.9 if index % 2 else 0.1
    matching_sets: list[set[str]] = []
    evaluation_counts: list[int] = []
    for order in itertools.permutations(predicate_ids):
        query = QuerySpec.model_validate(
            {
                "query_id": "equivalence",
                "dataset_path": "unused.parquet",
                "limit": 6,
                "threshold": 0.5,
                "max_calls": 18,
                "predicates": [
                    {"id": item, "instructions": f"{item}?"} for item in order
                ],
            }
        )
        result = await SequentialExecutor(MockProvider(probabilities)).execute(
            records, query, "run"
        )
        matching_sets.append({record.review_id for record in result.matching_records})
        evaluation_counts.append(result.completed_decisions)
    assert all(item == matching_sets[0] for item in matching_sets)
    assert len(set(evaluation_counts)) > 1
