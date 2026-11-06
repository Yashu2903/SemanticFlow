from __future__ import annotations

import hashlib
import itertools
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from semanticflow.optimizer.selectivity import wilson_interval

MATRIX_SCHEMA_VERSION = "m3-profile-matrix-v1"


@dataclass(frozen=True, slots=True)
class MatrixRow:
    """All recorded profiling outcomes and usage for one review, aligned to predicate order."""

    review_id: str
    outcomes: tuple[bool | None, ...]
    input_tokens: tuple[int | None, ...]
    output_tokens: tuple[int | None, ...]
    latency_seconds: tuple[float | None, ...]


class ProfileMatrix:
    """Review-by-predicate decision matrix built only from recorded profiling outcomes."""

    def __init__(
        self,
        predicate_ids: Sequence[str],
        rows: Sequence[MatrixRow],
        *,
        resampled: bool = False,
    ) -> None:
        ids = tuple(predicate_ids)
        if not 1 <= len(ids) <= 6:
            raise ValueError("profile matrix supports 1 to 6 predicates")
        if len(ids) != len(set(ids)):
            raise ValueError("profile matrix predicate IDs must be unique")
        if not rows:
            raise ValueError("profile matrix requires at least one row")
        for row in rows:
            widths = {
                len(row.outcomes),
                len(row.input_tokens),
                len(row.output_tokens),
                len(row.latency_seconds),
            }
            if widths != {len(ids)}:
                raise ValueError(f"row {row.review_id} does not match predicate count")
        if not resampled and len({row.review_id for row in rows}) != len(rows):
            raise ValueError("profile matrix review IDs must be unique")
        self.predicate_ids = ids
        self.rows = tuple(rows)
        self.resampled = resampled
        self._pass_masks: list[int] = []
        self._observed_masks: list[int] = []
        for row in self.rows:
            passed = observed = 0
            for index, outcome in enumerate(row.outcomes):
                if outcome is not None:
                    observed |= 1 << index
                    if outcome:
                        passed |= 1 << index
            self._pass_masks.append(passed)
            self._observed_masks.append(observed)
        self._count_cache: dict[int, tuple[int, int]] = {}

    @classmethod
    def from_traces(
        cls,
        traces: Iterable[Mapping[str, Any]],
        predicate_ids: Sequence[str],
        *,
        review_ids: Sequence[str],
        forbidden_review_ids: Iterable[str] = (),
        model_identifier: str | None = None,
        question_hashes: Mapping[str, str] | None = None,
    ) -> ProfileMatrix:
        """Build a matrix for exactly ``review_ids``; unknown cells stay ``None``.

        Traces for held-out (``forbidden_review_ids``) reviews are rejected, never filtered,
        so evaluation outcomes cannot leak into estimation by accident.
        """
        predicate_index = {predicate_id: index for index, predicate_id in enumerate(predicate_ids)}
        allowed = list(dict.fromkeys(review_ids))
        allowed_set = set(allowed)
        forbidden = set(forbidden_review_ids)
        if allowed_set & forbidden:
            raise ValueError("profiling review IDs overlap held-out evaluation IDs")
        cells: dict[tuple[str, int], Mapping[str, Any]] = {}
        for trace in traces:
            predicate_id = str(trace["predicate_id"])
            if predicate_id not in predicate_index:
                continue
            if model_identifier is not None and trace["model_identifier"] != model_identifier:
                continue
            if question_hashes is not None and (
                trace["question_hash"] != question_hashes[predicate_id]
            ):
                continue
            review_id = str(trace["review_id"])
            if review_id in forbidden:
                raise ValueError(f"held-out review {review_id} appears in profiling traces")
            if review_id not in allowed_set:
                raise ValueError(f"trace review {review_id} is outside the profiling split")
            key = (review_id, predicate_index[predicate_id])
            if key in cells:
                raise ValueError(f"duplicate profiling decision for {review_id}/{predicate_id}")
            cells[key] = trace
        width = len(predicate_index)
        rows: list[MatrixRow] = []
        for review_id in allowed:
            found = [cells.get((review_id, index)) for index in range(width)]
            rows.append(
                MatrixRow(
                    review_id=review_id,
                    outcomes=tuple(
                        bool(item["passed"]) if item is not None else None for item in found
                    ),
                    input_tokens=tuple(
                        _optional_int(item.get("input_tokens")) if item is not None else None
                        for item in found
                    ),
                    output_tokens=tuple(
                        _optional_int(item.get("output_tokens")) if item is not None else None
                        for item in found
                    ),
                    latency_seconds=tuple(
                        float(item["request_latency_seconds"]) if item is not None else None
                        for item in found
                    ),
                )
            )
        return cls(predicate_ids, rows)

    @property
    def size(self) -> int:
        return len(self.rows)

    @property
    def full_mask(self) -> int:
        return (1 << len(self.predicate_ids)) - 1

    @property
    def missing_cells(self) -> int:
        return sum(row.outcomes.count(None) for row in self.rows)

    @property
    def complete(self) -> bool:
        return self.missing_cells == 0

    def index(self, predicate_id: str) -> int:
        try:
            return self.predicate_ids.index(predicate_id)
        except ValueError as exc:
            raise ValueError(f"predicate {predicate_id} is not in the profile matrix") from exc

    def mask(self, predicate_ids: Iterable[str]) -> int:
        value = 0
        for predicate_id in predicate_ids:
            value |= 1 << self.index(predicate_id)
        return value

    def count(self, mask: int) -> tuple[int, int]:
        """Return (rows passing every predicate in mask, rows observing every predicate)."""
        cached = self._count_cache.get(mask)
        if cached is not None:
            return cached
        passing = support = 0
        for passed, observed in zip(self._pass_masks, self._observed_masks, strict=True):
            if observed & mask == mask:
                support += 1
                if passed & mask == mask:
                    passing += 1
        self._count_cache[mask] = (passing, support)
        return passing, support

    def complete_rows(self) -> ProfileMatrix:
        rows = [row for row in self.rows if None not in row.outcomes]
        if not rows:
            raise ValueError("profile matrix has no complete rows")
        return ProfileMatrix(self.predicate_ids, rows, resampled=self.resampled)

    def take(self, indices: Sequence[int]) -> ProfileMatrix:
        """Paired row selection; duplicate indices are allowed for bootstrap resamples."""
        return ProfileMatrix(
            self.predicate_ids, [self.rows[index] for index in indices], resampled=True
        )

    def select_reviews(self, review_ids: Sequence[str]) -> ProfileMatrix:
        by_id = {row.review_id: row for row in self.rows}
        missing = [review_id for review_id in review_ids if review_id not in by_id]
        if missing:
            raise ValueError(f"profile matrix lacks reviews: {missing[:3]}")
        return ProfileMatrix(self.predicate_ids, [by_id[review_id] for review_id in review_ids])

    def pattern_counts(self) -> Counter[tuple[bool, ...]]:
        return Counter(
            tuple(bool(value) for value in row.outcomes)
            for row in self.rows
            if None not in row.outcomes
        )

    def fingerprint(self) -> str:
        canonical = "\n".join(
            f"{row.review_id}\0{predicate_id}\0{row.outcomes[index]}"
            for row in sorted(self.rows, key=lambda item: item.review_id)
            for index, predicate_id in enumerate(self.predicate_ids)
        )
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def phi_coefficient(n11: int, n10: int, n01: int, n00: int) -> float | None:
    """Phi (Matthews) correlation of two binary outcomes; undefined for constant margins."""
    if min(n11, n10, n01, n00) < 0:
        raise ValueError("contingency counts cannot be negative")
    denominator = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    if denominator == 0:
        return None
    return (n11 * n00 - n10 * n01) / math.sqrt(denominator)


def joint_statistics(matrix: ProfileMatrix, *, version: str = "joint-profile-v1") -> dict[str, Any]:
    """Marginal, pairwise, conditional, and complete-conjunction statistics with supports."""
    ids = matrix.predicate_ids
    marginals = {}
    for predicate_id in ids:
        passing, support = matrix.count(matrix.mask([predicate_id]))
        interval = wilson_interval(passing, support) if support else None
        marginals[predicate_id] = {
            "passing": passing,
            "support": support,
            "selectivity": _ratio(passing, support),
            "wilson_lower": interval[0] if interval else None,
            "wilson_upper": interval[1] if interval else None,
        }
    pairwise = []
    for first, second in itertools.combinations(ids, 2):
        i, j = matrix.index(first), matrix.index(second)
        n11 = n10 = n01 = n00 = 0
        for row in matrix.rows:
            a, b = row.outcomes[i], row.outcomes[j]
            if a is None or b is None:
                continue
            if a and b:
                n11 += 1
            elif a:
                n10 += 1
            elif b:
                n01 += 1
            else:
                n00 += 1
        support = n11 + n10 + n01 + n00
        p_a = _ratio(n11 + n10, support)
        p_b = _ratio(n11 + n01, support)
        p_ab = _ratio(n11, support)
        lift = p_ab / (p_a * p_b) if p_ab is not None and p_a and p_b else None
        pairwise.append(
            {
                "predicates": [first, second],
                "support": support,
                "both_pass": n11,
                "first_only": n10,
                "second_only": n01,
                "neither": n00,
                "joint_selectivity": p_ab,
                "independence_joint_selectivity": (
                    p_a * p_b if p_a is not None and p_b is not None else None
                ),
                f"p_{second}_given_{first}": _ratio(n11, n11 + n10),
                f"p_{second}_given_{first}_support": n11 + n10,
                f"p_{first}_given_{second}": _ratio(n11, n11 + n01),
                f"p_{first}_given_{second}_support": n11 + n01,
                "phi": phi_coefficient(n11, n10, n01, n00),
                "lift": lift,
            }
        )
    conjunctions = []
    for size in range(1, len(ids) + 1):
        for subset in itertools.combinations(ids, size):
            passing, support = matrix.count(matrix.mask(subset))
            conjunctions.append(
                {
                    "predicates": list(subset),
                    "passing": passing,
                    "support": support,
                    "selectivity": _ratio(passing, support),
                }
            )
    complete = matrix.complete_rows() if matrix.missing_cells else matrix
    patterns = [
        {
            "outcomes": dict(zip(ids, pattern, strict=True)),
            "count": count,
        }
        for pattern, count in sorted(
            complete.pattern_counts().items(), key=lambda item: (-item[1], item[0])
        )
    ]
    return {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "joint_statistics_version": version,
        "predicate_ids": list(ids),
        "rows": matrix.size,
        "complete_rows": complete.size,
        "missing_cells": matrix.missing_cells,
        "matrix_complete": matrix.complete,
        "matrix_fingerprint": matrix.fingerprint(),
        "source": "recorded_profiling_outcomes",
        "marginals": marginals,
        "pairwise": pairwise,
        "conjunctions": conjunctions,
        "observed_outcome_patterns": patterns,
    }
