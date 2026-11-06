from __future__ import annotations

import hashlib
import itertools
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any

EvidenceKey = tuple[str, str, str, str]


@dataclass(frozen=True, slots=True)
class DecisionObservation:
    """One stored provider decision; never synthesized."""

    review_id: str
    predicate_id: str
    model_identifier: str
    question_hash: str
    probability: float
    threshold: float
    passed: bool
    input_tokens: int | None
    output_tokens: int | None
    latency_seconds: float
    request_attempts: int
    source: str
    run_id: str
    timestamp: str

    @property
    def key(self) -> EvidenceKey:
        return (self.review_id, self.predicate_id, self.model_identifier, self.question_hash)

    def passed_at(self, threshold: float) -> bool:
        if threshold == self.threshold:
            return self.passed
        return self.probability >= threshold

    @classmethod
    def from_trace(cls, row: Mapping[str, Any], source: str) -> DecisionObservation:
        return cls(
            review_id=str(row["review_id"]),
            predicate_id=str(row["predicate_id"]),
            model_identifier=str(row["model_identifier"]),
            question_hash=str(row["question_hash"]),
            probability=float(row["probability"]),
            threshold=float(row["threshold"]),
            passed=bool(row["passed"]),
            input_tokens=None if row.get("input_tokens") is None else int(row["input_tokens"]),
            output_tokens=None if row.get("output_tokens") is None else int(row["output_tokens"]),
            latency_seconds=float(row.get("request_latency_seconds", 0.0)),
            request_attempts=int(row.get("request_attempt_count", 1)),
            source=source,
            run_id=str(row.get("run_id", "")),
            timestamp=str(row.get("timestamp", "")),
        )


@dataclass(slots=True)
class EvidenceSource:
    label: str
    description: str
    path: str | None
    sha256: str | None
    rows: int
    models: dict[str, int] = field(default_factory=dict)
    predicates: dict[str, int] = field(default_factory=dict)
    run_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Decision file not found: {source}")
    return [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class DecisionEvidence:
    """Index of stored decisions keyed by review, predicate, model version, and question hash.

    A key may hold several observations (repeated calls). Repetitions are kept separately so
    variability is visible rather than averaged away.
    """

    def __init__(self) -> None:
        self._index: dict[EvidenceKey, list[DecisionObservation]] = defaultdict(list)
        self.sources: dict[str, EvidenceSource] = {}

    def add_rows(
        self,
        label: str,
        rows: Iterable[Mapping[str, Any]],
        *,
        description: str = "",
        path: str | Path | None = None,
        sha256: str | None = None,
    ) -> EvidenceSource:
        if label in self.sources:
            raise ValueError(f"evidence source {label!r} is already loaded")
        observations = [DecisionObservation.from_trace(row, label) for row in rows]
        seen: set[EvidenceKey] = set()
        for item in observations:
            if item.key in seen:
                raise ValueError(
                    f"source {label} records {item.review_id}/{item.predicate_id} more than once"
                )
            seen.add(item.key)
        info = EvidenceSource(
            label=label,
            description=description,
            path=str(path) if path is not None else None,
            sha256=sha256,
            rows=len(observations),
        )
        for item in observations:
            self._index[item.key].append(item)
            info.models[item.model_identifier] = info.models.get(item.model_identifier, 0) + 1
            info.predicates[item.predicate_id] = info.predicates.get(item.predicate_id, 0) + 1
        info.run_ids = sorted({item.run_id for item in observations})
        self.sources[label] = info
        return info

    def load_jsonl(self, path: str | Path, label: str, *, description: str = "") -> EvidenceSource:
        return self.add_rows(
            label,
            read_jsonl(path),
            description=description,
            path=path,
            sha256=file_sha256(path),
        )

    def observations(
        self,
        review_id: str,
        predicate_id: str,
        model_identifier: str,
        question_hash: str,
        *,
        sources: Sequence[str] | None = None,
    ) -> list[DecisionObservation]:
        found = self._index.get((review_id, predicate_id, model_identifier, question_hash), [])
        if sources is None:
            return list(found)
        allowed = set(sources)
        return [item for item in found if item.source in allowed]

    def all_observations(self) -> Iterable[DecisionObservation]:
        for items in self._index.values():
            yield from items

    def review_ids(self, *, sources: Sequence[str] | None = None) -> set[str]:
        allowed = None if sources is None else set(sources)
        return {
            item.review_id
            for item in self.all_observations()
            if allowed is None or item.source in allowed
        }

    def repeated_decision_report(
        self,
        *,
        model_identifier: str,
        question_hashes: Mapping[str, str],
        threshold: float,
        sources: Sequence[str],
        review_ids: Iterable[str] | None = None,
        near_threshold_margin: float = 0.05,
    ) -> dict[str, Any]:
        """Compare repeated observations of the same pair across sources."""
        allowed_reviews = None if review_ids is None else set(review_ids)
        by_pair: dict[tuple[str, str], dict[str, DecisionObservation]] = defaultdict(dict)
        for item in self.all_observations():
            if item.source not in sources or item.model_identifier != model_identifier:
                continue
            if question_hashes.get(item.predicate_id) != item.question_hash:
                continue
            if allowed_reviews is not None and item.review_id not in allowed_reviews:
                continue
            by_pair[(item.review_id, item.predicate_id)][item.source] = item
        comparisons = []
        for first, second in itertools.combinations(sources, 2):
            shared = sorted(
                key for key, found in by_pair.items() if first in found and second in found
            )
            differences = [
                abs(by_pair[key][first].probability - by_pair[key][second].probability)
                for key in shared
            ]
            crossings = [
                {
                    "review_id": key[0],
                    "predicate_id": key[1],
                    f"{first}_probability": by_pair[key][first].probability,
                    f"{second}_probability": by_pair[key][second].probability,
                    f"{first}_passed": by_pair[key][first].passed_at(threshold),
                    f"{second}_passed": by_pair[key][second].passed_at(threshold),
                }
                for key in shared
                if by_pair[key][first].passed_at(threshold)
                != by_pair[key][second].passed_at(threshold)
            ]
            by_predicate: dict[str, int] = defaultdict(int)
            for key in shared:
                by_predicate[key[1]] += 1
            comparisons.append(
                {
                    "sources": [first, second],
                    "shared_pairs": len(shared),
                    "shared_pairs_by_predicate": dict(sorted(by_predicate.items())),
                    "threshold_crossings": len(crossings),
                    "crossing_rate": len(crossings) / len(shared) if shared else None,
                    "mean_absolute_probability_difference": (
                        mean(differences) if differences else None
                    ),
                    "median_absolute_probability_difference": (
                        median(differences) if differences else None
                    ),
                    "maximum_absolute_probability_difference": (
                        max(differences) if differences else None
                    ),
                    "identical_probability_pairs": sum(value == 0 for value in differences),
                    "input_token_mismatches": sum(
                        by_pair[key][first].input_tokens != by_pair[key][second].input_tokens
                        for key in shared
                    ),
                    "output_token_mismatches": sum(
                        by_pair[key][first].output_tokens != by_pair[key][second].output_tokens
                        for key in shared
                    ),
                    "crossing_details": crossings,
                }
            )
        near_threshold: dict[str, dict[str, int]] = {}
        for source in sources:
            counts: dict[str, int] = defaultdict(int)
            total = 0
            for found in by_pair.values():
                item = found.get(source)
                if item is None:
                    continue
                total += 1
                if abs(item.probability - threshold) <= near_threshold_margin:
                    counts[item.predicate_id] += 1
            near_threshold[source] = {
                "decisions": total,
                "within_margin": sum(counts.values()),
                **{f"within_margin_{key}": value for key, value in sorted(counts.items())},
            }
        return {
            "model_identifier": model_identifier,
            "threshold": threshold,
            "near_threshold_margin": near_threshold_margin,
            "sources": list(sources),
            "pairs_with_repeated_observations": sum(
                len(found) > 1 for found in by_pair.values()
            ),
            "source_comparisons": comparisons,
            "near_threshold_decisions": near_threshold,
        }
