from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class VisibleCandidate:
    key: str
    workload_id: str
    screen_probability: float
    followup_cost_usd: float


class AcquisitionPolicy(Protocol):
    name: str
    requires_screen: bool

    def order(self, candidates: tuple[VisibleCandidate, ...], seed: int) -> list[str]: ...


def _tie_break(candidates: tuple[VisibleCandidate, ...], seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    return {
        candidate.key: rng.random() for candidate in sorted(candidates, key=lambda row: row.key)
    }


@dataclass(frozen=True)
class RandomPolicy:
    name: str = "random"
    requires_screen: bool = False

    def order(self, candidates: tuple[VisibleCandidate, ...], seed: int) -> list[str]:
        keys = sorted(candidate.key for candidate in candidates)
        random.Random(seed).shuffle(keys)
        return keys


@dataclass(frozen=True)
class StratifiedRandomPolicy:
    name: str = "stratified_random"
    requires_screen: bool = False

    def order(self, candidates: tuple[VisibleCandidate, ...], seed: int) -> list[str]:
        by_workload: dict[str, list[str]] = {}
        for candidate in candidates:
            by_workload.setdefault(candidate.workload_id, []).append(candidate.key)
        rng = random.Random(seed)
        for keys in by_workload.values():
            keys.sort()
            rng.shuffle(keys)
        order: list[str] = []
        workloads = sorted(by_workload)
        while any(by_workload.values()):
            for workload in workloads:
                if by_workload[workload]:
                    order.append(by_workload[workload].pop())
        return order


@dataclass(frozen=True)
class UncertaintyPolicy:
    margin: float | None = None
    name: str = "uncertainty_continuous"
    requires_screen: bool = True

    def order(self, candidates: tuple[VisibleCandidate, ...], seed: int) -> list[str]:
        ties = _tie_break(candidates, seed)
        if self.margin is None:
            return [
                row.key
                for row in sorted(
                    candidates,
                    key=lambda row: (abs(row.screen_probability - 0.5), ties[row.key]),
                )
            ]
        inside = [row for row in candidates if abs(row.screen_probability - 0.5) <= self.margin]
        outside = [row for row in candidates if row not in inside]
        inside.sort(key=lambda row: (abs(row.screen_probability - 0.5), ties[row.key]))
        outside.sort(key=lambda row: ties[row.key])
        return [row.key for row in (*inside, *outside)]


@dataclass(frozen=True)
class PositivePolicy:
    name: str = "positive_oriented"
    requires_screen: bool = True

    def order(self, candidates: tuple[VisibleCandidate, ...], seed: int) -> list[str]:
        ties = _tie_break(candidates, seed)
        return [
            row.key
            for row in sorted(
                candidates,
                key=lambda row: (-row.screen_probability, ties[row.key]),
            )
        ]


@dataclass(frozen=True)
class HybridPolicy:
    uncertainty_weight: float = 0.5
    positive_weight: float = 0.5
    cost_aware: bool = True
    name: str = "hybrid_weighted"
    requires_screen: bool = True

    def order(self, candidates: tuple[VisibleCandidate, ...], seed: int) -> list[str]:
        ties = _tie_break(candidates, seed)

        def score(row: VisibleCandidate) -> float:
            uncertainty = 1.0 - min(1.0, abs(row.screen_probability - 0.5) / 0.5)
            value = (
                self.uncertainty_weight * uncertainty
                + self.positive_weight * row.screen_probability
            )
            return value / row.followup_cost_usd if self.cost_aware else value

        return [row.key for row in sorted(candidates, key=lambda row: (-score(row), ties[row.key]))]


@dataclass(frozen=True)
class QuotaHybridPolicy:
    name: str = "hybrid_quota"
    requires_screen: bool = True

    def order(self, candidates: tuple[VisibleCandidate, ...], seed: int) -> list[str]:
        uncertainty = UncertaintyPolicy().order(candidates, seed)
        positive = PositivePolicy().order(candidates, seed + 1)
        output: list[str] = []
        seen: set[str] = set()
        while len(output) < len(candidates):
            for source in (uncertainty, positive):
                while source and source[0] in seen:
                    source.pop(0)
                if source:
                    key = source.pop(0)
                    seen.add(key)
                    output.append(key)
        return output


def deployable_policies() -> tuple[AcquisitionPolicy, ...]:
    return (
        RandomPolicy(),
        StratifiedRandomPolicy(),
        UncertaintyPolicy(),
        UncertaintyPolicy(margin=0.15, name="uncertainty_margin_0.15"),
        PositivePolicy(),
        HybridPolicy(),
        QuotaHybridPolicy(),
    )


def ablation_policies() -> tuple[AcquisitionPolicy, ...]:
    return (
        UncertaintyPolicy(margin=0.05, name="uncertainty_margin_0.05"),
        UncertaintyPolicy(margin=0.10, name="uncertainty_margin_0.10"),
        UncertaintyPolicy(margin=0.20, name="uncertainty_margin_0.20"),
        HybridPolicy(uncertainty_weight=0.0, positive_weight=1.0, name="hybrid_no_uncertainty"),
        HybridPolicy(uncertainty_weight=1.0, positive_weight=0.0, name="hybrid_no_positive"),
        HybridPolicy(cost_aware=False, name="hybrid_no_cost"),
    )
