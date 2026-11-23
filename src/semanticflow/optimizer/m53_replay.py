from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from semanticflow.optimizer.m53_evidence import EvidenceRecord, Usage
from semanticflow.optimizer.m53_policies import AcquisitionPolicy, VisibleCandidate


@dataclass(frozen=True)
class ReplaySnapshot:
    setting: str
    policy: str
    pool: str
    seed: int
    budget_kind: str
    budget_value: float
    requests: int
    attempts: int
    input_tokens: int
    output_tokens: int
    named_evaluations: int
    cost_usd: float
    screening_requests: int
    screening_input_tokens: int
    screening_output_tokens: int
    screening_cost_usd: float
    selected_records: int
    predicate_disagreements: int
    query_disagreements: int
    false_negatives: int
    false_positives: int
    positives: int
    known_predicate_disagreements: int
    known_query_disagreements: int
    known_false_negatives: int
    known_positives: int
    duplicate_acquisitions: int = 0

    def row(self) -> dict[str, object]:
        row = asdict(self)
        row["predicate_disagreement_fraction"] = _fraction(
            self.predicate_disagreements, self.known_predicate_disagreements
        )
        row["query_disagreement_fraction"] = _fraction(
            self.query_disagreements, self.known_query_disagreements
        )
        row["false_negative_fraction"] = _fraction(self.false_negatives, self.known_false_negatives)
        row["positive_fraction"] = _fraction(self.positives, self.known_positives)
        row["cost_per_predicate_disagreement"] = _cost(self.cost_usd, self.predicate_disagreements)
        row["cost_per_positive"] = _cost(self.cost_usd, self.positives)
        return row


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _cost(cost: float, count: int) -> float | None:
    return cost / count if count else None


def _visible(records: tuple[EvidenceRecord, ...]) -> tuple[VisibleCandidate, ...]:
    return tuple(
        VisibleCandidate(
            key=record.record_id,
            workload_id=record.workload_id,
            screen_probability=record.screen_probability,
            followup_cost_usd=record.followup_usage.cost_usd,
        )
        for record in records
    )


def _snapshot(
    *,
    setting: str,
    policy: str,
    pool: str,
    seed: int,
    budget_kind: str,
    budget_value: float,
    usage: Usage,
    screening_usage: Usage,
    selected: list[EvidenceRecord],
    records: tuple[EvidenceRecord, ...],
) -> ReplaySnapshot:
    return ReplaySnapshot(
        setting=setting,
        policy=policy,
        pool=pool,
        seed=seed,
        budget_kind=budget_kind,
        budget_value=budget_value,
        requests=usage.requests,
        attempts=usage.attempts,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        named_evaluations=usage.named_evaluations,
        cost_usd=usage.cost_usd,
        screening_requests=screening_usage.requests,
        screening_input_tokens=screening_usage.input_tokens,
        screening_output_tokens=screening_usage.output_tokens,
        screening_cost_usd=screening_usage.cost_usd,
        selected_records=len(selected),
        predicate_disagreements=sum(row.predicate_disagreement_count for row in selected),
        query_disagreements=sum(row.query_disagreement for row in selected),
        false_negatives=sum(row.false_negative for row in selected),
        false_positives=sum(row.false_positive for row in selected),
        positives=sum(row.separate_query_positive for row in selected),
        known_predicate_disagreements=sum(row.predicate_disagreement_count for row in records),
        known_query_disagreements=sum(row.query_disagreement for row in records),
        known_false_negatives=sum(row.false_negative for row in records),
        known_positives=sum(row.separate_query_positive for row in records),
    )


def replay_conditional(
    records: tuple[EvidenceRecord, ...],
    policy: AcquisitionPolicy,
    seed: int,
    budgets: tuple[int, ...],
    pool: str,
) -> list[ReplaySnapshot]:
    by_key = {record.record_id: record for record in records}
    order = policy.order(_visible(records), seed)
    if len(order) != len(set(order)) or set(order) != set(by_key):
        raise ValueError(f"policy {policy.name} returned invalid or duplicate actions")
    output: list[ReplaySnapshot] = []
    for budget in budgets:
        usage = Usage()
        selected: list[EvidenceRecord] = []
        for key in order:
            record = by_key[key]
            if usage.requests + record.followup_usage.requests > budget:
                continue
            usage += record.followup_usage
            selected.append(record)
        output.append(
            _snapshot(
                setting="conditional_followup",
                policy=policy.name,
                pool=pool,
                seed=seed,
                budget_kind="requests",
                budget_value=float(budget),
                usage=usage,
                screening_usage=Usage(),
                selected=selected,
                records=records,
            )
        )
    return output


def replay_end_to_end(
    records: tuple[EvidenceRecord, ...],
    policy: AcquisitionPolicy,
    seed: int,
    budgets: tuple[int, ...],
    pool: str,
    screening_window: int = 5,
) -> list[ReplaySnapshot]:
    by_key = {record.record_id: record for record in records}
    maximum_budget = max(budgets)
    usage = Usage()
    selected: list[EvidenceRecord] = []
    states: list[tuple[int, Usage, Usage, list[EvidenceRecord]]] = []
    screening_usage = Usage()
    if not policy.requires_screen:
        for key in policy.order(_visible(records), seed):
            record = by_key[key]
            if usage.requests + record.full_usage.requests > maximum_budget:
                continue
            usage += record.full_usage
            selected.append(record)
            states.append((usage.requests, usage, screening_usage, list(selected)))
    else:
        unseen = sorted(by_key)
        random.Random(seed).shuffle(unseen)
        visible: dict[str, EvidenceRecord] = {}
        while unseen or visible:
            for _ in range(screening_window):
                if not unseen:
                    break
                record = by_key[unseen.pop()]
                if usage.requests + record.screen_usage.requests > maximum_budget:
                    unseen.clear()
                    break
                usage += record.screen_usage
                screening_usage += record.screen_usage
                visible[record.record_id] = record
                states.append((usage.requests, usage, screening_usage, list(selected)))
            if not visible:
                break
            candidates = _visible(tuple(visible.values()))
            key = policy.order(candidates, seed + len(selected))[0]
            record = visible.pop(key)
            if usage.requests + record.followup_usage.requests > maximum_budget:
                break
            usage += record.followup_usage
            selected.append(record)
            states.append((usage.requests, usage, screening_usage, list(selected)))

    output: list[ReplaySnapshot] = []
    for budget in budgets:
        eligible = [state for state in states if state[0] <= budget]
        if eligible:
            _, state_usage, state_screening, state_selected = eligible[-1]
        else:
            state_usage, state_screening, state_selected = Usage(), Usage(), []
        output.append(
            _snapshot(
                setting="end_to_end",
                policy=policy.name,
                pool=pool,
                seed=seed,
                budget_kind="requests",
                budget_value=float(budget),
                usage=state_usage,
                screening_usage=state_screening,
                selected=state_selected,
                records=records,
            )
        )
    return output


def replay_conditional_dollar(
    records: tuple[EvidenceRecord, ...],
    policy: AcquisitionPolicy,
    seed: int,
    budgets: tuple[float, ...],
    pool: str,
) -> list[ReplaySnapshot]:
    by_key = {record.record_id: record for record in records}
    order = policy.order(_visible(records), seed)
    output: list[ReplaySnapshot] = []
    for budget in budgets:
        usage = Usage()
        selected: list[EvidenceRecord] = []
        for key in order:
            record = by_key[key]
            if usage.cost_usd + record.followup_usage.cost_usd > budget + 1e-15:
                continue
            usage += record.followup_usage
            selected.append(record)
        output.append(
            _snapshot(
                setting="conditional_followup",
                policy=policy.name,
                pool=pool,
                seed=seed,
                budget_kind="dollars",
                budget_value=budget,
                usage=usage,
                screening_usage=Usage(),
                selected=selected,
                records=records,
            )
        )
    return output


def replay_end_to_end_dollar(
    records: tuple[EvidenceRecord, ...],
    policy: AcquisitionPolicy,
    seed: int,
    budgets: tuple[float, ...],
    pool: str,
    screening_window: int = 5,
) -> list[ReplaySnapshot]:
    by_key = {record.record_id: record for record in records}
    output: list[ReplaySnapshot] = []
    for budget in budgets:
        usage = Usage()
        selected: list[EvidenceRecord] = []
        screening_usage = Usage()
        if not policy.requires_screen:
            for key in policy.order(_visible(records), seed):
                record = by_key[key]
                if usage.cost_usd + record.full_usage.cost_usd > budget + 1e-15:
                    continue
                usage += record.full_usage
                selected.append(record)
        else:
            unseen = sorted(by_key)
            random.Random(seed).shuffle(unseen)
            visible: dict[str, EvidenceRecord] = {}
            while unseen or visible:
                for _ in range(screening_window):
                    if not unseen:
                        break
                    record = by_key[unseen.pop()]
                    if usage.cost_usd + record.screen_usage.cost_usd > budget + 1e-15:
                        unseen.clear()
                        break
                    usage += record.screen_usage
                    screening_usage += record.screen_usage
                    visible[record.record_id] = record
                if not visible:
                    break
                candidates = _visible(tuple(visible.values()))
                key = policy.order(candidates, seed + len(selected))[0]
                record = visible.pop(key)
                if usage.cost_usd + record.followup_usage.cost_usd > budget + 1e-15:
                    break
                usage += record.followup_usage
                selected.append(record)
        output.append(
            _snapshot(
                setting="end_to_end",
                policy=policy.name,
                pool=pool,
                seed=seed,
                budget_kind="dollars",
                budget_value=budget,
                usage=usage,
                screening_usage=screening_usage,
                selected=selected,
                records=records,
            )
        )
    return output
