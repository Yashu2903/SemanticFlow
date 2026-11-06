from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from semanticflow.optimizer.evidence import DecisionEvidence, DecisionObservation
from semanticflow.optimizer.matrix import ProfileMatrix

REPLAY_SCHEMA_VERSION = "m3-counterfactual-replay-v2"

REPLAY_EVIDENCE: dict[str, Any] = {
    "evidence_kind": "offline_replay_estimate",
    "causal_measurement": False,
    "live_provider_calls": 0,
    "assumptions": [
        "A stored decision is what the model would return for that review and predicate at "
        "any time and in any evaluation order (stationarity and order independence).",
        "When repeated stored decisions disagree, the resolution policy chooses which one "
        "counts; another policy gives another estimate.",
        "Reviews lacking a needed decision are bounded, not observed; point estimates for "
        "them are imputation.",
    ],
    "determinacy_meaning": (
        "'exact' and 'fully determined' mean the stored decisions fix the replayed count "
        "without bounds or imputation under the stated policy; they do not make a replayed "
        "difference a measured or causal effect."
    ),
}

CellStatus = Literal[
    "observed",
    "observed_consistent_repeats",
    "conflict_resolved_by_priority",
    "conflict_unknown",
    "missing",
]


class ReplayTarget(BaseModel):
    """The query binding that observations must match exactly to be reused."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_identifier: str
    threshold: float = Field(ge=0.0, le=1.0)
    question_hashes: dict[str, str]

    @property
    def predicate_ids(self) -> list[str]:
        return list(self.question_hashes)


class ResolutionPolicy(BaseModel):
    """Which stored sources may supply a decision, and how disagreements are treated."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    source_priority: list[str] = Field(min_length=1)
    conflict_mode: Literal["prefer_priority", "treat_as_unknown"] = "prefer_priority"

    @model_validator(mode="after")
    def unique_sources(self) -> ResolutionPolicy:
        if len(self.source_priority) != len(set(self.source_priority)):
            raise ValueError("resolution sources must be unique")
        return self


class ReplayPricing(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_per_million_usd: float | None = None
    output_per_million_usd: float | None = None

    def cost(self, input_tokens: float, output_tokens: float) -> float | None:
        if self.input_per_million_usd is None or self.output_per_million_usd is None:
            return None
        return (
            input_tokens * self.input_per_million_usd
            + output_tokens * self.output_per_million_usd
        ) / 1_000_000


@dataclass(frozen=True, slots=True)
class ResolvedCell:
    predicate_id: str
    outcome: bool | None
    status: CellStatus
    observation: DecisionObservation | None
    observation_count: int


def resolve_cell(
    evidence: DecisionEvidence,
    review_id: str,
    predicate_id: str,
    target: ReplayTarget,
    policy: ResolutionPolicy,
) -> ResolvedCell:
    found = evidence.observations(
        review_id,
        predicate_id,
        target.model_identifier,
        target.question_hashes[predicate_id],
        sources=policy.source_priority,
    )
    if not found:
        return ResolvedCell(predicate_id, None, "missing", None, 0)
    rank = {source: index for index, source in enumerate(policy.source_priority)}
    ordered = sorted(found, key=lambda item: (rank[item.source], item.timestamp))
    outcomes = {item.passed_at(target.threshold) for item in ordered}
    chosen = ordered[0]
    if len(outcomes) == 1:
        status: CellStatus = "observed" if len(ordered) == 1 else "observed_consistent_repeats"
        return ResolvedCell(predicate_id, chosen.passed_at(target.threshold), status, chosen,
                            len(ordered))
    if policy.conflict_mode == "prefer_priority":
        return ResolvedCell(
            predicate_id,
            chosen.passed_at(target.threshold),
            "conflict_resolved_by_priority",
            chosen,
            len(ordered),
        )
    return ResolvedCell(predicate_id, None, "conflict_unknown", chosen, len(ordered))


class PatternPrior:
    """Dirichlet-smoothed outcome-pattern distribution from a complete profiling matrix.

    Used only for explicitly labelled imputation estimates, never for exact results.
    """

    def __init__(
        self, matrix: ProfileMatrix, predicate_ids: Sequence[str], *, pseudo_count: float = 0.5
    ) -> None:
        if not matrix.complete:
            raise ValueError("pattern prior requires a complete profiling matrix")
        if pseudo_count < 0:
            raise ValueError("pattern pseudo-count cannot be negative")
        self.predicate_ids = list(predicate_ids)
        positions = [matrix.index(predicate_id) for predicate_id in self.predicate_ids]
        counts: dict[tuple[bool, ...], int] = {}
        for pattern, count in matrix.pattern_counts().items():
            projected = tuple(pattern[index] for index in positions)
            counts[projected] = counts.get(projected, 0) + count
        cells = 2 ** len(self.predicate_ids)
        total = matrix.size + pseudo_count * cells
        if total == 0:
            raise ValueError("pattern prior has no mass")
        self.pseudo_count = pseudo_count
        self.sample_size = matrix.size
        self.probabilities = {
            pattern: (counts.get(pattern, 0) + pseudo_count) / total
            for pattern in itertools.product((False, True), repeat=len(self.predicate_ids))
        }

    def weight(self, outcomes: Mapping[str, bool]) -> float:
        return self.probabilities[tuple(outcomes[item] for item in self.predicate_ids)]


@dataclass(slots=True)
class _Path:
    reached: tuple[bool, ...]
    passed: tuple[bool, ...]
    evaluations: int
    matched: bool
    input_tokens: float
    output_tokens: float
    weight: float


def _path(
    order: Sequence[str],
    outcomes: Mapping[str, bool],
    usage: Mapping[str, tuple[float, float]],
    weight: float,
) -> _Path:
    reached: list[bool] = []
    passed: list[bool] = []
    alive = True
    input_tokens = output_tokens = 0.0
    for predicate_id in order:
        reached.append(alive)
        result = alive and outcomes[predicate_id]
        passed.append(result)
        if alive:
            input_tokens += usage[predicate_id][0]
            output_tokens += usage[predicate_id][1]
        alive = result
    return _Path(
        reached=tuple(reached),
        passed=tuple(passed),
        evaluations=sum(reached),
        matched=alive,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        weight=weight,
    )


@dataclass(slots=True)
class _ReviewContext:
    review_id: str
    cells: dict[str, ResolvedCell]
    usage: dict[str, tuple[float, float]]
    usage_estimated: set[str]
    assignments: list[tuple[dict[str, bool], float]]


def _usage_fill(
    evidence: DecisionEvidence, target: ReplayTarget, policy: ResolutionPolicy
) -> dict[str, tuple[float, float] | None]:
    """Per-predicate mean usage over the policy's sources, for pairs never observed."""
    totals: dict[str, list[float]] = {pid: [0.0, 0.0, 0.0] for pid in target.predicate_ids}
    for item in evidence.all_observations():
        if item.source not in policy.source_priority:
            continue
        if item.model_identifier != target.model_identifier:
            continue
        if target.question_hashes.get(item.predicate_id) != item.question_hash:
            continue
        if item.input_tokens is None or item.output_tokens is None:
            continue
        bucket = totals[item.predicate_id]
        bucket[0] += item.input_tokens
        bucket[1] += item.output_tokens
        bucket[2] += 1
    return {
        pid: (values[0] / values[2], values[1] / values[2]) if values[2] else None
        for pid, values in totals.items()
    }


def _contexts(
    evidence: DecisionEvidence,
    target: ReplayTarget,
    review_ids: Sequence[str],
    policy: ResolutionPolicy,
    prior: PatternPrior | None,
) -> tuple[list[_ReviewContext], dict[str, tuple[float, float] | None]]:
    fill = _usage_fill(evidence, target, policy)
    contexts: list[_ReviewContext] = []
    for review_id in review_ids:
        cells = {
            pid: resolve_cell(evidence, review_id, pid, target, policy)
            for pid in target.predicate_ids
        }
        usage: dict[str, tuple[float, float]] = {}
        estimated: set[str] = set()
        for pid, cell in cells.items():
            obs = cell.observation
            if obs is not None and obs.input_tokens is not None and obs.output_tokens is not None:
                usage[pid] = (float(obs.input_tokens), float(obs.output_tokens))
            else:
                usage[pid] = fill[pid] if fill[pid] is not None else (0.0, 0.0)
                estimated.add(pid)
        unknown = [pid for pid, cell in cells.items() if cell.outcome is None]
        known = {pid: bool(cell.outcome) for pid, cell in cells.items() if cell.outcome is not None}
        assignments: list[tuple[dict[str, bool], float]] = []
        for values in itertools.product((False, True), repeat=len(unknown)):
            outcomes = {**known, **dict(zip(unknown, values, strict=True))}
            weight = prior.weight(outcomes) if prior is not None else 1.0
            assignments.append((outcomes, weight))
        if prior is not None:
            total = sum(weight for _, weight in assignments)
            assignments = [(outcomes, weight / total) for outcomes, weight in assignments]
        contexts.append(_ReviewContext(review_id, cells, usage, estimated, assignments))
    return contexts, fill


def _validate(order: Sequence[str], target: ReplayTarget, review_ids: Sequence[str]) -> None:
    if not order:
        raise ValueError("replay order cannot be empty")
    if len(order) != len(set(order)):
        raise ValueError("replay order contains duplicate predicates")
    unknown = set(order) - set(target.predicate_ids)
    if unknown:
        raise ValueError(f"replay order references unbound predicates: {sorted(unknown)}")
    if len(review_ids) != len(set(review_ids)):
        raise ValueError("replay review IDs must be unique")
    if not review_ids:
        raise ValueError("replay requires at least one review")


def _bounds(lower: float, upper: float) -> dict[str, float | bool]:
    return {"lower": lower, "upper": upper, "exact": lower == upper}


def _replay_contexts(
    contexts: list[_ReviewContext],
    order: Sequence[str],
    pricing: ReplayPricing,
    *,
    estimate_label: str | None,
) -> dict[str, Any]:
    width = len(order)
    stage_in = [[0, 0] for _ in range(width)]
    stage_pass = [[0, 0] for _ in range(width)]
    stage_reject = [[0, 0] for _ in range(width)]
    expected_in = [0.0] * width
    expected_pass = [0.0] * width
    evaluations = [0, 0]
    expected_evaluations = 0.0
    tokens_in = [0.0, 0.0]
    tokens_out = [0.0, 0.0]
    expected_tokens = [0.0, 0.0]
    estimated_usage_pairs = 0
    resolved = 0
    definite: list[str] = []
    possible: list[str] = []
    expected_matches = 0.0
    missing_on_paths: list[dict[str, str]] = []
    conflicts_on_paths = 0
    attempts = 0
    latency = 0.0
    status_counts: dict[str, int] = {}
    for context in contexts:
        paths = [
            _path(order, outcomes, context.usage, weight)
            for outcomes, weight in context.assignments
        ]
        signatures = {(path.reached, path.passed) for path in paths}
        exact = len(signatures) == 1
        resolved += exact
        for stage in range(width):
            reached = [path.reached[stage] for path in paths]
            passed = [path.passed[stage] for path in paths]
            rejected = [r and not p for r, p in zip(reached, passed, strict=True)]
            stage_in[stage][0] += all(reached)
            stage_in[stage][1] += any(reached)
            stage_pass[stage][0] += all(passed)
            stage_pass[stage][1] += any(passed)
            stage_reject[stage][0] += all(rejected)
            stage_reject[stage][1] += any(rejected)
            expected_in[stage] += sum(path.weight * r for path, r in zip(paths, reached,
                                                                         strict=True))
            expected_pass[stage] += sum(path.weight * p for path, p in zip(paths, passed,
                                                                           strict=True))
        evaluations[0] += min(path.evaluations for path in paths)
        evaluations[1] += max(path.evaluations for path in paths)
        expected_evaluations += sum(path.weight * path.evaluations for path in paths)
        tokens_in[0] += min(path.input_tokens for path in paths)
        tokens_in[1] += max(path.input_tokens for path in paths)
        tokens_out[0] += min(path.output_tokens for path in paths)
        tokens_out[1] += max(path.output_tokens for path in paths)
        expected_tokens[0] += sum(path.weight * path.input_tokens for path in paths)
        expected_tokens[1] += sum(path.weight * path.output_tokens for path in paths)
        if all(path.matched for path in paths):
            definite.append(context.review_id)
        elif any(path.matched for path in paths):
            possible.append(context.review_id)
        expected_matches += sum(path.weight * path.matched for path in paths)
        for stage, predicate_id in enumerate(order):
            if not any(path.reached[stage] for path in paths):
                continue
            cell = context.cells[predicate_id]
            status_counts[cell.status] = status_counts.get(cell.status, 0) + 1
            if predicate_id in context.usage_estimated:
                estimated_usage_pairs += 1
            if cell.outcome is None:
                missing_on_paths.append(
                    {
                        "review_id": context.review_id,
                        "predicate_id": predicate_id,
                        "status": cell.status,
                    }
                )
            if cell.status == "conflict_resolved_by_priority":
                conflicts_on_paths += 1
            if all(path.reached[stage] for path in paths) and cell.observation is not None:
                attempts += cell.observation.request_attempts
                latency += cell.observation.latency_seconds
    review_count = len(contexts)
    observability = (
        "fully_observable"
        if resolved == review_count
        else "unobservable"
        if resolved == 0
        else "partially_observable"
    )
    cost_lower = pricing.cost(tokens_in[0], tokens_out[0])
    cost_upper = pricing.cost(tokens_in[1], tokens_out[1])
    tokens_exact = estimated_usage_pairs == 0 and tokens_in[0] == tokens_in[1]
    result: dict[str, Any] = {
        "predicate_order": list(order),
        "review_count": review_count,
        "observability": observability,
        "resolved_reviews": resolved,
        "unresolved_reviews": review_count - resolved,
        "logical_evaluations": _bounds(*evaluations),
        "stages": [
            {
                "stage": stage + 1,
                "predicate_id": predicate_id,
                "input_records": _bounds(*stage_in[stage]),
                "passed_records": _bounds(*stage_pass[stage]),
                "rejected_records": _bounds(*stage_reject[stage]),
            }
            for stage, predicate_id in enumerate(order)
        ],
        "final_matches": {
            "definite_count": len(definite),
            "possible_additional_count": len(possible),
            "bounds": _bounds(len(definite), len(definite) + len(possible)),
            "definite_review_ids": sorted(definite),
            "undetermined_review_ids": sorted(possible),
        },
        "usage": {
            "input_tokens": _bounds(*tokens_in),
            "output_tokens": _bounds(*tokens_out),
            "estimated_cost_usd": (
                {"lower": cost_lower, "upper": cost_upper, "exact": cost_lower == cost_upper}
                if cost_lower is not None and cost_upper is not None
                else None
            ),
            "exact_from_recorded_usage": tokens_exact,
            "pairs_with_estimated_usage": estimated_usage_pairs,
            "usage_estimate_method": (
                None if estimated_usage_pairs == 0 else "per_predicate_mean_of_recorded_usage"
            ),
        },
        "decision_status_on_possible_paths": dict(sorted(status_counts.items())),
        "missing_decisions_on_possible_paths": len(missing_on_paths),
        "missing_decision_examples": missing_on_paths[:20],
        "priority_resolved_conflicts_on_paths": conflicts_on_paths,
        "physical_metrics": {
            "status": "historical_observation_metrics_only",
            "reused_observation_request_attempts": attempts,
            "reused_observation_summed_latency_seconds": latency,
            "counterfactual_http_attempts": None,
            "counterfactual_retries": None,
            "counterfactual_wall_time_seconds": None,
            "reason": (
                "Retries, HTTP attempts, and latency of an unexecuted order are not "
                "determined by stored decisions; only logical evaluations are replayed."
            ),
        },
    }
    if estimate_label is not None:
        result["estimate"] = {
            "estimate_kind": estimate_label,
            "is_exact": False,
            "logical_evaluations": expected_evaluations,
            "stage_input_records": expected_in,
            "stage_passed_records": expected_pass,
            "final_matches": expected_matches,
            "input_tokens": expected_tokens[0],
            "output_tokens": expected_tokens[1],
            "estimated_cost_usd": pricing.cost(expected_tokens[0], expected_tokens[1]),
            "assumption": (
                "Unknown outcomes follow the smoothed profiling outcome-pattern distribution "
                "conditioned on this review's recorded outcomes; reviews are assumed "
                "exchangeable with the profiling sample."
            ),
        }
    return result


def replay_order(
    evidence: DecisionEvidence,
    target: ReplayTarget,
    review_ids: Sequence[str],
    order: Sequence[str],
    policy: ResolutionPolicy,
    *,
    pricing: ReplayPricing | None = None,
    prior: PatternPrior | None = None,
) -> dict[str, Any]:
    """Simulate sequential early rejection over stored decisions for one order."""
    _validate(order, target, review_ids)
    contexts, _ = _contexts(evidence, target, review_ids, policy, prior)
    result = _replay_contexts(
        contexts,
        order,
        pricing or ReplayPricing(),
        estimate_label="profile_pattern_conditional_imputation" if prior else None,
    )
    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "evidence": REPLAY_EVIDENCE,
        "policy": policy.model_dump(mode="json"),
        "target": target.model_dump(mode="json"),
        **result,
    }


def compare_orders(
    evidence: DecisionEvidence,
    target: ReplayTarget,
    review_ids: Sequence[str],
    first: Sequence[str],
    second: Sequence[str],
    policy: ResolutionPolicy,
    *,
    pricing: ReplayPricing | None = None,
    prior: PatternPrior | None = None,
) -> dict[str, Any]:
    """Replay two orders over the same resolved decisions; bound their difference jointly."""
    _validate(first, target, review_ids)
    _validate(second, target, review_ids)
    pricing = pricing or ReplayPricing()
    contexts, _ = _contexts(evidence, target, review_ids, policy, prior)
    label = "profile_pattern_conditional_imputation" if prior else None
    first_result = _replay_contexts(contexts, first, pricing, estimate_label=label)
    second_result = _replay_contexts(contexts, second, pricing, estimate_label=label)
    priced = pricing.cost(0.0, 0.0) is not None
    difference = [0, 0]
    input_difference = [0.0, 0.0]
    output_difference = [0.0, 0.0]
    cost_difference = [0.0, 0.0]
    expected = 0.0
    for context in contexts:
        deltas: list[int] = []
        input_deltas: list[float] = []
        output_deltas: list[float] = []
        cost_deltas: list[float] = []
        for outcomes, weight in context.assignments:
            a = _path(first, outcomes, context.usage, weight)
            b = _path(second, outcomes, context.usage, weight)
            deltas.append(a.evaluations - b.evaluations)
            input_deltas.append(a.input_tokens - b.input_tokens)
            output_deltas.append(a.output_tokens - b.output_tokens)
            if priced:
                cost_deltas.append(
                    pricing.cost(a.input_tokens, a.output_tokens)  # type: ignore[operator]
                    - pricing.cost(b.input_tokens, b.output_tokens)
                )
            expected += weight * (a.evaluations - b.evaluations)
        difference[0] += min(deltas)
        difference[1] += max(deltas)
        input_difference[0] += min(input_deltas)
        input_difference[1] += max(input_deltas)
        output_difference[0] += min(output_deltas)
        output_difference[1] += max(output_deltas)
        if priced:
            cost_difference[0] += min(cost_deltas)
            cost_difference[1] += max(cost_deltas)
    output: dict[str, Any] = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "evidence": REPLAY_EVIDENCE,
        "policy": policy.model_dump(mode="json"),
        "target": target.model_dump(mode="json"),
        "first": first_result,
        "second": second_result,
        "evaluation_difference_first_minus_second": _bounds(*difference),
        "input_token_difference_first_minus_second": _bounds(*input_difference),
        "output_token_difference_first_minus_second": _bounds(*output_difference),
        "estimated_cost_difference_usd": _bounds(*cost_difference) if priced else None,
        "difference_fully_determined_by_stored_decisions": difference[0] == difference[1],
    }
    if prior is not None:
        output["expected_evaluation_difference_estimate"] = expected
    return output


def repeated_variability_effect(
    evidence: DecisionEvidence,
    target: ReplayTarget,
    review_ids: Sequence[str],
    order: Sequence[str],
    sources: Sequence[str],
) -> dict[str, Any]:
    """Replay one order separately with each single source to expose response variability."""
    results = {}
    for source in sources:
        policy = ResolutionPolicy(name=f"only_{source}", source_priority=[source])
        replay = replay_order(evidence, target, review_ids, order, policy)
        results[source] = {
            "observability": replay["observability"],
            "logical_evaluations": replay["logical_evaluations"],
            "final_matches": replay["final_matches"]["bounds"],
        }
    return {"predicate_order": list(order), "by_source": results}
