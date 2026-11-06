import itertools

import pytest

from semanticflow.optimizer.cardinality import exact_matrix_cardinalities
from semanticflow.optimizer.evidence import DecisionEvidence
from semanticflow.optimizer.replay import (
    PatternPrior,
    ReplayPricing,
    ReplayTarget,
    ResolutionPolicy,
    compare_orders,
    replay_order,
    resolve_cell,
)
from tests.m3_support import MODEL, hashes, matrix_from_patterns, trace

IDS = ["A", "B", "C"]
TARGET = ReplayTarget(model_identifier=MODEL, threshold=0.5, question_hashes=hashes(IDS))
ONLY_RUN = ResolutionPolicy(name="run", source_priority=["run"])


def evidence_from(outcomes: dict[str, dict[str, bool]], label: str = "run") -> DecisionEvidence:
    evidence = DecisionEvidence()
    evidence.add_rows(
        label,
        [
            trace(review_id, predicate_id, passed)
            for review_id, row in outcomes.items()
            for predicate_id, passed in row.items()
        ],
    )
    return evidence


def complete_outcomes() -> dict[str, dict[str, bool]]:
    patterns = list(itertools.product((False, True), repeat=3)) * 2
    return {
        f"r{index:02}": dict(zip(IDS, pattern, strict=True))
        for index, pattern in enumerate(patterns)
    }


def test_complete_matrix_replay_is_exact_and_matches_matrix_replay_for_every_order() -> None:
    outcomes = complete_outcomes()
    evidence = evidence_from(outcomes)
    matrix = matrix_from_patterns(
        IDS, [tuple(row[item] for item in IDS) for row in outcomes.values()]
    )
    for order in itertools.permutations(IDS):
        report = replay_order(evidence, TARGET, list(outcomes), order, ONLY_RUN)
        assert report["observability"] == "fully_observable"
        stage_inputs = [stage["input_records"]["lower"] for stage in report["stages"]]
        assert all(stage["input_records"]["exact"] for stage in report["stages"])
        assert stage_inputs == exact_matrix_cardinalities(matrix, order)
        assert report["logical_evaluations"] == {
            "lower": sum(stage_inputs),
            "upper": sum(stage_inputs),
            "exact": True,
        }
        assert report["final_matches"]["bounds"]["lower"] == 2


def test_early_rejection_makes_missing_later_decisions_irrelevant() -> None:
    evidence = evidence_from({"r1": {"A": False}, "r2": {"A": True, "B": False}})
    report = replay_order(evidence, TARGET, ["r1", "r2"], ["A", "B", "C"], ONLY_RUN)
    assert report["observability"] == "fully_observable"
    assert report["logical_evaluations"]["lower"] == report["logical_evaluations"]["upper"] == 3
    assert report["missing_decisions_on_possible_paths"] == 0
    assert [stage["rejected_records"]["lower"] for stage in report["stages"]] == [1, 1, 0]


def test_missing_decisions_produce_bounds_and_partial_observability() -> None:
    evidence = evidence_from(
        {"r1": {"A": True, "B": True, "C": True}, "r2": {"A": True, "C": False}}
    )
    report = replay_order(evidence, TARGET, ["r1", "r2"], ["A", "B", "C"], ONLY_RUN)
    assert report["observability"] == "partially_observable"
    assert report["resolved_reviews"] == 1
    # r2 reaches B with B unknown: 2 evaluations if B fails, 3 if B passes.
    assert report["logical_evaluations"] == {"lower": 5, "upper": 6, "exact": False}
    assert report["final_matches"]["bounds"] == {"lower": 1, "upper": 1, "exact": True}
    assert report["missing_decision_examples"][0] == {
        "review_id": "r2",
        "predicate_id": "B",
        "status": "missing",
    }
    assert "estimate" not in report


def test_incomplete_experiment_without_any_decision_is_unobservable() -> None:
    evidence = evidence_from({"r1": {"A": True, "B": True, "C": True}})
    report = replay_order(evidence, TARGET, ["r_absent"], ["A", "B", "C"], ONLY_RUN)
    assert report["observability"] == "unobservable"
    assert report["logical_evaluations"] == {"lower": 1, "upper": 3, "exact": False}
    assert report["final_matches"]["bounds"] == {"lower": 0, "upper": 1, "exact": False}


def test_all_predicates_pass_gives_identical_outcomes_for_every_order() -> None:
    outcomes = {f"r{i}": dict.fromkeys(IDS, True) for i in range(4)}
    evidence = evidence_from(outcomes)
    reports = [
        replay_order(evidence, TARGET, list(outcomes), order, ONLY_RUN)
        for order in itertools.permutations(IDS)
    ]
    assert {report["logical_evaluations"]["lower"] for report in reports} == {12}
    assert {tuple(report["final_matches"]["definite_review_ids"]) for report in reports} == {
        tuple(sorted(outcomes))
    }
    comparison = compare_orders(
        evidence, TARGET, list(outcomes), ["A", "B", "C"], ["C", "B", "A"], ONLY_RUN
    )
    assert comparison["evaluation_difference_first_minus_second"]["exact"]
    assert comparison["evaluation_difference_first_minus_second"]["lower"] == 0


def test_zero_matches_when_every_review_fails_some_predicate() -> None:
    evidence = evidence_from(
        {
            "r1": {"A": False, "B": True, "C": True},
            "r2": {"A": True, "B": False, "C": True},
            "r3": {"A": True, "B": True, "C": False},
        }
    )
    for order in itertools.permutations(IDS):
        report = replay_order(evidence, TARGET, ["r1", "r2", "r3"], order, ONLY_RUN)
        assert report["final_matches"]["bounds"] == {"lower": 0, "upper": 0, "exact": True}


def test_compare_orders_bounds_difference_jointly_and_prices_both_token_kinds() -> None:
    rows = [
        trace("r1", "A", False, input_tokens=100, output_tokens=10),
        trace("r1", "B", True, input_tokens=20, output_tokens=2),
        trace("r1", "C", True, input_tokens=30, output_tokens=3),
    ]
    evidence = DecisionEvidence()
    evidence.add_rows("run", rows)
    pricing = ReplayPricing(input_per_million_usd=1.0, output_per_million_usd=10.0)
    comparison = compare_orders(
        evidence, TARGET, ["r1"], ["A", "B", "C"], ["B", "C", "A"], ONLY_RUN, pricing=pricing
    )
    assert comparison["evaluation_difference_first_minus_second"]["lower"] == 1 - 3
    assert comparison["input_token_difference_first_minus_second"]["lower"] == 100 - 150
    assert comparison["output_token_difference_first_minus_second"]["lower"] == 10 - 15
    cost = comparison["estimated_cost_difference_usd"]
    assert cost["exact"]
    assert cost["lower"] == pytest.approx((100 + 100) / 1e6 - (150 + 150) / 1e6)


def test_fully_determined_replays_are_still_labelled_estimates_not_causal_measurements() -> None:
    outcomes = complete_outcomes()
    evidence = evidence_from(outcomes)
    single = replay_order(evidence, TARGET, list(outcomes), IDS, ONLY_RUN)
    comparison = compare_orders(
        evidence, TARGET, list(outcomes), IDS, ["C", "B", "A"], ONLY_RUN
    )
    assert comparison["difference_fully_determined_by_stored_decisions"] is True
    assert "difference_is_exact" not in comparison
    for report in (single, comparison):
        assert report["evidence"]["evidence_kind"] == "offline_replay_estimate"
        assert report["evidence"]["causal_measurement"] is False
        assert report["evidence"]["live_provider_calls"] == 0
        assert any("order independence" in item for item in report["evidence"]["assumptions"])


def test_repeated_variability_is_visible_and_resolution_policy_is_explicit() -> None:
    evidence = DecisionEvidence()
    evidence.add_rows("old", [trace("r1", "A", True, probability=0.52)])
    evidence.add_rows("new", [trace("r1", "A", False, probability=0.47)])
    prefer_old = ResolutionPolicy(name="old_first", source_priority=["old", "new"])
    prefer_new = ResolutionPolicy(name="new_first", source_priority=["new", "old"])
    unknown = ResolutionPolicy(
        name="unknown", source_priority=["old", "new"], conflict_mode="treat_as_unknown"
    )
    assert resolve_cell(evidence, "r1", "A", TARGET, prefer_old).outcome is True
    assert resolve_cell(evidence, "r1", "A", TARGET, prefer_new).outcome is False
    cell = resolve_cell(evidence, "r1", "A", TARGET, unknown)
    assert (cell.outcome, cell.status, cell.observation_count) == (None, "conflict_unknown", 2)
    report = evidence.repeated_decision_report(
        model_identifier=MODEL,
        question_hashes=TARGET.question_hashes,
        threshold=0.5,
        sources=["old", "new"],
    )
    comparison = report["source_comparisons"][0]
    assert comparison["shared_pairs"] == 1
    assert comparison["threshold_crossings"] == 1
    assert comparison["maximum_absolute_probability_difference"] == pytest.approx(0.05)
    assert report["near_threshold_decisions"]["old"]["within_margin"] == 1


def test_observations_for_other_models_or_prompts_are_never_reused() -> None:
    evidence = DecisionEvidence()
    evidence.add_rows(
        "run",
        [
            trace("r1", "A", True, model="other-model"),
            trace("r1", "B", True, question_hash="different-prompt"),
        ],
    )
    assert resolve_cell(evidence, "r1", "A", TARGET, ONLY_RUN).status == "missing"
    assert resolve_cell(evidence, "r1", "B", TARGET, ONLY_RUN).status == "missing"


def test_threshold_change_rethresholds_from_recorded_probability() -> None:
    evidence = DecisionEvidence()
    evidence.add_rows("run", [trace("r1", "A", True, probability=0.6)])
    strict = ReplayTarget(model_identifier=MODEL, threshold=0.7, question_hashes=hashes(IDS))
    assert resolve_cell(evidence, "r1", "A", strict, ONLY_RUN).outcome is False


def test_duplicate_decisions_within_one_source_are_rejected() -> None:
    evidence = DecisionEvidence()
    with pytest.raises(ValueError, match="more than once"):
        evidence.add_rows("run", [trace("r1", "A", True), trace("r1", "A", False)])


def test_physical_metrics_are_not_counterfactually_invented() -> None:
    evidence = evidence_from({"r1": {"A": True, "B": True, "C": True}})
    physical = replay_order(evidence, TARGET, ["r1"], IDS, ONLY_RUN)["physical_metrics"]
    assert physical["counterfactual_http_attempts"] is None
    assert physical["counterfactual_retries"] is None
    assert physical["counterfactual_wall_time_seconds"] is None
    assert physical["reused_observation_request_attempts"] == 3


def test_pattern_prior_imputation_is_labelled_as_estimate_within_bounds() -> None:
    profiling = matrix_from_patterns(IDS, [(True, True, True)] * 3 + [(False, False, False)])
    evidence = evidence_from({"r1": {"A": True}})
    prior = PatternPrior(profiling, IDS)
    report = replay_order(evidence, TARGET, ["r1"], IDS, ONLY_RUN, prior=prior)
    estimate = report["estimate"]
    assert estimate["is_exact"] is False
    assert estimate["estimate_kind"] == "profile_pattern_conditional_imputation"
    bounds = report["logical_evaluations"]
    assert bounds["lower"] <= estimate["logical_evaluations"] <= bounds["upper"]


def test_invalid_orders_are_rejected() -> None:
    evidence = evidence_from({"r1": {"A": True}})
    with pytest.raises(ValueError, match="duplicate"):
        replay_order(evidence, TARGET, ["r1"], ["A", "A"], ONLY_RUN)
    with pytest.raises(ValueError, match="unbound"):
        replay_order(evidence, TARGET, ["r1"], ["A", "Z"], ONLY_RUN)
