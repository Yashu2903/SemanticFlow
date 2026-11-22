from __future__ import annotations

import pytest
from pydantic import ValidationError

from semanticflow.optimizer.m52_evidence import (
    EvidenceTrack,
    PairedQualityMeasurement,
    QualificationState,
    QualityEvidence,
)
from semanticflow.optimizer.m52_quality_uncertainty import (
    SequentialQualificationConfig,
    exact_one_sided_bounds,
    qualify_evidence,
    wilson_interval,
)


def measurement(
    index: int,
    reference: bool,
    candidate: bool,
    *,
    track: EvidenceTrack = EvidenceTrack.CERTIFICATION,
) -> PairedQualityMeasurement:
    return PairedQualityMeasurement(
        measurement_id=f"m{index}",
        record_id=f"r{index}",
        dataset="dataset",
        workload_id="workload",
        plan_id="plan",
        track=track,
        acquisition_method="test",
        source_partition="selection",
        acquisition_checkpoint=0,
        reference_decision=reference,
        candidate_decision=candidate,
        request_ids=(f"request{index}",),
        physical_requests=1,
        named_question_evaluations=1,
        input_tokens=100,
        output_tokens=1,
        token_priced_cost_usd=0.0000042,
    )


def evidence(rows: list[PairedQualityMeasurement]) -> QualityEvidence:
    return QualityEvidence(
        plan_id="plan",
        plan_fingerprint="fingerprint",
        dataset="dataset",
        dataset_fingerprint="dataset-fingerprint",
        workload_id="workload",
        workload_fingerprint="workload-fingerprint",
        physical_groups=(("a", "b"),),
        measurements=tuple(rows),
    )


@pytest.mark.parametrize(
    ("successes", "trials", "expected"),
    [(0, 3, 0.0), (1, 3, 1 / 3), (3, 3, 1.0), (21, 22, 21 / 22)],
)
def test_wilson_contains_point(successes: int, trials: int, expected: float) -> None:
    lower, upper = wilson_interval(successes, trials)
    assert lower <= expected <= upper


def test_exact_three_of_three_is_not_strong_recall_evidence() -> None:
    lower, upper = exact_one_sided_bounds(3, 3, 0.05)
    assert lower == pytest.approx(0.05 ** (1 / 3))
    assert upper == 1
    assert lower < 0.95


def test_exact_bounds_are_monotone_in_successes() -> None:
    bounds = [exact_one_sided_bounds(value, 20, 0.01) for value in range(21)]
    assert all(
        left[0] <= right[0] for left, right in zip(bounds, bounds[1:], strict=False)
    )
    assert all(
        left[1] <= right[1] for left, right in zip(bounds, bounds[1:], strict=False)
    )


def test_zero_positive_recall_is_undefined_and_support_insufficient() -> None:
    rows = [measurement(index, False, False) for index in range(40)]
    estimate = qualify_evidence(evidence(rows), SequentialQualificationConfig(), checkpoint=0)
    assert estimate.state == QualificationState.INSUFFICIENT_SUPPORT
    assert estimate.recall.estimate is None
    assert estimate.false_negative_rate.estimate is None


def test_three_positive_perfect_sample_remains_uncertain() -> None:
    rows = [measurement(index, index < 3, index < 3) for index in range(40)]
    estimate = qualify_evidence(
        evidence(rows),
        SequentialQualificationConfig(maximum_checkpoints=1, family_candidate_count=1),
        checkpoint=0,
    )
    assert estimate.counts.reference_positives == 3
    assert estimate.recall.estimate == 1
    assert estimate.recall.sequential_lower < 0.95
    assert estimate.state == QualificationState.UNCERTAIN


def test_large_perfect_representative_sample_can_qualify() -> None:
    rows = [measurement(index, True, True) for index in range(400)]
    estimate = qualify_evidence(
        evidence(rows),
        SequentialQualificationConfig(maximum_checkpoints=1, family_candidate_count=1),
        checkpoint=0,
    )
    assert estimate.state == QualificationState.QUALIFIED
    assert estimate.recall.sequential_lower >= 0.95
    assert estimate.changed_decision_rate.sequential_upper <= 0.02


def test_clear_quality_violation_is_rejected() -> None:
    rows = [measurement(index, True, index < 50) for index in range(100)]
    estimate = qualify_evidence(
        evidence(rows),
        SequentialQualificationConfig(maximum_checkpoints=1, family_candidate_count=1),
        checkpoint=0,
    )
    assert estimate.state == QualificationState.REJECTED


def test_discovery_rows_do_not_enter_certification_counts() -> None:
    rows = [
        measurement(index, True, True, track=EvidenceTrack.DISCOVERY) for index in range(10)
    ] + [measurement(100 + index, False, False) for index in range(40)]
    item = evidence(rows)
    assert item.counts().observations == 40
    assert item.counts().reference_positives == 0
    assert item.counts(EvidenceTrack.DISCOVERY).reference_positives == 10


def test_evidence_track_record_overlap_is_rejected() -> None:
    first = measurement(1, True, True, track=EvidenceTrack.DISCOVERY)
    second = measurement(2, True, True).model_copy(update={"record_id": first.record_id})
    with pytest.raises(ValidationError, match="must be disjoint"):
        evidence([first, second])


def test_error_spending_covers_frozen_family() -> None:
    config = SequentialQualificationConfig(
        family_alpha=0.05,
        maximum_checkpoints=8,
        family_candidate_count=4,
        simultaneous_bounds=4,
    )
    assert (
        config.allocated_one_sided_alpha
        * config.maximum_checkpoints
        * config.family_candidate_count
        * config.simultaneous_bounds
        == pytest.approx(config.family_alpha)
    )


def test_checkpoint_outside_frozen_schedule_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside"):
        qualify_evidence(evidence([]), SequentialQualificationConfig(), checkpoint=8)
