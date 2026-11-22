from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from semanticflow.optimizer.m51_experiment import load_yaml
from semanticflow.optimizer.m51_selection_analysis import load_evidence
from semanticflow.optimizer.m51_stage_d_analysis import normalize_stage_d_session
from semanticflow.optimizer.m52_acquisition import (
    AcquisitionReceipt,
    ActionKind,
    ProfilingAction,
)
from semanticflow.optimizer.m52_evidence import PairedQualityMeasurement

INPUT_PRICE_USD_PER_MILLION = 0.042


class ReplayVisibility(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    acquired_measurement_ids: frozenset[str] = frozenset()
    screened_record_ids: frozenset[str] = frozenset()
    paired_record_plan_ids: frozenset[str] = frozenset()


class ReplayAcquisition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    receipt: AcquisitionReceipt
    quality_measurement: PairedQualityMeasurement | None = None
    reference_decision: bool | None = None
    reference_minimum_margin: float | None = Field(default=None, ge=0, le=0.5)
    visible_trace_ids: tuple[str, ...] = ()


class HistoricalReplayMatrix:
    """A reveal-on-acquisition view over persisted M5.1 provider measurements."""

    def __init__(
        self,
        *,
        rows: Sequence[Mapping[str, Any]],
        plans: Sequence[Mapping[str, Any]],
        selections: Sequence[Mapping[str, Any]],
        config: Mapping[str, Any],
        source_partition: str,
    ) -> None:
        self.rows = tuple(dict(row) for row in rows)
        self.lookup = {(str(row["record_id"]), str(row["operator_id"])): dict(row) for row in rows}
        self.rows_by_request_id = {str(row["request_id"]): dict(row) for row in rows}
        self.plans = {str(plan["plan_id"]): dict(plan) for plan in plans}
        self.config = config
        self.source_partition = source_partition
        self._adaptive = {
            str(row["workload_id"]): self.plans[str(row["plan_id"])]
            for row in selections
            if row["strategy"] == "existing_adaptive"
        }
        self._visible: set[str] = set()
        self._screened: dict[tuple[str, str], dict[str, Any]] = {}
        self._paired: set[tuple[str, str]] = set()

    @classmethod
    def selection(cls) -> HistoricalReplayMatrix:
        evidence = load_evidence()
        selections_payload = json.loads(
            Path(
                "results/milestone_5_1/selection_stage_results/frozen_strategy_selections.json"
            ).read_text("utf-8")
        )
        return cls(
            rows=evidence["normalized"],
            plans=evidence["catalog"],
            selections=selections_payload["strategies"],
            config=evidence["config"],
            source_partition="selection",
        )

    @classmethod
    def stage_d(cls) -> HistoricalReplayMatrix:
        root = Path("results/milestone_5_1/selection_stage_results")
        schedule = json.loads((root / "stage_d_execution_schedule.json").read_text("utf-8"))
        rows, _, _ = normalize_stage_d_session(
            Path("results/milestone_5_1/stage_d_live_v3"), schedule
        )
        plans = json.loads((root / "frozen_physical_plan_catalog.json").read_text("utf-8"))["plans"]
        selections = json.loads((root / "frozen_strategy_selections.json").read_text("utf-8"))[
            "strategies"
        ]
        config = load_yaml(Path("configs/milestone_5_1/workloads.yaml"))
        return cls(
            rows=rows,
            plans=plans,
            selections=selections,
            config=config,
            source_partition="evaluation",
        )

    @property
    def visibility(self) -> ReplayVisibility:
        return ReplayVisibility(
            acquired_measurement_ids=frozenset(self._visible),
            screened_record_ids=frozenset(record_id for record_id, _ in self._screened),
            paired_record_plan_ids=frozenset(
                f"{record_id}|{plan_id}" for record_id, plan_id in self._paired
            ),
        )

    def records(self, dataset: str) -> list[str]:
        return sorted({str(row["record_id"]) for row in self.rows if row["dataset"] == dataset})

    def workload(self, workload_id: str) -> Mapping[str, Any]:
        return next(row for row in self.config["workloads"] if row["id"] == workload_id)

    def _path(self, plan: Mapping[str, Any], record_id: str) -> list[dict[str, Any]]:
        output = []
        active = True
        for group in plan["groups"]:
            if not active:
                break
            key = (record_id, str(group["operator_id"]))
            if key not in self.lookup:
                raise KeyError(f"historical measurement unavailable: {key}")
            row = self.lookup[key]
            output.append(row)
            active = bool(row["group_passed"])
        return output

    @staticmethod
    def _decision(path: Sequence[Mapping[str, Any]]) -> bool:
        return bool(path) and all(bool(row["group_passed"]) for row in path)

    @staticmethod
    def _minimum_margin(path: Sequence[Mapping[str, Any]]) -> float | None:
        probabilities = [
            abs(float(value) - 0.5)
            for row in path
            for value in row["predicate_probabilities"].values()
        ]
        return min(probabilities) if probabilities else None

    def _charge(self, rows: Sequence[Mapping[str, Any]], action_id: str) -> AcquisitionReceipt:
        acquired = [row for row in rows if str(row["request_id"]) not in self._visible]
        reused = [row for row in rows if str(row["request_id"]) in self._visible]
        self._visible.update(str(row["request_id"]) for row in acquired)
        return AcquisitionReceipt(
            action_id=action_id,
            acquired_measurement_ids=tuple(str(row["request_id"]) for row in acquired),
            reused_measurement_ids=tuple(str(row["request_id"]) for row in reused),
            physical_requests=len(acquired),
            named_question_evaluations=sum(
                int(row["named_question_evaluations"]) for row in acquired
            ),
            input_tokens=sum(int(row["input_tokens"]) for row in acquired),
            output_tokens=sum(int(row["output_tokens"]) for row in acquired),
            token_priced_cost_usd=sum(float(row["cost_usd"]) for row in acquired),
        )

    def acquire(self, action: ProfilingAction) -> ReplayAcquisition:
        if action.kind == ActionKind.STOP:
            raise ValueError("stop actions do not acquire evidence")
        assert action.record_id is not None
        assert action.plan_id is not None
        assert action.track is not None
        if any(item not in self._visible for item in action.prerequisite_measurement_ids):
            return ReplayAcquisition(
                receipt=AcquisitionReceipt(
                    action_id=action.action_id,
                    acquired_measurement_ids=(),
                    reused_measurement_ids=(),
                    physical_requests=0,
                    named_question_evaluations=0,
                    input_tokens=0,
                    output_tokens=0,
                    token_priced_cost_usd=0,
                    unsupported_reason="missing_action_prerequisites",
                )
            )
        plan = self.plans.get(action.plan_id)
        if plan is None:
            raise KeyError(f"unknown plan {action.plan_id}")
        workload_id = str(plan["workload_id"])
        adaptive = self._adaptive[workload_id]
        try:
            reference_path = self._path(adaptive, action.record_id)
        except KeyError as exc:
            return ReplayAcquisition(
                receipt=AcquisitionReceipt(
                    action_id=action.action_id,
                    acquired_measurement_ids=(),
                    reused_measurement_ids=(),
                    physical_requests=0,
                    named_question_evaluations=0,
                    input_tokens=0,
                    output_tokens=0,
                    token_priced_cost_usd=0,
                    unsupported_reason=str(exc),
                )
            )
        reference_decision = self._decision(reference_path)
        margin = self._minimum_margin(reference_path)
        screen_key = (action.record_id, workload_id)
        if action.kind == ActionKind.SCREEN_DISCOVERY_REFERENCE:
            receipt = self._charge(reference_path, action.action_id)
            self._screened[screen_key] = {
                "decision": reference_decision,
                "margin": margin,
                "request_ids": tuple(row["request_id"] for row in reference_path),
                "cost_usd": receipt.token_priced_cost_usd,
            }
            return ReplayAcquisition(
                receipt=receipt,
                reference_decision=reference_decision,
                reference_minimum_margin=margin,
                visible_trace_ids=tuple(row["request_id"] for row in reference_path),
            )
        if (
            action.kind == ActionKind.ACQUIRE_DISCOVERY_CANDIDATE
            and screen_key not in self._screened
        ):
            return ReplayAcquisition(
                receipt=AcquisitionReceipt(
                    action_id=action.action_id,
                    acquired_measurement_ids=(),
                    reused_measurement_ids=(),
                    physical_requests=0,
                    named_question_evaluations=0,
                    input_tokens=0,
                    output_tokens=0,
                    token_priced_cost_usd=0,
                    unsupported_reason="record_not_screened",
                )
            )
        try:
            candidate_path = self._path(plan, action.record_id)
        except KeyError as exc:
            return ReplayAcquisition(
                receipt=AcquisitionReceipt(
                    action_id=action.action_id,
                    acquired_measurement_ids=(),
                    reused_measurement_ids=(),
                    physical_requests=0,
                    named_question_evaluations=0,
                    input_tokens=0,
                    output_tokens=0,
                    token_priced_cost_usd=0,
                    unsupported_reason=str(exc),
                )
            )
        all_rows = [*reference_path, *candidate_path]
        receipt = self._charge(all_rows, action.action_id)
        candidate_decision = self._decision(candidate_path)
        self._paired.add((action.record_id, action.plan_id))
        measurement_id = (
            "m52quality_"
            + hashlib.sha256(
                f"{action.action_id}|{action.record_id}|{action.plan_id}".encode()
            ).hexdigest()[:24]
        )
        quality = PairedQualityMeasurement(
            measurement_id=measurement_id,
            record_id=action.record_id,
            dataset=str(plan["dataset"]),
            workload_id=workload_id,
            plan_id=action.plan_id,
            track=action.track,
            acquisition_method=action.acquisition_method,
            source_partition=self.source_partition,
            acquisition_checkpoint=int(action.score_components.get("checkpoint", 0)),
            reference_decision=reference_decision,
            candidate_decision=candidate_decision,
            reference_probability_min_margin=margin,
            inclusion_probability=(
                float(action.score_components["inclusion_probability"])
                if action.score_components.get("inclusion_probability") is not None
                else None
            ),
            request_ids=tuple(str(row["request_id"]) for row in all_rows),
            physical_requests=receipt.physical_requests,
            named_question_evaluations=receipt.named_question_evaluations,
            input_tokens=receipt.input_tokens,
            output_tokens=receipt.output_tokens,
            token_priced_cost_usd=receipt.token_priced_cost_usd,
        )
        return ReplayAcquisition(
            receipt=receipt,
            quality_measurement=quality,
            reference_decision=reference_decision,
            reference_minimum_margin=margin,
            visible_trace_ids=tuple(str(row["request_id"]) for row in all_rows),
        )

    def visible_screen(self, record_id: str, workload_id: str) -> Mapping[str, Any] | None:
        return self._screened.get((record_id, workload_id))

    def trace(self, request_id: str) -> Mapping[str, Any]:
        if request_id not in self._visible:
            raise PermissionError("future-information access: measurement has not been acquired")
        return self.rows_by_request_id[request_id]
