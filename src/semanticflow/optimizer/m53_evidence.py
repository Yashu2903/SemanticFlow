from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
M51_AUDIT = ROOT / "results" / "milestone_5_1" / "final_research_audit"
M51_LIVE = ROOT / "results" / "milestone_5_1" / "stage_d_live_v3" / "operators"
M51_SELECTION = ROOT / "results" / "milestone_5_1" / "selection_stage_results"
M523_LIVE = ROOT / "results" / "milestone_5_2_3" / "live_pilot_a_repaired_session_01"


@dataclass(frozen=True)
class Usage:
    requests: int = 0
    attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    named_evaluations: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            requests=self.requests + other.requests,
            attempts=self.attempts + other.attempts,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            named_evaluations=self.named_evaluations + other.named_evaluations,
            cost_usd=self.cost_usd + other.cost_usd,
        )


@dataclass(frozen=True)
class EvidenceRecord:
    source: str
    partition: str
    workload_id: str
    dataset: str
    record_id: str
    predicate_ids: tuple[str, ...]
    batch_operator_id: str
    screen_predicate_id: str
    separate_probabilities: dict[str, float]
    batch_probabilities: dict[str, float]
    separate_decisions: dict[str, bool]
    batch_decisions: dict[str, bool]
    request_ids: tuple[str, ...]
    screen_usage: Usage
    followup_usage: Usage
    full_usage: Usage

    @property
    def screen_probability(self) -> float:
        return self.separate_probabilities[self.screen_predicate_id]

    @property
    def predicate_disagreement_count(self) -> int:
        return sum(
            self.separate_decisions[predicate] != self.batch_decisions[predicate]
            for predicate in self.predicate_ids
        )

    @property
    def separate_query_positive(self) -> bool:
        return all(self.separate_decisions.values())

    @property
    def batch_query_positive(self) -> bool:
        return all(self.batch_decisions.values())

    @property
    def query_disagreement(self) -> bool:
        return self.separate_query_positive != self.batch_query_positive

    @property
    def false_negative(self) -> bool:
        return self.separate_query_positive and not self.batch_query_positive

    @property
    def false_positive(self) -> bool:
        return self.batch_query_positive and not self.separate_query_positive


@dataclass(frozen=True)
class WorkloadSpec:
    workload_id: str
    dataset: str
    predicates: tuple[str, ...]
    screen_predicate: str
    batch_operator: str
    single_operators: dict[str, str]


WORKLOADS: tuple[WorkloadSpec, ...] = (
    WorkloadSpec(
        workload_id="m51_banking_card_unauthorized_refund",
        dataset="banking77_queries",
        predicates=(
            "banking_card_related",
            "banking_unauthorized_activity",
            "banking_refund_or_reversal",
        ),
        screen_predicate="banking_card_related",
        batch_operator="fusion_7a530d955e1a88eb",
        single_operators={
            "banking_card_related": "fusion_41883a26b65ef52a",
            "banking_unauthorized_activity": "fusion_90a28cccc412ea36",
            "banking_refund_or_reversal": "fusion_f9b70e5f0e90e392",
        },
    ),
    WorkloadSpec(
        workload_id="m51_beauty_failure_usability",
        dataset="amazon_all_beauty",
        predicates=("beauty_malfunction", "beauty_hard_to_use"),
        screen_predicate="beauty_malfunction",
        batch_operator="fusion_36f72d9213396451",
        single_operators={
            "beauty_malfunction": "fusion_8ec118dea509fb59",
            "beauty_hard_to_use": "fusion_aacabf1abd806e2d",
        },
    ),
    WorkloadSpec(
        workload_id="m51_sms_prize_urgency_contact",
        dataset="sms_spam",
        predicates=("sms_prize_or_reward", "sms_urgency_pressure", "sms_contact_action"),
        screen_predicate="sms_prize_or_reward",
        batch_operator="fusion_3ac5aa84cc3217ed",
        single_operators={
            "sms_prize_or_reward": "fusion_eb694e96e28a402b",
            "sms_urgency_pressure": "fusion_7e7997e62429e776",
            "sms_contact_action": "fusion_1ee25f11a4ff50b7",
        },
    ),
)


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[str(row["review_id"])] = row
    return rows


def _usage(row: dict[str, str]) -> Usage:
    return Usage(
        requests=1,
        attempts=int(row["http_attempts"]),
        input_tokens=int(row["input_tokens"]),
        output_tokens=int(row["output_tokens"]),
        named_evaluations=int(row["named_question_evaluations"]),
        cost_usd=float(row["cost_usd"]),
    )


def _operator_decisions(dataset: str, operator_id: str) -> dict[str, dict[str, Any]]:
    directory = M51_LIVE / dataset / operator_id
    metadata = json.loads((directory / "operator.json").read_text(encoding="utf-8"))
    if metadata["group_id"] != operator_id or metadata["threshold"] != 0.5:
        raise ValueError(f"incompatible operator metadata: {operator_id}")
    return _read_jsonl(directory / "decisions.jsonl")


def load_stage_d_evidence() -> tuple[EvidenceRecord, ...]:
    expenditure: dict[tuple[str, str], dict[str, str]] = {}
    with (M51_AUDIT / "per_request_expenditure.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            expenditure[(row["operator_id"], row["record_id"])] = row

    output: list[EvidenceRecord] = []
    for spec in WORKLOADS:
        operators = {spec.batch_operator, *spec.single_operators.values()}
        decisions = {
            operator: _operator_decisions(spec.dataset, operator) for operator in operators
        }
        record_sets = [set(rows) for rows in decisions.values()]
        records = set.intersection(*record_sets)
        if len(records) != 400 or any(record_set != records for record_set in record_sets):
            raise ValueError(f"incomplete Stage D coverage for {spec.workload_id}")

        for record_id in sorted(records):
            separate_probabilities: dict[str, float] = {}
            separate_decisions: dict[str, bool] = {}
            request_ids: list[str] = []
            separate_usage: dict[str, Usage] = {}
            for predicate in spec.predicates:
                operator = spec.single_operators[predicate]
                decision = decisions[operator][record_id]
                probability = float(decision["predicate_probabilities"][predicate])
                separate_probabilities[predicate] = probability
                separate_decisions[predicate] = probability >= 0.5
                cost_row = expenditure[(operator, record_id)]
                request_ids.append(cost_row["request_id"])
                separate_usage[predicate] = _usage(cost_row)

            batch = decisions[spec.batch_operator][record_id]
            batch_probabilities = {
                predicate: float(batch["predicate_probabilities"][predicate])
                for predicate in spec.predicates
            }
            batch_decisions = {
                predicate: probability >= 0.5
                for predicate, probability in batch_probabilities.items()
            }
            batch_cost_row = expenditure[(spec.batch_operator, record_id)]
            request_ids.append(batch_cost_row["request_id"])
            batch_usage = _usage(batch_cost_row)
            screen_usage = separate_usage[spec.screen_predicate]
            followup_usage = batch_usage
            full_usage = batch_usage
            for predicate, usage in separate_usage.items():
                full_usage += usage
                if predicate != spec.screen_predicate:
                    followup_usage += usage

            output.append(
                EvidenceRecord(
                    source="m51_stage_d_evaluation",
                    partition="historical_evaluation",
                    workload_id=spec.workload_id,
                    dataset=spec.dataset,
                    record_id=record_id,
                    predicate_ids=spec.predicates,
                    batch_operator_id=spec.batch_operator,
                    screen_predicate_id=spec.screen_predicate,
                    separate_probabilities=separate_probabilities,
                    batch_probabilities=batch_probabilities,
                    separate_decisions=separate_decisions,
                    batch_decisions=batch_decisions,
                    request_ids=tuple(request_ids),
                    screen_usage=screen_usage,
                    followup_usage=followup_usage,
                    full_usage=full_usage,
                )
            )
    return tuple(output)


def load_m523_representative_evidence() -> tuple[EvidenceRecord, ...]:
    with (M523_LIVE / "representative_measurements.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        source_rows = list(csv.DictReader(handle))
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in source_rows:
        grouped.setdefault((row["workload_id"], row["record_id"]), []).append(row)

    specs = {spec.workload_id: spec for spec in WORKLOADS}
    output: list[EvidenceRecord] = []
    for (workload_id, record_id), rows in sorted(grouped.items()):
        spec = specs[workload_id]
        singles = {
            row["predicate_ids"]: row for row in rows if row["physical_operator"] == "single"
        }
        batches = [row for row in rows if row["physical_operator"] == "multi_decision"]
        if set(singles) != set(spec.predicates) or len(batches) != 1:
            raise ValueError(f"incomplete M5.2.3 representative evidence: {record_id}")
        batch = batches[0]
        separate_probabilities = {
            predicate: float(json.loads(singles[predicate]["probabilities_json"])[predicate])
            for predicate in spec.predicates
        }
        separate_decisions = {
            predicate: bool(json.loads(singles[predicate]["decisions_json"])[predicate])
            for predicate in spec.predicates
        }
        batch_probabilities_json = json.loads(batch["probabilities_json"])
        batch_decisions_json = json.loads(batch["decisions_json"])
        batch_probabilities = {
            predicate: float(batch_probabilities_json[predicate]) for predicate in spec.predicates
        }
        batch_decisions = {
            predicate: bool(batch_decisions_json[predicate]) for predicate in spec.predicates
        }

        def live_usage(row: dict[str, str]) -> Usage:
            predicates = row["predicate_ids"].split("|")
            return Usage(
                requests=1,
                attempts=int(row["http_attempts"]),
                input_tokens=int(row["input_tokens"]),
                output_tokens=int(row["output_tokens"]),
                named_evaluations=len(predicates),
                cost_usd=float(row["token_priced_cost_usd"]),
            )

        single_usage = {predicate: live_usage(row) for predicate, row in singles.items()}
        batch_usage = live_usage(batch)
        screen_usage = single_usage[spec.screen_predicate]
        full_usage = batch_usage
        followup_usage = batch_usage
        for predicate, usage in single_usage.items():
            full_usage += usage
            if predicate != spec.screen_predicate:
                followup_usage += usage
        output.append(
            EvidenceRecord(
                source="m523_live_representative",
                partition="representative",
                workload_id=workload_id,
                dataset=spec.dataset,
                record_id=record_id,
                predicate_ids=spec.predicates,
                batch_operator_id=spec.batch_operator,
                screen_predicate_id=spec.screen_predicate,
                separate_probabilities=separate_probabilities,
                batch_probabilities=batch_probabilities,
                separate_decisions=separate_decisions,
                batch_decisions=batch_decisions,
                request_ids=tuple(row["action_id"] for row in rows),
                screen_usage=screen_usage,
                followup_usage=followup_usage,
                full_usage=full_usage,
            )
        )
    if len(output) != 60:
        raise ValueError(f"expected 60 M5.2.3 representative records, got {len(output)}")
    return tuple(output)


def m523_discovery_coverage() -> list[dict[str, Any]]:
    with (M523_LIVE / "discovery_measurements.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["workload_id"], row["record_id"]), []).append(row)
    specs = {spec.workload_id: spec for spec in WORKLOADS}
    output = []
    for (workload_id, record_id), record_rows in sorted(grouped.items()):
        spec = specs[workload_id]
        output.append(
            {
                "source": "m523_live_discovery",
                "partition": "discovery_enriched",
                "workload_id": workload_id,
                "dataset": spec.dataset,
                "record_id": record_id,
                "original_record_identity": True,
                "predicate_definitions": True,
                "separate_probabilities": False,
                "batch_probabilities": any(
                    row["measurement_type"] == "discovery_followup" for row in record_rows
                ),
                "complete_query_outputs": False,
                "screening_evidence": True,
                "physical_request_identities": True,
                "actual_tokens": True,
                "actual_cost": True,
                "sampling_provenance": True,
                "decision_time_visibility": True,
                "complete_counterfactual": False,
                "limitation": "missing required separate and/or batched follow-up outcomes",
            }
        )
    return output


def selection_stage_coverage() -> list[dict[str, Any]]:
    with (M51_SELECTION / "complete_measurement_matrix.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    by_operator: dict[str, set[str]] = {}
    for row in rows:
        by_operator.setdefault(row["operator_id"], set()).add(row["record_id"])
    output: list[dict[str, Any]] = []
    for spec in WORKLOADS:
        operators = (spec.batch_operator, *spec.single_operators.values())
        record_sets = [by_operator.get(operator, set()) for operator in operators]
        compatible = set.intersection(*record_sets) if record_sets else set()
        for record_id in sorted(compatible):
            output.append(
                {
                    "source": "m51_selection_development",
                    "partition": "historical_development",
                    "workload_id": spec.workload_id,
                    "dataset": spec.dataset,
                    "record_id": record_id,
                    "original_record_identity": True,
                    "predicate_definitions": True,
                    "separate_probabilities": True,
                    "batch_probabilities": True,
                    "complete_query_outputs": True,
                    "screening_evidence": True,
                    "physical_request_identities": True,
                    "actual_tokens": True,
                    "actual_cost": True,
                    "sampling_provenance": True,
                    "decision_time_visibility": True,
                    "complete_counterfactual": True,
                    "limitation": "development evidence; excluded from primary evaluation",
                }
            )
    return output


def coverage_rows(records: tuple[EvidenceRecord, ...]) -> list[dict[str, Any]]:
    return [
        {
            "source": record.source,
            "partition": record.partition,
            "workload_id": record.workload_id,
            "dataset": record.dataset,
            "record_id": record.record_id,
            "original_record_identity": True,
            "predicate_definitions": True,
            "separate_probabilities": True,
            "batch_probabilities": True,
            "complete_query_outputs": True,
            "screening_evidence": True,
            "physical_request_identities": True,
            "actual_tokens": True,
            "actual_cost": True,
            "sampling_provenance": True,
            "decision_time_visibility": True,
            "complete_counterfactual": True,
            "limitation": "",
        }
        for record in records
    ]
