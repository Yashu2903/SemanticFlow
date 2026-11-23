from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

from semanticflow.optimizer.m52_evidence import EvidenceTrack
from semanticflow.optimizer.m522_action_universe import (
    AuthorizedActionUniverse,
    AuthorizedLiveAction,
    LiveMeasurementType,
    PhysicalOperatorType,
    action_identity,
    universe_fingerprint,
)

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ID = "semanticflow-m5.3.1-fresh-evidence-validation-v1"
MODEL = "jev-1.14.0"
THRESHOLD = 0.5
SELECTION_SEED = 53101
RECORDS_PER_WORKLOAD = 100
REPRESENTATIVE_PER_WORKLOAD = 50
ENRICHED_PER_WORKLOAD = 50
HARD_REQUEST_LIMIT = 1100
HARD_ATTEMPT_LIMIT = 3300
MAX_RETRIES = 2
PLANNING_COST_USD = 0.0155
DEFAULT_HARD_COST_USD = 0.025
REQUEST_BUDGETS = (5, 10, 20, 30, 50, 100)
DOLLAR_BUDGETS = (0.0001, 0.00025, 0.0005, 0.001, 0.0015)
REPLAY_SEEDS = tuple(range(53101, 53151))
DATA_DIRECTORY = ROOT / "data" / "milestone_5"
WORKLOAD_CONFIG = ROOT / "configs" / "milestone_5_1" / "workloads.yaml"
POLICY_SOURCE = ROOT / "src" / "semanticflow" / "optimizer" / "m53_policies.py"
FROZEN_SOURCE_PATHS = (
    "src/semanticflow/optimizer/m53_policies.py",
    "src/semanticflow/optimizer/m53_replay.py",
    "src/semanticflow/optimizer/m531_protocol.py",
    "src/semanticflow/optimizer/m531_live.py",
    "src/semanticflow/optimizer/m531_analysis.py",
    "src/semanticflow/optimizer/m522_live_acquisition.py",
    "src/semanticflow/optimizer/m522_jev_adapter.py",
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FreshRecord:
    workload_id: str
    dataset: str
    record_id: str
    source_record_id: str
    text: str
    text_sha256: str
    original_partition: str
    stratum: str
    enrichment_score: float


def _workloads() -> tuple[dict[str, Any], ...]:
    config = yaml.safe_load(WORKLOAD_CONFIG.read_text("utf-8"))
    wanted = {
        "m51_banking_card_unauthorized_refund",
        "m51_beauty_failure_usability",
        "m51_sms_prize_urgency_contact",
    }
    return tuple(item for item in config["workloads"] if item["id"] in wanted)


def _predicate_instructions() -> dict[str, dict[str, str]]:
    config = yaml.safe_load(WORKLOAD_CONFIG.read_text("utf-8"))
    return {dataset: dict(spec["predicates"]) for dataset, spec in config["datasets"].items()}


def historical_provider_identities() -> tuple[set[str], set[str], dict[str, str]]:
    """Return all stable IDs/text hashes visible in prior provider evidence.

    IDs are extracted from the immutable M5.1 and M5.2.3 artifact trees, then
    resolved to content hashes through every locally versioned dataset copy.
    """
    roots = (
        ROOT / "results" / "milestone_5_1",
        ROOT / "results" / "milestone_5_2_3",
    )
    pattern = re.compile(r"(?:amazon_all_beauty|banking77_queries|sms_spam):[0-9a-f]{24}")
    ids: set[str] = set()
    source_files: dict[str, str] = {}
    for root in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".csv", ".json", ".jsonl"}:
                continue
            try:
                content = path.read_text("utf-8")
            except UnicodeDecodeError:
                continue
            found = set(pattern.findall(content))
            if found:
                ids.update(found)
                source_files[path.relative_to(ROOT).as_posix()] = file_hash(path)
    hashes: set[str] = set()
    for directory in (ROOT / "data").glob("milestone_*"):
        for path in directory.glob("*.parquet"):
            table = pq.read_table(path, columns=["review_id", "text_sha256"])
            for row in table.to_pylist():
                if str(row["review_id"]) in ids:
                    hashes.add(str(row["text_sha256"]))
    return ids, hashes, source_files


LEXICONS = {
    "m51_banking_card_unauthorized_refund": (
        "card",
        "payment",
        "charge",
        "unauthor",
        "recogn",
        "refund",
        "revers",
        "stolen",
    ),
    "m51_beauty_failure_usability": (
        "broke",
        "broken",
        "fail",
        "work",
        "leak",
        "hard",
        "difficult",
        "use",
        "apply",
    ),
    "m51_sms_prize_urgency_contact": (
        "prize",
        "won",
        "reward",
        "voucher",
        "urgent",
        "now",
        "today",
        "call",
        "text",
        "click",
    ),
}


def _enrichment_score(workload_id: str, text: str) -> float:
    lowered = text.lower()
    matches = sum(term in lowered for term in LEXICONS[workload_id])
    # Prefer partial lexical coverage over obvious all-negative/all-positive extremes.
    target = 2 if workload_id != "m51_beauty_failure_usability" else 1
    return 1.0 / (1.0 + abs(matches - target)) + min(len(text), 240) / 24000


def select_fresh_records() -> tuple[tuple[FreshRecord, ...], dict[str, Any]]:
    prior_ids, prior_hashes, registry_sources = historical_provider_identities()
    selected: list[FreshRecord] = []
    eligibility: dict[str, Any] = {}
    for workload in _workloads():
        dataset = str(workload["dataset"])
        table = pq.read_table(DATA_DIRECTORY / f"{dataset}.parquet")
        candidates = [
            row
            for row in table.to_pylist()
            if str(row["review_id"]) not in prior_ids
            and str(row["text_sha256"]) not in prior_hashes
        ]
        if len(candidates) < RECORDS_PER_WORKLOAD:
            raise ValueError(f"only {len(candidates)} independent records for {workload['id']}")
        ordered = sorted(candidates, key=lambda row: str(row["review_id"]))
        rng = random.Random(SELECTION_SEED + sum(map(ord, str(workload["id"]))))
        rng.shuffle(ordered)
        representative = ordered[:REPRESENTATIVE_PER_WORKLOAD]
        representative_ids = {str(row["review_id"]) for row in representative}
        remainder = [row for row in candidates if str(row["review_id"]) not in representative_ids]
        enriched = sorted(
            remainder,
            key=lambda row: (
                -_enrichment_score(str(workload["id"]), str(row["review_text"])),
                str(row["text_sha256"]),
            ),
        )[:ENRICHED_PER_WORKLOAD]
        for stratum, rows in (
            ("representative_random", representative),
            ("lexical_enrichment", enriched),
        ):
            for row in rows:
                selected.append(
                    FreshRecord(
                        workload_id=str(workload["id"]),
                        dataset=dataset,
                        record_id=str(row["review_id"]),
                        source_record_id=str(row["source_record_id"]),
                        text=str(row["review_text"]),
                        text_sha256=str(row["text_sha256"]),
                        original_partition=str(row["partition"]),
                        stratum=stratum,
                        enrichment_score=_enrichment_score(
                            str(workload["id"]), str(row["review_text"])
                        ),
                    )
                )
        eligibility[str(workload["id"])] = {
            "source_records": table.num_rows,
            "independently_eligible": len(candidates),
            "selected": RECORDS_PER_WORKLOAD,
            "representative": REPRESENTATIVE_PER_WORKLOAD,
            "enriched": ENRICHED_PER_WORKLOAD,
        }
    if len({record.record_id for record in selected}) != len(selected):
        raise ValueError("fresh selection contains duplicate stable identities")
    if len({record.text_sha256 for record in selected}) != len(selected):
        raise ValueError("fresh selection contains duplicate content hashes")
    return tuple(sorted(selected, key=lambda row: (row.workload_id, row.stratum, row.record_id))), {
        "eligibility": eligibility,
        "historical_record_ids": len(prior_ids),
        "historical_content_hashes": len(prior_hashes),
        "historical_registry_sources": registry_sources,
        "historical_registry_fingerprint": canonical_hash(
            {"ids": sorted(prior_ids), "hashes": sorted(prior_hashes)}
        ),
    }


def frozen_policy_configuration() -> dict[str, Any]:
    definitions = {
        "random": {"requires_screen": False},
        "uncertainty_continuous": {"threshold": 0.5, "ranking": "absolute_distance"},
        "positive_oriented": {"ranking": "descending_screen_probability"},
        "hybrid_weighted": {
            "uncertainty_weight": 0.5,
            "positive_weight": 0.5,
            "cost_aware": True,
        },
        "hybrid_quota": {"alternates": ["uncertainty_continuous", "positive_oriented"]},
    }
    return {
        "definitions": definitions,
        "policy_source": POLICY_SOURCE.relative_to(ROOT).as_posix(),
        "policy_source_sha256": file_hash(POLICY_SOURCE),
        "fingerprint": canonical_hash(definitions),
    }


def build_experiment() -> tuple[
    dict[str, Any], tuple[FreshRecord, ...], AuthorizedActionUniverse, list[dict[str, Any]]
]:
    records, independence = select_fresh_records()
    instructions = _predicate_instructions()
    workload_index = {str(item["id"]): item for item in _workloads()}
    actions: list[AuthorizedLiveAction] = []
    plan: list[dict[str, Any]] = []
    for record in records:
        workload = workload_index[record.workload_id]
        predicates = tuple(map(str, workload["predicate_ids"]))
        for index, predicate in enumerate(predicates):
            role = "screen" if index == 0 else "separate_predicate"
            identity = action_identity(
                {
                    "experiment": EXPERIMENT_ID,
                    "record": record.record_id,
                    "role": role,
                    "predicates": [predicate],
                }
            )
            predicted_input = 325
            action = AuthorizedLiveAction(
                action_id=identity,
                workload_id=record.workload_id,
                dataset=record.dataset,
                source_partition=record.stratum,
                record_id=record.record_id,
                record_text_sha256=record.text_sha256,
                measurement_type=LiveMeasurementType.REPRESENTATIVE_REFERENCE,
                physical_operator=PhysicalOperatorType.SINGLE,
                track=EvidenceTrack.CERTIFICATION,
                predicate_ids=(predicate,),
                predicate_instructions=(instructions[record.dataset][predicate],),
                threshold=THRESHOLD,
                requested_model=MODEL,
                maximum_attempts=MAX_RETRIES + 1,
                predicted_input_tokens=predicted_input,
                predicted_output_tokens=35,
                predicted_cost_usd=predicted_input * 0.042 / 1_000_000,
            )
            actions.append(action)
            plan.append({"action_id": identity, "role": role, "predicate_group": predicate})
        identity = action_identity(
            {
                "experiment": EXPERIMENT_ID,
                "record": record.record_id,
                "role": "complete_multi_predicate",
                "predicates": predicates,
            }
        )
        predicted_input = {2: 364, 3: 398}[len(predicates)]
        action = AuthorizedLiveAction(
            action_id=identity,
            workload_id=record.workload_id,
            dataset=record.dataset,
            source_partition=record.stratum,
            record_id=record.record_id,
            record_text_sha256=record.text_sha256,
            measurement_type=LiveMeasurementType.REPRESENTATIVE_CANDIDATE,
            physical_operator=PhysicalOperatorType.MULTI_DECISION,
            track=EvidenceTrack.CERTIFICATION,
            predicate_ids=predicates,
            predicate_instructions=tuple(instructions[record.dataset][p] for p in predicates),
            threshold=THRESHOLD,
            requested_model=MODEL,
            maximum_attempts=MAX_RETRIES + 1,
            predicted_input_tokens=predicted_input,
            predicted_output_tokens={2: 47, 3: 69}[len(predicates)],
            predicted_cost_usd=predicted_input * 0.042 / 1_000_000,
        )
        actions.append(action)
        plan.append(
            {
                "action_id": identity,
                "role": "complete_multi_predicate",
                "predicate_group": "|".join(predicates),
            }
        )
    actions.sort(
        key=lambda action: (
            action.workload_id,
            action.record_id,
            len(action.predicate_ids),
            action.predicate_ids,
        )
    )
    universe = AuthorizedActionUniverse(
        experiment_id=EXPERIMENT_ID,
        actions=tuple(actions),
        maximum_actions_per_record=4,
        universe_fingerprint=universe_fingerprint(EXPERIMENT_ID, actions),
    )
    policy = frozen_policy_configuration()
    config: dict[str, Any] = {
        "schema_version": "m5.3.1-frozen-experiment-v1",
        "experiment_id": EXPERIMENT_ID,
        "model": MODEL,
        "threshold": THRESHOLD,
        "selection_seed": SELECTION_SEED,
        "records_per_workload": RECORDS_PER_WORKLOAD,
        "strata_per_workload": {
            "representative_random": REPRESENTATIVE_PER_WORKLOAD,
            "lexical_enrichment": ENRICHED_PER_WORKLOAD,
        },
        "request_budgets": REQUEST_BUDGETS,
        "dollar_budgets": DOLLAR_BUDGETS,
        "replay_seeds": REPLAY_SEEDS,
        "maximum_unique_requests": HARD_REQUEST_LIMIT,
        "maximum_http_attempts": HARD_ATTEMPT_LIMIT,
        "maximum_retries_per_request": MAX_RETRIES,
        "planning_cost_usd": PLANNING_COST_USD,
        "summed_action_cost_forecast_usd": sum(
            action.predicted_cost_usd for action in actions
        ),
        "default_hard_cost_usd": DEFAULT_HARD_COST_USD,
        "request_count": len(actions),
        "policy_configuration": policy,
        "action_universe_fingerprint": universe.universe_fingerprint,
        "independence": independence,
        "workload_config_sha256": file_hash(WORKLOAD_CONFIG),
        "source_dataset_sha256": {
            item["dataset"]: file_hash(DATA_DIRECTORY / f"{item['dataset']}.parquet")
            for item in _workloads()
        },
        "source_code_sha256": {
            path: file_hash(ROOT / path) for path in FROZEN_SOURCE_PATHS
        },
    }
    config["configuration_fingerprint"] = canonical_hash(config)
    roles = {item["action_id"]: item for item in plan}
    plan_rows = [
        {
            **roles[action.action_id],
            "workload_id": action.workload_id,
            "dataset": action.dataset,
            "record_id": action.record_id,
            "stratum": action.source_partition,
            "record_text_sha256": action.record_text_sha256,
            "physical_operator": action.physical_operator.value,
            "predicted_input_tokens": action.predicted_input_tokens,
            "predicted_output_tokens": action.predicted_output_tokens,
            "predicted_cost_usd": action.predicted_cost_usd,
            "maximum_reserved_cost_usd": (
                len(next(row.text for row in records if row.record_id == action.record_id).encode())
                + sum(len(item.encode()) for item in action.predicate_instructions)
                + 2048
            )
            * 0.042
            / 1_000_000,
            "maximum_attempts": action.maximum_attempts,
        }
        for action in actions
    ]
    return config, records, universe, plan_rows
