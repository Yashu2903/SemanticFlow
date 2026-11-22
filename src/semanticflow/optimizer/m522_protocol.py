from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from semanticflow.optimizer.m52_evidence import EvidenceTrack
from semanticflow.optimizer.m522_action_universe import (
    AuthorizedActionUniverse,
    AuthorizedLiveAction,
    LiveMeasurementType,
    PhysicalOperatorType,
    action_identity,
    universe_fingerprint,
)
from semanticflow.optimizer.m522_experiment_objective import (
    ObjectivePolicy,
    research_instrumentation_policy,
)

WORKLOAD_CONFIG = Path("configs/milestone_5_1/workloads.yaml")
PRICE_PER_MILLION_INPUT_TOKENS = 0.042

PILOT_WORKLOADS = (
    "m51_banking_card_unauthorized_refund",
    "m51_beauty_failure_usability",
    "m51_sms_prize_urgency_contact",
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class PilotAProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "m5.2.2-pilot-a-protocol-v1"
    experiment_id: str = "semanticflow-m5.2.2-pilot-a-instrumentation"
    objective: ObjectivePolicy
    scientific_objective: str
    statistical_non_certification_statement: str
    workload_ids: tuple[str, ...]
    record_partition: str = "development"
    representative_records_per_workload: int = Field(default=20, ge=1)
    discovery_records_per_workload: int = Field(default=10, ge=1)
    model: str = "jev-1.14.0"
    threshold: float = Field(default=0.5, ge=0, le=1)
    cache_enabled: bool = False
    concurrency: int = Field(default=1, ge=1)
    maximum_retries: int = Field(default=2, ge=0)
    physical_request_ceiling: int = Field(default=360, gt=0)
    http_attempt_ceiling: int = Field(default=1080, gt=0)
    expected_physical_requests: int = Field(default=280, gt=0)
    expected_named_question_evaluations: int = Field(default=430, gt=0)
    expected_cost_proxy_usd: float = Field(default=0.0040551, ge=0)
    conservative_cost_proxy_usd: float = Field(default=0.006708384375, ge=0)
    output_directory: str = "results/milestone_5_2_2/pilot_a_live"
    policy_version: str = "m5.2.2-cost-aware-instrumentation-v1"
    seed: int = 5221
    near_threshold_margin: float = Field(default=0.15, gt=0, le=0.5)
    protocol_fingerprint: str

    @model_validator(mode="after")
    def validate_protocol(self) -> PilotAProtocol:
        if self.objective.permits_production_deployment:
            raise ValueError("Pilot A cannot authorize production deployment")
        if self.expected_physical_requests > self.physical_request_ceiling:
            raise ValueError("expected actions exceed physical request ceiling")
        if self.physical_request_ceiling * (self.maximum_retries + 1) > self.http_attempt_ceiling:
            raise ValueError("attempt ceiling cannot cover the physical ceiling retry policy")
        expected = protocol_fingerprint(
            self.model_dump(mode="json", exclude={"protocol_fingerprint"})
        )
        if self.protocol_fingerprint != expected:
            raise ValueError("protocol fingerprint mismatch")
        return self


class PilotAAuthorizationManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "m5.2.2-pilot-a-authorization-proposal-v1"
    experiment_id: str
    protocol_fingerprint: str
    action_universe_fingerprint: str
    policy_fingerprint: str
    model: str
    physical_request_ceiling: int
    http_attempt_ceiling: int
    output_directory: str
    required_authorization_contract: str
    paid_calls_authorized: bool = False
    countersigned_by: str | None = None
    historical_authorization_reuse_allowed: bool = False
    scope_fingerprint: str


def protocol_fingerprint(payload: dict[str, Any]) -> str:
    return canonical_hash(payload)


def build_protocol() -> PilotAProtocol:
    payload: dict[str, Any] = {
        "objective": research_instrumentation_policy(),
        "scientific_objective": (
            "Validate bounded response-adaptive acquisition, paid-screen accounting, evidence "
            "isolation, provider contracts, and restart reconciliation."
        ),
        "statistical_non_certification_statement": (
            "Twenty representative records per workload cannot formally qualify batching under "
            "M5.2 or M5.2.1 confidence requirements."
        ),
        "workload_ids": PILOT_WORKLOADS,
    }
    defaults = PilotAProtocol.model_construct(**payload, protocol_fingerprint="")
    full = defaults.model_dump(mode="json", exclude={"protocol_fingerprint"})
    return PilotAProtocol(**full, protocol_fingerprint=protocol_fingerprint(full))


def _load_config() -> dict[str, Any]:
    return yaml.safe_load(WORKLOAD_CONFIG.read_text("utf-8"))


def _records(dataset_path: Path, partition: str) -> list[dict[str, Any]]:
    table = pq.read_table(
        dataset_path,
        columns=["review_id", "review_text", "partition", "text_sha256"],
    )
    table = table.filter(pc.equal(table["partition"], partition))
    return sorted(table.to_pylist(), key=lambda row: str(row["review_id"]))


def _predicted_tokens(predicate_count: int, *, screen: bool = False) -> tuple[int, int]:
    del screen
    # Rounded M5.1 Stage D request means by named-question count.
    historical = {1: (325, 35), 2: (364, 47), 3: (398, 69), 4: (430, 89)}
    if predicate_count not in historical:
        raise ValueError("Pilot A lacks historical token support for this group size")
    return historical[predicate_count]


def _make_action(
    *,
    workload: dict[str, Any],
    dataset: str,
    record: dict[str, Any],
    measurement_type: LiveMeasurementType,
    operator: PhysicalOperatorType,
    predicate_ids: tuple[str, ...],
    instructions: tuple[str, ...],
    track: EvidenceTrack,
    protocol: PilotAProtocol,
    prerequisites: tuple[str, ...] = (),
) -> AuthorizedLiveAction:
    identity_payload = {
        "experiment": protocol.experiment_id,
        "workload": workload["id"],
        "record": record["review_id"],
        "measurement_type": measurement_type.value,
        "operator": operator.value,
        "predicates": predicate_ids,
    }
    input_tokens, output_tokens = _predicted_tokens(
        len(predicate_ids), screen=measurement_type == LiveMeasurementType.DISCOVERY_SCREEN
    )
    return AuthorizedLiveAction(
        action_id=action_identity(identity_payload),
        workload_id=workload["id"],
        dataset=dataset,
        source_partition=protocol.record_partition,
        record_id=record["review_id"],
        record_text_sha256=record["text_sha256"],
        measurement_type=measurement_type,
        physical_operator=operator,
        track=track,
        predicate_ids=predicate_ids,
        predicate_instructions=instructions,
        threshold=protocol.threshold,
        requested_model=protocol.model,
        prerequisite_action_ids=prerequisites,
        maximum_attempts=protocol.maximum_retries + 1,
        predicted_input_tokens=input_tokens,
        predicted_output_tokens=output_tokens,
        predicted_cost_usd=input_tokens / 1_000_000 * PRICE_PER_MILLION_INPUT_TOKENS,
    )


def build_action_universe(protocol: PilotAProtocol) -> AuthorizedActionUniverse:
    config = _load_config()
    workloads = {row["id"]: row for row in config["workloads"]}
    actions: list[AuthorizedLiveAction] = []
    for workload_id in protocol.workload_ids:
        workload = workloads[workload_id]
        dataset = workload["dataset"]
        dataset_config = config["datasets"][dataset]
        predicates = dataset_config["predicates"]
        predicate_ids = tuple(workload["predicate_ids"])
        instructions = tuple(predicates[item] for item in predicate_ids)
        records = _records(Path(dataset_config["path"]), protocol.record_partition)
        needed = (
            protocol.representative_records_per_workload + protocol.discovery_records_per_workload
        )
        if len(records) < needed:
            raise ValueError(f"insufficient eligible records for {workload_id}")
        representative = records[: protocol.representative_records_per_workload]
        discovery = records[protocol.representative_records_per_workload : needed]
        for record in representative:
            reference_ids: list[str] = []
            for predicate_id, instruction in zip(predicate_ids, instructions, strict=True):
                action = _make_action(
                    workload=workload,
                    dataset=dataset,
                    record=record,
                    measurement_type=LiveMeasurementType.REPRESENTATIVE_REFERENCE,
                    operator=PhysicalOperatorType.SINGLE,
                    predicate_ids=(predicate_id,),
                    instructions=(instruction,),
                    track=EvidenceTrack.CERTIFICATION,
                    protocol=protocol,
                )
                actions.append(action)
                reference_ids.append(action.action_id)
            actions.append(
                _make_action(
                    workload=workload,
                    dataset=dataset,
                    record=record,
                    measurement_type=LiveMeasurementType.REPRESENTATIVE_CANDIDATE,
                    operator=PhysicalOperatorType.MULTI_DECISION,
                    predicate_ids=predicate_ids,
                    instructions=instructions,
                    track=EvidenceTrack.CERTIFICATION,
                    protocol=protocol,
                    prerequisites=tuple(reference_ids),
                )
            )
        for record in discovery:
            screen = _make_action(
                workload=workload,
                dataset=dataset,
                record=record,
                measurement_type=LiveMeasurementType.DISCOVERY_SCREEN,
                operator=PhysicalOperatorType.SINGLE,
                predicate_ids=(predicate_ids[0],),
                instructions=(instructions[0],),
                track=EvidenceTrack.DISCOVERY,
                protocol=protocol,
            )
            actions.append(screen)
            actions.append(
                _make_action(
                    workload=workload,
                    dataset=dataset,
                    record=record,
                    measurement_type=LiveMeasurementType.DISCOVERY_FOLLOWUP,
                    operator=PhysicalOperatorType.MULTI_DECISION,
                    predicate_ids=predicate_ids,
                    instructions=instructions,
                    track=EvidenceTrack.DISCOVERY,
                    protocol=protocol,
                    prerequisites=(screen.action_id,),
                )
            )
    ordered = tuple(sorted(actions, key=lambda action: action.action_id))
    return AuthorizedActionUniverse(
        experiment_id=protocol.experiment_id,
        actions=ordered,
        maximum_actions_per_record=4,
        universe_fingerprint=universe_fingerprint(protocol.experiment_id, ordered),
    )


def load_authorized_record_texts(
    protocol: PilotAProtocol, universe: AuthorizedActionUniverse
) -> dict[str, str]:
    config = _load_config()
    allowed = {action.record_id: action.record_text_sha256 for action in universe.actions}
    result: dict[str, str] = {}
    datasets = {action.dataset for action in universe.actions}
    for dataset in datasets:
        path = Path(config["datasets"][dataset]["path"])
        for record in _records(path, protocol.record_partition):
            record_id = str(record["review_id"])
            if record_id not in allowed:
                continue
            if record["text_sha256"] != allowed[record_id]:
                raise ValueError("authorized record text fingerprint changed")
            result[record_id] = str(record["review_text"])
    if set(result) != set(allowed):
        raise ValueError("authorized record texts are incomplete")
    return result


def policy_fingerprint(protocol: PilotAProtocol) -> str:
    return canonical_hash(
        {
            "version": protocol.policy_version,
            "seed": protocol.seed,
            "near_threshold_margin": protocol.near_threshold_margin,
            "tie_break": "score_desc_action_id_asc",
            "stopping": "all_representative_and_screens_complete_then_eligible_followups",
        }
    )


def build_authorization_proposal(
    protocol: PilotAProtocol, universe: AuthorizedActionUniverse
) -> PilotAAuthorizationManifest:
    policy_hash = policy_fingerprint(protocol)
    scope = {
        "experiment_id": protocol.experiment_id,
        "protocol_fingerprint": protocol.protocol_fingerprint,
        "action_universe_fingerprint": universe.universe_fingerprint,
        "policy_fingerprint": policy_hash,
        "model": protocol.model,
        "physical_request_ceiling": protocol.physical_request_ceiling,
        "http_attempt_ceiling": protocol.http_attempt_ceiling,
        "output_directory": protocol.output_directory,
    }
    scope_hash = canonical_hash(scope)
    return PilotAAuthorizationManifest(
        **scope,
        required_authorization_contract=("AUTHORIZE M5.2 SMALL LIVE PILOT " + scope_hash),
        paid_calls_authorized=False,
        scope_fingerprint=scope_hash,
    )


def validate_live_manifest(
    manifest: PilotAAuthorizationManifest,
    *,
    protocol: PilotAProtocol,
    universe: AuthorizedActionUniverse,
    supplied_authorization: str | None,
    output_directory: str,
) -> None:
    expected = build_authorization_proposal(protocol, universe)
    for field in (
        "experiment_id",
        "protocol_fingerprint",
        "action_universe_fingerprint",
        "policy_fingerprint",
        "model",
        "physical_request_ceiling",
        "http_attempt_ceiling",
        "output_directory",
        "required_authorization_contract",
        "scope_fingerprint",
    ):
        if getattr(manifest, field) != getattr(expected, field):
            raise PermissionError(f"authorization scope mismatch: {field}")
    if output_directory != manifest.output_directory:
        raise PermissionError("live output directory differs from authorized scope")
    if manifest.historical_authorization_reuse_allowed:
        raise PermissionError("historical authorization reuse is forbidden")
    if not manifest.paid_calls_authorized or not manifest.countersigned_by:
        raise PermissionError("a countersigned live manifest is required")
    if supplied_authorization != manifest.required_authorization_contract:
        raise PermissionError("exact scoped authorization contract is required")
