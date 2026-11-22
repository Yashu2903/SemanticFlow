from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field


class FrozenCostPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "m5.2-cost-prediction-v1"
    prediction_id: str
    model_version: str
    estimator_version: str
    input_measurement_ids: tuple[str, ...]
    available_data_partition: str
    creation_checkpoint: int = Field(ge=0)
    dataset_fingerprint: str
    workload_fingerprint: str
    candidate_plan_fingerprint: str
    predicted_cost_usd: float = Field(ge=0)
    predicted_cost_lower_usd: float = Field(ge=0)
    predicted_cost_upper_usd: float = Field(ge=0)
    predicted_requests: float = Field(ge=0)
    predicted_stage_reachability: tuple[float, ...]
    stage_mean_costs_usd: tuple[float, ...]
    uses_future_evaluation_measurements: bool = False


class FrozenOperatorStatistics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operator_id: str
    measurement_ids: tuple[str, ...]
    mean_cost_usd: float = Field(ge=0)
    cost_standard_error_usd: float = Field(ge=0)
    pass_rate: float = Field(ge=0, le=1)
    sample_count: int = Field(ge=1)


class M52CostEstimator:
    """Partition-locked baseline estimator with persisted input provenance."""

    estimator_version = "m5.2-selection-locked-unconditional-v1"

    def __init__(
        self,
        rows: Sequence[Mapping[str, object]],
        *,
        allowed_partition: str,
        model_version: str = "jev-1.14.0",
    ) -> None:
        if not rows:
            raise ValueError("cost estimator requires historical measurements")
        invalid = sorted(
            {str(row["partition"]) for row in rows if row["partition"] != allowed_partition}
        )
        if invalid:
            raise ValueError(f"future or unauthorized partitions exposed to estimator: {invalid}")
        self.allowed_partition = allowed_partition
        self.model_version = model_version
        grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["operator_id"])].append(row)
        self.statistics: dict[str, FrozenOperatorStatistics] = {}
        for operator_id, values in grouped.items():
            costs = [float(row["cost_usd"]) for row in values]
            standard_error = (
                statistics.stdev(costs) / math.sqrt(len(costs)) if len(costs) > 1 else 0.0
            )
            self.statistics[operator_id] = FrozenOperatorStatistics(
                operator_id=operator_id,
                measurement_ids=tuple(sorted(str(row["request_id"]) for row in values)),
                mean_cost_usd=statistics.mean(costs),
                cost_standard_error_usd=standard_error,
                pass_rate=statistics.mean(bool(row["group_passed"]) for row in values),
                sample_count=len(values),
            )

    def predict(
        self,
        plan: Mapping[str, object],
        *,
        record_count: int,
        checkpoint: int,
        dataset_fingerprint: str,
        workload_fingerprint: str,
    ) -> FrozenCostPrediction:
        if record_count <= 0:
            raise ValueError("record count must be positive")
        reach = 1.0
        cost = variance = requests = 0.0
        reachability: list[float] = []
        means: list[float] = []
        measurement_ids: set[str] = set()
        groups = plan["groups"]
        if not isinstance(groups, list):
            raise TypeError("plan groups must be a list")
        for group in groups:
            operator_id = str(group["operator_id"])
            if operator_id not in self.statistics:
                raise ValueError(f"operator {operator_id} lacks permitted cost evidence")
            stats = self.statistics[operator_id]
            entrants = record_count * reach
            cost += entrants * stats.mean_cost_usd
            variance += (entrants * stats.cost_standard_error_usd) ** 2
            requests += entrants
            reachability.append(reach)
            means.append(stats.mean_cost_usd)
            measurement_ids.update(stats.measurement_ids)
            reach *= stats.pass_rate
        uncertainty = 1.96 * math.sqrt(variance)
        payload = {
            "estimator_version": self.estimator_version,
            "partition": self.allowed_partition,
            "checkpoint": checkpoint,
            "plan_fingerprint": plan["plan_fingerprint"],
            "record_count": record_count,
            "cost": cost,
            "inputs": sorted(measurement_ids),
        }
        prediction_id = (
            "m52pred_"
            + hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()[:24]
        )
        return FrozenCostPrediction(
            prediction_id=prediction_id,
            model_version=self.model_version,
            estimator_version=self.estimator_version,
            input_measurement_ids=tuple(sorted(measurement_ids)),
            available_data_partition=self.allowed_partition,
            creation_checkpoint=checkpoint,
            dataset_fingerprint=dataset_fingerprint,
            workload_fingerprint=workload_fingerprint,
            candidate_plan_fingerprint=str(plan["plan_fingerprint"]),
            predicted_cost_usd=cost,
            predicted_cost_lower_usd=max(0.0, cost - uncertainty),
            predicted_cost_upper_usd=cost + uncertainty,
            predicted_requests=requests,
            predicted_stage_reachability=tuple(reachability),
            stage_mean_costs_usd=tuple(means),
        )
