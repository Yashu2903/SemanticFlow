from __future__ import annotations

import threading

from pydantic import BaseModel, ConfigDict, Field


class BudgetSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "m5.2.2-budget-snapshot-v1"
    physical_request_ceiling: int = Field(gt=0)
    http_attempt_ceiling: int = Field(gt=0)
    maximum_retries: int = Field(ge=0)
    reserved_action_ids: tuple[str, ...] = ()
    completed_action_ids: tuple[str, ...] = ()
    ambiguous_action_ids: tuple[str, ...] = ()
    reserved_attempt_capacity: int = Field(default=0, ge=0)
    actual_http_attempts: int = Field(default=0, ge=0)
    named_question_evaluations: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    actual_cost_proxy_usd: float = Field(default=0, ge=0)
    predicted_input_tokens: int = Field(default=0, ge=0)
    reserved_cost_proxy_usd: float = Field(default=0, ge=0)


class BudgetLedger:
    """Thread-safe unique-request and retry-inclusive attempt reservations."""

    def __init__(
        self,
        *,
        physical_request_ceiling: int,
        http_attempt_ceiling: int,
        maximum_retries: int,
        snapshot: BudgetSnapshot | None = None,
    ) -> None:
        self.physical_request_ceiling = physical_request_ceiling
        self.http_attempt_ceiling = http_attempt_ceiling
        self.maximum_retries = maximum_retries
        self._lock = threading.Lock()
        if snapshot is None:
            self._reserved: set[str] = set()
            self._completed: set[str] = set()
            self._ambiguous: set[str] = set()
            self._reserved_attempts = 0
            self._actual_attempts = 0
            self._named = 0
            self._input = 0
            self._output = 0
            self._cost = 0.0
            self._predicted_input = 0
            self._reserved_cost = 0.0
        else:
            if (
                snapshot.physical_request_ceiling != physical_request_ceiling
                or snapshot.http_attempt_ceiling != http_attempt_ceiling
                or snapshot.maximum_retries != maximum_retries
            ):
                raise ValueError("ledger snapshot ceilings differ from protocol")
            self._reserved = set(snapshot.reserved_action_ids)
            self._completed = set(snapshot.completed_action_ids)
            self._ambiguous = set(snapshot.ambiguous_action_ids)
            self._reserved_attempts = snapshot.reserved_attempt_capacity
            self._actual_attempts = snapshot.actual_http_attempts
            self._named = snapshot.named_question_evaluations
            self._input = snapshot.input_tokens
            self._output = snapshot.output_tokens
            self._cost = snapshot.actual_cost_proxy_usd
            self._predicted_input = snapshot.predicted_input_tokens
            self._reserved_cost = snapshot.reserved_cost_proxy_usd
        self._validate()

    def _validate(self) -> None:
        if len(self._reserved) > self.physical_request_ceiling:
            raise ValueError("reserved requests exceed ceiling")
        if self._reserved_attempts > self.http_attempt_ceiling:
            raise ValueError("reserved attempts exceed ceiling")
        if self._actual_attempts > self._reserved_attempts:
            raise ValueError("actual attempts exceed reserved capacity")
        if not self._completed <= self._reserved or not self._ambiguous <= self._reserved:
            raise ValueError("completed or ambiguous action lacks reservation")

    def reserve_action(
        self, action_id: str, *, predicted_input_tokens: int, predicted_cost_usd: float
    ) -> BudgetSnapshot:
        with self._lock:
            if action_id in self._reserved:
                raise PermissionError("action identity is already reserved")
            if len(self._reserved) + 1 > self.physical_request_ceiling:
                raise PermissionError("unique physical request ceiling exhausted")
            capacity = self.maximum_retries + 1
            if self._reserved_attempts + capacity > self.http_attempt_ceiling:
                raise PermissionError("HTTP attempt capacity exhausted")
            self._reserved.add(action_id)
            self._reserved_attempts += capacity
            self._predicted_input += predicted_input_tokens
            self._reserved_cost += predicted_cost_usd
            return self.snapshot()

    def charge_receipt(
        self,
        action_id: str,
        *,
        attempts: int,
        named_question_evaluations: int,
        input_tokens: int,
        output_tokens: int,
        cost_proxy_usd: float,
    ) -> BudgetSnapshot:
        with self._lock:
            if action_id not in self._reserved:
                raise PermissionError("cannot charge an unreserved action")
            if action_id in self._completed:
                raise PermissionError("completed action cannot be charged twice")
            if action_id in self._ambiguous:
                raise PermissionError("ambiguous action requires reconciliation")
            if attempts < 1 or attempts > self.maximum_retries + 1:
                raise ValueError("receipt attempt count violates retry policy")
            if self._actual_attempts + attempts > self._reserved_attempts:
                raise PermissionError("receipt exceeds reserved attempt capacity")
            self._actual_attempts += attempts
            self._named += named_question_evaluations
            self._input += input_tokens
            self._output += output_tokens
            self._cost += cost_proxy_usd
            self._completed.add(action_id)
            return self.snapshot()

    def release_unattempted_reservation(
        self,
        action_id: str,
        *,
        predicted_input_tokens: int,
        predicted_cost_usd: float,
    ) -> BudgetSnapshot:
        """Explicitly release capacity only when durable evidence proves no dispatch."""
        with self._lock:
            if action_id not in self._reserved:
                raise PermissionError("action identity is not reserved")
            if action_id in self._completed or action_id in self._ambiguous:
                raise PermissionError("executed action reservation cannot be released")
            self._reserved.remove(action_id)
            self._reserved_attempts -= self.maximum_retries + 1
            self._predicted_input -= predicted_input_tokens
            self._reserved_cost -= predicted_cost_usd
            return self.snapshot()

    def mark_ambiguous(self, action_id: str, *, attempts: int) -> BudgetSnapshot:
        with self._lock:
            if action_id not in self._reserved:
                raise PermissionError("ambiguous action lacks reservation")
            if action_id in self._completed:
                raise PermissionError("completed action cannot become ambiguous")
            if attempts < 1 or attempts > self.maximum_retries + 1:
                raise ValueError("invalid ambiguous attempt count")
            if self._actual_attempts + attempts > self._reserved_attempts:
                raise PermissionError("ambiguous attempts exceed reservation")
            self._actual_attempts += attempts
            self._ambiguous.add(action_id)
            return self.snapshot()

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            physical_request_ceiling=self.physical_request_ceiling,
            http_attempt_ceiling=self.http_attempt_ceiling,
            maximum_retries=self.maximum_retries,
            reserved_action_ids=tuple(sorted(self._reserved)),
            completed_action_ids=tuple(sorted(self._completed)),
            ambiguous_action_ids=tuple(sorted(self._ambiguous)),
            reserved_attempt_capacity=self._reserved_attempts,
            actual_http_attempts=self._actual_attempts,
            named_question_evaluations=self._named,
            input_tokens=self._input,
            output_tokens=self._output,
            actual_cost_proxy_usd=self._cost,
            predicted_input_tokens=self._predicted_input,
            reserved_cost_proxy_usd=self._reserved_cost,
        )
