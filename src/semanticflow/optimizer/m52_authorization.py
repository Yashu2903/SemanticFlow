from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field


class M52LivePilotScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "m5.2-live-pilot-scope-v1"
    scope_fingerprint: str
    authorization_phrase: str
    model: str = "jev-1.14.0"
    schedule_fingerprint: str
    physical_request_ceiling: int = Field(gt=0)
    retry_inclusive_attempt_ceiling: int = Field(gt=0)
    maximum_retries: int = Field(default=2, ge=0)
    historical_authorization: bool = False
    stage: str = "m5.2-small-live-pilot"


def schedule_fingerprint(schedule: Sequence[Mapping[str, object]]) -> str:
    return hashlib.sha256(
        json.dumps(list(schedule), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_live_authorization(
    scope: M52LivePilotScope,
    *,
    supplied_authorization: str | None,
    schedule: Sequence[Mapping[str, object]],
    requested_model: str,
    requested_maximum_retries: int,
) -> None:
    if scope.historical_authorization or scope.stage != "m5.2-small-live-pilot":
        raise PermissionError("historical or cross-stage authorization cannot be reused")
    if supplied_authorization != scope.authorization_phrase:
        raise PermissionError("exact M5.2 authorization phrase is required")
    if requested_model != scope.model:
        raise PermissionError("requested model differs from the frozen scope")
    if requested_maximum_retries > scope.maximum_retries:
        raise PermissionError("retry policy exceeds the frozen scope")
    if len(schedule) > scope.physical_request_ceiling:
        raise PermissionError("schedule exceeds the physical request ceiling")
    if schedule_fingerprint(schedule) != scope.schedule_fingerprint:
        raise PermissionError("schedule fingerprint mismatch")


class AtomicAttemptBudget:
    """Thread-safe fail-closed reservation; reservations are never silently refunded."""

    def __init__(self, ceiling: int) -> None:
        if ceiling <= 0:
            raise ValueError("attempt ceiling must be positive")
        self.ceiling = ceiling
        self._reserved = 0
        self._lock = threading.Lock()

    @property
    def reserved(self) -> int:
        with self._lock:
            return self._reserved

    def reserve(self, attempts: int = 1) -> int:
        if attempts <= 0:
            raise ValueError("attempt reservation must be positive")
        with self._lock:
            if self._reserved + attempts > self.ceiling:
                raise PermissionError("atomic HTTP attempt budget exhausted")
            self._reserved += attempts
            return self._reserved
