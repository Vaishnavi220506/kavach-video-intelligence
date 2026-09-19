"""Serializable incident models and explicit review/lifecycle states."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from ..risk.engine import RiskAssessment


NEW = "NEW"
REVIEWED = "REVIEWED"
FALSE_POSITIVE = "FALSE_POSITIVE"
OPEN = "OPEN"
CLOSED = "CLOSED"

IncidentStatus = Literal["NEW", "REVIEWED", "FALSE_POSITIVE"]
IncidentLifecycle = Literal["OPEN", "CLOSED"]
REVIEW_STATUSES = (NEW, REVIEWED, FALSE_POSITIVE)
LIFECYCLE_STATES = (OPEN, CLOSED)


class IncidentModelError(ValueError):
    """Raised when an incident model is invalid."""


@dataclass
class Incident:
    """A stable in-memory incident assembled from one or more events."""

    id: str
    type: str
    timestamp: float
    entities: tuple[str, ...]
    risk: RiskAssessment
    evidence: dict[str, object]
    status: IncidentStatus = NEW
    lifecycle: IncidentLifecycle = OPEN
    last_seen_timestamp: float | None = None
    occurrence_count: int = 1
    closed_timestamp: float | None = None

    def __post_init__(self) -> None:
        if not str(self.id).strip() or not str(self.type).strip():
            raise IncidentModelError("incident id and type cannot be empty")
        timestamp = float(self.timestamp)
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise IncidentModelError("incident timestamp must be finite and non-negative")
        entities = tuple(str(entity) for entity in self.entities if str(entity).strip())
        if not entities:
            raise IncidentModelError("incident must reference at least one entity")
        if not isinstance(self.risk, RiskAssessment):
            raise TypeError("risk must be a RiskAssessment")
        if self.status not in REVIEW_STATUSES:
            raise IncidentModelError(f"invalid incident review status: {self.status}")
        if self.lifecycle not in LIFECYCLE_STATES:
            raise IncidentModelError(f"invalid incident lifecycle: {self.lifecycle}")
        if int(self.occurrence_count) <= 0:
            raise IncidentModelError("occurrence_count must be positive")
        latest = timestamp if self.last_seen_timestamp is None else float(
            self.last_seen_timestamp
        )
        if not math.isfinite(latest) or latest < timestamp:
            raise IncidentModelError(
                "last_seen_timestamp must be finite and no earlier than timestamp"
            )
        if self.closed_timestamp is not None:
            closed = float(self.closed_timestamp)
            if not math.isfinite(closed) or closed < latest:
                raise IncidentModelError(
                    "closed_timestamp must be finite and no earlier than last seen"
                )
            self.closed_timestamp = closed
        if self.lifecycle == CLOSED and self.closed_timestamp is None:
            self.closed_timestamp = latest
        self.id = str(self.id)
        self.type = str(self.type)
        self.timestamp = timestamp
        self.entities = entities
        self.evidence = dict(self.evidence)
        self.last_seen_timestamp = latest
        self.occurrence_count = int(self.occurrence_count)

    def to_dict(self) -> dict[str, object]:
        """Return the incident with risk and evidence ready for serialization."""

        return {
            "id": self.id,
            "type": self.type,
            "timestamp": self.timestamp,
            "last_seen_timestamp": self.last_seen_timestamp,
            "entities": list(self.entities),
            "risk": self.risk.to_dict(),
            "evidence": dict(self.evidence),
            "status": self.status,
            "lifecycle": self.lifecycle,
            "occurrence_count": self.occurrence_count,
            "closed_timestamp": self.closed_timestamp,
        }
