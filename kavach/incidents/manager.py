"""Deduplicated, reviewable in-memory incident management."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable

from ..behaviours.base import BehaviourEvent
from ..risk.engine import RiskEngine
from .models import (
    CLOSED,
    FALSE_POSITIVE,
    NEW,
    OPEN,
    REVIEWED,
    Incident,
    IncidentLifecycle,
    IncidentStatus,
)


class IncidentManagerError(ValueError):
    """Raised when an incident operation or configuration is invalid."""


class IncidentUpdate:
    """Result of ingesting an event, including whether it was deduplicated."""

    __slots__ = ("created", "deduplicated", "incident")

    def __init__(
        self,
        incident: Incident,
        *,
        created: bool,
        deduplicated: bool,
    ) -> None:
        self.incident = incident
        self.created = bool(created)
        self.deduplicated = bool(deduplicated)

    def to_dict(self) -> dict[str, object]:
        """Return the update and its current incident."""

        return {
            "created": self.created,
            "deduplicated": self.deduplicated,
            "incident": self.incident.to_dict(),
        }


class IncidentManager:
    """Convert behaviour events into stable, bounded in-memory incidents.

    Incident identity is based on event type plus the set of entity IDs. A
    continuing event within the configured windows updates the same incident.
    This manager deliberately does not write a database; persistence is a
    later module.
    """

    def __init__(
        self,
        *,
        risk_engine: RiskEngine | None = None,
        dedup_window_seconds: float = 10.0,
        cooldown_seconds: float = 3.0,
        max_incidents: int = 1000,
        max_event_history: int = 5000,
    ) -> None:
        for name, value in (
            ("dedup_window_seconds", dedup_window_seconds),
            ("cooldown_seconds", cooldown_seconds),
        ):
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise IncidentManagerError(
                    f"{name} must be finite and non-negative"
                )
            setattr(self, name, numeric)
        if int(max_incidents) <= 0 or int(max_event_history) <= 0:
            raise IncidentManagerError("incident bounds must be positive")
        self.risk_engine = risk_engine or RiskEngine()
        self.max_incidents = int(max_incidents)
        self.max_event_history = int(max_event_history)
        self._incidents: dict[str, Incident] = {}
        self._fingerprints: dict[str, tuple[str, tuple[str, ...]]] = {}
        self._event_history: deque[BehaviourEvent] = deque(
            maxlen=self.max_event_history
        )
        self._next_sequence = 1
        self._last_event_timestamp: float | None = None

    @staticmethod
    def _fingerprint(event: BehaviourEvent) -> tuple[str, tuple[str, ...]]:
        return event.type, tuple(sorted(set(event.entities)))

    def _new_id(self) -> str:
        incident_id = f"INC-{self._next_sequence:06d}"
        self._next_sequence += 1
        return incident_id

    def _make_room(self) -> None:
        if len(self._incidents) < self.max_incidents:
            return
        removable = [
            incident
            for incident in self._incidents.values()
            if incident.lifecycle == CLOSED
        ]
        if not removable:
            raise IncidentManagerError(
                "maximum incident capacity reached and no closed incident is removable"
            )
        oldest = min(
            removable,
            key=lambda incident: incident.last_seen_timestamp or incident.timestamp,
        )
        self._incidents.pop(oldest.id, None)
        self._fingerprints.pop(oldest.id, None)

    def _find_open(
        self,
        fingerprint: tuple[str, tuple[str, ...]],
    ) -> Incident | None:
        matches = [
            incident
            for incident_id, incident in self._incidents.items()
            if self._fingerprints.get(incident_id) == fingerprint
            and incident.lifecycle == OPEN
            and incident.status != FALSE_POSITIVE
        ]
        return max(
            matches,
            key=lambda incident: incident.last_seen_timestamp or incident.timestamp,
            default=None,
        )

    @staticmethod
    def _validate_timestamp(
        event: BehaviourEvent,
        previous_timestamp: float | None,
    ) -> None:
        if previous_timestamp is not None and event.timestamp < previous_timestamp:
            raise IncidentManagerError(
                f"event timestamp {event.timestamp} is earlier than the previous "
                f"event timestamp {previous_timestamp}"
            )

    def ingest(
        self,
        event: BehaviourEvent,
        *,
        recent_events: Iterable[BehaviourEvent] = (),
    ) -> IncidentUpdate:
        """Score and ingest one event, merging repeated events when appropriate."""

        if not isinstance(event, BehaviourEvent):
            raise TypeError("event must be a BehaviourEvent")
        self._validate_timestamp(event, self._last_event_timestamp)
        external_events = tuple(recent_events)
        assessment = self.risk_engine.assess(
            event,
            recent_events=tuple(self._event_history) + external_events,
        )
        fingerprint = self._fingerprint(event)
        incident = self._find_open(fingerprint)
        if incident is not None:
            last_seen = incident.last_seen_timestamp or incident.timestamp
            elapsed = max(0.0, event.timestamp - last_seen)
            within_cooldown = elapsed <= self.cooldown_seconds
            within_dedup_window = elapsed <= self.dedup_window_seconds
            if within_cooldown or within_dedup_window:
                incident.last_seen_timestamp = event.timestamp
                incident.occurrence_count += 1
                incident.risk = assessment
                incident.evidence.update(
                    {
                        "latest_event": event.to_dict(),
                        "latest_risk_breakdown": assessment.breakdown.to_dict(),
                        "occurrence_count": incident.occurrence_count,
                        "last_merge_reason": (
                            "cooldown"
                            if within_cooldown
                            else "dedup_window"
                        ),
                    }
                )
                if incident.status == REVIEWED:
                    incident.status = NEW
                self._event_history.append(event)
                self._last_event_timestamp = event.timestamp
                return IncidentUpdate(
                    incident,
                    created=False,
                    deduplicated=True,
                )

        self._make_room()
        incident = Incident(
            id=self._new_id(),
            type=event.type,
            timestamp=event.timestamp,
            entities=event.entities,
            risk=assessment,
            evidence={
                "initial_event": event.to_dict(),
                "latest_event": event.to_dict(),
                "risk_breakdown": assessment.breakdown.to_dict(),
                "occurrence_count": 1,
            },
            status=NEW,
            lifecycle=OPEN,
        )
        self._incidents[incident.id] = incident
        self._fingerprints[incident.id] = fingerprint
        self._event_history.append(event)
        self._last_event_timestamp = event.timestamp
        return IncidentUpdate(incident, created=True, deduplicated=False)

    process = ingest

    def get(self, incident_id: str) -> Incident | None:
        """Return one incident by ID, or None when it is unknown."""

        return self._incidents.get(str(incident_id))

    def list_incidents(
        self,
        *,
        status: IncidentStatus | None = None,
        lifecycle: IncidentLifecycle | None = None,
    ) -> tuple[Incident, ...]:
        """List incidents in creation order with optional state filters."""

        if status is not None and status not in (NEW, REVIEWED, FALSE_POSITIVE):
            raise IncidentManagerError(f"invalid incident status: {status}")
        if lifecycle is not None and lifecycle not in (OPEN, CLOSED):
            raise IncidentManagerError(f"invalid incident lifecycle: {lifecycle}")
        return tuple(
            incident
            for incident in self._incidents.values()
            if (status is None or incident.status == status)
            and (lifecycle is None or incident.lifecycle == lifecycle)
        )

    def _require(self, incident_id: str) -> Incident:
        incident = self.get(incident_id)
        if incident is None:
            raise IncidentManagerError(f"unknown incident: {incident_id}")
        return incident

    def mark_reviewed(self, incident_id: str) -> Incident:
        """Mark an open incident as reviewed without changing its risk."""

        incident = self._require(incident_id)
        if incident.status == FALSE_POSITIVE:
            raise IncidentManagerError(
                "a false positive must be reopened before it can be reviewed"
            )
        incident.status = REVIEWED
        return incident

    def mark_false_positive(self, incident_id: str) -> Incident:
        """Mark an incident as a false positive and close its lifecycle."""

        incident = self._require(incident_id)
        incident.status = FALSE_POSITIVE
        incident.lifecycle = CLOSED
        incident.closed_timestamp = incident.last_seen_timestamp
        return incident

    def close(self, incident_id: str) -> Incident:
        """Close an incident while preserving its review status."""

        incident = self._require(incident_id)
        incident.lifecycle = CLOSED
        incident.closed_timestamp = incident.last_seen_timestamp
        return incident

    def reopen(self, incident_id: str) -> Incident:
        """Reopen an incident and return it to the NEW review queue."""

        incident = self._require(incident_id)
        incident.lifecycle = OPEN
        incident.status = NEW
        incident.closed_timestamp = None
        return incident

    @property
    def incidents(self) -> tuple[Incident, ...]:
        """Return all currently retained incidents."""

        return self.list_incidents()
