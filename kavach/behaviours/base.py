"""Shared contracts and debouncing for KAVACH Module 7 behaviours."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
from numbers import Real
from typing import TypeAlias

from ..intelligence.object_memory import ObjectMemory
from ..intelligence.scene_graph import SceneGraph


class BehaviourError(ValueError):
    """Raised when behaviour state, evidence, or configuration is invalid."""


EvidenceValue: TypeAlias = bool | float | int | str


@dataclass(frozen=True)
class BehaviourContext:
    """Structured state available to every behaviour detector."""

    timestamp: float
    memory: ObjectMemory
    scene_graph: SceneGraph
    frame_width: int | None = None
    frame_height: int | None = None

    def __post_init__(self) -> None:
        timestamp = float(self.timestamp)
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise BehaviourError("context timestamp must be finite and non-negative")
        if not isinstance(self.memory, ObjectMemory):
            raise TypeError("memory must be an ObjectMemory")
        if not isinstance(self.scene_graph, SceneGraph):
            raise TypeError("scene_graph must be a SceneGraph")
        for name in ("frame_width", "frame_height"):
            value = getattr(self, name)
            if value is not None and int(value) <= 0:
                raise BehaviourError(f"{name} must be positive when provided")
        object.__setattr__(self, "timestamp", timestamp)


@dataclass(frozen=True)
class BehaviourEvent:
    """One emitted temporal event with JSON-friendly structured evidence."""

    type: str
    timestamp: float
    entities: tuple[str, ...]
    confidence: float
    evidence: Mapping[str, EvidenceValue]

    def __post_init__(self) -> None:
        if not str(self.type).strip():
            raise BehaviourError("event type cannot be empty")
        timestamp = float(self.timestamp)
        confidence = float(self.confidence)
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise BehaviourError("event timestamp must be finite and non-negative")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise BehaviourError("event confidence must be between 0 and 1")
        entities = tuple(str(entity) for entity in self.entities if str(entity).strip())
        if not entities:
            raise BehaviourError("event must reference at least one entity")
        normalized: dict[str, EvidenceValue] = {}
        for key, value in self.evidence.items():
            if isinstance(value, bool):
                normalized[str(key)] = value
            elif isinstance(value, Real):
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise BehaviourError("numeric event evidence must be finite")
                normalized[str(key)] = numeric
            elif isinstance(value, str):
                normalized[str(key)] = value
            else:
                raise BehaviourError(
                    "event evidence values must be booleans, numbers, or strings"
                )
        if not normalized:
            raise BehaviourError("event must include structured evidence")
        object.__setattr__(self, "type", str(self.type))
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "entities", entities)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "evidence", normalized)

    def to_dict(self) -> dict[str, object]:
        """Return an event payload suitable for JSON serialization."""

        return {
            "type": self.type,
            "timestamp": self.timestamp,
            "entities": list(self.entities),
            "confidence": self.confidence,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class BehaviourCandidate:
    """An internal condition that can become an event after debounce checks."""

    event_type: str
    timestamp: float
    entities: tuple[str, ...]
    confidence: float
    evidence: Mapping[str, EvidenceValue]
    key: tuple[str, ...] | None = None


class BehaviourDetector(ABC):
    """Base class implementing rising-edge emission and cooldown."""

    event_type: str = "BEHAVIOUR_EVENT"

    def __init__(
        self,
        *,
        debounce_seconds: float = 0.0,
        cooldown_seconds: float = 3.0,
    ) -> None:
        for name, value in (
            ("debounce_seconds", debounce_seconds),
            ("cooldown_seconds", cooldown_seconds),
        ):
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise BehaviourError(f"{name} must be finite and non-negative")
            setattr(self, name, numeric)
        self._condition_since: dict[tuple[str, ...], float] = {}
        self._active_keys: set[tuple[str, ...]] = set()
        self._last_emitted: dict[tuple[str, ...], float] = {}
        self._last_timestamp: float | None = None

    @abstractmethod
    def evaluate(self, context: BehaviourContext) -> Iterable[BehaviourCandidate]:
        """Return conditions that are true in the current structured state."""

    def detect(self, context: BehaviourContext) -> list[BehaviourEvent]:
        """Evaluate once and emit only debounced, non-cooldown events."""

        if (
            self._last_timestamp is not None
            and context.timestamp < self._last_timestamp
        ):
            raise BehaviourError(
                f"timestamp {context.timestamp} is earlier than previous detector "
                f"timestamp {self._last_timestamp}"
            )
        candidates = tuple(self.evaluate(context))
        current_keys: set[tuple[str, ...]] = set()
        events: list[BehaviourEvent] = []
        for candidate in candidates:
            if candidate.event_type != self.event_type:
                raise BehaviourError(
                    f"{self.__class__.__name__} returned {candidate.event_type!r}; "
                    f"expected {self.event_type!r}"
                )
            key = tuple(candidate.key or candidate.entities)
            if not key:
                raise BehaviourError("behaviour candidate key cannot be empty")
            if key in current_keys:
                raise BehaviourError(f"duplicate candidate key: {key}")
            current_keys.add(key)
            self._condition_since.setdefault(key, context.timestamp)
            if key in self._active_keys:
                continue
            condition_duration = context.timestamp - self._condition_since[key]
            if condition_duration < self.debounce_seconds:
                continue
            last_emitted = self._last_emitted.get(key)
            if (
                last_emitted is not None
                and context.timestamp - last_emitted < self.cooldown_seconds
            ):
                continue
            event = BehaviourEvent(
                type=candidate.event_type,
                timestamp=candidate.timestamp,
                entities=candidate.entities,
                confidence=candidate.confidence,
                evidence=candidate.evidence,
            )
            events.append(event)
            self._active_keys.add(key)
            self._last_emitted[key] = context.timestamp

        for key in tuple(self._condition_since):
            if key not in current_keys:
                self._condition_since.pop(key, None)
                self._active_keys.discard(key)
        self._last_timestamp = context.timestamp
        return events

    def reset(self) -> None:
        """Clear debounce, active-condition, cooldown, and timestamp state."""

        self._condition_since.clear()
        self._active_keys.clear()
        self._last_emitted.clear()
        self._last_timestamp = None


def normalized_class_name(value: object) -> str:
    """Normalize class spelling for configurable behaviour matching."""

    return "_".join(str(value).strip().lower().replace("-", " ").split())


def class_names(value: object, default: tuple[str, ...]) -> tuple[str, ...]:
    """Normalize a config list or comma-separated class string."""

    if value is None:
        return tuple(normalized_class_name(item) for item in default)
    if isinstance(value, str):
        values = value.split(",")
    else:
        try:
            values = tuple(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise BehaviourError("class list must be a sequence or string") from exc
    normalized = tuple(normalized_class_name(item) for item in values if str(item).strip())
    return normalized or tuple(normalized_class_name(item) for item in default)
