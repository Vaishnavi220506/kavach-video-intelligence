"""Transparent risk scoring from structured KAVACH behaviour events.

This module intentionally contains no model inference or language-model
logic. It turns the evidence already produced by Module 7 into a bounded,
inspectable policy score.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import TypeAlias

from ..behaviours.base import BehaviourEvent

CONFIG_PATH = Path(__file__).with_name("config.yaml")

LOW = "LOW"
MEDIUM = "MEDIUM"
HIGH = "HIGH"
CRITICAL = "CRITICAL"
RISK_CATEGORIES = (LOW, MEDIUM, HIGH, CRITICAL)

SEVERITY = "severity"
DURATION = "duration"
MOTION_INTENSITY = "motion_intensity"
SPATIAL_CONTEXT = "spatial_context"
REPEAT_FREQUENCY = "repeat_frequency"
DETECTION_CONFIDENCE = "detection_confidence"
RISK_COMPONENTS = (
    SEVERITY,
    DURATION,
    MOTION_INTENSITY,
    SPATIAL_CONTEXT,
    REPEAT_FREQUENCY,
    DETECTION_CONFIDENCE,
)

NumericMapping: TypeAlias = Mapping[str, float]


class RiskEngineError(ValueError):
    """Base class for invalid risk inputs or scoring operations."""


class RiskConfigurationError(RiskEngineError):
    """Raised when a risk policy is missing or inconsistent."""


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RiskConfigurationError(f"{label} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise RiskConfigurationError(f"{label} must be finite")
    return converted


def _clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, float(value)))


def _normalized_name(value: object) -> str:
    return "_".join(str(value).strip().upper().replace("-", " ").split())


@dataclass(frozen=True)
class RiskBreakdown:
    """Normalized component values, weights, and their contributions."""

    components: Mapping[str, float]
    weights: Mapping[str, float]
    weighted_contributions: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        components = {
            name: _clamp(_finite_number(self.components.get(name, 0.0), name))
            for name in RISK_COMPONENTS
        }
        weights = {
            name: _finite_number(self.weights.get(name, 0.0), f"weight {name}")
            for name in RISK_COMPONENTS
        }
        if any(value < 0.0 for value in weights.values()):
            raise RiskConfigurationError("risk weights cannot be negative")
        total_weight = sum(weights.values())
        if not math.isclose(total_weight, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise RiskConfigurationError(
                f"risk weights must sum to 1.0, got {total_weight}"
            )
        contributions = {
            name: components[name] * weights[name] for name in RISK_COMPONENTS
        }
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "weighted_contributions", contributions)

    @property
    def score(self) -> float:
        """Return the weighted score on the 0–100 scale."""

        return _clamp(sum(self.weighted_contributions.values()))

    def to_dict(self) -> dict[str, object]:
        """Return an explanation-friendly JSON-compatible representation."""

        return {
            "components": dict(self.components),
            "weights": dict(self.weights),
            "weighted_contributions": dict(self.weighted_contributions),
            "score": self.score,
        }


@dataclass(frozen=True)
class RiskAssessment:
    """Risk result kept separate from the behaviour detector confidence."""

    event_type: str
    score: float
    category: str
    detection_confidence: float
    breakdown: RiskBreakdown

    def __post_init__(self) -> None:
        score = _finite_number(self.score, "risk score")
        confidence = _finite_number(
            self.detection_confidence,
            "detection confidence",
        )
        if not 0.0 <= score <= 100.0:
            raise RiskEngineError("risk score must be between 0 and 100")
        if not 0.0 <= confidence <= 1.0:
            raise RiskEngineError("detection confidence must be between 0 and 1")
        if self.category not in RISK_CATEGORIES:
            raise RiskEngineError(f"unknown risk category: {self.category!r}")
        if not isinstance(self.breakdown, RiskBreakdown):
            raise TypeError("breakdown must be a RiskBreakdown")
        object.__setattr__(self, "event_type", str(self.event_type))
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "detection_confidence", confidence)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible assessment."""

        return {
            "event_type": self.event_type,
            "score": self.score,
            "category": self.category,
            "detection_confidence": self.detection_confidence,
            "breakdown": self.breakdown.to_dict(),
        }


class RiskCategory:
    """Named risk categories used by the default policy."""

    LOW = LOW
    MEDIUM = MEDIUM
    HIGH = HIGH
    CRITICAL = CRITICAL


@dataclass(frozen=True)
class RiskConfig:
    """Validated risk policy and normalization references."""

    weights: NumericMapping
    category_thresholds: NumericMapping
    severity_by_event: NumericMapping
    default_severity: float = 40.0
    duration_reference_seconds: float = 5.0
    motion_reference_px_per_second: float = 500.0
    displacement_reference_px: float = 250.0
    spatial_distance_reference_px: float = 150.0
    repeat_reference_events: int = 3
    repeat_window_seconds: float = 30.0

    def __post_init__(self) -> None:
        weights = {
            name: _finite_number(self.weights.get(name, 0.0), f"weight {name}")
            for name in RISK_COMPONENTS
        }
        total_weight = sum(weights.values())
        if any(value < 0.0 for value in weights.values()) or not math.isclose(
            total_weight,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise RiskConfigurationError(
                f"risk weights must be non-negative and sum to 1.0, got {total_weight}"
            )

        thresholds = {
            category: _finite_number(
                self.category_thresholds.get(category, -1.0),
                f"threshold {category}",
            )
            for category in RISK_CATEGORIES
        }
        if thresholds[LOW] < 0.0 or any(
            thresholds[RISK_CATEGORIES[index]]
            > thresholds[RISK_CATEGORIES[index + 1]]
            for index in range(len(RISK_CATEGORIES) - 1)
        ):
            raise RiskConfigurationError(
                "risk category thresholds must be ordered from LOW to CRITICAL"
            )
        if thresholds[CRITICAL] > 100.0:
            raise RiskConfigurationError("CRITICAL threshold cannot exceed 100")

        severities = {
            _normalized_name(key): _clamp(
                _finite_number(value, f"severity {key}")
            )
            for key, value in self.severity_by_event.items()
        }
        for name, value in (
            ("default_severity", self.default_severity),
            ("duration_reference_seconds", self.duration_reference_seconds),
            ("motion_reference_px_per_second", self.motion_reference_px_per_second),
            ("displacement_reference_px", self.displacement_reference_px),
            ("spatial_distance_reference_px", self.spatial_distance_reference_px),
            ("repeat_window_seconds", self.repeat_window_seconds),
        ):
            numeric = _finite_number(value, name)
            if name == "default_severity":
                if not 0.0 <= numeric <= 100.0:
                    raise RiskConfigurationError(
                        "default_severity must be between 0 and 100"
                    )
            elif numeric <= 0.0:
                raise RiskConfigurationError(f"{name} must be positive")
        if int(self.repeat_reference_events) <= 0:
            raise RiskConfigurationError("repeat_reference_events must be positive")

        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "category_thresholds", thresholds)
        object.__setattr__(self, "severity_by_event", severities)
        object.__setattr__(self, "default_severity", float(self.default_severity))
        object.__setattr__(
            self,
            "duration_reference_seconds",
            float(self.duration_reference_seconds),
        )
        object.__setattr__(
            self,
            "motion_reference_px_per_second",
            float(self.motion_reference_px_per_second),
        )
        object.__setattr__(
            self,
            "displacement_reference_px",
            float(self.displacement_reference_px),
        )
        object.__setattr__(
            self,
            "spatial_distance_reference_px",
            float(self.spatial_distance_reference_px),
        )
        object.__setattr__(
            self,
            "repeat_reference_events",
            int(self.repeat_reference_events),
        )
        object.__setattr__(
            self,
            "repeat_window_seconds",
            float(self.repeat_window_seconds),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> RiskConfig:
        """Build a validated policy from the YAML-shaped mapping."""

        scoring = payload.get("scoring", {})
        if not isinstance(scoring, Mapping):
            raise RiskConfigurationError("scoring must be a mapping")
        weights = scoring.get("weights", {})
        thresholds = scoring.get("category_thresholds", {})
        severities = payload.get("severity_by_event", {})
        if not isinstance(weights, Mapping) or not isinstance(thresholds, Mapping):
            raise RiskConfigurationError(
                "scoring.weights and scoring.category_thresholds must be mappings"
            )
        if not isinstance(severities, Mapping):
            raise RiskConfigurationError("severity_by_event must be a mapping")
        return cls(
            weights={str(key): value for key, value in weights.items()},  # type: ignore[dict-item]
            category_thresholds={
                str(key).upper(): value for key, value in thresholds.items()
            },  # type: ignore[dict-item]
            severity_by_event={
                str(key): value for key, value in severities.items()
            },  # type: ignore[dict-item]
            default_severity=severities.get("DEFAULT", 40.0),
            duration_reference_seconds=scoring.get(
                "duration_reference_seconds",
                5.0,
            ),
            motion_reference_px_per_second=scoring.get(
                "motion_reference_px_per_second",
                500.0,
            ),
            displacement_reference_px=scoring.get(
                "displacement_reference_px",
                250.0,
            ),
            spatial_distance_reference_px=scoring.get(
                "spatial_distance_reference_px",
                150.0,
            ),
            repeat_reference_events=scoring.get("repeat_reference_events", 3),
            repeat_window_seconds=scoring.get("repeat_window_seconds", 30.0),
        )


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, object]:
    """Load the default YAML risk policy without importing YAML at module load."""

    config_path = Path(path)
    if not config_path.is_file():
        raise RiskConfigurationError(f"risk config does not exist: {config_path}")
    try:
        import yaml
    except ImportError as exc:
        raise RiskConfigurationError(
            "PyYAML is required to load the KAVACH risk configuration"
        ) from exc
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise RiskConfigurationError(
            f"could not read risk config {config_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RiskConfigurationError("risk config root must be a mapping")
    return payload


class RiskEngine:
    """Score BehaviourEvent objects using explicit, inspectable heuristics."""

    def __init__(self, config: RiskConfig | Mapping[str, object] | None = None) -> None:
        if config is None:
            config = RiskConfig.from_mapping(load_config())
        elif isinstance(config, Mapping):
            config = RiskConfig.from_mapping(config)
        if not isinstance(config, RiskConfig):
            raise TypeError("config must be a RiskConfig or mapping")
        self.config = config

    @classmethod
    def from_config(cls, path: str | Path = CONFIG_PATH) -> RiskEngine:
        """Build an engine from a YAML file."""

        return cls(RiskConfig.from_mapping(load_config(path)))

    @staticmethod
    def _numeric_evidence(
        evidence: Mapping[str, object],
        keys: Iterable[str],
    ) -> float | None:
        for key in keys:
            value = evidence.get(key)
            if isinstance(value, bool) or not isinstance(value, Real):
                continue
            numeric = float(value)
            if math.isfinite(numeric):
                return max(0.0, numeric)
        return None

    @staticmethod
    def _truthy_evidence(
        evidence: Mapping[str, object],
        keys: Iterable[str],
    ) -> bool:
        for key in keys:
            value = evidence.get(key)
            if isinstance(value, bool) and value:
                return True
            if isinstance(value, Real) and not isinstance(value, bool) and value != 0:
                return True
            if isinstance(value, str) and value.strip().lower() in {
                "true",
                "yes",
                "1",
            }:
                return True
        return False

    def _duration_score(self, event: BehaviourEvent) -> float:
        duration = self._numeric_evidence(
            event.evidence,
            ("duration_seconds", "pattern_duration_seconds"),
        )
        if duration is None:
            return 0.0
        return _clamp(duration / self.config.duration_reference_seconds * 100.0)

    def _motion_score(self, event: BehaviourEvent) -> float:
        evidence = event.evidence
        ratios: list[float] = []
        for key in (
            "speed_px_per_second",
            "human_speed_px_per_second",
            "forklift_speed_px_per_second",
            "downward_speed_px_per_second",
            "pre_motion_speed_px_per_second",
        ):
            value = self._numeric_evidence(evidence, (key,))
            if value is not None:
                ratios.append(value / self.config.motion_reference_px_per_second)
        for key in (
            "horizontal_displacement_px",
            "vertical_displacement_px",
            "downward_displacement_px",
        ):
            value = self._numeric_evidence(evidence, (key,))
            if value is not None:
                ratios.append(value / self.config.displacement_reference_px)
        return _clamp(max(ratios, default=0.0) * 100.0)

    def _spatial_score(self, event: BehaviourEvent) -> float:
        evidence = event.evidence
        scores: list[float] = []
        if self._truthy_evidence(
            evidence,
            (
                "inside",
                "inside_restricted_zone",
                "restricted_zone_entry",
                "inside_zone",
            ),
        ):
            scores.append(100.0)
        if self._truthy_evidence(evidence, ("near_restricted_zone",)):
            scores.append(80.0)
        if self._truthy_evidence(evidence, ("near_floor",)):
            scores.append(70.0)
        if self._truthy_evidence(evidence, ("approaching",)):
            scores.append(75.0)
        if self._truthy_evidence(evidence, ("rapid_downward_motion",)):
            scores.append(75.0)
        if self._truthy_evidence(evidence, ("separated_from_previous_relation",)):
            scores.append(60.0)

        distance = self._numeric_evidence(
            evidence,
            ("distance_px", "nearest_distance_px"),
        )
        if distance is not None:
            maximum = self._numeric_evidence(
                evidence,
                ("maximum_distance_px", "distance_threshold_px"),
            )
            maximum = maximum or self.config.spatial_distance_reference_px
            scores.append(_clamp(1.0 - distance / maximum) * 100.0)

        support = self._numeric_evidence(evidence, ("support_ratio",))
        if support is not None:
            scores.append(_clamp(1.0 - support, 0.0, 1.0) * 100.0)
        return _clamp(max(scores, default=0.0))

    @staticmethod
    def _event_fingerprint(event: BehaviourEvent) -> tuple[str, tuple[str, ...]]:
        return event.type, tuple(sorted(set(event.entities)))

    def _repeat_score(
        self,
        event: BehaviourEvent,
        recent_events: Iterable[BehaviourEvent],
    ) -> float:
        fingerprint = self._event_fingerprint(event)
        count = 0
        for previous in recent_events:
            if not isinstance(previous, BehaviourEvent):
                raise TypeError("recent_events must contain BehaviourEvent objects")
            age = event.timestamp - previous.timestamp
            if (
                age >= 0.0
                and age <= self.config.repeat_window_seconds
                and self._event_fingerprint(previous) == fingerprint
                and previous.timestamp < event.timestamp
            ):
                count += 1
        return _clamp(count / self.config.repeat_reference_events * 100.0)

    def _category_for(self, score: float) -> str:
        selected = LOW
        for category in RISK_CATEGORIES:
            if score >= self.config.category_thresholds[category]:
                selected = category
        return selected

    def assess(
        self,
        event: BehaviourEvent,
        *,
        recent_events: Iterable[BehaviourEvent] = (),
    ) -> RiskAssessment:
        """Return risk score and all normalized component evidence."""

        if not isinstance(event, BehaviourEvent):
            raise TypeError("event must be a BehaviourEvent")
        severity = self.config.severity_by_event.get(
            _normalized_name(event.type),
            self.config.default_severity,
        )
        components = {
            SEVERITY: severity,
            DURATION: self._duration_score(event),
            MOTION_INTENSITY: self._motion_score(event),
            SPATIAL_CONTEXT: self._spatial_score(event),
            REPEAT_FREQUENCY: self._repeat_score(event, recent_events),
            DETECTION_CONFIDENCE: event.confidence * 100.0,
        }
        breakdown = RiskBreakdown(components=components, weights=self.config.weights)
        score = breakdown.score
        return RiskAssessment(
            event_type=event.type,
            score=score,
            category=self._category_for(score),
            detection_confidence=event.confidence,
            breakdown=breakdown,
        )
