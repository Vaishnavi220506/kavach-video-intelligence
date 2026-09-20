"""Ordered registry and configuration loader for Module 7 behaviours."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from .activity import ObjectActivityDetector, ZoneTransitionDetector
from .aisle_obstruction import AisleObstructionDetector
from .anomaly import MotionAnomalyDetector
from .base import BehaviourContext, BehaviourDetector, BehaviourError, BehaviourEvent
from .collision import CollisionRiskDetector
from .dragging import DraggingDetector
from .drop import PossibleDropDetector
from .improper_placement import ImproperPlacementDetector
from .overhang import OverhangDetector
from .proximity import HumanForkliftProximityDetector
from .rough_handling import RoughHandlingDetector
from .stacking import UnstableStackDetector
from .throwing import ThrowingDetector
from .zone_violation import ZoneViolationDetector


class BehaviourConfigurationError(BehaviourError):
    """Raised when the YAML behaviour configuration is unavailable or invalid."""


CONFIG_PATH = Path(__file__).with_name("config.yaml")
_DETECTOR_SPECS = (
    ("zone_violation", ZoneViolationDetector),
    ("zone_transition", ZoneTransitionDetector),
    ("object_activity", ObjectActivityDetector),
    ("motion_anomaly", MotionAnomalyDetector),
    ("throwing", ThrowingDetector),
    ("rough_handling", RoughHandlingDetector),
    ("aisle_obstruction", AisleObstructionDetector),
    ("improper_placement", ImproperPlacementDetector),
    ("collision_risk", CollisionRiskDetector),
    ("dragging", DraggingDetector),
    ("drop", PossibleDropDetector),
    ("overhang", OverhangDetector),
    ("stacking", UnstableStackDetector),
    ("proximity", HumanForkliftProximityDetector),
)


class BehaviourRegistry:
    """Evaluate registered detectors in deterministic order."""

    def __init__(self, detectors: Iterable[BehaviourDetector] = ()) -> None:
        self._detectors: list[BehaviourDetector] = []
        for detector in detectors:
            self.register(detector)

    @property
    def detectors(self) -> tuple[BehaviourDetector, ...]:
        """Return detectors in their evaluation order."""

        return tuple(self._detectors)

    @property
    def event_types(self) -> tuple[str, ...]:
        """Return the event types currently registered."""

        return tuple(detector.event_type for detector in self._detectors)

    def register(self, detector: BehaviourDetector) -> None:
        """Append one detector, rejecting duplicate event types."""

        if not isinstance(detector, BehaviourDetector):
            raise TypeError("detector must be a BehaviourDetector")
        if detector.event_type in self.event_types:
            raise BehaviourError(f"duplicate behaviour event type: {detector.event_type}")
        self._detectors.append(detector)

    def detect(self, context: BehaviourContext) -> list[BehaviourEvent]:
        """Evaluate every detector and return newly emitted events."""

        events: list[BehaviourEvent] = []
        for detector in self._detectors:
            events.extend(detector.detect(context))
        return events

    def reset(self) -> None:
        """Reset every detector's debounce and cooldown state."""

        for detector in self._detectors:
            detector.reset()


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, object]:
    """Load the YAML detector configuration without silently using defaults."""

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise BehaviourConfigurationError(
            "PyYAML is required to load behaviour config.yaml"
        ) from exc
    config_path = Path(path)
    if not config_path.is_file():
        raise BehaviourConfigurationError(f"behaviour config not found: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise BehaviourConfigurationError(
            f"could not read behaviour config: {config_path}"
        ) from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise BehaviourConfigurationError("behaviour config root must be a mapping")
    if "behaviours" in loaded:
        loaded = loaded["behaviours"]
    if not isinstance(loaded, Mapping):
        raise BehaviourConfigurationError("behaviours config must be a mapping")
    return dict(loaded)


def build_default_registry(
    *,
    config: Mapping[str, object] | None = None,
    config_path: str | Path | None = None,
) -> BehaviourRegistry:
    """Build the configured activity, anomaly, and safety detectors."""

    loaded = (
        dict(config)
        if config is not None
        else load_config(CONFIG_PATH if config_path is None else config_path)
    )
    registry = BehaviourRegistry()
    for config_name, detector_type in _DETECTOR_SPECS:
        section = loaded.get(config_name, {})
        if not isinstance(section, Mapping):
            raise BehaviourConfigurationError(
                f"configuration for '{config_name}' must be a mapping"
            )
        parameters = dict(section)
        enabled = bool(parameters.pop("enabled", True))
        if enabled:
            try:
                registry.register(detector_type(**parameters))
            except (TypeError, ValueError, BehaviourError) as exc:
                raise BehaviourConfigurationError(
                    f"invalid configuration for '{config_name}': {exc}"
                ) from exc
    return registry
