"""Human-forklift proximity detector using distance and motion evidence."""

from __future__ import annotations

from collections.abc import Iterable

from ..intelligence.geometry import bottom_center, euclidean_distance
from ..intelligence.relationships import APPROACHING
from ..intelligence.scene_graph import SceneNode
from .base import (
    BehaviourCandidate,
    BehaviourContext,
    BehaviourDetector,
    BehaviourError,
    class_names,
)

UNSAFE_HUMAN_FORKLIFT_PROXIMITY = "UNSAFE_HUMAN_FORKLIFT_PROXIMITY"


class HumanForkliftProximityDetector(BehaviourDetector):
    """Detect close human/forklift pairs when motion evidence is present."""

    event_type = UNSAFE_HUMAN_FORKLIFT_PROXIMITY

    def __init__(
        self,
        *,
        human_classes: object = None,
        forklift_classes: object = None,
        max_distance_px: float = 100.0,
        min_motion_speed_px_per_second: float = 2.0,
        require_motion: bool = True,
        debounce_seconds: float = 0.25,
        cooldown_seconds: float = 3.0,
    ) -> None:
        super().__init__(
            debounce_seconds=debounce_seconds,
            cooldown_seconds=cooldown_seconds,
        )
        if float(max_distance_px) < 0.0 or float(min_motion_speed_px_per_second) < 0.0:
            raise BehaviourError("proximity thresholds must be non-negative")
        self.human_classes = class_names(human_classes, ("person", "worker"))
        self.forklift_classes = class_names(
            forklift_classes,
            ("forklift", "pallet_truck", "lift_truck"),
        )
        self.max_distance_px = float(max_distance_px)
        self.min_motion_speed_px_per_second = float(min_motion_speed_px_per_second)
        self.require_motion = bool(require_motion)

    @staticmethod
    def _class_name(node: SceneNode) -> str:
        assert node.state is not None
        return "_".join(
            str(node.state.class_name).strip().lower().replace("-", " ").split()
        )

    def evaluate(self, context: BehaviourContext) -> Iterable[BehaviourCandidate]:
        objects = [
            node
            for node in context.scene_graph.nodes
            if node.node_type == "object" and node.state is not None
        ]
        for human in objects:
            if self._class_name(human) not in self.human_classes:
                continue
            for forklift in objects:
                if self._class_name(forklift) not in self.forklift_classes:
                    continue
                if human.node_id == forklift.node_id:
                    continue
                human_point = bottom_center(human.state.bbox)
                forklift_point = bottom_center(forklift.state.bbox)
                distance_px = euclidean_distance(human_point, forklift_point)
                if distance_px > self.max_distance_px:
                    continue
                human_motion = _get_motion(context, human.state.track_id)
                forklift_motion = _get_motion(context, forklift.state.track_id)
                human_speed = 0.0 if human_motion is None else human_motion.speed_pixels_per_second
                forklift_speed = 0.0 if forklift_motion is None else forklift_motion.speed_pixels_per_second
                approaching = context.scene_graph.has_relation(
                    human.node_id,
                    APPROACHING,
                    forklift.node_id,
                )
                motion_present = (
                    approaching
                    or human_speed >= self.min_motion_speed_px_per_second
                    or forklift_speed >= self.min_motion_speed_px_per_second
                )
                if self.require_motion and not motion_present:
                    continue
                evidence: dict[str, object] = {
                    "distance_px": distance_px,
                    "maximum_distance_px": self.max_distance_px,
                    "human_speed_px_per_second": human_speed,
                    "forklift_speed_px_per_second": forklift_speed,
                    "approaching": approaching,
                    "motion_present": motion_present,
                    "ground_plane_calibration_available": False,
                }
                confidence = min(
                    0.94,
                    0.55
                    + max(0.0, 0.20 * (1.0 - distance_px / max(1.0, self.max_distance_px)))
                    + (0.10 if approaching else 0.0),
                )
                yield BehaviourCandidate(
                    event_type=self.event_type,
                    timestamp=context.timestamp,
                    entities=(human.node_id, forklift.node_id),
                    confidence=confidence,
                    evidence=evidence,
                    key=(human.node_id, forklift.node_id),
                )


def _get_motion(context: BehaviourContext, track_id: int):
    try:
        return context.memory.get_velocity(track_id)
    except (KeyError, ValueError, RuntimeError):
        return None

