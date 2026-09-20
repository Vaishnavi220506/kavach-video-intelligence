"""Explainable closing-trajectory collision-risk detector."""

from __future__ import annotations

from collections.abc import Iterable

from ..intelligence.geometry import bottom_center, euclidean_distance
from ..intelligence.relationships import APPROACHING
from ._utils import node_class, object_nodes
from .base import (
    BehaviourCandidate,
    BehaviourContext,
    BehaviourDetector,
    BehaviourError,
    class_names,
)

COLLISION_RISK = "COLLISION_RISK"


class CollisionRiskDetector(BehaviourDetector):
    """Flag close pairs whose image-space distance is shrinking."""

    event_type = COLLISION_RISK

    def __init__(
        self,
        *,
        human_classes: object = None,
        equipment_classes: object = None,
        warning_distance_px: float = 140.0,
        minimum_closing_speed_px_per_second: float = 8.0,
        require_approaching_relation: bool = False,
        debounce_seconds: float = 0.25,
        cooldown_seconds: float = 4.0,
    ) -> None:
        super().__init__(debounce_seconds=debounce_seconds, cooldown_seconds=cooldown_seconds)
        if float(warning_distance_px) < 0.0 or float(minimum_closing_speed_px_per_second) < 0.0:
            raise BehaviourError("collision thresholds must be non-negative")
        self.human_classes = class_names(human_classes, ("person", "worker"))
        self.equipment_classes = class_names(equipment_classes, ("forklift", "pallet_truck", "trolley", "truck"))
        self.warning_distance = float(warning_distance_px)
        self.minimum_closing_speed = float(minimum_closing_speed_px_per_second)
        self.require_approaching_relation = bool(require_approaching_relation)

    def evaluate(self, context: BehaviourContext) -> Iterable[BehaviourCandidate]:
        objects = object_nodes(context)
        for human in objects:
            assert human.state is not None
            if node_class(human) not in self.human_classes:
                continue
            human_history = context.memory.get_history(human.state.track_id)
            for equipment in objects:
                assert equipment.state is not None
                if node_class(equipment) not in self.equipment_classes or equipment.node_id == human.node_id:
                    continue
                equipment_history = context.memory.get_history(equipment.state.track_id)
                if len(human_history) < 2 or len(equipment_history) < 2:
                    continue
                current_distance = euclidean_distance(bottom_center(human.state.bbox), bottom_center(equipment.state.bbox))
                if current_distance > self.warning_distance:
                    continue
                previous_distance = euclidean_distance(bottom_center(human_history[-2].bbox), bottom_center(equipment_history[-2].bbox))
                dt = max(1e-9, context.timestamp - human_history[-2].timestamp)
                closing_speed = max(0.0, (previous_distance - current_distance) / dt)
                approaching = context.scene_graph.has_relation(human.node_id, APPROACHING, equipment.node_id)
                if closing_speed < self.minimum_closing_speed and not approaching:
                    continue
                if self.require_approaching_relation and not approaching:
                    continue
                yield BehaviourCandidate(
                    event_type=self.event_type,
                    timestamp=context.timestamp,
                    entities=(human.node_id, equipment.node_id),
                    confidence=min(0.93, 0.58 + min(0.22, closing_speed / 100.0) + (0.08 if approaching else 0.0)),
                    evidence={
                        "distance_px": current_distance,
                        "previous_distance_px": previous_distance,
                        "closing_speed_px_per_second": closing_speed,
                        "approaching": approaching,
                        "warning_distance_px": self.warning_distance,
                        "ground_plane_calibration_available": False,
                        "image_space_only": True,
                    },
                    key=(human.node_id, equipment.node_id),
                )

