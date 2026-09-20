"""Cautious throwing/release detector using tracked motion only."""

from __future__ import annotations

from collections.abc import Iterable

from ._utils import get_motion, is_near_floor, nearby_entity, node_class, object_nodes
from .base import (
    BehaviourCandidate,
    BehaviourContext,
    BehaviourDetector,
    BehaviourError,
    class_names,
)

POSSIBLE_THROWING = "POSSIBLE_THROWING"


class ThrowingDetector(BehaviourDetector):
    """Flag a high-speed object release pattern; never claim a throw as fact."""

    event_type = POSSIBLE_THROWING

    def __init__(
        self,
        *,
        throw_classes: object = None,
        person_classes: object = None,
        min_release_speed_px_per_second: float = 220.0,
        min_acceleration_px_per_second_squared: float = 120.0,
        near_floor_fraction: float = 0.85,
        require_person_nearby: bool = False,
        debounce_seconds: float = 0.0,
        cooldown_seconds: float = 4.0,
    ) -> None:
        super().__init__(debounce_seconds=debounce_seconds, cooldown_seconds=cooldown_seconds)
        if float(min_release_speed_px_per_second) < 0.0 or float(min_acceleration_px_per_second_squared) < 0.0:
            raise BehaviourError("throwing motion thresholds must be non-negative")
        if not 0.0 <= float(near_floor_fraction) <= 1.0:
            raise BehaviourError("near_floor_fraction must be between 0 and 1")
        self.throw_classes = class_names(throw_classes, ("carton", "box", "package"))
        self.person_classes = class_names(person_classes, ("person", "worker"))
        self.min_release_speed = float(min_release_speed_px_per_second)
        self.min_acceleration = float(min_acceleration_px_per_second_squared)
        self.near_floor_fraction = float(near_floor_fraction)
        self.require_person_nearby = bool(require_person_nearby)

    def evaluate(self, context: BehaviourContext) -> Iterable[BehaviourCandidate]:
        for node in object_nodes(context):
            assert node.state is not None
            if node_class(node) not in self.throw_classes:
                continue
            motion = get_motion(context, node.state.track_id)
            if motion is None or motion.speed_pixels_per_second < self.min_release_speed:
                continue
            acceleration = motion.acceleration_pixels_per_second_squared or 0.0
            if acceleration < self.min_acceleration:
                continue
            person = nearby_entity(context, node, self.person_classes)
            if self.require_person_nearby and person is None:
                continue
            evidence = {
                "high_release_speed": True,
                "speed_px_per_second": motion.speed_pixels_per_second,
                "acceleration_px_per_second_squared": acceleration,
                "direction": motion.direction,
                "near_floor": is_near_floor(context, node, self.near_floor_fraction),
                "person_nearby": person is not None,
                "image_space_only": True,
            }
            entities = (node.node_id,) if person is None else (node.node_id, person.node_id)
            yield BehaviourCandidate(
                event_type=self.event_type,
                timestamp=context.timestamp,
                entities=entities,
                confidence=min(0.92, 0.55 + min(0.25, acceleration / 1000.0)),
                evidence=evidence,
                key=(node.node_id,),
            )

