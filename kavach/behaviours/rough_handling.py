"""Possible rough-handling detector from acceleration and direction change."""

from __future__ import annotations

from collections.abc import Iterable
import math

from ..intelligence.motion import MotionError, calculate_motion
from ._utils import get_motion, nearby_entity, node_class, object_nodes
from .base import BehaviourCandidate, BehaviourContext, BehaviourDetector, BehaviourError, class_names


POSSIBLE_ROUGH_HANDLING = "POSSIBLE_ROUGH_HANDLING"


class RoughHandlingDetector(BehaviourDetector):
    """Flag abrupt object motion while keeping the result explicitly possible."""

    event_type = POSSIBLE_ROUGH_HANDLING

    def __init__(
        self,
        *,
        monitored_classes: object = None,
        min_speed_px_per_second: float = 45.0,
        min_acceleration_px_per_second_squared: float = 180.0,
        direction_change_degrees: float = 100.0,
        person_classes: object = None,
        debounce_seconds: float = 0.0,
        cooldown_seconds: float = 4.0,
    ) -> None:
        super().__init__(debounce_seconds=debounce_seconds, cooldown_seconds=cooldown_seconds)
        for name, value in (("min_speed_px_per_second", min_speed_px_per_second), ("min_acceleration_px_per_second_squared", min_acceleration_px_per_second_squared), ("direction_change_degrees", direction_change_degrees)):
            if float(value) < 0.0 or not math.isfinite(float(value)):
                raise BehaviourError(f"{name} must be finite and non-negative")
        if float(direction_change_degrees) > 180.0:
            raise BehaviourError("direction_change_degrees cannot exceed 180")
        self.monitored_classes = class_names(monitored_classes, ("carton", "box", "package", "pallet"))
        self.person_classes = class_names(person_classes, ("person", "worker"))
        self.min_speed = float(min_speed_px_per_second)
        self.min_acceleration = float(min_acceleration_px_per_second_squared)
        self.direction_change = float(direction_change_degrees)

    @staticmethod
    def _angle_change(previous, current) -> float:
        first = math.degrees(math.atan2(previous.velocity_y_pixels_per_second, previous.velocity_x_pixels_per_second))
        second = math.degrees(math.atan2(current.velocity_y_pixels_per_second, current.velocity_x_pixels_per_second))
        delta = abs(first - second) % 360.0
        return min(delta, 360.0 - delta)

    def evaluate(self, context: BehaviourContext) -> Iterable[BehaviourCandidate]:
        for node in object_nodes(context):
            assert node.state is not None
            if node_class(node) not in self.monitored_classes:
                continue
            history = context.memory.get_history(node.state.track_id)
            if len(history) < 3:
                continue
            current = get_motion(context, node.state.track_id)
            if current is None or current.speed_pixels_per_second < self.min_speed:
                continue
            try:
                previous = calculate_motion(history[-3], history[-2])
            except MotionError:
                continue
            acceleration = current.acceleration_pixels_per_second_squared or 0.0
            angle = self._angle_change(previous, current)
            if acceleration < self.min_acceleration and angle < self.direction_change:
                continue
            person = nearby_entity(context, node, self.person_classes)
            evidence = {
                "speed_px_per_second": current.speed_pixels_per_second,
                "acceleration_px_per_second_squared": acceleration,
                "direction_change_degrees": angle,
                "person_nearby": person is not None,
                "image_space_only": True,
            }
            entities = (node.node_id,) if person is None else (node.node_id, person.node_id)
            yield BehaviourCandidate(
                event_type=self.event_type,
                timestamp=context.timestamp,
                entities=entities,
                confidence=min(0.90, 0.52 + min(0.25, acceleration / 1000.0) + (0.08 if angle >= self.direction_change else 0.0)),
                evidence=evidence,
                key=(node.node_id,),
            )

