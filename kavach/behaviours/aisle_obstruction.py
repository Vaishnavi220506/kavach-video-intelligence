"""Stationary-object obstruction rule for configured aisle polygons."""

from __future__ import annotations

from collections.abc import Iterable

from ._utils import node_class, object_nodes
from .base import (
    BehaviourCandidate,
    BehaviourContext,
    BehaviourDetector,
    BehaviourError,
    class_names,
)

AISLE_OBSTRUCTION = "AISLE_OBSTRUCTION"


class AisleObstructionDetector(BehaviourDetector):
    """Flag a monitored object that remains stationary inside an aisle zone."""

    event_type = AISLE_OBSTRUCTION

    def __init__(
        self,
        *,
        aisle_zones: object = None,
        monitored_classes: object = None,
        min_stationary_seconds: float = 3.0,
        debounce_seconds: float = 0.25,
        cooldown_seconds: float = 5.0,
    ) -> None:
        super().__init__(debounce_seconds=debounce_seconds, cooldown_seconds=cooldown_seconds)
        if float(min_stationary_seconds) < 0.0:
            raise BehaviourError("min_stationary_seconds must be non-negative")
        if aisle_zones is None:
            self.aisle_zones = ()
        else:
            self.aisle_zones = class_names(aisle_zones, ())
        self.monitored_classes = class_names(monitored_classes, ("carton", "box", "package", "pallet", "trolley"))
        self.min_stationary_seconds = float(min_stationary_seconds)

    def evaluate(self, context: BehaviourContext) -> Iterable[BehaviourCandidate]:
        if not self.aisle_zones:
            return
        for node in object_nodes(context):
            assert node.state is not None
            if node_class(node) not in self.monitored_classes:
                continue
            inside = [
                edge.object
                for edge in context.scene_graph.relations
                if edge.subject == node.node_id and edge.relation == "inside_zone" and edge.object in self.aisle_zones
            ]
            if not inside:
                continue
            stationary_duration = context.memory.get_stationary_duration(node.state.track_id)
            if stationary_duration < self.min_stationary_seconds:
                continue
            zone = sorted(inside)[0]
            yield BehaviourCandidate(
                event_type=self.event_type,
                timestamp=context.timestamp,
                entities=(node.node_id, zone),
                confidence=min(0.88, 0.58 + min(0.25, stationary_duration / 20.0)),
                evidence={
                    "zone_name": zone,
                    "stationary": True,
                    "stationary_duration_seconds": stationary_duration,
                    "minimum_stationary_seconds": self.min_stationary_seconds,
                    "image_space_only": True,
                },
                key=(node.node_id, zone),
            )

