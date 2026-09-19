"""Stationary-object placement rule based on configurable allowed zones."""

from __future__ import annotations

from collections.abc import Iterable

from ._utils import node_class, object_nodes
from .base import BehaviourCandidate, BehaviourContext, BehaviourDetector, BehaviourError, class_names


IMPROPER_PLACEMENT = "IMPROPER_PLACEMENT"


class ImproperPlacementDetector(BehaviourDetector):
    """Flag a settled object outside all configured placement zones."""

    event_type = IMPROPER_PLACEMENT

    def __init__(
        self,
        *,
        allowed_zones: object = None,
        monitored_classes: object = None,
        min_stationary_seconds: float = 2.0,
        debounce_seconds: float = 0.25,
        cooldown_seconds: float = 5.0,
    ) -> None:
        super().__init__(debounce_seconds=debounce_seconds, cooldown_seconds=cooldown_seconds)
        if float(min_stationary_seconds) < 0.0:
            raise BehaviourError("min_stationary_seconds must be non-negative")
        self.allowed_zones = class_names(allowed_zones, ("staging_zone", "loading_zone", "pallet_zone"))
        self.monitored_classes = class_names(monitored_classes, ("carton", "box", "package", "pallet", "trolley"))
        self.min_stationary_seconds = float(min_stationary_seconds)

    def evaluate(self, context: BehaviourContext) -> Iterable[BehaviourCandidate]:
        for node in object_nodes(context):
            assert node.state is not None
            if node_class(node) not in self.monitored_classes:
                continue
            inside_any_zone = any(
                edge.subject == node.node_id
                and edge.relation == "inside_zone"
                and edge.object in self.allowed_zones
                for edge in context.scene_graph.relations
            )
            if inside_any_zone:
                continue
            stationary_duration = context.memory.get_stationary_duration(node.state.track_id)
            if stationary_duration < self.min_stationary_seconds:
                continue
            yield BehaviourCandidate(
                event_type=self.event_type,
                timestamp=context.timestamp,
                entities=(node.node_id,),
                confidence=min(0.84, 0.55 + min(0.22, stationary_duration / 20.0)),
                evidence={
                    "stationary": True,
                    "stationary_duration_seconds": stationary_duration,
                    "allowed_zones": ",".join(self.allowed_zones),
                    "inside_allowed_zone": False,
                    "image_space_only": True,
                },
                key=(node.node_id,),
            )

