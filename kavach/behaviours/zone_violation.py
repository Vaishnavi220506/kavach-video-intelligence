"""Restricted-zone violation detector for KAVACH Module 7."""

from __future__ import annotations

from collections.abc import Iterable

from ..intelligence.relationships import INSIDE_ZONE, zone_node_id
from ..intelligence.zones import RESTRICTED_ZONE
from .base import (
    BehaviourCandidate,
    BehaviourContext,
    BehaviourDetector,
    BehaviourError,
    class_names,
)


ZONE_VIOLATION = "ZONE_VIOLATION"


class ZoneViolationDetector(BehaviourDetector):
    """Emit one event when a configured class enters a restricted polygon."""

    event_type = ZONE_VIOLATION

    def __init__(
        self,
        *,
        zone_name: str = RESTRICTED_ZONE,
        monitored_classes: object = None,
        debounce_seconds: float = 0.25,
        cooldown_seconds: float = 3.0,
        base_confidence: float = 0.85,
    ) -> None:
        super().__init__(
            debounce_seconds=debounce_seconds,
            cooldown_seconds=cooldown_seconds,
        )
        if not str(zone_name).strip():
            raise BehaviourError("zone_name cannot be empty")
        if not 0.0 <= float(base_confidence) <= 1.0:
            raise BehaviourError("base_confidence must be between 0 and 1")
        self.zone_name = str(zone_name)
        self.monitored_classes = (
            None if monitored_classes is None else class_names(monitored_classes, ())
        )
        self.base_confidence = float(base_confidence)

    def evaluate(self, context: BehaviourContext) -> Iterable[BehaviourCandidate]:
        target_zone = zone_node_id(self.zone_name)
        for edge in context.scene_graph.relations:
            if edge.relation != INSIDE_ZONE or edge.object != target_zone:
                continue
            node = next(
                (candidate for candidate in context.scene_graph.nodes if candidate.node_id == edge.subject),
                None,
            )
            if node is None or node.state is None:
                continue
            class_name = str(node.state.class_name).strip().lower().replace("-", " ")
            normalized_class = "_".join(class_name.split())
            if self.monitored_classes is not None and normalized_class not in self.monitored_classes:
                continue
            evidence: dict[str, object] = dict(edge.evidence)
            evidence.update(
                {
                    "zone_name": self.zone_name,
                    "track_id": float(node.state.track_id),
                    "detector_confidence": float(node.state.confidence),
                }
            )
            confidence = max(
                0.0,
                min(1.0, self.base_confidence * (0.5 + 0.5 * float(node.state.confidence))),
            )
            yield BehaviourCandidate(
                event_type=self.event_type,
                timestamp=context.timestamp,
                entities=(edge.subject, edge.object),
                confidence=confidence,
                evidence=evidence,
                key=(edge.subject, edge.object),
            )
