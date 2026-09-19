"""General temporal activity signals derived from tracked object state.

These signals are intentionally separate from the warehouse-specific safety
rules. They make the pipeline useful on footage where a generic detector can
see people or vehicles but cannot reliably label cartons, pallets, or
forklifts. They are still evidence signals, not action-recognition claims.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..intelligence.geometry import bottom_center
from ..intelligence.motion import MotionError, calculate_motion
from ..intelligence.relationships import INSIDE_ZONE
from ..intelligence.scene_graph import SceneNode
from .base import (
    BehaviourCandidate,
    BehaviourContext,
    BehaviourDetector,
    BehaviourError,
    class_names,
)


ZONE_TRANSITION = "ZONE_TRANSITION"
OBJECT_ACTIVITY = "OBJECT_ACTIVITY"


def _node_class_name(node: SceneNode) -> str:
    assert node.state is not None
    return "_".join(
        str(node.state.class_name).strip().lower().replace("-", " ").split()
    )


def _current_zone_edges(context: BehaviourContext, node_id: str):
    return tuple(
        edge
        for edge in context.scene_graph.relations
        if edge.subject == node_id and edge.relation == INSIDE_ZONE
    )


class ZoneTransitionDetector(BehaviourDetector):
    """Emit a signal when an observed track enters or exits a configured zone."""

    event_type = ZONE_TRANSITION

    def __init__(
        self,
        *,
        monitored_classes: object = None,
        monitored_zones: object = None,
        debounce_seconds: float = 0.0,
        cooldown_seconds: float = 1.0,
    ) -> None:
        super().__init__(
            debounce_seconds=debounce_seconds,
            cooldown_seconds=cooldown_seconds,
        )
        self.monitored_classes = (
            None if monitored_classes is None else class_names(monitored_classes, ())
        )
        self.monitored_zones = (
            None if monitored_zones is None else class_names(monitored_zones, ())
        )
        self._previous_zones: dict[str, set[str]] = {}

    def evaluate(self, context: BehaviourContext) -> Iterable[BehaviourCandidate]:
        current_zones: dict[str, set[str]] = {}
        current_nodes = {
            node.node_id: node
            for node in context.scene_graph.nodes
            if node.node_type == "object" and node.state is not None
        }

        for node_id, node in current_nodes.items():
            if (
                self.monitored_classes is not None
                and _node_class_name(node) not in self.monitored_classes
            ):
                continue
            zone_edges = _current_zone_edges(context, node_id)
            zones = {
                str(edge.object)
                for edge in zone_edges
                if self.monitored_zones is None
                or str(edge.object) in self.monitored_zones
            }
            current_zones[node_id] = zones
            previous = self._previous_zones.get(node_id)
            if previous is None:
                # The first observation is a baseline, not proof of an entry.
                continue
            for zone_id in sorted(zones - previous):
                edge = next(edge for edge in zone_edges if edge.object == zone_id)
                yield self._candidate(
                    context,
                    node,
                    zone_id,
                    transition="entered",
                    boundary_distance_px=edge.evidence.get("boundary_distance_px", 0.0),
                )
            for zone_id in sorted(previous - zones):
                boundary_distance = 0.0
                if context.scene_graph.zones is not None:
                    zone = context.scene_graph.zones.get_zone(zone_id.upper())
                    if zone is not None:
                        boundary_distance = zone.distance_to_boundary(
                            bottom_center(node.state.bbox)  # type: ignore[union-attr]
                        )
                yield self._candidate(
                    context,
                    node,
                    zone_id,
                    transition="exited",
                    boundary_distance_px=boundary_distance,
                )

        self._previous_zones = current_zones

    def _candidate(
        self,
        context: BehaviourContext,
        node: SceneNode,
        zone_id: str,
        *,
        transition: str,
        boundary_distance_px: float,
    ) -> BehaviourCandidate:
        assert node.state is not None
        return BehaviourCandidate(
            event_type=self.event_type,
            timestamp=context.timestamp,
            entities=(node.node_id, zone_id),
            confidence=min(0.95, 0.65 + 0.25 * float(node.state.confidence)),
            evidence={
                "transition": transition,
                "zone_name": zone_id,
                "boundary_distance_px": float(boundary_distance_px),
                "track_id": float(node.state.track_id),
                "detector_confidence": float(node.state.confidence),
            },
            key=(node.node_id, zone_id, transition),
        )

    def reset(self) -> None:
        super().reset()
        self._previous_zones.clear()


class ObjectActivityDetector(BehaviourDetector):
    """Emit debounced moving/stationary activity for visible tracks."""

    event_type = OBJECT_ACTIVITY

    def __init__(
        self,
        *,
        monitored_classes: object = None,
        minimum_track_age_seconds: float = 0.5,
        minimum_moving_speed_px_per_second: float = 8.0,
        emit_initial_state: bool = True,
        debounce_seconds: float = 0.0,
        cooldown_seconds: float = 5.0,
    ) -> None:
        super().__init__(
            debounce_seconds=debounce_seconds,
            cooldown_seconds=cooldown_seconds,
        )
        if float(minimum_track_age_seconds) < 0.0:
            raise BehaviourError("minimum_track_age_seconds must be non-negative")
        if float(minimum_moving_speed_px_per_second) < 0.0:
            raise BehaviourError(
                "minimum_moving_speed_px_per_second must be non-negative"
            )
        self.monitored_classes = (
            None if monitored_classes is None else class_names(monitored_classes, ())
        )
        self.minimum_track_age_seconds = float(minimum_track_age_seconds)
        self.minimum_moving_speed_px_per_second = float(
            minimum_moving_speed_px_per_second
        )
        self.emit_initial_state = bool(emit_initial_state)
        self._last_state: dict[str, str] = {}
        self._initial_emitted: set[str] = set()

    def evaluate(self, context: BehaviourContext) -> Iterable[BehaviourCandidate]:
        active_nodes = {
            node.node_id: node
            for node in context.scene_graph.nodes
            if node.node_type == "object" and node.state is not None
        }
        for node_id, node in active_nodes.items():
            assert node.state is not None
            if (
                self.monitored_classes is not None
                and _node_class_name(node) not in self.monitored_classes
            ):
                continue
            try:
                motion = context.memory.get_velocity(node.state.track_id)
            except (KeyError, MotionError, RuntimeError, ValueError):
                motion = None
            if motion is None:
                continue
            state = (
                "moving"
                if motion.speed_pixels_per_second
                >= self.minimum_moving_speed_px_per_second
                else "stationary"
            )
            age = context.memory.get_track_age(node.state.track_id) or 0.0
            previous = self._last_state.get(node_id)
            transition = None
            if age < self.minimum_track_age_seconds:
                # Warm up the per-track state without generating a signal
                # from a detector's first few jittery observations.
                self._last_state[node_id] = state
                continue
            if self.emit_initial_state and node_id not in self._initial_emitted:
                transition = "initial"
                self._initial_emitted.add(node_id)
            elif previous is not None and state != previous:
                transition = f"{previous}_to_{state}"

            self._last_state[node_id] = state
            if transition is None:
                continue
            yield BehaviourCandidate(
                event_type=self.event_type,
                timestamp=context.timestamp,
                entities=(node_id,),
                confidence=min(
                    0.90,
                    0.55 + min(0.30, motion.speed_pixels_per_second / 500.0),
                ),
                evidence={
                    "activity_state": state,
                    "transition": transition,
                    "speed_px_per_second": motion.speed_pixels_per_second,
                    "direction": motion.direction,
                    "dx_px": motion.dx,
                    "dy_px": motion.dy,
                    "track_age_seconds": age,
                    "timestamp_aware": True,
                },
                key=(node_id, transition),
            )

        # A track that disappeared should be eligible for a fresh initial
        # signal if it is later assigned a new visible observation.
        for node_id in tuple(self._last_state):
            if node_id not in active_nodes:
                self._last_state.pop(node_id, None)
                self._initial_emitted.discard(node_id)

    def reset(self) -> None:
        super().reset()
        self._last_state.clear()
        self._initial_emitted.clear()


__all__ = ["OBJECT_ACTIVITY", "ZONE_TRANSITION", "ObjectActivityDetector", "ZoneTransitionDetector"]
