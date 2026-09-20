"""Possible dragging detector based on near-floor temporal motion."""

from __future__ import annotations

from collections.abc import Iterable

from ..intelligence.scene_graph import SceneNode
from .base import (
    BehaviourCandidate,
    BehaviourContext,
    BehaviourDetector,
    BehaviourError,
    class_names,
)

POSSIBLE_DRAGGING = "POSSIBLE_DRAGGING"


class DraggingDetector(BehaviourDetector):
    """Detect sustained horizontal movement of a low image-plane object."""

    event_type = POSSIBLE_DRAGGING

    def __init__(
        self,
        *,
        draggable_classes: object = None,
        person_classes: object = None,
        min_duration_seconds: float = 2.0,
        history_window_seconds: float = 5.0,
        min_horizontal_displacement_px: float = 40.0,
        max_vertical_displacement_px: float = 35.0,
        near_floor_fraction: float = 0.85,
        near_floor_min_px: float = 900.0,
        require_person_nearby: bool = False,
        debounce_seconds: float = 0.0,
        cooldown_seconds: float = 3.0,
    ) -> None:
        super().__init__(
            debounce_seconds=debounce_seconds,
            cooldown_seconds=cooldown_seconds,
        )
        for name, value in (
            ("min_duration_seconds", min_duration_seconds),
            ("history_window_seconds", history_window_seconds),
            ("min_horizontal_displacement_px", min_horizontal_displacement_px),
            ("max_vertical_displacement_px", max_vertical_displacement_px),
            ("near_floor_min_px", near_floor_min_px),
        ):
            if float(value) < 0.0:
                raise BehaviourError(f"{name} must be non-negative")
        if not 0.0 <= float(near_floor_fraction) <= 1.0:
            raise BehaviourError("near_floor_fraction must be between 0 and 1")
        self.draggable_classes = class_names(
            draggable_classes,
            ("carton", "box", "package", "pallet"),
        )
        self.person_classes = class_names(person_classes, ("person", "worker"))
        self.min_duration_seconds = float(min_duration_seconds)
        self.history_window_seconds = float(history_window_seconds)
        self.min_horizontal_displacement_px = float(min_horizontal_displacement_px)
        self.max_vertical_displacement_px = float(max_vertical_displacement_px)
        self.near_floor_fraction = float(near_floor_fraction)
        self.near_floor_min_px = float(near_floor_min_px)
        self.require_person_nearby = bool(require_person_nearby)

    def _near_floor(self, node: SceneNode, context: BehaviourContext) -> bool:
        assert node.state is not None
        bottom_y = float(node.state.bbox[3])
        if context.frame_height is not None:
            return bottom_y >= context.frame_height * self.near_floor_fraction
        return bottom_y >= self.near_floor_min_px

    def _nearby_person(
        self,
        node: SceneNode,
        context: BehaviourContext,
    ) -> str | None:
        nodes = {candidate.node_id: candidate for candidate in context.scene_graph.nodes}
        for nearby_id in context.scene_graph.get_nearby(node.node_id):
            nearby = nodes.get(nearby_id)
            if nearby is None or nearby.state is None:
                continue
            nearby_class = "_".join(
                str(nearby.state.class_name).strip().lower().replace("-", " ").split()
            )
            if nearby_class in self.person_classes:
                return nearby.node_id
        return None

    def evaluate(self, context: BehaviourContext) -> Iterable[BehaviourCandidate]:
        for node in context.scene_graph.nodes:
            if node.node_type != "object" or node.state is None:
                continue
            class_name = "_".join(
                str(node.state.class_name).strip().lower().replace("-", " ").split()
            )
            if class_name not in self.draggable_classes or not self._near_floor(node, context):
                continue
            history = context.memory.get_history(node.state.track_id)
            cutoff = context.timestamp - self.history_window_seconds
            recent = [state for state in history if state.timestamp >= cutoff]
            if len(recent) < 2:
                continue
            first, latest = recent[0], recent[-1]
            duration = latest.timestamp - first.timestamp
            horizontal_displacement = abs(latest.center[0] - first.center[0])
            vertical_displacement = abs(latest.center[1] - first.center[1])
            if (
                duration < self.min_duration_seconds
                or horizontal_displacement < self.min_horizontal_displacement_px
                or vertical_displacement > self.max_vertical_displacement_px
            ):
                continue
            person_id = self._nearby_person(node, context)
            if self.require_person_nearby and person_id is None:
                continue
            evidence: dict[str, object] = {
                "near_floor": True,
                "horizontal_displacement_px": horizontal_displacement,
                "vertical_displacement_px": vertical_displacement,
                "duration_seconds": duration,
                "person_nearby": person_id is not None,
                "history_window_seconds": self.history_window_seconds,
            }
            entities = (node.node_id,) if person_id is None else (node.node_id, person_id)
            confidence = min(
                0.95,
                0.55
                + min(0.25, horizontal_displacement / max(1.0, self.min_horizontal_displacement_px) * 0.10)
                + (0.10 if person_id is not None else 0.0),
            )
            yield BehaviourCandidate(
                event_type=self.event_type,
                timestamp=context.timestamp,
                entities=entities,
                confidence=confidence,
                evidence=evidence,
                key=(node.node_id,),
            )

