"""Small shared helpers for explainable warehouse behaviour rules."""

from __future__ import annotations

from collections.abc import Iterable

from ..intelligence.geometry import bottom_center, euclidean_distance
from ..intelligence.motion import MotionEstimate
from ..intelligence.scene_graph import SceneNode
from .base import BehaviourContext, class_names, normalized_class_name


def node_class(node: SceneNode) -> str:
    if node.state is None:
        return ""
    return normalized_class_name(node.state.class_name)


def object_nodes(context: BehaviourContext) -> tuple[SceneNode, ...]:
    return tuple(
        node
        for node in context.scene_graph.nodes
        if node.node_type == "object" and node.state is not None
    )


def get_motion(context: BehaviourContext, track_id: int) -> MotionEstimate | None:
    try:
        return context.memory.get_velocity(track_id)
    except (KeyError, ValueError, RuntimeError):
        return None


def nearby_entity(
    context: BehaviourContext,
    node: SceneNode,
    classes: Iterable[str],
    *,
    maximum_distance_px: float | None = None,
) -> SceneNode | None:
    if node.state is None:
        return None
    wanted = set(class_names(classes, ()))
    candidates = {item.node_id: item for item in object_nodes(context)}
    for other_id in context.scene_graph.get_nearby(node.node_id):
        other = candidates.get(other_id)
        if other is None or other.state is None or node_class(other) not in wanted:
            continue
        if maximum_distance_px is not None:
            distance = euclidean_distance(
                bottom_center(node.state.bbox),
                bottom_center(other.state.bbox),
            )
            if distance > maximum_distance_px:
                continue
        return other
    return None


def is_near_floor(context: BehaviourContext, node: SceneNode, fraction: float) -> bool:
    if node.state is None:
        return False
    if context.frame_height is None:
        return False
    return float(node.state.bbox[3]) >= float(context.frame_height) * float(fraction)

