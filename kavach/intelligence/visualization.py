"""OpenCV drawing helpers for KAVACH temporal and spatial intelligence."""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from .geometry import (
    bbox_center,
    euclidean_distance,
    support_ratio,
    validate_bbox,
)
from .object_memory import ObjectMemory
from .scene_graph import SceneGraph
from .zones import ZoneManager


def draw_trajectory(
    frame: np.ndarray,
    trajectory: Iterable[tuple[float, float]],
    *,
    color: tuple[int, int, int] = (255, 0, 255),
    thickness: int = 2,
    point_radius: int = 3,
) -> np.ndarray:
    """Return a copy of a BGR frame with a center-point trajectory drawn."""

    if not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim != 3:
        raise ValueError("frame must be a non-empty color NumPy array")
    if thickness <= 0 or point_radius <= 0:
        raise ValueError("thickness and point_radius must be positive")

    points = [
        (int(round(float(x))), int(round(float(y))))
        for x, y in trajectory
    ]
    annotated = frame.copy()
    for start, end in zip(points, points[1:], strict=False):
        cv2.line(annotated, start, end, color, thickness, cv2.LINE_AA)
    for point in points:
        cv2.circle(annotated, point, point_radius, color, -1, cv2.LINE_AA)
    return annotated


def draw_memory_trajectory(
    frame: np.ndarray,
    memory: ObjectMemory,
    track_id: int,
    *,
    seconds: float = 2.0,
    color: tuple[int, int, int] = (255, 0, 255),
) -> np.ndarray:
    """Draw one retained track trajectory and a small track label."""

    trajectory = memory.get_trajectory(track_id, seconds=seconds)
    annotated = draw_trajectory(frame, trajectory, color=color)
    if trajectory:
        x, y = (int(round(value)) for value in trajectory[-1])
        cv2.putText(
            annotated,
            f"Track #{int(track_id)}",
            (x + 6, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def _validate_frame(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim != 3:
        raise ValueError("frame must be a non-empty color NumPy array")


def draw_zones(
    frame: np.ndarray,
    zones: ZoneManager,
    *,
    alpha: float = 0.20,
    show_labels: bool = True,
) -> np.ndarray:
    """Return a copy of a BGR frame with configured polygons overlaid."""

    _validate_frame(frame)
    if not isinstance(zones, ZoneManager):
        raise TypeError("zones must be a ZoneManager")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    annotated = frame.copy()
    overlay = annotated.copy()
    for zone in zones.zones:
        points = np.asarray(zone.polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [points], zone.color)
        cv2.polylines(annotated, [points], True, zone.color, 2, cv2.LINE_AA)
        if show_labels:
            x, y = (int(round(value)) for value in zone.polygon[0])
            cv2.putText(
                annotated,
                zone.name,
                (x, max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                zone.color,
                2,
                cv2.LINE_AA,
            )
    return cv2.addWeighted(overlay, alpha, annotated, 1.0 - alpha, 0.0)


def draw_bbox_centers(
    frame: np.ndarray,
    bboxes: Iterable[tuple[float, float, float, float]],
    *,
    color: tuple[int, int, int] = (255, 0, 0),
    radius: int = 4,
) -> np.ndarray:
    """Draw geometric centers for a sequence of XYXY boxes."""

    _validate_frame(frame)
    if radius <= 0:
        raise ValueError("radius must be positive")
    annotated = frame.copy()
    for bbox in bboxes:
        x, y = (int(round(value)) for value in bbox_center(bbox))
        cv2.circle(annotated, (x, y), radius, color, -1, cv2.LINE_AA)
    return annotated


def draw_distance_indicator(
    frame: np.ndarray,
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    *,
    label: str | None = None,
    color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a point-to-point image-space distance indicator in pixels."""

    _validate_frame(frame)
    if thickness <= 0:
        raise ValueError("thickness must be positive")
    distance = euclidean_distance(point_a, point_b)
    start = tuple(int(round(value)) for value in point_a)
    end = tuple(int(round(value)) for value in point_b)
    annotated = frame.copy()
    cv2.line(annotated, start, end, color, thickness, cv2.LINE_AA)
    cv2.circle(annotated, start, 4, color, -1, cv2.LINE_AA)
    cv2.circle(annotated, end, 4, color, -1, cv2.LINE_AA)
    midpoint = ((start[0] + end[0]) // 2 + 5, (start[1] + end[1]) // 2 - 5)
    cv2.putText(
        annotated,
        label or f"{distance:.1f} px",
        midpoint,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA,
    )
    return annotated


def draw_support_overlap(
    frame: np.ndarray,
    supported_bbox: tuple[float, float, float, float],
    support_bbox: tuple[float, float, float, float],
    *,
    label: str | None = None,
    supported_color: tuple[int, int, int] = (255, 0, 0),
    support_color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw boxes and their horizontal support-overlap indicator.

    The ratio is a 2-D geometric cue only. It is not a claim that an object is
    physically supported by another object.
    """

    _validate_frame(frame)
    supported = validate_bbox(supported_bbox)
    support = validate_bbox(support_bbox)
    ratio = support_ratio(supported, support)
    annotated = frame.copy()
    for bbox, color in ((supported, supported_color), (support, support_color)):
        x1, y1, x2, y2 = (int(round(value)) for value in bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    start_x = max(supported[0], support[0])
    end_x = min(supported[2], support[2])
    if end_x > start_x:
        y = int(round((supported[3] + support[1]) / 2.0))
        cv2.line(
            annotated,
            (int(round(start_x)), y),
            (int(round(end_x)), y),
            (0, 255, 255),
            4,
            cv2.LINE_AA,
        )
    text = label or f"support overlap {ratio:.0%}"
    x, y = (int(round(value)) for value in bbox_center(supported))
    cv2.putText(
        annotated,
        text,
        (x, max(18, y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def draw_scene_graph(
    frame: np.ndarray,
    graph: SceneGraph,
    *,
    show_relation_labels: bool = True,
) -> np.ndarray:
    """Draw current object nodes and object-to-object graph edges.

    Zone polygons are drawn through ``draw_zones``. This is a debugging view,
    not a replacement for the structured graph or a learned visualizer.
    """

    _validate_frame(frame)
    if not isinstance(graph, SceneGraph):
        raise TypeError("graph must be a SceneGraph")

    annotated = draw_zones(frame, graph.zones) if graph.zones is not None else frame.copy()
    centers: dict[str, tuple[int, int]] = {}
    for node in graph.nodes:
        if node.state is None:
            continue
        x1, y1, x2, y2 = (int(round(value)) for value in validate_bbox(node.state.bbox))
        center = tuple(int(round(value)) for value in bbox_center(node.state.bbox))
        centers[node.node_id] = center
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(annotated, center, 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            annotated,
            node.node_id,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    relation_colors = {
        "near": (0, 255, 255),
        "overlapping": (0, 165, 255),
        "supported_by": (0, 255, 0),
        "moving_with": (255, 0, 255),
        "approaching": (0, 0, 255),
        "separating_from": (255, 0, 0),
        "above": (200, 200, 0),
        "below": (200, 200, 0),
        "left_of": (200, 200, 0),
        "right_of": (200, 200, 0),
    }
    drawn_symmetric: set[tuple[str, frozenset[str]]] = set()
    symmetric = {"near", "overlapping", "moving_with", "approaching", "separating_from"}
    for edge in graph.relations:
        if edge.subject not in centers or edge.object not in centers:
            continue
        if edge.relation in symmetric:
            key = (edge.relation, frozenset((edge.subject, edge.object)))
            if key in drawn_symmetric:
                continue
            drawn_symmetric.add(key)
        start = centers[edge.subject]
        end = centers[edge.object]
        color = relation_colors.get(edge.relation, (255, 255, 255))
        cv2.arrowedLine(annotated, start, end, color, 1, cv2.LINE_AA, tipLength=0.08)
        if show_relation_labels:
            midpoint = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
            cv2.putText(
                annotated,
                edge.relation,
                (midpoint[0] + 4, midpoint[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )
    return annotated
