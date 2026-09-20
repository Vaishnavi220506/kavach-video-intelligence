"""Reusable image-space geometry primitives for KAVACH Module 5."""

from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np

Point = tuple[float, float]
BoundingBox = tuple[float, float, float, float]
Polygon = tuple[Point, ...]


class GeometryError(ValueError):
    """Raised when a geometric input is invalid."""


def _finite_tuple(values: Sequence[float], expected: int, label: str) -> tuple[float, ...]:
    try:
        converted = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise GeometryError(f"{label} must contain numeric values") from exc
    if len(converted) != expected or not all(math.isfinite(value) for value in converted):
        raise GeometryError(f"{label} must contain {expected} finite values")
    return converted


def validate_bbox(bbox: Sequence[float]) -> BoundingBox:
    """Validate and normalize an XYXY bounding box."""

    x1, y1, x2, y2 = _finite_tuple(bbox, 4, "bbox")
    if x2 < x1 or y2 < y1:
        raise GeometryError("bbox must satisfy x1 <= x2 and y1 <= y2")
    return (x1, y1, x2, y2)


def validate_polygon(polygon: Sequence[Sequence[float]]) -> Polygon:
    """Validate and normalize a polygon with at least three vertices."""

    try:
        points = tuple(
            _finite_tuple(point, 2, "polygon point")
            for point in polygon
        )
    except TypeError as exc:
        raise GeometryError("polygon must be an iterable of points") from exc
    if len(points) < 3:
        raise GeometryError("polygon must contain at least three points")
    return tuple((point[0], point[1]) for point in points)


def bbox_center(bbox: Sequence[float]) -> Point:
    """Return the geometric center of an XYXY bounding box."""

    x1, y1, x2, y2 = validate_bbox(bbox)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bottom_center(bbox: Sequence[float]) -> Point:
    """Return the bottom-center point used by the upstream zone demo."""

    x1, _, x2, y2 = validate_bbox(bbox)
    return ((x1 + x2) / 2.0, y2)


def euclidean_distance(point_a: Sequence[float], point_b: Sequence[float]) -> float:
    """Return straight-line distance between two image points in pixels."""

    ax, ay = _finite_tuple(point_a, 2, "point_a")
    bx, by = _finite_tuple(point_b, 2, "point_b")
    return math.hypot(bx - ax, by - ay)


def bbox_intersection(
    bbox_a: Sequence[float],
    bbox_b: Sequence[float],
) -> BoundingBox | None:
    """Return the positive-area intersection of two boxes, if one exists."""

    ax1, ay1, ax2, ay2 = validate_bbox(bbox_a)
    bx1, by1, bx2, by2 = validate_bbox(bbox_b)
    intersection = (max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2))
    if intersection[2] <= intersection[0] or intersection[3] <= intersection[1]:
        return None
    return intersection


def bbox_iou(bbox_a: Sequence[float], bbox_b: Sequence[float]) -> float:
    """Return Intersection over Union for two bounding boxes."""

    a = validate_bbox(bbox_a)
    b = validate_bbox(bbox_b)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    intersection = bbox_intersection(a, b)
    if intersection is None:
        return 0.0
    intersection_area = (intersection[2] - intersection[0]) * (
        intersection[3] - intersection[1]
    )
    union = area_a + area_b - intersection_area
    return intersection_area / union if union > 0.0 else 0.0


def horizontal_overlap(bbox_a: Sequence[float], bbox_b: Sequence[float]) -> float:
    """Return horizontal intersection length in pixels."""

    ax1, _, ax2, _ = validate_bbox(bbox_a)
    bx1, _, bx2, _ = validate_bbox(bbox_b)
    return max(0.0, min(ax2, bx2) - max(ax1, bx1))


def vertical_overlap(bbox_a: Sequence[float], bbox_b: Sequence[float]) -> float:
    """Return vertical intersection length in pixels."""

    _, ay1, _, ay2 = validate_bbox(bbox_a)
    _, by1, _, by2 = validate_bbox(bbox_b)
    return max(0.0, min(ay2, by2) - max(ay1, by1))


def support_ratio(
    supported_bbox: Sequence[float],
    support_bbox: Sequence[float],
) -> float:
    """Return supported-box width covered by the support box horizontally.

    This is a geometric overlap ratio only. It does not prove that one object
    is physically resting on another; vertical ordering and contact reasoning
    belong to later modules.
    """

    supported = validate_bbox(supported_bbox)
    width = supported[2] - supported[0]
    if width <= 0.0:
        raise GeometryError("supported_bbox must have positive width")
    return min(1.0, horizontal_overlap(supported, support_bbox) / width)


def relative_above(bbox_a: Sequence[float], bbox_b: Sequence[float]) -> bool:
    """Return whether box A's center is above box B's center in image space."""

    return bbox_center(bbox_a)[1] < bbox_center(bbox_b)[1]


def relative_below(bbox_a: Sequence[float], bbox_b: Sequence[float]) -> bool:
    """Return whether box A's center is below box B's center in image space."""

    return bbox_center(bbox_a)[1] > bbox_center(bbox_b)[1]


def relative_left(bbox_a: Sequence[float], bbox_b: Sequence[float]) -> bool:
    """Return whether box A's center is left of box B's center."""

    return bbox_center(bbox_a)[0] < bbox_center(bbox_b)[0]


def relative_right(bbox_a: Sequence[float], bbox_b: Sequence[float]) -> bool:
    """Return whether box A's center is right of box B's center."""

    return bbox_center(bbox_a)[0] > bbox_center(bbox_b)[0]


def point_inside_polygon(
    point: Sequence[float],
    polygon: Sequence[Sequence[float]],
) -> bool:
    """Test a point against a polygon using OpenCV's boundary-inclusive test."""

    x, y = _finite_tuple(point, 2, "point")
    normalized_polygon = validate_polygon(polygon)
    contour = np.asarray(normalized_polygon, dtype=np.float32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(contour, (x, y), False) >= 0.0


def distance_to_line(
    point: Sequence[float],
    line_start: Sequence[float],
    line_end: Sequence[float],
) -> float:
    """Return perpendicular distance from a point to a finite line segment."""

    px, py = _finite_tuple(point, 2, "point")
    ax, ay = _finite_tuple(line_start, 2, "line_start")
    bx, by = _finite_tuple(line_end, 2, "line_end")
    dx = bx - ax
    dy = by - ay
    length_squared = dx * dx + dy * dy
    if length_squared <= 0.0:
        raise GeometryError("line_start and line_end must be different points")
    projection = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_squared))
    closest = (ax + projection * dx, ay + projection * dy)
    return euclidean_distance((px, py), closest)


def distance_to_region(
    point: Sequence[float],
    polygon: Sequence[Sequence[float]],
) -> float:
    """Return zero inside a polygon, otherwise distance to its boundary."""

    normalized_polygon = validate_polygon(polygon)
    if point_inside_polygon(point, normalized_polygon):
        return 0.0
    distances = [
        distance_to_line(
            point,
            normalized_polygon[index],
            normalized_polygon[(index + 1) % len(normalized_polygon)],
        )
        for index in range(len(normalized_polygon))
    ]
    return min(distances)

