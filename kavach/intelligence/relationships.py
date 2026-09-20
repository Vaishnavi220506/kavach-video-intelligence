"""Explainable spatial and temporal relation derivation for Module 6."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING

from ..perception.tracker import TrackedObject
from .geometry import (
    bbox_intersection,
    bbox_iou,
    bottom_center,
    euclidean_distance,
    relative_above,
    relative_left,
    support_ratio,
    validate_bbox,
)
from .zones import ZoneManager

if TYPE_CHECKING:
    from .object_memory import ObjectMemory


class RelationshipError(ValueError):
    """Raised when relation inputs or thresholds are invalid."""


NEAR = "near"
INSIDE_ZONE = "inside_zone"
ABOVE = "above"
BELOW = "below"
LEFT_OF = "left_of"
RIGHT_OF = "right_of"
OVERLAPPING = "overlapping"
SUPPORTED_BY = "supported_by"
MOVING_WITH = "moving_with"
APPROACHING = "approaching"
SEPARATING_FROM = "separating_from"

RELATION_NAMES = (
    NEAR,
    INSIDE_ZONE,
    ABOVE,
    BELOW,
    LEFT_OF,
    RIGHT_OF,
    OVERLAPPING,
    SUPPORTED_BY,
    MOVING_WITH,
    APPROACHING,
    SEPARATING_FROM,
)


def _normalise_label(value: object) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    return normalized.strip("_") or "object"


def object_node_id(track_id: int, class_name: str) -> str:
    """Return a stable readable node ID such as ``carton_7``."""

    return f"{_normalise_label(class_name)}_{int(track_id)}"


def zone_node_id(name: str) -> str:
    """Return a readable zone node ID such as ``staging_zone``."""

    return _normalise_label(name)


@dataclass(frozen=True)
class RelationThresholds:
    """Explicit thresholds for the explainable relation rules."""

    near_distance_px: float = 100.0
    overlap_iou_threshold: float = 0.01
    support_min_horizontal_overlap_ratio: float = 0.50
    support_max_vertical_gap_px: float = 20.0
    moving_with_max_distance_px: float = 120.0
    moving_with_velocity_delta_px_per_second: float = 25.0
    moving_with_min_speed_px_per_second: float = 2.0
    motion_min_distance_change_px: float = 2.0
    support_provider_classes: tuple[str, ...] = (
        "pallet",
        "platform",
        "shelf",
        "floor",
        "trolley",
        "pallet_truck",
    )

    def __post_init__(self) -> None:
        non_negative = (
            "near_distance_px",
            "support_max_vertical_gap_px",
            "moving_with_max_distance_px",
            "moving_with_velocity_delta_px_per_second",
            "moving_with_min_speed_px_per_second",
            "motion_min_distance_change_px",
        )
        for field_name in non_negative:
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise RelationshipError(f"{field_name} must be finite and non-negative")
        for field_name in (
            "overlap_iou_threshold",
            "support_min_horizontal_overlap_ratio",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise RelationshipError(f"{field_name} must be between 0 and 1")
        normalized = tuple(_normalise_label(value) for value in self.support_provider_classes)
        object.__setattr__(self, "support_provider_classes", normalized)


@dataclass(frozen=True)
class RelationEdge:
    """One directed graph edge with numeric evidence for debugging."""

    subject: str
    relation: str
    object: str
    timestamp: float
    evidence: Mapping[str, float]

    def __post_init__(self) -> None:
        if not str(self.subject).strip() or not str(self.object).strip():
            raise RelationshipError("relation endpoints cannot be empty")
        if str(self.relation) not in RELATION_NAMES:
            raise RelationshipError(f"unknown relation: {self.relation}")
        timestamp = float(self.timestamp)
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise RelationshipError("relation timestamp must be finite and non-negative")
        if not self.evidence:
            raise RelationshipError("every relation must include numeric evidence")
        normalized_evidence: dict[str, float] = {}
        for key, value in self.evidence.items():
            if not isinstance(value, Real) or isinstance(value, bool):
                raise RelationshipError("relation evidence values must be numeric")
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise RelationshipError("relation evidence must be finite")
            normalized_evidence[str(key)] = numeric_value
        object.__setattr__(self, "subject", str(self.subject))
        object.__setattr__(self, "relation", str(self.relation))
        object.__setattr__(self, "object", str(self.object))
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "evidence", normalized_evidence)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly edge representation."""

        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "timestamp": self.timestamp,
            "evidence": dict(self.evidence),
        }


def _edge(
    subject: str,
    relation: str,
    object_name: str,
    timestamp: float,
    evidence: Mapping[str, float],
) -> RelationEdge:
    return RelationEdge(
        subject=subject,
        relation=relation,
        object=object_name,
        timestamp=timestamp,
        evidence=evidence,
    )


def _append_symmetric(
    edges: list[RelationEdge],
    first: str,
    relation: str,
    second: str,
    timestamp: float,
    evidence: Mapping[str, float],
) -> None:
    """Represent a symmetric fact as two directed edges."""

    edges.append(_edge(first, relation, second, timestamp, evidence))
    edges.append(_edge(second, relation, first, timestamp, evidence))


def _previous_state(memory: ObjectMemory | None, track_id: int):
    if memory is None:
        return None
    history = memory.get_history(track_id)
    return history[-2] if len(history) >= 2 else None


def _motion(memory: ObjectMemory | None, track_id: int):
    if memory is None:
        return None
    try:
        return memory.get_velocity(track_id)
    except (KeyError, ValueError, RuntimeError):
        return None


def _validate_objects(objects: tuple[TrackedObject, ...]) -> None:
    seen: set[int] = set()
    for tracked in objects:
        track_id = int(tracked.track_id)
        if track_id in seen:
            raise RelationshipError(f"duplicate current track ID: {track_id}")
        seen.add(track_id)
        try:
            validate_bbox(tracked.bbox)
            euclidean_distance(tracked.center, tracked.center)
        except ValueError as exc:
            raise RelationshipError(f"invalid state for track {track_id}") from exc


def _timestamp(objects: tuple[TrackedObject, ...], timestamp: float | None) -> float:
    value = timestamp
    if value is None:
        value = 0.0 if not objects else float(objects[0].timestamp)
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise RelationshipError("timestamp must be finite and non-negative")
    return value


def build_relations(
    tracked_objects: Iterable[TrackedObject],
    *,
    timestamp: float | None = None,
    zones: ZoneManager | None = None,
    memory: ObjectMemory | None = None,
    thresholds: RelationThresholds | None = None,
) -> list[RelationEdge]:
    """Derive current directed relations from boxes, zones, and memory.

    The function performs no learned relation classification. Current spatial
    edges come from geometry thresholds; approaching, separating, and moving
    with use timestamped states from the supplied Module 4 ObjectMemory.
    """

    objects = tuple(tracked_objects)
    _validate_objects(objects)
    threshold = thresholds or RelationThresholds()
    current_timestamp = _timestamp(objects, timestamp)
    edges: list[RelationEdge] = []

    node_ids = {
        int(tracked.track_id): object_node_id(tracked.track_id, tracked.class_name)
        for tracked in objects
    }

    if zones is not None:
        for tracked in objects:
            point_x, point_y = bottom_center(tracked.bbox)
            subject = node_ids[int(tracked.track_id)]
            for zone in zones.zones:
                if zone.contains((point_x, point_y)):
                    edges.append(
                        _edge(
                            subject,
                            INSIDE_ZONE,
                            zone_node_id(zone.name),
                            current_timestamp,
                            {
                                "inside": 1.0,
                                "bottom_center_x": point_x,
                                "bottom_center_y": point_y,
                                "boundary_distance_px": zone.distance_to_boundary(
                                    (point_x, point_y)
                                ),
                            },
                        )
                    )

    for index, first in enumerate(objects):
        first_id = node_ids[int(first.track_id)]
        for second in objects[index + 1 :]:
            second_id = node_ids[int(second.track_id)]
            distance_px = euclidean_distance(first.center, second.center)

            if distance_px <= threshold.near_distance_px:
                _append_symmetric(
                    edges,
                    first_id,
                    NEAR,
                    second_id,
                    current_timestamp,
                    {
                        "distance_px": distance_px,
                        "threshold_px": threshold.near_distance_px,
                        "margin_px": threshold.near_distance_px - distance_px,
                    },
                )

            iou = bbox_iou(first.bbox, second.bbox)
            if iou >= threshold.overlap_iou_threshold and iou > 0.0:
                intersection = bbox_intersection(first.bbox, second.bbox)
                assert intersection is not None
                _append_symmetric(
                    edges,
                    first_id,
                    OVERLAPPING,
                    second_id,
                    current_timestamp,
                    {
                        "iou": iou,
                        "intersection_width_px": intersection[2] - intersection[0],
                        "intersection_height_px": intersection[3] - intersection[1],
                    },
                )

            direction_evidence = {
                "center_delta_x_px": second.center[0] - first.center[0],
                "center_delta_y_px": second.center[1] - first.center[1],
            }
            if relative_above(first.bbox, second.bbox):
                edges.append(_edge(first_id, ABOVE, second_id, current_timestamp, direction_evidence))
                edges.append(_edge(second_id, BELOW, first_id, current_timestamp, direction_evidence))
            elif relative_above(second.bbox, first.bbox):
                edges.append(_edge(second_id, ABOVE, first_id, current_timestamp, direction_evidence))
                edges.append(_edge(first_id, BELOW, second_id, current_timestamp, direction_evidence))

            if relative_left(first.bbox, second.bbox):
                edges.append(_edge(first_id, LEFT_OF, second_id, current_timestamp, direction_evidence))
                edges.append(_edge(second_id, RIGHT_OF, first_id, current_timestamp, direction_evidence))
            elif relative_left(second.bbox, first.bbox):
                edges.append(_edge(second_id, LEFT_OF, first_id, current_timestamp, direction_evidence))
                edges.append(_edge(first_id, RIGHT_OF, second_id, current_timestamp, direction_evidence))

            first_class = _normalise_label(first.class_name)
            second_class = _normalise_label(second.class_name)
            if (
                second_class in threshold.support_provider_classes
                and relative_above(first.bbox, second.bbox)
            ):
                horizontal_ratio = support_ratio(first.bbox, second.bbox)
                vertical_gap = max(0.0, second.bbox[1] - first.bbox[3])
                if (
                    horizontal_ratio >= threshold.support_min_horizontal_overlap_ratio
                    and vertical_gap <= threshold.support_max_vertical_gap_px
                ):
                    edges.append(
                        _edge(
                            first_id,
                            SUPPORTED_BY,
                            second_id,
                            current_timestamp,
                            {
                                "horizontal_overlap_px": max(
                                    0.0,
                                    min(first.bbox[2], second.bbox[2])
                                    - max(first.bbox[0], second.bbox[0]),
                                ),
                                "horizontal_overlap_ratio": horizontal_ratio,
                                "vertical_gap_px": vertical_gap,
                            },
                        )
                    )
            if (
                first_class in threshold.support_provider_classes
                and relative_above(second.bbox, first.bbox)
            ):
                horizontal_ratio = support_ratio(second.bbox, first.bbox)
                vertical_gap = max(0.0, first.bbox[1] - second.bbox[3])
                if (
                    horizontal_ratio >= threshold.support_min_horizontal_overlap_ratio
                    and vertical_gap <= threshold.support_max_vertical_gap_px
                ):
                    edges.append(
                        _edge(
                            second_id,
                            SUPPORTED_BY,
                            first_id,
                            current_timestamp,
                            {
                                "horizontal_overlap_px": max(
                                    0.0,
                                    min(second.bbox[2], first.bbox[2])
                                    - max(second.bbox[0], first.bbox[0]),
                                ),
                                "horizontal_overlap_ratio": horizontal_ratio,
                                "vertical_gap_px": vertical_gap,
                            },
                        )
                    )

            first_previous = _previous_state(memory, first.track_id)
            second_previous = _previous_state(memory, second.track_id)
            if first_previous is not None and second_previous is not None:
                previous_distance_px = euclidean_distance(
                    first_previous.center,
                    second_previous.center,
                )
                distance_change_px = distance_px - previous_distance_px
                first_dt = float(first.timestamp) - float(first_previous.timestamp)
                second_dt = float(second.timestamp) - float(second_previous.timestamp)
                elapsed = min(first_dt, second_dt)
                if elapsed > 0.0 and abs(distance_change_px) >= threshold.motion_min_distance_change_px:
                    evidence = {
                        "previous_distance_px": previous_distance_px,
                        "current_distance_px": distance_px,
                        "distance_change_px": distance_change_px,
                        "interval_seconds": elapsed,
                        "closing_rate_px_per_second": -distance_change_px / elapsed,
                    }
                    if distance_change_px < 0.0:
                        _append_symmetric(
                            edges,
                            first_id,
                            APPROACHING,
                            second_id,
                            current_timestamp,
                            evidence,
                        )
                    else:
                        _append_symmetric(
                            edges,
                            first_id,
                            SEPARATING_FROM,
                            second_id,
                            current_timestamp,
                            evidence,
                        )

            first_motion = _motion(memory, first.track_id)
            second_motion = _motion(memory, second.track_id)
            if first_motion is not None and second_motion is not None:
                velocity_delta = euclidean_distance(
                    (
                        first_motion.velocity_x_pixels_per_second,
                        first_motion.velocity_y_pixels_per_second,
                    ),
                    (
                        second_motion.velocity_x_pixels_per_second,
                        second_motion.velocity_y_pixels_per_second,
                    ),
                )
                if (
                    distance_px <= threshold.moving_with_max_distance_px
                    and first_motion.speed_pixels_per_second
                    >= threshold.moving_with_min_speed_px_per_second
                    and second_motion.speed_pixels_per_second
                    >= threshold.moving_with_min_speed_px_per_second
                    and velocity_delta
                    <= threshold.moving_with_velocity_delta_px_per_second
                ):
                    _append_symmetric(
                        edges,
                        first_id,
                        MOVING_WITH,
                        second_id,
                        current_timestamp,
                        {
                            "distance_px": distance_px,
                            "velocity_delta_px_per_second": velocity_delta,
                            "subject_speed_px_per_second": first_motion.speed_pixels_per_second,
                            "object_speed_px_per_second": second_motion.speed_pixels_per_second,
                        },
                    )

    return edges
