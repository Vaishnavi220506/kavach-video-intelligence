"""Readable, evidence-backed overlays for processed KAVACH videos."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import re

import cv2
import numpy as np

from ..behaviours.base import BehaviourEvent
from ..intelligence.relationships import object_node_id, zone_node_id
from ..intelligence.zones import ZoneManager
from ..perception.tracker import TrackedObject


ANOMALY_COLOR = (55, 85, 235)  # BGR: clear red
SAFETY_COLOR = (20, 145, 245)  # BGR: orange-red
ACTIVITY_COLOR = (220, 145, 35)  # BGR: blue
PANEL_COLOR = (12, 22, 38)
WHITE = (245, 248, 252)
MUTED = (190, 205, 220)


@dataclass(frozen=True)
class EventExplanation:
    """Short human-readable copy derived only from structured event evidence."""

    title: str
    reason: str
    detail: str
    color: tuple[int, int, int]
    kind: str


def _pretty(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).replace("_", " ").strip()).title()


def _canonical(value: object) -> str:
    return "_".join(re.sub(r"[^a-zA-Z0-9]+", " ", str(value)).lower().split())


def _number(evidence: Mapping[str, object], key: str, default: float = 0.0) -> float:
    value = evidence.get(key, default)
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _yes(value: object) -> str:
    return "yes" if bool(value) else "no"


def _px(value: float) -> str:
    return f"{value:.0f} px"


def _seconds(value: float) -> str:
    return f"{value:.1f}s"


def _event_kind(event_type: str) -> tuple[str, tuple[int, int, int]]:
    if event_type == "MOTION_ANOMALY":
        return "anomaly", ANOMALY_COLOR
    if event_type in {"OBJECT_ACTIVITY", "ZONE_TRANSITION"}:
        return "activity", ACTIVITY_COLOR
    return "safety", SAFETY_COLOR


def explain_event(event: BehaviourEvent) -> EventExplanation:
    """Translate one event into concise overlay copy without adding facts."""

    evidence = event.evidence
    event_type = str(event.type)
    kind, color = _event_kind(event_type)
    entities = ", ".join(event.entities) if event.entities else "the tracked object"

    if event_type == "MOTION_ANOMALY":
        reason = str(evidence.get("reason", "unusual motion")).replace("+", " and ")
        speed = _number(evidence, "speed_px_per_second")
        baseline = _number(evidence, "baseline_speed_px_per_second")
        ratio = _number(evidence, "speed_ratio")
        change = _number(evidence, "direction_change_degrees")
        detail_parts = []
        if speed:
            detail_parts.append(f"speed {_px(speed)}")
        if baseline:
            detail_parts.append(f"baseline {_px(baseline)}")
        if ratio > 1.0:
            detail_parts.append(f"{ratio:.1f}x baseline")
        if change:
            detail_parts.append(f"direction change {change:.0f} deg")
        detail = " | ".join(detail_parts) or "recent track motion changed abruptly"
        return EventExplanation(
            "ANOMALY | MOTION ANOMALY",
            f"{_pretty(entities)} showed {reason}.",
            f"Why flagged: {detail}. Image-space signal only; not proof of damage.",
            color,
            kind,
        )

    if event_type == "ZONE_VIOLATION":
        zone = _pretty(evidence.get("zone_name", "configured restricted zone"))
        boundary = _number(evidence, "boundary_distance_px")
        detail = f"bottom-center point is inside {zone.lower()}"
        if boundary:
            detail += f" | boundary distance {_px(boundary)}"
        return EventExplanation(
            "SAFETY SIGNAL | ZONE VIOLATION",
            f"{_pretty(entities)} entered a restricted area.",
            f"Why flagged: {detail}.",
            color,
            kind,
        )

    if event_type == "ZONE_TRANSITION":
        transition = _pretty(evidence.get("transition", "changed zone")).lower()
        zone = _pretty(evidence.get("zone_name", "configured zone"))
        return EventExplanation(
            "ACTIVITY | ZONE TRANSITION",
            f"{_pretty(entities)} {transition} {zone.lower()}.",
            "Why shown: polygon membership changed. This is context, not a safety violation by itself.",
            color,
            kind,
        )

    if event_type == "OBJECT_ACTIVITY":
        state = _pretty(evidence.get("activity_state", "changed state")).lower()
        speed = _number(evidence, "speed_px_per_second")
        detail = f"speed {_px(speed)}" if speed else "timestamped track movement"
        return EventExplanation(
            "ACTIVITY | OBJECT ACTIVITY",
            f"{_pretty(entities)} is {state}.",
            f"Why shown: {detail}. This is context, not a safety violation by itself.",
            color,
            kind,
        )

    if event_type == "POSSIBLE_DRAGGING":
        displacement = _number(evidence, "horizontal_displacement_px")
        duration = _number(evidence, "duration_seconds")
        vertical = _number(evidence, "vertical_displacement_px")
        return EventExplanation(
            "SAFETY SIGNAL | POSSIBLE DRAGGING",
            f"{_pretty(entities)} moved horizontally near the floor.",
            f"Why flagged: {_px(displacement)} horizontal movement | {_seconds(duration)} | {_px(vertical)} vertical drift | person nearby {_yes(evidence.get('person_nearby'))}.",
            color,
            kind,
        )

    if event_type == "POSSIBLE_DROP":
        downward = _number(evidence, "downward_speed_px_per_second")
        displacement = _number(evidence, "downward_displacement_px")
        return EventExplanation(
            "SAFETY SIGNAL | POSSIBLE DROP",
            f"{_pretty(entities)} shows a move, separation, downward-motion, and settling pattern.",
            f"Why flagged: downward {_px(downward)}/s | drop displacement {_px(displacement)} | physical impact is not established.",
            color,
            kind,
        )

    if event_type == "POSSIBLE_THROWING":
        speed = _number(evidence, "speed_px_per_second")
        acceleration = _number(evidence, "acceleration_px_per_second_squared")
        return EventExplanation(
            "SAFETY SIGNAL | POSSIBLE THROWING",
            f"{_pretty(entities)} shows a high-speed release-like motion.",
            f"Why flagged: speed {_px(speed)}/s | acceleration {acceleration:.0f} px/s^2 | image-space evidence only; intent is not established.",
            color,
            kind,
        )

    if event_type == "POSSIBLE_ROUGH_HANDLING":
        speed = _number(evidence, "speed_px_per_second")
        acceleration = _number(evidence, "acceleration_px_per_second_squared")
        change = _number(evidence, "direction_change_degrees")
        return EventExplanation(
            "SAFETY SIGNAL | POSSIBLE ROUGH HANDLING",
            f"{_pretty(entities)} shows abrupt handling motion.",
            f"Why flagged: speed {_px(speed)}/s | acceleration {acceleration:.0f} px/s^2 | direction change {change:.0f} deg.",
            color,
            kind,
        )

    if event_type == "AISLE_OBSTRUCTION":
        zone = _pretty(evidence.get("zone_name", "configured aisle"))
        duration = _number(evidence, "stationary_duration_seconds")
        return EventExplanation(
            "SAFETY SIGNAL | AISLE OBSTRUCTION",
            f"{_pretty(entities)} remained in {zone.lower()}.",
            f"Why flagged: stationary for {_seconds(duration)} inside a configured aisle polygon.",
            color,
            kind,
        )

    if event_type == "IMPROPER_PLACEMENT":
        duration = _number(evidence, "stationary_duration_seconds")
        return EventExplanation(
            "SAFETY SIGNAL | IMPROPER PLACEMENT",
            f"{_pretty(entities)} settled outside an approved placement zone.",
            f"Why flagged: stationary for {_seconds(duration)} | allowed-zone membership was false.",
            color,
            kind,
        )

    if event_type == "COLLISION_RISK":
        distance = _number(evidence, "distance_px")
        closing = _number(evidence, "closing_speed_px_per_second")
        return EventExplanation(
            "SAFETY SIGNAL | COLLISION RISK",
            f"{_pretty(entities)} are close and moving toward one another.",
            f"Why flagged: distance {_px(distance)} | closing speed {_px(closing)}/s | image-space approximation only.",
            color,
            kind,
        )

    if event_type == "UNSAFE_HUMAN_FORKLIFT_PROXIMITY":
        distance = _number(evidence, "distance_px")
        return EventExplanation(
            "SAFETY SIGNAL | HUMAN-FORKLIFT PROXIMITY",
            f"{_pretty(entities)} entered the configured proximity threshold.",
            f"Why flagged: distance {_px(distance)} | motion present {_yes(evidence.get('motion_present'))} | image-space approximation only.",
            color,
            kind,
        )

    if event_type == "PALLET_OVERHANG":
        ratio = _number(evidence, "support_ratio")
        return EventExplanation(
            "SAFETY SIGNAL | PALLET OVERHANG",
            f"{_pretty(entities)} has weak horizontal support.",
            f"Why flagged: support ratio {ratio:.2f} | configured minimum was {_number(evidence, 'minimum_support_ratio'):.2f}.",
            color,
            kind,
        )

    if event_type == "UNSTABLE_STACK":
        ratio = _number(evidence, "minimum_support_ratio")
        height = _number(evidence, "stack_height_objects")
        return EventExplanation(
            "SAFETY SIGNAL | UNSTABLE STACK",
            f"{_pretty(entities)} forms a stack with a weak support link.",
            f"Why flagged: {height:.0f} objects | weakest support ratio {ratio:.2f}.",
            color,
            kind,
        )

    keys = ", ".join(str(key).replace("_", " ") for key in list(evidence)[:3])
    return EventExplanation(
        f"SAFETY SIGNAL | {_pretty(event_type)}",
        f"{_pretty(entities)} triggered a configured review rule.",
        f"Why flagged: structured evidence includes {keys or 'the recorded rule evidence'}.",
        color,
        kind,
    )


def _truncate(text: str, maximum: int = 150) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    return clean if len(clean) <= maximum else clean[: maximum - 3].rstrip() + "..."


def _put_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    color: tuple[int, int, int],
    scale: float = 0.50,
    thickness: int = 2,
) -> tuple[int, int, int, int]:
    (width, height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness
    )
    x, y = origin
    y = max(height + baseline + 4, y)
    x = max(4, min(x, image.shape[1] - width - 8))
    cv2.rectangle(
        image,
        (x - 3, y - height - baseline - 4),
        (x + width + 5, y + 3),
        PANEL_COLOR,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x, y - 2),
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return (x - 3, y - height - baseline - 4, x + width + 5, y + 3)


def _rectangles_overlap(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
    padding: int = 4,
) -> bool:
    return not (
        first[2] + padding < second[0]
        or second[2] + padding < first[0]
        or first[3] + padding < second[1]
        or second[3] + padding < first[1]
    )


def _draw_flagged_label(
    image: np.ndarray,
    text: str,
    tracked: TrackedObject,
    *,
    banner_height: int,
    color: tuple[int, int, int],
    occupied: list[tuple[int, int, int, int]],
) -> None:
    """Place one flagged-object label near its box without stacking labels."""

    x1, y1, x2, y2 = (int(round(value)) for value in tracked.bbox)
    preferred_y = y1 - 8 if y1 > banner_height else y2 + 22
    candidates = [
        (x1, preferred_y),
        (x1, y2 + 22),
        (max(4, x1 - 90), preferred_y),
        (x2 + 12, preferred_y),
        (x1, preferred_y - 30),
    ]
    chosen: tuple[int, int] | None = None
    for candidate in candidates:
        (width, height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_DUPLEX, 0.50, 2
        )
        x = max(4, min(candidate[0], image.shape[1] - width - 8))
        y = max(height + baseline + 4, candidate[1])
        rect = (x - 3, y - height - baseline - 4, x + width + 5, y + 3)
        if not any(_rectangles_overlap(rect, previous) for previous in occupied):
            chosen = candidate
            break
    # If every nearby slot is occupied, leave the circle/bounding box clean
    # rather than drawing a misleading pile of text. The banner still names
    # the event and its entities.
    if chosen is not None:
        occupied.append(_put_label(image, text, chosen, color=color))


def _compact_banner_explanations(
    explanations: Sequence[EventExplanation],
    maximum: int,
) -> list[EventExplanation]:
    """Merge duplicate explanations so busy frames remain readable."""

    compacted: list[EventExplanation] = []
    counts: dict[tuple[str, str], int] = {}
    for explanation in explanations:
        key = (explanation.title, explanation.detail)
        counts[key] = counts.get(key, 0) + 1
        if any(
            item.title == explanation.title and item.detail == explanation.detail
            for item in compacted
        ):
            continue
        compacted.append(explanation)

    result: list[EventExplanation] = []
    for explanation in compacted[: max(1, int(maximum))]:
        count = counts[(explanation.title, explanation.detail)]
        if count > 1:
            explanation = EventExplanation(
                explanation.title,
                f"{explanation.reason.rstrip('.')} + {count - 1} similar tracked object(s).",
                explanation.detail,
                explanation.color,
                explanation.kind,
            )
        result.append(explanation)
    return result


def draw_explainable_overlays(
    frame: np.ndarray,
    tracked_objects: Sequence[TrackedObject],
    zones: ZoneManager,
    events: Sequence[BehaviourEvent] = (),
    *,
    max_banner_events: int = 2,
) -> np.ndarray:
    """Highlight event entities and write the evidence-based reason on-frame."""

    annotated = frame.copy()
    if not events:
        return annotated

    priority = {"safety": 0, "anomaly": 1, "activity": 2}
    event_explanations = [explain_event(event) for event in events]
    explanations = sorted(
        event_explanations,
        key=lambda item: priority.get(item.kind, 3),
    )
    entity_to_color: dict[str, tuple[int, int, int]] = {}
    event_entities: set[str] = set()
    zone_entities: set[str] = set()
    for event, explanation in zip(events, event_explanations):
        # Activity and zone-transition signals are useful context, but they
        # are not violations. Keep them in the banner when no safety signal
        # exists, without covering the frame with extra blue circles/labels.
        if explanation.kind == "activity":
            continue
        for entity in event.entities:
            canonical = _canonical(entity)
            event_entities.add(canonical)
            entity_to_color[canonical] = explanation.color
            if zones.get_zone(str(entity).upper()) is not None or any(
                _canonical(zone.name) == canonical for zone in zones.zones
            ):
                zone_entities.add(canonical)

    tracked_by_entity = {
        _canonical(object_node_id(item.track_id, item.class_name)): item
        for item in tracked_objects
    }
    highlighted: dict[str, TrackedObject] = {}
    for entity in sorted(event_entities):
        item = tracked_by_entity.get(entity)
        if item is not None:
            highlighted[entity] = item

    for entity, tracked in sorted(
        highlighted.items(),
        key=lambda item: (item[1].bbox[1], item[1].bbox[0], item[0]),
    ):
        color = entity_to_color.get(entity, SAFETY_COLOR)
        x1, y1, x2, y2 = (int(round(value)) for value in tracked.bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 4, cv2.LINE_AA)
        center = (int(round(tracked.center[0])), int(round(tracked.center[1])))
        axes = (
            max(22, abs(x2 - x1) // 2 + 14),
            max(28, abs(y2 - y1) // 2 + 14),
        )
        cv2.ellipse(annotated, center, axes, 0.0, 0.0, 360.0, color, 3, cv2.LINE_AA)
        cv2.circle(annotated, center, 5, color, -1, cv2.LINE_AA)

    # Make the relevant zone boundary unmistakable for zone violations and
    # zone transitions. Normal zone overlays remain visible underneath.
    for zone in zones.zones:
        if _canonical(zone.name) not in zone_entities:
            continue
        points = np.asarray(zone.polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [points], True, SAFETY_COLOR, 5, cv2.LINE_AA)

    # A line between two involved objects makes proximity/collision evidence
    # visually inspectable instead of leaving the relationship as text only.
    for event in events:
        if event.type not in {"COLLISION_RISK", "UNSAFE_HUMAN_FORKLIFT_PROXIMITY"}:
            continue
        points = [
            tracked_by_entity[_canonical(entity)].center
            for entity in event.entities
            if _canonical(entity) in tracked_by_entity
        ]
        if len(points) >= 2:
            first = tuple(int(round(value)) for value in points[0])
            second = tuple(int(round(value)) for value in points[1])
            cv2.line(annotated, first, second, SAFETY_COLOR, 4, cv2.LINE_AA)

    width = annotated.shape[1]
    safety_explanations = [
        explanation for explanation in explanations if explanation.kind != "activity"
    ]
    banner_source = safety_explanations or explanations
    banner_lines = _compact_banner_explanations(banner_source, max_banner_events)
    banner_height = 96 + max(0, len(banner_lines) - 1) * 52
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (width, banner_height), PANEL_COLOR, -1)
    annotated = cv2.addWeighted(overlay, 0.92, annotated, 0.08, 0.0)
    cv2.line(annotated, (0, banner_height), (width, banner_height), SAFETY_COLOR, 2)
    y = 27
    for explanation in banner_lines:
        cv2.putText(
            annotated,
            _truncate(explanation.title, 74),
            (18, y),
            cv2.FONT_HERSHEY_DUPLEX,
            0.64,
            explanation.color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            _truncate(explanation.reason, 122),
            (18, y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            WHITE,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            _truncate(explanation.detail, 150),
            (18, y + 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            MUTED,
            1,
            cv2.LINE_AA,
        )
        y += 52

    remaining = len(explanations) - len(banner_lines)
    if remaining > 0:
        cv2.putText(
            annotated,
            f"+ {remaining} more signal(s) at this timestamp",
            (width - 280, banner_height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            MUTED,
            1,
            cv2.LINE_AA,
        )

    # The banner is intentionally translucent, but never let it obscure the
    # thing being flagged. Reassert the circles and bounding boxes after the
    # copy has been composited so an object near the top of the frame stays
    # unmistakable in the rendered video.
    # Reassert the zone and relationship guides first; object labels are the
    # final layer so a polygon edge cannot slice through their text.
    for zone in zones.zones:
        if _canonical(zone.name) not in zone_entities:
            continue
        points = np.asarray(zone.polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [points], True, SAFETY_COLOR, 5, cv2.LINE_AA)

    for event in events:
        if event.type not in {"COLLISION_RISK", "UNSAFE_HUMAN_FORKLIFT_PROXIMITY"}:
            continue
        points = [
            tracked_by_entity[_canonical(entity)].center
            for entity in event.entities
            if _canonical(entity) in tracked_by_entity
        ]
        if len(points) >= 2:
            first = tuple(int(round(value)) for value in points[0])
            second = tuple(int(round(value)) for value in points[1])
            cv2.line(annotated, first, second, SAFETY_COLOR, 4, cv2.LINE_AA)

    occupied_labels: list[tuple[int, int, int, int]] = []
    for entity, tracked in sorted(
        highlighted.items(),
        key=lambda item: (item[1].bbox[1], item[1].bbox[0], item[0]),
    ):
        color = entity_to_color.get(entity, SAFETY_COLOR)
        x1, y1, x2, y2 = (int(round(value)) for value in tracked.bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 4, cv2.LINE_AA)
        center = (int(round(tracked.center[0])), int(round(tracked.center[1])))
        axes = (
            max(22, abs(x2 - x1) // 2 + 14),
            max(28, abs(y2 - y1) // 2 + 14),
        )
        cv2.ellipse(annotated, center, axes, 0.0, 0.0, 360.0, color, 3, cv2.LINE_AA)
        cv2.circle(annotated, center, 5, color, -1, cv2.LINE_AA)
        _draw_flagged_label(
            annotated,
            f"FLAGGED {tracked.class_name} #{tracked.track_id}",
            tracked,
            banner_height=banner_height,
            color=color,
            occupied=occupied_labels,
        )

    return annotated


__all__ = ["EventExplanation", "draw_explainable_overlays", "explain_event"]
