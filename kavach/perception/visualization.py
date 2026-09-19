"""KAVACH drawing helpers for frame-local detections."""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from .detector import Detection
from .tracker import TrackedObject


_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 128, 0),
    (0, 200, 255),
    (180, 0, 255),
    (0, 180, 0),
    (255, 0, 0),
)


def _color_for(class_id: int) -> tuple[int, int, int]:
    return _COLORS[class_id % len(_COLORS)]


def draw_detections(
    frame: np.ndarray,
    detections: Iterable[Detection],
    *,
    show_confidence: bool = True,
) -> np.ndarray:
    """Return a copy of a BGR frame with detection boxes and labels drawn."""

    if not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim != 3:
        raise ValueError("frame must be a non-empty color NumPy array")

    annotated = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox)
        color = _color_for(detection.class_id)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = detection.class_name
        if show_confidence:
            label = f"{label} {detection.confidence:.2f}"
        text_y = max(18, y1 - 6)
        cv2.putText(
            annotated,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return annotated


def draw_tracked_objects(
    frame: np.ndarray,
    objects: Iterable[TrackedObject],
    *,
    show_confidence: bool = True,
    show_labels: bool = True,
) -> np.ndarray:
    """Return a copy of a BGR frame with tracked boxes and optional labels."""

    if not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim != 3:
        raise ValueError("frame must be a non-empty color NumPy array")

    annotated = frame.copy()
    for tracked in objects:
        x1, y1, x2, y2 = (int(round(value)) for value in tracked.bbox)
        color = _color_for(tracked.track_id)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        if show_labels:
            label = f"{tracked.class_name} #{tracked.track_id}"
            if show_confidence:
                label = f"{label} {tracked.confidence:.2f}"
            cv2.putText(
                annotated,
                label,
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    return annotated
