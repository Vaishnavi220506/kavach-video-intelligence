"""KAVACH Module 3 multi-object tracking with explicit ByteTrack state."""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker

from .detector import (
    Detection,
    InvalidFrameError,
    WarehouseDetector,
)


class TrackingError(RuntimeError):
    """Base class for expected tracking errors."""


class InvalidTimestampError(TrackingError):
    """Raised when a frame timestamp cannot identify its video time."""


@dataclass(frozen=True)
class ByteTrackConfig:
    """Thresholds used by the explicit Ultralytics BYTETracker adapter."""

    tracker_type: str = "bytetrack"
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.10
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.80
    fuse_score: bool = True

    def __post_init__(self) -> None:
        if self.tracker_type != "bytetrack":
            raise ValueError("ByteTrackConfig.tracker_type must be 'bytetrack'")
        if not 0.0 <= self.track_low_thresh <= self.track_high_thresh <= 1.0:
            raise ValueError(
                "track_low_thresh and track_high_thresh must satisfy "
                "0.0 <= low <= high <= 1.0"
            )
        if not 0.0 <= self.new_track_thresh <= 1.0:
            raise ValueError("new_track_thresh must be between 0.0 and 1.0")
        if self.track_buffer < 0:
            raise ValueError("track_buffer cannot be negative")
        if not 0.0 <= self.match_thresh <= 1.0:
            raise ValueError("match_thresh must be between 0.0 and 1.0")

    def as_namespace(self) -> SimpleNamespace:
        """Return the argument shape expected by Ultralytics BYTETracker."""

        return SimpleNamespace(**asdict(self))

    def as_yaml(self) -> str:
        """Serialize the small tracker configuration without extra packages."""

        return (
            f"tracker_type: {self.tracker_type}\n"
            f"track_high_thresh: {self.track_high_thresh}\n"
            f"track_low_thresh: {self.track_low_thresh}\n"
            f"new_track_thresh: {self.new_track_thresh}\n"
            f"track_buffer: {self.track_buffer}\n"
            f"match_thresh: {self.match_thresh}\n"
            f"fuse_score: {str(self.fuse_score).lower()}\n"
        )


@dataclass(frozen=True)
class TrackedObject:
    """A frame-local observation associated with a persistent track ID."""

    track_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    timestamp: float
    frame_number: int | None = None


class _DetectionBatch:
    """Minimal Results-like input accepted by Ultralytics BYTETracker."""

    def __init__(self, detections: Sequence[Detection]) -> None:
        self.xyxy = np.asarray([d.bbox for d in detections], dtype=np.float32)
        if not len(detections):
            self.xyxy = np.empty((0, 4), dtype=np.float32)
        widths = self.xyxy[:, 2] - self.xyxy[:, 0]
        heights = self.xyxy[:, 3] - self.xyxy[:, 1]
        centers_x = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2.0
        centers_y = (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2.0
        self.xywh = np.column_stack((centers_x, centers_y, widths, heights)).astype(
            np.float32
        )
        self.conf = np.asarray([d.confidence for d in detections], dtype=np.float32)
        self.cls = np.asarray([d.class_id for d in detections], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, index: Any) -> _DetectionBatch:
        selected = object.__new__(_DetectionBatch)
        selected.xyxy = np.atleast_2d(self.xyxy[index]).astype(np.float32)
        selected.xywh = np.atleast_2d(self.xywh[index]).astype(np.float32)
        selected.conf = np.atleast_1d(self.conf[index]).astype(np.float32)
        selected.cls = np.atleast_1d(self.cls[index]).astype(np.float32)
        if len(selected.conf) == 0:
            selected.xyxy = np.empty((0, 4), dtype=np.float32)
            selected.xywh = np.empty((0, 4), dtype=np.float32)
        return selected


def _write_runtime_config(config: ByteTrackConfig) -> str:
    """Write custom thresholds to a short-lived YAML file for provenance."""

    import tempfile

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".yaml",
        prefix="kavach_bytetrack_",
        delete=False,
    )
    try:
        handle.write(config.as_yaml())
    finally:
        handle.close()
    return handle.name


class MultiObjectTracker:
    """Associate Module 2 detections with persistent ByteTrack IDs.

    The detector is separate from the tracker. ``update`` accepts a frame
    from ``VideoReader``, obtains one frame-local detection list, advances one
    ByteTrack step, and returns current tracked objects. ByteTrack internally
    retains a bounded lost-track buffer; this class intentionally does not
    expose or build a KAVACH ObjectMemory history.
    """

    DEFAULT_CONFIG_PATH = Path(__file__).with_name("bytetrack.yaml")

    def __init__(
        self,
        detector: Any | None = None,
        *,
        model_path: str | Path = "yolo11n.pt",
        detection_confidence: float | None = None,
        device: str | None = None,
        allowed_classes: Iterable[str] | None = None,
        config: ByteTrackConfig | None = None,
    ) -> None:
        self.config = config or ByteTrackConfig()
        self.config_path = self.DEFAULT_CONFIG_PATH
        self._runtime_config_path: str | None = None
        if config is not None and config != ByteTrackConfig():
            self._runtime_config_path = _write_runtime_config(config)
            self.config_path = Path(self._runtime_config_path)

        if detector is None:
            model_confidence = (
                self.config.track_low_thresh
                if detection_confidence is None
                else detection_confidence
            )
            detector = WarehouseDetector(
                model_path=model_path,
                confidence=model_confidence,
                device=device,
                allowed_classes=allowed_classes,
            )
        elif detection_confidence is not None and not 0.0 <= detection_confidence <= 1.0:
            raise ValueError("detection_confidence must be between 0.0 and 1.0")

        self.detector = detector
        self.device = getattr(detector, "device", device or "unknown")
        self._class_names = dict(getattr(detector, "class_names", {}))
        self._tracker = BYTETracker(args=self.config.as_namespace())
        self.frames_processed = 0
        self._last_timestamp: float | None = None
        self._closed = False

    @property
    def tracker_name(self) -> str:
        """Return the explicitly configured backend name."""

        return self.config.tracker_type

    @property
    def track_buffer(self) -> int:
        """Maximum number of missing frames ByteTrack retains internally."""

        return self.config.track_buffer

    def _validate_frame_and_timestamp(
        self, frame: np.ndarray, timestamp: float
    ) -> None:
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise InvalidFrameError("frame must be a non-empty NumPy array")
        if not math.isfinite(float(timestamp)) or float(timestamp) < 0.0:
            raise InvalidTimestampError(
                f"timestamp must be a finite non-negative number, got {timestamp!r}"
            )

    def update(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_number: int | None = None,
    ) -> list[TrackedObject]:
        """Detect and track one frame from a VideoReader packet."""

        self._validate_frame_and_timestamp(frame, timestamp)
        detections = self.detector.detect(frame)
        return self.update_detections(
            detections,
            timestamp,
            frame=frame,
            frame_number=frame_number,
        )

    def update_detections(
        self,
        detections: Sequence[Detection],
        timestamp: float,
        *,
        frame: np.ndarray | None = None,
        frame_number: int | None = None,
    ) -> list[TrackedObject]:
        """Advance ByteTrack using already computed Module 2 detections."""

        if not math.isfinite(float(timestamp)) or float(timestamp) < 0.0:
            raise InvalidTimestampError(
                f"timestamp must be a finite non-negative number, got {timestamp!r}"
            )
        if frame is not None and (not isinstance(frame, np.ndarray) or frame.size == 0):
            raise InvalidFrameError("frame must be a non-empty NumPy array")

        tracks = self._tracker.update(_DetectionBatch(tuple(detections)), img=frame)
        current_frame_number = (
            self.frames_processed if frame_number is None else int(frame_number)
        )
        self.frames_processed += 1
        self._last_timestamp = float(timestamp)
        return self._standardize_tracks(
            tracks,
            float(timestamp),
            current_frame_number,
        )

    def _standardize_tracks(
        self,
        tracks: np.ndarray,
        timestamp: float,
        frame_number: int,
    ) -> list[TrackedObject]:
        if tracks is None or len(tracks) == 0:
            return []

        standardized: list[TrackedObject] = []
        for track in np.asarray(tracks).reshape(-1, tracks.shape[-1]):
            x1, y1, x2, y2 = (float(value) for value in track[:4])
            track_id = int(track[4])
            confidence = float(track[5])
            class_id = int(track[6])
            class_name = self._class_names.get(class_id, str(class_id))
            standardized.append(
                TrackedObject(
                    track_id=track_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    timestamp=timestamp,
                    frame_number=frame_number,
                )
            )
        return standardized

    def reset(self) -> None:
        """Start a new tracking session and reset the ByteTrack ID counter."""

        self._tracker.reset()
        self.frames_processed = 0
        self._last_timestamp = None

    def close(self) -> None:
        """Release any temporary custom configuration file."""

        if self._closed:
            return
        self._closed = True
        if self._runtime_config_path:
            try:
                os.unlink(self._runtime_config_path)
            except FileNotFoundError:
                pass
            self._runtime_config_path = None

    def __enter__(self) -> MultiObjectTracker:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
