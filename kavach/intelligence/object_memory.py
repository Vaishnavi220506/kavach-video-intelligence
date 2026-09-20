"""Bounded timestamped object histories for KAVACH Module 4."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..perception.tracker import TrackedObject
from .motion import MotionError, MotionEstimate, calculate_motion


class ObjectMemoryError(RuntimeError):
    """Base class for expected object-memory errors."""


class NonMonotonicTimestampError(ObjectMemoryError):
    """Raised when updates move backward in source-video time."""


class InvalidMemoryConfigurationError(ObjectMemoryError):
    """Raised when a history bound or motion setting is invalid."""


@dataclass(frozen=True)
class ObjectState:
    """One timestamped observation stored for one tracked object."""

    track_id: int
    timestamp: float
    frame_number: int
    center: tuple[float, float]
    bbox: tuple[float, float, float, float]
    class_name: str
    confidence: float


def _finite_values(values: Sequence[float], label: str) -> tuple[float, ...]:
    converted = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in converted):
        raise ObjectMemoryError(f"{label} must contain only finite values")
    return converted


class ObjectMemory:
    """Keep bounded, timestamp-aware histories indexed by track ID.

    The memory stores only observations that arrive through ``update``. It
    does not invent states while an object is missing, and it does not convert
    pixels to metres. Motion values are explicitly reported in pixels per
    second until camera calibration exists.
    """

    def __init__(
        self,
        *,
        max_history_seconds: float = 30.0,
        max_states_per_track: int = 600,
        max_tracks: int = 1000,
        stationary_speed_threshold: float = 2.0,
    ) -> None:
        if (
            not math.isfinite(max_history_seconds)
            or max_history_seconds <= 0.0
            or max_states_per_track <= 0
            or max_tracks <= 0
            or not math.isfinite(stationary_speed_threshold)
            or stationary_speed_threshold < 0.0
        ):
            raise InvalidMemoryConfigurationError(
                "history bounds must be positive and stationary threshold "
                "must be finite and non-negative"
            )

        self.max_history_seconds = float(max_history_seconds)
        self.max_states_per_track = int(max_states_per_track)
        self.max_tracks = int(max_tracks)
        self.stationary_speed_threshold = float(stationary_speed_threshold)
        self._history: dict[int, deque[ObjectState]] = {}
        self._first_seen: dict[int, float] = {}
        self._last_seen: dict[int, float] = {}
        self._last_update_timestamp: float | None = None
        self._implicit_frame_number = 0

    @property
    def history(self) -> dict[int, list[ObjectState]]:
        """Return a defensive copy of the bounded history mapping."""

        return {track_id: list(states) for track_id, states in self._history.items()}

    @property
    def track_ids(self) -> tuple[int, ...]:
        """Return track IDs currently retained by memory."""

        return tuple(self._history.keys())

    def _validate_update_timestamp(self, timestamp: float) -> float:
        value = float(timestamp)
        if not math.isfinite(value) or value < 0.0:
            raise ObjectMemoryError(
                f"timestamp must be finite and non-negative, got {timestamp!r}"
            )
        if (
            self._last_update_timestamp is not None
            and value < self._last_update_timestamp
        ):
            raise NonMonotonicTimestampError(
                f"timestamp {value} is earlier than the previous update "
                f"{self._last_update_timestamp}"
            )
        return value

    def _evict_track_if_needed(self, track_id: int) -> None:
        if track_id in self._history or len(self._history) < self.max_tracks:
            return
        oldest_id = min(self._last_seen, key=self._last_seen.get)
        self._history.pop(oldest_id, None)
        self._first_seen.pop(oldest_id, None)
        self._last_seen.pop(oldest_id, None)

    def _make_state(
        self,
        tracked: TrackedObject,
        timestamp: float,
        frame_number: int,
    ) -> ObjectState:
        center = _finite_values(tracked.center, "center")
        bbox = _finite_values(tracked.bbox, "bbox")
        confidence = float(tracked.confidence)
        if len(center) != 2 or len(bbox) != 4:
            raise ObjectMemoryError("center must have 2 values and bbox must have 4")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ObjectMemoryError("confidence must be finite and between 0 and 1")
        return ObjectState(
            track_id=int(tracked.track_id),
            timestamp=timestamp,
            frame_number=int(frame_number),
            center=(center[0], center[1]),
            bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
            class_name=str(tracked.class_name),
            confidence=confidence,
        )

    def _append_state(self, state: ObjectState) -> None:
        track_id = state.track_id
        self._evict_track_if_needed(track_id)
        if track_id not in self._history:
            self._history[track_id] = deque(maxlen=self.max_states_per_track)
            self._first_seen[track_id] = state.timestamp
        self._history[track_id].append(state)
        self._last_seen[track_id] = state.timestamp
        cutoff = state.timestamp - self.max_history_seconds
        while self._history[track_id] and self._history[track_id][0].timestamp < cutoff:
            self._history[track_id].popleft()

    def update(
        self,
        tracked_objects: Iterable[TrackedObject],
        timestamp: float,
        frame_number: int | None = None,
    ) -> None:
        """Append the current active tracks at one source-video timestamp.

        ``frame_number`` is optional for callers without source metadata. When
        it is omitted, the memory uses an update counter. A
        ``TrackedObject.frame_number`` value always takes precedence, which
        lets a ``VideoReader`` packet preserve its actual source index.
        """

        timestamp_value = self._validate_update_timestamp(timestamp)
        if frame_number is not None and int(frame_number) < 0:
            raise ObjectMemoryError("frame_number must be non-negative")
        fallback_frame = (
            self._implicit_frame_number
            if frame_number is None
            else int(frame_number)
        )
        objects = tuple(tracked_objects)
        for tracked in objects:
            object_frame = getattr(tracked, "frame_number", None)
            if object_frame is None:
                object_frame = fallback_frame
            if int(object_frame) < 0:
                raise ObjectMemoryError("frame_number must be non-negative")
            self._append_state(
                self._make_state(tracked, timestamp_value, int(object_frame))
            )

        self._last_update_timestamp = timestamp_value
        self._implicit_frame_number = max(
            self._implicit_frame_number + 1,
            fallback_frame + 1,
        )

    def get(self, track_id: int) -> ObjectState | None:
        """Return the latest state for a track, or ``None`` if unknown."""

        states = self._history.get(int(track_id))
        return states[-1] if states else None

    def get_history(self, track_id: int) -> list[ObjectState]:
        """Return a defensive list of all retained states for a track."""

        return list(self._history.get(int(track_id), ()))

    def get_trajectory(
        self,
        track_id: int,
        seconds: float = 2.0,
    ) -> list[tuple[float, float]]:
        """Return recent center points within ``seconds`` of the latest state."""

        seconds_value = float(seconds)
        if not math.isfinite(seconds_value) or seconds_value < 0.0:
            raise ObjectMemoryError("seconds must be finite and non-negative")
        states = self._history.get(int(track_id))
        if not states:
            return []
        cutoff = states[-1].timestamp - seconds_value
        return [state.center for state in states if state.timestamp >= cutoff]

    def get_velocity(self, track_id: int) -> MotionEstimate | None:
        """Return the latest timestamp-aware image motion estimate."""

        states = self._history.get(int(track_id))
        if not states or len(states) < 2:
            return None
        previous_previous: ObjectState | None = (
            states[-3] if len(states) >= 3 else None
        )
        try:
            return calculate_motion(
                states[-2],
                states[-1],
                previous_previous=previous_previous,
                stationary_speed_threshold=self.stationary_speed_threshold,
            )
        except MotionError as exc:
            raise ObjectMemoryError(str(exc)) from exc

    def is_stationary(self, track_id: int) -> bool:
        """Return whether the latest measurable motion is below the threshold."""

        velocity = self.get_velocity(track_id)
        return velocity is not None and velocity.stationary

    def get_stationary_duration(self, track_id: int) -> float:
        """Return the duration of the latest uninterrupted stationary run."""

        states = self._history.get(int(track_id))
        if not states or len(states) < 2:
            return 0.0

        start = states[-1].timestamp
        for index in range(len(states) - 1, 0, -1):
            try:
                motion = calculate_motion(
                    states[index - 1],
                    states[index],
                    stationary_speed_threshold=self.stationary_speed_threshold,
                )
            except MotionError:
                break
            if not motion.stationary:
                break
            start = states[index - 1].timestamp
        return max(0.0, states[-1].timestamp - start)

    def get_track_age(self, track_id: int) -> float | None:
        """Return latest-seen minus first-seen time for a retained track."""

        latest = self.get(track_id)
        first = self._first_seen.get(int(track_id))
        if latest is None or first is None:
            return None
        return max(0.0, latest.timestamp - first)

    def get_last_seen(self, track_id: int) -> float | None:
        """Return the last source timestamp at which a track was observed."""

        return self._last_seen.get(int(track_id))

    def clear(self) -> None:
        """Remove all histories and reset the update clock."""

        self._history.clear()
        self._first_seen.clear()
        self._last_seen.clear()
        self._last_update_timestamp = None
        self._implicit_frame_number = 0

