"""Reusable OpenCV video reader with metadata and frame packets."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .source import (
    MissingVideoMetadataError,
    UnreadableVideoError,
    UnsupportedVideoError,
    VideoError,
    VideoSource,
    ZeroFPSVideoError,
    resolve_video_source,
)


@dataclass(frozen=True)
class FramePacket:
    """One decoded frame and the timing information derived from its index."""

    frame: np.ndarray
    frame_number: int
    timestamp: float


class VideoReader:
    """Read a local video one frame at a time.

    Frame numbers are zero-based, matching OpenCV's frame indexing convention.
    For a constant-FPS file, the timestamp of frame ``n`` is ``n / fps``
    seconds from the beginning of the video.
    """

    def __init__(self, source: VideoSource):
        self.source: Path = resolve_video_source(source)
        self._capture = self._open_capture(self.source)
        self._closed = False
        self._frames_read = 0
        self._eof = False

        try:
            self._fps = self._read_fps()
            self._width = self._read_positive_int(
                cv2.CAP_PROP_FRAME_WIDTH, "width"
            )
            self._height = self._read_positive_int(
                cv2.CAP_PROP_FRAME_HEIGHT, "height"
            )
            self._total_frames = self._read_positive_int(
                cv2.CAP_PROP_FRAME_COUNT, "total frame count"
            )
        except VideoError:
            self.close()
            raise

        self._duration = self._total_frames / self._fps

    @staticmethod
    def _open_capture(source: Path) -> cv2.VideoCapture:
        try:
            capture = cv2.VideoCapture(str(source))
        except cv2.error as exc:
            raise UnsupportedVideoError(
                f"OpenCV could not open video '{source}'."
            ) from exc

        if not capture.isOpened():
            capture.release()
            raise UnsupportedVideoError(
                f"OpenCV could not open video '{source}'. "
                "The file may be unsupported or its codec may be unavailable."
            )
        return capture

    def _read_fps(self) -> float:
        fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ZeroFPSVideoError(
                f"Video '{self.source}' reports an invalid FPS value: {fps!r}."
            )
        return fps

    def _read_positive_int(self, property_id: int, label: str) -> int:
        value = float(self._capture.get(property_id))
        if not math.isfinite(value) or value <= 0:
            raise MissingVideoMetadataError(
                f"Video '{self.source}' is missing valid {label} metadata "
                f"(reported value: {value!r})."
            )
        result = int(round(value))
        if result <= 0:
            raise MissingVideoMetadataError(
                f"Video '{self.source}' is missing valid {label} metadata "
                f"(reported value: {value!r})."
            )
        return result

    @property
    def fps(self) -> float:
        """Frames per second reported by the video container."""

        return self._fps

    @property
    def width(self) -> int:
        """Frame width in pixels."""

        return self._width

    @property
    def height(self) -> int:
        """Frame height in pixels."""

        return self._height

    @property
    def total_frames(self) -> int:
        """Total frame count reported by the video container."""

        return self._total_frames

    @property
    def duration(self) -> float:
        """Approximate duration in seconds, calculated as ``total_frames / fps``."""

        return self._duration

    @property
    def is_opened(self) -> bool:
        """Whether the underlying OpenCV capture is still available."""

        return not self._closed and self._capture.isOpened()

    def __iter__(self) -> Iterator[FramePacket]:
        """Yield decoded frames until the stream reaches its end."""

        if self._closed:
            raise VideoError("Cannot iterate a closed VideoReader.")
        if self._eof:
            return

        while True:
            try:
                success, frame = self._capture.read()
            except cv2.error as exc:
                self.close()
                raise UnreadableVideoError(
                    f"OpenCV failed while reading video '{self.source}' "
                    f"at frame {self._frames_read}."
                ) from exc
            if not success:
                self._eof = True
                if self._frames_read == 0:
                    self.close()
                    raise UnreadableVideoError(
                        f"Video '{self.source}' opened but no readable frames "
                        "were returned."
                    )
                break

            if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                self.close()
                raise UnreadableVideoError(
                    f"Video '{self.source}' returned an empty frame at frame "
                    f"{self._frames_read}."
                )

            frame_number = self._frames_read
            timestamp = frame_number / self._fps
            self._frames_read += 1
            yield FramePacket(
                frame=frame,
                frame_number=frame_number,
                timestamp=timestamp,
            )

    def close(self) -> None:
        """Release the OpenCV capture."""

        if not self._closed:
            self._capture.release()
            self._closed = True

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
