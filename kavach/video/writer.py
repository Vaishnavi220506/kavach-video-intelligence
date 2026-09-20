"""Reusable OpenCV video writer with frame-shape validation."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .source import VideoError


class VideoWriterError(VideoError):
    """Raised when an output video cannot be created or written."""


class VideoFrameError(VideoWriterError):
    """Raised when a frame does not match the configured output video."""


class VideoWriter:
    """Write BGR frames to a video file using OpenCV's configured codec."""

    def __init__(
        self,
        output: str | Path,
        fps: float,
        width: int,
        height: int,
        codec: str = "avc1",
    ):
        self.output = Path(output).expanduser()
        self._validate_metadata(fps, width, height, codec)
        self.fps = float(fps)
        self.width = int(width)
        self.height = int(height)
        self.codec = codec
        self.output.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(
            str(self.output),
            fourcc,
            self.fps,
            (self.width, self.height),
        )
        # H.264 is friendlier to browser video elements than MPEG-4 Part 2.
        # Some OpenCV builds do not ship an H.264 encoder, so retain a safe
        # local fallback instead of failing a complete analysis run solely
        # because of codec availability.
        if not self._writer.isOpened() and codec == "avc1":
            self._writer.release()
            codec = "mp4v"
            self.codec = codec
            self._writer = cv2.VideoWriter(
                str(self.output),
                cv2.VideoWriter_fourcc(*codec),
                self.fps,
                (self.width, self.height),
            )
        self._closed = False
        self.frames_written = 0

        if not self._writer.isOpened():
            self._writer.release()
            raise VideoWriterError(
                f"OpenCV could not create output video '{self.output}' "
                f"with codec '{codec}'."
            )

    @staticmethod
    def _validate_metadata(
        fps: float, width: int, height: int, codec: str
    ) -> None:
        if not math.isfinite(float(fps)) or float(fps) <= 0:
            raise VideoWriterError(f"Writer FPS must be positive, got {fps!r}.")
        if int(width) <= 0 or int(height) <= 0:
            raise VideoWriterError(
                f"Writer resolution must be positive, got {width}x{height}."
            )
        if not isinstance(codec, str) or len(codec) != 4:
            raise VideoWriterError(
                f"Codec must be a four-character code, got {codec!r}."
            )

    @property
    def is_opened(self) -> bool:
        """Whether the underlying OpenCV writer is still available."""

        return not self._closed and self._writer.isOpened()

    def write(self, frame: np.ndarray) -> None:
        """Write one BGR color frame after checking its dimensions."""

        if self._closed:
            raise VideoWriterError("Cannot write to a closed VideoWriter.")
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise VideoFrameError("Frame must be a non-empty NumPy array.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise VideoFrameError(
                f"Frame must have shape (height, width, 3), got {frame.shape}."
            )
        expected_shape = (self.height, self.width)
        if frame.shape[:2] != expected_shape:
            raise VideoFrameError(
                f"Frame resolution {frame.shape[1]}x{frame.shape[0]} does not "
                f"match writer resolution {self.width}x{self.height}."
            )

        self._writer.write(frame)
        self.frames_written += 1

    def close(self) -> None:
        """Release the OpenCV writer."""

        if not self._closed:
            self._writer.release()
            self._closed = True

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
