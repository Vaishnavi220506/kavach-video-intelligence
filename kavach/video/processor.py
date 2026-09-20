"""Small frame-processing helpers for the Module 1 video foundation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .reader import FramePacket, VideoReader
from .source import VideoSource
from .writer import VideoWriter, VideoWriterError

FrameProcessor = Callable[[FramePacket], np.ndarray]


@dataclass(frozen=True)
class VideoProcessingResult:
    """Summary of a completed reader-to-writer pass."""

    input_path: Path
    output_path: Path
    frames_written: int
    fps: float
    duration: float


def annotate_frame_metadata(packet: FramePacket) -> np.ndarray:
    """Return a copy of a frame annotated with its number and timestamp."""

    annotated = packet.frame.copy()
    label = f"Frame: {packet.frame_number} | Time: {packet.timestamp:.2f}s"
    cv2.putText(
        annotated,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return annotated


def process_video(
    source: VideoSource,
    output: str | Path,
    frame_processor: FrameProcessor = annotate_frame_metadata,
    codec: str = "avc1",
) -> VideoProcessingResult:
    """Read, transform, and write a video in one controlled pass.

    ``frame_processor`` receives each :class:`FramePacket` and must return a
    BGR NumPy frame with the same resolution as the input. The default
    processor demonstrates the Module 1 contract by drawing frame number and
    timestamp without involving detection or tracking.
    """

    with VideoReader(source) as reader:
        output_path = Path(output).expanduser().resolve()
        if output_path == reader.source:
            raise VideoWriterError(
                "Output video must use a different path from the input video."
            )
        with VideoWriter(
            output=output_path,
            fps=reader.fps,
            width=reader.width,
            height=reader.height,
            codec=codec,
        ) as writer:
            for packet in reader:
                writer.write(frame_processor(packet))

            frames_written = writer.frames_written

        return VideoProcessingResult(
            input_path=reader.source,
            output_path=output_path,
            frames_written=frames_written,
            fps=reader.fps,
            duration=frames_written / reader.fps,
        )
