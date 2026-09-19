"""Video source types, path validation, and shared video exceptions."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import TypeAlias


VideoSource: TypeAlias = str | PathLike[str]


class VideoError(RuntimeError):
    """Base class for expected video-pipeline errors."""


class InvalidVideoSourceError(VideoError):
    """Raised when a video path is missing, empty, or is not a regular file."""


class UnsupportedVideoError(VideoError):
    """Raised when OpenCV cannot open the file/container/codec."""


class UnreadableVideoError(VideoError):
    """Raised when an opened stream cannot produce its first frame."""


class VideoMetadataError(VideoError):
    """Base class for missing or invalid video metadata."""


class ZeroFPSVideoError(VideoMetadataError):
    """Raised when the source reports no usable frames-per-second value."""


class MissingVideoMetadataError(VideoMetadataError):
    """Raised when required dimensions or frame-count metadata is unavailable."""


def resolve_video_source(source: VideoSource) -> Path:
    """Validate and normalize a filesystem video source.

    OpenCV supports many source types, but KAVACH Module 1 intentionally keeps
    the foundation focused on local video files. Streamlit uploads are first
    written to a temporary local file and then use this same path-based API.
    """

    if source is None:
        raise InvalidVideoSourceError("Video source cannot be None.")

    try:
        path = Path(source).expanduser()
    except (TypeError, ValueError) as exc:
        raise InvalidVideoSourceError(
            f"Video source must be a filesystem path, got {source!r}."
        ) from exc

    if not str(path).strip():
        raise InvalidVideoSourceError("Video source cannot be empty.")
    if not path.exists():
        raise InvalidVideoSourceError(f"Video file does not exist: {path}")
    if not path.is_file():
        raise InvalidVideoSourceError(f"Video source is not a file: {path}")

    return path.resolve()

