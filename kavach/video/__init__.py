"""Video ingestion and output primitives for KAVACH."""

from .processor import (
    FrameProcessor,
    VideoProcessingResult,
    annotate_frame_metadata,
    process_video,
)
from .reader import FramePacket, VideoReader
from .source import (
    InvalidVideoSourceError,
    MissingVideoMetadataError,
    UnreadableVideoError,
    UnsupportedVideoError,
    VideoError,
    VideoMetadataError,
    VideoSource,
    ZeroFPSVideoError,
)
from .writer import VideoFrameError, VideoWriter, VideoWriterError

__all__ = [
    "FramePacket",
    "FrameProcessor",
    "InvalidVideoSourceError",
    "MissingVideoMetadataError",
    "UnreadableVideoError",
    "UnsupportedVideoError",
    "VideoError",
    "VideoFrameError",
    "VideoMetadataError",
    "VideoProcessingResult",
    "VideoReader",
    "VideoSource",
    "VideoWriter",
    "VideoWriterError",
    "ZeroFPSVideoError",
    "annotate_frame_metadata",
    "process_video",
]

