"""Dashboard-facing source and end-to-end orchestration helpers."""

from .analysis import (
    DashboardAnalysisResult,
    analyse_video,
    build_default_zones,
    draw_dashboard_frame,
)
from .overlays import EventExplanation, draw_explainable_overlays, explain_event
from .source import (
    MAX_VIDEO_BYTES,
    VideoSourceError,
    download_video_url,
    persist_uploaded_video,
    validate_direct_video_url,
    video_file_sha256,
)

__all__ = [
    "MAX_VIDEO_BYTES",
    "DashboardAnalysisResult",
    "EventExplanation",
    "VideoSourceError",
    "analyse_video",
    "build_default_zones",
    "download_video_url",
    "draw_dashboard_frame",
    "draw_explainable_overlays",
    "explain_event",
    "persist_uploaded_video",
    "validate_direct_video_url",
    "video_file_sha256",
]
