"""Dashboard-facing source and end-to-end orchestration helpers."""

from .analysis import (
    DashboardAnalysisResult,
    analyse_video,
    build_default_zones,
    draw_dashboard_frame,
)
from .source import (
    MAX_VIDEO_BYTES,
    VideoSourceError,
    download_video_url,
    persist_uploaded_video,
    validate_direct_video_url,
    video_file_sha256,
)
from .overlays import EventExplanation, draw_explainable_overlays, explain_event

__all__ = [
    "DashboardAnalysisResult",
    "MAX_VIDEO_BYTES",
    "VideoSourceError",
    "analyse_video",
    "build_default_zones",
    "download_video_url",
    "draw_dashboard_frame",
    "draw_explainable_overlays",
    "EventExplanation",
    "explain_event",
    "persist_uploaded_video",
    "validate_direct_video_url",
    "video_file_sha256",
]
