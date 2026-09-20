import io
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from kavach.behaviours import MOTION_ANOMALY, BehaviourEvent
from kavach.dashboard import (
    VideoSourceError,
    build_default_zones,
    download_video_url,
    draw_dashboard_frame,
    explain_event,
    persist_uploaded_video,
    validate_direct_video_url,
    video_file_sha256,
)
from kavach.perception import TrackedObject
from kavach.video import FramePacket


class _FakeResponse(io.BytesIO):
    status = 200

    def __init__(self, content: bytes, content_type: str = "video/mp4") -> None:
        super().__init__(content)
        self.headers = {
            "Content-Length": str(len(content)),
            "Content-Type": content_type,
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class DashboardSourceTests(unittest.TestCase):
    def test_direct_url_validation_rejects_non_video_url_shapes(self) -> None:
        with self.assertRaises(VideoSourceError):
            validate_direct_video_url("file:///tmp/video.mp4")
        with self.assertRaises(VideoSourceError):
            validate_direct_video_url("https://user:password@example.com/video.mp4")
        with self.assertRaises(VideoSourceError):
            validate_direct_video_url("https://")

    @patch("kavach.dashboard.source.urlopen")
    def test_direct_url_download_is_bounded_and_temporary(self, urlopen) -> None:
        urlopen.return_value = _FakeResponse(b"video-bytes")
        with tempfile.TemporaryDirectory() as temp_dir:
            output = download_video_url(
                "https://example.com/warehouse.mp4",
                output_dir=temp_dir,
                max_bytes=100,
            )
            self.assertTrue(output.is_file())
            self.assertEqual(output.read_bytes(), b"video-bytes")
            self.assertEqual(video_file_sha256(output), video_file_sha256(output))

    @patch("kavach.dashboard.source.urlopen")
    def test_direct_url_rejects_html_response(self, urlopen) -> None:
        urlopen.return_value = _FakeResponse(b"<html>", content_type="text/html")
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(VideoSourceError):
                download_video_url("https://example.com/page", output_dir=temp_dir)

    def test_upload_is_safely_named_and_zones_are_configured(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = persist_uploaded_video(
                "../../warehouse clip.unknown",
                b"sample-video",
                output_dir=temp_dir,
            )
            self.assertTrue(output.is_file())
            self.assertEqual(output.suffix, ".mp4")
            self.assertNotIn("..", output.name)
        zones = build_default_zones(640, 480)
        self.assertEqual(
            {zone.name for zone in zones.zones},
            {"STAGING_ZONE", "LOADING_ZONE", "RESTRICTED_ZONE", "PALLET_ZONE"},
        )


class DashboardRenderingTests(unittest.TestCase):
    def test_zone_explanation_names_the_reason_and_boundary(self) -> None:
        event = BehaviourEvent(
            type="ZONE_VIOLATION",
            timestamp=0.5,
            entities=("person_3", "restricted_zone"),
            confidence=0.8,
            evidence={"zone_name": "RESTRICTED_ZONE", "boundary_distance_px": 12},
        )

        explanation = explain_event(event)

        self.assertIn("restricted area", explanation.reason)
        self.assertIn("bottom-center point", explanation.detail)
        self.assertIn("12 px", explanation.detail)

    def test_motion_anomaly_highlights_the_matching_tracked_object(self) -> None:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        packet = FramePacket(frame=frame, frame_number=15, timestamp=0.5)
        tracked = TrackedObject(
            track_id=7,
            class_name="person",
            confidence=0.9,
            bbox=(30.0, 60.0, 120.0, 180.0),
            center=(75.0, 120.0),
            timestamp=0.5,
            frame_number=15,
        )
        event = BehaviourEvent(
            type=MOTION_ANOMALY,
            timestamp=0.5,
            entities=("person_7",),
            confidence=0.8,
            evidence={"speed_px_per_second": 300.0},
        )

        annotated = draw_dashboard_frame(
            frame,
            packet,
            [tracked],
            build_default_zones(320, 240),
            [event],
        )

        self.assertTrue(np.any(annotated[60, 30:121, 2] > 200))
        self.assertTrue(np.any(annotated != frame))


if __name__ == "__main__":
    unittest.main()
