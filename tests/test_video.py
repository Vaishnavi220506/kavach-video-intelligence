"""Tests for the Module 1 video foundation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from kavach.video import (
    InvalidVideoSourceError,
    MissingVideoMetadataError,
    UnreadableVideoError,
    VideoReader,
    VideoWriter,
    ZeroFPSVideoError,
    process_video,
)


class FakeCapture:
    """Small OpenCV capture double for metadata/error tests."""

    def __init__(self, *, fps=10.0, width=64.0, height=48.0, frames=1.0, readable=True):
        self.values = {
            cv2.CAP_PROP_FPS: fps,
            cv2.CAP_PROP_FRAME_WIDTH: width,
            cv2.CAP_PROP_FRAME_HEIGHT: height,
            cv2.CAP_PROP_FRAME_COUNT: frames,
        }
        self.readable = readable
        self.released = False

    def isOpened(self):
        return True

    def get(self, property_id):
        return self.values.get(property_id, 0.0)

    def read(self):
        if self.readable:
            return True, np.zeros((48, 64, 3), dtype=np.uint8)
        return False, None

    def release(self):
        self.released = True


class VideoFoundationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.input_path = self.root / "input.avi"
        self.output_path = self.root / "processed.avi"
        self._create_input_video()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_input_video(self):
        writer = VideoWriter(
            self.input_path,
            fps=10.0,
            width=64,
            height=48,
            codec="MJPG",
        )
        try:
            for frame_number in range(5):
                frame = np.full(
                    (48, 64, 3),
                    (frame_number * 20, 40, 80),
                    dtype=np.uint8,
                )
                writer.write(frame)
        finally:
            writer.close()

    def test_reader_exposes_metadata_and_timestamps(self):
        with VideoReader(self.input_path) as reader:
            self.assertAlmostEqual(reader.fps, 10.0, places=2)
            self.assertEqual(reader.width, 64)
            self.assertEqual(reader.height, 48)
            self.assertEqual(reader.total_frames, 5)
            self.assertAlmostEqual(reader.duration, 0.5, places=2)

            packets = list(reader)

        self.assertEqual(len(packets), 5)
        self.assertEqual([packet.frame_number for packet in packets], list(range(5)))
        for packet in packets:
            self.assertAlmostEqual(
                packet.timestamp,
                packet.frame_number / reader.fps,
                places=6,
            )
            self.assertEqual(packet.frame.shape, (48, 64, 3))

    def test_process_video_writes_metadata_annotated_frames(self):
        result = process_video(
            self.input_path,
            self.output_path,
            codec="MJPG",
        )

        self.assertTrue(self.output_path.exists())
        self.assertEqual(result.frames_written, 5)
        self.assertAlmostEqual(result.duration, 0.5, places=2)

        with VideoReader(self.output_path) as reader:
            output_packets = list(reader)

        self.assertEqual(len(output_packets), 5)
        self.assertEqual(output_packets[0].frame.shape, (48, 64, 3))
        # The green annotation changes at least one pixel in the first frame.
        self.assertFalse(np.array_equal(output_packets[0].frame, np.full((48, 64, 3), (0, 40, 80), dtype=np.uint8)))

    def test_invalid_source_is_rejected(self):
        with self.assertRaises(InvalidVideoSourceError):
            VideoReader(self.root / "missing.mp4")

    def test_zero_fps_is_rejected(self):
        fake_capture = FakeCapture(fps=0.0)
        with patch("kavach.video.reader.cv2.VideoCapture", return_value=fake_capture):
            with self.assertRaises(ZeroFPSVideoError):
                VideoReader(self.input_path)
        self.assertTrue(fake_capture.released)

    def test_missing_metadata_is_rejected(self):
        fake_capture = FakeCapture(width=0.0)
        with patch("kavach.video.reader.cv2.VideoCapture", return_value=fake_capture):
            with self.assertRaises(MissingVideoMetadataError):
                VideoReader(self.input_path)
        self.assertTrue(fake_capture.released)

    def test_unreadable_stream_is_rejected_on_iteration(self):
        fake_capture = FakeCapture(readable=False)
        with patch("kavach.video.reader.cv2.VideoCapture", return_value=fake_capture):
            reader = VideoReader(self.input_path)
            with self.assertRaises(UnreadableVideoError):
                list(reader)
            reader.close()


if __name__ == "__main__":
    unittest.main()

