from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from kavach.dashboard import analyse_video
from kavach.perception import WarehouseDetector
from kavach.storage import EventDatabase
from kavach.video import VideoReader, VideoWriter


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.asarray(xyxy, dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class _SequenceModel:
    names = {0: "person", 1: "carton"}

    def __init__(self, frames):
        self.frames = list(frames)
        self.index = 0

    def predict(self, **kwargs):
        detections = self.frames[min(self.index, len(self.frames) - 1)]
        self.index += 1
        if not detections:
            boxes = _Boxes(np.empty((0, 4)), np.empty(0), np.empty(0))
        else:
            boxes = _Boxes(
                [item["bbox"] for item in detections],
                [item["confidence"] for item in detections],
                [item["class_id"] for item in detections],
            )
        return [_Result(boxes)]


def _write_video(path: Path, *, frames: int) -> None:
    image = np.zeros((96, 128, 3), dtype=np.uint8)
    with VideoWriter(path, fps=8.0, width=128, height=96, codec="mp4v") as writer:
        for index in range(frames):
            frame = image.copy()
            cv2.putText(frame, str(index), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            writer.write(frame)


class EndToEndPipelineTests(unittest.TestCase):
    def test_two_videos_traverse_reader_to_dashboard_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_one = root / "one.mp4"
            source_two = root / "two.mp4"
            _write_video(source_one, frames=4)
            _write_video(source_two, frames=3)
            database = EventDatabase(root / "events.sqlite3")
            try:
                detector_one = WarehouseDetector(
                    model=_SequenceModel([
                        [{"bbox": [10, 10, 30, 60], "confidence": 0.9, "class_id": 0}],
                        [{"bbox": [12, 10, 32, 60], "confidence": 0.9, "class_id": 0}],
                        [{"bbox": [14, 10, 34, 60], "confidence": 0.9, "class_id": 0}],
                        [{"bbox": [16, 10, 36, 60], "confidence": 0.9, "class_id": 0}],
                    ]),
                    allowed_classes=(),
                    confidence=0.25,
                    device="cpu",
                )
                detector_two = WarehouseDetector(
                    model=_SequenceModel([[], [], []]),
                    allowed_classes=(),
                    confidence=0.25,
                    device="cpu",
                )
                first = analyse_video(
                    source_one,
                    video_id="video-one",
                    output=root / "one-processed.mp4",
                    detector=detector_one,
                    database=database,
                )
                second = analyse_video(
                    source_two,
                    video_id="video-two",
                    output=root / "two-processed.mp4",
                    detector=detector_two,
                    database=database,
                )
                self.assertEqual(first.frames_processed, 4)
                self.assertEqual(second.frames_processed, 3)
                self.assertTrue(Path(first.output_path).is_file())
                self.assertTrue(Path(second.output_path).is_file())
                self.assertEqual(
                    {item["video_id"] for item in database.list_videos()},
                    {"video-one", "video-two"},
                )
                with VideoReader(first.output_path) as reader:
                    self.assertEqual(reader.total_frames, 4)
                    self.assertEqual(reader.width, 128)
                    self.assertEqual(reader.height, 96)
                self.assertIsInstance(first.events, tuple)
                self.assertIsInstance(second.events, tuple)
            finally:
                database.close()


if __name__ == "__main__":
    unittest.main()
