"""Unit tests for the KAVACH Module 2 perception boundary."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from kavach.perception import (
    Detection,
    InvalidFrameError,
    WarehouseDetector,
    draw_detections,
)


class _TensorLike:
    def __init__(self, value: object) -> None:
        self.value = np.asarray(value)

    def cpu(self) -> "_TensorLike":
        return self

    def numpy(self) -> np.ndarray:
        return self.value


class _FakeBoxes:
    xyxy = _TensorLike([[10, 20, 110, 220], [5, 5, 30, 40], [1, 2, 3, 4]])
    conf = _TensorLike([0.91, 0.83, 0.99])
    cls = _TensorLike([0, 1, 2])


class _FakeResult:
    boxes = _FakeBoxes()


class _FakeModel:
    names = {0: "person", 1: "truck", 2: "cat"}

    def __init__(self) -> None:
        self.predict_calls: list[dict[str, object]] = []

    def predict(self, **kwargs: object) -> list[_FakeResult]:
        self.predict_calls.append(kwargs)
        return [_FakeResult()]


class DetectionTests(unittest.TestCase):
    def test_detection_is_standardized_and_irrelevant_classes_are_filtered(self) -> None:
        model = _FakeModel()
        detector = WarehouseDetector(
            model_path="fake.pt",
            model=model,
            confidence=0.4,
            device="cpu",
        )

        detections = detector.detect(np.zeros((240, 320, 3), dtype=np.uint8))

        self.assertEqual([d.class_name for d in detections], ["person", "truck"])
        self.assertEqual(detections[0].class_id, 0)
        self.assertEqual(detections[0].bbox, (10.0, 20.0, 110.0, 220.0))
        self.assertEqual(detections[0].center, (60.0, 120.0))
        self.assertAlmostEqual(detections[1].confidence, 0.83)
        self.assertEqual(model.predict_calls[0]["conf"], 0.4)
        self.assertEqual(model.predict_calls[0]["device"], "cpu")
        self.assertEqual(model.predict_calls[0]["classes"], [0, 1])

    def test_model_is_loaded_once_and_reused_for_multiple_frames(self) -> None:
        model = _FakeModel()
        with patch("kavach.perception.detector.YOLO", return_value=model) as loader:
            detector = WarehouseDetector("model.pt", device="cpu")
            detector.detect(np.zeros((32, 32, 3), dtype=np.uint8))
            detector.detect(np.zeros((32, 32, 3), dtype=np.uint8))

        loader.assert_called_once_with("model.pt")
        self.assertEqual(len(model.predict_calls), 2)

    def test_explicit_empty_allowed_classes_disables_filter(self) -> None:
        detector = WarehouseDetector(
            model_path="fake.pt",
            model=_FakeModel(),
            allowed_classes=(),
            device="cpu",
        )

        detections = detector.detect(np.zeros((32, 32, 3), dtype=np.uint8))

        self.assertEqual([d.class_name for d in detections], ["person", "truck", "cat"])

    def test_invalid_frame_is_rejected(self) -> None:
        detector = WarehouseDetector(model_path="fake.pt", model=_FakeModel(), device="cpu")

        with self.assertRaises(InvalidFrameError):
            detector.detect(np.empty((0, 0, 3), dtype=np.uint8))
        with self.assertRaises(InvalidFrameError):
            detector.detect(np.zeros((2, 2, 1, 1), dtype=np.uint8))

    def test_benchmark_returns_timing_summary(self) -> None:
        detector = WarehouseDetector(model_path="fake.pt", model=_FakeModel(), device="cpu")
        frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(4)]

        result = detector.benchmark(frames, max_frames=3, warmup_frames=1)

        self.assertEqual(result.frames, 3)
        self.assertGreater(result.average_inference_time_ms, 0.0)
        self.assertGreater(result.approximate_fps, 0.0)
        self.assertEqual(result.device, "cpu")
        self.assertEqual(result.model, "fake.pt")

    def test_drawing_returns_annotated_copy(self) -> None:
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        detection = Detection(
            class_name="person",
            class_id=0,
            confidence=0.95,
            bbox=(10.0, 20.0, 60.0, 70.0),
            center=(35.0, 45.0),
        )

        annotated = draw_detections(frame, [detection])

        self.assertEqual(annotated.shape, frame.shape)
        self.assertTrue(np.array_equal(frame, np.zeros_like(frame)))
        self.assertGreater(int(np.count_nonzero(annotated)), 0)
        self.assertFalse(np.array_equal(annotated, frame))


if __name__ == "__main__":
    unittest.main()
