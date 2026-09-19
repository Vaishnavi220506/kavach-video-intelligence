"""Unit tests for the KAVACH Module 3 ByteTrack boundary."""

from __future__ import annotations

import unittest

import numpy as np

from kavach.perception import (
    ByteTrackConfig,
    Detection,
    InvalidTimestampError,
    MultiObjectTracker,
    TrackedObject,
    draw_tracked_objects,
)


def _detection(
    x1: float,
    y1: float = 20.0,
    x2: float | None = None,
    *,
    confidence: float = 0.9,
    class_id: int = 0,
    class_name: str = "person",
) -> Detection:
    x2 = x1 + 20.0 if x2 is None else x2
    y2 = y1 + 40.0
    return Detection(
        class_name=class_name,
        class_id=class_id,
        confidence=confidence,
        bbox=(x1, y1, x2, y2),
        center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
    )


class _SequenceDetector:
    class_names = {0: "person", 1: "truck"}
    device = "cpu"

    def __init__(self, sequence: list[list[Detection]]) -> None:
        self.sequence = list(sequence)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        del frame
        if not self.sequence:
            return []
        return self.sequence.pop(0)


def _tracker(
    sequence: list[list[Detection]],
    *,
    config: ByteTrackConfig | None = None,
) -> MultiObjectTracker:
    return MultiObjectTracker(
        detector=_SequenceDetector(sequence),
        config=config,
    )


class TrackingTests(unittest.TestCase):
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    def test_single_object_keeps_id_across_frames(self) -> None:
        tracker = _tracker(
            [[_detection(10)], [_detection(12)], [_detection(14)]],
        )

        first = tracker.update(self.frame, 0.0)
        second = tracker.update(self.frame, 0.1)
        third = tracker.update(self.frame, 0.2)

        self.assertEqual(len(first), len(second), len(third))
        self.assertEqual(first[0].track_id, second[0].track_id)
        self.assertEqual(second[0].track_id, third[0].track_id)
        self.assertEqual(third[0].timestamp, 0.2)

    def test_multiple_objects_keep_two_distinct_ids(self) -> None:
        tracker = _tracker(
            [
                [_detection(10), _detection(100)],
                [_detection(12), _detection(98)],
            ]
        )

        first = tracker.update(self.frame, 0.0)
        second = tracker.update(self.frame, 0.1)

        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 2)
        self.assertEqual(len({obj.track_id for obj in first}), 2)
        self.assertEqual({obj.track_id for obj in first}, {obj.track_id for obj in second})

    def test_crossing_objects_are_processed_without_extra_active_tracks(self) -> None:
        tracker = _tracker(
            [
                [_detection(10, x2=50), _detection(70, x2=110)],
                [_detection(25, x2=65), _detection(55, x2=95)],
                [_detection(40, x2=80), _detection(40, x2=80)],
                [_detection(55, x2=95), _detection(25, x2=65)],
            ]
        )

        outputs = [tracker.update(self.frame, index / 10.0) for index in range(4)]

        self.assertTrue(all(len(objects) == 2 for objects in outputs))
        self.assertLessEqual(
            len({obj.track_id for objects in outputs for obj in objects}), 2
        )

    def test_short_occlusion_reacquires_original_id(self) -> None:
        tracker = _tracker(
            [[_detection(10)], [], [_detection(12)]],
            config=ByteTrackConfig(track_buffer=2),
        )

        visible_before = tracker.update(self.frame, 0.0)
        missing = tracker.update(self.frame, 0.1)
        visible_after = tracker.update(self.frame, 0.2)

        self.assertEqual(len(visible_before), 1)
        self.assertEqual(missing, [])
        self.assertEqual(len(visible_after), 1)
        self.assertEqual(visible_before[0].track_id, visible_after[0].track_id)

    def test_low_confidence_detection_can_recover_existing_track(self) -> None:
        tracker = _tracker(
            [
                [_detection(10, confidence=0.9)],
                [_detection(12, confidence=0.15)],
            ],
            config=ByteTrackConfig(
                track_high_thresh=0.25,
                track_low_thresh=0.10,
                new_track_thresh=0.25,
            ),
        )

        first = tracker.update(self.frame, 0.0)
        second = tracker.update(self.frame, 0.1)

        self.assertEqual(first[0].track_id, second[0].track_id)
        self.assertAlmostEqual(second[0].confidence, 0.15)

    def test_invalid_timestamp_is_rejected(self) -> None:
        tracker = _tracker([[_detection(10)]])

        with self.assertRaises(InvalidTimestampError):
            tracker.update(self.frame, -0.1)

    def test_tracked_visualization_contains_id_label(self) -> None:
        tracked = TrackedObject(
            track_id=7,
            class_name="person",
            confidence=0.9,
            bbox=(10.0, 20.0, 40.0, 60.0),
            center=(25.0, 40.0),
            timestamp=1.0,
        )

        annotated = draw_tracked_objects(self.frame, [tracked])

        self.assertEqual(annotated.shape, self.frame.shape)
        self.assertGreater(int(np.count_nonzero(annotated)), 0)
        self.assertTrue(np.array_equal(self.frame, np.zeros_like(self.frame)))


if __name__ == "__main__":
    unittest.main()
