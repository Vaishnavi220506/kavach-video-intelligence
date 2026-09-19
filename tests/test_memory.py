"""Unit tests for KAVACH Module 4 object memory and motion."""

from __future__ import annotations

import unittest

import numpy as np

from kavach.intelligence import (
    NonMonotonicTimestampError,
    ObjectMemory,
    ObjectState,
    draw_memory_trajectory,
)
from kavach.perception import TrackedObject


def _tracked(
    track_id: int,
    center_x: float,
    center_y: float = 50.0,
    *,
    timestamp: float = 0.0,
    frame_number: int | None = None,
    confidence: float = 0.9,
) -> TrackedObject:
    bbox = (center_x - 5.0, center_y - 5.0, center_x + 5.0, center_y + 5.0)
    return TrackedObject(
        track_id=track_id,
        class_name="person",
        confidence=confidence,
        bbox=bbox,
        center=(center_x, center_y),
        timestamp=timestamp,
        frame_number=frame_number,
    )


class ObjectMemoryTests(unittest.TestCase):
    def test_history_preserves_state_fields_and_recent_trajectory(self) -> None:
        memory = ObjectMemory(max_history_seconds=10.0)
        memory.update([_tracked(7, 0.0, frame_number=10)], 0.0)
        memory.update([_tracked(7, 3.0, frame_number=25)], 0.5)
        memory.update([_tracked(7, 6.0, frame_number=40)], 1.0)

        state = memory.get(7)

        self.assertIsInstance(state, ObjectState)
        self.assertEqual(state.frame_number, 40)
        self.assertEqual(state.class_name, "person")
        self.assertEqual(state.bbox, (1.0, 45.0, 11.0, 55.0))
        self.assertEqual(memory.get_trajectory(7, seconds=0.6), [(3.0, 50.0), (6.0, 50.0)])
        self.assertEqual(len(memory.history[7]), 3)

    def test_speed_uses_timestamp_not_processing_rate(self) -> None:
        memory = ObjectMemory()
        memory.update([_tracked(2, 0.0, frame_number=0)], 0.0)
        memory.update([_tracked(2, 10.0, frame_number=100)], 2.0)

        motion = memory.get_velocity(2)

        self.assertIsNotNone(motion)
        self.assertAlmostEqual(motion.dx, 10.0)
        self.assertAlmostEqual(motion.dy, 0.0)
        self.assertAlmostEqual(motion.displacement_pixels, 10.0)
        self.assertAlmostEqual(motion.speed_pixels_per_second, 5.0)
        self.assertEqual(motion.direction, "right")
        self.assertEqual(motion.duration_seconds, 2.0)

    def test_stationary_duration_is_trailing_stationary_time(self) -> None:
        memory = ObjectMemory()
        for timestamp, frame in [(0.0, 0), (1.0, 1), (3.0, 3)]:
            memory.update([_tracked(5, 20.0, frame_number=frame)], timestamp)

        self.assertTrue(memory.is_stationary(5))
        self.assertAlmostEqual(memory.get_stationary_duration(5), 3.0)
        self.assertAlmostEqual(memory.get_track_age(5), 3.0)
        self.assertAlmostEqual(memory.get_last_seen(5), 3.0)

    def test_acceleration_is_derived_from_three_timestamped_states(self) -> None:
        memory = ObjectMemory()
        memory.update([_tracked(4, 0.0)], 0.0)
        memory.update([_tracked(4, 1.0)], 1.0)
        memory.update([_tracked(4, 3.0)], 2.0)

        motion = memory.get_velocity(4)

        self.assertIsNotNone(motion)
        self.assertAlmostEqual(motion.speed_pixels_per_second, 2.0)
        self.assertAlmostEqual(motion.acceleration_pixels_per_second_squared, 1.0)

    def test_history_is_bounded_by_time_and_state_count(self) -> None:
        memory = ObjectMemory(max_history_seconds=2.0, max_states_per_track=3)
        for timestamp in range(6):
            memory.update([_tracked(1, float(timestamp), frame_number=timestamp)], float(timestamp))

        history = memory.get_history(1)

        self.assertLessEqual(len(history), 3)
        self.assertEqual([state.timestamp for state in history], [3.0, 4.0, 5.0])

    def test_missing_update_does_not_fabricate_a_state(self) -> None:
        memory = ObjectMemory()
        memory.update([_tracked(8, 1.0)], 0.0)
        memory.update([], 1.0)

        self.assertEqual(len(memory.get_history(8)), 1)
        self.assertEqual(memory.get_last_seen(8), 0.0)

    def test_non_monotonic_updates_are_rejected(self) -> None:
        memory = ObjectMemory()
        memory.update([_tracked(3, 1.0)], 2.0)

        with self.assertRaises(NonMonotonicTimestampError):
            memory.update([_tracked(3, 2.0)], 1.0)

    def test_trajectory_visualization_returns_annotated_copy(self) -> None:
        memory = ObjectMemory()
        memory.update([_tracked(9, 10.0, 10.0)], 0.0)
        memory.update([_tracked(9, 30.0, 20.0)], 1.0)
        frame = np.zeros((80, 100, 3), dtype=np.uint8)

        annotated = draw_memory_trajectory(frame, memory, 9, seconds=2.0)

        self.assertEqual(annotated.shape, frame.shape)
        self.assertGreater(int(np.count_nonzero(annotated)), 0)
        self.assertTrue(np.array_equal(frame, np.zeros_like(frame)))


if __name__ == "__main__":
    unittest.main()

