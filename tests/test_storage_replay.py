"""Deterministic SQLite and evidence-replay tests for KAVACH Module 9."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from kavach.behaviours import BehaviourEvent
from kavach.incidents import EvidenceReplay, IncidentManager, ReplayError
from kavach.risk import RiskEngine
from kavach.storage import DatabaseError, EventDatabase
from kavach.video import VideoWriter


class StorageReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.video_path = self.root / "warehouse.avi"
        self.database_path = self.root / "events.sqlite3"
        self.clip_dir = self.root / "clips"
        self._create_video()
        self.database = EventDatabase(self.database_path)
        self.database.register_video(
            "warehouse-1",
            self.video_path,
            fps=10.0,
            width=64,
            height=48,
            total_frames=80,
            duration=8.0,
        )
        self.manager = IncidentManager(risk_engine=RiskEngine())

    def tearDown(self) -> None:
        self.database.close()
        self.temp_dir.cleanup()

    def _create_video(self) -> None:
        with VideoWriter(
            self.video_path,
            fps=10.0,
            width=64,
            height=48,
            codec="MJPG",
        ) as writer:
            for frame_number in range(80):
                writer.write(
                    np.full(
                        (48, 64, 3),
                        (frame_number % 255, 40, 80),
                        dtype=np.uint8,
                    )
                )

    @staticmethod
    def _event(
        timestamp: float,
        *,
        event_type: str = "ZONE_VIOLATION",
        entity: str = "person_3",
    ) -> BehaviourEvent:
        return BehaviourEvent(
            type=event_type,
            timestamp=timestamp,
            entities=(entity, "restricted_zone"),
            confidence=0.85,
            evidence={
                "inside": True,
                "zone_name": "RESTRICTED_ZONE",
                "boundary_distance_px": 0.0,
            },
        )

    def _store(self, event: BehaviourEvent) -> str:
        incident = self.manager.ingest(event).incident
        return self.database.insert_event("warehouse-1", incident)

    def test_insert_and_query_preserve_structured_event_fields(self) -> None:
        first_id = self._store(self._event(2.0))
        second_id = self._store(
            self._event(
                5.0,
                event_type="POSSIBLE_DROP",
                entity="carton_7",
            )
        )

        first = self.database.get_event(first_id)
        self.assertIsNotNone(first)
        assert first is not None
        self.assertEqual(first["video_id"], "warehouse-1")
        self.assertEqual(first["behaviour"], "ZONE_VIOLATION")
        self.assertEqual(first["entities"], ["person_3", "restricted_zone"])
        self.assertIsInstance(first["risk"], dict)
        self.assertIsInstance(first["evidence"], dict)
        self.assertEqual(first["review_status"], "NEW")
        self.assertIsNone(first["clip_path"])

        self.assertEqual(
            [row["event_id"] for row in self.database.get_events("warehouse-1")],
            [first_id, second_id],
        )
        self.assertEqual(
            len(self.database.get_events_by_type("ZONE_VIOLATION")),
            1,
        )
        self.assertEqual(
            len(self.database.get_events_by_risk(minimum_score=0.0)),
            2,
        )
        self.assertEqual(
            len(self.database.get_events_between_times("warehouse-1", 1.0, 2.0)),
            1,
        )
        self.assertEqual(
            [row["event_id"] for row in self.database.get_events_for_entity("carton_7")],
            [second_id],
        )

        statistics = self.database.get_event_statistics("warehouse-1")
        self.assertEqual(statistics["total_events"], 2)
        self.assertEqual(statistics["by_behaviour"]["ZONE_VIOLATION"], 1)
        self.assertEqual(statistics["by_review_status"]["NEW"], 2)

    def test_review_status_and_clip_path_are_updateable(self) -> None:
        event_id = self._store(self._event(2.0))

        reviewed = self.database.update_review_status(event_id, "REVIEWED")
        self.assertEqual(reviewed["review_status"], "REVIEWED")
        false_positive = self.database.update_review_status(
            event_id,
            "FALSE_POSITIVE",
        )
        self.assertEqual(false_positive["review_status"], "FALSE_POSITIVE")
        self.assertEqual(false_positive["lifecycle"], "CLOSED")
        updated = self.database.set_clip_path(event_id, self.clip_dir / "clip.avi")
        self.assertEqual(
            updated["clip_path"],
            str((self.clip_dir / "clip.avi").resolve()),
        )

    def test_events_survive_database_reopen(self) -> None:
        event_id = self._store(self._event(2.0))
        self.database.close()
        self.database = EventDatabase(self.database_path)

        event = self.database.get_event(event_id)

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event["event_id"], event_id)
        self.assertEqual(event["video_id"], "warehouse-1")

    def test_analysis_metadata_survives_database_reopen(self) -> None:
        self.database.set_video_analysis_metadata(
            "warehouse-1",
            {
                "processing_fps": 12.5,
                "device": "cpu",
                "unique_objects": {"person": 2},
            },
        )
        self.database.close()
        self.database = EventDatabase(self.database_path)

        video = self.database.get_video("warehouse-1")

        assert video is not None
        self.assertIn("analysis_metadata_json", video)
        self.assertIn("12.5", str(video["analysis_metadata_json"]))

    def test_clear_events_for_video_is_idempotent_for_reruns(self) -> None:
        event_id = self._store(self._event(2.0))

        clip_paths = self.database.clear_events_for_video("warehouse-1")

        self.assertEqual(clip_paths, [])
        self.assertIsNone(self.database.get_event(event_id))
        self.assertEqual(self.database.get_event_statistics("warehouse-1")["total_events"], 0)

    def test_replay_clamps_start_and_end_and_associates_clip(self) -> None:
        start_id = self._store(self._event(0.2))
        end_id = self._store(
            self._event(
                7.8,
                event_type="POSSIBLE_DROP",
                entity="carton_7",
            )
        )
        replay = EvidenceReplay(
            self.database,
            output_dir=self.clip_dir,
            pre_seconds=3.0,
            post_seconds=3.0,
            codec="MJPG",
        )

        start_result = replay.create_clip(
            start_id,
            output=self.clip_dir / "start.avi",
        )
        end_result = replay.create_clip(
            end_id,
            output=self.clip_dir / "end.avi",
        )

        self.assertEqual(start_result.start_frame, 0)
        self.assertEqual(start_result.end_frame, 32)
        self.assertEqual(start_result.frames_written, 33)
        self.assertEqual(end_result.start_frame, 48)
        self.assertEqual(end_result.end_frame, 79)
        self.assertEqual(end_result.frames_written, 32)
        self.assertTrue(start_result.clip_path.is_file())
        self.assertTrue(end_result.clip_path.is_file())
        self.assertEqual(
            self.database.get_event(start_id)["clip_path"],
            str(start_result.clip_path),
        )
        self.assertEqual(
            self.database.get_event(end_id)["clip_path"],
            str(end_result.clip_path),
        )

    def test_existing_clip_is_reused_without_reencoding(self) -> None:
        event_id = self._store(self._event(2.0))
        replay = EvidenceReplay(
            self.database,
            output_dir=self.clip_dir,
            codec="MJPG",
        )
        output = self.clip_dir / "cached.avi"
        first = replay.create_clip(event_id, output=output)
        first_size = output.stat().st_size
        second = replay.create_clip(event_id, output=output)

        self.assertFalse(first.reused)
        self.assertTrue(second.reused)
        self.assertEqual(first_size, output.stat().st_size)
        self.assertEqual(first.frames_written, second.frames_written)

    def test_unknown_event_and_invalid_query_are_rejected(self) -> None:
        replay = EvidenceReplay(self.database, output_dir=self.clip_dir)
        with self.assertRaises(ReplayError):
            replay.create_clip("missing-event")
        with self.assertRaises(DatabaseError):
            self.database.get_events_between_times("warehouse-1", 3.0, 1.0)


if __name__ == "__main__":
    unittest.main()
