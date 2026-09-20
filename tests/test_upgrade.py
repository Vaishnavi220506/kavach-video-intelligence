"""Tests for the warehouse-model, evidence, and prevention upgrade."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from kavach.behaviours import BehaviourEvent
from kavach.incidents import EvidenceStore, IncidentManager
from kavach.perception import EvaluationError, evaluate_checkpoint
from kavach.prevention import PreventionEngine
from kavach.risk import RiskEngine
from kavach.storage import EventDatabase


class UpgradeTests(unittest.TestCase):
    def test_snapshot_hash_can_detect_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = EvidenceStore(Path(directory) / "snapshots")
            artifact = store.capture_snapshot(
                "video-1",
                "INC-000001",
                np.zeros((24, 32, 3), dtype=np.uint8),
                timestamp=1.5,
                frame_number=12,
            )
            self.assertTrue(store.verify(artifact))
            Path(artifact.path).write_bytes(b"changed")
            self.assertFalse(store.verify(artifact))

    def test_review_history_and_evidence_patch_are_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = EventDatabase(Path(directory) / "events.sqlite3")
            try:
                database.register_video("video-1", Path(directory) / "input.mp4")
                event = BehaviourEvent(
                    type="POSSIBLE_THROWING",
                    timestamp=1.0,
                    entities=("carton_7",),
                    confidence=0.8,
                    evidence={"speed_px_per_second": 250.0},
                )
                incident = IncidentManager(risk_engine=RiskEngine()).ingest(event).incident
                event_id = database.insert_event("video-1", incident)
                database.update_event_evidence(event_id, {"custom_note": "inspect"})
                reviewed = database.update_review_status(event_id, "REVIEWED")
                self.assertEqual(reviewed["review_status"], "REVIEWED")
                self.assertEqual(reviewed["evidence"]["custom_note"], "inspect")
                self.assertEqual(reviewed["evidence"]["review_history"][-1]["status"], "REVIEWED")
            finally:
                database.close()

    def test_prevention_rule_is_structured_and_deterministic(self) -> None:
        recommendations = PreventionEngine().recommend({"type": "COLLISION_RISK"})
        self.assertEqual(recommendations[0]["rule_id"], "P-COLLISION-01")
        self.assertEqual(recommendations[0]["root_cause"], "layout")

    def test_evaluation_refuses_missing_labeled_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.write_bytes(b"not-a-model")
            with self.assertRaises(EvaluationError):
                evaluate_checkpoint(checkpoint, Path(directory) / "missing.yaml")


if __name__ == "__main__":
    unittest.main()

