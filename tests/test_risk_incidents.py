"""Deterministic tests for KAVACH Module 8 risk and incident intelligence."""

from __future__ import annotations

import unittest

from kavach.behaviours import BehaviourEvent
from kavach.incidents import (
    CLOSED,
    FALSE_POSITIVE,
    NEW,
    OPEN,
    REVIEWED,
    IncidentManager,
    IncidentManagerError,
)
from kavach.risk import (
    HIGH,
    RiskEngine,
)


def _event(
    timestamp: float,
    *,
    event_type: str = "UNSAFE_HUMAN_FORKLIFT_PROXIMITY",
    entities: tuple[str, ...] = ("person_3", "forklift_1"),
    confidence: float = 0.91,
    evidence: dict[str, object] | None = None,
) -> BehaviourEvent:
    return BehaviourEvent(
        type=event_type,
        timestamp=timestamp,
        entities=entities,
        confidence=confidence,
        evidence=evidence
        or {
            "distance_px": 20.0,
            "maximum_distance_px": 100.0,
            "approaching": True,
            "human_speed_px_per_second": 50.0,
        },
    )


class RiskEngineTests(unittest.TestCase):
    def test_score_is_bounded_and_confidence_is_separate(self) -> None:
        event = _event(1.0, confidence=0.91)

        assessment = RiskEngine().assess(event)

        self.assertGreaterEqual(assessment.score, 0.0)
        self.assertLessEqual(assessment.score, 100.0)
        self.assertEqual(assessment.category, HIGH)
        self.assertEqual(assessment.detection_confidence, 0.91)
        self.assertNotEqual(assessment.score, assessment.detection_confidence * 100.0)

    def test_breakdown_explains_weighted_score(self) -> None:
        assessment = RiskEngine().assess(
            _event(
                1.0,
                event_type="POSSIBLE_DRAGGING",
                entities=("carton_7", "person_3"),
                confidence=0.72,
                evidence={
                    "near_floor": True,
                    "horizontal_displacement_px": 148.0,
                    "duration_seconds": 2.3,
                    "person_nearby": True,
                },
            )
        )
        breakdown = assessment.breakdown

        self.assertEqual(
            set(breakdown.components),
            {
                "severity",
                "duration",
                "motion_intensity",
                "spatial_context",
                "repeat_frequency",
                "detection_confidence",
            },
        )
        self.assertAlmostEqual(
            sum(breakdown.weighted_contributions.values()),
            assessment.score,
        )
        self.assertGreater(breakdown.components["duration"], 0.0)
        self.assertGreater(breakdown.components["spatial_context"], 0.0)

    def test_repeat_frequency_uses_matching_prior_events(self) -> None:
        engine = RiskEngine()
        first = _event(1.0)
        second = _event(2.0)

        assessment = engine.assess(second, recent_events=(first,))

        self.assertAlmostEqual(
            assessment.breakdown.components["repeat_frequency"],
            100.0 / 3.0,
        )

    def test_unknown_event_uses_configured_default_severity(self) -> None:
        assessment = RiskEngine().assess(
            _event(
                1.0,
                event_type="FUTURE_EVENT",
                entities=("object_1",),
                confidence=0.5,
                evidence={"custom_signal": True},
            )
        )

        self.assertEqual(assessment.breakdown.components["severity"], 40.0)

    def test_yaml_policy_loads_with_explicit_categories(self) -> None:
        engine = RiskEngine()

        self.assertEqual(engine.config.category_thresholds["CRITICAL"], 75.0)
        self.assertAlmostEqual(sum(engine.config.weights.values()), 1.0)


class IncidentManagerTests(unittest.TestCase):
    def test_repeated_events_share_one_incident(self) -> None:
        manager = IncidentManager(
            risk_engine=RiskEngine(),
            dedup_window_seconds=10.0,
            cooldown_seconds=3.0,
        )

        first = manager.ingest(_event(1.0))
        repeated = manager.ingest(_event(2.0))

        self.assertTrue(first.created)
        self.assertFalse(repeated.created)
        self.assertTrue(repeated.deduplicated)
        self.assertEqual(first.incident.id, repeated.incident.id)
        self.assertEqual(repeated.incident.occurrence_count, 2)
        self.assertEqual(repeated.incident.evidence["last_merge_reason"], "cooldown")

    def test_event_after_dedup_window_creates_new_incident(self) -> None:
        manager = IncidentManager(
            risk_engine=RiskEngine(),
            dedup_window_seconds=5.0,
            cooldown_seconds=1.0,
        )

        first = manager.ingest(_event(1.0))
        later = manager.ingest(_event(7.0))

        self.assertTrue(later.created)
        self.assertNotEqual(first.incident.id, later.incident.id)
        self.assertEqual(len(manager.incidents), 2)

    def test_review_and_lifecycle_transitions_are_explicit(self) -> None:
        manager = IncidentManager(risk_engine=RiskEngine())
        incident = manager.ingest(_event(1.0)).incident

        self.assertEqual(incident.status, NEW)
        self.assertEqual(incident.lifecycle, OPEN)
        manager.mark_reviewed(incident.id)
        self.assertEqual(incident.status, REVIEWED)
        manager.close(incident.id)
        self.assertEqual(incident.lifecycle, CLOSED)
        manager.reopen(incident.id)
        self.assertEqual(incident.status, NEW)
        self.assertEqual(incident.lifecycle, OPEN)
        manager.mark_false_positive(incident.id)
        self.assertEqual(incident.status, FALSE_POSITIVE)
        self.assertEqual(incident.lifecycle, CLOSED)

    def test_false_positive_is_not_merged_into(self) -> None:
        manager = IncidentManager(risk_engine=RiskEngine())
        first = manager.ingest(_event(1.0)).incident
        manager.mark_false_positive(first.id)

        second = manager.ingest(_event(2.0))

        self.assertTrue(second.created)
        self.assertNotEqual(first.id, second.incident.id)

    def test_non_monotonic_events_are_rejected(self) -> None:
        manager = IncidentManager(risk_engine=RiskEngine())
        manager.ingest(_event(3.0))

        with self.assertRaises(IncidentManagerError):
            manager.ingest(_event(2.0))


if __name__ == "__main__":
    unittest.main()
