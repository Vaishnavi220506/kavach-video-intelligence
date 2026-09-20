"""Tests for the shared client-facing evidence contract.

`kavach.presentation` is the single definition of the shape the API returns
and the static demo fixtures are generated in. If it drifts, the published
site and the live service start describing the same record differently, which
is exactly the failure this module exists to prevent.
"""

from __future__ import annotations

import json
import unittest

from kavach.presentation import (
    ACTIVITY_EVENT_TYPES,
    ANOMALY_EVENT_TYPES,
    SAFETY_EVENT_TYPES,
    SUPPORTED_BEHAVIOURS,
    event_payload,
    evidence_graph,
    signal_kind,
)


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "event_id": "INC-000001",
        "video_id": "video-1",
        "timestamp": 75.0,
        "behaviour": "POSSIBLE_DROP",
        "entities": ["carton_8", "pallet_4"],
        "risk": {"score": 56.1, "category": "MEDIUM"},
        "evidence": {},
        "review_status": "NEW",
    }
    row.update(overrides)
    return row


class SignalClassificationTests(unittest.TestCase):
    def test_every_supported_behaviour_classifies(self) -> None:
        for behaviour in SUPPORTED_BEHAVIOURS:
            self.assertIn(signal_kind(behaviour), {"activity", "anomaly", "safety"})

    def test_classes_do_not_overlap(self) -> None:
        self.assertFalse(ACTIVITY_EVENT_TYPES & ANOMALY_EVENT_TYPES)
        self.assertFalse(ACTIVITY_EVENT_TYPES & SAFETY_EVENT_TYPES)
        self.assertFalse(ANOMALY_EVENT_TYPES & SAFETY_EVENT_TYPES)

    def test_unknown_behaviour_is_treated_as_safety(self) -> None:
        # Failing closed matters: an unrecognised rule should surface for
        # review rather than being filed away as background activity.
        self.assertEqual(signal_kind("SOMETHING_NEW"), "safety")


class EventPayloadTests(unittest.TestCase):
    def test_adds_display_and_routing_fields(self) -> None:
        payload = event_payload(_row())
        self.assertEqual(payload["timestamp_display"], "01:15")
        self.assertEqual(payload["signal_kind"], "safety")
        self.assertEqual(payload["replay_url"], "/api/events/INC-000001/replay")
        self.assertEqual(payload["clip_url"], "/api/events/INC-000001/clip")

    def test_does_not_mutate_the_source_row(self) -> None:
        row = _row()
        event_payload(row)
        self.assertNotIn("timestamp_display", row)

    def test_flattens_the_latest_event_evidence(self) -> None:
        payload = event_payload(
            _row(
                evidence={
                    "initial_event": {"evidence": {"distance_px": 10.0}},
                    "latest_event": {"evidence": {"distance_px": 53.3}},
                }
            )
        )
        # latest_event wins: a supervisor reads the current state of a
        # continuing incident, not how it started.
        self.assertEqual(payload["evidence_summary"], {"distance_px": 53.3})

    def test_falls_back_to_raw_evidence(self) -> None:
        payload = event_payload(_row(evidence={"speed_px_per_second": 4.0}))
        self.assertEqual(payload["evidence_summary"], {"speed_px_per_second": 4.0})

    def test_prevention_and_root_cause_are_surfaced(self) -> None:
        payload = event_payload(
            _row(
                evidence={
                    "prevention": [{"rule_id": "P-PROX-01"}],
                    "root_cause_category": "layout",
                }
            )
        )
        self.assertEqual(payload["prevention"], [{"rule_id": "P-PROX-01"}])
        self.assertEqual(payload["root_cause_category"], "layout")

    def test_verification_can_be_skipped_for_static_fixtures(self) -> None:
        # The published site ships no snapshot files. An unconditional False
        # would read as a failed integrity check rather than an absent one, so
        # the key is omitted instead.
        row = _row(
            evidence={
                "evidence_artifacts": [
                    {"path": "missing.jpg", "sha256": "0" * 64},
                ]
            }
        )
        skipped = event_payload(row, verify_artifacts=False)
        self.assertNotIn("verified", skipped["evidence_artifacts"][0])
        self.assertEqual(
            skipped["evidence_artifacts"][0]["url"],
            "/api/events/INC-000001/snapshot/0",
        )

        checked = event_payload(row, verify_artifacts=True)
        self.assertIs(checked["evidence_artifacts"][0]["verified"], False)

    def test_an_unverifiable_artifact_does_not_break_the_record(self) -> None:
        # EvidenceStore.verify raises when an artifact record has no digest.
        # One malformed row must not fail the whole video response.
        payload = event_payload(
            _row(evidence={"evidence_artifacts": [{"path": "no-digest.jpg"}]}),
            verify_artifacts=True,
        )
        self.assertIs(payload["evidence_artifacts"][0]["verified"], False)

    def test_payload_is_json_serialisable(self) -> None:
        json.dumps(event_payload(_row()), default=str)


class EvidenceGraphTests(unittest.TestCase):
    def test_links_every_entity_to_its_incident(self) -> None:
        graph = evidence_graph([event_payload(_row())])
        node_ids = {node["id"] for node in graph["nodes"]}
        self.assertIn("event:INC-000001", node_ids)
        self.assertIn("entity:carton_8", node_ids)
        self.assertIn("entity:pallet_4", node_ids)
        self.assertEqual(len(graph["edges"]), 2)

    def test_shared_entities_are_one_node(self) -> None:
        events = [
            event_payload(_row()),
            event_payload(_row(event_id="INC-000002", entities=["carton_8"])),
        ]
        graph = evidence_graph(events)
        entity_nodes = [node for node in graph["nodes"] if node["kind"] == "entity"]
        self.assertEqual(len([n for n in entity_nodes if n["id"] == "entity:carton_8"]), 1)
        self.assertEqual(len(graph["edges"]), 3)

    def test_handles_an_event_with_no_entities(self) -> None:
        graph = evidence_graph([event_payload(_row(entities=[]))])
        self.assertEqual(graph["edges"], [])
        self.assertEqual(len(graph["nodes"]), 1)


if __name__ == "__main__":
    unittest.main()
