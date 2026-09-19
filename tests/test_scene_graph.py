"""Synthetic tests for KAVACH Module 6 scene graph relations."""

from __future__ import annotations

import unittest

import numpy as np

from kavach.intelligence import (
    ABOVE,
    APPROACHING,
    MOVING_WITH,
    SEPARATING_FROM,
    STAGING_ZONE,
    ObjectMemory,
    RelationThresholds,
    SceneGraph,
    ZoneManager,
    bbox_center,
    draw_scene_graph,
)
from kavach.perception import TrackedObject


def _tracked(
    track_id: int,
    class_name: str,
    bbox: tuple[float, float, float, float],
    timestamp: float,
    frame_number: int,
) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        class_name=class_name,
        confidence=0.9,
        bbox=bbox,
        center=bbox_center(bbox),
        timestamp=timestamp,
        frame_number=frame_number,
    )


def _zone_manager() -> ZoneManager:
    return ZoneManager.from_config(
        {
            STAGING_ZONE: {
                "polygon": ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)),
                "color": (0, 255, 0),
            }
        }
    )


class SceneGraphTests(unittest.TestCase):
    def test_nodes_reference_current_states_and_relations_keep_evidence(self) -> None:
        person = _tracked(3, "person", (10.0, 10.0, 50.0, 50.0), 1.0, 10)
        carton = _tracked(7, "carton", (35.0, 20.0, 75.0, 40.0), 1.0, 10)
        pallet = _tracked(2, "pallet", (35.0, 40.0, 75.0, 70.0), 1.0, 10)
        graph = SceneGraph(
            zones=_zone_manager(),
            thresholds=RelationThresholds(
                near_distance_px=100.0,
                overlap_iou_threshold=0.05,
                support_min_horizontal_overlap_ratio=0.8,
                support_max_vertical_gap_px=5.0,
            ),
        )

        snapshot = graph.update([person, carton, pallet], timestamp=1.0)

        self.assertEqual(
            {node.node_id for node in snapshot.nodes},
            {"person_3", "carton_7", "pallet_2", "staging_zone"},
        )
        carton_node = next(node for node in graph.nodes if node.node_id == "carton_7")
        self.assertIs(carton_node.state, carton)
        self.assertTrue(graph.has_relation("carton_7", "supported_by", "pallet_2"))
        self.assertTrue(graph.has_relation(7, ABOVE, 2))
        self.assertIn("person_3", graph.get_nearby(7))
        self.assertIn("pallet_2", graph.get_nearby(7))
        self.assertEqual(
            set(graph.get_objects_in_zone(STAGING_ZONE)),
            {"person_3", "carton_7", "pallet_2"},
        )

        for edge in graph.relations:
            self.assertTrue(edge.evidence)
            self.assertTrue(all(isinstance(value, float) for value in edge.evidence.values()))
        support_edge = next(
            edge
            for edge in graph.relations
            if edge.subject == "carton_7" and edge.relation == "supported_by"
        )
        self.assertAlmostEqual(support_edge.evidence["horizontal_overlap_ratio"], 1.0)
        self.assertAlmostEqual(support_edge.evidence["vertical_gap_px"], 0.0)

    def test_approaching_and_separating_use_object_memory(self) -> None:
        memory = ObjectMemory(stationary_speed_threshold=0.1)
        graph = SceneGraph(
            memory=memory,
            thresholds=RelationThresholds(
                near_distance_px=120.0,
                motion_min_distance_change_px=1.0,
            ),
        )
        frames = (
            (
                _tracked(3, "person", (10.0, 40.0, 30.0, 60.0), 0.0, 0),
                _tracked(7, "carton", (90.0, 40.0, 110.0, 60.0), 0.0, 0),
            ),
            (
                _tracked(3, "person", (20.0, 40.0, 40.0, 60.0), 1.0, 1),
                _tracked(7, "carton", (80.0, 40.0, 100.0, 60.0), 1.0, 1),
            ),
            (
                _tracked(3, "person", (15.0, 40.0, 35.0, 60.0), 2.0, 2),
                _tracked(7, "carton", (85.0, 40.0, 105.0, 60.0), 2.0, 2),
            ),
        )

        for timestamp, objects in enumerate(frames):
            memory.update(objects, float(timestamp))
            graph.update(objects, timestamp=float(timestamp))
            if timestamp == 1:
                self.assertTrue(graph.has_relation(3, APPROACHING, 7))
            if timestamp == 2:
                self.assertTrue(graph.has_relation(3, SEPARATING_FROM, 7))

        self.assertEqual(len(graph.snapshots), 3)
        self.assertIn("separating_from", graph.debug_log())
        self.assertEqual(graph.to_dict()["timestamp"], 2.0)

    def test_moving_with_requires_similar_timestamped_motion(self) -> None:
        memory = ObjectMemory(stationary_speed_threshold=0.1)
        graph = SceneGraph(
            memory=memory,
            thresholds=RelationThresholds(
                moving_with_max_distance_px=100.0,
                moving_with_velocity_delta_px_per_second=1.0,
                moving_with_min_speed_px_per_second=1.0,
            ),
        )
        first = (
            _tracked(1, "person", (10.0, 40.0, 30.0, 60.0), 0.0, 0),
            _tracked(2, "forklift", (50.0, 40.0, 80.0, 60.0), 0.0, 0),
        )
        second = (
            _tracked(1, "person", (20.0, 40.0, 40.0, 60.0), 1.0, 1),
            _tracked(2, "forklift", (60.0, 40.0, 90.0, 60.0), 1.0, 1),
        )
        memory.update(first, 0.0)
        graph.update(first, timestamp=0.0)
        memory.update(second, 1.0)
        graph.update(second, timestamp=1.0)

        self.assertTrue(graph.has_relation("person_1", MOVING_WITH, "forklift_2"))

    def test_debug_visualization_returns_an_annotated_copy(self) -> None:
        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        objects = [_tracked(7, "carton", (20.0, 20.0, 50.0, 50.0), 0.0, 0)]
        graph = SceneGraph(zones=_zone_manager())
        graph.update(objects, timestamp=0.0)

        annotated = draw_scene_graph(frame, graph)

        self.assertEqual(annotated.shape, frame.shape)
        self.assertGreater(int(np.count_nonzero(annotated)), 0)
        self.assertTrue(np.array_equal(frame, np.zeros_like(frame)))


if __name__ == "__main__":
    unittest.main()

