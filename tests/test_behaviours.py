"""Deterministic synthetic tests for KAVACH Module 7 behaviours."""

from __future__ import annotations

import unittest

from kavach.behaviours import (
    AISLE_OBSTRUCTION,
    COLLISION_RISK,
    IMPROPER_PLACEMENT,
    MOTION_ANOMALY,
    OBJECT_ACTIVITY,
    PALLET_OVERHANG,
    POSSIBLE_DRAGGING,
    POSSIBLE_DROP,
    POSSIBLE_ROUGH_HANDLING,
    POSSIBLE_THROWING,
    UNSAFE_HUMAN_FORKLIFT_PROXIMITY,
    UNSTABLE_STACK,
    ZONE_VIOLATION,
    AisleObstructionDetector,
    BehaviourContext,
    CollisionRiskDetector,
    DraggingDetector,
    HumanForkliftProximityDetector,
    ImproperPlacementDetector,
    MotionAnomalyDetector,
    ObjectActivityDetector,
    OverhangDetector,
    PossibleDropDetector,
    RoughHandlingDetector,
    ThrowingDetector,
    UnstableStackDetector,
    ZoneTransitionDetector,
    ZoneViolationDetector,
    build_default_registry,
)
from kavach.intelligence import ObjectMemory, RelationThresholds, SceneGraph, ZoneManager
from kavach.perception import TrackedObject


def _bbox(center_x: float, center_y: float, width: float = 20.0, height: float = 20.0):
    return (
        center_x - width / 2.0,
        center_y - height / 2.0,
        center_x + width / 2.0,
        center_y + height / 2.0,
    )


def _tracked(
    track_id: int,
    class_name: str,
    center_x: float,
    center_y: float,
    timestamp: float,
    frame_number: int,
    *,
    width: float = 20.0,
    height: float = 20.0,
) -> TrackedObject:
    bbox = _bbox(center_x, center_y, width, height)
    return TrackedObject(
        track_id=track_id,
        class_name=class_name,
        confidence=0.9,
        bbox=bbox,
        center=(center_x, center_y),
        timestamp=timestamp,
        frame_number=frame_number,
    )


def _context(
    memory: ObjectMemory,
    graph: SceneGraph,
    objects: tuple[TrackedObject, ...],
    timestamp: float,
    *,
    frame_width: int = 300,
    frame_height: int = 100,
) -> BehaviourContext:
    memory.update(objects, timestamp)
    graph.update(objects, timestamp=timestamp)
    return BehaviourContext(
        timestamp=timestamp,
        memory=memory,
        scene_graph=graph,
        frame_width=frame_width,
        frame_height=frame_height,
    )


def _restricted_zone() -> ZoneManager:
    return ZoneManager.from_config(
        {
            "RESTRICTED_ZONE": [
                (0.0, 0.0),
                (100.0, 0.0),
                (100.0, 100.0),
                (0.0, 100.0),
            ]
        }
    )


class BehaviourDetectorTests(unittest.TestCase):
    def test_zone_transition_emits_entry_and_exit(self) -> None:
        memory = ObjectMemory()
        zones = _restricted_zone()
        graph = SceneGraph(zones=zones)
        detector = ZoneTransitionDetector(cooldown_seconds=0.0)

        outside = (_tracked(3, "person", 140.0, 40.0, 0.0, 0),)
        inside = (_tracked(3, "person", 40.0, 40.0, 1.0, 1),)
        self.assertEqual(detector.detect(_context(memory, graph, outside, 0.0)), [])
        entered = detector.detect(_context(memory, graph, inside, 1.0))
        self.assertEqual([event.type for event in entered], ["ZONE_TRANSITION"])
        self.assertEqual(entered[0].evidence["transition"], "entered")
        exited = detector.detect(_context(memory, graph, outside, 2.0))
        self.assertEqual([event.type for event in exited], ["ZONE_TRANSITION"])
        self.assertEqual(exited[0].evidence["transition"], "exited")

    def test_object_activity_emits_initial_state_once_and_state_change(self) -> None:
        memory = ObjectMemory(stationary_speed_threshold=1.0)
        graph = SceneGraph()
        detector = ObjectActivityDetector(
            minimum_track_age_seconds=0.0,
            minimum_moving_speed_px_per_second=8.0,
            cooldown_seconds=0.0,
        )
        first = (_tracked(7, "person", 10.0, 50.0, 0.0, 0),)
        moving = (_tracked(7, "person", 30.0, 50.0, 1.0, 1),)
        moving_again = (_tracked(7, "person", 40.0, 50.0, 2.0, 2),)
        self.assertEqual(detector.detect(_context(memory, graph, first, 0.0)), [])
        initial = detector.detect(_context(memory, graph, moving, 1.0))
        self.assertEqual([event.type for event in initial], [OBJECT_ACTIVITY])
        self.assertEqual(initial[0].evidence["activity_state"], "moving")
        self.assertEqual(detector.detect(_context(memory, graph, moving_again, 2.0)), [])
        stationary = (_tracked(7, "person", 40.0, 50.0, 3.0, 3),)
        stopped = detector.detect(_context(memory, graph, stationary, 3.0))
        self.assertEqual([event.type for event in stopped], [OBJECT_ACTIVITY])
        self.assertEqual(stopped[0].evidence["transition"], "moving_to_stationary")

    def test_motion_anomaly_detects_speed_spike_with_evidence(self) -> None:
        memory = ObjectMemory(stationary_speed_threshold=1.0)
        graph = SceneGraph()
        detector = MotionAnomalyDetector(
            minimum_history_states=5,
            speed_ratio_threshold=2.0,
            minimum_speed_spike_px_per_second=30.0,
            minimum_direction_speed_px_per_second=1000.0,
            cooldown_seconds=0.0,
        )
        frames = (
            (_tracked(7, "person", 0.0, 40.0, 0.0, 0),),
            (_tracked(7, "person", 5.0, 40.0, 1.0, 1),),
            (_tracked(7, "person", 10.0, 40.0, 2.0, 2),),
            (_tracked(7, "person", 15.0, 40.0, 3.0, 3),),
            (_tracked(7, "person", 80.0, 40.0, 4.0, 4),),
            (_tracked(7, "person", 145.0, 40.0, 5.0, 5),),
        )
        for index, objects in enumerate(frames):
            events = detector.detect(_context(memory, graph, objects, float(index)))
        self.assertEqual([event.type for event in events], [MOTION_ANOMALY])
        self.assertEqual(events[0].evidence["reason"], "speed_spike")
        self.assertGreater(events[0].evidence["speed_ratio"], 2.0)

    def test_motion_anomaly_does_not_flag_steady_motion(self) -> None:
        memory = ObjectMemory(stationary_speed_threshold=1.0)
        graph = SceneGraph()
        detector = MotionAnomalyDetector(
            minimum_history_states=5,
            minimum_speed_spike_px_per_second=30.0,
            minimum_direction_speed_px_per_second=1000.0,
        )
        for index in range(6):
            objects = (_tracked(7, "person", float(index * 10), 40.0, float(index), index),)
            events = detector.detect(_context(memory, graph, objects, float(index)))
        self.assertEqual(events, [])

    def test_debounce_requires_condition_to_persist(self) -> None:
        memory = ObjectMemory()
        graph = SceneGraph(zones=_restricted_zone())
        detector = ZoneViolationDetector(
            monitored_classes=["person"],
            debounce_seconds=0.5,
            cooldown_seconds=0.0,
        )
        object_at_zone = (_tracked(3, "person", 40.0, 40.0, 0.0, 0),)

        first = detector.detect(_context(memory, graph, object_at_zone, 0.0))
        brief = detector.detect(_context(memory, graph, object_at_zone, 0.25))
        persisted = detector.detect(_context(memory, graph, object_at_zone, 0.5))

        self.assertEqual(first, [])
        self.assertEqual(brief, [])
        self.assertEqual([event.type for event in persisted], [ZONE_VIOLATION])

    def test_zone_violation_and_cooldown_emit_once_per_incident(self) -> None:
        memory = ObjectMemory()
        graph = SceneGraph(zones=_restricted_zone())
        detector = ZoneViolationDetector(
            monitored_classes=["person"],
            debounce_seconds=0.0,
            cooldown_seconds=3.0,
        )
        object_at_zone = (_tracked(3, "person", 40.0, 40.0, 0.0, 0),)

        first = detector.detect(_context(memory, graph, object_at_zone, 0.0))
        still_inside = detector.detect(_context(memory, graph, object_at_zone, 1.0))
        self.assertEqual([event.type for event in first], [ZONE_VIOLATION])
        self.assertEqual(still_inside, [])
        self.assertEqual(first[0].entities, ("person_3", "restricted_zone"))
        self.assertEqual(first[0].to_dict()["type"], ZONE_VIOLATION)

        detector.detect(_context(memory, graph, (), 2.0))
        suppressed_reentry = detector.detect(
            _context(memory, graph, object_at_zone, 2.5)
        )
        emitted_after_cooldown = detector.detect(
            _context(memory, graph, object_at_zone, 3.0)
        )
        self.assertEqual(suppressed_reentry, [])
        self.assertEqual([event.type for event in emitted_after_cooldown], [ZONE_VIOLATION])

    def test_dragging_uses_near_floor_horizontal_displacement_and_person(self) -> None:
        memory = ObjectMemory()
        graph = SceneGraph(thresholds=RelationThresholds(near_distance_px=80.0))
        detector = DraggingDetector(
            min_duration_seconds=2.0,
            min_horizontal_displacement_px=40.0,
            near_floor_fraction=0.85,
            require_person_nearby=True,
            cooldown_seconds=3.0,
        )
        frames = (
            (_tracked(7, "carton", 20.0, 85.0, 0.0, 0), _tracked(3, "person", 40.0, 80.0, 0.0, 0)),
            (_tracked(7, "carton", 50.0, 85.0, 1.0, 1), _tracked(3, "person", 70.0, 80.0, 1.0, 1)),
            (_tracked(7, "carton", 80.0, 85.0, 2.0, 2), _tracked(3, "person", 100.0, 80.0, 2.0, 2)),
        )

        self.assertEqual(detector.detect(_context(memory, graph, frames[0], 0.0)), [])
        self.assertEqual(detector.detect(_context(memory, graph, frames[1], 1.0)), [])
        events = detector.detect(_context(memory, graph, frames[2], 2.0))
        duplicate = detector.detect(_context(memory, graph, frames[2], 2.5))

        self.assertEqual([event.type for event in events], [POSSIBLE_DRAGGING])
        self.assertEqual(duplicate, [])
        self.assertEqual(events[0].evidence["near_floor"], True)
        self.assertEqual(events[0].evidence["person_nearby"], True)
        self.assertAlmostEqual(events[0].evidence["horizontal_displacement_px"], 60.0)

    def test_possible_drop_requires_move_separation_downward_motion_and_settle(self) -> None:
        memory = ObjectMemory(stationary_speed_threshold=1.0)
        graph = SceneGraph(
            memory=memory,
            thresholds=RelationThresholds(near_distance_px=50.0),
        )
        detector = PossibleDropDetector(
            min_pre_motion_speed_px_per_second=5.0,
            min_downward_speed_px_per_second=20.0,
            min_downward_displacement_px=20.0,
            max_settled_speed_px_per_second=5.0,
            min_deceleration_px_per_second=10.0,
            require_separation=True,
        )
        frames = (
            (_tracked(7, "carton", 30.0, 30.0, 0.0, 0), _tracked(3, "person", 60.0, 30.0, 0.0, 0)),
            (_tracked(7, "carton", 40.0, 30.0, 1.0, 1), _tracked(3, "person", 70.0, 30.0, 1.0, 1)),
            (_tracked(7, "carton", 50.0, 70.0, 2.0, 2), _tracked(3, "person", 150.0, 30.0, 2.0, 2)),
            (_tracked(7, "carton", 50.0, 71.0, 3.0, 3), _tracked(3, "person", 150.0, 30.0, 3.0, 3)),
        )
        for timestamp, objects in enumerate(frames):
            events = detector.detect(_context(memory, graph, objects, float(timestamp)))
            if timestamp < 3:
                self.assertEqual(events, [])

        self.assertEqual([event.type for event in events], [POSSIBLE_DROP])
        self.assertEqual(events[0].evidence["separated_from_previous_relation"], True)
        self.assertEqual(events[0].evidence["rapid_downward_motion"], True)
        self.assertEqual(events[0].evidence["abrupt_deceleration"], True)

    def test_overhang_uses_low_support_ratio_over_pallet(self) -> None:
        memory = ObjectMemory()
        graph = SceneGraph()
        detector = OverhangDetector(
            min_support_ratio=0.70,
            max_vertical_gap_px=2.0,
            debounce_seconds=0.0,
        )
        objects = (
            _tracked(7, "carton", 50.0, 10.0, 0.0, 0, width=100.0, height=20.0),
            _tracked(2, "pallet", 50.0, 40.0, 0.0, 0, width=50.0, height=40.0),
        )

        events = detector.detect(_context(memory, graph, objects, 0.0))

        self.assertEqual([event.type for event in events], [PALLET_OVERHANG])
        self.assertAlmostEqual(events[0].evidence["support_ratio"], 0.5)
        self.assertAlmostEqual(events[0].evidence["left_overhang_px"], 25.0)

    def test_unstable_stack_uses_adjacent_support_ratios(self) -> None:
        memory = ObjectMemory()
        graph = SceneGraph()
        detector = UnstableStackDetector(
            min_stack_objects=3,
            min_stable_support_ratio=0.75,
            max_vertical_gap_px=2.0,
            debounce_seconds=0.0,
        )
        objects = (
            _tracked(1, "carton", 80.0, 10.0, 0.0, 0, width=30.0, height=20.0),
            _tracked(2, "carton", 60.0, 30.0, 0.0, 0, width=40.0, height=20.0),
            _tracked(3, "carton", 50.0, 55.0, 0.0, 0, width=100.0, height=30.0),
        )

        events = detector.detect(_context(memory, graph, objects, 0.0))

        self.assertEqual([event.type for event in events], [UNSTABLE_STACK])
        self.assertEqual(events[0].entities, ("carton_1", "carton_2", "carton_3"))
        self.assertAlmostEqual(events[0].evidence["minimum_support_ratio"], 0.5)

    def test_human_forklift_proximity_requires_distance_and_motion(self) -> None:
        memory = ObjectMemory(stationary_speed_threshold=1.0)
        graph = SceneGraph(
            memory=memory,
            thresholds=RelationThresholds(near_distance_px=100.0),
        )
        detector = HumanForkliftProximityDetector(
            max_distance_px=100.0,
            min_motion_speed_px_per_second=2.0,
            debounce_seconds=0.0,
        )
        first = (
            _tracked(3, "person", 20.0, 50.0, 0.0, 0),
            _tracked(1, "forklift", 150.0, 45.0, 0.0, 0, width=30.0, height=30.0),
        )
        second = (
            _tracked(3, "person", 30.0, 50.0, 1.0, 1),
            _tracked(1, "forklift", 120.0, 45.0, 1.0, 1, width=30.0, height=30.0),
        )
        detector.detect(_context(memory, graph, first, 0.0))
        events = detector.detect(_context(memory, graph, second, 1.0))

        self.assertEqual([event.type for event in events], [UNSAFE_HUMAN_FORKLIFT_PROXIMITY])
        self.assertEqual(events[0].evidence["approaching"], True)
        self.assertAlmostEqual(events[0].evidence["distance_px"], 90.0)

    def test_default_registry_preserves_requested_order(self) -> None:
        registry = build_default_registry()

        self.assertEqual(
            registry.event_types,
            (
                ZONE_VIOLATION,
                "ZONE_TRANSITION",
                "OBJECT_ACTIVITY",
                "MOTION_ANOMALY",
                POSSIBLE_THROWING,
                POSSIBLE_ROUGH_HANDLING,
                AISLE_OBSTRUCTION,
                IMPROPER_PLACEMENT,
                COLLISION_RISK,
                POSSIBLE_DRAGGING,
                POSSIBLE_DROP,
                PALLET_OVERHANG,
                UNSTABLE_STACK,
                UNSAFE_HUMAN_FORKLIFT_PROXIMITY,
            ),
        )

    def test_throwing_uses_high_speed_and_acceleration(self) -> None:
        memory = ObjectMemory()
        graph = SceneGraph()
        detector = ThrowingDetector(
            min_release_speed_px_per_second=100.0,
            min_acceleration_px_per_second_squared=100.0,
            cooldown_seconds=0.0,
        )
        frames = (
            (_tracked(7, "carton", 0.0, 40.0, 0.0, 0),),
            (_tracked(7, "carton", 10.0, 40.0, 1.0, 1),),
            (_tracked(7, "carton", 20.0, 40.0, 2.0, 2),),
            (_tracked(7, "carton", 300.0, 40.0, 3.0, 3),),
        )
        events = []
        for timestamp, objects in enumerate(frames):
            events = detector.detect(_context(memory, graph, objects, float(timestamp)))
        self.assertEqual([event.type for event in events], [POSSIBLE_THROWING])
        self.assertGreater(events[0].evidence["speed_px_per_second"], 100.0)

    def test_rough_handling_uses_acceleration_or_direction_change(self) -> None:
        memory = ObjectMemory()
        graph = SceneGraph()
        detector = RoughHandlingDetector(
            min_speed_px_per_second=5.0,
            min_acceleration_px_per_second_squared=20.0,
            direction_change_degrees=100.0,
            cooldown_seconds=0.0,
        )
        frames = (
            (_tracked(7, "carton", 0.0, 40.0, 0.0, 0),),
            (_tracked(7, "carton", 10.0, 40.0, 1.0, 1),),
            (_tracked(7, "carton", 20.0, 40.0, 2.0, 2),),
            (_tracked(7, "carton", 0.0, 40.0, 3.0, 3),),
        )
        for timestamp, objects in enumerate(frames):
            events = detector.detect(_context(memory, graph, objects, float(timestamp)))
        self.assertEqual([event.type for event in events], [POSSIBLE_ROUGH_HANDLING])

    def test_aisle_obstruction_requires_configured_zone_and_stationary_time(self) -> None:
        memory = ObjectMemory()
        graph = SceneGraph(zones=_restricted_zone())
        detector = AisleObstructionDetector(
            aisle_zones=["restricted_zone"],
            monitored_classes=["carton"],
            min_stationary_seconds=3.0,
            debounce_seconds=0.0,
            cooldown_seconds=0.0,
        )
        objects = (_tracked(7, "carton", 50.0, 50.0, 0.0, 0),)
        detector.detect(_context(memory, graph, objects, 0.0))
        detector.detect(_context(memory, graph, objects, 2.0))
        events = detector.detect(_context(memory, graph, objects, 4.0))
        self.assertEqual([event.type for event in events], [AISLE_OBSTRUCTION])

    def test_improper_placement_requires_settling_outside_allowed_zones(self) -> None:
        memory = ObjectMemory()
        graph = SceneGraph(zones=_restricted_zone())
        detector = ImproperPlacementDetector(
            allowed_zones=["staging_zone"],
            monitored_classes=["carton"],
            min_stationary_seconds=2.0,
            debounce_seconds=0.0,
            cooldown_seconds=0.0,
        )
        objects = (_tracked(7, "carton", 150.0, 50.0, 0.0, 0),)
        detector.detect(_context(memory, graph, objects, 0.0))
        events = detector.detect(_context(memory, graph, objects, 2.0))
        self.assertEqual([event.type for event in events], [IMPROPER_PLACEMENT])

    def test_collision_risk_uses_closing_distance(self) -> None:
        memory = ObjectMemory()
        graph = SceneGraph()
        detector = CollisionRiskDetector(
            warning_distance_px=100.0,
            minimum_closing_speed_px_per_second=5.0,
            debounce_seconds=0.0,
            cooldown_seconds=0.0,
        )
        first = (
            _tracked(3, "person", 0.0, 40.0, 0.0, 0),
            _tracked(1, "forklift", 150.0, 40.0, 0.0, 0),
        )
        second = (
            _tracked(3, "person", 30.0, 40.0, 1.0, 1),
            _tracked(1, "forklift", 100.0, 40.0, 1.0, 1),
        )
        detector.detect(_context(memory, graph, first, 0.0))
        events = detector.detect(_context(memory, graph, second, 1.0))
        self.assertEqual([event.type for event in events], [COLLISION_RISK])
        self.assertGreater(events[0].evidence["closing_speed_px_per_second"], 5.0)
