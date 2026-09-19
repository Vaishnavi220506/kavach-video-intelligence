"""Run the small synthetic behaviour evaluation without claiming video accuracy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kavach.behaviours import BehaviourContext, build_default_registry
from kavach.intelligence import (
    ObjectMemory,
    RESTRICTED_ZONE,
    SceneGraph,
    ZoneManager,
)
from kavach.perception import TrackedObject


FRAME_WIDTH = 200
FRAME_HEIGHT = 1000


def _object(track_id: int, class_name: str, bbox: list[float], timestamp: float, frame: int) -> TrackedObject:
    x1, y1, x2, y2 = bbox
    return TrackedObject(
        track_id=track_id,
        class_name=class_name,
        confidence=0.90,
        bbox=(x1, y1, x2, y2),
        center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        timestamp=timestamp,
        frame_number=frame,
    )


def _zones() -> ZoneManager:
    return ZoneManager(
        {
            RESTRICTED_ZONE: [[0, 0], [200, 0], [200, 800], [0, 800]],
            "aisle_zone": [[0, 0], [200, 0], [200, 800], [0, 800]],
            "staging_zone": [[0, 800], [200, 800], [200, 1000], [0, 1000]],
        }
    )


def _scenarios() -> dict[str, list[list[dict[str, Any]]]]:
    return {
        "zone_inside": [
            [{"id": 1, "class": "person", "bbox": [80, 100, 120, 200]}],
            [{"id": 1, "class": "person", "bbox": [80, 100, 120, 200]}],
        ],
        "zone_clear": [
            [{"id": 1, "class": "person", "bbox": [80, 850, 120, 950]}],
            [{"id": 1, "class": "person", "bbox": [80, 850, 120, 950]}],
        ],
        "dragging": [
            [{"id": 7, "class": "carton", "bbox": [10, 800, 40, 900]}],
            [{"id": 7, "class": "carton", "bbox": [60, 800, 90, 900]}],
            [{"id": 7, "class": "carton", "bbox": [110, 800, 140, 900]}],
            [{"id": 7, "class": "carton", "bbox": [160, 800, 190, 900]}],
        ],
        "dragging_vertical": [
            [{"id": 7, "class": "carton", "bbox": [10, 700, 40, 800]}],
            [{"id": 7, "class": "carton", "bbox": [60, 780, 90, 880]}],
            [{"id": 7, "class": "carton", "bbox": [110, 860, 140, 960]}],
        ],
        "drop": [
            [
                {"id": 7, "class": "carton", "bbox": [35, 680, 65, 720]},
                {"id": 2, "class": "pallet", "bbox": [20, 720, 80, 760]},
            ],
            [
                {"id": 7, "class": "carton", "bbox": [55, 680, 85, 720]},
                {"id": 2, "class": "pallet", "bbox": [20, 720, 80, 760]},
            ],
            [{"id": 7, "class": "carton", "bbox": [55, 710, 85, 750]}],
            [{"id": 7, "class": "carton", "bbox": [55, 710, 85, 750]}],
        ],
        "drop_smooth": [
            [{"id": 7, "class": "carton", "bbox": [35, 680, 65, 720]}],
            [{"id": 7, "class": "carton", "bbox": [55, 680, 85, 720]}],
            [{"id": 7, "class": "carton", "bbox": [75, 680, 105, 720]}],
            [{"id": 7, "class": "carton", "bbox": [95, 680, 125, 720]}],
        ],
        "overhang": [[
            {"id": 7, "class": "carton", "bbox": [0, 100, 100, 200]},
            {"id": 2, "class": "pallet", "bbox": [20, 200, 80, 240]},
        ]],
        "supported": [[
            {"id": 7, "class": "carton", "bbox": [20, 100, 80, 200]},
            {"id": 2, "class": "pallet", "bbox": [20, 200, 80, 240]},
        ]],
        "unstable_stack": [[
            {"id": 1, "class": "carton", "bbox": [0, 0, 100, 40]},
            {"id": 2, "class": "carton", "bbox": [30, 40, 70, 80]},
            {"id": 3, "class": "carton", "bbox": [30, 80, 70, 120]},
        ], [
            {"id": 1, "class": "carton", "bbox": [0, 0, 100, 40]},
            {"id": 2, "class": "carton", "bbox": [30, 40, 70, 80]},
            {"id": 3, "class": "carton", "bbox": [30, 80, 70, 120]},
        ]],
        "stable_stack": [[
            {"id": 1, "class": "carton", "bbox": [20, 0, 80, 40]},
            {"id": 2, "class": "carton", "bbox": [20, 40, 80, 80]},
            {"id": 3, "class": "carton", "bbox": [20, 80, 80, 120]},
        ], [
            {"id": 1, "class": "carton", "bbox": [20, 0, 80, 40]},
            {"id": 2, "class": "carton", "bbox": [20, 40, 80, 80]},
            {"id": 3, "class": "carton", "bbox": [20, 80, 80, 120]},
        ]],
        "proximity": [
            [
                {"id": 1, "class": "person", "bbox": [20, 40, 40, 90]},
                {"id": 2, "class": "forklift", "bbox": [70, 40, 100, 100]},
            ],
            [
                {"id": 1, "class": "person", "bbox": [40, 40, 60, 90]},
                {"id": 2, "class": "forklift", "bbox": [70, 40, 100, 100]},
            ],
        ],
        "proximity_far": [
            [
                {"id": 1, "class": "person", "bbox": [0, 40, 20, 90]},
                {"id": 2, "class": "forklift", "bbox": [150, 40, 180, 100]},
            ],
            [
                {"id": 1, "class": "person", "bbox": [0, 40, 20, 90]},
                {"id": 2, "class": "forklift", "bbox": [150, 40, 180, 100]},
            ],
        ],
        "throwing": [
            [{"id": 7, "class": "carton", "bbox": [10, 450, 40, 500]}],
            [{"id": 7, "class": "carton", "bbox": [15, 450, 45, 500]}],
            [{"id": 7, "class": "carton", "bbox": [180, 450, 210, 500]}],
        ],
        "throwing_smooth": [
            [{"id": 7, "class": "carton", "bbox": [10, 450, 40, 500]}],
            [{"id": 7, "class": "carton", "bbox": [30, 450, 60, 500]}],
            [{"id": 7, "class": "carton", "bbox": [50, 450, 80, 500]}],
            [{"id": 7, "class": "carton", "bbox": [70, 450, 100, 500]}],
        ],
        "rough_handling": [
            [{"id": 7, "class": "carton", "bbox": [10, 450, 40, 500]}],
            [{"id": 7, "class": "carton", "bbox": [20, 450, 50, 500]}],
            [{"id": 7, "class": "carton", "bbox": [100, 450, 130, 500]}],
        ],
        "rough_handling_smooth": [
            [{"id": 7, "class": "carton", "bbox": [10, 450, 40, 500]}],
            [{"id": 7, "class": "carton", "bbox": [30, 450, 60, 500]}],
            [{"id": 7, "class": "carton", "bbox": [50, 450, 80, 500]}],
            [{"id": 7, "class": "carton", "bbox": [70, 450, 100, 500]}],
        ],
        "aisle_obstruction": [
            [{"id": 7, "class": "pallet", "bbox": [80, 300, 120, 400]}],
            [{"id": 7, "class": "pallet", "bbox": [80, 300, 120, 400]}],
            [{"id": 7, "class": "pallet", "bbox": [80, 300, 120, 400]}],
            [{"id": 7, "class": "pallet", "bbox": [80, 300, 120, 400]}],
            [{"id": 7, "class": "pallet", "bbox": [80, 300, 120, 400]}],
            [{"id": 7, "class": "pallet", "bbox": [80, 300, 120, 400]}],
            [{"id": 7, "class": "pallet", "bbox": [80, 300, 120, 400]}],
        ],
        "aisle_clear": [
            [{"id": 7, "class": "pallet", "bbox": [10, 300, 50, 400]}],
            [{"id": 7, "class": "pallet", "bbox": [30, 300, 70, 400]}],
            [{"id": 7, "class": "pallet", "bbox": [50, 300, 90, 400]}],
            [{"id": 7, "class": "pallet", "bbox": [70, 300, 110, 400]}],
            [{"id": 7, "class": "pallet", "bbox": [90, 300, 130, 400]}],
            [{"id": 7, "class": "pallet", "bbox": [110, 300, 150, 400]}],
            [{"id": 7, "class": "pallet", "bbox": [130, 300, 170, 400]}],
        ],
        "improper_placement": [
            [{"id": 7, "class": "carton", "bbox": [80, 400, 120, 500]}],
            [{"id": 7, "class": "carton", "bbox": [80, 400, 120, 500]}],
            [{"id": 7, "class": "carton", "bbox": [80, 400, 120, 500]}],
            [{"id": 7, "class": "carton", "bbox": [80, 400, 120, 500]}],
            [{"id": 7, "class": "carton", "bbox": [80, 400, 120, 500]}],
        ],
        "proper_placement": [
            [{"id": 7, "class": "carton", "bbox": [80, 850, 120, 950]}],
            [{"id": 7, "class": "carton", "bbox": [80, 850, 120, 950]}],
            [{"id": 7, "class": "carton", "bbox": [80, 850, 120, 950]}],
            [{"id": 7, "class": "carton", "bbox": [80, 850, 120, 950]}],
            [{"id": 7, "class": "carton", "bbox": [80, 850, 120, 950]}],
        ],
        "collision_risk": [
            [
                {"id": 1, "class": "person", "bbox": [0, 40, 20, 90]},
                {"id": 2, "class": "forklift", "bbox": [140, 40, 170, 100]},
            ],
            [
                {"id": 1, "class": "person", "bbox": [30, 40, 50, 90]},
                {"id": 2, "class": "forklift", "bbox": [120, 40, 150, 100]},
            ],
        ],
        "collision_risk_far": [
            [
                {"id": 1, "class": "person", "bbox": [0, 40, 20, 90]},
                {"id": 2, "class": "forklift", "bbox": [150, 40, 180, 100]},
            ],
            [
                {"id": 1, "class": "person", "bbox": [0, 40, 20, 90]},
                {"id": 2, "class": "forklift", "bbox": [150, 40, 180, 100]},
            ],
        ],
    }


def _config_for(behaviour: str) -> dict[str, dict[str, object]]:
    sections = (
        "zone_violation", "zone_transition", "object_activity", "motion_anomaly",
        "throwing", "rough_handling", "aisle_obstruction", "improper_placement",
        "collision_risk", "dragging", "drop", "overhang", "stacking", "proximity",
    )
    config = {section: {"enabled": False} for section in sections}
    if behaviour == "ZONE_VIOLATION":
        config["zone_violation"] = {
            "enabled": True,
            "monitored_classes": ["person"],
            "debounce_seconds": 0.25,
            "cooldown_seconds": 3.0,
        }
    elif behaviour == "POSSIBLE_DRAGGING":
        config["dragging"] = {
            "enabled": True,
            "draggable_classes": ["carton"],
            "min_duration_seconds": 1.5,
            "history_window_seconds": 5.0,
            "min_horizontal_displacement_px": 40.0,
            "max_vertical_displacement_px": 35.0,
            "near_floor_fraction": 0.85,
            "require_person_nearby": False,
            "debounce_seconds": 0.0,
            "cooldown_seconds": 3.0,
        }
    elif behaviour == "POSSIBLE_DROP":
        config["drop"] = {
            "enabled": True,
            "drop_classes": ["carton"],
            "min_pre_motion_speed_px_per_second": 20.0,
            "min_downward_speed_px_per_second": 35.0,
            "min_downward_displacement_px": 12.0,
            "max_settled_speed_px_per_second": 5.0,
            "min_deceleration_px_per_second": 15.0,
            "max_pattern_seconds": 4.0,
            "near_floor_fraction": 0.85,
            "require_separation": True,
            "debounce_seconds": 0.0,
            "cooldown_seconds": 3.0,
        }
    elif behaviour == "PALLET_OVERHANG":
        config["overhang"] = {
            "enabled": True,
            "support_provider_classes": ["pallet"],
            "min_support_ratio": 0.70,
            "max_vertical_gap_px": 20.0,
            "debounce_seconds": 0.0,
            "cooldown_seconds": 3.0,
        }
    elif behaviour == "UNSTABLE_STACK":
        config["stacking"] = {
            "enabled": True,
            "stackable_classes": ["carton"],
            "min_stack_objects": 3,
            "min_stable_support_ratio": 0.75,
            "max_vertical_gap_px": 20.0,
            "debounce_seconds": 0.5,
            "cooldown_seconds": 3.0,
        }
    elif behaviour == "UNSAFE_HUMAN_FORKLIFT_PROXIMITY":
        config["proximity"] = {
            "enabled": True,
            "human_classes": ["person"],
            "forklift_classes": ["forklift"],
            "max_distance_px": 100.0,
            "min_motion_speed_px_per_second": 2.0,
            "require_motion": True,
            "debounce_seconds": 0.0,
            "cooldown_seconds": 3.0,
        }
    elif behaviour == "POSSIBLE_THROWING":
        config["throwing"] = {
            "enabled": True,
            "throw_classes": ["carton"],
            "min_release_speed_px_per_second": 220.0,
            "min_acceleration_px_per_second_squared": 120.0,
            "debounce_seconds": 0.0,
            "cooldown_seconds": 3.0,
        }
    elif behaviour == "POSSIBLE_ROUGH_HANDLING":
        config["rough_handling"] = {
            "enabled": True,
            "monitored_classes": ["carton"],
            "min_speed_px_per_second": 45.0,
            "min_acceleration_px_per_second_squared": 180.0,
            "direction_change_degrees": 100.0,
            "debounce_seconds": 0.0,
            "cooldown_seconds": 3.0,
        }
    elif behaviour == "AISLE_OBSTRUCTION":
        config["aisle_obstruction"] = {
            "enabled": True,
            "aisle_zones": ["aisle_zone"],
            "monitored_classes": ["pallet"],
            "min_stationary_seconds": 3.0,
            "debounce_seconds": 0.0,
            "cooldown_seconds": 3.0,
        }
    elif behaviour == "IMPROPER_PLACEMENT":
        config["improper_placement"] = {
            "enabled": True,
            "allowed_zones": ["staging_zone"],
            "monitored_classes": ["carton"],
            "min_stationary_seconds": 2.0,
            "debounce_seconds": 0.0,
            "cooldown_seconds": 3.0,
        }
    elif behaviour == "COLLISION_RISK":
        config["collision_risk"] = {
            "enabled": True,
            "human_classes": ["person"],
            "equipment_classes": ["forklift"],
            "warning_distance_px": 100.0,
            "minimum_closing_speed_px_per_second": 8.0,
            "debounce_seconds": 0.0,
            "cooldown_seconds": 3.0,
        }
    return config


def _predicted_behaviours(case: dict[str, object], scenarios: dict[str, list[list[dict[str, Any]]]]) -> set[str]:
    behaviour = str(case["behaviour"])
    memory = ObjectMemory(max_history_seconds=10.0, max_states_per_track=100)
    zones = _zones()
    graph = SceneGraph(memory=memory, zones=zones, max_snapshots=20)
    registry = build_default_registry(config=_config_for(behaviour))
    predicted: set[str] = set()
    for frame_number, raw_objects in enumerate(scenarios[str(case["scenario"])]):
        timestamp = frame_number * 0.5
        objects = tuple(
            _object(
                int(item["id"]),
                str(item["class"]),
                list(item["bbox"]),
                timestamp,
                frame_number,
            )
            for item in raw_objects
        )
        memory.update(objects, timestamp, frame_number=frame_number)
        graph.update(objects, timestamp=timestamp, memory=memory)
        context = BehaviourContext(
            timestamp=timestamp,
            memory=memory,
            scene_graph=graph,
            frame_width=FRAME_WIDTH,
            frame_height=FRAME_HEIGHT,
        )
        predicted.update(event.type for event in registry.detect(context))
    return predicted


def _metric(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "precision": precision, "recall": recall, "F1": f1}


def run(dataset_path: str | Path) -> dict[str, object]:
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    scenarios = _scenarios()
    results = []
    by_behaviour: dict[str, dict[str, int]] = {}
    for case in dataset["cases"]:
        expected = set(case["expected"])
        predicted = _predicted_behaviours(case, scenarios)
        behaviour = str(case["behaviour"])
        bucket = by_behaviour.setdefault(behaviour, {"TP": 0, "FP": 0, "FN": 0})
        if behaviour in expected and behaviour in predicted:
            bucket["TP"] += 1
        elif behaviour in expected and behaviour not in predicted:
            bucket["FN"] += 1
        elif behaviour not in expected and behaviour in predicted:
            bucket["FP"] += 1
        results.append({
            "id": case["id"],
            "expected": sorted(expected),
            "predicted": sorted(predicted),
            "match": expected == predicted,
        })
    metrics = {
        name: _metric(values["TP"], values["FP"], values["FN"])
        for name, values in by_behaviour.items()
    }
    return {
        "dataset": dataset["name"],
        "data_type": dataset["data_type"],
        "case_count": len(results),
        "case_results": results,
        "metrics_by_behaviour": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=Path(__file__).with_name("controlled_dataset.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.dataset)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
