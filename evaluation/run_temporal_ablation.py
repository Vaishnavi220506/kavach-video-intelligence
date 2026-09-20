"""Show the difference between frame-local detections and temporal evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kavach.behaviours import BehaviourContext, build_default_registry
from kavach.intelligence import ObjectMemory, SceneGraph, ZoneManager
from kavach.perception import TrackedObject


def run() -> dict[str, object]:
    # The same carton is visible in every frame and moves horizontally near
    # the image floor. The frame-level view can report presence and boxes, but
    # only the temporal layer can accumulate duration and displacement.
    states = [
        (0.0, [10, 800, 40, 900]),
        (0.5, [60, 800, 90, 900]),
        (1.0, [110, 800, 140, 900]),
        (1.5, [160, 800, 190, 900]),
    ]
    memory = ObjectMemory(max_history_seconds=5.0)
    graph = SceneGraph(memory=memory, zones=ZoneManager(), max_snapshots=20)
    registry = build_default_registry(
        config={
            "zone_violation": {"enabled": False},
            "dragging": {
                "enabled": True,
                "draggable_classes": ["carton"],
                "min_duration_seconds": 1.5,
                "history_window_seconds": 5.0,
                "min_horizontal_displacement_px": 40.0,
                "max_vertical_displacement_px": 35.0,
                "near_floor_fraction": 0.85,
                "debounce_seconds": 0.0,
                "cooldown_seconds": 3.0,
            },
            "drop": {"enabled": False},
            "overhang": {"enabled": False},
            "stacking": {"enabled": False},
            "proximity": {"enabled": False},
        }
    )
    temporal_events = []
    for frame_number, (timestamp, bbox) in enumerate(states):
        x1, y1, x2, y2 = bbox
        tracked = TrackedObject(
            track_id=7,
            class_name="carton",
            confidence=0.9,
            bbox=tuple(bbox),
            center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            timestamp=timestamp,
            frame_number=frame_number,
        )
        objects = (tracked,)
        memory.update(objects, timestamp, frame_number=frame_number)
        graph.update(objects, timestamp=timestamp, memory=memory)
        context = BehaviourContext(
            timestamp=timestamp,
            memory=memory,
            scene_graph=graph,
            frame_width=200,
            frame_height=1000,
        )
        temporal_events.extend(event.to_dict() for event in registry.detect(context))
    return {
        "frame_level_observation": {
            "per_frame": [
                {"frame": index, "detected_class": "carton", "bbox": bbox}
                for index, (_, bbox) in enumerate(states)
            ],
            "conclusion": "presence and box coordinates only; no dragging event",
        },
        "temporal_kavach_observation": {
            "events": temporal_events,
            "conclusion": "possible dragging is supported after timestamped displacement and duration accumulate",
        },
    }


def main() -> None:
    output = Path("reports/temporal_ablation.json")
    result = run()
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
