"""Generate the committed demo dataset used by the public KAVACH website.

The published site has no backend: Vercel serves a static bundle, so the React
app cannot call the FastAPI service. This script produces the fixtures the
static build reads instead.

Nothing here fabricates results. A scripted set of tracked-object trajectories
is fed through the same engines the live pipeline uses -- ``ObjectMemory``,
``SceneGraph``, ``BehaviourRegistry``, ``RiskEngine``, ``IncidentManager``,
``PreventionEngine`` and ``EventDatabase`` -- with the repository's own
``kavach/behaviours/config.yaml`` thresholds and the dashboard's default zone
polygons. Every behaviour, risk score, component breakdown, evidence record and
statistic in the output is real engine output.

What *is* synthetic is the input: these are structured trajectories in the style
of ``evaluation/controlled_dataset.json``, not detections from camera footage.
The website labels the dataset as a synthetic reconstruction wherever it is
displayed. Detection is deliberately bypassed because YOLO cannot be run against
frames that do not exist; the perception layer's accuracy is reported separately
in ``docs/MODEL_UPGRADE.md`` and must not be inferred from this dataset.

Usage::

    python scripts/build_demo_dataset.py
    python scripts/build_demo_dataset.py --output frontend/public/demo
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kavach.behaviours import BehaviourContext, build_default_registry
from kavach.dashboard.analysis import build_default_zones
from kavach.incidents import IncidentManager
from kavach.intelligence import ObjectMemory, SceneGraph
from kavach.perception import TrackedObject
from kavach.presentation import (
    ANOMALY_SCOPE_NOTE,
    MODEL_SCOPE_NOTE,
    SUPPORTED_BEHAVIOURS,
    event_payload,
    evidence_graph,
)
from kavach.prevention import PreventionEngine
from kavach.risk import RiskEngine
from kavach.storage import EventDatabase

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
SOURCE_FPS = 25.0
STEP_SECONDS = 0.2
VIDEO_ID = "demo-warehouse-reconstruction"
SOURCE_NAME = "synthetic_reconstruction.mp4"

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Segment:
    """One entity's linear motion between two keyframes.

    ``start`` and ``end`` are inclusive image-space boxes. A track is absent
    from a frame whenever no segment covers that timestamp, which is how the
    scenario expresses occlusion and separation from a support surface.
    """

    track_id: int
    class_name: str
    start_time: float
    end_time: float
    start_box: Box
    end_box: Box

    def box_at(self, timestamp: float) -> Box | None:
        if timestamp < self.start_time - 1e-9 or timestamp > self.end_time + 1e-9:
            return None
        span = self.end_time - self.start_time
        ratio = 0.0 if span <= 0 else (timestamp - self.start_time) / span
        ratio = min(1.0, max(0.0, ratio))
        return tuple(  # type: ignore[return-value]
            start + (end - start) * ratio
            for start, end in zip(self.start_box, self.end_box, strict=False)
        )


def hold(track_id: int, class_name: str, start: float, end: float, box: Box) -> Segment:
    """A stationary entity, used for placement, stacking and support cases."""

    return Segment(track_id, class_name, start, end, box, box)


def move(
    track_id: int,
    class_name: str,
    start: float,
    end: float,
    start_box: Box,
    end_box: Box,
) -> Segment:
    return Segment(track_id, class_name, start, end, start_box, end_box)


def shift(box: Box, dx: float, dy: float) -> Box:
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


# --- The scripted shift excerpt -------------------------------------------
#
# Zone polygons come from ``build_default_zones`` at 1920x1080:
#   STAGING_ZONE     x 38-614     y 691-1058
#   LOADING_ZONE     x 1306-1882  y 691-1058
#   RESTRICTED_ZONE  x 576-1344   y 367-1015
#   PALLET_ZONE      x 691-1229   y 821-1058
#
# Thresholds referenced below are the committed values in
# kavach/behaviours/config.yaml. Each block is written to exercise one rule
# with clear margin, so the dataset stays reproducible when a threshold moves
# slightly. Timings leave gaps between blocks so per-rule cooldowns expire.

WORKER_A = 1
WORKER_B = 2
FORKLIFT = 3
PALLET = 4
CARTON_A = 7
CARTON_B = 8
CARTON_C = 9
CARTON_D = 10


def build_segments() -> list[Segment]:
    segments: list[Segment] = []

    # 0.0-10.0s  Worker A walks the staging aisle. Establishes track age and a
    # motion baseline so later anomaly scoring has history to compare against.
    segments.append(move(WORKER_A, "person", 0.0, 10.0, (120, 700, 200, 900), (520, 700, 600, 900)))

    # 6.0-13.0s  Carton dragged along the floor: >40px horizontal, <35px
    # vertical, >2.0s, bottom below the 918px near-floor line.
    segments.append(move(CARTON_A, "carton", 6.0, 13.0, (240, 900, 320, 980), (660, 906, 740, 986)))

    # 14.0-24.0s  Forklift crosses while Worker B closes on it. Drives both
    # UNSAFE_HUMAN_FORKLIFT_PROXIMITY (<100px, moving) and COLLISION_RISK
    # (<140px with >8px/s closing speed). Both rules measure the bottom-centre
    # floor point, so the pair converges to roughly 60px of separation there
    # and then keeps drifting slowly, which satisfies the 0.25s debounce and
    # the proximity rule's require_motion condition.
    segments.append(move(FORKLIFT, "forklift", 14.0, 22.0, (1500, 640, 1740, 860), (1000, 640, 1240, 860)))
    segments.append(move(FORKLIFT, "forklift", 22.2, 24.0, (1000, 640, 1240, 860), (960, 640, 1200, 860)))
    segments.append(move(WORKER_B, "person", 14.0, 22.0, (640, 660, 720, 860), (1020, 660, 1100, 860)))
    segments.append(move(WORKER_B, "person", 22.2, 24.0, (1020, 660, 1100, 860), (1000, 660, 1080, 860)))

    # 26.0-34.0s  Worker A steps inside RESTRICTED_ZONE and stays past the
    # 0.25s debounce.
    segments.append(move(WORKER_A, "person", 26.0, 34.0, (700, 500, 780, 700), (1000, 520, 1080, 720)))

    # 36.0-40.0s  Drop: the carton rides the pallet, moves horizontally, then
    # falls and settles. The rule reads exactly four consecutive states --
    # pre-motion, fall, settle -- so the landing frames must be perfectly
    # still, or the settled-speed test fails. The pallet is absent from the
    # falling frames, which is how `require_separation` is met.
    segments.append(hold(PALLET, "pallet", 36.0, 36.8, (950, 900, 1180, 980)))
    segments.append(move(CARTON_B, "carton", 36.0, 36.8, (980, 820, 1080, 900), (1060, 820, 1160, 900)))
    segments.append(hold(CARTON_B, "carton", 37.0, 40.0, (1062, 900, 1162, 980)))

    # 42.0-45.0s  Throwing: release speed above 220px/s with strong positive
    # acceleration across successive states.
    segments.append(move(CARTON_C, "carton", 42.0, 42.4, (300, 480, 380, 560), (360, 482, 440, 562)))
    segments.append(move(CARTON_C, "carton", 42.6, 43.0, (460, 500, 540, 580), (640, 540, 720, 620)))
    segments.append(move(CARTON_C, "carton", 43.2, 43.8, (820, 600, 900, 680), (1180, 700, 1260, 780)))

    # 47.0-52.0s  Rough handling: high speed, high acceleration, and a
    # direction reversal beyond 100 degrees.
    segments.append(move(CARTON_D, "carton", 47.0, 47.6, (400, 300, 480, 380), (760, 300, 840, 380)))
    segments.append(move(CARTON_D, "carton", 47.8, 48.4, (800, 300, 880, 380), (440, 306, 520, 386)))
    segments.append(move(CARTON_D, "carton", 48.6, 49.2, (480, 300, 560, 380), (840, 294, 920, 374)))

    # 54.0-60.0s  Pallet overhang: the carton is supported by less than 70% of
    # its width and sits within the 20px vertical gap.
    segments.append(hold(PALLET, "pallet", 54.0, 60.0, (700, 900, 930, 980)))
    segments.append(hold(CARTON_A, "carton", 54.0, 60.0, (840, 818, 1040, 898)))

    # 62.0-68.0s  Unstable stack: three cartons, each poorly supported by the
    # one beneath it, gaps under 20px.
    segments.append(hold(CARTON_B, "carton", 62.0, 68.0, (760, 900, 960, 980)))
    segments.append(hold(CARTON_C, "carton", 62.0, 68.0, (880, 820, 1080, 896)))
    segments.append(hold(CARTON_D, "carton", 62.0, 68.0, (1000, 740, 1200, 816)))

    # 70.0-76.0s  Improper placement: a carton left stationary for more than
    # 2.0s outside staging, loading and pallet zones.
    segments.append(hold(CARTON_A, "carton", 70.0, 76.0, (180, 240, 300, 360)))

    # 78.0-84.0s  Motion anomaly: a slow baseline followed by a speed spike far
    # beyond the 260px/s floor and the 4x ratio.
    segments.append(move(WORKER_B, "person", 78.0, 81.0, (300, 600, 380, 800), (390, 600, 470, 800)))
    segments.append(move(WORKER_B, "person", 81.2, 82.0, (500, 600, 580, 800), (1100, 604, 1180, 804)))
    segments.append(move(WORKER_B, "person", 82.2, 84.0, (1140, 604, 1220, 804), (1320, 608, 1400, 808)))

    return segments


def frames_from_segments(
    segments: Sequence[Segment],
    *,
    step: float = STEP_SECONDS,
) -> list[tuple[int, float, tuple[TrackedObject, ...]]]:
    """Sample every segment onto a fixed cadence of frames."""

    end_time = max(segment.end_time for segment in segments)
    frame_count = int(round(end_time / step)) + 1
    frames: list[tuple[int, float, tuple[TrackedObject, ...]]] = []
    for index in range(frame_count):
        timestamp = round(index * step, 6)
        frame_number = int(round(timestamp * SOURCE_FPS))
        objects: list[TrackedObject] = []
        seen: set[int] = set()
        for segment in segments:
            if segment.track_id in seen:
                continue
            box = segment.box_at(timestamp)
            if box is None:
                continue
            seen.add(segment.track_id)
            x1, y1, x2, y2 = box
            objects.append(
                TrackedObject(
                    track_id=segment.track_id,
                    class_name=segment.class_name,
                    confidence=0.91,
                    bbox=(x1, y1, x2, y2),
                    center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    timestamp=timestamp,
                    frame_number=frame_number,
                )
            )
        frames.append((frame_number, timestamp, tuple(objects)))
    return frames


def run_pipeline(
    frames: Sequence[tuple[int, float, tuple[TrackedObject, ...]]],
    database: EventDatabase,
) -> dict[str, Any]:
    """Drive the real engines over the sampled frames and persist incidents."""

    memory = ObjectMemory(max_history_seconds=12.0, max_states_per_track=200)
    zones = build_default_zones(FRAME_WIDTH, FRAME_HEIGHT)
    graph = SceneGraph(memory=memory, zones=zones, max_snapshots=64)
    registry = build_default_registry()
    incident_manager = IncidentManager(risk_engine=RiskEngine())

    track_observations = 0
    unique_objects: dict[str, set[int]] = {}

    for frame_number, timestamp, objects in frames:
        track_observations += len(objects)
        for tracked in objects:
            unique_objects.setdefault(tracked.class_name, set()).add(tracked.track_id)
        memory.update(objects, timestamp, frame_number=frame_number)
        graph.update(objects, timestamp=timestamp, memory=memory)
        context = BehaviourContext(
            timestamp=timestamp,
            memory=memory,
            scene_graph=graph,
            frame_width=FRAME_WIDTH,
            frame_height=FRAME_HEIGHT,
        )
        for event in registry.detect(context):
            incident_manager.ingest(event)

    prevention = PreventionEngine()
    for incident in incident_manager.incidents:
        recommendations = prevention.recommend(incident.to_dict())
        if recommendations:
            incident.evidence["prevention"] = recommendations
            incident.evidence["root_cause_category"] = prevention.root_cause(
                incident.to_dict()
            )
        database.insert_event(VIDEO_ID, incident)

    return {
        "track_observations": track_observations,
        "unique_objects": {
            name: len(ids) for name, ids in sorted(unique_objects.items())
        },
        "frames_processed": len(frames),
    }


def build_tracks_payload(
    frames: Sequence[tuple[int, float, tuple[TrackedObject, ...]]],
) -> list[dict[str, Any]]:
    """Export the trajectories so the site can replay them on a canvas.

    The website draws this instead of shipping a video file. It is the same
    geometry the behaviour rules consumed, which keeps the replay and the
    evidence in exact agreement rather than approximately illustrating it.
    """

    payload: list[dict[str, Any]] = []
    for _frame_number, timestamp, objects in frames:
        payload.append(
            {
                "t": round(timestamp, 3),
                "o": [
                    {
                        "id": tracked.track_id,
                        "c": tracked.class_name,
                        "b": [round(value, 1) for value in tracked.bbox],
                    }
                    for tracked in objects
                ],
            }
        )
    return payload


def zones_payload() -> list[dict[str, Any]]:
    zones = build_default_zones(FRAME_WIDTH, FRAME_HEIGHT)
    exported: list[dict[str, Any]] = []
    for zone in zones.zones:
        exported.append(
            {
                "name": zone.name,
                "polygon": [
                    [round(float(point[0]), 1), round(float(point[1]), 1)]
                    for point in zone.polygon
                ],
            }
        )
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "frontend" / "public" / "demo",
        help="Directory that receives the generated JSON fixtures.",
    )
    args = parser.parse_args()

    segments = build_segments()
    frames = frames_from_segments(segments)
    duration = frames[-1][1] if frames else 0.0

    with tempfile.TemporaryDirectory() as workdir:
        database_path = Path(workdir) / "demo.sqlite3"
        with EventDatabase(database_path) as database:
            database.register_video(
                VIDEO_ID,
                SOURCE_NAME,
                fps=SOURCE_FPS,
                width=FRAME_WIDTH,
                height=FRAME_HEIGHT,
                total_frames=int(round(duration * SOURCE_FPS)) + 1,
                duration=duration,
            )
            run_stats = run_pipeline(frames, database)
            events = [dict(event) for event in database.get_events(VIDEO_ID)]
            statistics = database.get_event_statistics(VIDEO_ID)

    # Row insert times are wall-clock and carry no meaning for a static
    # fixture. Dropping them keeps the generated files byte-identical between
    # runs, so CI can assert reproducibility and a regeneration does not churn
    # the diff.
    for event in events:
        event.pop("created_at", None)
        event.pop("updated_at", None)

    # Present the rows through the same module the API uses, so the static
    # fixture is shaped exactly like a live /api/videos/{id} response and the
    # frontend needs no second code path. Artifact verification is off: the
    # published site ships no snapshot files, and an unconditional False would
    # read as a failed integrity check rather than an absent one.
    presented = [event_payload(event, verify_artifacts=False) for event in events]

    behaviour_counts: dict[str, int] = {}
    for event in presented:
        behaviour = str(event.get("behaviour", ""))
        behaviour_counts[behaviour] = behaviour_counts.get(behaviour, 0) + 1

    video = {
        "video_id": VIDEO_ID,
        "name": SOURCE_NAME,
        "fps": SOURCE_FPS,
        "width": FRAME_WIDTH,
        "height": FRAME_HEIGHT,
        "total_frames": int(round(duration * SOURCE_FPS)) + 1,
        "duration": duration,
        # There is no rendered output video: the site replays the trajectory
        # geometry on a canvas instead, which keeps the replay and the evidence
        # in exact agreement.
        "processed": False,
        "processed_url": None,
        "events": presented,
        "statistics": statistics,
        "graph": evidence_graph(presented),
        "supported_behaviours": list(SUPPORTED_BEHAVIOURS),
        "model_scope_note": MODEL_SCOPE_NOTE,
        "anomaly_scope_note": ANOMALY_SCOPE_NOTE,
        "model_options": [],
        "frames_processed": run_stats["frames_processed"],
        "track_observations": run_stats["track_observations"],
        "unique_objects": run_stats["unique_objects"],
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "video.json").write_text(
        json.dumps(video, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (args.output / "tracks.json").write_text(
        json.dumps(
            {
                "width": FRAME_WIDTH,
                "height": FRAME_HEIGHT,
                "fps": SOURCE_FPS,
                "step_seconds": STEP_SECONDS,
                "duration": round(duration, 3),
                "zones": zones_payload(),
                "frames": build_tracks_payload(frames),
            },
            separators=(",", ":"),
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output / "meta.json").write_text(
        json.dumps(
            {
                "video_id": VIDEO_ID,
                "name": SOURCE_NAME,
                "provenance": (
                    "Synthetic reconstruction. Scripted tracked-object "
                    "trajectories were processed by the real KAVACH behaviour, "
                    "risk, incident and prevention engines using the "
                    "repository's committed thresholds. Object detection was "
                    "not involved: these are not detections from camera "
                    "footage, and no perception accuracy may be inferred from "
                    "this dataset."
                ),
                "generator": "scripts/build_demo_dataset.py",
                "width": FRAME_WIDTH,
                "height": FRAME_HEIGHT,
                "fps": SOURCE_FPS,
                "duration": round(duration, 3),
                "event_count": len(events),
                "behaviour_counts": behaviour_counts,
                **run_stats,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"frames            : {len(frames)}")
    print(f"duration (s)      : {duration:.1f}")
    print(f"events stored     : {len(events)}")
    for behaviour, count in sorted(behaviour_counts.items()):
        print(f"  {behaviour:<34} {count}")
    print(f"written to        : {args.output}")


if __name__ == "__main__":
    main()
