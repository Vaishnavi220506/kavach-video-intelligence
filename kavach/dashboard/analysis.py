"""Dashboard orchestration for the existing KAVACH processing modules."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from ..behaviours import BehaviourContext, BehaviourEvent, build_default_registry
from ..incidents import EvidenceStore, IncidentManager
from ..intelligence import (
    LOADING_ZONE,
    PALLET_ZONE,
    RESTRICTED_ZONE,
    STAGING_ZONE,
    ObjectMemory,
    SceneGraph,
    ZoneManager,
    draw_zones,
)
from ..perception import MultiObjectTracker, TrackedObject, WarehouseDetector, draw_tracked_objects
from ..prevention import PreventionEngine
from ..risk import RiskEngine
from ..storage import EventDatabase
from ..video import FramePacket, VideoReader, VideoWriter
from .overlays import draw_explainable_overlays

ProgressCallback = Callable[
    [float, np.ndarray, FramePacket, Sequence[TrackedObject], Sequence[BehaviourEvent]],
    None,
]


@dataclass(frozen=True)
class DashboardAnalysisResult:
    """Serializable summary retained in Streamlit session state."""

    video_id: str
    source_path: str
    output_path: str
    fps: float
    width: int
    height: int
    total_frames: int
    duration: float
    frames_processed: int
    track_observations: int
    unique_objects: Mapping[str, int]
    events: tuple[dict[str, object], ...]
    processing_seconds: float
    detector_model: str
    device: str

    @property
    def processing_fps(self) -> float:
        """Return source frames processed per wall-clock second."""

        if self.processing_seconds <= 0.0:
            return 0.0
        return self.frames_processed / self.processing_seconds

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["unique_objects"] = dict(self.unique_objects)
        result["events"] = list(self.events)
        result["processing_fps"] = self.processing_fps
        return result


def build_default_zones(width: int, height: int) -> ZoneManager:
    """Build editable dashboard defaults in frame-relative coordinates.

    These are starting polygons for demonstration footage, not a universal
    warehouse layout. A future camera-configuration UI should replace them.
    """

    width_value = int(width)
    height_value = int(height)
    if width_value <= 0 or height_value <= 0:
        raise ValueError("width and height must be positive")
    return ZoneManager.from_config(
        {
            STAGING_ZONE: {
                "polygon": [
                    [0.02 * width_value, 0.64 * height_value],
                    [0.32 * width_value, 0.64 * height_value],
                    [0.32 * width_value, 0.98 * height_value],
                    [0.02 * width_value, 0.98 * height_value],
                ],
                "color": (255, 180, 0),
            },
            LOADING_ZONE: {
                "polygon": [
                    [0.68 * width_value, 0.64 * height_value],
                    [0.98 * width_value, 0.64 * height_value],
                    [0.98 * width_value, 0.98 * height_value],
                    [0.68 * width_value, 0.98 * height_value],
                ],
                "color": (0, 180, 255),
            },
            RESTRICTED_ZONE: {
                "polygon": [
                    [0.30 * width_value, 0.34 * height_value],
                    [0.70 * width_value, 0.34 * height_value],
                    [0.70 * width_value, 0.94 * height_value],
                    [0.30 * width_value, 0.94 * height_value],
                ],
                "color": (0, 0, 255),
            },
            PALLET_ZONE: {
                "polygon": [
                    [0.36 * width_value, 0.76 * height_value],
                    [0.64 * width_value, 0.76 * height_value],
                    [0.64 * width_value, 0.98 * height_value],
                    [0.36 * width_value, 0.98 * height_value],
                ],
                "color": (0, 200, 0),
            },
        }
    )


def draw_dashboard_frame(
    frame: np.ndarray,
    packet: FramePacket,
    tracked_objects: Sequence[TrackedObject],
    zones: ZoneManager,
    events: Sequence[BehaviourEvent] = (),
) -> np.ndarray:
    """Annotate zones, YOLO/ByteTrack IDs, source timing, and current events.

    Motion anomalies are not detected by YOLO itself. YOLO supplies the visual
    object observation, ByteTrack supplies its short-term identity, and the
    temporal behaviour layer compares that identity's timestamped motion. The
    processed video makes that chain visible with a highlighted object,
    category label, and evidence-based explanation.
    """

    has_safety_signal = any(
        event.type not in {"OBJECT_ACTIVITY", "ZONE_TRANSITION"} for event in events
    )
    # The banner names the relevant zone on signal frames. Hide the normal
    # zone-name labels there so they do not compete with flagged-object labels.
    annotated = draw_zones(frame, zones, show_labels=not has_safety_signal)
    annotated = draw_tracked_objects(
        annotated,
        tracked_objects,
        show_labels=not has_safety_signal,
    )
    annotated = draw_explainable_overlays(
        annotated,
        tracked_objects,
        zones,
        events,
    )
    cv2.putText(
        annotated,
        f"Frame {packet.frame_number} | {packet.timestamp:.2f}s",
        (18, annotated.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def _object_key(tracked: TrackedObject) -> str:
    return f"{str(tracked.class_name).strip().lower().replace(' ', '_')} #{tracked.track_id}"


def analyse_video(
    source: str | Path,
    *,
    video_id: str,
    output: str | Path,
    detector: WarehouseDetector,
    database: EventDatabase,
    progress_callback: ProgressCallback | None = None,
    max_frames: int | None = None,
    evidence_store: EvidenceStore | None = None,
) -> DashboardAnalysisResult:
    """Run the existing KAVACH modules once and persist resulting incidents."""

    if not isinstance(detector, WarehouseDetector):
        raise TypeError("detector must be a WarehouseDetector")
    if not isinstance(database, EventDatabase):
        raise TypeError("database must be an EventDatabase")
    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if source_path == output_path:
        raise ValueError("analysis output must differ from the source video")

    started = time.perf_counter()
    track_observations = 0
    frames_processed = 0
    object_ids: dict[str, set[str]] = {}
    snapshot_store = evidence_store or EvidenceStore(
        output_path.parent.parent / "evidence" / "snapshots"
    )
    zones: ZoneManager | None = None
    if max_frames is not None and int(max_frames) <= 0:
        raise ValueError("max_frames must be positive when provided")
    with VideoReader(source_path) as reader:
        database.register_video(
            video_id,
            reader.source,
            fps=reader.fps,
            width=reader.width,
            height=reader.height,
            total_frames=reader.total_frames,
            duration=reader.duration,
        )
        zones = build_default_zones(reader.width, reader.height)
        memory = ObjectMemory()
        graph = SceneGraph(memory=memory, zones=zones)
        registry = build_default_registry()
        incident_manager = IncidentManager(risk_engine=RiskEngine.from_config())
        # Keep a new signal visible for a short source-time window. Without
        # this, a one-frame event could be missed while watching the output
        # video even though its exact evidence frame is stored separately.
        visible_event_expiry: dict[tuple[str, tuple[str, ...]], float] = {}
        visible_event_values: dict[
            tuple[str, tuple[str, ...]], BehaviourEvent
        ] = {}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with MultiObjectTracker(detector=detector) as tracker:
            with VideoWriter(
                output=output_path,
                fps=reader.fps,
                width=reader.width,
                height=reader.height,
                codec="avc1",
            ) as writer:
                for packet in reader:
                    if max_frames is not None and frames_processed >= int(max_frames):
                        break
                    tracked_objects = tracker.update(
                        packet.frame,
                        packet.timestamp,
                        packet.frame_number,
                    )
                    track_observations += len(tracked_objects)
                    frames_processed += 1
                    for tracked in tracked_objects:
                        object_ids.setdefault(str(tracked.class_name), set()).add(
                            _object_key(tracked)
                        )

                    memory.update(
                        tracked_objects,
                        packet.timestamp,
                        frame_number=packet.frame_number,
                    )
                    graph.update(
                        tracked_objects,
                        timestamp=packet.timestamp,
                        memory=memory,
                    )
                    context = BehaviourContext(
                        timestamp=packet.timestamp,
                        memory=memory,
                        scene_graph=graph,
                        frame_width=reader.width,
                        frame_height=reader.height,
                    )
                    current_events = registry.detect(context)
                    incident_updates = []
                    for event in current_events:
                        incident_updates.append(incident_manager.ingest(event))
                        event_key = (event.type, tuple(event.entities))
                        visible_event_values[event_key] = event
                        visible_event_expiry[event_key] = packet.timestamp + 2.0
                    for event_key in tuple(visible_event_expiry):
                        if packet.timestamp > visible_event_expiry[event_key]:
                            visible_event_expiry.pop(event_key, None)
                            visible_event_values.pop(event_key, None)
                    display_events = tuple(
                        visible_event_values[event_key]
                        for event_key in visible_event_expiry
                        if event_key in visible_event_values
                    )

                    annotated = draw_dashboard_frame(
                        packet.frame,
                        packet,
                        tracked_objects,
                        zones,
                        display_events,
                    )
                    writer.write(annotated)
                    for update in incident_updates:
                        artifact = snapshot_store.capture_snapshot(
                            video_id,
                            update.incident.id,
                            annotated,
                            timestamp=packet.timestamp,
                            frame_number=packet.frame_number,
                        )
                        artifacts = update.incident.evidence.setdefault(
                            "evidence_artifacts", []
                        )
                        if isinstance(artifacts, list) and not any(
                            isinstance(item, dict)
                            and item.get("path") == artifact.path
                            for item in artifacts
                        ):
                            # A long-lived incident is still bounded to avoid
                            # unbounded JPEG growth during a noisy run.
                            artifacts.append(artifact.to_dict())
                            del artifacts[:-20]
                    if progress_callback is not None:
                        stride = max(1, reader.total_frames // 100)
                        if packet.frame_number % stride == 0:
                            progress_callback(
                                min(
                                    1.0,
                                    (packet.frame_number + 1) / reader.total_frames,
                                ),
                                annotated,
                                packet,
                                tracked_objects,
                                display_events,
                            )

        for incident in incident_manager.incidents:
            prevention = PreventionEngine()
            recommendations = prevention.recommend(incident.to_dict())
            if recommendations:
                incident.evidence["prevention"] = recommendations
                incident.evidence["root_cause_category"] = prevention.root_cause(
                    incident.to_dict()
                )
            database.insert_event(video_id, incident)
        events = tuple(database.get_events(video_id))
        elapsed = time.perf_counter() - started
        return DashboardAnalysisResult(
            video_id=str(video_id),
            source_path=str(reader.source),
            output_path=str(output_path),
            fps=reader.fps,
            width=reader.width,
            height=reader.height,
            total_frames=reader.total_frames,
            duration=reader.duration,
            frames_processed=frames_processed,
            track_observations=track_observations,
            unique_objects={
                name: len(values) for name, values in sorted(object_ids.items())
            },
            events=events,
            processing_seconds=elapsed,
            detector_model=detector.model_name,
            device=detector.device,
        )
