"""Local API that exposes the existing KAVACH pipeline to the web UI.

The API is intentionally thin: video analysis still runs through the existing
VideoReader → detector → tracker → intelligence → incident pipeline. This
module only adds HTTP transport, job status, media delivery, and a readable
frontend-facing shape for the stored evidence.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kavach.assistant import GroundedAssistant, OllamaClient, OllamaError
from kavach.behaviours import (
    AISLE_OBSTRUCTION,
    COLLISION_RISK,
    IMPROPER_PLACEMENT,
    MOTION_ANOMALY,
    OBJECT_ACTIVITY,
    POSSIBLE_ROUGH_HANDLING,
    POSSIBLE_THROWING,
    ZONE_TRANSITION,
)
from kavach.dashboard import (
    DashboardAnalysisResult,
    analyse_video,
    download_video_url,
    persist_uploaded_video,
    video_file_sha256,
)
from kavach.incidents import EvidenceReplay, EvidenceStore, ReplayError
from kavach.perception import ModelLoadError, WarehouseDetector
from kavach.perception.classes import WAREHOUSE_VOCABULARY
from kavach.storage import DatabaseError, EventDatabase
from kavach.video import VideoError, VideoReader
from kavach.assistant.retrieval import format_timestamp


OUTPUT_ROOT = PROJECT_ROOT / "outputs"
SOURCE_DIRECTORY = OUTPUT_ROOT / "sources"
PROCESSED_DIRECTORY = OUTPUT_ROOT / "processed"
CLIP_DIRECTORY = OUTPUT_ROOT / "clips"
EVIDENCE_DIRECTORY = OUTPUT_ROOT / "evidence" / "snapshots"
DATABASE_PATH = OUTPUT_ROOT / "kavach.sqlite3"
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

ACTIVITY_EVENT_TYPES = frozenset({ZONE_TRANSITION, OBJECT_ACTIVITY})
ANOMALY_EVENT_TYPES = frozenset({MOTION_ANOMALY})
SAFETY_EVENT_TYPES = frozenset(
    {
        "ZONE_VIOLATION",
        "POSSIBLE_DRAGGING",
        "POSSIBLE_DROP",
        "PALLET_OVERHANG",
        "UNSTABLE_STACK",
        "UNSAFE_HUMAN_FORKLIFT_PROXIMITY",
        POSSIBLE_THROWING,
        POSSIBLE_ROUGH_HANDLING,
        AISLE_OBSTRUCTION,
        IMPROPER_PLACEMENT,
        COLLISION_RISK,
    }
)
SUPPORTED_BEHAVIOURS = tuple(
    sorted(ACTIVITY_EVENT_TYPES | ANOMALY_EVENT_TYPES | SAFETY_EVENT_TYPES)
)

database = EventDatabase(DATABASE_PATH)
replay = EvidenceReplay(database, output_dir=CLIP_DIRECTORY)
assistant = GroundedAssistant(database, replay=replay)

_detector_cache: dict[tuple[str, float], WarehouseDetector] = {}
_detector_lock = threading.Lock()
_jobs: dict[str, dict[str, object]] = {}
_jobs_lock = threading.Lock()
_analysis_executor = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="kavach-analysis",
)

app = FastAPI(
    title="KAVACH Local API",
    version="1.0.0",
    description="Local evidence-backed video intelligence API.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UrlRequest(BaseModel):
    """Direct video URL request."""

    url: str = Field(min_length=1, max_length=2048)


class AnalysisRequest(BaseModel):
    """Safe subset of analysis options exposed to the web UI."""

    confidence: float = Field(default=0.25, ge=0.05, le=0.95)
    model: str = Field(default="bundled", pattern="^(bundled|custom|warehouse|mvp|world)$")


class ChatRequest(BaseModel):
    """Grounded assistant question."""

    video_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=2000)
    use_llm: bool = True


class ReviewRequest(BaseModel):
    """Human review state for one stored incident."""

    status: str = Field(pattern="^(NEW|REVIEWED|FALSE_POSITIVE)$")


def _not_found(message: str) -> HTTPException:
    return HTTPException(status_code=404, detail=message)


def _video_or_404(video_id: str) -> dict[str, object]:
    video = database.get_video(video_id)
    if video is None:
        raise _not_found(f"Unknown video: {video_id}")
    return video


def _event_or_404(event_id: str) -> dict[str, object]:
    event = database.get_event(event_id)
    if event is None:
        raise _not_found(f"Unknown event: {event_id}")
    return event


def _processed_path(video_id: str) -> Path:
    return (PROCESSED_DIRECTORY / f"{video_id}.mp4").resolve()


def _model_options() -> list[dict[str, object]]:
    """Describe local checkpoint choices without loading heavyweight weights."""

    bundled = PROJECT_ROOT / "yolo11n.pt"
    custom = (
        PROJECT_ROOT
        / "notebooks"
        / "runs"
        / "trained_models"
        / "yolo_worker_safety"
        / "weights"
        / "best.pt"
    )
    warehouse = PROJECT_ROOT / "weights" / "kavach_warehouse.pt"
    mvp = PROJECT_ROOT / "runs" / "mvp" / "person_carton_v2_100e" / "weights" / "best.pt"
    world = PROJECT_ROOT / "weights" / "yolov8s-worldv2.pt"
    return [
        {
            "key": "bundled",
            "label": "YOLO11n · COCO baseline",
            "available": bundled.is_file(),
            "scope": "General COCO classes; warehouse-specific labels are not guaranteed.",
        },
        {
            "key": "custom",
            "label": "Worker Safety · custom checkpoint",
            "available": custom.is_file(),
            "scope": "Existing checkpoint exposes worker only; add warehouse labels before reporting warehouse accuracy.",
        },
        {
            "key": "warehouse",
            "label": "KAVACH Warehouse · evaluated checkpoint",
            "available": warehouse.is_file(),
            "scope": "Reserved for a validated multi-class warehouse checkpoint placed at weights/kavach_warehouse.pt.",
        },
        {
            "key": "mvp",
            "label": "KAVACH MVP v2 · person + carton",
            "available": mvp.is_file(),
            "scope": "Small manually reviewed prototype; carton labels cover foreground/handled cartons only and are not general warehouse accuracy.",
        },
        {
            "key": "world",
            "label": "YOLO-World · warehouse prompts",
            "available": world.is_file(),
            "scope": "Open-vocabulary exploration only; prompts are not accuracy claims and must be labelled/evaluated.",
        },
    ]


def _event_payload(event: dict[str, object]) -> dict[str, object]:
    payload = dict(event)
    timestamp = float(payload.get("timestamp", 0.0))
    payload["timestamp"] = timestamp
    payload["timestamp_display"] = format_timestamp(timestamp)
    event_id = str(payload.get("event_id", ""))
    payload["replay_url"] = f"/api/events/{event_id}/replay"
    payload["clip_url"] = (
        f"/api/events/{event_id}/clip" if event_id else None
    )
    behaviour = str(payload.get("behaviour", ""))
    if behaviour in ACTIVITY_EVENT_TYPES:
        payload["signal_kind"] = "activity"
    elif behaviour in ANOMALY_EVENT_TYPES:
        payload["signal_kind"] = "anomaly"
    else:
        payload["signal_kind"] = "safety"
    raw_evidence = payload.get("evidence")
    evidence_summary: dict[str, object] = {}
    if isinstance(raw_evidence, dict):
        for candidate_key in ("latest_event", "initial_event"):
            candidate = raw_evidence.get(candidate_key)
            if isinstance(candidate, dict) and isinstance(candidate.get("evidence"), dict):
                evidence_summary = dict(candidate["evidence"])
                break
        if not evidence_summary:
            evidence_summary = dict(raw_evidence)
    payload["evidence_summary"] = evidence_summary
    artifacts: list[dict[str, object]] = []
    if isinstance(raw_evidence, dict):
        raw_artifacts = raw_evidence.get("evidence_artifacts")
        if isinstance(raw_artifacts, list):
            for index, artifact in enumerate(raw_artifacts):
                if not isinstance(artifact, dict):
                    continue
                item = dict(artifact)
                item["url"] = f"/api/events/{event_id}/snapshot/{index}"
                item["verified"] = EvidenceStore.verify(item)
                artifacts.append(item)
    payload["evidence_artifacts"] = artifacts
    payload["prevention"] = raw_evidence.get("prevention", []) if isinstance(raw_evidence, dict) else []
    payload["root_cause_category"] = raw_evidence.get("root_cause_category") if isinstance(raw_evidence, dict) else None
    return payload


def _evidence_graph(events: list[dict[str, object]]) -> dict[str, object]:
    """Build a transparent entity→incident graph from stored event evidence.

    This is deliberately not presented as a persisted frame-level scene graph;
    Module 6 graph snapshots are in-memory. The web graph only visualizes the
    relationships explicitly retained in SQLite incident records.
    """

    nodes: dict[str, dict[str, object]] = {}
    edges: list[dict[str, object]] = []
    for event in events:
        event_id = str(event.get("event_id", "event"))
        risk = event.get("risk") if isinstance(event.get("risk"), dict) else {}
        event_node = f"event:{event_id}"
        nodes[event_node] = {
            "id": event_node,
            "label": str(event.get("behaviour", "Event")).replace("_", " "),
            "kind": "incident",
            "risk": str(risk.get("category", "UNKNOWN")),
            "timestamp": float(event.get("timestamp", 0.0)),
        }
        for raw_entity in event.get("entities", []):
            entity = str(raw_entity)
            entity_node = f"entity:{entity}"
            nodes.setdefault(
                entity_node,
                {
                    "id": entity_node,
                    "label": entity.replace("_", " "),
                    "kind": "entity",
                },
            )
            edges.append(
                {
                    "source": entity_node,
                    "target": event_node,
                    "label": "involved in",
                }
            )
    return {"nodes": list(nodes.values()), "edges": edges}


def _video_summary(video: dict[str, object]) -> dict[str, object]:
    video_id = str(video["video_id"])
    output = _processed_path(video_id)
    payload: dict[str, object] = {
        "video_id": video_id,
        "name": Path(str(video["source_path"])).name,
        "fps": video.get("fps"),
        "width": video.get("width"),
        "height": video.get("height"),
        "total_frames": video.get("total_frames"),
        "duration": video.get("duration"),
        "processed": output.is_file(),
        "processed_url": (
            f"/api/videos/{video_id}/processed" if output.is_file() else None
        ),
    }
    raw_metadata = video.get("analysis_metadata_json")
    if raw_metadata:
        try:
            metadata = json.loads(str(raw_metadata))
        except json.JSONDecodeError:
            metadata = {}
        if isinstance(metadata, dict):
            payload.update(metadata)
    return payload


def _video_payload(video_id: str) -> dict[str, object]:
    video = _video_or_404(video_id)
    events = [_event_payload(item) for item in database.get_events(video_id)]
    stats = database.get_event_statistics(video_id)
    payload: dict[str, object] = {
        **_video_summary(video),
        "events": events,
        "statistics": stats,
        "graph": _evidence_graph(events),
        "supported_behaviours": list(SUPPORTED_BEHAVIOURS),
        "model_scope_note": (
            "The default model detects only classes present in its trained "
            "vocabulary. Warehouse labels such as forklift or carton are not "
            "claimed unless the selected weights actually expose them."
        ),
        "anomaly_scope_note": (
            "MOTION_ANOMALY is an explainable image-space novelty signal based "
            "on tracked speed and direction changes; it is not a learned "
            "damage or defect classifier."
        ),
        "model_options": _model_options(),
    }
    with _jobs_lock:
        result = _jobs.get(video_id, {}).get("result")
    if isinstance(result, dict):
        for key in ("frames_processed", "track_observations", "unique_objects", "processing_seconds", "processing_fps", "detector_model", "device"):
            if key in result:
                payload[key] = result[key]
    return payload


def _model_path(model: str) -> Path:
    if model == "warehouse":
        return PROJECT_ROOT / "weights" / "kavach_warehouse.pt"
    if model == "mvp":
        return PROJECT_ROOT / "runs" / "mvp" / "person_carton_v2_100e" / "weights" / "best.pt"
    if model == "world":
        return PROJECT_ROOT / "weights" / "yolov8s-worldv2.pt"
    if model == "custom":
        custom = (
            PROJECT_ROOT
            / "notebooks"
            / "runs"
            / "trained_models"
            / "yolo_worker_safety"
            / "weights"
            / "best.pt"
        )
        if custom.is_file():
            return custom
    return PROJECT_ROOT / "yolo11n.pt"


def _get_detector(model: str, confidence: float) -> WarehouseDetector:
    path = _model_path(model).resolve()
    if not path.is_file():
        raise ModelLoadError(f"Model weights were not found: {path}")
    key = (str(path), round(float(confidence), 4))
    with _detector_lock:
        detector = _detector_cache.get(key)
        if detector is None:
            if model == "world":
                detector = WarehouseDetector.from_yolo_world(
                    model_path=path,
                    vocabulary=WAREHOUSE_VOCABULARY,
                    confidence=float(confidence),
                    device="auto",
                )
            else:
                detector = WarehouseDetector(
                    model_path=path,
                    confidence=float(confidence),
                    device="auto",
                    allowed_classes=(),
                    model=YOLO(str(path)),
                )
            _detector_cache[key] = detector
        return detector


def _set_job(video_id: str, **values: object) -> None:
    with _jobs_lock:
        current = dict(_jobs.get(video_id, {}))
        current.update(values)
        _jobs[video_id] = current


def _run_analysis(
    video_id: str,
    source: Path,
    request: AnalysisRequest,
) -> None:
    _set_job(
        video_id,
        status="running",
        progress=0.0,
        message="Loading the perception model...",
    )

    def on_progress(
        ratio: float,
        _frame: Any,
        packet: Any,
        tracked: Any,
        events: Any,
    ) -> None:
        _set_job(
            video_id,
            status="running",
            progress=round(float(ratio), 4),
            frame_number=int(packet.frame_number),
            timestamp=float(packet.timestamp),
            active_tracks=len(tracked),
            latest_events=len(events),
            message=f"Processing source frame {packet.frame_number + 1}",
        )

    try:
        detector = _get_detector(request.model, request.confidence)
        old_clips = database.clear_events_for_video(video_id)
        clip_root = CLIP_DIRECTORY.resolve()
        for old_clip in old_clips:
            resolved_clip = old_clip.resolve()
            if resolved_clip == clip_root or clip_root not in resolved_clip.parents:
                continue
            try:
                resolved_clip.unlink(missing_ok=True)
            except OSError:
                # A stale replay should not prevent the new analysis from
                # completing; it is safe to report the new result regardless.
                pass
        output = _processed_path(video_id)
        result = analyse_video(
            source,
            video_id=video_id,
            output=output,
            detector=detector,
            database=database,
            progress_callback=on_progress,
        )
        database.set_video_analysis_metadata(
            video_id,
            {
                "frames_processed": result.frames_processed,
                "track_observations": result.track_observations,
                "unique_objects": dict(result.unique_objects),
                "processing_seconds": result.processing_seconds,
                "processing_fps": result.processing_fps,
                "detector_model": result.detector_model,
                "device": result.device,
                "supported_behaviours": list(SUPPORTED_BEHAVIOURS),
                "model_key": request.model,
                "detector_classes": sorted(detector.class_names.values()),
            },
        )
        _set_job(
            video_id,
            status="complete",
            progress=1.0,
            message="Analysis complete",
            result=result.to_dict(),
        )
    except Exception as exc:  # background errors must become visible to UI
        _set_job(
            video_id,
            status="error",
            progress=0.0,
            message=str(exc),
            error_type=type(exc).__name__,
        )


@app.get("/api/health")
def health() -> dict[str, object]:
    """Fast liveness check that does not block on model or Ollama startup."""

    return {"status": "ok", "service": "kavach-api"}


@app.get("/api/ollama")
def ollama_status() -> dict[str, object]:
    """Report local Ollama availability on demand."""

    client = OllamaClient()
    available = client.is_available()
    models: list[str] = []
    if available:
        try:
            models = list(client.list_models())
        except OllamaError:
            models = []
    return {
        "available": available,
        "model": client.model,
        "models": models,
    }


@app.get("/api/videos")
def list_videos() -> dict[str, object]:
    try:
        videos = [_video_summary(item) for item in database.list_videos()]
    except DatabaseError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"videos": videos}


@app.get("/api/videos/{video_id}")
def get_video(video_id: str) -> dict[str, object]:
    try:
        return _video_payload(video_id)
    except DatabaseError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/videos/upload")
async def upload_video(file: UploadFile = File(...)) -> dict[str, object]:
    """Persist and register one uploaded local video without analysing it yet."""

    content = await file.read()
    try:
        source = persist_uploaded_video(
            file.filename or "upload.mp4",
            content,
            output_dir=SOURCE_DIRECTORY,
        )
        video_id = "video-" + video_file_sha256(source)[:16]
        with VideoReader(source) as reader:
            database.register_video(
                video_id,
                reader.source,
                fps=reader.fps,
                width=reader.width,
                height=reader.height,
                total_frames=reader.total_frames,
                duration=reader.duration,
            )
        return _video_payload(video_id)
    except (VideoError, DatabaseError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/videos/url")
def ingest_url(request: UrlRequest) -> dict[str, object]:
    """Download one direct video URL, then register its metadata."""

    try:
        source = download_video_url(request.url, output_dir=SOURCE_DIRECTORY)
        video_id = "video-" + video_file_sha256(source)[:16]
        with VideoReader(source) as reader:
            database.register_video(
                video_id,
                reader.source,
                fps=reader.fps,
                width=reader.width,
                height=reader.height,
                total_frames=reader.total_frames,
                duration=reader.duration,
            )
        return _video_payload(video_id)
    except (VideoError, DatabaseError, ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/videos/{video_id}/analyse")
def start_analysis(video_id: str, request: AnalysisRequest) -> dict[str, object]:
    video = _video_or_404(video_id)
    source = Path(str(video["source_path"])).expanduser().resolve()
    if not source.is_file():
        raise HTTPException(
            status_code=400,
            detail="The registered source file is no longer available. Upload it again.",
        )
    with _jobs_lock:
        current = _jobs.get(video_id)
        if current and current.get("status") == "running":
            return {"video_id": video_id, **current}
        _jobs[video_id] = {
            "status": "queued",
            "progress": 0.0,
            "message": "Queued for analysis",
        }
    _analysis_executor.submit(_run_analysis, video_id, source, request)
    return {"video_id": video_id, "status": "queued", "progress": 0.0}


@app.get("/api/videos/{video_id}/job")
def analysis_status(video_id: str) -> dict[str, object]:
    _video_or_404(video_id)
    with _jobs_lock:
        current = dict(_jobs.get(video_id, {}))
    output = _processed_path(video_id)
    if not current and output.is_file():
        current = {
            "status": "complete",
            "progress": 1.0,
            "message": "Processed output is available",
        }
    return {"video_id": video_id, **current}


@app.get("/api/videos/{video_id}/processed")
def processed_video(video_id: str) -> FileResponse:
    _video_or_404(video_id)
    output = _processed_path(video_id)
    if not output.is_file():
        raise _not_found("Processed video is not available; analyse the source first.")
    return FileResponse(output, media_type="video/mp4", filename=output.name)


@app.get("/api/events")
def query_events(
    video_id: str | None = None,
    behaviour: str | None = None,
    risk: str | None = None,
    entity: str | None = None,
    start: float | None = Query(default=None, ge=0.0),
    end: float | None = Query(default=None, ge=0.0),
) -> dict[str, object]:
    """Return readable event rows with simple server-side filters."""

    events: list[dict[str, object]] = []
    candidates = database.list_videos()
    selected_ids = [
        str(item["video_id"])
        for item in candidates
        if video_id is None or str(item["video_id"]) == video_id
    ]
    for selected_id in selected_ids:
        events.extend(database.get_events(selected_id))
    normalized_behaviour = behaviour.upper() if behaviour else None
    normalized_risk = risk.upper() if risk else None
    normalized_entity = entity.lower().replace("#", "_").replace(" ", "_") if entity else None
    filtered = []
    for event in events:
        event_risk = event.get("risk") if isinstance(event.get("risk"), dict) else {}
        timestamp = float(event.get("timestamp", 0.0))
        if normalized_behaviour and str(event.get("behaviour")) != normalized_behaviour:
            continue
        if normalized_risk and str(event_risk.get("category")) != normalized_risk:
            continue
        if normalized_entity and not any(
            normalized_entity in str(item).lower() for item in event.get("entities", [])
        ):
            continue
        if start is not None and timestamp < start:
            continue
        if end is not None and timestamp > end:
            continue
        filtered.append(_event_payload(event))
    return {"events": filtered, "total": len(filtered)}


@app.get("/api/events/{event_id}")
def get_event(event_id: str) -> dict[str, object]:
    return _event_payload(_event_or_404(event_id))


@app.post("/api/events/{event_id}/replay")
def create_replay(event_id: str) -> dict[str, object]:
    _event_or_404(event_id)
    try:
        result = replay.create_clip(event_id)
    except ReplayError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        **result.to_dict(),
        "clip_url": f"/api/events/{event_id}/clip",
    }


@app.get("/api/events/{event_id}/clip")
def event_clip(event_id: str) -> FileResponse:
    event = _event_or_404(event_id)
    clip_path = event.get("clip_path")
    if not clip_path or not Path(str(clip_path)).is_file():
        raise _not_found("Replay clip has not been generated yet.")
    path = Path(str(clip_path)).resolve()
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@app.get("/api/events/{event_id}/snapshot/{snapshot_index}")
def event_snapshot(event_id: str, snapshot_index: int) -> FileResponse:
    """Serve a snapshot only when its stored SHA-256 still verifies."""

    if snapshot_index < 0:
        raise _not_found("snapshot index must be non-negative")
    event = _event_or_404(event_id)
    evidence = event.get("evidence")
    artifacts = evidence.get("evidence_artifacts") if isinstance(evidence, dict) else None
    if not isinstance(artifacts, list) or snapshot_index >= len(artifacts):
        raise _not_found("evidence snapshot is not available")
    artifact = artifacts[snapshot_index]
    if not isinstance(artifact, dict) or not EvidenceStore.verify(artifact):
        raise _not_found("evidence snapshot failed integrity verification")
    path = Path(str(artifact["path"])).expanduser().resolve()
    return FileResponse(path, media_type="image/jpeg", filename=path.name)


@app.get("/api/models")
def models() -> dict[str, object]:
    """Return detector choices and their honest class-scope notes."""

    return {"models": _model_options()}


@app.get("/api/review/queue")
def review_queue(video_id: str | None = None) -> dict[str, object]:
    """Return NEW events that still need supervisor review."""

    selected = [video_id] if video_id else [item["video_id"] for item in database.list_videos()]
    events: list[dict[str, object]] = []
    for selected_id in selected:
        events.extend(
            event
            for event in database.get_events(str(selected_id))
            if event.get("review_status") == "NEW"
        )
    return {"events": [_event_payload(event) for event in events], "total": len(events)}


@app.post("/api/events/{event_id}/review")
def update_review(event_id: str, request: ReviewRequest) -> dict[str, object]:
    try:
        return _event_payload(database.update_review_status(event_id, request.status))
    except DatabaseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, object]:
    _video_or_404(request.video_id)
    try:
        response = assistant.ask(
            request.question,
            video_id=request.video_id,
            use_llm=request.use_llm,
        )
        payload = response.to_dict()
        references = []
        for reference in payload.get("event_references", []):
            item = dict(reference)
            event_id = str(item.get("event_id", ""))
            item["replay_url"] = f"/api/events/{event_id}/replay"
            item["clip_url"] = f"/api/events/{event_id}/clip"
            references.append(item)
        payload["event_references"] = references
        return payload
    except (DatabaseError, OllamaError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/videos/{video_id}/analytics")
def analytics(video_id: str) -> dict[str, object]:
    payload = _video_payload(video_id)
    events = payload["events"]
    if not isinstance(events, list):
        events = []
    return {
        "statistics": payload["statistics"],
        "events": events,
        "graph": payload["graph"],
    }


# Serve the production React build from the same local origin when it exists.
# The Vite dev server remains useful during frontend development, while this
# mount gives users one reproducible command for a self-contained website.
if FRONTEND_DIST.is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_DIST / "assets"),
        name="frontend-assets",
    )

    @app.get("/{path:path}", include_in_schema=False)
    def frontend_app(path: str) -> FileResponse:
        candidate = (FRONTEND_DIST / path).resolve()
        if candidate.is_file() and FRONTEND_DIST in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html", media_type="text/html")
