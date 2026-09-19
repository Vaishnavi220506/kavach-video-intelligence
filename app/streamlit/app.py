"""KAVACH supervisor dashboard.

This file is intentionally an orchestration/UI layer. The computer-vision,
temporal reasoning, risk, storage, replay, and assistant implementations live
under ``kavach/`` and are reused without dashboard-specific copies.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import cv2
import streamlit as st
from ultralytics import YOLO

# Streamlit executes this file from ``app/streamlit``. Add the repository
# root explicitly so the reusable top-level ``kavach`` package is importable
# both from the dashboard and from direct Streamlit launch commands.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kavach.assistant import GroundedAssistant, OllamaError
from kavach.dashboard import (
    DashboardAnalysisResult,
    VideoSourceError,
    analyse_video,
    download_video_url,
    persist_uploaded_video,
    video_file_sha256,
)
from kavach.incidents import EvidenceReplay, ReplayError
from kavach.perception import ModelLoadError, WarehouseDetector
from kavach.storage import DatabaseError, EventDatabase
from kavach.video import VideoError, VideoReader


APP_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = APP_ROOT / "outputs" / "kavach.sqlite3"
PROCESSED_DIRECTORY = APP_ROOT / "outputs" / "processed"


def _initialize_state() -> None:
    defaults = {
        "source_path": None,
        "source_key": None,
        "source_name": None,
        "source_origin": None,
        "video_id": None,
        "analysis_result": None,
        "source_temp_dir": None,
        "chat_messages": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


@st.cache_resource(show_spinner=False)
def get_database(database_path: str) -> EventDatabase:
    """Keep one SQLite connection per Streamlit process."""

    return EventDatabase(database_path)


@st.cache_resource(show_spinner="Loading YOLO weights once...")
def load_yolo_model(model_path: str) -> YOLO:
    """Cache heavyweight model weights across Streamlit reruns."""

    return YOLO(model_path)


@st.cache_resource(show_spinner=False)
def get_detector(model_path: str, confidence: float) -> WarehouseDetector:
    """Reuse the cached YOLO object while allowing confidence changes."""

    return WarehouseDetector(
        model_path=model_path,
        confidence=confidence,
        device="auto",
        # Preserve the upstream application's broad model output. Behaviour
        # detectors still decide which semantic classes are relevant.
        allowed_classes=(),
        model=load_yolo_model(model_path),
    )


@st.cache_resource(show_spinner=False)
def get_assistant(database_path: str) -> GroundedAssistant:
    database = get_database(database_path)
    replay = EvidenceReplay(database, output_dir=APP_ROOT / "outputs" / "clips")
    return GroundedAssistant(database, replay=replay)


@st.cache_resource(show_spinner=False)
def get_replay(database_path: str) -> EvidenceReplay:
    return EvidenceReplay(
        get_database(database_path),
        output_dir=APP_ROOT / "outputs" / "clips",
    )


def _model_path(choice: str) -> Path:
    if choice == "Custom trained model":
        custom = APP_ROOT / "notebooks" / "runs" / "trained_models" / "yolo_worker_safety" / "weights" / "best.pt"
        if custom.is_file():
            return custom
        st.info("Custom weights were not found; the bundled YOLO11n model will be used.")
    return APP_ROOT / "yolo11n.pt"


def _session_source_directory() -> Path:
    value = st.session_state.get("source_temp_dir")
    if value:
        return Path(value)
    directory = Path(tempfile.mkdtemp(prefix="kavach_dashboard_"))
    st.session_state["source_temp_dir"] = str(directory)
    return directory


def _set_source(path: Path, *, key: str, name: str, origin: str) -> None:
    previous_key = st.session_state.get("source_key")
    st.session_state["source_path"] = str(path)
    st.session_state["source_key"] = key
    st.session_state["source_name"] = name
    st.session_state["source_origin"] = origin
    if previous_key != key:
        st.session_state["video_id"] = None
        st.session_state["analysis_result"] = None
        st.session_state["chat_messages"] = []


def _source_picker() -> None:
    st.subheader("Video source")
    source_mode = st.radio(
        "Choose a source",
        ("Local upload", "Direct video URL"),
        horizontal=True,
        key="source_mode",
    )
    if source_mode == "Local upload":
        uploaded = st.file_uploader(
            "Upload a video file",
            type=["avi", "m4v", "mkv", "mov", "mp4", "mpeg", "webm"],
            help="The file is written to a session-owned temporary directory.",
        )
        if uploaded is not None:
            content = uploaded.getvalue()
            key = "upload:" + hashlib.sha256(content).hexdigest()
            if st.session_state.get("source_key") != key:
                try:
                    path = persist_uploaded_video(
                        uploaded.name,
                        content,
                        output_dir=_session_source_directory(),
                    )
                    _set_source(
                        path,
                        key=key,
                        name=uploaded.name,
                        origin="local upload",
                    )
                except VideoSourceError as exc:
                    st.error(str(exc))
    else:
        with st.form("direct_video_url_form", clear_on_submit=False):
            url = st.text_input(
                "Direct video URL",
                placeholder="https://example.com/warehouse.mp4",
                help="Direct HTTP(S) video responses only; webpage URLs are not supported.",
            )
            submitted = st.form_submit_button("Load URL")
        if submitted:
            try:
                path = download_video_url(
                    url,
                    output_dir=_session_source_directory(),
                )
                _set_source(
                    path,
                    key="url:" + video_file_sha256(path),
                    name=Path(path).name,
                    origin="direct video URL",
                )
                st.success("Direct video downloaded to temporary session storage.")
            except VideoSourceError as exc:
                st.error(str(exc))

    source_path = st.session_state.get("source_path")
    if source_path:
        st.caption(
            f"Selected: {st.session_state.get('source_name')} "
            f"({st.session_state.get('source_origin')})"
        )


def _format_timestamp(seconds: object) -> str:
    total = max(0, int(round(float(seconds))))
    minutes, remaining = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    return (
        f"{hours:02d}:{minutes:02d}:{remaining:02d}"
        if hours
        else f"{minutes:02d}:{remaining:02d}"
    )


def _risk(event: Mapping[str, object]) -> Mapping[str, object]:
    value = event.get("risk")
    return value if isinstance(value, Mapping) else {}


def _event_table(events: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for event in events:
        risk = _risk(event)
        rows.append(
            {
                "timestamp": _format_timestamp(event.get("timestamp", 0.0)),
                "event_id": event.get("event_id"),
                "behaviour": event.get("behaviour"),
                "risk": risk.get("category", "UNKNOWN"),
                "risk_score": risk.get("score", 0.0),
                "entities": ", ".join(str(x) for x in event.get("entities", [])),
                "status": event.get("review_status"),
                "occurrences": event.get("occurrence_count", 1),
            }
        )
    return rows


def _show_replay(event_id: str, *, key_prefix: str) -> None:
    database = get_database(str(DATABASE_PATH))
    event = database.get_event(event_id)
    if event is None:
        st.error(f"Event {event_id} is no longer in the database.")
        return
    clip_path = event.get("clip_path")
    if clip_path and Path(str(clip_path)).is_file():
        st.video(str(clip_path))
        return
    if st.button("Create replay clip", key=f"{key_prefix}_{event_id}"):
        try:
            result = get_replay(str(DATABASE_PATH)).create_clip(event_id)
            st.success(
                f"Replay ready: {_format_timestamp(result.start_time)}–"
                f"{_format_timestamp(result.end_time)}"
            )
            st.video(str(result.clip_path))
        except ReplayError as exc:
            st.error(f"Replay unavailable: {exc}")


def _render_event_cards(events: list[dict[str, object]], *, key_prefix: str) -> None:
    if not events:
        st.info("No stored events match the current filters.")
        return
    st.dataframe(_event_table(events), use_container_width=True, hide_index=True)
    for event in events:
        risk = _risk(event)
        event_id = str(event.get("event_id"))
        with st.expander(
            f"{_format_timestamp(event.get('timestamp', 0.0))} · "
            f"{event.get('behaviour')} · {risk.get('category', 'UNKNOWN')} · {event_id}"
        ):
            st.write(
                {
                    "entities": event.get("entities", []),
                    "risk_score": risk.get("score"),
                    "detection_confidence": risk.get("detection_confidence"),
                    "review_status": event.get("review_status"),
                    "occurrence_count": event.get("occurrence_count"),
                }
            )
            st.code(
                json.dumps(
                    {
                        "risk": risk,
                        "evidence": event.get("evidence", {}),
                    },
                    indent=2,
                    default=str,
                ),
                language="json",
            )
            _show_replay(event_id, key_prefix=key_prefix)


def _active_video_id(database: EventDatabase) -> str | None:
    current = st.session_state.get("video_id")
    if current:
        return str(current)
    videos = database.list_videos()
    if len(videos) == 1:
        return str(videos[0]["video_id"])
    return None


def _video_selector(database: EventDatabase, *, key: str) -> str | None:
    videos = database.list_videos()
    if not videos:
        return None
    labels = {
        str(video["video_id"]): Path(str(video["source_path"])).name
        for video in videos
    }
    active = _active_video_id(database)
    options = list(labels)
    index = options.index(active) if active in options else 0
    selected = st.selectbox(
        "Video",
        options,
        index=index,
        format_func=lambda value: labels[value],
        key=key,
    )
    return str(selected)


def _analyse_section() -> None:
    st.header("ANALYSE")
    st.write("Select one local video or one direct video URL, then explicitly start analysis.")
    _source_picker()
    source_path = st.session_state.get("source_path")
    model_choice = st.selectbox(
        "Perception model",
        ("Bundled YOLO11n", "Custom trained model"),
        key="model_choice",
    )
    confidence = st.slider(
        "Detection confidence",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        key="detection_confidence",
    )
    analyse_clicked = st.button(
        "Analyse video",
        type="primary",
        disabled=not bool(source_path),
        key="analyse_video_button",
    )
    if analyse_clicked and source_path:
        source = Path(str(source_path))
        video_id = "video-" + video_file_sha256(source)[:16]
        model_path = _model_path(model_choice)
        PROCESSED_DIRECTORY.mkdir(parents=True, exist_ok=True)
        output_path = PROCESSED_DIRECTORY / f"{video_id}.mp4"
        progress = st.progress(0.0, text="Starting video analysis...")
        preview = st.empty()
        status = st.empty()

        def update_progress(ratio, frame, packet, tracked, events):
            progress.progress(
                min(1.0, float(ratio)),
                text=f"Frame {packet.frame_number + 1} / source frames",
            )
            preview.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                # Streamlit 1.32 exposes this image option as
                # ``use_column_width``; ``use_container_width`` belongs to
                # newer widget APIs and raises a TypeError here.
                use_column_width=True,
            )
            status.caption(
                f"{packet.timestamp:.2f}s · {len(tracked)} active tracks · "
                f"{len(events)} new behaviour events"
            )

        try:
            detector = get_detector(str(model_path), float(confidence))
            result = analyse_video(
                source,
                video_id=video_id,
                output=output_path,
                detector=detector,
                database=get_database(str(DATABASE_PATH)),
                progress_callback=update_progress,
            )
            progress.progress(1.0, text="Analysis complete")
            st.session_state["video_id"] = result.video_id
            st.session_state["analysis_result"] = result.to_dict()
            st.success(
                f"Processed {result.frames_processed} frames in "
                f"{result.processing_seconds:.2f}s."
            )
        except (VideoError, ModelLoadError, DatabaseError, ValueError, RuntimeError) as exc:
            st.error(f"Analysis failed: {exc}")

    result_data = st.session_state.get("analysis_result")
    if not result_data:
        st.info("No analysis result is loaded in this session yet.")
        return
    result = DashboardAnalysisResult(
        video_id=str(result_data["video_id"]),
        source_path=str(result_data["source_path"]),
        output_path=str(result_data["output_path"]),
        fps=float(result_data["fps"]),
        width=int(result_data["width"]),
        height=int(result_data["height"]),
        total_frames=int(result_data["total_frames"]),
        duration=float(result_data["duration"]),
        frames_processed=int(result_data["frames_processed"]),
        track_observations=int(result_data["track_observations"]),
        unique_objects=dict(result_data["unique_objects"]),
        events=tuple(result_data["events"]),
        processing_seconds=float(result_data["processing_seconds"]),
        detector_model=str(result_data["detector_model"]),
        device=str(result_data["device"]),
    )
    st.subheader("Analysis result")
    metrics = st.columns(6)
    metrics[0].metric("FPS", f"{result.fps:.2f}")
    metrics[1].metric("Resolution", f"{result.width}×{result.height}")
    metrics[2].metric("Duration", _format_timestamp(result.duration))
    metrics[3].metric("Frames", result.frames_processed)
    metrics[4].metric("Processing FPS", f"{result.processing_fps:.2f}")
    metrics[5].metric("Events", len(result.events))
    st.caption(f"Model: {result.detector_model} · Device: {result.device}")
    st.video(result.output_path)
    st.write("Unique tracked objects")
    st.dataframe(
        [{"class": name, "unique_track_ids": count} for name, count in result.unique_objects.items()],
        use_container_width=True,
        hide_index=True,
    )
    st.write("Detected events")
    _render_event_cards(list(result.events), key_prefix="analyse_replay")


def _events_section() -> None:
    st.header("EVENTS")
    database = get_database(str(DATABASE_PATH))
    video_id = _video_selector(database, key="events_video")
    if video_id is None:
        st.info("Analyse a video first to populate the event store.")
        return
    all_events = database.get_events(video_id)
    behaviour_options = sorted({str(event["behaviour"]) for event in all_events})
    risk_options = sorted({str(_risk(event).get("category", "UNKNOWN")) for event in all_events})
    filter_columns = st.columns(4)
    behaviour = filter_columns[0].selectbox("Behaviour", ["All", *behaviour_options], key="event_behaviour")
    risk = filter_columns[1].selectbox("Risk", ["All", *risk_options], key="event_risk")
    entity = filter_columns[2].text_input("Entity", placeholder="carton_12 or person_3", key="event_entity")
    maximum = max(
        [float(event.get("timestamp", 0.0)) for event in all_events]
        + [float((database.get_video(video_id) or {}).get("duration") or 1.0)]
    )
    time_window = filter_columns[3].slider(
        "Time window (seconds)",
        0.0,
        max(1.0, maximum),
        (0.0, max(1.0, maximum)),
        key="event_time_window",
    )
    normalized_entity = "_".join(entity.lower().replace("#", " ").replace("-", " ").split())
    events = []
    for event in all_events:
        if behaviour != "All" and event.get("behaviour") != behaviour:
            continue
        if risk != "All" and _risk(event).get("category") != risk:
            continue
        if normalized_entity and normalized_entity not in {
            str(value).lower() for value in event.get("entities", [])
        }:
            continue
        timestamp = float(event.get("timestamp", 0.0))
        if not time_window[0] <= timestamp <= time_window[1]:
            continue
        events.append(event)
    _render_event_cards(events, key_prefix="events_replay")


def _ask_section() -> None:
    st.header("ASK KAVACH")
    st.write("Questions are answered from SQLite event records first; Ollama only verbalizes supplied context.")
    database = get_database(str(DATABASE_PATH))
    video_id = _video_selector(database, key="ask_video")
    for message in st.session_state.get("chat_messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                for reference in message.get("references", []):
                    st.caption(
                        f"{reference.get('timestamp_display')} · "
                        f"{reference.get('event_id')} · {reference.get('behaviour')}"
                    )
                    _show_replay(
                        str(reference.get("event_id")),
                        key_prefix="ask_replay",
                    )
    question = st.chat_input("Ask about the analysed video")
    if question:
        if video_id is None:
            st.session_state["chat_messages"].append(
                {"role": "assistant", "content": "Analyse a video before asking questions."}
            )
            st.rerun()
        st.session_state["chat_messages"].append({"role": "user", "content": question})
        try:
            response = get_assistant(str(DATABASE_PATH)).ask(
                question,
                video_id=video_id,
                use_llm=True,
            )
            content = response.answer
            if response.llm_error:
                content += f"\n\n_Answer used deterministic fallback: {response.llm_error}_"
            st.session_state["chat_messages"].append(
                {
                    "role": "assistant",
                    "content": content,
                    "references": list(response.event_references),
                }
            )
        except (OllamaError, DatabaseError, RuntimeError, ValueError) as exc:
            st.session_state["chat_messages"].append(
                {"role": "assistant", "content": f"Unable to answer from stored evidence: {exc}"}
            )
        st.rerun()


def _analytics_section() -> None:
    st.header("ANALYTICS")
    database = get_database(str(DATABASE_PATH))
    video_id = _video_selector(database, key="analytics_video")
    if video_id is None:
        st.info("Analyse a video first to populate analytics.")
        return
    statistics = database.get_event_statistics(video_id)
    metrics = st.columns(3)
    metrics[0].metric("Total events", int(statistics["total_events"]))
    metrics[1].metric("Average risk", f"{float(statistics['average_risk_score']):.1f}/100")
    events = database.get_events(video_id)
    repeat_count = sum(max(0, int(event.get("occurrence_count", 1)) - 1) for event in events)
    metrics[2].metric("Merged repeat observations", repeat_count)

    st.subheader("Events by behaviour")
    st.dataframe(
        [{"behaviour": name, "count": count} for name, count in statistics["by_behaviour"].items()],
        use_container_width=True,
        hide_index=True,
    )
    st.subheader("Risk distribution")
    st.dataframe(
        [{"risk": name, "count": count} for name, count in statistics["by_risk"].items()],
        use_container_width=True,
        hide_index=True,
    )
    st.subheader("Event timeline")
    st.dataframe(_event_table(events), use_container_width=True, hide_index=True)
    st.caption("Counts are stored incident rows. Repeat observations are shown separately from detection confidence.")


def main() -> None:
    st.set_page_config(page_title="KAVACH Supervisor Dashboard", page_icon="🦺", layout="wide")
    _initialize_state()
    st.title("KAVACH — Explainable Temporal Video Intelligence")
    st.caption("Supervisor workspace for analysed video, evidence-backed incidents, and grounded local search.")
    tabs = st.tabs(["ANALYSE", "EVENTS", "ASK KAVACH", "ANALYTICS"])
    with tabs[0]:
        _analyse_section()
    with tabs[1]:
        _events_section()
    with tabs[2]:
        _ask_section()
    with tabs[3]:
        _analytics_section()


if __name__ == "__main__":
    main()
