"""SQLite persistence for KAVACH videos, incidents, and evidence metadata."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..incidents.models import Incident
from ..risk import RISK_CATEGORIES

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
DEFAULT_DATABASE_PATH = Path("outputs") / "kavach.sqlite3"


class EventStorageError(RuntimeError):
    """Base class for expected SQLite event-storage errors."""


class DatabaseError(EventStorageError):
    """Raised when a database operation or record is invalid."""


@dataclass(frozen=True)
class VideoRecord:
    """Video metadata retained so replay can locate and bound a source."""

    video_id: str
    source_path: str
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    total_frames: int | None = None
    duration: float | None = None

    def __post_init__(self) -> None:
        if not str(self.video_id).strip() or not str(self.source_path).strip():
            raise DatabaseError("video_id and source_path cannot be empty")
        for name in ("fps", "duration"):
            value = getattr(self, name)
            if value is not None:
                numeric = float(value)
                if not math.isfinite(numeric) or numeric < 0.0:
                    raise DatabaseError(f"{name} must be finite and non-negative")
        for name in ("width", "height", "total_frames"):
            value = getattr(self, name)
            if value is not None and int(value) <= 0:
                raise DatabaseError(f"{name} must be positive")
        if self.fps is not None and float(self.fps) <= 0.0:
            raise DatabaseError("fps must be positive when provided")
        object.__setattr__(self, "video_id", str(self.video_id))
        object.__setattr__(self, "source_path", str(self.source_path))
        if self.fps is not None:
            object.__setattr__(self, "fps", float(self.fps))
        if self.duration is not None:
            object.__setattr__(self, "duration", float(self.duration))
        for name in ("width", "height", "total_frames"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, int(value))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: object, label: str) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise DatabaseError(f"{label} must be JSON serializable") from exc


def _decoded(value: str, label: str) -> object:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise DatabaseError(f"stored {label} is not valid JSON") from exc


class EventDatabase:
    """Store structured incident events in a local SQLite database."""

    def __init__(
        self,
        path: str | Path = DEFAULT_DATABASE_PATH,
        *,
        initialize: bool = True,
    ) -> None:
        self.path = Path(path).expanduser()
        if self.path != Path(":memory:"):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        # Streamlit may call a cached resource from different script-runner
        # threads. SQLite connections are thread-affine by default, so keep
        # one shared connection safe with an explicit re-entrant lock.
        self._lock = threading.RLock()
        try:
            self._connection = sqlite3.connect(
                str(self.path),
                check_same_thread=False,
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            if initialize:
                self._initialize()
        except sqlite3.Error as exc:
            raise DatabaseError(f"could not open database {self.path}: {exc}") from exc
        self._closed = False

    def _initialize(self) -> None:
        with self._lock:
            try:
                schema = SCHEMA_PATH.read_text(encoding="utf-8")
                self._connection.executescript(schema)
                columns = {
                    str(row[1])
                    for row in self._connection.execute(
                        "PRAGMA table_info(videos)"
                    ).fetchall()
                }
                if "analysis_metadata_json" not in columns:
                    self._connection.execute(
                        "ALTER TABLE videos ADD COLUMN analysis_metadata_json TEXT"
                    )
                self._connection.commit()
            except (OSError, sqlite3.Error) as exc:
                self._connection.close()
                raise DatabaseError(
                    f"could not initialize database schema {SCHEMA_PATH}: {exc}"
                ) from exc

    def _ensure_open(self) -> None:
        if self._closed:
            raise DatabaseError("database connection is closed")

    @staticmethod
    def _validate_video_id(video_id: str) -> str:
        value = str(video_id).strip()
        if not value:
            raise DatabaseError("video_id cannot be empty")
        return value

    @staticmethod
    def _validate_time(value: float, label: str) -> float:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise DatabaseError(f"{label} must be finite and non-negative")
        return numeric

    def register_video(
        self,
        video_id: str,
        source_path: str | Path,
        *,
        fps: float | None = None,
        width: int | None = None,
        height: int | None = None,
        total_frames: int | None = None,
        duration: float | None = None,
    ) -> VideoRecord:
        """Insert or update a source video and its optional metadata."""

        self._ensure_open()
        record = VideoRecord(
            video_id=self._validate_video_id(video_id),
            source_path=str(source_path),
            fps=fps,
            width=width,
            height=height,
            total_frames=total_frames,
            duration=duration,
        )
        now = _now()
        with self._lock:
            try:
                with self._connection:
                    self._connection.execute(
                    """
                    INSERT INTO videos (
                        video_id, source_path, fps, width, height,
                        total_frames, duration, analysis_metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(video_id) DO UPDATE SET
                        source_path = excluded.source_path,
                        fps = COALESCE(excluded.fps, videos.fps),
                        width = COALESCE(excluded.width, videos.width),
                        height = COALESCE(excluded.height, videos.height),
                        total_frames = COALESCE(
                            excluded.total_frames, videos.total_frames
                        ),
                        duration = COALESCE(excluded.duration, videos.duration),
                        analysis_metadata_json = COALESCE(
                            excluded.analysis_metadata_json,
                            videos.analysis_metadata_json
                        )
                    """,
                        (
                            record.video_id,
                            record.source_path,
                            record.fps,
                            record.width,
                            record.height,
                            record.total_frames,
                            record.duration,
                            None,
                            now,
                        ),
                    )
            except sqlite3.Error as exc:
                raise DatabaseError(f"could not register video: {exc}") from exc
        return record

    upsert_video = register_video

    def set_video_analysis_metadata(
        self,
        video_id: str,
        metadata: Mapping[str, object],
    ) -> dict[str, object]:
        """Persist serializable metrics produced by an analysis run."""

        self._ensure_open()
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        value = self._validate_video_id(video_id)
        metadata_json = _json(dict(metadata), "analysis metadata")
        with self._lock:
            try:
                with self._connection:
                    cursor = self._connection.execute(
                        """
                        UPDATE videos
                        SET analysis_metadata_json = ?
                        WHERE video_id = ?
                        """,
                        (metadata_json, value),
                    )
                if cursor.rowcount == 0:
                    raise DatabaseError(f"unknown video: {video_id}")
            except sqlite3.Error as exc:
                raise DatabaseError(
                    f"could not store analysis metadata for {value}: {exc}"
                ) from exc
        result = self.get_video(value)
        assert result is not None
        return result

    def get_video(self, video_id: str) -> dict[str, object] | None:
        """Return one stored video record."""

        self._ensure_open()
        value = self._validate_video_id(video_id)
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM videos WHERE video_id = ?",
                (value,),
            ).fetchone()
        return None if row is None else dict(row)

    def list_videos(self) -> list[dict[str, object]]:
        """Return registered videos in creation order."""

        self._ensure_open()
        with self._lock:
            try:
                rows = self._connection.execute(
                    """
                    SELECT * FROM videos
                    ORDER BY created_at, video_id
                    """
                ).fetchall()
            except sqlite3.Error as exc:
                raise DatabaseError(f"could not list videos: {exc}") from exc
        return [dict(row) for row in rows]

    def clear_events_for_video(self, video_id: str) -> list[Path]:
        """Remove a video's prior analysis rows before an explicit re-run.

        The returned clip paths let the caller remove generated replay files
        without touching arbitrary user files. Re-running the same video is
        therefore idempotent in the event store instead of appending a second
        copy of every incident.
        """

        self._ensure_open()
        value = self._validate_video_id(video_id)
        with self._lock:
            try:
                rows = self._connection.execute(
                    "SELECT clip_path FROM events WHERE video_id = ?",
                    (value,),
                ).fetchall()
                clip_paths = [
                    Path(str(row["clip_path"])).expanduser()
                    for row in rows
                    if row["clip_path"]
                ]
                with self._connection:
                    self._connection.execute(
                        "DELETE FROM events WHERE video_id = ?",
                        (value,),
                    )
                return clip_paths
            except sqlite3.Error as exc:
                raise DatabaseError(
                    f"could not clear events for video {value}: {exc}"
                ) from exc

    def insert_event(
        self,
        video_id: str,
        incident: Incident,
        *,
        clip_path: str | Path | None = None,
    ) -> str:
        """Insert or update one Module 8 Incident as a database event row."""

        self._ensure_open()
        if not isinstance(incident, Incident):
            raise TypeError("incident must be an Incident")
        video_value = self._validate_video_id(video_id)
        clip_value = None if clip_path is None else str(Path(clip_path).expanduser().resolve())
        risk = incident.risk.to_dict()
        entities = tuple(dict.fromkeys(incident.entities))
        now = _now()
        with self._lock:
            try:
                with self._connection:
                    self._connection.execute(
                    """
                    INSERT INTO events (
                        event_id, video_id, timestamp, behaviour, entities_json,
                        risk_score, risk_category, detection_confidence, risk_json,
                        evidence_json, review_status, lifecycle, clip_path,
                        first_seen_timestamp, last_seen_timestamp, occurrence_count,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(event_id) DO UPDATE SET
                        video_id = excluded.video_id,
                        timestamp = excluded.timestamp,
                        behaviour = excluded.behaviour,
                        entities_json = excluded.entities_json,
                        risk_score = excluded.risk_score,
                        risk_category = excluded.risk_category,
                        detection_confidence = excluded.detection_confidence,
                        risk_json = excluded.risk_json,
                        evidence_json = excluded.evidence_json,
                        review_status = excluded.review_status,
                        lifecycle = excluded.lifecycle,
                        clip_path = COALESCE(excluded.clip_path, events.clip_path),
                        first_seen_timestamp = excluded.first_seen_timestamp,
                        last_seen_timestamp = excluded.last_seen_timestamp,
                        occurrence_count = excluded.occurrence_count,
                        updated_at = excluded.updated_at
                    """,
                        (
                            incident.id,
                            video_value,
                            incident.timestamp,
                            incident.type,
                            _json(list(entities), "entities"),
                            incident.risk.score,
                            incident.risk.category,
                            incident.risk.detection_confidence,
                            _json(risk, "risk"),
                            _json(incident.evidence, "evidence"),
                            incident.status,
                            incident.lifecycle,
                            clip_value,
                            incident.timestamp,
                            incident.last_seen_timestamp,
                            incident.occurrence_count,
                            now,
                            now,
                        ),
                    )
                    self._connection.execute(
                        "DELETE FROM event_entities WHERE event_id = ?",
                        (incident.id,),
                    )
                    self._connection.executemany(
                    """
                    INSERT INTO event_entities (event_id, entity_id)
                    VALUES (?, ?)
                    """,
                        [(incident.id, entity) for entity in entities],
                    )
            except sqlite3.IntegrityError as exc:
                raise DatabaseError(
                    f"could not insert event {incident.id}; register its video first: {exc}"
                ) from exc
            except sqlite3.Error as exc:
                raise DatabaseError(f"could not insert event {incident.id}: {exc}") from exc
        return incident.id

    upsert_event = insert_event

    def _row_to_event(self, row: sqlite3.Row) -> dict[str, object]:
        data = dict(row)
        data["entities"] = _decoded(data.pop("entities_json"), "entities")
        data["risk"] = _decoded(data.pop("risk_json"), "risk")
        data["evidence"] = _decoded(data.pop("evidence_json"), "evidence")
        return data

    def get_event(self, event_id: str) -> dict[str, object] | None:
        """Return one event by its stable Incident ID."""

        self._ensure_open()
        value = str(event_id).strip()
        if not value:
            raise DatabaseError("event_id cannot be empty")
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM events WHERE event_id = ?",
                (value,),
            ).fetchone()
        return None if row is None else self._row_to_event(row)

    def update_event_evidence(
        self,
        event_id: str,
        patch: Mapping[str, object],
    ) -> dict[str, object]:
        """Merge structured evidence into an existing event row."""

        if not isinstance(patch, Mapping):
            raise TypeError("evidence patch must be a mapping")
        self._ensure_open()
        value = str(event_id).strip()
        if not value:
            raise DatabaseError("event_id cannot be empty")
        with self._lock:
            event = self.get_event(value)
            if event is None:
                raise DatabaseError(f"unknown event: {event_id}")
            evidence = event.get("evidence")
            merged = dict(evidence) if isinstance(evidence, Mapping) else {}
            merged.update(dict(patch))
            try:
                with self._connection:
                    self._connection.execute(
                        "UPDATE events SET evidence_json = ?, updated_at = ? WHERE event_id = ?",
                        (_json(merged, "evidence"), _now(), value),
                    )
            except sqlite3.Error as exc:
                raise DatabaseError(f"could not update evidence for {value}: {exc}") from exc
        result = self.get_event(value)
        assert result is not None
        return result

    def _query_events(
        self,
        sql: str,
        parameters: Iterable[object] = (),
    ) -> list[dict[str, object]]:
        self._ensure_open()
        with self._lock:
            try:
                rows = self._connection.execute(sql, tuple(parameters)).fetchall()
            except sqlite3.Error as exc:
                raise DatabaseError(f"event query failed: {exc}") from exc
        return [self._row_to_event(row) for row in rows]

    def get_events(self, video_id: str) -> list[dict[str, object]]:
        """Return all events for one video in source-time order."""

        value = self._validate_video_id(video_id)
        return self._query_events(
            """
            SELECT * FROM events
            WHERE video_id = ?
            ORDER BY timestamp, event_id
            """,
            (value,),
        )

    def get_events_by_type(
        self,
        behaviour: str,
        *,
        video_id: str | None = None,
    ) -> list[dict[str, object]]:
        """Return events matching one behaviour type."""

        value = str(behaviour).strip()
        if not value:
            raise DatabaseError("behaviour cannot be empty")
        if video_id is None:
            return self._query_events(
                """
                SELECT * FROM events
                WHERE behaviour = ?
                ORDER BY timestamp, event_id
                """,
                (value,),
            )
        return self._query_events(
            """
            SELECT * FROM events
            WHERE behaviour = ? AND video_id = ?
            ORDER BY timestamp, event_id
            """,
            (value, self._validate_video_id(video_id)),
        )

    def get_events_by_risk(
        self,
        risk: str | None = None,
        *,
        minimum_score: float | None = None,
        video_id: str | None = None,
    ) -> list[dict[str, object]]:
        """Filter by risk category and/or minimum numeric risk score."""

        clauses: list[str] = []
        parameters: list[object] = []
        if risk is not None:
            category = str(risk).strip().upper()
            if category not in RISK_CATEGORIES:
                raise DatabaseError(f"unknown risk category: {risk}")
            clauses.append("risk_category = ?")
            parameters.append(category)
        if minimum_score is not None:
            score = self._validate_time(minimum_score, "minimum_score")
            if score > 100.0:
                raise DatabaseError("minimum_score cannot exceed 100")
            clauses.append("risk_score >= ?")
            parameters.append(score)
        if video_id is not None:
            clauses.append("video_id = ?")
            parameters.append(self._validate_video_id(video_id))
        where = "" if not clauses else "WHERE " + " AND ".join(clauses)
        return self._query_events(
            f"""
            SELECT * FROM events
            {where}
            ORDER BY risk_score DESC, timestamp, event_id
            """,
            parameters,
        )

    def get_events_between_times(
        self,
        video_id: str,
        start_time: float,
        end_time: float,
    ) -> list[dict[str, object]]:
        """Return events in an inclusive source-time interval."""

        video_value = self._validate_video_id(video_id)
        start = self._validate_time(start_time, "start_time")
        end = self._validate_time(end_time, "end_time")
        if end < start:
            raise DatabaseError("end_time cannot be earlier than start_time")
        return self._query_events(
            """
            SELECT * FROM events
            WHERE video_id = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp, event_id
            """,
            (video_value, start, end),
        )

    def get_events_for_entity(
        self,
        entity_id: str,
        *,
        video_id: str | None = None,
    ) -> list[dict[str, object]]:
        """Return events linked to one normalized entity ID."""

        entity = str(entity_id).strip()
        if not entity:
            raise DatabaseError("entity_id cannot be empty")
        parameters: list[object] = [entity]
        video_clause = ""
        if video_id is not None:
            video_clause = "AND events.video_id = ?"
            parameters.append(self._validate_video_id(video_id))
        return self._query_events(
            f"""
            SELECT events.*
            FROM events
            INNER JOIN event_entities
                ON event_entities.event_id = events.event_id
            WHERE event_entities.entity_id = ?
            {video_clause}
            ORDER BY events.timestamp, events.event_id
            """,
            parameters,
        )

    def get_event_statistics(
        self,
        video_id: str | None = None,
    ) -> dict[str, object]:
        """Return counts and average risk useful for later dashboards."""

        self._ensure_open()
        parameters: tuple[object, ...] = ()
        where = ""
        if video_id is not None:
            where = "WHERE video_id = ?"
            parameters = (self._validate_video_id(video_id),)
        with self._lock:
            try:
                total_row = self._connection.execute(
                    f"""
                    SELECT COUNT(*) AS total_events,
                           AVG(risk_score) AS average_risk_score
                    FROM events {where}
                    """,
                    parameters,
                ).fetchall()
                total_row = total_row[0]
                grouped = {}
                for column, output_key in (
                    ("behaviour", "by_behaviour"),
                    ("risk_category", "by_risk"),
                    ("review_status", "by_review_status"),
                ):
                    rows = self._connection.execute(
                        f"""
                        SELECT {column} AS group_name, COUNT(*) AS count
                        FROM events {where}
                        GROUP BY {column}
                        ORDER BY group_name
                        """,
                        parameters,
                    ).fetchall()
                    grouped[output_key] = {
                        str(row["group_name"]): int(row["count"]) for row in rows
                    }
            except sqlite3.Error as exc:
                raise DatabaseError(f"could not calculate event statistics: {exc}") from exc
        average = total_row["average_risk_score"]
        return {
            "total_events": int(total_row["total_events"]),
            "average_risk_score": 0.0 if average is None else float(average),
            **grouped,
        }

    def update_review_status(self, event_id: str, status: str) -> dict[str, object]:
        """Update the human review status of a stored event."""

        allowed = ("NEW", "REVIEWED", "FALSE_POSITIVE")
        normalized = str(status).strip().upper()
        if normalized not in allowed:
            raise DatabaseError(f"unknown review status: {status}")
        self._ensure_open()
        value = str(event_id).strip()
        with self._lock:
            existing = self.get_event(value)
            if existing is None:
                raise DatabaseError(f"unknown event: {event_id}")
            evidence = existing.get("evidence")
            merged_evidence = dict(evidence) if isinstance(evidence, Mapping) else {}
            history = merged_evidence.get("review_history")
            review_history = list(history) if isinstance(history, list) else []
            review_history.append({"status": normalized, "timestamp": _now()})
            merged_evidence["review_history"] = review_history[-20:]
            updated_at = _now()
            with self._connection:
                cursor = self._connection.execute(
                """
                UPDATE events
                SET review_status = ?,
                    lifecycle = CASE
                        WHEN ? = 'FALSE_POSITIVE' THEN 'CLOSED'
                        ELSE lifecycle
                    END,
                    updated_at = ?,
                    evidence_json = ?
                WHERE event_id = ?
                """,
                    (normalized, normalized, updated_at, _json(merged_evidence, "evidence"), value),
                )
            if cursor.rowcount == 0:
                raise DatabaseError(f"unknown event: {event_id}")
            event = self.get_event(value)
        assert event is not None
        return event

    def set_clip_path(
        self,
        event_id: str,
        clip_path: str | Path,
    ) -> dict[str, object]:
        """Associate an evidence clip path with one stored event."""

        self._ensure_open()
        value = str(event_id).strip()
        path = str(Path(clip_path).expanduser().resolve())
        if not path:
            raise DatabaseError("clip_path cannot be empty")
        with self._lock:
            with self._connection:
                cursor = self._connection.execute(
                """
                UPDATE events
                SET clip_path = ?, updated_at = ?
                WHERE event_id = ?
                """,
                    (path, _now(), value),
                )
            if cursor.rowcount == 0:
                raise DatabaseError(f"unknown event: {event_id}")
            event = self.get_event(value)
        assert event is not None
        return event

    def close(self) -> None:
        """Close the SQLite connection."""

        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def __enter__(self) -> EventDatabase:
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
