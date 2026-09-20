"""Create short, bounded evidence clips for stored incidents."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2

from ..storage.database import EventDatabase, EventStorageError
from ..video.source import VideoError, resolve_video_source
from ..video.writer import VideoWriter, VideoWriterError
from .evidence import file_sha256

DEFAULT_CLIP_DIRECTORY = Path("outputs") / "clips"


class ReplayError(RuntimeError):
    """Raised when an evidence clip cannot be generated."""


@dataclass(frozen=True)
class ReplayResult:
    """Metadata describing one generated or reused evidence clip."""

    event_id: str
    video_id: str
    clip_path: Path
    event_timestamp: float
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    frames_written: int
    fps: float
    reused: bool = False
    sha256: str | None = None

    @property
    def duration(self) -> float:
        """Return the requested clip duration in seconds."""

        return max(0.0, self.end_time - self.start_time)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly replay metadata."""

        return {
            "event_id": self.event_id,
            "video_id": self.video_id,
            "clip_path": str(self.clip_path),
            "event_timestamp": self.event_timestamp,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "frames_written": self.frames_written,
            "fps": self.fps,
            "duration": self.duration,
            "reused": self.reused,
            "sha256": self.sha256,
        }


def _safe_filename(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return normalized.strip("._") or "event"


def _metadata_value(
    capture: cv2.VideoCapture,
    property_id: int,
    fallback: object,
    label: str,
) -> float:
    value = float(capture.get(property_id))
    if not math.isfinite(value) or value <= 0.0:
        try:
            value = float(fallback)
        except (TypeError, ValueError):
            value = 0.0
    if not math.isfinite(value) or value <= 0.0:
        raise ReplayError(f"source video is missing valid {label} metadata")
    return value


def _browser_compatible_clip(path: Path) -> bool:
    """Return whether an existing clip uses a browser-friendly MP4 codec."""

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return False
        fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(
            chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4)
        ).lower()
        return fourcc in {"avc1", "h264", "avc3"}
    finally:
        capture.release()


class EvidenceReplay:
    """Seek to an incident window and encode only that short clip."""

    def __init__(
        self,
        database: EventDatabase,
        *,
        output_dir: str | Path = DEFAULT_CLIP_DIRECTORY,
        pre_seconds: float = 3.0,
        post_seconds: float = 3.0,
        codec: str = "avc1",
    ) -> None:
        if not isinstance(database, EventDatabase):
            raise TypeError("database must be an EventDatabase")
        for name, value in (
            ("pre_seconds", pre_seconds),
            ("post_seconds", post_seconds),
        ):
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ReplayError(f"{name} must be finite and non-negative")
        if not isinstance(codec, str) or len(codec) != 4:
            raise ReplayError("codec must be a four-character code")
        self.database = database
        self.output_dir = Path(output_dir).expanduser()
        self.pre_seconds = float(pre_seconds)
        self.post_seconds = float(post_seconds)
        self.codec = codec

    def _output_path(
        self,
        event: dict[str, object],
        output: str | Path | None,
    ) -> Path:
        if output is not None:
            return Path(output).expanduser().resolve()
        stored = event.get("clip_path")
        if stored:
            return Path(str(stored)).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return (
            self.output_dir
            / f"{_safe_filename(str(event['video_id']))}_"
            f"{_safe_filename(str(event['event_id']))}.mp4"
        ).resolve()

    def create_clip(
        self,
        event_id: str,
        *,
        output: str | Path | None = None,
        force: bool = False,
    ) -> ReplayResult:
        """Create or reuse a clip spanning event time minus/plus the window."""

        event = self.database.get_event(event_id)
        if event is None:
            raise ReplayError(f"unknown stored event: {event_id}")
        video_id = str(event["video_id"])
        video = self.database.get_video(video_id)
        if video is None:
            raise ReplayError(f"unknown stored video: {video_id}")
        try:
            source = resolve_video_source(str(video["source_path"]))
        except VideoError as exc:
            raise ReplayError(
                f"source video for event {event_id} is unavailable: {exc}"
            ) from exc

        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            capture.release()
            raise ReplayError(f"OpenCV could not open replay source: {source}")
        try:
            fps = _metadata_value(
                capture,
                cv2.CAP_PROP_FPS,
                video.get("fps"),
                "FPS",
            )
            total_frames = int(
                round(
                    _metadata_value(
                        capture,
                        cv2.CAP_PROP_FRAME_COUNT,
                        video.get("total_frames"),
                        "total frame count",
                    )
                )
            )
            width = int(
                round(
                    _metadata_value(
                        capture,
                        cv2.CAP_PROP_FRAME_WIDTH,
                        video.get("width"),
                        "frame width",
                    )
                )
            )
            height = int(
                round(
                    _metadata_value(
                        capture,
                        cv2.CAP_PROP_FRAME_HEIGHT,
                        video.get("height"),
                        "frame height",
                    )
                )
            )
            if total_frames <= 0:
                raise ReplayError("source video has no positive frame count")

            event_timestamp = float(event["timestamp"])
            source_duration = total_frames / fps
            bounded_event_time = max(0.0, min(event_timestamp, source_duration))
            start_time = max(0.0, bounded_event_time - self.pre_seconds)
            end_time = min(source_duration, bounded_event_time + self.post_seconds)
            start_frame = max(0, int(math.floor(start_time * fps + 1e-9)))
            end_frame = min(
                total_frames - 1,
                int(math.floor(end_time * fps + 1e-9)),
            )
            if end_frame < start_frame:
                raise ReplayError("calculated replay window contains no frames")

            output_path = self._output_path(event, output)
            reusable = (
                output_path.is_file()
                and output_path.stat().st_size > 0
                and (
                    self.codec != "avc1"
                    or _browser_compatible_clip(output_path)
                )
            )
            if not force and reusable:
                self.database.set_clip_path(event_id, output_path)
                digest = file_sha256(output_path)
                self.database.update_event_evidence(
                    event_id,
                    {
                        "replay_artifact": {
                            "kind": "evidence_clip",
                            "path": str(output_path),
                            "sha256": digest,
                            "bytes": output_path.stat().st_size,
                            "start_time": start_time,
                            "end_time": end_time,
                        }
                    },
                )
                return ReplayResult(
                    event_id=str(event_id),
                    video_id=video_id,
                    clip_path=output_path,
                    event_timestamp=event_timestamp,
                    start_time=start_time,
                    end_time=end_time,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    frames_written=end_frame - start_frame + 1,
                    fps=fps,
                    reused=True,
                    sha256=digest,
                )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frames_written = 0
            try:
                with VideoWriter(
                    output=output_path,
                    fps=fps,
                    width=width,
                    height=height,
                    codec=self.codec,
                ) as writer:
                    for _ in range(start_frame, end_frame + 1):
                        success, frame = capture.read()
                        if not success:
                            if frames_written == 0:
                                raise ReplayError(
                                    f"replay source ended before frame {start_frame}"
                                )
                            break
                        writer.write(frame)
                        frames_written = writer.frames_written
            except (cv2.error, VideoWriterError) as exc:
                raise ReplayError(
                    f"could not write replay clip for event {event_id}: {exc}"
                ) from exc
            if frames_written <= 0:
                raise ReplayError(f"no frames were written for event {event_id}")

            try:
                self.database.set_clip_path(event_id, output_path)
                digest = file_sha256(output_path)
                self.database.update_event_evidence(
                    event_id,
                    {
                        "replay_artifact": {
                            "kind": "evidence_clip",
                            "path": str(output_path),
                            "sha256": digest,
                            "bytes": output_path.stat().st_size,
                            "start_time": start_time,
                            "end_time": end_time,
                        }
                    },
                )
            except EventStorageError as exc:
                raise ReplayError(
                    f"clip was created but could not be associated with event "
                    f"{event_id}: {exc}"
                ) from exc
            return ReplayResult(
                event_id=str(event_id),
                video_id=video_id,
                clip_path=output_path,
                event_timestamp=event_timestamp,
                start_time=start_time,
                end_time=end_time,
                start_frame=start_frame,
                end_frame=end_frame,
                frames_written=frames_written,
                fps=fps,
                sha256=digest,
            )
        except cv2.error as exc:
            raise ReplayError(
                f"OpenCV failed while reading replay source {source}: {exc}"
            ) from exc
        finally:
            capture.release()

    replay = create_clip


IncidentReplay = EvidenceReplay
