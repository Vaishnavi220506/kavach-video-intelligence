"""Evidence snapshots and integrity verification for KAVACH incidents."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class EvidenceError(RuntimeError):
    """Raised when an evidence artifact cannot be created or verified."""


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""

    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise EvidenceError(f"evidence file does not exist: {candidate}")
    digest = hashlib.sha256()
    try:
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise EvidenceError(f"could not hash evidence file {candidate}: {exc}") from exc
    return digest.hexdigest()


@dataclass(frozen=True)
class EvidenceArtifact:
    """A file-backed snapshot with enough metadata to verify it later."""

    kind: str
    path: str
    sha256: str
    bytes: int
    timestamp: float
    frame_number: int
    mime_type: str = "image/jpeg"

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "mime_type": self.mime_type,
        }


def _safe(value: object) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return normalized.strip("._") or "event"


class EvidenceStore:
    """Write bounded incident snapshots below one controlled output directory."""

    def __init__(
        self,
        output_dir: str | Path = Path("outputs") / "evidence" / "snapshots",
        *,
        jpeg_quality: int = 92,
    ) -> None:
        if not 1 <= int(jpeg_quality) <= 100:
            raise EvidenceError("jpeg_quality must be between 1 and 100")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.jpeg_quality = int(jpeg_quality)

    def capture_snapshot(
        self,
        video_id: str,
        event_id: str,
        frame: np.ndarray,
        *,
        timestamp: float,
        frame_number: int,
    ) -> EvidenceArtifact:
        """Write one JPEG snapshot and return its verifiable artifact record."""

        if not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim not in (2, 3):
            raise EvidenceError("snapshot frame must be a non-empty image array")
        numeric_timestamp = float(timestamp)
        if not math.isfinite(numeric_timestamp) or numeric_timestamp < 0.0:
            raise EvidenceError("snapshot timestamp must be finite and non-negative")
        numeric_frame = int(frame_number)
        if numeric_frame < 0:
            raise EvidenceError("snapshot frame_number must be non-negative")

        directory = self.output_dir / _safe(video_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{_safe(event_id)}_frame_{numeric_frame:08d}.jpg"
        try:
            success = bool(
                cv2.imwrite(
                    str(path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                )
            )
        except cv2.error as exc:
            raise EvidenceError(f"could not encode evidence snapshot: {exc}") from exc
        if not success or not path.is_file() or path.stat().st_size <= 0:
            raise EvidenceError(f"OpenCV did not create evidence snapshot: {path}")
        return EvidenceArtifact(
            kind="incident_snapshot",
            path=str(path.resolve()),
            sha256=file_sha256(path),
            bytes=path.stat().st_size,
            timestamp=numeric_timestamp,
            frame_number=numeric_frame,
        )

    @staticmethod
    def verify(artifact: EvidenceArtifact | dict[str, object]) -> bool:
        """Return whether the artifact still matches its recorded digest."""

        if isinstance(artifact, EvidenceArtifact):
            path, expected = artifact.path, artifact.sha256
        elif isinstance(artifact, dict):
            path, expected = artifact.get("path"), artifact.get("sha256")
        else:
            raise EvidenceError("artifact must be EvidenceArtifact or mapping")
        if not path or not expected:
            raise EvidenceError("artifact must include path and sha256")
        try:
            return file_sha256(str(path)).casefold() == str(expected).casefold()
        except EvidenceError:
            return False

