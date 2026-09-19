"""Safe materialization helpers for dashboard video sources.

The dashboard accepts local uploads and direct HTTP(S) video URLs. It does
not claim to download arbitrary web pages or video-platform URLs.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re
import tempfile
from typing import BinaryIO
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


MAX_VIDEO_BYTES = 1_073_741_824
DOWNLOAD_TIMEOUT_SECONDS = 30.0
CHUNK_SIZE = 1024 * 1024
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".webm"}
ALLOWED_CONTENT_TYPES = {
    "application/octet-stream",
    "binary/octet-stream",
}


class VideoSourceError(RuntimeError):
    """Raised when a dashboard source cannot be safely materialized."""


def validate_direct_video_url(url: str) -> str:
    """Validate and normalize a direct HTTP(S) URL.

    A syntactically valid URL is not proof that it points to a video. The
    downloader checks the response content type and the OpenCV reader later
    verifies that the bytes are a supported video stream.
    """

    value = str(url).strip()
    if not value:
        raise VideoSourceError("video URL cannot be empty")
    if len(value) > 2048:
        raise VideoSourceError("video URL is too long")
    parsed = urlparse(value)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise VideoSourceError("only direct HTTP(S) video URLs are supported")
    if not parsed.hostname:
        raise VideoSourceError("video URL must include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise VideoSourceError("video URLs with embedded credentials are not allowed")
    return value


def _content_length(response: object) -> int | None:
    headers = getattr(response, "headers", None)
    value: object = None
    if headers is not None:
        try:
            value = headers.get("Content-Length")
        except AttributeError:
            value = None
    if value is None and hasattr(response, "getheader"):
        value = response.getheader("Content-Length")  # type: ignore[attr-defined]
    if value in (None, ""):
        return None
    try:
        length = int(value)
    except (TypeError, ValueError) as exc:
        raise VideoSourceError("source returned an invalid Content-Length") from exc
    if length < 0:
        raise VideoSourceError("source returned a negative Content-Length")
    return length


def _content_type(response: object) -> str:
    headers = getattr(response, "headers", None)
    if headers is None:
        return ""
    try:
        value = headers.get("Content-Type", "")
    except AttributeError:
        return ""
    return str(value).split(";", 1)[0].strip().lower()


def _suffix_from_url(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix in VIDEO_SUFFIXES else ".mp4"


def _safe_name(name: str, fallback: str = "upload") -> str:
    value = Path(str(name)).name
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return value or fallback


def _write_stream(
    response: BinaryIO,
    target: Path,
    *,
    max_bytes: int,
) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    total = 0
    try:
        with partial.open("wb") as handle:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise VideoSourceError(
                        f"video exceeds the {max_bytes} byte download limit"
                    )
                handle.write(chunk)
        if total == 0:
            raise VideoSourceError("video source returned an empty response")
        partial.replace(target)
        return target
    except VideoSourceError:
        partial.unlink(missing_ok=True)
        raise
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise VideoSourceError(f"could not write temporary video: {exc}") from exc


def download_video_url(
    url: str,
    *,
    output_dir: str | Path | None = None,
    timeout_seconds: float = DOWNLOAD_TIMEOUT_SECONDS,
    max_bytes: int = MAX_VIDEO_BYTES,
) -> Path:
    """Download one direct video URL into a bounded temporary file."""

    normalized_url = validate_direct_video_url(url)
    if timeout_seconds <= 0.0:
        raise VideoSourceError("timeout_seconds must be positive")
    if max_bytes <= 0:
        raise VideoSourceError("max_bytes must be positive")

    directory = (
        Path(output_dir).expanduser()
        if output_dir is not None
        else Path(tempfile.mkdtemp(prefix="kavach_url_"))
    )
    filename = "download_" + sha256(normalized_url.encode("utf-8")).hexdigest()[:12]
    target = directory / (filename + _suffix_from_url(normalized_url))
    request = Request(
        normalized_url,
        headers={
            "Accept": "video/*,application/octet-stream;q=0.9,*/*;q=0.1",
            "User-Agent": "KAVACH/1.0 direct-video-source",
        },
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # nosec B310
            status = getattr(response, "status", None)
            if status is not None and int(status) >= 400:
                raise VideoSourceError(f"video URL returned HTTP status {status}")
            length = _content_length(response)
            if length is not None and length > max_bytes:
                raise VideoSourceError(
                    f"video exceeds the {max_bytes} byte download limit"
                )
            content_type = _content_type(response)
            if content_type and not (
                content_type.startswith("video/")
                or content_type in ALLOWED_CONTENT_TYPES
            ):
                raise VideoSourceError(
                    "URL did not return a direct video response "
                    f"(Content-Type: {content_type})"
                )
            return _write_stream(response, target, max_bytes=max_bytes)
    except VideoSourceError:
        raise
    except HTTPError as exc:
        raise VideoSourceError(
            f"could not download video URL (HTTP {exc.code})"
        ) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise VideoSourceError(f"could not download video URL: {exc}") from exc


def persist_uploaded_video(
    filename: str,
    content: bytes,
    *,
    output_dir: str | Path,
    max_bytes: int = MAX_VIDEO_BYTES,
) -> Path:
    """Persist one bounded Streamlit upload in a session-owned directory."""

    if not isinstance(content, bytes) or not content:
        raise VideoSourceError("uploaded video is empty")
    if len(content) > max_bytes:
        raise VideoSourceError(f"uploaded video exceeds the {max_bytes} byte limit")
    directory = Path(output_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_name(filename, "upload.mp4")
    suffix = Path(safe_name).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        safe_name = Path(safe_name).stem + ".mp4"
    digest = sha256(content).hexdigest()[:12]
    target = directory / f"upload_{digest}_{safe_name}"
    if not target.is_file():
        try:
            target.write_bytes(content)
        except OSError as exc:
            raise VideoSourceError(f"could not write uploaded video: {exc}") from exc
    return target


def video_file_sha256(path: str | Path) -> str:
    """Return a stable content hash used as the dashboard video ID."""

    source = Path(path).expanduser()
    digest = sha256()
    try:
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
                digest.update(chunk)
    except OSError as exc:
        raise VideoSourceError(f"could not hash video source: {exc}") from exc
    return digest.hexdigest()
