"""Extract reviewable frames from one or more warehouse videos.

This command only creates candidate images and a provenance manifest. It does
not create labels and it must not be treated as an auto-annotation tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Iterable

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kavach.video import VideoError, VideoReader


def _safe_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_") or "video"
    digest = hashlib.sha1(
        f"{path.resolve()}|{path.stat().st_size}|{path.stat().st_mtime_ns}".encode(
            "utf-8"
        )
    ).hexdigest()[:8]
    return f"{stem}_{digest}"


def extract_frames(
    videos: Iterable[str | Path],
    *,
    output_dir: str | Path,
    split: str = "train",
    every_seconds: float = 1.0,
    jpeg_quality: int = 95,
    max_frames_per_video: int | None = None,
) -> dict[str, object]:
    """Extract deterministic, timestamped candidate frames.

    Frames are assigned to one split explicitly. The caller should reserve
    complete videos for validation rather than randomly splitting adjacent
    frames from the same source.
    """

    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    if every_seconds <= 0:
        raise ValueError("every_seconds must be positive")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")
    if max_frames_per_video is not None and max_frames_per_video <= 0:
        raise ValueError("max_frames_per_video must be positive")

    destination = Path(output_dir).expanduser().resolve()
    image_dir = destination / "images" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "frame_manifest.jsonl"
    records: list[dict[str, object]] = []

    for video_value in videos:
        source = Path(video_value).expanduser().resolve()
        source_key = _safe_stem(source)
        frames_from_video = 0
        next_sample = 0.0
        try:
            with VideoReader(source) as reader:
                metadata = {
                    "fps": reader.fps,
                    "width": reader.width,
                    "height": reader.height,
                    "total_frames": reader.total_frames,
                    "duration_seconds": reader.duration,
                }
                for packet in reader:
                    if packet.timestamp + 1e-9 < next_sample:
                        continue
                    image_name = f"{source_key}__f{packet.frame_number:07d}.jpg"
                    image_path = image_dir / image_name
                    written = cv2.imwrite(
                        str(image_path),
                        packet.frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
                    )
                    if not written:
                        raise RuntimeError(f"OpenCV could not write {image_path}")
                    records.append(
                        {
                            "image": str(image_path.relative_to(destination)),
                            "source": str(source),
                            "frame_number": packet.frame_number,
                            "timestamp_seconds": packet.timestamp,
                            "split": split,
                            "label_status": "unannotated",
                            "metadata": metadata,
                        }
                    )
                    frames_from_video += 1
                    next_sample += every_seconds
                    if (
                        max_frames_per_video is not None
                        and frames_from_video >= max_frames_per_video
                    ):
                        break
        except VideoError:
            raise
        if frames_from_video == 0:
            raise RuntimeError(f"No frames were extracted from {source}")

    with manifest_path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    return {
        "output_dir": str(destination),
        "split": split,
        "frames_extracted": len(records),
        "manifest": str(manifest_path),
        "label_status": "unannotated; manual bounding-box labels are still required",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video", action="append", required=True, help="Local video; repeat for multiple sources."
    )
    parser.add_argument("--output", type=Path, default=Path("data/warehouse_dataset"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--every-seconds", type=float, default=1.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-frames-per-video", type=int, default=None)
    args = parser.parse_args()
    try:
        result = extract_frames(
            args.video,
            output_dir=args.output,
            split=args.split,
            every_seconds=args.every_seconds,
            jpeg_quality=args.jpeg_quality,
            max_frames_per_video=args.max_frames_per_video,
        )
    except (OSError, RuntimeError, ValueError, VideoError) as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
