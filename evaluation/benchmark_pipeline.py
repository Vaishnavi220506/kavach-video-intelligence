"""Measure the real KAVACH pipeline on a local video when assets are available."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import tracemalloc
from collections import Counter
from pathlib import Path

from kavach.assistant import GroundedAssistant, OllamaClient, OllamaError
from kavach.dashboard import analyse_video, video_file_sha256
from kavach.incidents import EvidenceReplay
from kavach.perception import WarehouseDetector
from kavach.storage import EventDatabase
from kavach.video import VideoReader


def run(
    video: str | Path,
    *,
    model_path: str | Path = "yolo11n.pt",
    detection_frames: int = 30,
    max_frames: int | None = 300,
    include_llm: bool = True,
) -> dict[str, object]:
    source = Path(video).expanduser().resolve()
    model = Path(model_path).expanduser().resolve()
    tracemalloc.start()
    detector = WarehouseDetector(
        model_path=model,
        confidence=0.25,
        device="auto",
        allowed_classes=(),
    )
    with VideoReader(source) as reader:
        detection_benchmark = detector.benchmark(
            (packet.frame for packet in reader),
            max_frames=detection_frames,
            warmup_frames=1,
        )
    _, detection_peak = tracemalloc.get_traced_memory()

    with tempfile.TemporaryDirectory(prefix="kavach_benchmark_") as temp_dir:
        root = Path(temp_dir)
        database = EventDatabase(root / "events.sqlite3")
        video_id = "video-" + video_file_sha256(source)[:16]
        started = time.perf_counter()
        result = analyse_video(
            source,
            video_id=video_id,
            output=root / "processed.mp4",
            detector=detector,
            database=database,
            max_frames=max_frames,
        )
        wall_clock_seconds = time.perf_counter() - started
        _, end_to_end_peak = tracemalloc.get_traced_memory()

        llm_result: dict[str, object] | None = None
        if include_llm:
            assistant = GroundedAssistant(
                database,
                ollama_client=OllamaClient(),
                replay=EvidenceReplay(database, output_dir=root / "clips"),
            )
            try:
                llm_result = assistant.benchmark(
                    "Summarize this video",
                    video_id=video_id,
                    repeats=1,
                ).to_dict()
            except OllamaError as exc:
                llm_result = {"available": False, "error": str(exc)}
        database.close()
    tracemalloc.stop()
    return {
        "video": str(video),
        "model": result.detector_model,
        "device": result.device,
        "source_metadata": {
            "fps": result.fps,
            "width": result.width,
            "height": result.height,
            "total_frames": result.total_frames,
            "duration_seconds": result.duration,
        },
        "object_detection": {
            "frames_benchmarked": detection_benchmark.frames,
            "average_inference_time_ms": detection_benchmark.average_inference_time_ms,
            "approximate_fps": detection_benchmark.approximate_fps,
        },
        "end_to_end": {
            "frames_processed": result.frames_processed,
            "frame_limit": max_frames,
            "processing_seconds": result.processing_seconds,
            "wall_clock_seconds": wall_clock_seconds,
            "processing_fps": result.processing_fps,
            "stored_events": len(result.events),
            "unique_objects": dict(result.unique_objects),
            "events_by_behaviour": dict(
                Counter(str(event.get("behaviour", "UNKNOWN")) for event in result.events)
            ),
        },
        "memory": {
            "python_tracemalloc_peak_bytes_after_detection": detection_peak,
            "python_tracemalloc_peak_bytes_after_end_to_end": end_to_end_peak,
            "note": "Python allocation tracing only; excludes native Torch/OpenCV allocation and full process RSS.",
        },
        "llm": llm_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=Path("data/videos/sample_warehouse.mp4"))
    parser.add_argument("--model", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument("--detection-frames", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reports/pipeline_benchmark.json"))
    args = parser.parse_args()
    result = run(
        args.video,
        model_path=args.model,
        detection_frames=args.detection_frames,
        max_frames=args.max_frames,
        include_llm=not args.skip_llm,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
