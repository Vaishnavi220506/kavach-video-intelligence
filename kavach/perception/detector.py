"""KAVACH object-detection boundary for Ultralytics models."""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO

from .classes import (
    STANDARD_COCO_WAREHOUSE_CLASSES,
    WAREHOUSE_VOCABULARY,
    class_ids_for_names,
    model_class_names,
    normalize_class_name,
)

BoundingBox = tuple[float, float, float, float]
Point = tuple[float, float]


class DetectionError(RuntimeError):
    """Base class for expected perception errors."""


class InvalidFrameError(DetectionError):
    """Raised when a caller provides an empty or invalid image array."""


class ModelLoadError(DetectionError):
    """Raised when an Ultralytics model cannot be loaded."""


@dataclass(frozen=True)
class Detection:
    """One frame-local object detection.

    Coordinates are pixel coordinates in the input frame. ``bbox`` follows
    the Ultralytics/XYXY convention: left, top, right, bottom. This object
    intentionally has no persistent ID; tracking belongs to a later module.
    """

    class_name: str
    class_id: int
    confidence: float
    bbox: BoundingBox
    center: Point


@dataclass(frozen=True)
class InferenceBenchmark:
    """Summary of timed, frame-local detector inference."""

    frames: int
    average_inference_time_ms: float
    approximate_fps: float
    device: str
    model: str


def resolve_device(requested: str | None) -> str:
    """Resolve automatic device selection and safely fall back to CPU."""

    cuda_available = bool(torch.cuda.is_available())
    if requested is None or requested.casefold() == "auto":
        return "cuda:0" if cuda_available else "cpu"

    requested = requested.strip()
    if requested.casefold().startswith("cuda") and not cuda_available:
        return "cpu"
    return requested


def _as_numpy(value: Any) -> np.ndarray:
    """Convert an Ultralytics tensor-like value to a NumPy array."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class WarehouseDetector:
    """Load one Ultralytics model and expose stable frame-local detections.

    The default filter is deliberately conservative for the repository's
    standard YOLO11n COCO checkpoint: only the exact ``person`` and ``truck``
    classes overlap the requested warehouse vocabulary. Pass
    ``allowed_classes`` for a custom model, or use ``from_yolo_world`` for an
    explicitly evaluated open-vocabulary experiment.
    """

    def __init__(
        self,
        model_path: str | Path = "yolo11n.pt",
        confidence: float = 0.25,
        device: str | None = None,
        allowed_classes: Iterable[str] | None = None,
        *,
        model: Any | None = None,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")

        self.model_path = str(model_path)
        self.confidence = float(confidence)
        self.device = resolve_device(device)

        if allowed_classes is None:
            selected_classes: tuple[str, ...] = STANDARD_COCO_WAREHOUSE_CLASSES
        else:
            selected_classes = tuple(str(name) for name in allowed_classes)
        self.allowed_classes = selected_classes
        self._filter_enabled = bool(selected_classes)

        if model is not None:
            self.model = model
        else:
            try:
                self.model = YOLO(self.model_path)
            except Exception as exc:  # Ultralytics raises several model-specific types.
                raise ModelLoadError(
                    f"Unable to load Ultralytics model '{self.model_path}'."
                ) from exc

        self._names = model_class_names(getattr(self.model, "names", {}))
        self._allowed_class_ids = class_ids_for_names(
            self._names, self.allowed_classes
        )
        self._allowed_class_names = {
            normalize_class_name(name) for name in self.allowed_classes
        }

    @classmethod
    def from_yolo_world(
        cls,
        model_path: str | Path = "yolov8s-worldv2.pt",
        vocabulary: Iterable[str] = WAREHOUSE_VOCABULARY,
        confidence: float = 0.25,
        device: str | None = None,
    ) -> WarehouseDetector:
        """Create an optional YOLO-World detector with explicit text prompts.

        YOLO-World needs an additional CLIP text encoder and its weights. It
        is intentionally opt-in and is not the default detector for Module 2.
        The caller must validate these prompts on labeled warehouse data.
        """

        prompts = tuple(str(name) for name in vocabulary)
        try:
            from ultralytics import YOLOWorld

            model = YOLOWorld(str(model_path))
            model.set_classes(list(prompts))
        except Exception as exc:
            raise ModelLoadError(
                "Unable to initialize the optional YOLO-World detector. "
                "Check the YOLO-World/CLIP dependencies and model weights."
            ) from exc

        return cls(
            model_path=model_path,
            confidence=confidence,
            device=device,
            allowed_classes=prompts,
            model=model,
        )

    @property
    def model_name(self) -> str:
        """Human-readable model filename or identifier."""

        return Path(self.model_path).name or self.model_path

    @property
    def class_names(self) -> dict[int, str]:
        """The names exposed by the loaded model."""

        return dict(self._names)

    def _predict(self, frame: np.ndarray) -> Any:
        kwargs: dict[str, Any] = {
            "conf": self.confidence,
            "device": self.device,
            "verbose": False,
        }
        if self._filter_enabled and self._allowed_class_ids:
            kwargs["classes"] = list(self._allowed_class_ids)
        return self.model.predict(source=frame, **kwargs)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run one frame of detection and return standardized results."""

        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise InvalidFrameError("frame must be a non-empty NumPy array")
        if frame.ndim not in (2, 3):
            raise InvalidFrameError(
                f"frame must have 2 or 3 dimensions, got shape {frame.shape}"
            )

        results = self._predict(frame)
        if not results:
            return []

        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return []

        xyxy = _as_numpy(getattr(boxes, "xyxy", np.empty((0, 4)))).reshape(-1, 4)
        confidences = _as_numpy(getattr(boxes, "conf", np.empty(0))).reshape(-1)
        class_ids = _as_numpy(getattr(boxes, "cls", np.empty(0))).reshape(-1)
        detections: list[Detection] = []

        for box, confidence, class_id in zip(xyxy, confidences, class_ids, strict=False):
            numeric_class_id = int(class_id)
            class_name = self._names.get(numeric_class_id, str(numeric_class_id))
            if self._filter_enabled and normalize_class_name(class_name) not in self._allowed_class_names:
                continue

            x1, y1, x2, y2 = (float(value) for value in box)
            detections.append(
                Detection(
                    class_name=class_name,
                    class_id=numeric_class_id,
                    confidence=float(confidence),
                    bbox=(x1, y1, x2, y2),
                    center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                )
            )

        return detections

    def benchmark(
        self,
        frames: Iterable[np.ndarray],
        *,
        max_frames: int = 30,
        warmup_frames: int = 1,
    ) -> InferenceBenchmark:
        """Time frame-local inference over an iterable of image arrays."""

        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if warmup_frames < 0:
            raise ValueError("warmup_frames cannot be negative")

        iterator: Iterator[np.ndarray] = iter(frames)
        for _ in range(warmup_frames):
            try:
                self.detect(next(iterator))
            except StopIteration:
                break

        durations: list[float] = []
        while len(durations) < max_frames:
            try:
                frame = next(iterator)
            except StopIteration:
                break
            start = time.perf_counter()
            self.detect(frame)
            durations.append(time.perf_counter() - start)

        if not durations:
            raise ValueError("frames did not contain any frame to benchmark")

        average_seconds = sum(durations) / len(durations)
        return InferenceBenchmark(
            frames=len(durations),
            average_inference_time_ms=average_seconds * 1000.0,
            approximate_fps=1.0 / average_seconds,
            device=self.device,
            model=self.model_name,
        )

