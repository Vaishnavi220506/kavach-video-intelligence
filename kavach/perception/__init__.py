"""KAVACH Module 2 warehouse object perception."""

from .classes import (
    STANDARD_COCO_WAREHOUSE_CLASSES,
    WAREHOUSE_TRAINING_CLASSES,
    WAREHOUSE_VOCABULARY,
    class_ids_for_names,
    normalize_class_name,
)
from .dataset import DatasetIssue, DatasetReport, validate_yolo_dataset

__all__ = [
    "Detection",
    "DetectionError",
    "ByteTrackConfig",
    "InferenceBenchmark",
    "InvalidFrameError",
    "InvalidTimestampError",
    "MultiObjectTracker",
    "ModelLoadError",
    "STANDARD_COCO_WAREHOUSE_CLASSES",
    "WAREHOUSE_TRAINING_CLASSES",
    "WAREHOUSE_VOCABULARY",
    "WarehouseDetector",
    "class_ids_for_names",
    "draw_detections",
    "draw_tracked_objects",
    "normalize_class_name",
    "resolve_device",
    "TrackedObject",
    "TrackingError",
    "ClassMetrics",
    "DetectionEvaluation",
    "EvaluationError",
    "evaluate_checkpoint",
    "DatasetIssue",
    "DatasetReport",
    "validate_yolo_dataset",
]


def __getattr__(name: str):
    """Load Torch/Ultralytics-backed exports only when they are used."""

    groups = {
        "Detection": (".detector", "Detection"),
        "DetectionError": (".detector", "DetectionError"),
        "InferenceBenchmark": (".detector", "InferenceBenchmark"),
        "InvalidFrameError": (".detector", "InvalidFrameError"),
        "ModelLoadError": (".detector", "ModelLoadError"),
        "WarehouseDetector": (".detector", "WarehouseDetector"),
        "resolve_device": (".detector", "resolve_device"),
        "ClassMetrics": (".evaluation", "ClassMetrics"),
        "DetectionEvaluation": (".evaluation", "DetectionEvaluation"),
        "EvaluationError": (".evaluation", "EvaluationError"),
        "evaluate_checkpoint": (".evaluation", "evaluate_checkpoint"),
        "ByteTrackConfig": (".tracker", "ByteTrackConfig"),
        "InvalidTimestampError": (".tracker", "InvalidTimestampError"),
        "MultiObjectTracker": (".tracker", "MultiObjectTracker"),
        "TrackedObject": (".tracker", "TrackedObject"),
        "TrackingError": (".tracker", "TrackingError"),
        "draw_detections": (".visualization", "draw_detections"),
        "draw_tracked_objects": (".visualization", "draw_tracked_objects"),
    }
    try:
        module_name, attribute = groups[name]
    except KeyError as exc:
        raise AttributeError(f"module 'kavach.perception' has no attribute {name!r}") from exc
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
