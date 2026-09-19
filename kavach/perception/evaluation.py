"""Ultralytics validation adapter for camera-specific warehouse datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any

import yaml

from .classes import WAREHOUSE_TRAINING_CLASSES
from .dataset import validate_yolo_dataset
from .detector import resolve_device


class EvaluationError(RuntimeError):
    """Raised when a detector evaluation cannot be run honestly."""


@dataclass(frozen=True)
class ClassMetrics:
    class_name: str
    precision: float
    recall: float
    map50: float
    map50_95: float

    def to_dict(self) -> dict[str, object]:
        return {
            "class_name": self.class_name,
            "precision": self.precision,
            "recall": self.recall,
            "mAP50": self.map50,
            "mAP50_95": self.map50_95,
        }


@dataclass(frozen=True)
class DetectionEvaluation:
    model: str
    dataset: str
    split: str
    device: str
    classes: tuple[ClassMetrics, ...]
    mean_precision: float
    mean_recall: float
    map50: float
    map50_95: float

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "dataset": self.dataset,
            "split": self.split,
            "device": self.device,
            "classes": [item.to_dict() for item in self.classes],
            "mean_precision": self.mean_precision,
            "mean_recall": self.mean_recall,
            "mAP50": self.map50,
            "mAP50_95": self.map50_95,
        }


def _values(metrics: Any, attribute: str) -> list[float]:
    value = getattr(metrics, attribute, None)
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(item) for item in value]


def evaluate_checkpoint(
    model_path: str | Path,
    dataset_yaml: str | Path,
    *,
    split: str = "val",
    device: str | None = None,
    imgsz: int = 640,
    conf: float | None = None,
    expected_classes: tuple[str, ...] | None = WAREHOUSE_TRAINING_CLASSES,
) -> DetectionEvaluation:
    """Run official Ultralytics validation and preserve per-class metrics.

    The function refuses missing datasets. It never turns training-log values
    into a claim about a different dataset or a different model checkpoint.
    """

    model_file = Path(model_path).expanduser().resolve()
    dataset_file = Path(dataset_yaml).expanduser().resolve()
    if not model_file.is_file():
        raise EvaluationError(f"model checkpoint was not found: {model_file}")
    if not dataset_file.is_file():
        raise EvaluationError(
            f"labeled dataset YAML was not found: {dataset_file}. "
            "Add a camera-specific YOLO dataset before reporting accuracy."
        )
    dataset_report = validate_yolo_dataset(
        dataset_file,
        expected_classes=expected_classes,
        require_all_expected_classes=expected_classes is not None,
    )
    if not dataset_report.is_valid:
        details = "\n".join(
            f"- {issue.code}: {issue.message}" for issue in dataset_report.issues[:8]
        )
        raise EvaluationError(
            "The dataset failed validation; no accuracy was reported.\n" + details
        )
    try:
        from ultralytics import YOLO

        model = YOLO(str(model_file))
        # Ultralytics interprets relative ``path`` values against its global
        # datasets directory rather than beside the YAML. Give validation an
        # ephemeral absolute-path copy so the repository remains portable.
        config = yaml.safe_load(dataset_file.read_text(encoding="utf-8"))
        config["path"] = str(dataset_report.dataset_root)
        with tempfile.TemporaryDirectory(prefix="kavach-eval-") as temp_dir:
            effective_dataset = Path(temp_dir) / dataset_file.name
            effective_dataset.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            kwargs: dict[str, object] = {
                "data": str(effective_dataset),
                "split": split,
                "device": resolve_device(device),
                "imgsz": int(imgsz),
                "verbose": False,
                "plots": False,
            }
            if conf is not None:
                kwargs["conf"] = float(conf)
            result = model.val(**kwargs)
    except Exception as exc:
        raise EvaluationError(f"Ultralytics validation failed: {exc}") from exc

    box = getattr(result, "box", None)
    if box is None:
        raise EvaluationError("Ultralytics validation returned no box metrics")
    names = getattr(result, "names", None) or getattr(model, "names", {})
    if isinstance(names, dict):
        ordered_names = [str(names[key]) for key in sorted(names)]
    else:
        ordered_names = [str(item) for item in names]
    precision = _values(box, "p")
    recall = _values(box, "r")
    map50 = _values(box, "ap50")
    map50_95 = _values(box, "ap")
    class_count = min(len(ordered_names), len(precision), len(recall), len(map50), len(map50_95))
    classes = tuple(
        ClassMetrics(
            ordered_names[index],
            precision[index],
            recall[index],
            map50[index],
            map50_95[index],
        )
        for index in range(class_count)
    )
    return DetectionEvaluation(
        model=str(model_file),
        dataset=str(dataset_file),
        split=split,
        device=resolve_device(device),
        classes=classes,
        mean_precision=float(getattr(box, "mp", 0.0)),
        mean_recall=float(getattr(box, "mr", 0.0)),
        map50=float(getattr(box, "map50", 0.0)),
        map50_95=float(getattr(box, "map", 0.0)),
    )
