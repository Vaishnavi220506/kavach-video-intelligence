"""Explain per-image detector errors on a labelled YOLO split."""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kavach.perception.dataset import validate_yolo_dataset
from kavach.perception.detector import resolve_device


@dataclass(frozen=True)
class Box:
    class_id: int
    class_name: str
    bbox: tuple[float, float, float, float]
    confidence: float | None = None


def box_iou(first: Iterable[float], second: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in first)
    bx1, by1, bx2, by2 = (float(value) for value in second)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def match_boxes(
    ground_truth: list[Box],
    predictions: list[Box],
    *,
    iou_threshold: float = 0.5,
) -> tuple[Counter[str], list[dict[str, object]]]:
    """Greedily match predictions and return counts plus explainable cases."""

    counts: Counter[str] = Counter()
    cases: list[dict[str, object]] = []
    matched: set[int] = set()
    ordered_predictions = sorted(
        predictions,
        key=lambda item: item.confidence if item.confidence is not None else 0.0,
        reverse=True,
    )
    for prediction in ordered_predictions:
        same_class = [
            (index, box_iou(prediction.bbox, truth.bbox))
            for index, truth in enumerate(ground_truth)
            if index not in matched and truth.class_id == prediction.class_id
        ]
        same_class.sort(key=lambda item: item[1], reverse=True)
        if same_class and same_class[0][1] >= iou_threshold:
            index, overlap = same_class[0]
            matched.add(index)
            counts[f"tp:{prediction.class_name}"] += 1
            continue

        any_class = [
            (index, box_iou(prediction.bbox, truth.bbox))
            for index, truth in enumerate(ground_truth)
            if index not in matched
        ]
        any_class.sort(key=lambda item: item[1], reverse=True)
        if any_class and any_class[0][1] >= iou_threshold:
            index, overlap = any_class[0]
            truth = ground_truth[index]
            matched.add(index)
            counts[f"fp:{prediction.class_name}"] += 1
            counts[f"fn:{truth.class_name}"] += 1
            cases.append(
                {
                    "reason": "class_confusion",
                    "predicted": prediction.class_name,
                    "expected": truth.class_name,
                    "confidence": prediction.confidence,
                    "iou": overlap,
                    "bbox": list(prediction.bbox),
                }
            )
        else:
            counts[f"fp:{prediction.class_name}"] += 1
            cases.append(
                {
                    "reason": "false_positive",
                    "predicted": prediction.class_name,
                    "expected": None,
                    "confidence": prediction.confidence,
                    "iou": 0.0 if not any_class else any_class[0][1],
                    "bbox": list(prediction.bbox),
                }
            )

    for index, truth in enumerate(ground_truth):
        if index in matched:
            continue
        counts[f"fn:{truth.class_name}"] += 1
        cases.append(
            {
                "reason": "false_negative",
                "predicted": None,
                "expected": truth.class_name,
                "confidence": None,
                "iou": 0.0,
                "bbox": list(truth.bbox),
            }
        )
    return counts, cases


def _label_dir(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].casefold() == "images":
            parts[index] = "labels"
            return Path(*parts)
    return image_dir.parent / "labels" / image_dir.name


def _image_dirs(dataset_yaml: Path, split: str, dataset_root: Path) -> tuple[Path, ...]:
    config = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    values = config[split] if isinstance(config[split], list) else [config[split]]
    result: list[Path] = []
    for value in values:
        path = Path(str(value)).expanduser()
        result.append((path if path.is_absolute() else dataset_root / path).resolve())
    return tuple(result)


def _ground_truth(image_path: Path, label_path: Path, names: tuple[str, ...]) -> list[Box]:
    from PIL import Image

    with Image.open(image_path) as image:
        width, height = image.size
    boxes: list[Box] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        class_id = int(fields[0])
        center_x, center_y, box_width, box_height = (float(value) for value in fields[1:])
        boxes.append(
            Box(
                class_id,
                names[class_id],
                (
                    (center_x - box_width / 2.0) * width,
                    (center_y - box_height / 2.0) * height,
                    (center_x + box_width / 2.0) * width,
                    (center_y + box_height / 2.0) * height,
                ),
            )
        )
    return boxes


def evaluate_confusion_cases(
    model_path: str | Path,
    dataset_yaml: str | Path,
    *,
    split: str = "val",
    device: str | None = None,
    imgsz: int = 640,
    conf: float = 0.25,
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    """Run predictions and explain TP/FP/FN and class-confusion cases."""

    dataset_file = Path(dataset_yaml).expanduser().resolve()
    report = validate_yolo_dataset(
        dataset_file,
        expected_classes=None,
        require_all_expected_classes=False,
    )
    if not report.is_valid:
        raise ValueError(report.format_text())
    config = yaml.safe_load(dataset_file.read_text(encoding="utf-8"))
    names = tuple(str(name) for name in config["names"] if not isinstance(config["names"], dict)) if isinstance(config["names"], list) else tuple(str(config["names"][key]) for key in sorted(config["names"], key=lambda value: int(value)))
    image_paths = [
        image_path
        for image_dir in _image_dirs(dataset_file, split, report.dataset_root)
        for image_path in sorted(image_dir.iterdir())
        if image_path.is_file() and image_path.suffix.casefold() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]
    if not image_paths:
        raise ValueError(f"no images found for split {split!r}")

    from ultralytics import YOLO

    model = YOLO(str(Path(model_path).expanduser().resolve()))
    results = model.predict(
        source=[str(path) for path in image_paths],
        device=resolve_device(device),
        imgsz=int(imgsz),
        conf=float(conf),
        verbose=False,
    )
    totals: Counter[str] = Counter()
    all_cases: list[dict[str, object]] = []
    per_image: list[dict[str, object]] = []
    for image_path, result in zip(image_paths, results, strict=False):
        labels = _ground_truth(
            image_path,
            _label_dir(image_path.parent) / f"{image_path.stem}.txt",
            names,
        )
        boxes = getattr(result, "boxes", None)
        predicted: list[Box] = []
        if boxes is not None and len(boxes):
            coordinates = boxes.xyxy.cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            class_ids = boxes.cls.cpu().tolist()
            result_names = getattr(result, "names", None) or names
            for coordinate, confidence, class_id_value in zip(coordinates, confidences, class_ids, strict=False):
                class_id = int(class_id_value)
                # Indexing works the same for a list of names and a dict
                # keyed by class id, so no branch is needed here.
                class_name = result_names[class_id]
                predicted.append(Box(class_id, str(class_name), tuple(float(v) for v in coordinate), float(confidence)))
        counts, cases = match_boxes(labels, predicted, iou_threshold=iou_threshold)
        totals.update(counts)
        for case in cases:
            case["image"] = str(image_path)
        all_cases.extend(cases)
        per_image.append({"image": str(image_path), "ground_truth": len(labels), "predictions": len(predicted), "errors": len(cases)})

    per_class: dict[str, dict[str, float | int]] = {}
    for class_name in names:
        tp = totals[f"tp:{class_name}"]
        fp = totals[f"fp:{class_name}"]
        fn = totals[f"fn:{class_name}"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[class_name] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}

    confusion_matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for case in all_cases:
        if case["reason"] == "class_confusion":
            confusion_matrix[str(case["expected"])][str(case["predicted"])] += 1
    return {
        "model": str(Path(model_path).expanduser().resolve()),
        "dataset": str(dataset_file),
        "split": split,
        "device": resolve_device(device),
        "confidence_threshold": conf,
        "iou_threshold": iou_threshold,
        "images": len(image_paths),
        "per_class": per_class,
        "confusion_matrix": {expected: dict(predicted) for expected, predicted in confusion_matrix.items()},
        "cases": all_cases,
        "per_image": per_image,
    }

