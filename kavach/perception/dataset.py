"""Validation and reporting for the KAVACH supervised YOLO dataset.

This module deliberately separates *candidate frame collection* from
*ground-truth annotation*. A frame extracted from a video is not a label, and
an annotation produced by a generic model is not a reliable ground truth.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .classes import WAREHOUSE_TRAINING_CLASSES, normalize_class_name

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


@dataclass(frozen=True)
class DatasetIssue:
    """One actionable dataset validation issue."""

    code: str
    message: str
    split: str | None = None
    path: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "split": self.split,
            "path": self.path,
        }


@dataclass(frozen=True)
class DatasetReport:
    """Machine-readable summary of a YOLO dataset audit."""

    dataset_yaml: Path
    dataset_root: Path
    classes: tuple[str, ...]
    image_counts: dict[str, int]
    label_counts: dict[str, int]
    class_counts: dict[str, int]
    issues: tuple[DatasetIssue, ...]

    @property
    def is_valid(self) -> bool:
        return not self.issues

    @property
    def total_images(self) -> int:
        return sum(self.image_counts.values())

    @property
    def total_labels(self) -> int:
        return sum(self.label_counts.values())

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_yaml": str(self.dataset_yaml),
            "dataset_root": str(self.dataset_root),
            "classes": list(self.classes),
            "image_counts": dict(self.image_counts),
            "label_counts": dict(self.label_counts),
            "class_counts": dict(self.class_counts),
            "total_images": self.total_images,
            "total_labels": self.total_labels,
            "valid": self.is_valid,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def format_text(self) -> str:
        """Return a concise report suitable for a terminal or CI log."""

        lines = [
            f"Dataset: {self.dataset_yaml}",
            f"Root: {self.dataset_root}",
            f"Classes: {', '.join(self.classes) or '(none)'}",
            f"Images: {self.total_images} | label files: {self.total_labels}",
            "Splits: "
            + ", ".join(
                f"{split}={self.image_counts.get(split, 0)} images/"
                f"{self.label_counts.get(split, 0)} labels"
                for split in self.image_counts
            ),
        ]
        if self.class_counts:
            lines.append(
                "Instances: "
                + ", ".join(
                    f"{name}={self.class_counts.get(name, 0)}"
                    for name in self.classes
                )
            )
        if self.issues:
            lines.append("Issues:")
            lines.extend(f"- [{issue.code}] {issue.message}" for issue in self.issues)
        else:
            lines.append("Status: VALID for the requested schema")
        return "\n".join(lines)


def _normalise_names(value: Any) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        try:
            ordered = [value[key] for key in sorted(value, key=lambda item: int(item))]
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("names mapping keys must be integer class IDs") from exc
    elif isinstance(value, (list, tuple)):
        ordered = list(value)
    else:
        raise ValueError("names must be a list or an integer-keyed mapping")

    names = tuple(str(name).strip() for name in ordered)
    if not names or any(not name for name in names):
        raise ValueError("names must contain at least one non-empty class name")
    normalized = [normalize_class_name(name) for name in names]
    if len(set(normalized)) != len(normalized):
        raise ValueError("names must not contain duplicate class names")
    return names


def _resolve_path(value: object, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _split_values(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return (str(value),)


def _label_directory(image_directory: Path) -> Path:
    """Map the conventional ``images/<split>`` directory to labels."""

    parts = list(image_directory.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].casefold() == "images":
            parts[index] = "labels"
            return Path(*parts)
    return image_directory.parent / "labels" / image_directory.name


def _image_files(directory: Path) -> tuple[Path, ...]:
    if not directory.is_dir():
        return ()
    return tuple(
        sorted(
            item
            for item in directory.iterdir()
            if item.is_file() and item.suffix.casefold() in IMAGE_EXTENSIONS
        )
    )


def _validate_label_file(
    label_path: Path,
    *,
    split: str,
    class_count: int,
    class_names: tuple[str, ...],
    class_counts: Counter[str],
    issues: list[DatasetIssue],
) -> None:
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        issues.append(
            DatasetIssue(
                "unreadable_label",
                f"Could not read label file: {exc}",
                split,
                str(label_path),
            )
        )
        return

    # An existing empty file is a valid negative image in YOLO format. A
    # missing label file is reported separately by the caller because it is
    # usually an annotation workflow mistake.
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            issues.append(
                DatasetIssue(
                    "malformed_label",
                    f"Expected 5 values on line {line_number}, found {len(fields)}.",
                    split,
                    str(label_path),
                )
            )
            continue
        try:
            class_id_float = float(fields[0])
            values = [float(item) for item in fields[1:]]
        except ValueError:
            issues.append(
                DatasetIssue(
                    "non_numeric_label",
                    f"Line {line_number} contains non-numeric values.",
                    split,
                    str(label_path),
                )
            )
            continue
        class_id = int(class_id_float)
        if class_id_float != class_id or not 0 <= class_id < class_count:
            issues.append(
                DatasetIssue(
                    "class_id_out_of_range",
                    f"Line {line_number} uses class ID {fields[0]!r}; "
                    f"valid IDs are 0..{class_count - 1}.",
                    split,
                    str(label_path),
                )
            )
            continue
        if not all(math.isfinite(value) for value in values):
            issues.append(
                DatasetIssue(
                    "non_finite_box",
                    f"Line {line_number} contains NaN or infinity.",
                    split,
                    str(label_path),
                )
            )
            continue
        center_x, center_y, width, height = values
        if not all(0.0 <= value <= 1.0 for value in values) or width <= 0 or height <= 0:
            issues.append(
                DatasetIssue(
                    "box_not_normalized",
                    f"Line {line_number} must use normalized xywh values in [0, 1] "
                    "with positive width and height.",
                    split,
                    str(label_path),
                )
            )
            continue
        class_counts[class_names[class_id]] += 1


def validate_yolo_dataset(
    dataset_yaml: str | Path,
    *,
    expected_classes: Iterable[str] | None = WAREHOUSE_TRAINING_CLASSES,
    require_all_expected_classes: bool = True,
) -> DatasetReport:
    """Validate a conventional YOLO detection dataset without running training.

    ``expected_classes`` defaults to the KAVACH warehouse schema. Pass
    ``None`` when validating a different, already-defined class schema. The
    validator checks the actual files and labels, not only the YAML names.
    """

    dataset_file = Path(dataset_yaml).expanduser().resolve()
    issues: list[DatasetIssue] = []
    config: Mapping[str, Any] = {}
    classes: tuple[str, ...] = ()
    root = dataset_file.parent

    if not dataset_file.is_file():
        issues.append(
            DatasetIssue("missing_yaml", f"Dataset YAML was not found: {dataset_file}")
        )
        return DatasetReport(dataset_file, root, classes, {}, {}, {}, tuple(issues))

    try:
        loaded = yaml.safe_load(dataset_file.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise ValueError("top-level YAML value must be a mapping")
        config = loaded
        classes = _normalise_names(config.get("names"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        issues.append(DatasetIssue("invalid_yaml", f"Could not parse dataset YAML: {exc}"))
        return DatasetReport(dataset_file, root, classes, {}, {}, {}, tuple(issues))

    if "path" in config:
        root = _resolve_path(config["path"], dataset_file.parent)

    expected = tuple(str(name) for name in expected_classes) if expected_classes is not None else None
    if expected is not None:
        normalized_actual = tuple(normalize_class_name(name) for name in classes)
        normalized_expected = tuple(normalize_class_name(name) for name in expected)
        if normalized_actual != normalized_expected:
            issues.append(
                DatasetIssue(
                    "class_schema_mismatch",
                    "Expected canonical KAVACH classes "
                    f"{list(expected)}, found {list(classes)}.",
                )
            )

    image_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    class_counts: Counter[str] = Counter()

    for split in ("train", "val", "test"):
        if split not in config or config[split] in (None, ""):
            if split in ("train", "val"):
                issues.append(DatasetIssue("missing_split", f"Dataset YAML has no '{split}' split.", split))
            continue
        split_image_dirs = tuple(_resolve_path(item, root) for item in _split_values(config[split]))
        split_images: list[Path] = []
        split_labels = 0
        for image_dir in split_image_dirs:
            if not image_dir.is_dir():
                issues.append(
                    DatasetIssue(
                        "missing_image_directory",
                        f"Image directory was not found: {image_dir}",
                        split,
                        str(image_dir),
                    )
                )
                continue
            split_images.extend(_image_files(image_dir))
            label_dir = _label_directory(image_dir)
            if not label_dir.is_dir():
                issues.append(
                    DatasetIssue(
                        "missing_label_directory",
                        f"Label directory was not found: {label_dir}",
                        split,
                        str(label_dir),
                    )
                )
                continue
            label_paths = {item.stem: item for item in label_dir.glob("*.txt") if item.is_file()}
            for image_path in _image_files(image_dir):
                label_path = label_paths.get(image_path.stem)
                if label_path is None:
                    issues.append(
                        DatasetIssue(
                            "missing_label",
                            "Every image needs a matching YOLO label file; "
                            "use an empty file only for a reviewed negative image.",
                            split,
                            str(image_path),
                        )
                    )
                    continue
                split_labels += 1
                _validate_label_file(
                    label_path,
                    split=split,
                    class_count=len(classes),
                    class_names=classes,
                    class_counts=class_counts,
                    issues=issues,
                )
            image_stems = {item.stem for item in _image_files(image_dir)}
            for stem, label_path in sorted(label_paths.items()):
                if stem not in image_stems:
                    issues.append(
                        DatasetIssue(
                            "orphan_label",
                            "Label has no matching image.",
                            split,
                            str(label_path),
                        )
                    )
        image_counts[split] = len(split_images)
        label_counts[split] = split_labels
        if not split_images:
            issues.append(DatasetIssue("empty_split", f"'{split}' contains no images.", split))

    if expected is not None and require_all_expected_classes:
        missing = [name for name in expected if class_counts.get(name, 0) == 0]
        if missing:
            issues.append(
                DatasetIssue(
                    "missing_class_examples",
                    "No labelled instances were found for: " + ", ".join(missing),
                )
            )

    return DatasetReport(
        dataset_file,
        root,
        classes,
        image_counts,
        label_counts,
        dict(class_counts),
        tuple(issues),
    )


__all__ = [
    "IMAGE_EXTENSIONS",
    "DatasetIssue",
    "DatasetReport",
    "validate_yolo_dataset",
]
