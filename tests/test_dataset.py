"""Tests for the warehouse dataset contract and validator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from kavach.perception import WAREHOUSE_TRAINING_CLASSES, validate_yolo_dataset


def _write_dataset(root: Path, *, invalid_label: bool = False) -> Path:
    for split in ("train", "val"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        image_path = image_dir / "frame_0001.jpg"
        assert cv2.imwrite(str(image_path), np.zeros((100, 120, 3), dtype=np.uint8))
        if invalid_label:
            label_path = label_dir / "frame_0001.txt"
            label_path.write_text("0 0.5 0.5 1.4 0.2\n", encoding="utf-8")
        else:
            label_lines = [
                f"{class_id} {0.1 + class_id * 0.09:.3f} 0.5 0.05 0.2"
                for class_id in range(len(WAREHOUSE_TRAINING_CLASSES))
            ]
            (label_dir / "frame_0001.txt").write_text(
                "\n".join(label_lines) + "\n", encoding="utf-8"
            )
    yaml_path = root / "warehouse.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "names": dict(enumerate(WAREHOUSE_TRAINING_CLASSES)),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return yaml_path


class DatasetTests(unittest.TestCase):
    def test_valid_dataset_reports_all_classes_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = validate_yolo_dataset(_write_dataset(Path(directory)))
            self.assertTrue(report.is_valid, report.format_text())
            self.assertEqual(report.total_images, 2)
            self.assertEqual(report.total_labels, 2)
            self.assertEqual(report.class_counts["forklift"], 2)

    def test_missing_label_is_not_silently_treated_as_negative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            yaml_path = _write_dataset(root)
            (root / "labels" / "val" / "frame_0001.txt").unlink()
            report = validate_yolo_dataset(yaml_path)
            self.assertFalse(report.is_valid)
            self.assertIn("missing_label", {issue.code for issue in report.issues})

    def test_invalid_box_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = validate_yolo_dataset(
                _write_dataset(Path(directory), invalid_label=True),
                require_all_expected_classes=False,
            )
            self.assertFalse(report.is_valid)
            self.assertIn("box_not_normalized", {issue.code for issue in report.issues})

    def test_missing_yaml_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = validate_yolo_dataset(Path(directory) / "missing.yaml")
            self.assertFalse(report.is_valid)
            self.assertEqual(report.issues[0].code, "missing_yaml")


if __name__ == "__main__":
    unittest.main()
