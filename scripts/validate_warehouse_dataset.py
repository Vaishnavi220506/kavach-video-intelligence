"""Validate the labelled KAVACH warehouse YOLO dataset before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kavach.perception import validate_yolo_dataset
from kavach.perception import WAREHOUSE_TRAINING_CLASSES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/warehouse.yaml"))
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a human-readable report.")
    parser.add_argument(
        "--allow-missing-classes",
        action="store_true",
        help="Inspect an incomplete class set without marking it invalid for that reason.",
    )
    parser.add_argument(
        "--schema",
        choices=("warehouse", "any"),
        default="warehouse",
        help="Require the canonical eight-class schema, or validate the classes declared in the YAML.",
    )
    args = parser.parse_args()
    report = validate_yolo_dataset(
        args.data,
        expected_classes=(
            WAREHOUSE_TRAINING_CLASSES if args.schema == "warehouse" else None
        ),
        require_all_expected_classes=not args.allow_missing_classes,
    )
    print(json.dumps(report.to_dict(), indent=2) if args.json else report.format_text())
    return 0 if report.is_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
