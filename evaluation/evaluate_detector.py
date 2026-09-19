"""CLI for honest per-class YOLO validation on a labeled dataset.

Example:
    .venv/Scripts/python.exe evaluation/evaluate_detector.py \
        --model weights/kavach_warehouse.pt --data data/warehouse.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kavach.perception import (
    EvaluationError,
    WAREHOUSE_TRAINING_CLASSES,
    evaluate_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="YOLO .pt checkpoint")
    parser.add_argument("--data", required=True, help="YOLO dataset YAML")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument(
        "--schema",
        choices=("warehouse", "any"),
        default="warehouse",
        help="Require the canonical KAVACH warehouse classes by default.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    try:
        result = evaluate_checkpoint(
            args.model,
            args.data,
            split=args.split,
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            expected_classes=(
                WAREHOUSE_TRAINING_CLASSES if args.schema == "warehouse" else None
            ),
        )
    except EvaluationError as exc:
        parser.error(str(exc))
        return 2
    payload = json.dumps(result.to_dict(), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
