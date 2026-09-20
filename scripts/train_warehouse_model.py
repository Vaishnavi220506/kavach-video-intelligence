"""Train a canonical KAVACH warehouse detector after strict validation.

The script is intentionally a thin, reproducible Ultralytics wrapper. It
does not auto-label data, alter KAVACH behaviour rules, or replace the current
baseline checkpoint until a trained model is evaluated and deliberately
selected by the user.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kavach.perception import WAREHOUSE_TRAINING_CLASSES, validate_yolo_dataset


def _absolute_ultralytics_yaml(data_file: Path, dataset_root: Path, destination: Path) -> Path:
    """Make a run-local YAML because Ultralytics resolves relative paths globally."""

    config = yaml.safe_load(data_file.read_text(encoding="utf-8"))
    config["path"] = str(dataset_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return destination


def train(args: argparse.Namespace) -> dict[str, object]:
    expected_classes = (
        WAREHOUSE_TRAINING_CLASSES if args.schema == "warehouse" else None
    )
    report = validate_yolo_dataset(
        args.data,
        expected_classes=expected_classes,
        require_all_expected_classes=expected_classes is not None,
    )
    if not report.is_valid:
        raise ValueError(
            "Dataset validation failed; training was not started.\n"
            + report.format_text()
        )

    # Some Windows Python builds ship a Torch wheel whose optional Dynamo /
    # Inductor modules are internally inconsistent. Ultralytics toggles the
    # deterministic flag even when it is false, which would abort a normal
    # CPU run before the first batch. Keep the normal path strict, but allow a
    # non-deterministic run to proceed when only that optional hook is broken.
    import torch

    if not args.deterministic:
        # The current Windows CPU wheel also wraps several optimizer methods
        # in a lazy Dynamo import. Its optional modules are inconsistent on
        # this host, so keep those wrappers as plain Python calls for this
        # small CPU training run.
        def safe_disable_dynamo(fn: object = None, recursive: bool = True) -> object:
            if fn is None:
                return lambda wrapped: wrapped
            return fn

        torch._disable_dynamo = safe_disable_dynamo

        original_use_deterministic_algorithms = torch.use_deterministic_algorithms

        def safe_use_deterministic_algorithms(mode: bool, *call_args: object, **call_kwargs: object) -> object:
            try:
                return original_use_deterministic_algorithms(mode, *call_args, **call_kwargs)
            except ImportError as exc:
                if mode or "NP_SUPPORTED_MODULES" not in str(exc):
                    raise
                return None

        torch.use_deterministic_algorithms = safe_use_deterministic_algorithms

    from ultralytics import YOLO

    from kavach.perception.detector import resolve_device

    model_path = Path(args.base_model).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Base model was not found: {model_path}")
    resolved_device = resolve_device(args.device)
    model = YOLO(str(model_path))
    run_data = _absolute_ultralytics_yaml(
        Path(args.data).expanduser().resolve(),
        report.dataset_root,
        Path(args.output_dir).expanduser().resolve() / args.name / "dataset.absolute.yaml",
    )
    results = model.train(
        data=str(run_data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=resolved_device,
        project=str(Path(args.output_dir).expanduser().resolve()),
        name=args.name,
        exist_ok=args.exist_ok,
        patience=args.patience,
        workers=0,
        cache=False,
        seed=42,
        deterministic=args.deterministic,
        plots=True,
        verbose=True,
    )
    run_dir = Path(getattr(results, "save_dir", Path(args.output_dir) / args.name)).resolve()
    checkpoint = run_dir / "weights" / "best.pt"
    summary = {
        "model": str(model_path),
        "dataset": str(Path(args.data).expanduser().resolve()),
        "device": resolved_device,
        "classes": list(report.classes),
        "epochs_requested": args.epochs,
        "run_directory": str(run_dir),
        "best_checkpoint": str(checkpoint) if checkpoint.is_file() else None,
        "note": "Run evaluation/evaluate_detector.py on the held-out val split before enabling this checkpoint.",
    }
    summary_path = run_dir / "kavach_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/warehouse.yaml"))
    parser.add_argument("--base-model", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/warehouse"))
    parser.add_argument("--name", default="yolo11n_warehouse")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--schema",
        choices=("warehouse", "any"),
        default="warehouse",
        help="Require the canonical eight-class schema, or use the classes declared in the YAML for a scoped prototype.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Request deterministic Torch algorithms when the installed Torch build supports them.",
    )
    parser.add_argument("--exist-ok", action="store_true")
    args = parser.parse_args()
    try:
        summary = train(args)
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
