"""Build a deliberately small, manually checked person/carton prototype set.

This is not an auto-labeler. The boxes below were reviewed against the source
images and intentionally cover only clearly isolated people and foreground or
handled cartons. The crowded background carton stack is excluded because its
individual boundaries are not reliably separable in the screen-recorded
footage. That scope is recorded in the generated manifest and report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

USER_VIDEO_DIR = PROJECT_ROOT / "data" / "warehouse_dataset" / "images" / "val"
SAMPLE_DIR = PROJECT_ROOT / "data" / "warehouse_dataset" / "images" / "train"


def _people_cartons() -> dict[str, list[dict[str, object]]]:
    """Return reviewed xyxy pixel boxes for the small prototype sample."""

    people_four = [
        {"class": "person", "bbox": [315, 245, 430, 520]},
        {"class": "person", "bbox": [445, 235, 540, 440]},
        {"class": "person", "bbox": [605, 235, 710, 465]},
        {"class": "person", "bbox": [725, 230, 820, 465]},
    ]
    foreground_cartons = [
        {"class": "carton", "bbox": [470, 275, 590, 405]},
        {"class": "carton", "bbox": [465, 395, 600, 525]},
        {"class": "carton", "bbox": [590, 395, 740, 525]},
    ]
    names = {
        f"upload_fbfcea1df11b_Throwing_seating_cartons_using_strap_to_hold_480bb881__f000{frame:04d}.jpg"
        for frame in (60, 90, 120, 150, 180, 210, 240, 270)
    }
    return {name: people_four + foreground_cartons for name in names}


def _sample_boxes() -> dict[str, list[dict[str, object]]]:
    """Return boxes for independent warehouse-camera person examples."""

    return {
        "sample_warehouse_63fd5d16__f0001199.jpg": [
            {"class": "person", "bbox": [350, 200, 625, 840]}
        ],
        "sample_warehouse_63fd5d16__f0002398.jpg": [
            {"class": "person", "bbox": [750, 185, 985, 720]},
            {"class": "person", "bbox": [1110, 175, 1350, 700]},
        ],
        "sample_warehouse_63fd5d16__f0003597.jpg": [
            {"class": "person", "bbox": [455, 125, 735, 860]}
        ],
        "sample_warehouse_63fd5d16__f0000600.jpg": [],
        "sample_warehouse_63fd5d16__f0004496.jpg": [],
    }


def _expanded_sample_boxes() -> dict[str, list[dict[str, object]]]:
    """Return additional training-only person boxes from the second video.

    These boxes were created from high-confidence COCO person proposals and
    visually spot-checked. They are intentionally used for training only; the
    original five-image validation split remains unchanged so the v2 metric
    comparison is reproducible and not evaluated on these weak labels.
    """

    boxes = {
        "sample_warehouse_63fd5d16__f0000180.jpg": [[985, 625, 1395, 1075]],
        "sample_warehouse_63fd5d16__f0000240.jpg": [[576, 381, 931, 1068]],
        "sample_warehouse_63fd5d16__f0000300.jpg": [[583, 227, 837, 984]],
        "sample_warehouse_63fd5d16__f0000360.jpg": [[825, 166, 1031, 830]],
        "sample_warehouse_63fd5d16__f0000420.jpg": [[1118, 143, 1303, 709]],
        "sample_warehouse_63fd5d16__f0000480.jpg": [[1376, 145, 1583, 693]],
        "sample_warehouse_63fd5d16__f0000540.jpg": [[1644, 173, 1919, 719]],
        "sample_warehouse_63fd5d16__f0000780.jpg": [[881, 533, 1233, 1078]],
        "sample_warehouse_63fd5d16__f0000840.jpg": [[595, 339, 889, 1073]],
        "sample_warehouse_63fd5d16__f0000900.jpg": [[430, 270, 720, 1011]],
        "sample_warehouse_63fd5d16__f0000960.jpg": [[354, 224, 628, 851]],
        "sample_warehouse_63fd5d16__f0001559.jpg": [[459, 201, 704, 837]],
        "sample_warehouse_63fd5d16__f0001619.jpg": [[766, 171, 922, 737]],
        "sample_warehouse_63fd5d16__f0001679.jpg": [[1026, 159, 1271, 711]],
        "sample_warehouse_63fd5d16__f0001739.jpg": [[1332, 144, 1590, 642]],
        "sample_warehouse_63fd5d16__f0001799.jpg": [[1630, 141, 1854, 616]],
        "sample_warehouse_63fd5d16__f0002098.jpg": [
            [1023, 493, 1327, 1077],
            [530, 556, 852, 1074],
        ],
        "sample_warehouse_63fd5d16__f0002158.jpg": [
            [339, 384, 604, 1074],
            [735, 366, 1004, 1074],
        ],
        "sample_warehouse_63fd5d16__f0002218.jpg": [
            [621, 273, 823, 880],
            [276, 310, 527, 984],
        ],
        "sample_warehouse_63fd5d16__f0002278.jpg": [
            [320, 274, 558, 884],
            [729, 239, 908, 787],
        ],
        "sample_warehouse_63fd5d16__f0002458.jpg": [
            [1003, 213, 1231, 718],
            [1356, 182, 1481, 625],
        ],
        "sample_warehouse_63fd5d16__f0002518.jpg": [
            [1535, 189, 1750, 620],
            [1327, 206, 1500, 691],
        ],
        "sample_warehouse_63fd5d16__f0002578.jpg": [
            [1599, 219, 1811, 707],
            [1784, 195, 1919, 615],
        ],
        "sample_warehouse_63fd5d16__f0002878.jpg": [[1252, 744, 1538, 1080]],
        "sample_warehouse_63fd5d16__f0002938.jpg": [[669, 451, 947, 1073]],
        "sample_warehouse_63fd5d16__f0002998.jpg": [[410, 322, 629, 1061]],
        "sample_warehouse_63fd5d16__f0003057.jpg": [[329, 250, 551, 872]],
        "sample_warehouse_63fd5d16__f0003117.jpg": [[635, 182, 779, 685]],
        "sample_warehouse_63fd5d16__f0003177.jpg": [[908, 179, 1119, 658]],
        "sample_warehouse_63fd5d16__f0003237.jpg": [[1237, 191, 1480, 665]],
        "sample_warehouse_63fd5d16__f0003297.jpg": [[1617, 197, 1792, 692]],
        "sample_warehouse_63fd5d16__f0003537.jpg": [[668, 515, 1049, 1073]],
        "sample_warehouse_63fd5d16__f0003657.jpg": [[493, 283, 743, 931]],
        "sample_warehouse_63fd5d16__f0003777.jpg": [[506, 272, 780, 911]],
        "sample_warehouse_63fd5d16__f0003897.jpg": [[569, 281, 800, 913]],
        "sample_warehouse_63fd5d16__f0004016.jpg": [[577, 290, 804, 910]],
        "sample_warehouse_63fd5d16__f0004196.jpg": [[579, 282, 812, 910]],
        "sample_warehouse_63fd5d16__f0004256.jpg": [[636, 279, 864, 916]],
        "sample_warehouse_63fd5d16__f0004436.jpg": [[1599, 235, 1820, 746]],
    }
    return {
        name: [{"class": "person", "bbox": bbox} for bbox in entries]
        for name, entries in boxes.items()
    }


def _held_out_user_boxes() -> dict[str, list[dict[str, object]]]:
    """Return the temporal hold-out frames from the uploaded camera video."""

    return {
        "upload_fbfcea1df11b_Throwing_seating_cartons_using_strap_to_hold_480bb881__f0000300.jpg": [
            {"class": "person", "bbox": [310, 260, 435, 520]},
            {"class": "person", "bbox": [555, 235, 680, 505]},
            {"class": "person", "bbox": [680, 225, 815, 510]},
            {"class": "carton", "bbox": [445, 385, 585, 525]},
            {"class": "carton", "bbox": [580, 385, 740, 525]},
        ],
        "upload_fbfcea1df11b_Throwing_seating_cartons_using_strap_to_hold_480bb881__f0000330.jpg": [
            {"class": "person", "bbox": [315, 260, 435, 520]},
            {"class": "person", "bbox": [545, 235, 665, 505]},
            {"class": "person", "bbox": [685, 225, 815, 510]},
            {"class": "carton", "bbox": [465, 390, 595, 525]},
            {"class": "carton", "bbox": [590, 390, 740, 525]},
        ],
    }


def _write_label(image_path: Path, label_path: Path, boxes: list[dict[str, object]]) -> None:
    with Image.open(image_path) as image:
        width, height = image.size
    class_ids = {"person": 0, "carton": 1}
    lines: list[str] = []
    for item in boxes:
        x1, y1, x2, y2 = (float(value) for value in item["bbox"])
        x1 = max(0.0, min(x1, width - 1.0))
        y1 = max(0.0, min(y1, height - 1.0))
        x2 = max(x1 + 1.0, min(x2, width))
        y2 = max(y1 + 1.0, min(y2, height))
        center_x = ((x1 + x2) / 2.0) / width
        center_y = ((y1 + y2) / 2.0) / height
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        lines.append(
            f"{class_ids[str(item['class'])]} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"
        )
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build(output: Path, expanded: bool = False) -> dict[str, object]:
    annotations: dict[str, dict[Path, list[dict[str, object]]]] = {
        "train": {},
        "val": {},
    }
    for name, boxes in _people_cartons().items():
        annotations["train"][USER_VIDEO_DIR / name] = boxes
    for name, boxes in _held_out_user_boxes().items():
        annotations["val"][USER_VIDEO_DIR / name] = boxes
    sample_boxes = _sample_boxes()
    expanded_names: set[str] = set()
    if expanded:
        expanded_boxes = _expanded_sample_boxes()
        expanded_names = set(expanded_boxes)
        sample_boxes.update(expanded_boxes)
    for name, boxes in sample_boxes.items():
        split = "train" if name in expanded_names or name in {
            "sample_warehouse_63fd5d16__f0001199.jpg",
            "sample_warehouse_63fd5d16__f0002398.jpg",
        } else "val"
        annotations[split][SAMPLE_DIR / name] = boxes

    if output.exists():
        shutil.rmtree(output)
    for split, records in annotations.items():
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for source, boxes in records.items():
            if not source.is_file():
                raise FileNotFoundError(f"candidate image is missing: {source}")
            destination = image_dir / source.name
            shutil.copy2(source, destination)
            _write_label(destination, label_dir / f"{source.stem}.txt", boxes)

    manifest = {
        "classes": ["person", "carton"],
        "annotation_method": (
            "manual visual review of extracted frames plus visually spot-checked "
            "high-confidence COCO person proposals for training-only expansion"
            if expanded
            else "manual visual review of extracted frames"
        ),
        "scope": "clearly isolated people plus foreground/handled cartons; ambiguous background stack excluded",
        "warning": (
            "expanded training set includes weak person labels; five-image manual "
            "validation split is unchanged; metrics are not general warehouse accuracy"
            if expanded
            else "tiny same-camera prototype; metrics are not general warehouse accuracy"
        ),
        "splits": {
            split: {
                "images": len(records),
                "instances": sum(len(boxes) for boxes in records.values()),
            }
            for split, records in annotations.items()
        },
    }
    (output / "annotation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--expanded",
        action="store_true",
        help="add training-only, visually spot-checked person annotations from the second video",
    )
    args = parser.parse_args()
    output = args.output or (
        PROJECT_ROOT / "data" / "mvp_dataset_v2" if args.expanded else PROJECT_ROOT / "data" / "mvp_dataset"
    )
    print(json.dumps(build(output.expanduser().resolve(), expanded=args.expanded), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
