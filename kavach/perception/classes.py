"""KAVACH semantic vocabulary and class-filtering helpers.

This module is a KAVACH contribution. The names below are application-level
labels and are not a claim that the current pretrained model can detect every
label reliably.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

# The desired warehouse vocabulary for future model evaluation. A prompt or a
# configured class name alone does not establish detector accuracy.
WAREHOUSE_VOCABULARY: tuple[str, ...] = (
    "person",
    "cardboard box",
    "carton",
    "package",
    "pallet",
    "trolley",
    "pallet truck",
    "forklift",
    "truck",
)

# Canonical supervised-training schema for the KAVACH warehouse detector.
# ``carton`` is the single label for a visible cardboard box/carton. Keeping
# one canonical name avoids teaching a detector that the same physical object
# is two different classes (``cardboard box`` and ``carton``).
WAREHOUSE_TRAINING_CLASSES: tuple[str, ...] = (
    "person",
    "carton",
    "package",
    "pallet",
    "trolley",
    "pallet_truck",
    "forklift",
    "truck",
)

# Exact names from the standard COCO model that overlap the current target
# vocabulary. YOLO11n on COCO does not include forklift, pallet, box, or
# trolley classes.
STANDARD_COCO_WAREHOUSE_CLASSES: tuple[str, ...] = ("person", "truck")


def normalize_class_name(name: object) -> str:
    """Return a stable, case-insensitive class-name representation."""

    return str(name).strip().casefold()


def model_class_names(names: Mapping[int, object] | Iterable[object]) -> dict[int, str]:
    """Normalize Ultralytics' class-name mapping or list into a dict."""

    if isinstance(names, Mapping):
        return {int(class_id): str(name) for class_id, name in names.items()}
    return {class_id: str(name) for class_id, name in enumerate(names)}


def class_ids_for_names(
    names: Mapping[int, object] | Iterable[object],
    wanted_names: Iterable[str] | None,
) -> tuple[int, ...]:
    """Resolve wanted names to model class IDs without inventing aliases."""

    if wanted_names is None:
        return ()

    wanted = {normalize_class_name(name) for name in wanted_names}
    normalized_names = model_class_names(names)
    return tuple(
        class_id
        for class_id, name in normalized_names.items()
        if normalize_class_name(name) in wanted
    )
