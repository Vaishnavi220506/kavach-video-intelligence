"""Pallet overhang detector based on support geometry."""

from __future__ import annotations

from collections.abc import Iterable

from ..intelligence.geometry import horizontal_overlap, relative_above, support_ratio
from ..intelligence.scene_graph import SceneNode
from .base import (
    BehaviourCandidate,
    BehaviourContext,
    BehaviourDetector,
    BehaviourError,
    class_names,
)

PALLET_OVERHANG = "PALLET_OVERHANG"


class OverhangDetector(BehaviourDetector):
    """Detect a low-support object over a pallet-like provider."""

    event_type = PALLET_OVERHANG

    def __init__(
        self,
        *,
        support_provider_classes: object = None,
        min_support_ratio: float = 0.70,
        max_vertical_gap_px: float = 20.0,
        debounce_seconds: float = 0.25,
        cooldown_seconds: float = 3.0,
    ) -> None:
        super().__init__(
            debounce_seconds=debounce_seconds,
            cooldown_seconds=cooldown_seconds,
        )
        if not 0.0 <= float(min_support_ratio) <= 1.0:
            raise BehaviourError("min_support_ratio must be between 0 and 1")
        if float(max_vertical_gap_px) < 0.0:
            raise BehaviourError("max_vertical_gap_px must be non-negative")
        self.support_provider_classes = class_names(
            support_provider_classes,
            ("pallet", "platform", "shelf", "floor", "pallet_truck"),
        )
        self.min_support_ratio = float(min_support_ratio)
        self.max_vertical_gap_px = float(max_vertical_gap_px)

    @staticmethod
    def _class_name(node: SceneNode) -> str:
        assert node.state is not None
        return "_".join(
            str(node.state.class_name).strip().lower().replace("-", " ").split()
        )

    def evaluate(self, context: BehaviourContext) -> Iterable[BehaviourCandidate]:
        object_nodes = [
            node
            for node in context.scene_graph.nodes
            if node.node_type == "object" and node.state is not None
        ]
        for top in object_nodes:
            if self._class_name(top) in self.support_provider_classes:
                continue
            for support in object_nodes:
                if top.node_id == support.node_id or support.state is None:
                    continue
                if self._class_name(support) not in self.support_provider_classes:
                    continue
                if not relative_above(top.state.bbox, support.state.bbox):
                    continue
                vertical_gap = max(0.0, support.state.bbox[1] - top.state.bbox[3])
                if vertical_gap > self.max_vertical_gap_px:
                    continue
                overlap_px = horizontal_overlap(top.state.bbox, support.state.bbox)
                ratio = support_ratio(top.state.bbox, support.state.bbox)
                if ratio >= self.min_support_ratio:
                    continue
                left_overhang = max(0.0, support.state.bbox[0] - top.state.bbox[0])
                right_overhang = max(0.0, top.state.bbox[2] - support.state.bbox[2])
                evidence: dict[str, object] = {
                    "support_ratio": ratio,
                    "minimum_support_ratio": self.min_support_ratio,
                    "horizontal_overlap_px": overlap_px,
                    "left_overhang_px": left_overhang,
                    "right_overhang_px": right_overhang,
                    "vertical_gap_px": vertical_gap,
                    "top_width_px": top.state.bbox[2] - top.state.bbox[0],
                    "support_width_px": support.state.bbox[2] - support.state.bbox[0],
                }
                yield BehaviourCandidate(
                    event_type=self.event_type,
                    timestamp=context.timestamp,
                    entities=(top.node_id, support.node_id),
                    confidence=min(0.90, 0.55 + (1.0 - ratio) * 0.30),
                    evidence=evidence,
                    key=(top.node_id, support.node_id),
                )

