"""Unstable-stack detector using adjacent support geometry."""

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

UNSTABLE_STACK = "UNSTABLE_STACK"


class UnstableStackDetector(BehaviourDetector):
    """Find a vertical stack with an adjacent support ratio below a limit."""

    event_type = UNSTABLE_STACK

    def __init__(
        self,
        *,
        stackable_classes: object = None,
        min_stack_objects: int = 3,
        min_stable_support_ratio: float = 0.75,
        max_vertical_gap_px: float = 20.0,
        debounce_seconds: float = 0.5,
        cooldown_seconds: float = 3.0,
    ) -> None:
        super().__init__(
            debounce_seconds=debounce_seconds,
            cooldown_seconds=cooldown_seconds,
        )
        if int(min_stack_objects) < 2:
            raise BehaviourError("min_stack_objects must be at least 2")
        if not 0.0 <= float(min_stable_support_ratio) <= 1.0:
            raise BehaviourError("min_stable_support_ratio must be between 0 and 1")
        if float(max_vertical_gap_px) < 0.0:
            raise BehaviourError("max_vertical_gap_px must be non-negative")
        self.stackable_classes = class_names(
            stackable_classes,
            ("carton", "box", "package", "pallet"),
        )
        self.min_stack_objects = int(min_stack_objects)
        self.min_stable_support_ratio = float(min_stable_support_ratio)
        self.max_vertical_gap_px = float(max_vertical_gap_px)

    @staticmethod
    def _class_name(node: SceneNode) -> str:
        assert node.state is not None
        return "_".join(
            str(node.state.class_name).strip().lower().replace("-", " ").split()
        )

    def _support_measure(
        self,
        top: SceneNode,
        lower: SceneNode,
    ) -> tuple[float, float, float] | None:
        assert top.state is not None and lower.state is not None
        if not relative_above(top.state.bbox, lower.state.bbox):
            return None
        gap = max(0.0, lower.state.bbox[1] - top.state.bbox[3])
        if gap > self.max_vertical_gap_px:
            return None
        overlap = horizontal_overlap(top.state.bbox, lower.state.bbox)
        if overlap <= 0.0:
            return None
        return support_ratio(top.state.bbox, lower.state.bbox), gap, overlap

    def _find_chains(
        self,
        support_map: dict[str, list[tuple[str, tuple[float, float, float]]]],
        start: str,
        chain: tuple[str, ...] = (),
    ) -> Iterable[tuple[tuple[str, ...], tuple[tuple[float, float, float], ...]]]:
        current_chain = chain + (start,)
        if len(current_chain) >= self.min_stack_objects:
            yield current_chain, ()
            return
        for lower, measure in support_map.get(start, ()):
            if lower in current_chain:
                continue
            for result_chain, result_measures in self._find_chains(
                support_map,
                lower,
                current_chain,
            ):
                yield result_chain, (measure,) + result_measures

    def evaluate(self, context: BehaviourContext) -> Iterable[BehaviourCandidate]:
        object_nodes = [
            node
            for node in context.scene_graph.nodes
            if node.node_type == "object"
            and node.state is not None
            and self._class_name(node) in self.stackable_classes
        ]
        support_map: dict[str, list[tuple[str, tuple[float, float, float]]]] = {}
        for top in object_nodes:
            for lower in object_nodes:
                if top.node_id == lower.node_id:
                    continue
                measure = self._support_measure(top, lower)
                if measure is not None:
                    support_map.setdefault(top.node_id, []).append((lower.node_id, measure))

        emitted: set[tuple[str, ...]] = set()
        for node in object_nodes:
            for chain, measures in self._find_chains(support_map, node.node_id):
                if chain in emitted or len(measures) != len(chain) - 1:
                    continue
                minimum_ratio = min(measure[0] for measure in measures)
                if minimum_ratio >= self.min_stable_support_ratio:
                    continue
                emitted.add(chain)
                evidence: dict[str, object] = {
                    "stack_height_objects": float(len(chain)),
                    "minimum_support_ratio": minimum_ratio,
                    "minimum_stable_support_ratio": self.min_stable_support_ratio,
                    "weak_support_links": float(
                        sum(
                            measure[0] < self.min_stable_support_ratio
                            for measure in measures
                        )
                    ),
                }
                for index, (ratio, gap, overlap) in enumerate(measures):
                    evidence[f"support_ratio_{index}"] = ratio
                    evidence[f"vertical_gap_{index}_px"] = gap
                    evidence[f"horizontal_overlap_{index}_px"] = overlap
                yield BehaviourCandidate(
                    event_type=self.event_type,
                    timestamp=context.timestamp,
                    entities=chain,
                    confidence=min(0.90, 0.55 + (1.0 - minimum_ratio) * 0.30),
                    evidence=evidence,
                    key=chain,
                )

