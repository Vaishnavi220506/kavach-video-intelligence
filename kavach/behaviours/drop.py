"""Possible-drop pattern detector using bounded temporal evidence."""

from __future__ import annotations

import itertools
from collections.abc import Iterable

from ..intelligence.motion import MotionError, calculate_motion
from ..intelligence.relationships import MOVING_WITH, NEAR, SUPPORTED_BY
from .base import (
    BehaviourCandidate,
    BehaviourContext,
    BehaviourDetector,
    BehaviourError,
    class_names,
)

POSSIBLE_DROP = "POSSIBLE_DROP"


class PossibleDropDetector(BehaviourDetector):
    """Emit a cautious event for a move-separation-downward-settle pattern."""

    event_type = POSSIBLE_DROP

    def __init__(
        self,
        *,
        drop_classes: object = None,
        min_pre_motion_speed_px_per_second: float = 20.0,
        min_downward_speed_px_per_second: float = 35.0,
        min_downward_displacement_px: float = 12.0,
        max_settled_speed_px_per_second: float = 5.0,
        min_deceleration_px_per_second: float = 15.0,
        max_pattern_seconds: float = 4.0,
        near_floor_fraction: float = 0.85,
        require_separation: bool = True,
        debounce_seconds: float = 0.0,
        cooldown_seconds: float = 3.0,
    ) -> None:
        super().__init__(
            debounce_seconds=debounce_seconds,
            cooldown_seconds=cooldown_seconds,
        )
        for name, value in (
            ("min_pre_motion_speed_px_per_second", min_pre_motion_speed_px_per_second),
            ("min_downward_speed_px_per_second", min_downward_speed_px_per_second),
            ("min_downward_displacement_px", min_downward_displacement_px),
            ("max_settled_speed_px_per_second", max_settled_speed_px_per_second),
            ("min_deceleration_px_per_second", min_deceleration_px_per_second),
            ("max_pattern_seconds", max_pattern_seconds),
        ):
            if float(value) < 0.0:
                raise BehaviourError(f"{name} must be non-negative")
        if not 0.0 <= float(near_floor_fraction) <= 1.0:
            raise BehaviourError("near_floor_fraction must be between 0 and 1")
        self.drop_classes = class_names(
            drop_classes,
            ("carton", "box", "package", "pallet"),
        )
        self.min_pre_motion_speed_px_per_second = float(min_pre_motion_speed_px_per_second)
        self.min_downward_speed_px_per_second = float(min_downward_speed_px_per_second)
        self.min_downward_displacement_px = float(min_downward_displacement_px)
        self.max_settled_speed_px_per_second = float(max_settled_speed_px_per_second)
        self.min_deceleration_px_per_second = float(min_deceleration_px_per_second)
        self.max_pattern_seconds = float(max_pattern_seconds)
        self.near_floor_fraction = float(near_floor_fraction)
        self.require_separation = bool(require_separation)

    def _recent_separation(
        self,
        context: BehaviourContext,
        node_id: str,
    ) -> tuple[bool, str | None, str | None]:
        snapshots = context.scene_graph.snapshots
        for previous, current in itertools.pairwise(snapshots):
            previous_pairs = set(self._snapshot_relations(previous, node_id))
            current_pairs = set(self._snapshot_relations(current, node_id))
            disappeared = previous_pairs - current_pairs
            if disappeared:
                relation, other = sorted(disappeared)[0]
                return True, relation, other
        return False, None, None

    @staticmethod
    def _snapshot_relations(snapshot, node_id: str) -> set[tuple[str, str]]:
        relations: set[tuple[str, str]] = set()
        for edge in snapshot.relations:
            if edge.relation not in {NEAR, SUPPORTED_BY, MOVING_WITH}:
                continue
            if edge.subject == node_id:
                relations.add((edge.relation, edge.object))
            elif edge.object == node_id:
                relations.add((edge.relation, edge.subject))
        return relations

    def evaluate(self, context: BehaviourContext) -> Iterable[BehaviourCandidate]:
        for node in context.scene_graph.nodes:
            if node.node_type != "object" or node.state is None:
                continue
            class_name = "_".join(
                str(node.state.class_name).strip().lower().replace("-", " ").split()
            )
            if class_name not in self.drop_classes:
                continue
            history = context.memory.get_history(node.state.track_id)
            if len(history) < 4:
                continue
            states = history[-4:]
            if states[-1].timestamp - states[0].timestamp > self.max_pattern_seconds:
                continue
            try:
                pre_motion = calculate_motion(states[0], states[1])
                downward_motion = calculate_motion(states[1], states[2])
                settled_motion = calculate_motion(states[2], states[3])
            except MotionError:
                continue

            moving_before = pre_motion.speed_pixels_per_second >= self.min_pre_motion_speed_px_per_second
            rapid_downward = (
                downward_motion.velocity_y_pixels_per_second
                >= self.min_downward_speed_px_per_second
                and downward_motion.dy >= self.min_downward_displacement_px
            )
            settled = settled_motion.speed_pixels_per_second <= self.max_settled_speed_px_per_second
            abrupt_deceleration = (
                downward_motion.speed_pixels_per_second - settled_motion.speed_pixels_per_second
                >= self.min_deceleration_px_per_second
            )
            separated, separation_relation, separated_entity = self._recent_separation(
                context,
                node.node_id,
            )
            if not (moving_before and rapid_downward and settled and abrupt_deceleration):
                continue
            if self.require_separation and not separated:
                continue

            near_floor = False
            if context.frame_height is not None:
                near_floor = float(states[-1].bbox[3]) >= context.frame_height * self.near_floor_fraction
            evidence: dict[str, object] = {
                "moving_before": moving_before,
                "separated_from_previous_relation": separated,
                "rapid_downward_motion": rapid_downward,
                "abrupt_deceleration": abrupt_deceleration,
                "settled_after_downward_motion": settled,
                "pre_motion_speed_px_per_second": pre_motion.speed_pixels_per_second,
                "downward_speed_px_per_second": downward_motion.speed_pixels_per_second,
                "downward_displacement_px": downward_motion.dy,
                "post_motion_speed_px_per_second": settled_motion.speed_pixels_per_second,
                "pattern_duration_seconds": states[-1].timestamp - states[0].timestamp,
                "near_floor": near_floor,
            }
            if separation_relation is not None:
                evidence["separation_relation"] = separation_relation
            entities = (node.node_id,)
            if separated_entity is not None:
                entities += (separated_entity,)
            confidence = min(
                0.92,
                0.55
                + (0.10 if separated else 0.0)
                + min(0.15, downward_motion.speed_pixels_per_second / 500.0)
                + (0.10 if near_floor else 0.0),
            )
            yield BehaviourCandidate(
                event_type=self.event_type,
                timestamp=context.timestamp,
                entities=entities,
                confidence=confidence,
                evidence=evidence,
                key=(node.node_id,),
            )
