"""Explainable motion-anomaly signal for general video inputs.

This is a novelty heuristic over tracked image-plane motion. It is useful for
flagging abrupt changes that deserve review, but it is not a trained anomaly
classifier and cannot establish damage, intent, or a physical incident.
"""

from __future__ import annotations

from collections.abc import Iterable
import math
from statistics import median

from ..intelligence.motion import MotionError, calculate_motion
from ..intelligence.scene_graph import SceneNode
from .base import (
    BehaviourCandidate,
    BehaviourContext,
    BehaviourDetector,
    BehaviourError,
    class_names,
)


MOTION_ANOMALY = "MOTION_ANOMALY"


def _class_name(node: SceneNode) -> str:
    assert node.state is not None
    return "_".join(
        str(node.state.class_name).strip().lower().replace("-", " ").split()
    )


def _direction_change(previous, current) -> float:
    previous_angle = math.atan2(
        previous.velocity_y_pixels_per_second,
        previous.velocity_x_pixels_per_second,
    )
    current_angle = math.atan2(
        current.velocity_y_pixels_per_second,
        current.velocity_x_pixels_per_second,
    )
    delta = abs(math.degrees(current_angle - previous_angle)) % 360.0
    return min(delta, 360.0 - delta)


class MotionAnomalyDetector(BehaviourDetector):
    """Flag robust speed spikes or abrupt direction reversals per track."""

    event_type = MOTION_ANOMALY

    def __init__(
        self,
        *,
        monitored_classes: object = None,
        history_window_seconds: float = 4.0,
        minimum_history_states: int = 5,
        minimum_track_age_seconds: float = 1.0,
        minimum_baseline_speed_px_per_second: float = 15.0,
        speed_ratio_threshold: float = 4.0,
        minimum_speed_spike_px_per_second: float = 260.0,
        minimum_direction_speed_px_per_second: float = 80.0,
        direction_change_threshold_degrees: float = 135.0,
        minimum_consecutive_anomalous_states: int = 2,
        debounce_seconds: float = 0.0,
        cooldown_seconds: float = 5.0,
    ) -> None:
        super().__init__(
            debounce_seconds=debounce_seconds,
            cooldown_seconds=cooldown_seconds,
        )
        if float(history_window_seconds) <= 0.0:
            raise BehaviourError("history_window_seconds must be positive")
        if int(minimum_history_states) < 4:
            raise BehaviourError("minimum_history_states must be at least 4")
        if int(minimum_consecutive_anomalous_states) <= 0:
            raise BehaviourError("minimum_consecutive_anomalous_states must be positive")
        if float(speed_ratio_threshold) <= 1.0:
            raise BehaviourError("speed_ratio_threshold must be greater than 1")
        if float(minimum_track_age_seconds) < 0.0:
            raise BehaviourError("minimum_track_age_seconds must be non-negative")
        for name, value in (
            ("minimum_baseline_speed_px_per_second", minimum_baseline_speed_px_per_second),
            ("minimum_speed_spike_px_per_second", minimum_speed_spike_px_per_second),
            ("minimum_direction_speed_px_per_second", minimum_direction_speed_px_per_second),
            ("direction_change_threshold_degrees", direction_change_threshold_degrees),
        ):
            if float(value) < 0.0 or not math.isfinite(float(value)):
                raise BehaviourError(f"{name} must be finite and non-negative")
        if float(direction_change_threshold_degrees) > 180.0:
            raise BehaviourError("direction_change_threshold_degrees cannot exceed 180")
        self.monitored_classes = (
            None if monitored_classes is None else class_names(monitored_classes, ())
        )
        self.history_window_seconds = float(history_window_seconds)
        self.minimum_history_states = int(minimum_history_states)
        self.minimum_track_age_seconds = float(minimum_track_age_seconds)
        self.minimum_baseline_speed_px_per_second = float(
            minimum_baseline_speed_px_per_second
        )
        self.speed_ratio_threshold = float(speed_ratio_threshold)
        self.minimum_speed_spike_px_per_second = float(
            minimum_speed_spike_px_per_second
        )
        self.minimum_direction_speed_px_per_second = float(
            minimum_direction_speed_px_per_second
        )
        self.direction_change_threshold_degrees = float(
            direction_change_threshold_degrees
        )
        self.minimum_consecutive_anomalous_states = int(
            minimum_consecutive_anomalous_states
        )
        self._consecutive_anomalous_states: dict[str, int] = {}

    def evaluate(self, context: BehaviourContext) -> Iterable[BehaviourCandidate]:
        for node in context.scene_graph.nodes:
            if node.node_type != "object" or node.state is None:
                continue
            if (
                self.monitored_classes is not None
                and _class_name(node) not in self.monitored_classes
            ):
                continue
            history = context.memory.get_history(node.state.track_id)
            if len(history) < self.minimum_history_states:
                continue
            track_age = context.memory.get_track_age(node.state.track_id) or 0.0
            if track_age < self.minimum_track_age_seconds:
                continue
            cutoff = history[-1].timestamp - self.history_window_seconds
            recent = [state for state in history if state.timestamp >= cutoff]
            if len(recent) < self.minimum_history_states:
                continue
            motions = []
            for previous, current in zip(recent[:-1], recent[1:]):
                try:
                    motions.append(calculate_motion(previous, current))
                except MotionError:
                    continue
            if len(motions) < 3:
                continue
            current_motion = motions[-1]
            previous_motion = motions[-2]
            baseline_speeds = [motion.speed_pixels_per_second for motion in motions[:-1]]
            baseline = median(baseline_speeds)
            speed_ratio = current_motion.speed_pixels_per_second / max(1.0, baseline)
            speed_spike = (
                current_motion.speed_pixels_per_second
                >= self.minimum_speed_spike_px_per_second
                and (
                    (
                        baseline >= self.minimum_baseline_speed_px_per_second
                        and speed_ratio >= self.speed_ratio_threshold
                    )
                    or (
                        baseline < self.minimum_baseline_speed_px_per_second
                        and current_motion.speed_pixels_per_second
                        >= self.minimum_speed_spike_px_per_second * 2.0
                    )
                )
            )
            direction_change = _direction_change(previous_motion, current_motion)
            direction_shift = (
                current_motion.speed_pixels_per_second
                >= self.minimum_direction_speed_px_per_second
                and previous_motion.speed_pixels_per_second
                >= self.minimum_direction_speed_px_per_second
                and direction_change >= self.direction_change_threshold_degrees
            )
            anomalous = speed_spike or direction_shift
            if not anomalous:
                self._consecutive_anomalous_states.pop(node.node_id, None)
                continue
            count = self._consecutive_anomalous_states.get(node.node_id, 0) + 1
            self._consecutive_anomalous_states[node.node_id] = count
            if count < self.minimum_consecutive_anomalous_states:
                continue
            reasons = []
            if speed_spike:
                reasons.append("speed_spike")
            if direction_shift:
                reasons.append("direction_change")
            evidence = {
                "reason": "+".join(reasons),
                "speed_px_per_second": current_motion.speed_pixels_per_second,
                "baseline_speed_px_per_second": baseline,
                "speed_ratio": speed_ratio,
                "direction_change_degrees": direction_change,
                "current_direction": current_motion.direction,
                "previous_direction": previous_motion.direction,
                "acceleration_px_per_second_squared": (
                    current_motion.acceleration_pixels_per_second_squared or 0.0
                ),
                "history_window_seconds": self.history_window_seconds,
                "track_age_seconds": track_age,
                "minimum_track_age_seconds": self.minimum_track_age_seconds,
                "minimum_baseline_speed_px_per_second": self.minimum_baseline_speed_px_per_second,
                "consecutive_anomalous_states": float(count),
                "minimum_consecutive_anomalous_states": float(
                    self.minimum_consecutive_anomalous_states
                ),
                "image_space_only": True,
            }
            confidence = min(
                0.92,
                0.48
                + (0.22 if speed_spike else 0.0)
                + (0.18 if direction_shift else 0.0)
                + min(0.10, max(0.0, speed_ratio - 1.0) / 20.0),
            )
            yield BehaviourCandidate(
                event_type=self.event_type,
                timestamp=context.timestamp,
                entities=(node.node_id,),
                confidence=confidence,
                evidence=evidence,
                key=(node.node_id,),
            )

        active_node_ids = {
            node.node_id
            for node in context.scene_graph.nodes
            if node.node_type == "object"
        }
        for node_id in tuple(self._consecutive_anomalous_states):
            if node_id not in active_node_ids:
                self._consecutive_anomalous_states.pop(node_id, None)

    def reset(self) -> None:
        super().reset()
        self._consecutive_anomalous_states.clear()


__all__ = ["MOTION_ANOMALY", "MotionAnomalyDetector"]
