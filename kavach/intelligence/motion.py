"""Timestamp-aware motion calculations for KAVACH object histories."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol


Point = tuple[float, float]


class MotionState(Protocol):
    """Small state interface required by the motion functions."""

    timestamp: float
    center: Point


class MotionError(ValueError):
    """Raised when motion cannot be calculated from invalid timestamps."""


@dataclass(frozen=True)
class MotionEstimate:
    """Motion between two timestamped observations in image pixels."""

    dx: float
    dy: float
    displacement_pixels: float
    velocity_x_pixels_per_second: float
    velocity_y_pixels_per_second: float
    speed_pixels_per_second: float
    direction: str
    duration_seconds: float
    stationary: bool
    acceleration_pixels_per_second_squared: float | None = None


def _direction(dx: float, dy: float, stationary: bool) -> str:
    """Convert image-coordinate movement into a human-readable direction."""

    if stationary:
        return "stationary"

    angle = math.degrees(math.atan2(dy, dx))
    if -22.5 <= angle < 22.5:
        return "right"
    if 22.5 <= angle < 67.5:
        return "down-right"
    if 67.5 <= angle < 112.5:
        return "down"
    if 112.5 <= angle < 157.5:
        return "down-left"
    if angle >= 157.5 or angle < -157.5:
        return "left"
    if -157.5 <= angle < -112.5:
        return "up-left"
    if -112.5 <= angle < -67.5:
        return "up"
    return "up-right"


def calculate_motion(
    previous: MotionState,
    current: MotionState,
    *,
    previous_previous: MotionState | None = None,
    stationary_speed_threshold: float = 2.0,
) -> MotionEstimate:
    """Calculate displacement and pixel-per-second motion from timestamps.

    ``dx`` and ``dy`` are image-pixel differences. A positive ``dx`` means
    movement right; a positive ``dy`` means movement down because OpenCV image
    coordinates start at the top-left. No physical-world unit is inferred.
    """

    if stationary_speed_threshold < 0.0 or not math.isfinite(stationary_speed_threshold):
        raise MotionError("stationary_speed_threshold must be finite and non-negative")

    duration = float(current.timestamp) - float(previous.timestamp)
    if not math.isfinite(duration) or duration <= 0.0:
        raise MotionError(
            "current timestamp must be greater than previous timestamp "
            "for motion calculation"
        )

    dx = float(current.center[0]) - float(previous.center[0])
    dy = float(current.center[1]) - float(previous.center[1])
    displacement = math.hypot(dx, dy)
    velocity_x = dx / duration
    velocity_y = dy / duration
    speed = displacement / duration
    stationary = speed <= stationary_speed_threshold

    acceleration: float | None = None
    if previous_previous is not None:
        previous_motion = calculate_motion(
            previous_previous,
            previous,
            stationary_speed_threshold=stationary_speed_threshold,
        )
        acceleration_x = (
            velocity_x - previous_motion.velocity_x_pixels_per_second
        ) / duration
        acceleration_y = (
            velocity_y - previous_motion.velocity_y_pixels_per_second
        ) / duration
        acceleration = math.hypot(acceleration_x, acceleration_y)

    return MotionEstimate(
        dx=dx,
        dy=dy,
        displacement_pixels=displacement,
        velocity_x_pixels_per_second=velocity_x,
        velocity_y_pixels_per_second=velocity_y,
        speed_pixels_per_second=speed,
        direction=_direction(dx, dy, stationary),
        duration_seconds=duration,
        stationary=stationary,
        acceleration_pixels_per_second_squared=acceleration,
    )

