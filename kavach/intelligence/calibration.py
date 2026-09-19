"""Optional image-to-ground-plane calibration for KAVACH Module 5.

This module converts image points to coordinates on an approximately planar
ground surface when a user supplies corresponding calibration points. It does
not recover object height or general 3-D position from one camera.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import Point, euclidean_distance


class CalibrationError(ValueError):
    """Raised when a ground-plane calibration cannot be constructed or used."""


def _points_array(points: Sequence[Sequence[float]], label: str) -> np.ndarray:
    try:
        array = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"{label} must contain numeric 2-D points") from exc
    if array.ndim != 2 or array.shape[1:] != (2,) or array.shape[0] < 4:
        raise CalibrationError(f"{label} must contain at least four 2-D points")
    if not np.isfinite(array).all():
        raise CalibrationError(f"{label} must contain only finite values")
    return array


@dataclass(frozen=True)
class GroundPlaneCalibration:
    """Homography-backed mapping from image coordinates to ground coordinates.

    ``ground_points`` define the units used by the caller (for example metres
    or feet). The mapping is meaningful only for points on the calibrated
    approximately planar surface.
    """

    image_points: tuple[Point, ...]
    ground_points: tuple[Point, ...]
    homography: np.ndarray
    inlier_mask: tuple[int, ...] | None = None

    @classmethod
    def from_correspondences(
        cls,
        image_points: Sequence[Sequence[float]],
        ground_points: Sequence[Sequence[float]],
        *,
        method: int = 0,
        ransac_reprojection_threshold: float = 3.0,
    ) -> "GroundPlaneCalibration":
        """Estimate a homography from four or more corresponding point pairs."""

        image_array = _points_array(image_points, "image_points")
        ground_array = _points_array(ground_points, "ground_points")
        if image_array.shape[0] != ground_array.shape[0]:
            raise CalibrationError("image_points and ground_points must have equal length")
        if ransac_reprojection_threshold <= 0.0:
            raise CalibrationError("ransac_reprojection_threshold must be positive")

        try:
            homography, mask = cv2.findHomography(
                image_array,
                ground_array,
                method=method,
                ransacReprojThreshold=float(ransac_reprojection_threshold),
            )
        except cv2.error as exc:
            raise CalibrationError("OpenCV could not estimate the homography") from exc
        if homography is None or homography.shape != (3, 3) or not np.isfinite(homography).all():
            raise CalibrationError("OpenCV could not estimate a finite 3x3 homography")
        scale = float(homography[2, 2])
        if abs(scale) > np.finfo(float).eps:
            homography = homography / scale
        inlier_mask = None if mask is None else tuple(int(value) for value in mask.ravel())
        return cls(
            image_points=tuple((float(x), float(y)) for x, y in image_array),
            ground_points=tuple((float(x), float(y)) for x, y in ground_array),
            homography=np.asarray(homography, dtype=np.float64),
            inlier_mask=inlier_mask,
        )

    def image_to_ground(self, point: Sequence[float]) -> Point:
        """Project one image point onto the calibrated ground plane."""

        try:
            point_array = np.asarray(point, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise CalibrationError("point must contain two numeric values") from exc
        if point_array.shape != (2,) or not np.isfinite(point_array).all():
            raise CalibrationError("point must contain two finite values")
        try:
            projected = cv2.perspectiveTransform(
                point_array.reshape(1, 1, 2),
                self.homography,
            ).reshape(2)
        except cv2.error as exc:
            raise CalibrationError("OpenCV could not project the image point") from exc
        if not np.isfinite(projected).all():
            raise CalibrationError("homography produced a non-finite ground point")
        return (float(projected[0]), float(projected[1]))

    def image_distance_on_ground(
        self,
        point_a: Sequence[float],
        point_b: Sequence[float],
    ) -> float:
        """Return distance after projection, in the supplied ground units."""

        return euclidean_distance(self.image_to_ground(point_a), self.image_to_ground(point_b))
