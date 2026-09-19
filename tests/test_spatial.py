"""Synthetic tests for KAVACH Module 5 geometry, zones, and calibration."""

from __future__ import annotations

import unittest

import numpy as np

from kavach.intelligence import (
    CalibrationError,
    GeometryError,
    GroundPlaneCalibration,
    LOADING_ZONE,
    PALLET_ZONE,
    RESTRICTED_ZONE,
    STAGING_ZONE,
    ZoneManager,
    bbox_center,
    bbox_intersection,
    bbox_iou,
    bottom_center,
    distance_to_line,
    distance_to_region,
    draw_bbox_centers,
    draw_distance_indicator,
    draw_support_overlap,
    draw_zones,
    euclidean_distance,
    horizontal_overlap,
    point_inside_polygon,
    relative_above,
    relative_below,
    relative_left,
    relative_right,
    support_ratio,
    validate_bbox,
    validate_polygon,
    vertical_overlap,
)


BOX_A = (0.0, 0.0, 10.0, 10.0)
BOX_B = (5.0, 5.0, 15.0, 15.0)
SQUARE = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))


class GeometryTests(unittest.TestCase):
    def test_bbox_points_and_distance(self) -> None:
        self.assertEqual(bbox_center(BOX_A), (5.0, 5.0))
        self.assertEqual(bottom_center(BOX_A), (5.0, 10.0))
        self.assertAlmostEqual(euclidean_distance((0.0, 0.0), (3.0, 4.0)), 5.0)

    def test_intersection_iou_and_axis_overlaps(self) -> None:
        self.assertEqual(bbox_intersection(BOX_A, BOX_B), (5.0, 5.0, 10.0, 10.0))
        self.assertIsNone(bbox_intersection(BOX_A, (20.0, 20.0, 30.0, 30.0)))
        self.assertAlmostEqual(bbox_iou(BOX_A, BOX_B), 1.0 / 7.0)
        self.assertAlmostEqual(horizontal_overlap(BOX_A, BOX_B), 5.0)
        self.assertAlmostEqual(vertical_overlap(BOX_A, BOX_B), 5.0)

    def test_support_and_relative_relationships(self) -> None:
        support = (2.0, 10.0, 8.0, 20.0)
        self.assertAlmostEqual(support_ratio(BOX_A, support), 0.6)
        self.assertTrue(relative_above(BOX_A, BOX_B))
        self.assertTrue(relative_left(BOX_A, BOX_B))
        self.assertTrue(relative_below(BOX_B, BOX_A))
        self.assertTrue(relative_right(BOX_B, BOX_A))

    def test_polygon_membership_and_region_distance(self) -> None:
        self.assertTrue(point_inside_polygon((5.0, 5.0), SQUARE))
        self.assertTrue(point_inside_polygon((0.0, 5.0), SQUARE))
        self.assertFalse(point_inside_polygon((15.0, 5.0), SQUARE))
        self.assertAlmostEqual(distance_to_line((5.0, 5.0), (0.0, 0.0), (10.0, 0.0)), 5.0)
        self.assertAlmostEqual(distance_to_region((5.0, 5.0), SQUARE), 0.0)
        self.assertAlmostEqual(distance_to_region((15.0, 5.0), SQUARE), 5.0)

    def test_invalid_geometry_is_rejected(self) -> None:
        with self.assertRaises(GeometryError):
            validate_bbox((0.0, 5.0, 2.0))
        with self.assertRaises(GeometryError):
            validate_bbox((5.0, 0.0, 2.0, 2.0))
        with self.assertRaises(GeometryError):
            validate_polygon(((0.0, 0.0), (1.0, 1.0)))
        with self.assertRaises(GeometryError):
            distance_to_line((1.0, 1.0), (0.0, 0.0), (0.0, 0.0))
        with self.assertRaises(GeometryError):
            support_ratio((0.0, 0.0, 0.0, 5.0), BOX_A)


class ZoneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = ZoneManager.from_config(
            {
                STAGING_ZONE: {
                    "polygon": SQUARE,
                    "color": (255, 0, 0),
                },
                LOADING_ZONE: ((20.0, 0.0), (30.0, 0.0), (30.0, 10.0), (20.0, 10.0)),
                RESTRICTED_ZONE: ((40.0, 0.0), (50.0, 0.0), (50.0, 10.0), (40.0, 10.0)),
                PALLET_ZONE: ((60.0, 0.0), (70.0, 0.0), (70.0, 10.0), (60.0, 10.0)),
            }
        )

    def test_named_polygons_are_configurable_and_queryable(self) -> None:
        self.assertEqual(len(self.manager.zones), 4)
        self.assertTrue(self.manager.contains(STAGING_ZONE, (5.0, 5.0)))
        self.assertEqual(self.manager.zone_for_point((25.0, 5.0)), LOADING_ZONE)
        self.assertIsNone(self.manager.zone_for_point((35.0, 5.0)))
        self.assertAlmostEqual(self.manager.distance_to_zone(RESTRICTED_ZONE, (55.0, 5.0)), 5.0)

    def test_zone_visualization_draws_a_copy(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        annotated = draw_zones(frame, self.manager)
        self.assertEqual(annotated.shape, frame.shape)
        self.assertGreater(int(np.count_nonzero(annotated)), 0)
        self.assertTrue(np.array_equal(frame, np.zeros_like(frame)))


class CalibrationTests(unittest.TestCase):
    def test_planar_homography_projects_points_and_distances(self) -> None:
        calibration = GroundPlaneCalibration.from_correspondences(
            image_points=SQUARE,
            ground_points=((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)),
        )
        projected = calibration.image_to_ground((5.0, 5.0))
        self.assertAlmostEqual(projected[0], 1.0, places=5)
        self.assertAlmostEqual(projected[1], 1.0, places=5)
        self.assertAlmostEqual(
            calibration.image_distance_on_ground((0.0, 0.0), (10.0, 0.0)),
            2.0,
            places=5,
        )
        self.assertEqual(calibration.homography.shape, (3, 3))

    def test_calibration_requires_four_matching_points(self) -> None:
        with self.assertRaises(CalibrationError):
            GroundPlaneCalibration.from_correspondences(
                ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
                ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
            )
        with self.assertRaises(CalibrationError):
            GroundPlaneCalibration.from_correspondences(
                SQUARE,
                ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
            )


class SpatialVisualizationTests(unittest.TestCase):
    def test_spatial_indicators_return_annotated_copies(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        centers = draw_bbox_centers(frame, [BOX_A, BOX_B])
        distance = draw_distance_indicator(centers, (5.0, 5.0), (15.0, 15.0))
        support = draw_support_overlap(distance, BOX_A, (2.0, 10.0, 8.0, 20.0))

        self.assertEqual(support.shape, frame.shape)
        self.assertGreater(int(np.count_nonzero(support)), 0)
        self.assertTrue(np.array_equal(frame, np.zeros_like(frame)))


if __name__ == "__main__":
    unittest.main()
