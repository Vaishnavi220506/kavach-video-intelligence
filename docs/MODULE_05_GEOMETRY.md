# KAVACH Module 5: Spatial Geometry, Zones and Calibration Foundations

## Scope

Module 5 adds reusable image-space geometry, configurable warehouse zones, and optional ground-plane calibration support.

This module is intentionally below scene-graph scope. It does not add object relationships as a persistent graph, behavior recognition, drop detection, risk reasoning, a database, an LLM, or a new dashboard.

The main idea is:

video frame + detection boxes
→ geometry measurements
→ point/polygon zone queries and spatial cues
→ optional calibrated ground-plane coordinates

## 1. Coordinates and pixels

An image is a grid of pixels. In OpenCV, the origin is at the top-left:

- x increases toward the right
- y increases toward the bottom

A point is represented as (x, y). A bounding box is represented as (x1, y1, x2, y2), where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.

These coordinates describe the image. They are not automatically metres.

## 2. Repository architecture

The new Module 5 code lives beside the Module 4 temporal memory code:

~~~text
kavach/intelligence/
├── geometry.py       # Pure image-space geometry primitives
├── zones.py          # Named, configurable polygon zones
├── calibration.py    # Optional image-to-ground homography
├── visualization.py  # Trajectory and spatial overlays
├── object_memory.py  # Module 4 bounded timestamped histories
└── motion.py         # Module 4 pixel motion calculations
~~~

Important public objects:

- GeometryError: rejects malformed boxes, points, polygons, and degenerate line inputs.
- Zone: one named polygon and its BGR display color.
- ZoneManager: stores configured zones and answers containment/distance queries.
- GroundPlaneCalibration: maps image points to a supplied ground coordinate system.
- draw_zones, draw_bbox_centers, draw_distance_indicator, and draw_support_overlap: optional OpenCV overlays.

The geometry functions are reusable and do not know about YOLO, tracking, or object memory. That separation lets later KAVACH modules use the same measurements without duplicating math.

## 3. Geometry primitives

geometry.py implements:

- bbox_center(bbox): returns the center of an XYXY box.
- bottom_center(bbox): returns the point halfway across the bottom edge. This is often a more useful approximate contact point for zone tests than the box center.
- euclidean_distance(a, b): straight-line distance between two image points, in pixels.
- bbox_intersection(a, b): returns the positive-area overlapping box, or None.
- bbox_iou(a, b): intersection over union, from 0.0 to 1.0.
- horizontal_overlap(a, b) and vertical_overlap(a, b): overlap lengths along one image axis.
- support_ratio(supported_bbox, support_bbox): the fraction of the first box's width covered horizontally by the second box.
- relative_above, relative_below, relative_left, and relative_right: compare box centers in image coordinates.
- point_inside_polygon(point, polygon): boundary-inclusive point-in-polygon test.
- distance_to_line(point, start, end): distance to a finite line segment.
- distance_to_region(point, polygon): 0 for a point inside the polygon; otherwise the closest boundary distance in pixels.

Support ratio is deliberately only a geometric cue. It does not prove that an object is physically resting on another object. Vertical ordering, contact stability, object identity, and time belong to later reasoning.

### Bounding-box example

For (10, 20, 30, 60):

- width = 30 - 10 = 20 pixels
- height = 60 - 20 = 40 pixels
- center = (20, 40)
- bottom-center = (20, 60)

## 4. What is IoU?

IoU means Intersection over Union:

~~~text
IoU = area of intersection / area of union
~~~

If two boxes overlap exactly, IoU is 1.0. If they do not overlap, IoU is 0.0. Detectors and trackers use IoU as a useful measure of how much two boxes refer to the same image region.

For the test boxes (0, 0, 10, 10) and (5, 5, 15, 15), the intersection is 25 square pixels and the union is 175 square pixels, so IoU is 1/7.

## 5. Polygons and zones

A polygon is a sequence of at least three image points that outlines a region. KAVACH zones are configured polygons, not hardcoded regions:

- STAGING_ZONE
- LOADING_ZONE
- RESTRICTED_ZONE
- PALLET_ZONE

Example configuration:

~~~python
from kavach.intelligence import ZoneManager

zones = ZoneManager.from_config({
    "STAGING_ZONE": {
        "polygon": [(80, 80), (500, 80), (500, 350), (80, 350)],
        "color": (255, 180, 0),  # OpenCV BGR
    },
    "RESTRICTED_ZONE": [
        (900, 100),
        (1200, 100),
        (1200, 500),
        (900, 500),
    ],
})
~~~

The coordinates above are examples only. A real deployment must configure polygons for its camera view.

ZoneManager.contains(name, point) uses OpenCV's pointPolygonTest through the shared geometry primitive. Boundary points count as inside. zone_for_point(point) returns the first matching configured zone, so overlapping zones should be avoided or deliberately ordered.

For tracked objects, a later integration can test either the box center or its bottom-center. The choice matters: the center describes the visual object midpoint, while the bottom-center is a rough image-plane contact location.

## 6. Overlap and support intuition

Two boxes can overlap horizontally without one object supporting another. The support ratio measures only this question:

> How much of the first box's horizontal width is covered by the second box?

For a carton box from x=100 to x=200, and a pallet from x=120 to x=180, the horizontal support ratio is 60 / 100 = 0.60.

This can become one input to later spatial reasoning. It must not be treated as a physical fact by itself because image perspective, occlusion, depth, and detector box noise are not resolved by a 2-D overlap.

## 7. Pixel distance versus real-world distance

Without calibration:

- euclidean_distance returns pixels.
- distance_to_region returns pixels.
- Module 4 motion returns pixels/second.
- A pixel is not a fixed physical length across the image.

Perspective makes distant objects appear smaller. Camera zoom, camera movement, lens distortion, and object depth also affect pixel measurements. Therefore KAVACH must not label an uncalibrated pixel distance as metres, nor label pixel motion as metres/second.

## 8. Optional ground-plane calibration

GroundPlaneCalibration provides architecture for mapping image points onto an approximately planar ground surface:

~~~python
from kavach.intelligence import GroundPlaneCalibration

calibration = GroundPlaneCalibration.from_correspondences(
    image_points=[(100, 100), (900, 100), (1100, 700), (50, 700)],
    ground_points=[(0, 0), (10, 0), (10, 8), (0, 8)],
)

ground_point = calibration.image_to_ground((500, 400))
ground_distance = calibration.image_distance_on_ground((500, 400), (600, 400))
~~~

The homography is a 3x3 projective transform estimated from four or more corresponding points. The image points and ground points must refer to the same physical locations. Ground coordinates use whatever units the caller supplied; if those points are in metres, the projected distances are in metres on the calibrated ground plane.

The assumptions are important:

- the relevant ground surface is approximately planar
- calibration correspondences are available and accurate
- the camera is fixed relative to that surface
- the queried point lies on, or is a good approximation to, the ground plane

A ground-plane homography does not automatically recover object height, full 3-D position, or the top of a carried load. A single monocular camera cannot provide reliable drop height merely from bounding-box pixels. This module therefore does not implement drop-height measurement or claim it can infer it.

## 9. Visualization

The optional drawing helpers return annotated copies and leave their input frames unchanged:

- draw_zones: fills and outlines configured polygons and labels them.
- draw_bbox_centers: marks box centers.
- draw_distance_indicator: draws a line between two points and labels its pixel distance.
- draw_support_overlap: draws two boxes and highlights horizontal overlap with a percentage cue.

These are diagnostic visualizations. They do not create a scene graph or alter detector/tracker behavior.

## 10. Tests and example output

The synthetic test suite covers:

- box centers and bottom-centers
- Euclidean distance
- intersection and IoU
- horizontal and vertical overlap
- support ratio
- relative directional comparisons
- point-in-polygon and region distance
- invalid geometry rejection
- configurable named zones and visualization
- planar homography projection and ground-plane distance
- visualization output immutability

Run:

~~~text
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v
.\\.venv\\Scripts\\python.exe -m compileall -q kavach tests
git diff --check
~~~

Latest validation: 37 tests passed in 0.129 seconds. Python bytecode
compilation completed successfully, and git diff --check reported no
whitespace errors.

Example spatial output for the synthetic boxes used by the tests:

~~~text
center=(5.0, 5.0), bottom_center=(5.0, 10.0)
intersection=(5.0, 5.0, 10.0, 10.0), IoU=0.142857
support_overlap=60%, outside_region_distance=5.0 px
ground_projection_of_(5, 5)=(1.0, 1.0)
~~~

## 11. Known limitations

- Geometry is image-space geometry unless a caller explicitly supplies a valid calibration.
- A polygon must be configured for the particular camera view; there is no universal warehouse layout.
- Overlapping zones are returned in insertion order by zone_for_point.
- point_inside_polygon assumes a valid, intentionally defined polygon; self-intersecting polygons are not repaired.
- Distance to a region is the closest polygon-edge distance, not a signed distance.
- Box centers and bottom-centers are approximations derived from detector boxes.
- Support ratio is not physical support or contact detection.
- Homography quality depends on point correspondences and planar-ground assumptions.
- Calibration does not recover height or general 3-D geometry.
- No scene graph, behavior detection, risk engine, database, LLM, or dashboard redesign is included.

Module 5 ends at reusable spatial measurements, configurable zones, and explicitly bounded calibration support. Module 6 should build on these primitives only after their assumptions are understood.
