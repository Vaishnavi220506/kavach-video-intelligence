# Module 4 — Temporal Object Memory

## Status and scope

Module 3 gave KAVACH short-term identities such as Person #1. Module 4 stores
recent observations associated with each track so KAVACH can ask:

    What has Person #1 been doing during the last few seconds?

This module adds bounded timestamped history, image-space motion calculations,
and trajectory drawing. It does not detect actions, drops, dragging, risk,
incidents, scene graphs, databases, LLM features, or physical-world speed.

All new code and this document are KAVACH CONTRIBUTION work. The upstream
repository's MIT license, copyright notice, and attribution remain in LICENSE.

## Architecture

    VideoReader
        → FramePacket(frame, frame_number, timestamp)
        → MultiObjectTracker
        → list[TrackedObject]
        → ObjectMemory.update(tracked_objects, timestamp, frame_number)
        → history[track_id] = bounded list[ObjectState]
        → motion / trajectory queries

The package is:

    kavach/
    └── intelligence/
        ├── __init__.py
        ├── motion.py          # timestamp-aware image motion calculations
        ├── object_memory.py   # bounded history indexed by track ID
        └── visualization.py   # trajectory drawing

The memory consumes TrackedObject values from Module 3. It does not run the
detector or tracker again, and it does not create states for absent tracks.

## ObjectState

Each retained state contains:

    ObjectState(
        track_id=7,
        timestamp=5.5055,
        frame_number=330,
        center=(801.54, 538.73),
        bbox=(685.85, 193.64, 917.23, 883.82),
        class_name="person",
        confidence=0.910,
    )

The history mapping is indexed by track ID:

    history[7] = [state_t1, state_t2, state_t3, ...]

ObjectMemory.history returns a defensive copy. The internal deque is bounded.

## Public API

    from kavach.intelligence import ObjectMemory

    memory = ObjectMemory(
        max_history_seconds=30.0,
        max_states_per_track=600,
        max_tracks=1000,
    )

    memory.update(tracked_objects, timestamp, frame_number=frame_number)

    latest = memory.get(track_id)
    history = memory.get_history(track_id)
    trajectory = memory.get_trajectory(track_id, seconds=2.0)
    motion = memory.get_velocity(track_id)
    stationary = memory.is_stationary(track_id)
    stationary_for = memory.get_stationary_duration(track_id)
    age = memory.get_track_age(track_id)
    last_seen = memory.get_last_seen(track_id)

Frame number is optional for callers without source metadata. When it is
available on TrackedObject, as it is when Module 3 receives a VideoReader
packet, it is preserved as the actual source frame number. Otherwise memory
uses an update counter rather than pretending processing speed is source time.

## Memory bounds and missing observations

The defaults are 30 seconds per track, at most 600 states per track, and at
most 1,000 tracks. Both time and state-count limits apply. When a new track
would exceed the track-count limit, the least recently seen retained track is
evicted.

When a track is not detected, no fake state is appended. Its last_seen remains
the timestamp of its last real observation. Absence of a detection is not proof
of a precise object position.

## Beginner concepts

### What is temporal information?

Temporal information describes how an observation changes over time. One image
can show where an object is. Timestamped observations can show whether it
moved, how long it stayed nearly still, and when it was last seen.

### What is a trajectory?

A trajectory is the path traced by an object's representative position. This
module uses the center of each bounding box:

    trajectory = [(x1, y1), (x2, y2), (x3, y3), ...]

It is a path in image pixels, not a physical path in metres. The helpers
draw_trajectory(frame, trajectory) and
draw_memory_trajectory(frame, memory, track_id, seconds=2) draw the path.

### What are dx and dy?

For two consecutive states:

    dx = current_x - previous_x
    dy = current_y - previous_y

In OpenCV coordinates, positive dx means right and positive dy means down.
A negative dy means upward image movement.

### What is distance?

Straight-line pixel displacement is:

    distance_pixels = sqrt(dx² + dy²)

It is the shortest straight-line distance between two observed centers, not the
total length of every bend in a path.

### What is speed?

The module uses source timestamps:

    dt = current_timestamp - previous_timestamp
    speed_pixels_per_second = distance_pixels / dt

The unit is deliberately pixels/second. Until camera calibration and a
ground-plane model exist, KAVACH must not call this metres/second or physical
velocity.

### Why do timestamps matter?

Ten pixels in one second is 10 pixels/second. Ten pixels in two seconds is 5
pixels/second. The displacement is equal, but the speed is not. A computer
may process frames faster or slower than the video's source rate, and frames
may be skipped. Motion therefore uses timestamp differences, not wall-clock
processing time or an assumed FPS.

Memory rejects updates that move backward in source time. Motion rejects a
non-positive interval rather than dividing by zero.

### What is direction?

MotionEstimate.direction categorizes image motion as right, left, up, down, a
diagonal direction, or stationary. It is image-space direction; camera
perspective and camera movement have not been corrected.

### What is stationary duration?

The memory examines the latest consecutive motion intervals. If their speed
stays below the configured stationary threshold, 2.0 pixels/second by default,
get_stationary_duration reports the duration of that trailing stationary run.
With only one state there is not enough evidence, so is_stationary is false and
duration is zero.

### What is acceleration?

With at least three states, the module compares the latest velocity with the
previous velocity and reports optional image-space acceleration magnitude. It
is not calibrated physical acceleration. Noisy box centers, camera movement,
and irregular detections can make it unstable.

### Why tracking alone is insufficient

Tracking can say that the current observation is probably still Person #1. It
does not by itself provide a queryable record of earlier centers, timestamps,
or confidence values. ObjectMemory supplies that bounded evidence window for
later temporal reasoning.

## Real tracked-video test

The real path was:

    data/videos/sample_warehouse.mp4
      → VideoReader
      → yolo11n.pt detector
      → explicit ByteTrack
      → ObjectMemory

Reader metadata:

    fps=59.94006
    resolution=1920x1080
    total_frames=4548
    duration=75.876s

The first 331 source frames were processed sequentially. Memory retained two
track IDs during that prefix, including 155 states for Person #1. The latest
state for that track was:

    track_id=1
    timestamp=5.5055s
    frame_number=330
    center=(801.540, 538.729)
    confidence=0.910

Its 2-second query returned 120 center points. The latest timestamp-aware
motion estimate was:

    dx=2.464 pixels
    dy=-7.316 pixels
    displacement=7.720 pixels
    speed=462.717 pixels/second
    direction=up
    acceleration=20255.600 pixels/second²
    track_age=2.569 seconds
    last_seen=5.506 seconds
    stationary=False
    stationary_duration=0.000 seconds

The acceleration value is shown for transparency, not as a reliable physical
measurement. Its size illustrates why future modules must smooth and validate
motion before using it for operational conclusions.

## Synthetic tests

Synthetic tests verified timestamp-based speed, positive dx and rightward
direction, stationary duration, three-state acceleration, time and count
history bounds, missing updates without fabricated states, non-monotonic time
rejection, and trajectory visualization.

Run from the repository root:

    .\.venv\Scripts\python.exe -m compileall -q kavach tests
    .\.venv\Scripts\python.exe -m unittest discover -s tests -v

The combined Modules 1–4 suite reports:

    Ran 27 tests in 0.076s
    OK

The real tracked-video memory test was run separately because it requires model
weights and is much slower than the synthetic unit tests.

## Files changed in Module 4

Added:

- kavach/intelligence/__init__.py
- kavach/intelligence/motion.py
- kavach/intelligence/object_memory.py
- kavach/intelligence/visualization.py
- tests/test_memory.py
- docs/MODULE_04_MEMORY.md

Modified:

- kavach/perception/tracker.py — propagates optional source frame numbers.
- app/streamlit/app.py — passes VideoReader frame numbers through the existing
  tracker call; no memory or dashboard visualization was added.
- docs/MODULE_03_TRACKING.md — documents the optional frame number field.
- docs/MODULE_02_DETECTION.md — clarifies its historical Module 2 scope.

## Known limitations

- Histories are bounded recent evidence, not permanent storage.
- Memory inherits any incorrect identity or ID switch from ByteTrack.
- Bounding-box centers are rough object-position estimates.
- Pixel motion is affected by camera movement, perspective, zoom, and depth.
- Values are pixels, pixels/second, and pixels/second². No metres/second claim
  is valid before calibration and scene geometry.
- Stationary detection is threshold-based and affected by detector jitter.
- Acceleration is sensitive to timestamp gaps and noisy measurements.
- The module does not detect drops, carrying, dragging, actions, or anomalies.
- There is no persistence across application restarts or event database.

Module 4 ends at bounded temporal object memory and image-space motion.
Module 5 extends these foundations with reusable spatial geometry, named
polygon zones, and optional ground-plane calibration support. See
MODULE_05_GEOMETRY.md for its scope and assumptions.
