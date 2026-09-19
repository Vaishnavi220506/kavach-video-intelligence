# Module 3 — Multi-Object Tracking

## Status and scope

Module 3 turns frame-local Module 2 detections into observations associated
with persistent track IDs. The intended transformation is:

```text
Frame 1 → person
Frame 2 → person

becomes

Frame 1 → person #3
Frame 2 → person #3
```

This module adds only bounded tracker state. It does not add KAVACH's
long-term `ObjectMemory`, trajectory history, scene graphs, behaviour
detection, risk reasoning, incidents, databases, LLM features, or a
dashboard redesign. Those remain future work.

The new code is **KAVACH CONTRIBUTION** code. The upstream repository's MIT
license, copyright notice, and attribution remain in `LICENSE`. Module 0
documents the upstream/KAVACH boundary.

## What the upstream repository actually did

The upstream Streamlit application and notebooks 06 and 07 call:

```python
results = model.track(frame, persist=True, verbose=False)
```

They do not pass a `tracker=` argument. In the installed environment used for
this project:

```text
Ultralytics = 8.4.145
DEFAULT_CFG.tracker = tracktrack.yaml
runtime tracker = TRACKTRACK
```

Therefore the upstream code did not explicitly use ByteTrack. README wording
or a notebook comment cannot change the tracker selected by the installed
runtime.

## Module 3 architecture

KAVACH uses the Module 2 detector as an input boundary and instantiates
Ultralytics' `BYTETracker` directly with an explicit `ByteTrackConfig`:

```text
VideoReader
    → FramePacket(frame, timestamp)
    → WarehouseDetector.detect(frame)
    → list[Detection]
    → _DetectionBatch
    → BYTETracker.update(...)
    → list[TrackedObject]
    → draw_tracked_objects(...)
```

Using `BYTETracker` directly avoids depending on whatever tracker happens to
be the Ultralytics global default. It also makes the boundary clear: the
detector creates observations; ByteTrack associates observations across
successive calls.

### `kavach/perception/tracker.py`

Important pieces:

- `ByteTrackConfig`: validated high/low confidence thresholds, new-track
  threshold, lost-track buffer, association threshold, and score fusion.
- `TrackedObject`: the standard output record.
- `MultiObjectTracker.update(frame, timestamp)`: runs the detector and
  advances ByteTrack for one frame.
- `MultiObjectTracker.update_detections(detections, timestamp)`: accepts an
  already computed Module 2 detection list. This makes the detection →
  association boundary directly testable.
- `MultiObjectTracker.reset()`: starts a new tracking session.

```python
from kavach.perception import MultiObjectTracker
from kavach.video import VideoReader

tracker = MultiObjectTracker(model_path="yolo11n.pt", device="auto")

with VideoReader("data/videos/sample_warehouse.mp4") as reader:
    for packet in reader:
        objects = tracker.update(packet.frame, packet.timestamp)
        for obj in objects:
            print(obj.class_name, obj.track_id, obj.timestamp)
```

When no detector is supplied, the tracker creates one detector and reuses it.
By default its detector confidence is set to ByteTrack's low threshold, 0.10,
so lower-confidence detections can reach ByteTrack's second association
stage. Supplying a higher `detection_confidence`, such as 0.25, is valid but
means detections below 0.25 never reach the tracker.

### `TrackedObject`

The output shape is:

```python
TrackedObject(
    track_id=3,
    class_name="person",
    confidence=0.89,
    bbox=(577.5, 227.1, 841.0, 980.3),
    center=(709.25, 603.7),
    timestamp=5.005,
    frame_number=300,
)
```

`track_id` is persistent within one tracker session. `bbox` uses XYXY pixel
coordinates, `center` is the geometric center, and `timestamp` comes from the
`FramePacket`/caller. `frame_number` is propagated when supplied by
`VideoReader`; it is optional for callers that do not have source frame
metadata. The record remains frame-local; it is not a durable history record.

### `kavach/perception/bytetrack.yaml`

The checked-in KAVACH configuration documents the selected ByteTrack values:

| Setting | Value | Broad meaning |
|---|---:|---|
| `track_high_thresh` | 0.25 | Strong detections used in first association |
| `track_low_thresh` | 0.10 | Lower bound for second-stage recovery |
| `new_track_thresh` | 0.25 | Minimum score for starting a new track |
| `track_buffer` | 30 frames | How long a lost track is retained internally |
| `match_thresh` | 0.80 | Association distance/IoU matching threshold |
| `fuse_score` | true | Combines detection score with matching cost |

`ByteTrackConfig` passes the same explicit values to `BYTETracker`. A custom
configuration can be supplied to `MultiObjectTracker(config=...)`; values are
validated before tracking starts. The default `track_buffer` is measured in
frames, not seconds, so the real-world grace interval depends on the source
FPS.

## Beginner concepts

### Detection versus tracking

Detection looks at one frame at a time and says, “There is a person in this
rectangle.” Tracking compares the current detections with recent tracker
state and says, “This rectangle is probably the same person as before.”

Tracking does not make the detector more accurate. It adds continuity to its
observations.

### What is a Track ID?

A Track ID is an integer label assigned to one ongoing track, for example
`Person #3`. It is not a person's real identity and it is not a face
recognition identity. It only means that this tracker believes several
observations belong to one moving object during this session.

IDs are normally local to a video-processing run. Restarting the tracker can
reuse an ID number for a different object.

### What is association?

Association is the matching step between the tracks known from the previous
frame and the detections found in the current frame. The tracker asks which
new box is the most plausible continuation of each existing track.

The answer uses multiple signals, including predicted position, box overlap,
and confidence. The current Ultralytics ByteTrack implementation performs
IoU/score-based association; Module 3 does not add a separate semantic class
gating layer. Module 3 keeps the association inside ByteTrack rather than
creating an application-level history structure.

### Why does IoU help matching?

IoU (Intersection over Union) compares two boxes:

```text
IoU = intersection area / union area
```

If a person moves only a little, the previous box and current box overlap a
lot, so IoU is high and association is easier. If the person moves far, the
boxes may overlap very little and association becomes uncertain. IoU is a
useful geometric clue, not proof that two boxes belong to the same object.

### What happens during occlusion?

Occlusion means another object or part of the scene temporarily hides an
object. The detector may return no box for the hidden object. ByteTrack keeps
the track in a lost state for up to `track_buffer` frames while its motion
model predicts where it may be. If a compatible detection returns soon
enough, ByteTrack can reactivate the old ID.

The current `MultiObjectTracker` returns only currently active tracked
objects. During a missing frame the hidden object may therefore be absent
from the returned list even though ByteTrack is retaining it internally.

### What is an ID switch?

An ID switch occurs when the tracker assigns an existing ID to the wrong
physical object, or gives the same physical object a new ID after losing it.
Crossing people, identical cartons, crowded scenes, camera motion, poor
detections, and long occlusions all make switches more likely.

The tests include a synthetic crossing case and did not produce an extra
active track or an observed switch in that case. This is not a guarantee:
the crossing objects are synthetic and the tracker has no ground-truth way to
know which identical object is which when their boxes overlap. The real sample
segment contained one visible person, so it cannot measure multi-person ID
switches.

### What does ByteTrack broadly do?

ByteTrack is a tracking-by-detection approach:

1. The detector produces boxes and confidence scores.
2. ByteTrack predicts where existing tracks should be.
3. It first matches strong detections to existing tracks.
4. It then uses remaining lower-confidence detections to recover tracks that
   would otherwise look temporarily missing.
5. Unmatched detections may start new tracks if their score is high enough.
6. Tracks that remain unmatched are kept in a bounded lost buffer and then
   removed.

The implementation used here is Ultralytics' `BYTETracker`; KAVACH supplies
the configuration and converts its output to `TrackedObject` records.

### Why does ByteTrack use lower-confidence detections?

A low-confidence detection can be a weak but real view of an object: perhaps
the person is partly hidden, blurred, or far away. If all low-confidence
detections are discarded immediately, the tracker may create a new ID when
the object becomes clear again. ByteTrack uses low-confidence detections for
matching existing tracks, while using the stronger threshold for starting
new tracks. That helps continuity without allowing every weak box to create
a new object.

### An intuitive Kalman prediction explanation

Imagine watching a carton move from left to right. Even before the next
detection arrives, the tracker can estimate its next position from its recent
position and motion. A Kalman filter is a mathematical way to combine:

- a prediction based on recent motion; and
- a new, imperfect measurement from the detector.

If the new measurement is noisy, the estimate stays closer to the predicted
path. If the prediction is uncertain but the detector is strong, the estimate
moves closer to the new box. This helps association during small gaps, but it
cannot recover an object after an arbitrarily long disappearance or a sudden
unmodeled movement.

### Why does KAVACH require tracking?

KAVACH eventually needs to reason about physical operations over time. A
single frame can show a person and a forklift, but it cannot say whether the
same person approached the forklift, entered a zone, waited there, or left.
Track IDs provide the continuity needed by later temporal modules. Module 3
provides those short-term identities; Module 4 will own explicit long-term
object memory.

## Real sample tracking run

The bundled sample was processed through `VideoReader`, `WarehouseDetector`,
and `MultiObjectTracker`:

```text
input=data/videos/sample_warehouse.mp4
reader_fps=59.94006
resolution=1920x1080
total_frames=4548
duration=75.876s
tracker=bytetrack
track_buffer=30 frames
device=cpu
```

For a sequential 31-frame segment (frames 300–330, approximately 5.005–
5.505 seconds), one visible person produced:

```text
frame=300 time=5.005s → person #1 confidence=0.885
frame=301 time=5.022s → person #1 confidence=0.884
frame=302 time=5.038s → person #1 confidence=0.896
frame=303 time=5.055s → person #1 confidence=0.893
visible_frames=31
unique_ids=[1]
```

No ID switch was observed in this one-person segment. This is an identity
stability smoke test, not a tracking benchmark.

## Performance

The same sequential segment measured end-to-end detector plus ByteTrack update
time on CPU:

```text
timed_frames=31
average_update_time=30.71 ms/frame
approximate_fps=32.57
device=cpu
model=yolo11n.pt
```

This includes Module 2 inference and tracking association, but not Streamlit
rendering. It was measured on the current development environment and will
vary with hardware, model size, input resolution, and runtime versions.

## Tests and validation

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m compileall -q kavach tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The Module 3 tests use the real Ultralytics `BYTETracker` with synthetic
Module 2 detections, so they do not download model weights. They cover:

- one object keeping one ID across frames;
- multiple objects keeping distinct IDs;
- objects crossing;
- a one-frame temporary occlusion and re-identification;
- low-confidence recovery through ByteTrack's second stage;
- invalid timestamps; and
- ID-bearing visualization.

Current combined result for Modules 1–3:

```text
Ran 19 tests in 0.075s
OK
```

## Files changed in Module 3

Added:

- `kavach/perception/tracker.py`
- `kavach/perception/bytetrack.yaml`
- `tests/test_tracking.py`
- `docs/MODULE_03_TRACKING.md`

Modified:

- `kavach/perception/__init__.py` — exports tracking types and visualization.
- `kavach/perception/visualization.py` — adds `draw_tracked_objects`.
- `app/streamlit/app.py` — minimally routes the existing video loop through
  `WarehouseDetector` and `MultiObjectTracker`, preserving the upload, zone,
  alert, and layout behavior while adding explicit IDs.

The upstream tracking notebooks remain unchanged historical examples. The
Streamlit app now uses the explicit KAVACH ByteTrack boundary; no dashboard
redesign was made and the `VideoReader` contract is unchanged.

## Known limitations

- ByteTrack IDs are session-local and are not real-world identities.
- A lost object is not returned during missing frames; only ByteTrack's
  internal bounded buffer retains it for possible reactivation.
- ID switches remain possible during crossings, similar-looking objects,
  long occlusions, camera motion, or detector failures.
- The direct ByteTrack adapter does not add application-level class gating;
  heavily overlapping detections from different classes need careful
  evaluation.
- The current tracker does not log or automatically score ID switches because
  that requires labeled identity ground truth.
- No application-owned trajectory or long-term object history exists yet.
- The tested real video segment contains one visible person, so multi-object
  real-world identity stability remains unverified.
- `track_buffer` is frame-based; a future production configuration should
  translate it deliberately for each camera's FPS.
- The checked-in YAML documents the configuration, while `ByteTrackConfig`
  supplies the actual values to the direct BYTETracker adapter.

Module 3 ends at bounded multi-object tracking. Module 4 is not implemented.
