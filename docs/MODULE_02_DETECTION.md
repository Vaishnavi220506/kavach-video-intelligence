# Module 2 — Warehouse Object Perception

## Status and scope

Module 2 adds a small, reusable object-detection boundary around the
repository's existing Ultralytics stack. It accepts one OpenCV/NumPy frame and
returns frame-local detections with a stable KAVACH data shape.

At the completion of Module 2, this module intentionally did not add
tracking, persistent IDs, trajectory
history, behaviour detection, scene graphs, risk reasoning, an event database,
an LLM, or a dashboard redesign. The existing Streamlit application remains
on its upstream `model.track(...)` path. Wiring this new detector into that
application would change its tracking behavior and was deferred to Module 3.

The new perception package is **KAVACH CONTRIBUTION** code. The upstream
repository remains attributed under its MIT license in `LICENSE` and in the
Module 0 audit. This document records a derivative-project change; it does not
replace upstream copyright or attribution.

## Why this boundary exists

The upstream code calls Ultralytics directly from the Streamlit frame loop.
That works for a demonstration, but it makes later pipeline stages depend on
Ultralytics result objects. `WarehouseDetector` gives KAVACH a small contract:

```text
VideoReader
    → FramePacket.frame (OpenCV BGR NumPy array)
    → WarehouseDetector.detect(frame)
    → list[Detection]
    → optional draw_detections(frame, detections)
```

The detector is created once. Its model is loaded once in the constructor and
reused for every subsequent call to `detect`.

## Repository additions

```text
kavach/
├── __init__.py
├── video/                  # Module 1: source, frames, writing
└── perception/             # Module 2: frame-local object perception
    ├── __init__.py
    ├── classes.py          # Warehouse vocabulary and exact class matching
    ├── detector.py         # WarehouseDetector, Detection, benchmark result
    └── visualization.py    # Optional OpenCV box/label drawing
```

### `kavach.perception.classes`

Important values and functions:

- `WAREHOUSE_VOCABULARY`: the target semantic vocabulary being investigated:
  `person`, `cardboard box`, `carton`, `package`, `pallet`, `trolley`,
  `pallet truck`, `forklift`, and `truck`.
- `STANDARD_COCO_WAREHOUSE_CLASSES`: the exact overlap between that target
  vocabulary and the current standard COCO checkpoint: `person` and `truck`.
- `normalize_class_name(...)`: case-insensitive comparison helper.
- `model_class_names(...)`: converts Ultralytics' list/dict class map to one
  consistent dictionary.
- `class_ids_for_names(...)`: resolves only exact names that really exist in
  the loaded model. It does not invent a forklift class from a configuration
  string.

### `kavach.perception.detector`

`WarehouseDetector` owns model initialization, device selection, confidence,
class filtering, result conversion, and a simple inference benchmark.

```python
from kavach.perception import WarehouseDetector
from kavach.video import VideoReader

detector = WarehouseDetector(
    model_path="yolo11n.pt",
    confidence=0.25,
    device="auto",
)

with VideoReader("data/videos/sample_warehouse.mp4") as reader:
    for packet in reader:
        detections = detector.detect(packet.frame)
        for detection in detections:
            print(detection.class_name, detection.bbox, detection.center)
```

The default class filter is deliberately conservative. For the standard
YOLO11n COCO weights it asks Ultralytics for the `person` and `truck` class
IDs, then checks the returned names again before creating a `Detection`.
Passing `allowed_classes=()` disables filtering for an exploratory use case.
Passing a custom model and `allowed_classes=("worker",)` supports the included
one-class checkpoint without renaming it to `person`.

### Standardized `Detection`

Each detection contains:

```python
Detection(
    class_name="person",
    class_id=0,
    confidence=0.908,
    bbox=(352.1, 203.4, 611.7, 843.8),
    center=(481.9, 523.6),
)
```

The coordinates are pixel coordinates in the same frame supplied to
`detect`. `center` is the geometric center of the box. It is not a person's
feet point and it is not a track position. There is deliberately no
`track_id` field in this module.

### `kavach.perception.visualization`

`draw_detections(frame, detections)` returns a copy of the BGR frame with a
colored rectangle and label such as `person 0.91`. It does not alter the input
array and it does not perform inference.

## Beginner concepts

### What is object detection?

Object detection answers two questions for an image:

1. What kinds of objects are present?
2. Where is each object located?

The output is therefore a set of object records. A scene containing two
people produces two records, each with its own box and confidence.

### What does YOLO do?

YOLO means “You Only Look Once.” In the common Ultralytics usage here, one
model call examines one image and predicts candidate boxes, class scores, and
confidence values. Ultralytics then applies post-processing and returns a
result object. KAVACH converts that library-specific result into its own
`Detection` records.

The current repository's default `yolo11n.pt` is a small YOLO11n model trained
on the COCO vocabulary. It is not a warehouse-specific forklift model.

### What is a bounding box?

A bounding box is a rectangle around a detected object. It is an approximate
location, not a pixel-perfect outline. A box can include some background and
can miss part of an object when the view is difficult.

### What are `x1, y1, x2, y2`?

KAVACH uses the XYXY convention:

```text
(x1, y1) = top-left corner
(x2, y2) = bottom-right corner
```

`x` increases from left to right and `y` increases from top to bottom in an
OpenCV image. Thus the box width is `x2 - x1`, and its height is `y2 - y1`.
Coordinates are floating-point values because the neural-network result is
not required to land on whole pixels. Visualization rounds them for drawing.

### What is confidence?

Confidence is the model's score for how strongly a candidate detection fits a
class and a location. A higher value is not a proof of correctness. The
`confidence` constructor argument is a threshold: predictions below it are
discarded by Ultralytics before KAVACH receives them.

Changing the threshold is a precision/recall trade-off. A high threshold can
remove weak true detections; a low threshold can admit more false positives.

### What is a class?

A class is a category name learned by the model, such as `person` or `truck`.
The integer `class_id` is the model's index for that name. Class IDs are only
meaningful together with the model's class map. A class name in an application
configuration does not teach a model a new category.

### What is inference?

Inference is using a trained model to make predictions on new input. In this
module, `detector.detect(frame)` is one inference request. It is different
from training, which changes model weights using labeled examples.

### What is IoU?

IoU means Intersection over Union. It compares two rectangles:

```text
IoU = area of overlap / area of union
```

An IoU of 1 means the boxes are identical; 0 means they do not overlap. IoU
is used when evaluating detections against labels and when deciding whether
two boxes overlap enough to be considered duplicates.

### Conceptually, what is NMS?

Non-Maximum Suppression (NMS) reduces duplicate boxes. If the model predicts
many highly overlapping boxes for one person, NMS keeps the strongest box and
removes weaker boxes whose overlap is high enough. The exact thresholds are
model/runtime settings; Module 2 uses Ultralytics' normal prediction
post-processing rather than reimplementing NMS.

### Detection versus classification

Image classification gives one or more labels for the whole image, such as
“warehouse” or “outdoor.” It does not say where an object is. Object detection
classifies individual objects and supplies a box for each one.

### Detection versus tracking

Detection asks, “What is in this frame?” Tracking adds continuity across
frames: “Is this the same person I saw previously?” A tracker may assign a
persistent ID and maintain motion state. Module 2 intentionally performs only
the first operation. `WarehouseDetector` uses `model.predict(...)`, not
`model.track(...)`, and stores no history.

### Why YOLO is only the perception layer of KAVACH

YOLO can provide observations such as “a person was detected here at this
frame.” It does not, by itself, explain how long the person stayed, whether a
forklift was approaching, whether a zone rule was violated, what evidence
should be retained, or how a supervisor should search the incident later.
Those require the later KAVACH stages. Module 2 establishes a clean visual
observation boundary for them.

## Model investigation and honest capability boundary

### Existing standard model: retained for Module 2 baseline

The repository's `yolo11n.pt` exposes the standard 80-class COCO map. The
target vocabulary overlap is only:

| Warehouse concept | Current standard model | Evidence in the real sample |
|---|---|---|
| person | Exact COCO class | Observed repeatedly |
| truck | Exact COCO class | Not observed in the sampled frames |
| cardboard box / carton | No exact class | Not available from this model |
| package | No exact class | Not available from this model |
| pallet | No exact class | Not available from this model |
| trolley | No exact class | Not available from this model |
| pallet truck | No exact class | Not available from this model |
| forklift | No exact class | Not available from this model |

“Not available” means the current model has no exact output category for the
concept. “Not observed” means the class exists in the model's map but did not
appear in the tested sample frames. Neither statement is a claim of detector
quality on a labeled warehouse benchmark.

### YOLO-World/open-vocabulary investigation

Ultralytics documents YOLO-World as an open-vocabulary model that can receive
text prompts through `set_classes`, which is why it is a plausible future
candidate for warehouse concepts. See the [Ultralytics YOLO-World
documentation](https://docs.ultralytics.com/models/yolo-world).

The optional `WarehouseDetector.from_yolo_world(...)` factory keeps that path
explicit and opt-in. In this environment, initializing the model required an
additional CLIP text encoder and a large auxiliary download. The attempted
CLIP weights failed SHA-256 checksum validation, so no YOLO-World detection
result is reported as a successful test and it was not added as the default
dependency or model. A future evaluation should use pinned dependencies and a
labeled warehouse test set before selecting it.

## Real sample run

The bundled file used was:

```text
data/videos/sample_warehouse.mp4
```

The reader reported:

```text
fps=59.94006
resolution=1920x1080
total_frames=4548
duration=75.876s
device=cpu
model=yolo11n.pt
```

The detector was run at `confidence=0.25` on 16 frames spaced every 300
frames, approximately one frame every five seconds. The unique detected
class was:

```text
['person']
```

Representative standardized output from the same run:

| Frame | Time (s) | Class | Confidence | XYXY box | Center |
|---:|---:|---|---:|---|---|
| 1200 | 20.020 | person | 0.908 | `(352.1, 203.4, 611.7, 843.8)` | `(481.9, 523.6)` |
| 2400 | 40.040 | person | 0.908 | `(1116.6, 189.6, 1314.6, 651.0)` | `(1215.6, 420.3)` |
| 2400 | 40.040 | person | 0.861 | `(766.0, 204.6, 914.2, 714.1)` | `(840.1, 459.4)` |
| 3600 | 60.060 | person | 0.922 | `(465.2, 346.5, 766.8, 1071.0)` | `(616.0, 708.8)` |

This is a smoke test, not an accuracy evaluation. No labels, precision,
recall, mAP, or class-specific reliability claim can be derived from it.

## Basic performance benchmark

The benchmark used the same `yolo11n.pt` model, CPU device, and 1920×1080
sample frames. It warmed up once and timed 10 subsequent calls to
`WarehouseDetector.detect`:

```text
frames=10
average_inference_time=22.13 ms/frame
approximate_inference_fps=45.19
device=cpu
model=yolo11n.pt
```

This is model inference throughput for the tested calls, not end-to-end
Streamlit throughput. Video decoding, display updates, annotation, and
tracking are not included. Performance will change with image size, model
variant, confidence settings, CPU/GPU hardware, and installed runtime.

## Tests and validation

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m compileall -q kavach tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Module 2 adds six unit tests using a fake Ultralytics-compatible model, so
they do not download weights or depend on GPU availability. Together with the
six Module 1 tests, the current suite result is:

```text
Ran 12 tests in 0.032s
OK
```

The real sample was also opened through `VideoReader`, and detector calls
were made on its `FramePacket.frame` values. The Streamlit application was
not redesigned or switched to this detector in this module.

## Files changed in Module 2

Added:

- `kavach/perception/__init__.py`
- `kavach/perception/classes.py`
- `kavach/perception/detector.py`
- `kavach/perception/visualization.py`
- `tests/test_detection.py`
- `docs/MODULE_02_DETECTION.md`

Modified:

- `kavach/__init__.py` — exports the new perception boundary.

At the completion of Module 2, no YOLO weights, tracker configuration,
Streamlit layout, video reader implementation, database, or later KAVACH
intelligence had been changed. Module 3 subsequently owns the tracker
integration.

## Known limitations

- The standard pretrained model is not warehouse-specific and does not detect
  most target warehouse concepts as exact classes.
- The sample run is a smoke test, not a labeled accuracy benchmark.
- The optional YOLO-World path requires extra runtime assets and was not
  validated successfully in this environment.
- Detection results are frame-local. There are no IDs, histories, temporal
  smoothing, or behavior labels.
- The detector expects an already-decoded image array; camera/URL source
  handling remains outside Module 1's local-file reader.
- The benchmark is intentionally simple and should not be treated as a
  deployment capacity measurement.

Module 2 ends at frame-local warehouse object perception. Module 3 is not
implemented here.
