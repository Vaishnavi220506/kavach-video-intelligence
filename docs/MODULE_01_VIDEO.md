# Module 1 — Video and OpenCV Foundation

## Purpose

Module 1 builds the reusable video foundation that later KAVACH modules can
depend on. It reads a local video, exposes metadata and timing, yields frames
one at a time, and can write processed frames to a new video file.

The existing Streamlit dashboard still uses the same YOLO call, zone geometry,
alerts, model choices, and layout. Its upload path now uses `VideoReader` so
the new foundation is exercised without redesigning the dashboard.

This module does not change YOLO, tracking configuration, scene understanding,
behaviour detection, risk reasoning, storage, LLM features, or the dashboard
design.

## 1. What is a video?

A video is a sequence of still images shown quickly one after another. Each
image is called a **frame**. The video also contains timing information that
tells us how quickly those frames should be shown.

For example, a 10 FPS video with 100 frames is approximately 10 seconds long:

```text
duration = total_frames / fps
         = 100 / 10
         = 10 seconds
```

This is an approximation for ordinary constant-FPS video. Variable-frame-rate
video can need more precise presentation timestamps than the basic metadata
available through this module.

## 2. What is a frame?

A frame is one image from the video. OpenCV returns a frame as a NumPy array.
For a normal color video, its shape is:

```text
(height, width, 3)
```

The three channels are stored in BGR order by OpenCV: blue, green, red. When
the existing Streamlit app displays a frame, it converts BGR to RGB because
the display component expects RGB ordering.

`FramePacket` groups the image with the two pieces of timing information that
KAVACH will need later:

```python
packet.frame          # NumPy image array in OpenCV BGR order
packet.frame_number   # zero-based integer: 0, 1, 2, ...
packet.timestamp      # seconds from the beginning of the video
```

## 3. What is FPS?

FPS means **frames per second**. It says how many frames the video is intended
to show in one second.

If `fps == 25`, frame 0 is at 0.00 seconds, frame 1 is at 0.04 seconds, and
frame 25 is at 1.00 seconds. A higher FPS usually gives smoother motion, but
it also means more frames for a computer-vision pipeline to process.

`VideoReader.fps` comes from OpenCV's `CAP_PROP_FPS` property. A zero, negative,
NaN, or infinite FPS is rejected because it cannot produce a meaningful
timestamp.

## 4. What is resolution?

Resolution is the number of pixels in each frame, written as:

```text
width × height
```

For example, the bundled sample video is 1920×1080. That means each frame has
1,920 columns of pixels and 1,080 rows of pixels.

The reader exposes `width` and `height` separately. The writer uses the same
dimensions for every output frame and rejects frames with a different shape,
because a video stream cannot safely mix arbitrary frame sizes.

## 5. How does `cv2.VideoCapture` work?

`cv2.VideoCapture` is OpenCV's interface for opening a camera or video file.
Module 1 uses it for local files:

```python
cap = cv2.VideoCapture("input.mp4")
```

The reader then asks OpenCV for metadata:

```python
fps = cap.get(cv2.CAP_PROP_FPS)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
```

`VideoReader` wraps those calls and validates the values. It raises clear
module errors for a missing file, an unsupported container/codec, invalid FPS,
missing dimensions/frame count, or a stream that cannot produce a frame.

The reader is also a context manager, so the capture is released reliably:

```python
with VideoReader("input.mp4") as reader:
    for packet in reader:
        use(packet.frame)
```

## 6. What does `cap.read()` return?

OpenCV's `cap.read()` returns a pair:

```python
success, frame = cap.read()
```

- `success` is `True` when OpenCV decoded a frame.
- `frame` is the decoded NumPy image when successful.
- At the normal end of a file, `success` becomes `False`.
- If the stream opens but the first frame cannot be decoded, Module 1 raises
  `UnreadableVideoError` instead of silently producing no output.

The equivalent loop inside `VideoReader` adds a zero-based frame number and
timestamp before yielding a `FramePacket`.

## 7. How are timestamps calculated?

For ordinary constant-FPS video, Module 1 uses:

```python
timestamp_seconds = frame_number / fps
```

The frame number is zero-based, so the first frame has timestamp `0.0`.

For a 10 FPS source:

| Frame number | Timestamp |
|---:|---:|
| 0 | 0.00 s |
| 1 | 0.10 s |
| 2 | 0.20 s |
| 10 | 1.00 s |

This formula is deliberately visible in `kavach/video/reader.py` so it can be
explained and tested. Later modules can introduce container timestamps if
variable-frame-rate or synchronized multi-camera input becomes necessary.

## 8. What does `VideoWriter` do?

`cv2.VideoWriter` encodes frames into a new video file. It needs four pieces of
metadata and a codec:

```python
writer = cv2.VideoWriter(
    "processed.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)
```

The Module 1 `VideoWriter` wraps that setup and provides:

```python
with VideoWriter("processed.mp4", fps=10, width=640, height=480) as writer:
    writer.write(bgr_frame)
```

It creates the parent directory when needed, checks that the codec is a
four-character code, verifies that frames are non-empty three-channel arrays,
checks their resolution, counts written frames, and releases the writer.

The codec is important: it determines how frames are compressed and whether a
particular OpenCV installation can create the requested output. If OpenCV
cannot open the writer, Module 1 raises `VideoWriterError`.

## 9. The reader-to-writer processing path

`process_video` demonstrates the complete Module 1 foundation without using
YOLO or any later KAVACH concept:

```text
INPUT VIDEO
    ↓
VideoReader / VideoCapture
    ↓
FramePacket(frame, frame_number, timestamp)
    ↓
annotate_frame_metadata
    ↓
VideoWriter / cv2.VideoWriter
    ↓
OUTPUT VIDEO
```

The default processor draws text such as:

```text
Frame: 24 | Time: 0.40s
```

An application can provide another frame callback later, as long as it
returns a BGR frame with the same width and height. This keeps video I/O
separate from future detection and reasoning stages.

Example:

```python
from kavach.video import process_video

result = process_video("data/videos/input.mp4", "outputs/annotated.mp4")
print(result.frames_written, result.duration)
```

## 10. How this module connects to KAVACH

Video is the first physical input to KAVACH. Before a detector can identify an
object, the system must reliably know which image it is looking at and when
that image occurred.

Module 1 provides that boundary:

- `VideoReader` owns opening and decoding the source.
- `FramePacket` carries the image and basic temporal coordinates.
- `VideoWriter` owns creation of a processed-video artifact.
- `process_video` demonstrates a clean reader → transform → writer pass.
- The Streamlit app reuses `VideoReader` while keeping its upstream behavior.

Later stages can consume a `FramePacket` and attach detections, tracks, or
other derived information without reimplementing `VideoCapture`, frame
numbering, timestamp calculation, or `VideoWriter` setup.

## Files changed

Added for Module 1:

- `kavach/__init__.py`
- `kavach/video/__init__.py`
- `kavach/video/source.py`
- `kavach/video/reader.py`
- `kavach/video/writer.py`
- `kavach/video/processor.py`
- `tests/test_video.py`
- `docs/MODULE_01_VIDEO.md`

Modified:

- `app/streamlit/app.py` — replaced direct `VideoCapture` handling with
  `VideoReader`, added readable video errors, and cleaned up the upload
  temporary file. The dashboard layout and YOLO/zone logic were preserved.

Pre-existing and preserved from Module 0:

- `requirements.txt`
- `.downloads/warehouse_assets.zip`
- `.downloads/extracted/data/videos/sample_warehouse.mp4`

## Architecture changes

The repository now has a reusable video boundary:

```text
kavach/
├── __init__.py
└── video/
    ├── __init__.py
    ├── source.py       # source validation and shared errors
    ├── reader.py       # VideoReader and FramePacket
    ├── writer.py       # VideoWriter and frame validation
    └── processor.py    # metadata annotation and reader-to-writer helper
```

The architecture intentionally does not include YOLO, tracking, zones,
events, storage, or dashboard code inside this package.

## Test commands

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m compileall -q kavach app tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests use a small synthetic MJPG video so they do not depend on the
bundled 75-second sample or on a downloaded model.

## Test results

The Module 1 test suite passed:

```text
Ran 6 tests in 0.077s
OK
```

The passing tests cover:

- metadata extraction;
- zero-based frame numbering;
- timestamp calculation from FPS;
- reader-to-writer processing;
- metadata annotation output;
- invalid source errors;
- zero-FPS errors;
- missing metadata errors; and
- unreadable-stream errors.

The application source also compiled successfully with exit code 0.

The real bundled sample also completed a full reader-to-writer round trip:

```text
input_frames=4548
frames_written=4548
fps=59.94006
duration_seconds=75.876
output_metadata=1920x1080, 4548 frames
```

The real output was written to a temporary file for verification and removed
after the check.

## Example output

For a synthetic 64×48 video containing five frames at 10 FPS:

```text
fps=10.0
resolution=64x48
total_frames=5
duration=0.5s
frame=0 timestamp=0.00s
frame=1 timestamp=0.10s
frame=2 timestamp=0.20s
frames_written=5
output=processed.avi
```

The generated output contains the same five frames with the green metadata
label drawn onto each frame.

## Known limitations

- The reader is intentionally path-based and currently targets local video
  files, not cameras, URLs, or live streams.
- Duration is calculated as `total_frames / fps`; variable-frame-rate timing
  needs a more advanced timestamp source.
- OpenCV metadata can be codec-dependent. Files that do not expose positive
  FPS, dimensions, or frame-count metadata are rejected rather than guessed.
- Codec availability depends on the local OpenCV/FFmpeg build. KAVACH prefers
  `avc1` for browser playback and falls back to `mp4v` when the local build
  cannot create an H.264-compatible stream. `MJPG` remains useful for simple
  local test files.
- The Streamlit application still processes uploaded videos synchronously and
  does not save its annotated feed as an output video. `process_video` provides
  that reusable capability for scripts and future pipeline stages.
- The module does not yet attach detection, tracking, zone, or event data to a
  `FramePacket`; those belong to later modules.

Module 1 ends here. No Module 2 work was started.
