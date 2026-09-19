# KAVACH Baseline Audit

## Audit status

This document records the Module 0 audit of the upstream repository
`sanaurrehmanarain/forklift-safety-ai` before KAVACH feature development.

- Audit date: 2026-09-13
- Upstream remote: `https://github.com/sanaurrehmanarain/forklift-safety-ai.git`
- Audited commit: `78a1d68a7831f538d2f06e0e1d494685209cc771`
  (`Revise asset download and placement instructions`)
- Working tree state at audit start: two tracked files already modified and
  two downloaded asset files already untracked; those changes were preserved.
- Scope: inspection, runtime verification, and baseline documentation only.
- KAVACH features intentionally not implemented: scene graphs, temporal
  behaviour detection, risk reasoning, incident extraction, evidence replay,
  event database, LLM/search, new tracking architecture, anomaly detection,
  and a new dashboard.

## Attribution boundary

The upstream repository is MIT licensed. The original `LICENSE` file, its
copyright notice, and the upstream citation metadata remain in place. The
existing application, notebooks, assets, model artifacts, and configuration
described below are **UPSTREAM CODE/ARTIFACTS**.

This file is a **KAVACH CONTRIBUTION**: it is an audit and baseline record for
the derivative project. Future KAVACH code should preserve the upstream MIT
notice and clearly identify new files and substantial changes as KAVACH work.
No upstream copyright text was removed or rewritten in Module 0.

## 1. Original project purpose

The upstream project is a small warehouse-safety demonstration. It combines a
Streamlit upload interface, OpenCV frame iteration, Ultralytics YOLO inference
and tracking, and a fixed polygon zone test. The intended user-visible result
is a live annotated video feed with a red safety-zone alert.

The README presents a larger machine-learning lifecycle: sample-video
collection, auto-annotation, dataset preparation, YOLO training, tracking,
zone logic, and deployment. The implementation is primarily a single
Streamlit script plus exploratory/training notebooks; there is no service
layer, persistence layer, test suite, or production pipeline abstraction.

Important correction: the repository does not contain a forklift detector. The
base model is standard YOLO11n trained on COCO, and the included custom model
has one class named `worker`.

## 2. Repository architecture

The committed repository has 46 tracked files. The effective source tree is:

```text
.
├── app/streamlit/app.py
├── assets/                              # README/application screenshots
├── data/dataset.yaml                    # YOLO dataset configuration
├── notebooks/                           # data, training, tracking, and deployment experiments
│   ├── 01_data_collection.ipynb
│   ├── 02_annotation_analysis.ipynb
│   ├── 03_data_analysis.ipynb
│   ├── 04_training.ipynb
│   ├── 05_model_comparison.ipynb
│   ├── 06_tracking.ipynb
│   ├── 07_zone_detection.ipynb
│   ├── 08_evaluation.ipynb
│   ├── 09_error_analysis.ipynb
│   ├── 10_model_optimization.ipynb
│   ├── 11_deployment.ipynb
│   ├── yolo11n.pt
│   └── runs/trained_models/yolo_worker_safety/  # committed training outputs
├── yolo11n.pt                            # root copy of the base weights
├── project_setup.ipynb                   # scaffold-generation notebook
├── requirements.txt
├── Dockerfile
├── docker-compose.yml                    # empty file
├── LICENSE
├── CITATION.cff
└── README.md
```

The README also describes `data/raw`, `data/processed`, `data/labels`,
`outputs`, `reports`, `trained_models`, `configs`, `src`, `tests`, and other
directories. Those directories are not represented in the committed tree
unless they contain files; `project_setup.ipynb` creates many of them when
run. There are no committed Python modules under `src/`, no committed tests,
and no committed application code under `app/fastapi/`.

### File-level map and disposition

| Path | Purpose, important functions, inputs, outputs, dependencies | KAVACH disposition |
|---|---|---|
| `README.md` | Upstream project description, setup instructions, architecture claims, notebook roadmap, screenshots, and attribution. It does not execute the application. | **MODIFY later**: retain upstream attribution, then add a clearly separated KAVACH section and correct claims. Not changed in Module 0. |
| `app/streamlit/app.py` | Only runtime entry point. Defines `load_model(choice)` and top-level Streamlit UI/video loop. Input: MP4 upload and confidence slider. Output: live RGB frames and per-frame status message; no saved output file. Depends on Streamlit, OpenCV, NumPy, Ultralytics. | **KEEP as seed; MODIFY later** to extract reusable stages and correct class/tracker handling. Not changed in Module 0. |
| `requirements.txt` | Declares Streamlit, Ultralytics, headless OpenCV, NumPy, and (in the pre-existing worktree edit) `lap`. | **MODIFY later**: make the environment reproducible and version-compatible. The upstream HEAD says Ultralytics 8.1.0; the worktree currently says 8.4.145. Preserved as found. |
| `Dockerfile` | Python 3.10-slim image, installs requirements, copies the whole repository, exposes 8501, runs Streamlit. | **KEEP initially; MODIFY later** for deterministic model/assets, a `.dockerignore`, health checks, and a maintainable runtime. |
| `docker-compose.yml` | Zero-byte placeholder. It defines no service and cannot provide a Compose deployment. | **REPLACE later** if Compose is needed. |
| `LICENSE` | Upstream MIT license with Sana Ur Rehman Arain copyright notice. | **KEEP unchanged**. Required attribution must remain in derivative copies. |
| `CITATION.cff` | Upstream software citation metadata for `forklift-safety-ai`. | **KEEP now; MODIFY later** only by adding KAVACH derivative metadata without deleting upstream credit. |
| `data/dataset.yaml` | YOLO data config: one `worker` class and train/val paths. Its `path` is an absolute Windows path from another machine. | **MODIFY later** to use portable/configured paths. |
| `yolo11n.pt` | Root copy of pretrained Ultralytics YOLO11n COCO weights. | **KEEP for baseline; replace/registry-manage later**. |
| `notebooks/yolo11n.pt` | Duplicate copy of the base YOLO11n weights used by notebook-relative execution. | **REMOVE later** after consolidating weight location and provenance; do not delete during the audit. |
| `notebooks/runs/trained_models/yolo_worker_safety/weights/best.pt` | Included one-class custom detector checkpoint. `YOLO(...).names` is `{0: 'worker'}`. | **KEEP as baseline evidence; REPLACE later** with reproducible KAVACH model artifacts. |
| `notebooks/runs/trained_models/yolo_worker_safety/weights/last.pt` | Last checkpoint from the recorded training run. | **KEEP as provenance; REPLACE/consolidate later**. |
| `notebooks/runs/trained_models/yolo_worker_safety/args.yaml` | Recorded training arguments and output directory. Contains an absolute path from the original author's machine and `tracker: tracktrack.yaml`. | **KEEP for evidence; MODIFY/replace later** with portable run metadata. |
| `notebooks/runs/trained_models/yolo_worker_safety/results.csv` and PNG/JPG artifacts | Training curves, confusion matrices, labels, batch previews, and validation previews committed as experiment evidence. | **KEEP for baseline; REPLACE later** with reproducible evaluation artifacts and provenance. |
| `assets/*` | Banner, app screenshot, zone screenshots, and worker-distribution image referenced by the README. They are not loaded by the application. | **KEEP as upstream visual reference; MODIFY later** if KAVACH branding is introduced. |
| `project_setup.ipynb` | Creates a proposed directory scaffold and empty placeholder files. It is not needed to run the application. | **REMOVE later** or archive as upstream history once the KAVACH layout is real. |
| `notebooks/01_data_collection.ipynb` | Downloads the Intel sample video if missing and extracts approximately one frame per second to `data/raw`. Uses OpenCV and `urllib.request`. | **KEEP lineage; MODIFY later** for controlled ingestion and provenance. |
| `notebooks/02_annotation_analysis.ipynb` | Loads base YOLO11n, keeps COCO class 0 (`person`) as `worker`, writes YOLO labels, and creates a worker-count histogram. Uses Ultralytics, OpenCV, Matplotlib, Pandas. | **KEEP as historical baseline; MODIFY later** because auto-labels are not ground truth. |
| `notebooks/03_data_analysis.ipynb` | Pairs images/labels, shuffles with seed 42, copies an 80/20 train/val split, and rewrites `data/dataset.yaml` with an absolute path. Uses `os`, `random`, `shutil`, `yaml`. | **MODIFY later** for idempotency, validation, and portable paths. |
| `notebooks/04_training.ipynb` | Detects CUDA/MPS/CPU, loads YOLO11n, trains 10 epochs with batch 8 and image size 640, and displays `results.png`. | **KEEP as baseline lineage; MODIFY later** for a reproducible training entry point. |
| `notebooks/05_model_comparison.ipynb` | Trains YOLO11n and YOLO11s, predicts on validation images, records mAP50 and elapsed inference time, and plots a comparison. | **REPLACE later** with a controlled benchmark. It does not implement the RT-DETR/Faster R-CNN comparison claimed by the README. |
| `notebooks/06_tracking.ipynb` | Reads a fixed sample video, calls `model.track(..., persist=True)`, plots annotated frames, and writes `outputs/tracked_warehouse.mp4`. | **MODIFY later** into a tested tracking component. |
| `notebooks/07_zone_detection.ipynb` | Calls base YOLO11n tracking, reads optional track IDs, tests the bottom-center point against a fixed polygon, draws alerts, and writes `outputs/zone_alert_warehouse.mp4`. | **MODIFY later** as the spatial-logic seed; class filtering and configuration are missing. |
| `notebooks/08_evaluation.ipynb` | Calls `model.val()` and displays a confusion matrix/metrics for the custom model. | **MODIFY later**: current relative weight path does not match the committed `notebooks/runs/...` artifact location. |
| `notebooks/09_error_analysis.ipynb` | Runs low-confidence predictions with `save=True` and `save_crop=True` for visual review. | **MODIFY later**: it saves detected crops, not actual false negatives, and has the same weight-path problem. |
| `notebooks/10_model_optimization.ipynb` | Exports the custom model to ONNX and OpenVINO. | **MODIFY later** after model/version paths and deployment targets are defined. |
| `notebooks/11_deployment.ipynb` | Contains an illustrative FastAPI `/predict` image endpoint in a notebook cell. It is not an application file and is not runnable from the repository as documented without extra packages. | **REPLACE later** with an actual service only when requested by a future module. |

## 3. Current data flow

### Streamlit application path

The actual path in `app/streamlit/app.py` is:

```text
Streamlit starts
  → render model selector, confidence slider, MP4 uploader, and placeholders
  → load_model(choice) at app top level
       → base choice: YOLO("yolo11n.pt")
       → custom choice: YOLO("notebooks/runs/.../weights/best.pt") if present
       → otherwise warn and fall back to YOLO11n
  → uploaded MP4 bytes written to a delete=False temporary file
  → cv2.VideoCapture(temp_file)
  → read width/height and construct a fixed rectangle:
       x = 10%..90% of frame width; y = 30%..95% of frame height
  → for every readable frame:
       model.track(frame, conf=slider_value, persist=True, verbose=False)
       → read results[0].boxes.xyxy
       → for every returned box, use bottom-center point (center_x, y2)
       → cv2.pointPolygonTest(zone_polygon, foot_point)
       → inside: red box and alert flag
       → outside: orange box
       → alert_placeholder.error(...) or .success(...)
       → draw polygon and translucent fill
       → convert BGR → RGB and display in the placeholder
  → release VideoCapture and show "Video Processing Complete!"
```

The application does not pass `tracker=...`, `classes=...`, `save=...`, or an
output path. It does not write an annotated MP4, JSON event, database row, or
incident record. The visible output is the most recently rendered frame and
the most recently rendered status message.

### Notebook tracking/zone path

The notebook path is similar, but `06_tracking.ipynb` uses
`results[0].plot()` and writes `outputs/tracked_warehouse.mp4`, while
`07_zone_detection.ipynb` reads `results[0].boxes.id` when available, adds an
ID label, tests the same bottom-center point, and writes
`outputs/zone_alert_warehouse.mp4`.

## 4. Exact current AI capabilities

| Question | Verified result | Evidence |
|---|---|---|
| Which YOLO model is used? | Base option: `yolo11n.pt`, standard Ultralytics YOLO11n detection checkpoint. Custom option: `notebooks/runs/.../best.pt`. | `app/streamlit/app.py:31-38`; runtime inspection of both weights. |
| Which classes are detected? | Base checkpoint has the 80 COCO classes, including `person`, but no forklift class. Custom checkpoint has exactly `{0: 'worker'}`. | Runtime inspection of `YOLO(...).names`. |
| Is `model.track()` used? | Yes, once per frame in the Streamlit app and in notebooks 06 and 07. | `app/streamlit/app.py:64`; notebook cells. |
| Which tracker is actually used? | In the tested environment (Ultralytics 8.4.145), the default resolved to `tracktrack.yaml` and instantiated `TRACKTRACK`. | `ultralytics.cfg.DEFAULT_CFG.tracker == 'tracktrack.yaml'`; observed `type(model.predictor.trackers[0]).__name__ == 'TRACKTRACK'`. |
| Is ByteTrack explicitly configured? | No. No application call passes `tracker="bytetrack.yaml"`; no project tracker YAML exists. README/notebook wording says ByteTrack, but the code does not enforce it. | App and notebooks; installed Ultralytics tracker source. |
| Are track IDs preserved? | The library produced an ID and kept it across a 30-frame sequential smoke test with `persist=True` (`ID 1` observed on all non-empty frames). The Streamlit app never reads `boxes.id`, so it does not preserve or expose IDs at the application/domain level. Notebook 07 reads IDs when available. | Runtime smoke test; `app/streamlit/app.py:70-76`; notebook 07. |
| Are historical trajectories maintained? | No project-owned trajectory history, per-object buffers, or trajectory output exists. Only the internal tracker state exists between calls. | No history data structure in app/notebooks. |
| Is there temporal reasoning? | No. Each frame independently sets a current alert flag. There is no dwell time, entry/exit transition, debounce, velocity, acceleration, or temporal window. | App loop and zone logic. |
| Is there action recognition? | No. | No action model or action labels in repository. |
| Is there anomaly detection? | No. | No anomaly algorithm, score, or model in repository. |
| Is object state maintained? | Only internal Ultralytics tracker state during sequential calls. No application object state is stored or emitted. | `persist=True`; no domain state structures. |
| Is an LLM used? | No. | No LLM, embedding, RAG, or language-search dependency/code. |
| Is there a database or event store? | No. | No database dependency, schema, or write path. |
| Is there forklift/machinery reasoning? | No. The base COCO model can emit generic COCO detections, but the app does not filter classes and does not model forklift interactions. | Weight names and app loop. |

## 5. Existing features and limitations

### Features that work in the current baseline

- Streamlit page with model selection, confidence threshold, and MP4 upload.
- OpenCV frame-by-frame decoding from a temporary MP4.
- YOLO11n base inference or the included one-class custom detector.
- Ultralytics tracking invocation with `persist=True`.
- Fixed polygon zone construction based on frame dimensions.
- Bottom-center/"feet" point-in-polygon test using OpenCV.
- Per-frame red/orange boxes, colored zone overlay, and Streamlit alert text.
- Notebook examples for frame extraction, auto-labeling, training, tracking,
  zone output, evaluation, export, and an illustrative image API.

### Limitations verified from code

1. **The app treats every base-model detection as a worker.** There is no
   class filter. A base YOLO11n detection of any COCO class can be used in the
   zone test and can trigger the message that a worker entered the zone.
2. **The README overstates tracking.** Tracking is invoked, but the app does
   not display IDs, confidence, class names, or track state. The active tracker
   is not guaranteed to be ByteTrack.
3. **No temporal safety semantics exist.** Alerts are instantaneous per-frame
   booleans and can flicker. There is no incident start/end, persistence
   threshold, cooldown, or event identity.
4. **No forklift detector or interaction model exists.** The custom model is
   one class (`worker`); the base model is COCO and has no forklift category.
5. **The zone is hard-coded.** Users cannot edit polygon vertices, use a
   camera-specific configuration, or define multiple zones.
6. **No processed video is generated by the app.** The README's `outputs/`
   description applies only to notebook examples; Streamlit renders frames in
   memory.
7. **The model paths depend on the current working directory.** Launching
   from the repository root works; launching from elsewhere can cause a
   missing weight error or an unintended download of `yolo11n.pt`.
8. **Uploaded temporary files are not removed.** `NamedTemporaryFile` uses
   `delete=False`, but the app never deletes the resulting file.
9. **Video open/metadata errors are not handled.** There is no explicit check
   for `cap.isOpened()`, zero dimensions, unsupported codecs, or a failed
   writer because the app has no writer.
10. **The runtime is synchronous.** Every frame runs inference and replaces a
    Streamlit image placeholder. There is no FPS control, batching, queue,
    background worker, or back-pressure strategy.
11. **The dependency baseline is inconsistent.** Upstream HEAD declares
    Ultralytics 8.1.0. The pre-existing worktree changes it to 8.4.145 and
    adds `lap`; the installed environment also has `opencv-python` 4.11.0 in
    addition to the declared headless OpenCV 4.9.0.80.
12. **The virtual environment is not self-bootstrapping.** The existing
    `.venv` has the needed packages but no `pip` module. A clean setup must
    create a new environment and install the declared dependencies.
13. **The dataset configuration is machine-specific.** `data/dataset.yaml`
    contains an absolute path from `C:\Users\Yahya\...`, and the recorded
    training `args.yaml` contains another absolute path.
14. **Notebook paths are inconsistent.** The committed run is under
    `notebooks/runs/...`, while notebooks 04, 08, and 09 use paths that assume
    a different working directory/layout.
15. **The Compose file is empty.** `docker-compose.yml` cannot start the
    application.
16. **There are no automated tests.** No tracked test files or test runner
    configuration is present.
17. **The evaluation/error-analysis claims are stronger than the code.** The
    error-analysis notebook saves detections/crops; it does not identify false
    negatives without ground truth. The comparison notebook is unexecuted and
    compares only two YOLO variants, not the architectures named in README.

## 6. Reusable components for KAVACH

### KEEP

- Upstream MIT `LICENSE` and citation/attribution information.
- Streamlit upload interaction as a temporary baseline UX reference.
- OpenCV decode, BGR/RGB conversion, drawing primitives, and polygon math.
- Ultralytics model loading and inference integration as the initial detector
  adapter.
- The sample MP4 as a development fixture after its provenance/license is
  recorded. It is ignored by `.gitignore` and is not a tracked source file.
- Baseline weights and notebook run artifacts as historical evidence, until a
  reproducible KAVACH model/artifact policy replaces them.

### MODIFY

- `app/streamlit/app.py`: extract the frame pipeline, make paths/configuration
  explicit, filter classes, make tracker selection explicit, and add testable
  outputs in a later module.
- `data/dataset.yaml` and notebook path handling: remove machine-specific
  absolute paths and validate data before training.
- `notebooks/01-04.ipynb`: preserve lineage but make data/model provenance and
  reproducibility explicit.
- `notebooks/06_tracking.ipynb` and `07_zone_detection.ipynb`: use the same
  configured detector/tracker and expose per-object data.
- `README.md`: distinguish upstream behavior from KAVACH work and correct
  ByteTrack, forklift, deployment, and notebook claims.
- `requirements.txt`/Dockerfile: pin and test one coherent runtime.

### REPLACE

- Empty `docker-compose.yml` when container orchestration is needed.
- The illustrative `11_deployment.ipynb` cell when a real API/service becomes
  an authorized future module.
- Duplicate model placement (`notebooks/yolo11n.pt`) once a model registry or
  single asset location exists.
- Ad-hoc notebook-generated outputs as the source of truth for future model
  evaluation.

### REMOVE later, with attribution preserved

- `project_setup.ipynb` scaffold code after the real KAVACH structure exists.
- Dead placeholder directories/files if they are still unused.
- Redundant model copies after checksums and provenance are recorded.

No item in these disposition lists was deleted or redesigned in Module 0.

## 7. What KAVACH will add later (not implemented here)

The target architecture may add explicit stages for video ingestion,
detection, multi-object tracking, temporal object memory, spatial reasoning,
a dynamic scene graph, temporal behaviour detection, risk reasoning, incident
extraction, evidence replay, a structured event database, a local LLM,
natural-language video search, and a supervisor dashboard. Those are roadmap
items only. This baseline contains none of them.

## 8. Known technical debt

- Absolute paths in dataset and training metadata.
- Relative model paths tied to launch/notebook working directory.
- Duplicate base weights with no checksum manifest.
- Dependency drift between upstream HEAD, the worktree, and installed OpenCV.
- No lock file, environment bootstrap script, or automated test suite.
- No input validation, codec/error reporting, cleanup, or resource context
  managers in the Streamlit loop.
- Model loaded at top-level and cached by choice; there is no explicit
  lifecycle for tracker reset between uploads or reruns.
- Hard-coded zone geometry and no camera calibration/configuration.
- Base-model detections are not class-filtered before worker safety alerts.
- No confidence/ID/class information is preserved in an event record.
- No output video, structured log, metrics endpoint, or incident artifact from
  the application.
- Notebook outputs are committed while their data inputs are mostly ignored;
  the experiment cannot be independently reproduced from the Git tree alone.
- `docker-compose.yml` is empty and there is no Docker health check.
- README names `11_development.ipynb`, while the actual file is
  `11_deployment.ipynb`.
- The `CITATION.cff` preferred-citation author fields are inconsistent with
  the main author entry and should be reviewed before a KAVACH release.

## 9. Baseline run instructions

These instructions run the upstream application from the repository root.

### Environment

The README requires Python 3.10+. The verified existing environment was:

```text
Python 3.11.16
streamlit 1.32.0
ultralytics 8.4.145
opencv-python-headless 4.9.0.80
opencv-python 4.11.0.86 (also installed in the existing environment)
numpy 1.26.4
lap 0.5.13
```

The environment above reflects the pre-existing working tree edit to
`requirements.txt`; it is not a clean install of the upstream HEAD file,
which pins Ultralytics 8.1.0.

For a clean environment on Windows:

```powershell
git clone https://github.com/sanaurrehmanarain/forklift-safety-ai.git
cd forklift-safety-ai
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run from the repository root so relative model paths resolve:

```powershell
streamlit run app/streamlit/app.py
```

Open `http://localhost:8501`, keep the default **Base Model (Robust)** or
choose **Custom 10-Epoch Model**, and upload an MP4. The application accepts
`.mp4` through the uploader. The sample fixture is expected at
`data/videos/sample_warehouse.mp4`; the repository instructions obtain it
from the upstream release asset workflow.

Required model assets:

- `yolo11n.pt` for the base option. Ultralytics may download it if it is not
  found in the process working directory.
- `notebooks/runs/trained_models/yolo_worker_safety/weights/best.pt` for the
  custom option; otherwise the app falls back to the base model.

Expected output is an in-browser, frame-by-frame annotated feed with a green
or red polygon, orange boxes outside the zone, red boxes inside the zone, and
an `Area Clear.` or `ALERT: Worker detected in restricted forklift zone!`
message. The Streamlit app does not create a processed video file.

## 10. Baseline run result

The upstream application code as present in the audited worktree was started
without any additional source changes:

- Streamlit server started successfully on port 8501.
- `GET http://127.0.0.1:8501` returned HTTP 200 and Streamlit HTML.
- The bundled sample opened successfully with OpenCV: 1920x1080, 59.94 FPS,
  4,548 frames, approximately 75.9 seconds.
- Base YOLO11n loaded successfully and produced `person` detections on sample
  frames.
- A 10-frame `model.track(..., persist=True)` smoke test completed without a
  runtime error.
- A 30-frame sequential tracking smoke test produced stable track ID `1` and
  confirmed the tested runtime tracker as `TRACKTRACK` using
  `tracktrack.yaml`.
- At sample frame 1200, the base model produced one `person` box whose
  bottom-center point was inside the app's fixed polygon, so the zone-alert
  branch was exercised successfully.
- The custom checkpoint loaded successfully and reported one class: `worker`.

This is a startup plus frame-processing smoke baseline, not a claim that a
full 75.9-second upload was benchmarked end-to-end in the browser. The app's
CPU-bound synchronous loop is sufficient for functional verification but has
no measured real-time guarantee.

## 11. Files changed in Module 0

Added by this audit:

- `docs/KAVACH_BASELINE.md` — this KAVACH audit and baseline record.

Intentionally preserved, pre-existing worktree changes (not authored or
rewritten by this audit):

- `app/streamlit/app.py` — `use_container_width` was already changed to
  `use_column_width`.
- `requirements.txt` — Ultralytics was already changed from 8.1.0 to 8.4.145
  and `lap==0.5.13` was already added.
- `.downloads/warehouse_assets.zip` and its extracted sample MP4 — already
  untracked downloaded assets.

No KAVACH runtime feature, new model, new tracker, database, dashboard, or
service was added. Nothing was pushed.
