# KAVACH Architecture

KAVACH is an explainable, temporal video-intelligence pipeline built by
progressively extending the upstream `forklift-safety-ai` repository.

## End-to-end data flow

```text
Video source
  → VideoReader / FramePacket
  → WarehouseDetector / Detection
  → MultiObjectTracker / TrackedObject
  → ObjectMemory / ObjectState history
  → geometry + ZoneManager
  → SceneGraph / evidence-bearing RelationEdge
  → BehaviourRegistry / BehaviourEvent
  → RiskEngine + IncidentManager / Incident
  → EventDatabase / SQLite row
  → EvidenceReplay / short clip
  → RetrievalService / QueryIntent results
  → GroundedAssistant / optional Ollama wording
  → Streamlit dashboard
```

The LLM is downstream of computer vision and structured storage. It is not
asked to inspect raw frames or decide what occurred.

## Module interfaces

### Video foundation — `kavach/video/`

- `VideoReader(source)` validates a local file and exposes `fps`, `width`,
  `height`, `total_frames`, and approximate `duration`.
- Iteration yields `FramePacket(frame, frame_number, timestamp)`.
- `VideoWriter(output, fps, width, height, codec)` writes BGR frames with one
  consistent output geometry.
- `process_video` is the small reader-to-writer demonstration helper.

Timestamps are derived from source frame index divided by source FPS. The
reader does not use wall-clock processing speed as video time.

### Perception — `kavach/perception/`

- `WarehouseDetector.detect(frame)` returns frame-local `Detection` values with
  class, confidence, XYXY bounding box, and center.
- `dataset.validate_yolo_dataset(yaml)` audits the separate supervised-model
  data contract before training or accuracy evaluation. Candidate frame
  extraction is kept in `scripts/prepare_warehouse_dataset.py`; it does not
  generate ground truth.
- The model is loaded once by the caller/dashboard cache.
- `resolve_device` selects CUDA when available and falls back to CPU.
- `MultiObjectTracker.update(...)` uses an explicit Ultralytics ByteTrack
  adapter and returns `TrackedObject` values with IDs and source timestamps.

Track IDs are not permanent identities. They can switch under occlusion,
similar-looking objects, missed detections, or camera motion.

### Temporal memory — `kavach/intelligence/object_memory.py`

`ObjectMemory.update(tracked_objects, timestamp, frame_number)` stores bounded
histories by track ID. It derives timestamp-aware displacement, speed in
pixels/second, direction, acceleration where stable, stationary duration, age,
and last-seen time. It does not invent states while an object is missing.

### Geometry and zones — `kavach/intelligence/geometry.py` and `zones.py`

Geometry functions operate on image-space boxes and polygons: centers,
intersection, IoU, overlap, support, relative position, point-in-polygon, and
distance. `ZoneManager` stores named configurable polygons, including the
default staging, loading, restricted, and pallet zone names.

Pixel measurements are not metres without calibration. The optional homography
support assumes a planar ground surface and known point correspondences; it
does not recover object height or general 3-D geometry.

### Dynamic scene graph — `scene_graph.py` and `relationships.py`

`SceneGraph.update(tracked_objects, timestamp, memory)` creates current object
and zone nodes and derives explainable edges such as `near`, `inside_zone`,
`supported_by`, `approaching`, and `moving_with`. Each edge retains numeric
evidence. The graph retains a bounded snapshot history for temporal rules.

### Behaviours — `kavach/behaviours/`

`BehaviourRegistry` evaluates the original six configured safety detectors,
general activity/novelty signals, and five additional cautious rules:

1. zone violation
2. possible dragging
3. possible drop
4. pallet overhang
5. unstable stack
6. unsafe human–forklift proximity
7. possible throwing
8. possible rough handling
9. aisle obstruction
10. improper placement
11. collision risk

The registry also emits configurable `ZONE_TRANSITION`, `OBJECT_ACTIVITY`, and
`MOTION_ANOMALY` signals. These are intentionally separate from safety
incidents: activity describes what changed, while the anomaly signal flags an
unusual image-space motion pattern relative to that track's recent baseline.

Each detector consumes `BehaviourContext`, not raw pixels. `BehaviourDetector`
provides debounce and cooldown so a continuing condition does not create one
incident per frame. The output is a structured `BehaviourEvent` with evidence.

### Risk and incidents — `kavach/risk/` and `kavach/incidents/`

`RiskEngine.assess` combines configured severity, duration, motion, spatial
context, repeat frequency, and detection confidence into a bounded 0–100 risk
score. Detection confidence remains a separate field.

`IncidentManager.ingest` deduplicates repeated behaviour events, maintains
incident lifecycle/review state, and keeps the latest evidence and risk
assessment.

`PreventionEngine` adds deterministic root-cause categories and SOP-oriented
recommendations after behaviour/risk computation. It never asks the LLM to
invent a cause or recommendation.

### Storage and replay — `kavach/storage/` and `kavach/incidents/replay.py`

`EventDatabase` stores videos, incidents, JSON evidence, review status, and
optional clip paths in SQLite. It provides event, type, risk, time, entity, and
statistics queries.

`EvidenceReplay.create_clip(event_id)` seeks directly to the incident window,
clamps it to video boundaries, encodes only the short clip, stores its path,
and reuses an existing clip when possible. `EvidenceStore` captures bounded
JPEG snapshots at event frames and records SHA-256 digests; replay clips also
retain their digest in structured evidence.

### Assistant — `kavach/assistant/`

- `QueryRouter` converts a narrow supported question vocabulary into a
  `QueryIntent`.
- `RetrievalService` executes the intent against SQLite and returns records,
  statistics, and configured operational rules.
- `GroundedAssistant.ask` performs deterministic counts and filtering first.
  Only explanatory, around-time, and summary queries may call Ollama.
- `prompts.py` limits factual context to retrieved records. Unsupported model
  language is rejected and replaced by a deterministic response.
- Event references preserve the actual event ID, timestamp, and clip path.

### Model evaluation — `kavach/perception/evaluation.py`

`evaluate_checkpoint` delegates validation to Ultralytics and preserves
per-class precision, recall, mAP50, and mAP50-95. It requires an actual
labeled dataset YAML and does not reinterpret old training logs as metrics for
a different checkpoint or dataset.

### Dashboard — `app/streamlit/app.py` and `kavach/dashboard/`

The dashboard is a thin supervisor interface. It provides source selection,
explicit analysis, event filtering/replay, grounded chat, and compact
analytics. It also exposes detector choice, evidence snapshot integrity,
review status, and prevention guidance. Heavy resources are cached and chat
reruns do not trigger analysis.

## Persistence boundaries

SQLite is the system of record for stored incidents. Object memory, scene-graph
snapshots, and ByteTrack state are bounded in-memory analysis state. Processed
videos and evidence clips are files referenced by the database.

## Upstream versus KAVACH

The original repository remains the source for its MIT-licensed Streamlit
prototype, YOLO/OpenCV usage, notebooks, sample assets, and original
attribution. KAVACH additions are the reusable module interfaces, explicit
ByteTrack adapter, temporal memory, explainable geometry/scene graph,
behaviour/risk/incident layers, SQLite/replay, grounded local assistant,
dashboard orchestration, tests, evaluation, and documentation. See the root
README and `docs/KAVACH_BASELINE.md` for the audit boundary.
