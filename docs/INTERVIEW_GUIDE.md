# KAVACH Interview Guide

## Why YOLO?

YOLO provides a practical one-stage detector with bounding boxes and class
confidence in one inference pass. It is a useful perception baseline for a
local laptop, but its class vocabulary and performance depend on the selected
weights. A COCO checkpoint should not be described as a reliable forklift or
carton detector merely because those names appear in configuration.

## Why ByteTrack?

Detection answers “what is visible in this frame?” ByteTrack associates
detections over time and supplies a track ID. It is lightweight and makes use
of lower-confidence detections during association, which can help recover an
existing object after a weak detector response. IDs can still switch.

## Why temporal memory?

One frame cannot establish duration, displacement, acceleration, separation, or
stationarity. `ObjectMemory` stores timestamped states per track, so KAVACH can
reason about a recent trajectory instead of treating every frame as unrelated.

## What is a scene graph?

A scene graph represents current objects and zones as nodes and their
relationships as directed, evidence-bearing edges. KAVACH derives edges from
geometry and temporal motion rather than asking a black-box classifier to
name relationships.

## How are behaviours detected?

The detectors are explicit rules over `ObjectMemory`, `SceneGraph`, and zones.
The original six safety detectors are supplemented by cautious rules for
throwing, rough handling, aisle obstruction, improper placement, and collision
risk. For example, possible dragging requires a near-floor carton, sustained
horizontal displacement, and enough duration. Possible drop requires a
move/separation/downward-motion/settling pattern and is deliberately worded as
possible, not as proof of physical impact.

## How is warehouse detection validated?

The API separates the bundled COCO baseline, the existing worker-only custom
checkpoint, and a reserved multi-class warehouse checkpoint. The evaluator
computes per-class precision, recall, mAP50, and mAP50-95 from labelled data.
Until a valid warehouse validation set is supplied, KAVACH reports no
warehouse accuracy number.

## How do review and evidence integrity work?

Each event can carry a bounded JPEG snapshot and a SHA-256 digest. The API
verifies the digest before serving a snapshot. A supervisor can move an event
through `NEW`, `REVIEWED`, or `FALSE_POSITIVE`; the history is retained in the
event evidence JSON. This gives future evaluation a reviewed source of truth.

## How are recommendations generated?

Root-cause categories and prevention actions come from
`kavach/prevention/engine.py`. They are deterministic mappings from behaviour
types and evidence to cautious SOP suggestions. The LLM does not invent or
assign them.

## Why rule-based temporal reasoning?

It is inspectable and teachable: every event can show thresholds, timestamps,
distances, speeds, and overlap ratios. It is also appropriate for a small
portfolio baseline. The tradeoff is that rules are sensitive to camera angle,
detector errors, occlusion, and threshold selection.

## Confidence versus risk

Detection confidence describes how strongly the detector supports an observed
class. Risk is a separate weighted score based on severity, duration, motion,
spatial context, repeats, and confidence. A confidence of 0.91 is not a risk
score of 91.

## Why SQLite?

SQLite is local, transactional, dependency-light, and sufficient for a
single-machine prototype. It provides indexed event queries and preserves
structured JSON evidence without introducing a server. It is not presented as
the final choice for large multi-site deployment.

## Why a local LLM?

Ollama allows local natural-language explanations without sending incident
records to a hosted service. It is optional and downstream of the structured
pipeline. Model size and laptop hardware affect latency.

## What is grounding?

Grounding means supplying the model only the retrieved event records, risk
breakdown, statistics, and configured operational rules that are relevant to a
question. KAVACH also appends verified event references itself and rejects a
small set of unsupported outcome/confidence claims.

## Why does the LLM not analyse raw video?

The computer-vision pipeline is responsible for observable facts. Letting a
general language model decide what happened would make counts, timestamps,
objects, and risk evidence difficult to audit. The assistant verbalizes and
searches structured results; it is not the incident detector.

## Main limitations

- The bundled YOLO11n/COCO model has limited warehouse-specific semantics.
- ByteTrack IDs are not guaranteed permanent identities.
- Rules use image-space approximations without universal calibration.
- A monocular view cannot reliably establish height, physical impact, damage,
  or intent.
- The controlled evaluation is synthetic and deliberately small.
- The sample benchmark is a bounded measurement, not a real-time claim.

## What would you improve next?

The highest-value next step would be a labeled, camera-specific evaluation set
and calibrated deployment measurements. Improvements should then be selected
from observed false positives/negatives: better warehouse weights, camera
calibration, occlusion handling, threshold tuning, and only then broader
deployment or model changes.
