# KAVACH Portfolio Release Notes

## Honest project description

KAVACH is a local, explainable video-intelligence prototype for physical
operations. It transforms video frames into detector observations, explicit
ByteTrack identities, bounded temporal object histories, geometry-based scene
relationships, eleven configurable temporal safety behaviour events, general
activity/zone-transition signals, an explainable motion-anomaly signal,
transparent risk and incident records, integrity-checked evidence snapshots,
SQLite storage, short evidence replays, supervisor review, and grounded
natural-language search through a local Ollama model. The React website
exposes analysis, event review, grounded questions, and compact analytics;
Streamlit remains as a fallback.

It is a portfolio and learning system. It is not presented as real-time,
production-ready, metre-calibrated, damage-detecting, or accuracy-validated on
a large warehouse dataset.

## Resume bullets

- Built a modular Python/OpenCV video-intelligence pipeline that connects
  YOLO11n perception, explicit ByteTrack association, timestamped object
  memory, geometry-derived scene relationships, eleven configurable temporal
  safety rules, general activity signals, and explainable motion novelty.
- Designed explainable risk and incident layers that keep detection confidence
  separate from 0–100 risk, retain numeric evidence, debounce repeated
  conditions, and persist reviewable incidents in SQLite.
- Added evidence snapshots with SHA-256 verification, supervisor review states,
  and deterministic root-cause/prevention recommendations.
- Added a strict per-class detector evaluation boundary for precision, recall,
  mAP50, and mAP50-95 without reporting metrics when labelled warehouse data
  is absent.
- Implemented bounded evidence replay and a grounded local Ollama assistant in
  which deterministic SQLite retrieval precedes generation and factual event
  references retain timestamps and clip paths.
- Added a Streamlit supervisor workspace for local/direct-video sources,
  explicit analysis, event filtering/replay, grounded questions, and compact
  behaviour/risk analytics with cached heavyweight resources.
- Added reproducible synthetic behaviour evaluation, temporal ablation tests,
  multi-video integration tests, and bounded CPU benchmark reporting; the
  included sample measured 19.85 detection FPS and 13.46 end-to-end FPS on the
  validated machine configuration.

## Release checklist

- [x] Upstream repository audited and attribution preserved.
- [x] Existing modules covered by unit tests.
- [x] Two synthetic videos traverse the dashboard orchestration path.
- [x] Controlled behaviour dataset and TP/FP/FN evaluator included.
- [x] Temporal ablation example included.
- [x] CPU detection, end-to-end, Python heap, and Ollama measurements recorded.
- [x] Dashboard source, events, assistant, and analytics sections implemented.
- [x] Reproducibility commands documented.
- [x] Limitations and false-positive modes documented.
- [ ] Real camera-specific labeled evaluation dataset.
- [ ] Camera calibration and deployment-scale resource measurements.

The final two unchecked items are intentionally not implied by this release.
