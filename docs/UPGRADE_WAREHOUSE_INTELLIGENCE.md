# KAVACH Warehouse Intelligence Upgrade

This upgrade adds operational scaffolding without changing the KAVACH pipeline
boundary.

## What is now available

### Warehouse dataset foundation

The repository now includes a strict supervised-dataset workflow in
`docs/MODEL_UPGRADE.md` and `data/README.md`. It can extract timestamped
candidate frames from multiple local videos, but it deliberately does not
auto-label them. The validator requires the canonical eight-class schema,
matching YOLO label files, normalized boxes, train/validation splits, and at
least one labelled example for every class before training can start.

This is important because the current videos are not ground truth. Running the
generic detector over them cannot produce honest precision, recall, or mAP.

### Detector choices and evaluation

The website exposes the local checkpoints that are actually present, plus
optional slots for two warehouse-oriented choices:

- `YOLO11n · COCO baseline` — general-purpose baseline.
- `Worker Safety · custom checkpoint` — the existing checkpoint, whose actual
  vocabulary is `worker` only.
- `KAVACH Warehouse · evaluated checkpoint` — a slot for
  `weights/kavach_warehouse.pt`.
- `YOLO-World · warehouse prompts` — an opt-in open-vocabulary experiment using
  `weights/yolov8s-worldv2.pt`.

Neither name is treated as proof of forklift, carton, or pallet accuracy. A
real multi-class warehouse checkpoint should be evaluated with:

```powershell
.\.venv\Scripts\python.exe evaluation\evaluate_detector.py `
  --model weights\kavach_warehouse.pt `
  --data data\warehouse.yaml `
  --output reports\warehouse_detection_metrics.json
```

The evaluator reports per-class precision, recall, mAP50, and mAP50-95. It
fails clearly when the labeled dataset is missing or invalid. At present the
repository contains no valid local multi-class warehouse validation set, so
no warehouse detection metric is claimed.

### Evidence integrity

Every emitted event gets a bounded JPEG evidence snapshot containing:

- source timestamp and frame number
- controlled file path
- byte size
- SHA-256 digest

Replay clips also store their digest. The API serves a snapshot only when its
current digest matches the stored digest.

### Supervisor review

Stored records support the review states `NEW`, `REVIEWED`, and
`FALSE_POSITIVE`. Each update appends a bounded review history to the event
evidence. The React Events page filters the queue and exposes the transition
controls next to the evidence and prevention explanation.

### Additional behaviours

The registry now includes cautious, configurable rules for:

- `POSSIBLE_THROWING`
- `POSSIBLE_ROUGH_HANDLING`
- `AISLE_OBSTRUCTION`
- `IMPROPER_PLACEMENT`
- `COLLISION_RISK`

They use tracked temporal motion, geometry, zones, and proximity. They do not
claim physical damage, intent, or collision from a single monocular frame.
Aisle obstruction remains silent until an aisle polygon is configured for the
camera.

### Prevention guidance

`kavach/prevention/engine.py` maps event types to transparent root-cause
categories and SOP-oriented actions. It is a deterministic rule layer; it is
not an LLM output and does not replace supervisor review.

## Deliberately deferred

Camera calibration, digital twin, live multi-camera ingestion, and Behaviour
DNA/vector search remain deferred until reliable warehouse detections and
reviewed incidents are available.

## Validation performed

- Python tests: 100 passed.
- Candidate-frame extraction: 76 train frames from the bundled warehouse
  sample and 16 validation candidates from the uploaded throwing/carton clip.
  These images are unannotated and are not accuracy evidence.
- Controlled behaviour evaluation: 22 synthetic cases across 11 rules, with
  TP=1, FP=0, FN=0 for each designed rule case.
- React production build: passed.
- Live sample re-analysis: completed successfully.
- Live snapshot endpoint: HTTP 200 with digest verification.
- Live replay endpoint: clip created/reused and SHA-256 persisted.
- Detector evaluator: correctly refused the repository's missing external
  dataset path instead of fabricating metrics.
