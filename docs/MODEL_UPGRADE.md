# KAVACH model upgrade: warehouse detector

This is the first step toward a warehouse-specific perception model. It adds
the data and measurement discipline required for a real model upgrade; it does
not silently replace the current detector.

## Current truth

The running baseline is still `yolo11n.pt`, a general COCO checkpoint. The old
notebook output at
`notebooks/runs/trained_models/yolo_worker_safety/weights/best.pt` is a custom
one-class `worker` checkpoint trained from an external path that is not
present in this repository. Neither checkpoint provides measured
multi-class warehouse accuracy.

The two local videos were useful for integration testing, but they are not a
labelled evaluation set. Running a detector over a video produces predictions,
not ground truth. Therefore KAVACH currently reports no warehouse precision,
recall, or mAP.

## What was added

- `kavach/perception/dataset.py` validates the actual YAML, image folders, label
  files, class IDs, normalized boxes, split contents, and per-class examples.
- `scripts/prepare_warehouse_dataset.py` extracts timestamped candidate frames
  from one or more local videos and writes a provenance manifest. It never
  creates labels.
- `scripts/validate_warehouse_dataset.py` provides a human-readable or JSON
  preflight report.
- `scripts/train_warehouse_model.py` is a deterministic, Windows-friendly
  Ultralytics training wrapper. It refuses to start if the dataset is
  incomplete.
- `data/warehouse.yaml.example` and `data/README.md` define the canonical
  class contract and annotation rules.

## Canonical warehouse schema

The supervised class order is:

| ID | Class | Annotation note |
|---:|---|---|
| 0 | person | Visible person, including partial visibility when the policy allows it |
| 1 | carton | The canonical label for a cardboard box/carton |
| 2 | package | Use only when the agreed visual definition separates it from carton |
| 3 | pallet | A visible pallet structure |
| 4 | trolley | A manually moved warehouse trolley/cart |
| 5 | pallet_truck | A pallet jack/pallet truck |
| 6 | forklift | Forklift vehicle, including the agreed partial-visibility policy |
| 7 | truck | Road/yard truck or lorry |

Do not use both `cardboard box` and `carton` for the same physical concept.
Do not add a class merely because a prompt or YAML name can be written. A
class is useful only after it has consistent labels and held-out evaluation
examples.

## Reproducible workflow

1. Collect several camera-relevant warehouse videos.
2. Reserve at least one complete video for validation. Do not randomly split
   adjacent frames from the same video across train and validation.
3. Extract candidate frames:

   ```powershell
   .\.venv\Scripts\python.exe scripts\prepare_warehouse_dataset.py `
     --video data\videos\sample_warehouse.mp4 `
     --split train `
     --output data\warehouse_dataset `
     --every-seconds 1

   .\.venv\Scripts\python.exe scripts\prepare_warehouse_dataset.py `
     --video path\to\held_out_video.mp4 `
     --split val `
     --output data\warehouse_dataset `
     --every-seconds 1
   ```

4. Copy the YAML template and manually draw boxes in an annotation tool:

   ```powershell
   Copy-Item data\warehouse.yaml.example data\warehouse.yaml
   ```

5. Validate the labels before training:

   ```powershell
   .\.venv\Scripts\python.exe scripts\validate_warehouse_dataset.py `
     --data data\warehouse.yaml
   ```

6. Train only after validation reports `VALID`:

   ```powershell
   .\.venv\Scripts\python.exe scripts\train_warehouse_model.py `
     --data data\warehouse.yaml `
     --base-model yolo11n.pt `
     --output-dir runs\warehouse
   ```

7. Evaluate the resulting `best.pt` on the held-out video split:

   ```powershell
   .\.venv\Scripts\python.exe evaluation\evaluate_detector.py `
     --model runs\warehouse\yolo11n_warehouse\weights\best.pt `
     --data data\warehouse.yaml `
     --split val `
     --output reports\warehouse_detection_metrics.json
   ```

The evaluator reports per-class precision, recall, mAP50, and mAP50-95. The
numbers become portfolio claims only after the split, class counts, and failure
examples are reviewed.

## Why this is not auto-labeling

The current COCO model can find generic `person` and `truck` observations in
some frames, but it does not reliably understand `forklift`, `carton`, or
`pallet`. Reusing those predictions as ground truth would bake the current
mistakes into the new model and make the evaluation circular. Candidate frames
must therefore be labelled and quality-checked by a human or an independently
reviewed annotation process.

## Current baseline observations

The existing two-video benchmark remains useful for pipeline performance, not
warehouse detection accuracy. The generic model produced only generic COCO
labels on those clips, with inconsistent semantic coverage. In particular,
the system has no evidence yet for reliable forklift/carton/pallet detection.
The measured baseline and event counts remain in `reports/` and
`docs/EVALUATION.md`.

## Compact MVP training runs

On 2026-09-16, a deliberately scoped prototype was trained so the repository
has a real end-to-end learning result without inventing labels for all eight
warehouse classes. The MVP uses two classes: `person` and `carton`.

- Dataset: 15 manually reviewed frames; 10 train / 5 validation images.
- Label scope: 42 people and 28 clearly isolated foreground/handled cartons.
  Ambiguous background cartons were excluded rather than guessed.
- Base checkpoint: `yolo11n.pt`.
- Training: 25 epochs, 640px images, batch 2, CPU, seed 42.
- Checkpoint: `runs/mvp/person_carton_v1/weights/best.pt`.
- Official Ultralytics validation: overall mAP50 `0.393`, mAP50-95 `0.214`.

Per-class validation metrics from `reports/mvp_detection_metrics.json`:

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| person | 0.395 | 0.286 | 0.290 | 0.131 |
| carton | 0.891 | 0.500 | 0.496 | 0.297 |

The fixed operating-point report in
`reports/mvp_confusion_cases_conf005.json` uses confidence `0.05` and IoU
`0.50`: `person` precision/recall/F1 are `0.143/0.143/0.143`, and `carton`
precision/recall/F1 are `0.667/0.500/0.571`. It found 8 false negatives and 7
false positives, with no observed class-confusion pair. At the application's
default confidence `0.25`, this tiny split produced no true-positive match in
the explicit matcher, which is a useful warning against deploying this
checkpoint as the default.

These v1 numbers are a smoke-test measurement, not general warehouse accuracy.
The validation set contains adjacent frames from the same uploaded camera
video plus a few independent-camera person/negative examples. A separate
carton video and a complete held-out camera video are still required for a
portfolio-grade claim.

### v2 retraining result

To improve the low recall, the same carefully scoped dataset was retrained for
100 epochs with patience 30. This changes optimization time only; it does not
add labels or invent warehouse classes.

- Checkpoint: `runs/mvp/person_carton_v2_100e/weights/best.pt`.
- Official Ultralytics validation: overall precision `0.886`, recall `0.578`,
  mAP50 `0.636`, mAP50-95 `0.308`.

Per-class validation metrics:

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| person | 0.772 | 0.429 | 0.528 | 0.220 |
| carton | 1.000 | 0.727 | 0.745 | 0.396 |

At the application's confidence threshold of `0.25`, the explicit fixed-IoU
matcher found the following operating-point results:

| Class | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| person | 2 | 5 | 5 | 0.286 | 0.286 | 0.286 |
| carton | 3 | 0 | 1 | 1.000 | 0.750 | 0.857 |

The v2 checkpoint is now the optional `mvp` choice exposed by the API. The
default `bundled` detector remains unchanged. The v2 result is an improvement
over v1 on this held-out split, but it is not evidence of broad warehouse
generalization: the validation set still contains only five images and includes
adjacent frames from the available local videos.

The confusion report at confidence `0.05` is retained in
`reports/mvp_v2_confusion_cases_conf005.json`. It shows 18 person false
positives and 3 carton false positives at that deliberately permissive
threshold. This is why the app should not lower its confidence threshold just
to increase recall.

The default `bundled` detector remains unchanged, so the narrow prototype does
not silently replace the current general-purpose pipeline.

### Expanded-data experiment (not selected)

I also tested a compact expansion using 49 training images and the same five
manual validation images. The extra training images came from the second local
warehouse video and added 46 visually spot-checked person proposals. This was
kept as a separate dataset in `data/mvp_dataset_v2` so the earlier experiment
could be reproduced unchanged.

The 60-epoch run was stopped after 24 epochs once it was clear that it was not
approaching v2; its best saved checkpoint measured overall precision `0.739`,
recall `0.655`, mAP50 `0.552`, and mAP50-95 `0.274`. At confidence
`0.25`, its fixed matcher produced person precision/recall/F1 of
`0.500/0.571/0.533` and carton precision/recall/F1 of `1.000/0.750/0.857`.

Although its fixed `0.25` operating point found more people than v2, its
official mAP was lower and the added labels are weaker than a complete manual
annotation pass. Therefore v2 remains the selected `mvp` checkpoint in the
application. This experiment confirms that adding labelled diversity is the
right direction, but weak labels and five validation images are not enough to
justify replacing the model. The next quality upgrade should be a larger,
fully manual dataset from independent videos.

The reusable confusion tool is:

```powershell
python evaluation\analyze_confusions.py `
  --model runs\mvp\person_carton_v2_100e\weights\best.pt `
  --data data\mvp.yaml `
  --schema any `
  --split val `
  --device cpu `
  --conf 0.05 `
  --output reports\mvp_v2_confusion_cases_conf005.json
```

The original eight-class `data/warehouse.yaml.example` workflow is still
correctly blocked until real labels exist for person, carton, package, pallet,
trolley, pallet truck, forklift, and truck. The MVP must not be presented as
that eight-class model.

The first failed attempts also exposed two local runtime problems: the
repository virtual environment's Torch DLL was blocked by Windows policy, and
the system Torch installation had mismatched optional Dynamo modules. The
training run succeeded with a matched system CPU stack after pinning Torch and
TorchVision in `requirements.txt`; the application environment should be
recreated from the requirements before restarting the server.

## Known limitations

- No labelled multi-class warehouse dataset is bundled yet.
- The extraction command samples frames; it does not decide what each object
  is and does not balance rare classes automatically.
- Class definitions, partial-object policy, and difficult-case policy need to
  be finalized before annotation.
- Metrics from a single camera or one video will not establish generalization.
- Training a detector will not, by itself, improve tracking, behaviour rules,
  calibration, or risk scoring. Those layers remain separate.

This document is a KAVACH contribution. The upstream notebooks and original
project remain preserved as attribution context; the original MIT license is
still applicable.
