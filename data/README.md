# KAVACH warehouse detector data

The repository contains demonstration videos, not a labelled multi-class
warehouse training set. The generic `yolo11n.pt` checkpoint and the old
one-class notebook checkpoint must not be treated as warehouse ground truth.

## Canonical labels

The supervised detector uses this exact order:

```text
0 person
1 carton          # cardboard box/carton
2 package
3 pallet
4 trolley
5 pallet_truck
6 forklift
7 truck
```

Use `carton` as the single canonical label for an ordinary cardboard
box/carton. Do not label one object as both `carton` and `package`; define the
visual distinction before annotation and apply it consistently.

## Directory layout

After copying `warehouse.yaml.example` to `warehouse.yaml`:

```text
data/
  warehouse.yaml
  warehouse_dataset/
    images/train/
    images/val/
    images/test/       # optional
    labels/train/
    labels/val/
    labels/test/       # optional
    frame_manifest.jsonl
```

Each image needs a same-stem YOLO label file. An empty label file is valid only
for a deliberately reviewed negative image. A missing label file is treated as
incomplete annotation.

YOLO labels contain one object per line:

```text
class_id center_x center_y width height
```

The four box values are normalized to `0..1`, relative to image width and
height. The class ID must match the YAML order.

## Collection and split policy

Extract frames from several camera-relevant videos, then keep at least one
complete video held out for validation. Randomly splitting adjacent frames
from one clip can leak near-identical images into both train and validation.

```powershell
.\.venv\Scripts\python.exe scripts\prepare_warehouse_dataset.py `
  --video data\videos\sample_warehouse.mp4 `
  --split train `
  --output data\warehouse_dataset `
  --every-seconds 1

.\.venv\Scripts\python.exe scripts\prepare_warehouse_dataset.py `
  --video path\to\held_out_camera_video.mp4 `
  --split val `
  --output data\warehouse_dataset `
  --every-seconds 1
```

The extraction command writes images and source-frame provenance only; it
never invents labels. Manually annotate the images in a bounding-box tool,
review blur/occlusion/partial-object cases, then validate:

```powershell
Copy-Item data\warehouse.yaml.example data\warehouse.yaml
.\.venv\Scripts\python.exe scripts\validate_warehouse_dataset.py `
  --data data\warehouse.yaml
```

Training is allowed only when the validator reports `Status: VALID for the
requested schema`.

## Small MVP experiment

For a lightweight BTech-scale experiment, the repository also includes
`mvp.yaml` and `scripts/build_mvp_dataset.py`. The builder copies a small,
manually reviewed person/carton subset from the local candidate frames and
records its intentionally narrow scope. It is useful for learning the full
train → validate → inspect-errors loop, but it is not a substitute for the
eight-class warehouse dataset:

```powershell
python scripts\build_mvp_dataset.py
python scripts\validate_warehouse_dataset.py --data data\mvp.yaml --schema any
python scripts\train_warehouse_model.py --data data\mvp.yaml --schema any `
  --base-model yolo11n.pt --output-dir runs\mvp --name person_carton_v2_100e `
  --epochs 100 --patience 30
python evaluation\evaluate_detector.py --model runs\mvp\person_carton_v2_100e\weights\best.pt `
  --data data\mvp.yaml --schema any --output reports\mvp_v2_detection_metrics.json
python evaluation\analyze_confusions.py --model runs\mvp\person_carton_v2_100e\weights\best.pt `
  --data data\mvp.yaml --schema any --conf 0.05 `
  --output reports\mvp_v2_confusion_cases_conf005.json
```

The generated MVP dataset is ignored by Git because it is derived from local
video/frame assets. Keep the annotation manifest and measured reports with
any local experiment record.

The compact expansion experiment keeps the original five-image validation set
unchanged and adds training-only person examples from the second warehouse
video. The extra labels are documented as visually spot-checked COCO proposals,
not as a replacement for a fully manual annotation pass:

```powershell
python scripts\build_mvp_dataset.py --expanded
python scripts\validate_warehouse_dataset.py --data data\mvp_v2.yaml --schema any
python scripts\train_warehouse_model.py --data data\mvp_v2.yaml --schema any `
  --base-model yolo11n.pt --output-dir runs\mvp --name person_carton_v3_expanded `
  --epochs 60 --patience 15 --batch 2 --device cpu
python evaluation\evaluate_detector.py --model runs\mvp\person_carton_v3_expanded\weights\best.pt `
  --data data\mvp_v2.yaml --schema any --output reports\mvp_v3_detection_metrics.json
```
