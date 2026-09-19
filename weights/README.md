# KAVACH detector weights

The repository includes `yolo11n.pt`, a general COCO checkpoint. It is a
baseline, not a validated warehouse detector: its labels do not include a
reliable `forklift`, `carton`, or `pallet` class.

The existing notebook checkpoint is also exposed by the API as the **Worker
Safety** option. Its inspected label map contains only `worker`, so it must not
be used to claim multi-class warehouse accuracy.

## Adding a validated warehouse checkpoint

Before training, prepare and manually annotate a camera-relevant dataset using
[`docs/MODEL_UPGRADE.md`](../docs/MODEL_UPGRADE.md) and
[`data/README.md`](../data/README.md). The repository's frame extractor is a
candidate-image tool only; it never turns current model predictions into
ground truth.

Place a trained, multi-class Ultralytics checkpoint at:

```text
weights/kavach_warehouse.pt
```

Then evaluate it against a real, camera-relevant labelled dataset before
enabling it for operational use:

```powershell
.\.venv\Scripts\python.exe evaluation\evaluate_detector.py `
  --model weights\kavach_warehouse.pt `
  --data data\warehouse.yaml `
  --split val `
  --output reports\warehouse_detection_metrics.json
```

The evaluator reports per-class precision, recall, mAP50, and mAP50-95. It
fails when the checkpoint or dataset is missing; KAVACH does not fabricate
metrics from unlabeled video.

## Optional open-vocabulary experiment

`WarehouseDetector.from_yolo_world(...)` remains available for an explicitly
labelled experiment. To expose it in the web UI, place the compatible
`yolov8s-worldv2.pt` weights at `weights/yolov8s-worldv2.pt`. Open-vocabulary
prompts are useful for exploration, but prompt names are not evidence of
reliable class accuracy. They must be validated with the same evaluator and a
warehouse-specific labelled set.
