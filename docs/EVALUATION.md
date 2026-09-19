# KAVACH Evaluation and Performance

## Reproducible artifacts

- `evaluation/controlled_dataset.json` contains 22 manually labelled,
  synthetic structured-trajectory cases: one positive and one negative case
  for each of the six original behaviour rules plus five added operational
  rules.
- `evaluation/run_controlled_evaluation.py` runs each case through the actual
  memory, geometry/scene-graph, and behaviour interfaces and calculates TP,
  FP, FN, precision, recall, and F1.
- `evaluation/run_temporal_ablation.py` demonstrates frame-local presence
  versus a timestamped possible-dragging event.
- `evaluation/benchmark_pipeline.py` measures actual local YOLO and
  end-to-end processing on a chosen video prefix and optionally benchmarks
  Ollama.

## Controlled behaviour results

The evaluated dataset is synthetic and intentionally small. Results are
case-level rule sanity checks, not detector accuracy or warehouse performance.

| Behaviour | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Zone violation | 1 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| Possible dragging | 1 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| Possible drop | 1 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| Pallet overhang | 1 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| Unstable stack | 1 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| Human–forklift proximity | 1 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| Possible throwing | 1 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| Possible rough handling | 1 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| Aisle obstruction | 1 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| Improper placement | 1 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| Collision risk | 1 | 0 | 0 | 1.00 | 1.00 | 1.00 |

These numbers are reproducible on the included synthetic cases only. There is
no labeled real warehouse behaviour dataset in this repository, so no real
video precision, recall, or F1 is claimed.

## Measured local performance

The bundled sample metadata is 1920×1080, 59.94 source FPS, 4,548 frames, and
approximately 75.88 seconds.

| Measurement | Configuration | Result |
|---|---|---:|
| Object detection | YOLO11n, CPU, five measured frames after one warmup | 19.85 FPS average; 50.38 ms average inference |
| End-to-end CV | YOLO11n, CPU, first 30 source frames | 13.46 FPS; 2.23 seconds; 30 frames processed |
| Grounded Ollama | `llama3.2:3b`, one summary request | 8.81 seconds |
| Ollama model allocation | `/api/ps` counters | 2,554,708,622 bytes model size and VRAM counters |
| Python heap tracing | `tracemalloc`, 30-frame benchmark | 111,233,651 bytes peak after end-to-end |

The detection and LLM measurements are machine- and warmup-dependent. The
end-to-end result is a bounded prefix measurement, not a claim that the system
keeps up with a 59.94 FPS stream or operates in real time. The Ollama memory
figure is not full process RAM.

## Temporal ablation

For one synthetic carton trajectory, every frame contains a `carton` detection
and an XYXY box. A frame-level system can report presence and per-frame
coordinates, but it does not have a duration or displacement conclusion. The
temporal KAVACH path accumulates four timestamped states and emits:

```text
POSSIBLE_DRAGGING
timestamp: 1.5 s
horizontal displacement: 150 px
duration: 1.5 s
near floor: true
```

This is the intended contribution of memory plus temporal rules; it is not
proof of a human action or physical cause.

## False-positive analysis

The following are concrete failure modes of the current evidence model. They
are documented risks, not claimed frequencies:

1. A carton moving horizontally on a conveyor can satisfy the near-floor and
   displacement rule and be reported as `POSSIBLE_DRAGGING`, even without a
   person. Person proximity is optional by default.
2. A one-frame detector miss can remove a support or near relation. When the
   object reappears lower and stationary, that interruption can resemble the
   `POSSIBLE_DROP` pattern.
3. A person and forklift can be close in image pixels but far apart in the
   warehouse because of perspective. Conversely, a distant pair can project
   close together. No universal metre threshold is assumed.
4. Overlapping projected boxes can make an object look supported or
   unsupported even when the physical 3-D arrangement differs.
5. The restricted-zone detector uses the bottom-center point. A loose box,
   partial occlusion, camera tilt, or a poorly configured polygon can move the
   point across a boundary without a meaningful operational change.
6. A fast conveyor or camera shake can satisfy the throwing/rough-handling
   motion thresholds without a human handling action.
7. Aisle obstruction and improper placement are only meaningful after the
   supervisor configures camera-specific polygons for aisle and approved
   placement areas.

The appropriate response is camera-specific labeling, calibration where
valid, threshold measurement, and review of evidence clips—not stronger prose
from the LLM.

## Cross-video smoke benchmark

Two real videos were run through the same YOLO11n CPU configuration with a
bounded 300-frame prefix. These are reproducibility checks, not accuracy
metrics:

| Video | Source | Detection FPS | End-to-end FPS | Events | Observation |
|---|---|---:|---:|---:|---|
| `sample_warehouse.mp4` | 1920×1080, 59.94 FPS | 34.04 | 18.36 | 4 | `fire hydrant`, `person`, and `sports ball` in this prefix; no reliable warehouse-specific class claim |
| `Throwing_seating_cartons_using_strap_to_hold.mp4` | 1280×720, 30 FPS | 35.85 | 13.37 | 25 | `person`, `truck`, `book`, `keyboard`, and `tennis racket`; COCO labels still do not prove cartons or throwing |

The detailed JSON artifacts are written to
`reports/sample_warehouse_benchmark.json` and
`reports/throwing_cartons_benchmark.json`, including unique class/track counts
and event-type counts. The second filename describes the source file, not a
validated throwing detection.
