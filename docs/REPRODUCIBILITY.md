# KAVACH Reproducibility

## Environment

- Python 3.10 or newer; the validated local environment used Python 3.11.
- Windows, Linux, or macOS with an OpenCV-compatible video codec.
- CPU fallback is supported. CUDA is used when PyTorch reports it available.
- Ollama is optional for the dashboard CV pipeline and required only for local
  LLM wording/search questions that are not deterministic database queries.

## Install

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
```

The requirements file pins the main Streamlit, Ultralytics, OpenCV, NumPy,
ByteTrack support, and YAML dependencies. No new Python dependency is needed
for the Module 11 dashboard or Module 10 Ollama client.

The final validation environment reported Python 3.11.16, Streamlit 1.32.0,
OpenCV 4.11.0, Ultralytics 8.4.145, CPU-only Torch 2.14.0, and CUDA
unavailable. OpenCV was already 4.11.0 in that environment while the existing
requirements file pins 4.9.0.80; repeat the benchmark after a clean install
if exact dependency parity is required.

## Model setup

The bundled `yolo11n.pt` is used by default. The optional custom path is
`notebooks/runs/trained_models/yolo_worker_safety/weights/best.pt` when those
weights are present. Model class support must be evaluated; configuration
names alone do not establish warehouse accuracy.

The web UI also exposes `KAVACH Warehouse` when
`weights/kavach_warehouse.pt` exists, and `YOLO-World` when
`weights/yolov8s-worldv2.pt` exists. These are opt-in paths. Copy
`data/warehouse.yaml.example` to `data/warehouse.yaml`, point it at a real
labelled dataset, and run:

```powershell
.\.venv\Scripts\python.exe evaluation\evaluate_detector.py `
  --model weights\kavach_warehouse.pt `
  --data data\warehouse.yaml `
  --output reports\warehouse_detection_metrics.json
```

The evaluator refuses missing or invalid datasets and therefore cannot produce
metrics until that dataset is supplied.

## Warehouse detector data workflow

The supervised model upgrade uses the canonical classes in
`data/warehouse.yaml.example`. Extracting frames is separate from labelling:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_warehouse_dataset.py `
  --video data\videos\sample_warehouse.mp4 `
  --split train `
  --output data\warehouse_dataset `
  --every-seconds 1

.\.venv\Scripts\python.exe scripts\validate_warehouse_dataset.py `
  --data data\warehouse.yaml
```

The second command must report `VALID` after a human has created matching YOLO
label files. Training is then started explicitly:

```powershell
.\.venv\Scripts\python.exe scripts\train_warehouse_model.py `
  --data data\warehouse.yaml `
  --base-model yolo11n.pt `
  --output-dir runs\warehouse
```

Keep validation and test videos separate from training videos. The training
wrapper uses seed 42, deterministic mode, CPU-safe worker settings, and no
image cache by default; record the actual device and checkpoint path from its
summary before comparing models.

## Ollama setup

Install Ollama using its official distribution, start the local service, and
pull the model:

```bash
ollama serve
ollama pull llama3.2:3b
ollama list
```

The client defaults to `http://127.0.0.1:11434` and `llama3.2:3b`. A missing
service produces an explicit fallback/error in the assistant rather than
silently inventing answers.

## Run the React website

The primary supervisor interface is the React website. Build it once so the
local API can serve a self-contained site:

```powershell
cd frontend
npm install
npm run build
cd ..
```

Start the API from the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/`. For frontend development, keep the API on
port 8000 and run `npm run dev -- --host 0.0.0.0 --port 5173` inside
`frontend`, then open `http://localhost:5173/`.

The React site supports local uploads, direct video-file URLs, explicit
analysis, processed video playback, 10-second seeking, event markers,
on-demand evidence replay, grounded questions, and structured analytics.

## Streamlit fallback

The original-compatible Streamlit interface remains available for comparison
and fallback:

```bash
streamlit run app/streamlit/app.py
```

Use the **ANALYSE** tab, select an upload or direct video URL, and explicitly
press **Analyse video**. The bundled sample is at
`data/videos/sample_warehouse.mp4` when the sample asset is present.

## Tests

```bash
python -m unittest discover -s tests -v
python -m compileall -q kavach app tests evaluation
git diff --check
```

## Evaluation and benchmarks

```bash
python -m evaluation.run_controlled_evaluation \
  --output reports/controlled_evaluation.json

python -m evaluation.run_temporal_ablation

python -m evaluation.benchmark_pipeline \
  --video data/videos/sample_warehouse.mp4 \
  --model yolo11n.pt \
  --max-frames 30 \
  --skip-llm
```

The benchmark uses a bounded prefix by default so it is practical on a laptop.
Pass a different `--max-frames` deliberately and report that bound with the
result. The memory measurement uses Python `tracemalloc` only; it excludes
native Torch/OpenCV allocations and full-process RSS.

## Outputs

- `outputs/processed/` — annotated analysis videos.
- `outputs/clips/` — on-demand evidence replay clips.
- `outputs/kavach.sqlite3` — local event database.
- `reports/` — reproducible evaluation/benchmark JSON reports.

Sample configuration files are kept with their owning modules:
`kavach/perception/bytetrack.yaml`, `kavach/behaviours/config.yaml`, and
`kavach/risk/config.yaml`.
