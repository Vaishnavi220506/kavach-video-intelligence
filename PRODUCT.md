# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Three confirmed audiences, in priority order:

1. **Recruiters and interviewers** evaluating the author and this project in a
   hiring or technical-review context. They skim first and read second: they
   need to understand what the system does, why it is technically non-trivial,
   and what is actually proven, within the first screen.
2. **Academic and faculty reviewers** assessing method and rigour — the
   perception/reasoning/risk separation, the controlled evaluation, the
   ablation, and the stated limitations.
3. **Warehouse and operations practitioners** judging whether the workflow
   resembles real supervisory review: what was observed, when, which tracked
   entities, why the risk score, and where the evidence clip is.

All three arrive without context and without a running backend. None of them
will install Python before deciding whether the project is worth their time.

## Product Purpose

KAVACH turns video of physical operations into structured, reviewable evidence:
detections, persistent short-term track identities, timestamped object
histories, geometry-derived relationships, temporal behaviour events,
transparent risk and incident records, short evidence replay clips, and
grounded natural-language search over those records.

Success is a supervisor answering "what happened, when, who was involved, why
was this scored this way, and show me the clip" in seconds, from evidence they
can audit rather than a model's assertion.

## Positioning

Computer vision generates structured evidence **first**; the language model is
strictly downstream of it. Deterministic SQLite retrieval computes counts and
filters, and the LLM receives only already-retrieved records, risk evidence,
statistics, and configured operational rules. When evidence is missing the
assistant reports insufficient evidence rather than generating a plausible
answer.

The claim a neighbouring project cannot truthfully copy is the audit trail:
every risk score decomposes into named contributing components, and every
answer carries verified event/timestamp/clip references back to rows in the
database.

## Operating Context

The full system runs locally because it requires a backend runtime: video
decoding, YOLO inference, SQLite storage, and an Ollama service. The four
working sections of the supervisor interface are:

- **ANALYSE** — ingest a local file or a direct HTTP(S) video URL; inspect
  metadata, progress, tracked objects, events, and the processed video.
- **EVENTS** — filter stored incidents by behaviour, risk, time, or entity;
  generate and replay short evidence clips; set review status.
- **ASK KAVACH** — grounded questions such as "show all dragging incidents" or
  "why was Event #32 considered high risk?".
- **ANALYTICS** — behaviour counts, risk distribution, timeline, evidence graph.

A Streamlit dashboard exists as an original-compatible fallback.

## Capabilities and Constraints

Confirmed stable hand-off types: `VideoReader`→`FramePacket`,
`WarehouseDetector.detect`→`Detection`, `MultiObjectTracker.update`→
`TrackedObject`, `ObjectMemory.update`→bounded `ObjectState` history,
`SceneGraph.update`→evidence-bearing relation edges,
`BehaviourRegistry.detect`→`BehaviourEvent`, `RiskEngine`/`IncidentManager`→
scored `Incident`, `EventDatabase`→SQLite rows and JSON evidence,
`EvidenceReplay.create_clip`→clip plus metadata, `GroundedAssistant.ask`→
answer plus verified references.

Eleven operational behaviour rules, plus activity and image-space anomaly
signals. Risk score and detection confidence are deliberately separate
quantities.

Hard constraints that no future work may soften:

- The default YOLO11n/COCO model has limited warehouse-specific semantics, and
  a model's configured class names do not establish warehouse-class accuracy.
- No valid local eight-class warehouse validation dataset is included.
- Track IDs can switch under occlusion, missed detections, similar objects, or
  camera motion.
- Behaviour thresholds are image-space approximations and camera dependent.
  Pixel distances are not metres without camera-specific calibration.
- A monocular camera does not establish metres, height, impact, damage, injury,
  or intent. "Possible drop" is cautious temporal evidence, not a claim of
  physical impact.
- Default zones are demonstration polygons requiring camera configuration.
- Direct URL support means direct HTTP(S) video resources only, not arbitrary
  webpages or video-platform extraction.
- Live streams, authentication, multi-site operation, and deployment-scale
  monitoring are out of scope.
- Human review remains necessary. Events are evidence for review, not an
  autonomous safety or disciplinary decision.

## Brand Commitments

- Name: **KAVACH** (Sanskrit/Hindi: armour, shield). Existing wordmark treatment
  is the letter `K` in a bordered square with the lockup "KAVACH / VIDEO
  INTELLIGENCE".
- Voice: precise, measured, technical, and conspicuously honest about limits.
  The existing README states what is *not* proven as prominently as what is.
  This restraint is the project's most distinctive quality and is binding —
  hype, rounded-up metrics, or implied production readiness would damage it.
- Existing assets in `assets/` belong to the original zone-alert prototype and
  are retained under a documented provenance boundary. They must not be
  presented as measured KAVACH benchmarks.
- MIT licensed; the `LICENSE` notice must remain with copies and derivatives.

## Evidence on Hand

Real, reproducible artifacts in the repository — these are the only numbers any
surface may cite:

- `reports/controlled_evaluation.json` — 22 synthetic structured-trajectory
  cases, one positive and one negative per behaviour rule; every designed case
  reproduces (TP=1, FP=0, FN=0, precision/recall/F1 = 1.00 per rule). This is a
  rule sanity check, explicitly **not** real warehouse accuracy.
- `evaluation/controlled_dataset.json` and `evaluation/run_controlled_evaluation.py`
  — the trajectory definitions and runner that produce the above.
- `reports/mvp_v2_detection_metrics.json` and `docs/MODEL_UPGRADE.md` — the
  two-class person+carton MVP trained on 15 manually reviewed frames
  (10 train / 5 validation); the 100-epoch checkpoint measured precision 0.886,
  recall 0.578, mAP50 0.636, mAP50-95 0.308. Smoke-test values only.
- `reports/sample_warehouse_benchmark.json`, `reports/pipeline_benchmark.json`,
  `reports/pipeline_benchmark_with_llm.json` — bounded CPU measurements on the
  validated machine (1920×1080 @ 59.94 FPS source, 4,548 frames, 75.88 s;
  YOLO11n detection ~19.85 FPS over five measured frames; end-to-end ~13.46 FPS
  over 30 frames; one grounded Ollama `llama3.2:3b` summary in 8.81 s).
  These do not establish real-time performance or production readiness.
- `reports/temporal_ablation.json` — temporal ablation results.
- `docs/` — 21 module and method documents including `ARCHITECTURE.md`,
  `EVALUATION.md`, `REPRODUCIBILITY.md`, `KAVACH_BASELINE.md` (provenance).

Absences future work must not fabricate: **no bundled sample video**
(`data/videos/sample_warehouse.mp4` is referenced by benchmarks but not
committed), **no committed SQLite database of analysed events**, no real
labelled warehouse video dataset, no users, no testimonials, no customers, no
pricing, no deployment or uptime record, and no hosted backend.

## Product Principles

1. **Evidence precedes explanation.** Structured records are generated by
   vision and geometry; language only describes what is already stored. Any
   surface that implies the LLM watches the video is wrong.
2. **Every score decomposes.** A risk number that cannot be opened into named
   contributing components is not shippable.
3. **State the limit next to the claim.** Honesty about what is unproven is the
   product's differentiator, not a disclaimer to be buried in a footer.
4. **Synthetic is labelled synthetic.** Controlled trajectory cases and any
   reconstruction derived from them are always identified as such, at the point
   of display, never only in small print.
5. **Review stays human.** The system produces evidence for a supervisor's
   decision and never presents itself as the decision.

## Accessibility & Inclusion

No product-specific standard was established by the user. The existing
implementation already commits to a skip link, `aria-pressed`/`aria-label` on
icon controls, visible focus outlines, a `prefers-reduced-motion` block, and a
320px minimum width — these are treated as a floor to preserve, not a ceiling.
