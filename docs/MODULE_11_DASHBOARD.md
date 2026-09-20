# Module 11 — Supervisor Dashboard and Video Source Experience

Module 11 adds a supervisor workspace around the existing KAVACH backend. The
primary interface is now a React website served by a small local FastAPI
adapter. The original Streamlit interface remains as a compatibility fallback.
Neither interface reimplements detection, tracking, behaviour rules, risk,
SQLite, replay, or the grounded assistant.

## Surfaces

The bundle serves two surfaces. The public **record** presents the project as
the artifact it produces and is readable without any backend. The **workspace**
is the operator surface and keeps four sections. The section names below were
shortened during the redesign; their responsibilities are unchanged.

1. **Overview** (was ANALYSE) — the reconstruction plate, what the record
   contains, and the highest scoring findings. With a backend, select a local
   upload or direct video URL, start analysis, and inspect metadata and
   progress.
2. **Findings** (was EVENTS) — filter stored incidents by behaviour, risk and
   signal class. Each finding exposes its score components, measured evidence,
   prevention rule, review state, and a replay action.
3. **Ask** (was ASK KAVACH) — send questions to the Module 10 assistant.
   Returned event references retain their real timestamps and can create or
   replay evidence clips. Without a backend the surface still answers a fixed
   set of questions by deterministic retrieval over the loaded record, and
   states that no text was generated.
4. **Analysis** (was ANALYTICS) — event counts by behaviour, risk
   distribution, and the entities the findings name.

## Data source

The workspace does not know whether it is talking to the API or reading the
committed dataset. It asks a source object for records and for whether a
capability is available; a capability the current source cannot provide is
explained at the control rather than failing when pressed. Both paths are
shaped by `kavach/presentation.py`, so the two cannot describe the same event
differently. See the Website section of the README.

## Architecture

```text
local upload / direct HTTP(S) video URL
              ↓
session-owned temporary source file
              ↓
VideoReader → WarehouseDetector → MultiObjectTracker
              ↓
ObjectMemory → SceneGraph → BehaviourRegistry
              ↓
IncidentManager → EventDatabase(SQLite)
              ↓
processed video + on-demand EvidenceReplay clips
              ↓
GroundedAssistant → Ollama only after deterministic retrieval
```

The orchestration is in `kavach/dashboard/analysis.py`. Source validation and
download are in `kavach/dashboard/source.py`. The primary web adapter is in
`app/api/main.py`; the React UI is in `frontend/src/`. The fallback Streamlit
UI is in `app/streamlit/app.py`.

## Source handling

- Uploads are written once to a session-owned temporary directory and are
  identified by a content hash.
- URL input accepts direct `http://` or `https://` resources only.
- URLs with embedded credentials, unsupported schemes, empty hosts, and
  oversized responses are rejected.
- Responses must be video content, `application/octet-stream`, or have no
  content type. An HTML page is rejected before OpenCV is called.
- The download has a bounded byte limit and is streamed in chunks.
- A direct video URL is not the same as a video-platform webpage. Arbitrary
  website extraction is intentionally unsupported.
- OpenCV still performs the final check that the downloaded bytes are a
  readable video stream.

## Web state and performance

The React site uses browser state for the active view, selected video, event,
replay clip, and chat history. Heavy model work remains behind the API and is
started explicitly as a background job. Sending a chat message does not
reanalyse the video. The API caches the detector per model/confidence pair and
keeps one analysis worker so a laptop is not overwhelmed by concurrent runs.

The processed video and replay clips are delivered as HTTP media with byte
ranges. KAVACH prefers an H.264-compatible `avc1` writer for browser playback,
with an OpenCV `mp4v` fallback when the local build lacks that encoder. The
website provides native controls, a position slider, incident markers, and
10-second forward/backward controls.

## Streamlit state and performance

The app uses session state for the selected source, its content key, the active
video ID, the last analysis result, and chat history. Analysis starts only
after the **Analyse video** button is pressed. Sending an Ask KAVACH message
does not invoke video processing again.

`st.cache_resource` keeps the SQLite connection, YOLO weights, detector, replay
service, and assistant objects across Streamlit reruns. The Ollama client does
not load a model on every request; Ollama owns its local model process.

The processed output video is written once per explicit analysis run under
`outputs/processed/`. Replay clips are generated only when requested and are
reused if already present.

## User flow

1. Build the React bundle and start the local API.
2. Choose **Local upload** or **Direct video URL**.
3. Load/select a source and choose the YOLO confidence threshold.
4. Press **Analyse video** once.
5. Press **Analyse video** and watch the job progress.
6. Review the processed video, metadata, tracks, and events.
7. Use **EVENTS** for filtering, evidence detail, review status, and replay.
8. Use **Ask** for evidence-grounded search or explanations; click a
   returned reference to load its replay into the player.
9. Use **Analysis** for compact stored-event summaries and entity relations.

The exact commands are in `docs/WEB_APP.md`. The Streamlit fallback follows
the same source and analysis flow.

## Boundaries

This is a local supervisor prototype. The default zone polygons are
frame-relative demonstration defaults and should be configured for a camera.
Live-stream input is not implemented. Upload and direct URL source files are
temporary session files, so long-term replay requires retaining the source
outside the session. The dashboard does not claim real-time processing,
physical metres, damage detection, or production deployment.
