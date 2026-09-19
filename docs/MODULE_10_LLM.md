# KAVACH Module 10: Grounded Local LLM and Natural-Language Video Search

## Scope

Module 10 adds a local-language interface over the structured evidence already
created by Modules 7–9:

~~~text
Video
→ computer vision
→ structured behaviour events
→ SQLite database
→ deterministic retrieval and counting
→ Ollama context
→ natural-language response
~~~

The local LLM does not decide what happened in the video. It does not receive
video frames, run detection, assign track IDs, calculate risk, count rows, or
create timestamps. It can only explain the records that Python retrieves and
places in its prompt.

The default local model is llama3.2:3b. It was already installed in the
development environment, so Module 10 does not download or pull a model
automatically.

No later dashboard, RAG store, cloud model, or new computer-vision feature was
implemented.

## 1. Repository architecture

~~~text
kavach/
├── assistant/
│   ├── __init__.py          # Public Module 10 API
│   ├── ollama_client.py     # Local Ollama HTTP client and benchmark
│   ├── retrieval.py         # SQLite retrieval and grounding context
│   ├── query_router.py      # Deterministic natural-language query routing
│   ├── prompts.py           # Grounding system prompt and message builder
│   └── assistant.py         # Route → retrieve → deterministic/LLM answer
├── storage/
│   └── database.py          # Module 9 list_videos support for video context
└── __init__.py              # Package-level assistant exports

tests/
└── test_assistant.py        # Routing, retrieval, grounding, HTTP parsing
~~~

### OllamaClient

~~~python
from kavach.assistant import OllamaClient

client = OllamaClient(
    host="http://127.0.0.1:11434",
    model="llama3.2:3b",
)
response = client.chat(messages)
~~~

OllamaClient:

- calls the local Ollama JSON API over HTTP
- sends non-streaming chat requests
- reports installed models
- reports whether the local service is reachable
- reports Ollama model allocation and VRAM counters when available
- measures response latency and token counters
- never downloads a model automatically

The host is configurable, but the default is loopback:

~~~text
http://127.0.0.1:11434
~~~

### QueryRouter

QueryRouter converts a small, auditable vocabulary into a QueryIntent. It
does not ask Ollama to decide which database operation to run.

Recognized examples include:

~~~text
Show all dragging incidents
→ EVENTS_BY_TYPE(POSSIBLE_DRAGGING)

Find every time a worker was close to a forklift
→ EVENTS_BY_TYPE(UNSAFE_HUMAN_FORKLIFT_PROXIMITY)

What happened around 12 minutes?
→ EVENTS_AROUND_TIME(timestamp=720 seconds, window=±10 seconds)

Why was Event #32 considered high risk?
→ EVENT_EXPLANATION(INC-000032)

Show incidents involving Carton #12
→ EVENTS_BY_ENTITY(carton_12)

Show all high risk events
→ EVENTS_BY_RISK(HIGH)

Which behaviour occurred most frequently?
→ STATISTICS
~~~

The time-search window is a retrieval rule and is separate from the Module 9
evidence-clip window. A time query uses ±10 seconds by default; a replay clip
uses up to 3 seconds before and after the event.

### RetrievalService

RetrievalService executes a QueryIntent with EventDatabase methods such as:

- get_event
- get_events
- get_events_by_type
- get_events_by_risk
- get_events_between_times
- get_events_for_entity
- get_event_statistics

It returns:

- the retrieved event rows
- total matching row count
- statistics when requested
- whether the context was truncated
- configured operational rules

Full event rows may be limited to 100 records in an LLM context so a large
video does not produce an unbounded prompt. Deterministic totals still come
from SQLite statistics, not from the truncated text context.

### GroundedAssistant

~~~python
from kavach.assistant import GroundedAssistant

assistant = GroundedAssistant(database)
response = assistant.ask(
    "Show every dragging event",
    video_id="warehouse-1",
)

print(response.answer)
print(response.event_references)
~~~

GroundedAssistant:

1. routes the question
2. executes the database retrieval
3. answers deterministic questions in Python
4. sends only retrieved context to Ollama for explanation or summarization
5. appends verified event references, timestamps, and clip paths

The response includes machine-readable event_references, so a later UI can
make each clip clickable without asking the model to invent a link.

## 2. What is an LLM?

An LLM, or large language model, is a neural network trained to predict and
generate sequences of language tokens. It is useful for turning structured
facts into a readable explanation or summary.

An LLM is not automatically a database, video decoder, object detector, or
source of truth. It can produce fluent text even when it does not have enough
evidence. That is why KAVACH places it after computer vision, retrieval, and
deterministic calculations.

## 3. What is a prompt?

A prompt is the input message supplied to the model. Module 10 sends:

- a system message containing behavior and grounding rules
- a user message containing the question
- a JSON retrieval context containing only stored facts

The prompt does not contain raw frames. It does not ask the LLM to watch the
video.

## 4. What is a system prompt?

A system prompt gives the model its role and constraints. KAVACH's system
prompt explicitly says:

- use only supplied event records, risk evidence, statistics, and configured
  operational rules
- never invent events, timestamps, objects, entity IDs, counts, risk evidence,
  clip contents, damage outcomes, injuries, causes, intent, or actions
- do not treat configured class or behaviour names as proof that an event
  occurred
- do not analyze raw video
- do not recount deterministic totals from arbitrary prose
- state that there is insufficient evidence when the context does not support
  a claim
- keep POSSIBLE_DROP wording cautious

It also explicitly says detection confidence is a rule heuristic, not a
probability. The Python application appends the verified references after the
model response.

## 5. What is context?

Context is the information supplied alongside a question so the model can
answer from relevant evidence.

For a risk explanation, context may contain:

~~~json
{
  "events": [
    {
      "event_id": "INC-000032",
      "timestamp": 720.0,
      "behaviour": "UNSAFE_HUMAN_FORKLIFT_PROXIMITY",
      "entities": ["person_3", "forklift_1"],
      "risk": {
        "score": 78.0,
        "category": "CRITICAL",
        "detection_confidence": 0.91,
        "breakdown": {
          "components": {
            "severity": 85.0,
            "spatial_context": 80.0
          }
        }
      },
      "evidence": {
        "distance_px": 20.0,
        "approaching": true
      }
    }
  ]
}
~~~

The model may explain why that stored record has a CRITICAL category. It may
not add another event, change the timestamp, or claim that the person was
injured.

## 6. What is hallucination?

A hallucination is generated text that sounds plausible but is not supported
by the available evidence.

In this project, hallucinations could look like:

- inventing a second forklift
- placing an event at 13:45 when the database has no such row
- changing a possible drop into confirmed impact
- claiming a carton was damaged
- giving a count different from SQLite
- treating confidence 0.91 as a 91% probability

The system prompt reduces this risk, while deterministic retrieval and
programmatic references reduce the model's authority. No prompt can turn a
small local model into a formal proof system, so the limitations remain
explicit.

## 7. What is grounding?

Grounding means tying an answer to supplied evidence instead of allowing the
model to rely on an unsupported story.

KAVACH grounding has three layers:

~~~text
1. QueryRouter chooses a known query shape.
2. RetrievalService obtains authoritative SQLite rows/statistics.
3. Prompts tell Ollama that this context is its only factual source.
~~~

For event lists, Python adds the final verified references:

~~~text
02:43 — INC-000007 — POSSIBLE_DRAGGING — clip: outputs/clips/...
11:29 — INC-000018 — POSSIBLE_DRAGGING — clip: outputs/clips/...
18:04 — INC-000031 — POSSIBLE_DRAGGING — clip: outputs/clips/...
~~~

If clips have not been generated yet, clip_path is null. The event can still
be replayed later through Module 9 EvidenceReplay.

## 8. What is retrieval?

Retrieval is the step that selects relevant stored records before generation.
For example:

~~~text
Question: Show incidents involving Carton #12
→ router extracts entity_id=carton_12
→ database searches event_entities
→ matching events become context
→ Python returns actual event IDs and timestamps
~~~

Retrieval is not generation. It is closer to looking up pages in a filing
system before asking someone to summarize those pages.

## 9. What is RAG?

RAG means Retrieval-Augmented Generation:

~~~text
retrieve relevant information
→ place it in a model context
→ generate an answer constrained by that information
~~~

Module 10 is a deliberately small, local RAG-like pattern. It uses SQLite
retrieval rather than a vector database because the questions are primarily
structured filters over event type, entity, risk, timestamp, and statistics.

Embeddings and semantic vector search are not needed for these first queries
and were not added.

## 10. Why does the LLM not analyze raw video here?

Raw video analysis belongs to the computer-vision pipeline:

~~~text
frames → detection → tracking → temporal memory
→ geometry → scene graph → behaviour event
~~~

That pipeline has access to pixels, FPS, track IDs, and geometric evidence.
The LLM receives none of those raw frames in Module 10. Giving the LLM raw
video would make it harder to guarantee exact timestamps, stable identities,
repeatable counts, and auditable evidence.

The LLM's role is narrower:

~~~text
structured event records → grounded natural-language explanation
~~~

## 11. Why do deterministic queries happen before generation?

Questions such as:

- How many dragging incidents?
- Which behaviour occurred most frequently?
- Show all high-risk events
- Show every dragging event
- Give the review timestamps

have answers that should be computed directly from the database.

Python performs those operations first because:

- SQL counts rows exactly
- filters are repeatable
- event IDs and timestamps come directly from records
- a model cannot accidentally change a count through phrasing
- the answer remains available when Ollama is offline

The LLM is used for prose-heavy tasks such as summarizing a retrieved video
or explaining the stored risk breakdown. Even then, risk values, counts,
timestamps, and event references come from retrieval rather than model
calculation.

## 12. Supported response behavior

### Show every dragging event

This is deterministic. The response lists every matching stored event with
its timestamp, ID, behaviour, and clip path when available:

~~~text
Found 3 stored POSSIBLE_DRAGGING.

Verified event references:
- 02:43 — INC-000007 — POSSIBLE_DRAGGING — clip: ...
- 11:29 — INC-000018 — POSSIBLE_DRAGGING — clip: ...
- 18:04 — INC-000031 — POSSIBLE_DRAGGING — clip: ...
~~~

### Why was Event #32 considered high risk?

Python first retrieves INC-000032 and its risk breakdown. Ollama may then
explain the supplied components. The response still includes the exact
stored score, category, confidence, evidence, and event reference.

### What happened around 12 minutes?

Python converts 12 minutes to 720 seconds and retrieves the configured
±10-second window. Ollama may summarize those retrieved records. The final
response includes actual matching event IDs and source timestamps.

### Summarize this video

Python retrieves video statistics and a bounded list of event records. Ollama
may turn those facts into a readable summary. If the event list was bounded,
the context includes truncated=true so the model cannot treat the sample as
the complete event list.

### Insufficient evidence

If no event matches or the requested video is not uniquely identified, the
assistant returns an insufficient-evidence response. It does not fill the
gap with a plausible warehouse story.

## 13. Local Ollama setup

Ollama must be installed and running locally. Verify the service and model:

~~~powershell
ollama --version
ollama list
~~~

The development environment reported:

~~~text
Ollama version: 0.34.0
Installed model: llama3.2:3b
Model size shown by ollama list: 2.0 GB
Service reachable: yes
Model available: yes
~~~

If the model is missing, install it outside the Python application:

~~~powershell
ollama pull llama3.2:3b
~~~

The KAVACH client intentionally does not run that command automatically.

## 14. Benchmark

The benchmark sends one short local chat request and records:

- end-to-end latency
- prompt token count when Ollama reports it
- response token count when Ollama reports it
- loaded model size and VRAM counters from Ollama /api/ps

Measured on the current development machine:

~~~text
model: llama3.2:3b
service/model availability: available
requests: 1
first-request latency: 42.781 seconds
prompt tokens: 46
response tokens: 2
Ollama total model allocation: 2,554,708,622 bytes
Ollama reported VRAM allocation: 2,554,708,622 bytes
~~~

The first request includes model loading. A warm grounded explanation call
with a 508-token prompt and a 64-token output limit returned:

~~~text
model: llama3.2:3b
warm response latency: 1.240 seconds
prompt tokens: 508
response tokens: 64
~~~

The /api/ps values are Ollama model-allocation counters, not a complete
operating-system RAM measurement. The reported VRAM value is environment
dependent and should not be treated as a hardware-independent guarantee.

Run a benchmark yourself:

~~~python
from kavach.assistant import OllamaClient

client = OllamaClient()
benchmark = client.benchmark(
    [
        {"role": "system", "content": "Reply with exactly OK."},
        {"role": "user", "content": "Reply with exactly OK."},
    ],
    repeats=1,
    options={"temperature": 0, "num_predict": 8},
)
print(benchmark.to_dict())
~~~

## 15. Tests and validation

Run the full repository suite:

~~~powershell
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v
.\\.venv\\Scripts\\python.exe -m compileall -q kavach tests
git diff --check
~~~

Module 10 tests cover:

- supported natural-language query routing
- time conversion from minutes to source seconds
- entity and behaviour extraction
- deterministic results with actual event IDs and timestamps
- database statistics without LLM counting
- risk explanation grounding
- verified references appended to responses
- insufficient-evidence behavior
- Ollama JSON response parsing
- installed-model and resource-counter parsing

Latest validation:

~~~text
Module 10 tests: 9 passed
Full suite before the dashboard/evaluation integration: 74 tests passed
Compilation: successful
git diff --check: no whitespace errors
~~~

## 16. Files changed

### Added

- kavach/assistant/__init__.py
- kavach/assistant/ollama_client.py
- kavach/assistant/retrieval.py
- kavach/assistant/query_router.py
- kavach/assistant/prompts.py
- kavach/assistant/assistant.py
- tests/test_assistant.py
- docs/MODULE_10_LLM.md

### Updated

- kavach/storage/database.py: added list_videos() for unambiguous video
  selection
- kavach/__init__.py: exposed GroundedAssistant and OllamaClient
- kavach/assistant/prompts.py: added explicit non-probability confidence and
  anti-invention rules

No existing CV or dashboard code was redesigned.

## 17. Known limitations

- the query router recognizes a focused vocabulary, not arbitrary language
- the assistant requires SQLite event records; it cannot search an
  unregistered video
- the existing Streamlit path does not automatically populate the Module 9
  database yet
- Ollama must be installed, running, and supplied with the selected local
  model
- first-request latency includes model loading and can be high on a laptop
- the /api/ps resource report is not a full RAM/VRAM profiler
- a prompted LLM can still produce unsupported prose despite grounding rules;
  the application therefore keeps deterministic fields and references
  authoritative
- event references are exact only for records returned by SQLite; no new
  timestamp or object is created from model text
- if the context is truncated, summaries are based on the supplied bounded
  records plus full SQLite statistics, not an unbounded raw event list
- natural-language search does not inspect raw video or discover events that
  the CV pipeline failed to record
- clip paths remain unavailable until Module 9 replay creates and associates a
  clip
- no conversational memory, access control, multi-user service, or audit
  log for model prompts has been added

Module 10 ends at grounded local natural-language retrieval and explanation.
No later module is implemented.
