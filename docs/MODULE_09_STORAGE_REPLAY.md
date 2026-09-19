# KAVACH Module 9: Event Database and Video Evidence Replay

## Scope

Module 8 creates explainable risk assessments and stable in-memory incidents.
Module 9 gives those records a local home and links them to short pieces of
the original video:

~~~text
Video metadata + Incident
→ SQLite INSERT
→ searchable event row
→ incident timestamp
→ seek to T - 3 s through T + 3 s
→ outputs/clips/<video>_<event>.mp4
→ clip path stored with the event
~~~

SQLite is the first storage backend. The database is deliberately separate
from the later supervisor dashboard, local LLM, and natural-language search
modules.

The implementation does not change YOLO, ByteTrack, ObjectMemory, the scene
graph, behaviour detectors, risk policy, or the Streamlit dashboard.

## 1. Repository architecture

~~~text
kavach/
├── storage/
│   ├── __init__.py       # Public storage API
│   ├── database.py       # SQLite connection, inserts, queries, updates
│   └── schema.sql        # Tables, constraints, foreign keys, indexes
└── incidents/
    └── replay.py         # Bounded OpenCV evidence-clip generation

tests/
└── test_storage_replay.py  # Synthetic database and video tests
~~~

### EventDatabase

~~~python
from kavach.storage import EventDatabase

database = EventDatabase("outputs/kavach.sqlite3")
database.register_video(
    "warehouse-1",
    "/path/to/warehouse.mp4",
    fps=30.0,
    width=1920,
    height=1080,
    total_frames=9000,
    duration=300.0,
)
database.insert_event("warehouse-1", incident)
~~~

EventDatabase:

- creates the SQLite file and schema when constructed
- registers source-video metadata
- stores Module 8 Incident records
- stores full risk and evidence payloads as JSON
- keeps scalar columns for fast filtering
- keeps a normalized entity-link table for entity queries
- updates review status and clip path
- returns ordinary Python dictionaries suitable for later API/UI work

The default database path is outputs/kavach.sqlite3.

### EvidenceReplay

~~~python
from kavach.incidents import EvidenceReplay

replay = EvidenceReplay(database, output_dir="outputs/clips")
result = replay.create_clip("INC-000001")
print(result.clip_path)
~~~

EvidenceReplay opens the source video, calculates a bounded time interval,
seeks to the starting frame, writes only that interval, and associates the
result with the stored event. If the existing clip path is present and the
file is non-empty, the default behavior reuses it instead of encoding again.

## 2. What is SQLite?

SQLite is a small database engine stored in a file. It does not require a
separate database server for this local-first project.

The file contains structured records that can be queried with SQL. KAVACH
uses Python's built-in sqlite3 library, so the first storage layer does not
add another package.

This is useful for the current stage because:

- one local application can read and write it
- the data survives a Python process restart
- tables and indexes make common event queries simple
- the file can later be migrated to a larger database if needed

## 3. Tables, rows, and primary keys

A table is a collection of records with named columns.

The Module 9 schema contains:

### videos

One row represents one source video:

~~~text
video_id | source_path | fps | width | height | total_frames | duration
~~~

video_id is the primary key. A primary key is the value that uniquely
identifies one row.

### events

One row represents one stable Module 8 incident stored for a video:

~~~text
event_id
video_id
timestamp
behaviour
entities_json
risk_score
risk_category
detection_confidence
risk_json
evidence_json
review_status
lifecycle
clip_path
first_seen_timestamp
last_seen_timestamp
occurrence_count
~~~

event_id is the primary key. In the current implementation it is the stable
incident ID, such as INC-000001.

The scalar risk_score, risk_category, and detection_confidence columns make
filtering easy. The complete risk breakdown is also preserved in risk_json,
so a later reviewer does not lose the explanation behind the score.

### event_entities

This small link table stores one row per event/entity relationship:

~~~text
event_id | entity_id
~~~

For example:

~~~text
INC-000001 | person_3
INC-000001 | forklift_1
~~~

This makes get_events_for_entity reliable and indexable without searching
inside a JSON string.

## 4. What is a row and what is INSERT?

A row is one record in a table. For example, one event row could be:

~~~text
INC-000001 | warehouse-1 | 82.3 | UNSAFE_HUMAN_FORKLIFT_PROXIMITY
~~~

INSERT is the SQL operation that adds a row. EventDatabase.insert_event()
uses a parameterized INSERT so values are passed separately from the SQL
text. This is safer than building SQL with string concatenation.

If the same event_id is inserted again, the implementation updates that
event row. This supports storing a Module 8 incident again after its
occurrence count, review state, or latest risk evidence changes.

## 5. What is SELECT and WHERE?

SELECT reads rows. WHERE limits which rows are returned.

The Python methods map to these ideas:

~~~python
database.get_event("INC-000001")
database.get_events("warehouse-1")
database.get_events_by_type("POSSIBLE_DROP")
database.get_events_by_risk("HIGH")
database.get_events_by_risk(minimum_score=75.0)
database.get_events_between_times("warehouse-1", 80.0, 90.0)
database.get_events_for_entity("forklift_1")
database.get_event_statistics("warehouse-1")
~~~

The returned event dictionaries contain:

- event_id and video_id
- timestamp and behaviour
- entities as a decoded list
- risk as a decoded dictionary
- evidence as a decoded dictionary
- review_status and lifecycle
- clip_path
- occurrence count and first/last seen times

Queries are ordered by source-video time except risk queries, which put the
highest numeric risk first.

## 6. What is an index?

An index is an additional lookup structure that helps a database find rows
without scanning every row.

The schema indexes:

- video plus timestamp
- behaviour type
- risk category plus risk score
- review status
- entity ID

Indexes use some storage space, but they make the common KAVACH questions
faster as the event table grows.

## 7. Why store structured evidence as JSON?

Event evidence has different keys for different behaviours:

~~~text
POSSIBLE_DRAGGING:
  horizontal_displacement_px
  duration_seconds
  near_floor

UNSAFE_HUMAN_FORKLIFT_PROXIMITY:
  distance_px
  approaching
  human_speed_px_per_second
~~~

JSON lets the database retain this variable evidence without creating a new
SQL column for every future rule. The important fields used for searching,
such as behaviour, risk category, score, timestamp, and review status, remain
normal SQL columns.

This is a deliberate hybrid:

~~~text
stable query fields → typed SQLite columns
variable explanation fields → JSON text
entity membership → indexed link table
~~~

The JSON is data, not executable code. Module 9 never asks an LLM to infer
what happened from a prose memory of the video.

## 8. Why store events instead of asking an LLM to remember video?

Video is a large time-indexed signal. A language model is not a durable event
database and should not be treated as one.

Storing structured events provides:

- exact timestamps
- stable entity identifiers
- numeric risk values
- preserved detector evidence
- deterministic queries
- repeatable statistics
- review status that survives process restarts
- a direct link to visual evidence

An LLM may later help translate a user question into a database query or
explain a stored result. It should not be the authoritative memory of which
events occurred.

## 9. How evidence clips are generated

For an event at source time T, the default window is:

~~~text
requested start = T - 3 seconds
requested end   = T + 3 seconds
~~~

The window is clamped to the source:

~~~text
actual start = max(0, T - 3)
actual end   = min(video duration, T + 3)
~~~

The times become frame indexes using source FPS:

~~~text
start_frame = floor(actual start × FPS)
end_frame   = floor(actual end × FPS)
~~~

The replay writer then:

1. opens the registered source path with OpenCV
2. reads FPS, dimensions, and frame count
3. seeks to start_frame
4. reads through end_frame only
5. writes those frames with VideoWriter
6. stores the absolute clip path in the event row

At the beginning of a video, the clip starts at frame zero. Near the end,
the clip stops at the last available frame. A short video may therefore
produce a clip shorter than six seconds.

The default output directory is:

~~~text
outputs/clips/
~~~

The filename is derived from video_id and event_id. A caller may provide a
different output path for tests or controlled workflows.

If a non-empty associated clip already exists, create_clip() returns its
metadata with reused=True. Passing force=True deliberately regenerates it.
This avoids re-encoding the source for every repeated replay request.

## 10. Complete example

~~~python
from kavach.incidents import IncidentManager, EvidenceReplay
from kavach.storage import EventDatabase

database = EventDatabase("outputs/kavach.sqlite3")
database.register_video("warehouse-1", "/data/warehouse.mp4")

update = IncidentManager().ingest(behaviour_event)
database.insert_event("warehouse-1", update.incident)

replay = EvidenceReplay(database)
clip = replay.create_clip(update.incident.id)
print(clip.to_dict())
~~~

The resulting stored event has:

~~~text
event_id: INC-000001
video_id: warehouse-1
timestamp: 82.3
behaviour: UNSAFE_HUMAN_FORKLIFT_PROXIMITY
risk: {score, category, detection_confidence, breakdown}
evidence: {distance_px, approaching, ...}
review_status: NEW
clip_path: outputs/clips/warehouse-1_INC-000001.mp4
~~~

## 11. Tests and validation

Run the full repository suite:

~~~powershell
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v
.\\.venv\\Scripts\\python.exe -m compileall -q kavach tests
git diff --check
~~~

Module 9 tests cover:

- SQLite schema creation
- video registration
- event INSERT and update behavior
- JSON evidence and risk round-trip
- queries by video, type, risk, time, and entity
- event statistics
- review-status and clip-path updates
- persistence after closing and reopening the database
- beginning-of-video replay clamping
- end-of-video replay clamping
- clip association
- replay reuse without re-encoding
- invalid event/query handling

Latest validation:

~~~text
Module 9 storage/replay tests: 6 passed
Full suite: 65 tests passed
Compilation: successful
git diff --check: no whitespace errors
~~~

## 12. Files changed

### Added

- kavach/storage/__init__.py
- kavach/storage/database.py
- kavach/storage/schema.sql
- kavach/incidents/replay.py
- tests/test_storage_replay.py
- docs/MODULE_09_STORAGE_REPLAY.md

### Updated

- kavach/incidents/__init__.py: exports replay classes and results
- kavach/__init__.py: exposes EventDatabase and EvidenceReplay

No dashboard redesign or raw-video LLM feature was added. Module 10 is
documented separately as a grounded assistant over stored records.

## 13. Known limitations

- the database is local SQLite, not a multi-user service
- the application does not automatically populate the database from the
  existing Streamlit loop yet; callers must register videos and insert
  incidents explicitly
- source paths must remain available for replay
- OpenCV seeking can be codec-dependent, especially for inter-frame codecs
- replay clips contain source frames only; they are not yet annotated with
  scene-graph or behaviour overlays
- replay uses source FPS and frame-count metadata; unusual variable-frame-rate
  files may need a more precise timestamp index
- risk and behaviour quality still depend on upstream heuristic limitations
- clip files and database rows can be out of sync if a file is manually moved
  or deleted
- no retention policy or background cleanup has been added
- SQLite JSON evidence is convenient for this stage but not a replacement for
  a validated event schema at production scale

Module 9 ends at local structured event persistence and short video evidence
replay. Module 10 consumes these stored records without changing their
provenance.
