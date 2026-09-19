# KAVACH Module 8: Explainable Risk and Incident Intelligence

## Scope

Module 7 produces structured temporal behaviour events. Module 8 turns those
events into:

~~~text
BehaviourEvent
→ normalized evidence components
→ transparent weighted risk score
→ LOW / MEDIUM / HIGH / CRITICAL category
→ deduplicated in-memory Incident
→ review and lifecycle state
~~~

The implementation contains no LLM, database, new detector, new tracker, or
dashboard redesign. Risk is produced by explicit Python policy rules and YAML
configuration.

## 1. Input boundary

The risk layer consumes a Module 7 BehaviourEvent:

~~~python
{
    "type": "UNSAFE_HUMAN_FORKLIFT_PROXIMITY",
    "timestamp": 82.3,
    "entities": ["person_3", "forklift_1"],
    "confidence": 0.91,
    "evidence": {
        "distance_px": 20.0,
        "maximum_distance_px": 100.0,
        "approaching": True
    }
}
~~~

The event already contains the result of the video, detection, tracking,
memory, geometry, scene-graph, and behaviour layers. Module 8 does not reopen
the video or run YOLO again.

The Module 7 confidence is a rule-based heuristic in the range 0 to 1. It is
not a calibrated probability. The evidence remains available to the risk
engine so the score can be inspected later.

## 2. Repository architecture

~~~text
kavach/
├── risk/
│   ├── __init__.py       # Public Module 8 risk API
│   ├── engine.py         # Validated policy, scoring, categories, breakdown
│   └── config.yaml       # Weights, thresholds, severities, references
└── incidents/
    ├── __init__.py       # Public incident API and state names
    ├── models.py         # Incident and review/lifecycle model
    └── manager.py        # Deduplication, cooldown, updates, transitions

tests/
└── test_risk_incidents.py  # Deterministic synthetic Module 8 tests
~~~

### RiskEngine

~~~python
from kavach.risk import RiskEngine

engine = RiskEngine()
assessment = engine.assess(event, recent_events=previous_events)
~~~

RiskEngine:

- loads the policy once when constructed
- validates weights and category thresholds
- reads only structured event evidence
- returns a RiskAssessment with a score, category, confidence, and breakdown
- has no model or language-model dependency

### RiskAssessment and RiskBreakdown

RiskAssessment contains:

- event_type
- score: a normalized number from 0 to 100
- category: LOW, MEDIUM, HIGH, or CRITICAL
- detection_confidence: the original event confidence from 0 to 1
- breakdown: all component values, weights, and weighted contributions

RiskBreakdown is intentionally serializable with to_dict(). A reviewer can
see which parts of the policy contributed to the final number.

### Incident

Incident stores:

- id: a stable manager-generated ID such as INC-000001
- type
- timestamp: first event timestamp
- last_seen_timestamp
- entities
- risk: the current RiskAssessment
- evidence: initial/latest event and risk-breakdown evidence
- occurrence_count
- status: NEW, REVIEWED, or FALSE_POSITIVE
- lifecycle: OPEN or CLOSED

The manager is intentionally in-memory. It does not create the Module 9
structured event database.

## 3. What is confidence versus risk?

Confidence answers:

> How strongly did the upstream behaviour rule match its available evidence?

For example, the proximity detector may emit confidence 0.91. This is stored
as detection_confidence: 0.91.

Risk answers a different question:

> How concerning is this event under the current safety policy?

Risk includes configured severity and context in addition to confidence. The
two values are deliberately separate:

~~~text
detection confidence = 0.91
risk score            = 50.3
risk category         = HIGH
~~~

The implementation never interprets 0.91 as risk 91. A high-confidence
observation can be low risk, and a lower-confidence observation may still
deserve review if the configured severity and context are high.

## 4. What is severity?

Severity is the policy’s starting concern level for an event type. The
default configuration is intentionally a documented starting policy, not a
claim that every warehouse has the same hazard ranking:

| Behaviour event | Default severity |
| --- | ---: |
| ZONE_VIOLATION | 55 |
| POSSIBLE_DRAGGING | 45 |
| POSSIBLE_DROP | 65 |
| PALLET_OVERHANG | 60 |
| UNSTABLE_STACK | 70 |
| UNSAFE_HUMAN_FORKLIFT_PROXIMITY | 85 |

These values should be reviewed with domain experts and validated against
real incidents before being treated as an operational safety policy.

## 5. Weighted score

Every component is normalized to 0–100. The default weights are:

| Component | Weight | What it represents |
| --- | ---: | --- |
| severity | 0.35 | configured concern for the event type |
| duration | 0.15 | how long a duration-bearing condition lasted |
| motion_intensity | 0.15 | image-space speed or displacement evidence |
| spatial_context | 0.20 | restricted entry, proximity, support, or other spatial cues |
| repeat_frequency | 0.10 | matching prior events in a recent time window |
| detection_confidence | 0.05 | the upstream rule confidence, converted to 0–100 |

The score is:

~~~text
score =
    severity × 0.35
  + duration × 0.15
  + motion_intensity × 0.15
  + spatial_context × 0.20
  + repeat_frequency × 0.10
  + detection_confidence × 0.05
~~~

The weights sum to 1.0, so the result remains bounded from 0 to 100.

### Normalization

Normalization puts different measurements on the same scale. For example:

- duration uses a configured reference of 5 seconds
- speed uses a configured reference of 500 pixels/second
- displacement uses a configured reference of 250 pixels
- a distance cue becomes more concerning as its pixel distance approaches
  zero
- support ratio becomes more concerning as the ratio approaches zero
- confidence 0.91 becomes a confidence component of 91.0, not a final risk
  score of 91

These are image-space and policy references. They are not calibrated metres,
physical force, injury probability, or a universal safety standard.

### Spatial evidence

The initial engine uses the evidence already emitted by Module 7. Examples
include:

- inside a restricted zone → strong spatial-context signal
- near-floor, rapid downward motion, or approaching → contextual cue
- small distance_px relative to maximum_distance_px → stronger proximity cue
- low support_ratio → stronger support-risk cue

The engine does not invent a relationship that was not present in the event.

### Repeat frequency

When recent matching events are supplied, the engine counts previous events
with the same type and entity set inside the configured repeat window. Three
matching previous events reach the default repeat reference of 100.0. This
component is bounded and remains visible in the breakdown.

## 6. Risk categories

The default category thresholds are:

~~~text
0  – 24.999  → LOW
25 – 49.999  → MEDIUM
50 – 74.999  → HIGH
75 – 100     → CRITICAL
~~~

Thresholds are configuration, not hidden logic. A score of 50.3 is therefore
HIGH under the default policy.

Categories are labels for triage. They do not mean that the system has
proved an injury, damage, impact, legal violation, or physical danger.

## 7. From events to stable incidents

Create a manager and ingest events:

~~~python
from kavach.incidents import IncidentManager

manager = IncidentManager(risk_engine=engine)
update = manager.ingest(event)

print(update.incident.id)
print(update.created)
print(update.deduplicated)
~~~

The manager fingerprints an event using:

~~~text
(event type, sorted unique entity IDs)
~~~

For example, both of these share a fingerprint:

~~~text
(UNSAFE_HUMAN_FORKLIFT_PROXIMITY, (forklift_1, person_3))
~~~

If the same fingerprint arrives again while its incident is open:

- within cooldown_seconds, it is merged as detector repetition
- within dedup_window_seconds, it is merged as a continuing incident
- occurrence_count increases
- last_seen_timestamp and latest evidence are updated
- the risk assessment is refreshed
- a previously REVIEWED open incident returns to NEW because new evidence
  needs review

Outside the deduplication window, a new stable incident ID is created. A
FALSE_POSITIVE incident is never merged into; a later occurrence gets a new
incident so the original review decision remains auditable.

This prevents one three-second behaviour event from becoming one incident
per processed frame.

## 8. Incident lifecycle and review

Review status and lifecycle describe different things:

~~~text
Review status: NEW → REVIEWED
              └→ FALSE_POSITIVE

Lifecycle:    OPEN → CLOSED
              └→ OPEN  (reopen)
~~~

The available manager operations are:

- mark_reviewed(id): a human has reviewed the incident
- mark_false_positive(id): reject it as a false positive and close it
- close(id): stop the active lifecycle while preserving review status
- reopen(id): return it to OPEN and NEW for another review
- get(id) and list_incidents(...): retrieve current in-memory records

The system does not automatically mark events as reviewed. Human review is
kept explicit.

## 9. Complete example

Input event:

~~~python
event = BehaviourEvent(
    type="UNSAFE_HUMAN_FORKLIFT_PROXIMITY",
    timestamp=82.3,
    entities=("person_3", "forklift_1"),
    confidence=0.91,
    evidence={
        "distance_px": 20.0,
        "maximum_distance_px": 100.0,
        "approaching": True,
    },
)
~~~

With the default policy, the important normalized values are:

~~~text
severity              = 85.0
duration              = 0.0
motion_intensity      = 0.0
spatial_context       = 80.0
repeat_frequency      = 0.0
detection_confidence  = 91.0
~~~

The weighted contribution is:

~~~text
(85 × .35) + (0 × .15) + (0 × .15) + (80 × .20)
+ (0 × .10) + (91 × .05)
= 50.3
~~~

The result is:

~~~text
RiskAssessment(
    score=50.3,
    category="HIGH",
    detection_confidence=0.91,
)
~~~

The incident manager then creates:

~~~text
INC-000001
type: UNSAFE_HUMAN_FORKLIFT_PROXIMITY
timestamp: 82.3
entities: person_3, forklift_1
status: NEW
lifecycle: OPEN
occurrence_count: 1
~~~

If the same event arrives at 83.3, the manager returns INC-000001 again with
deduplicated=True and occurrence_count=2. It does not create INC-000002.

## 10. Tests and validation

Run the full repository suite:

~~~powershell
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v
.\\.venv\\Scripts\\python.exe -m compileall -q kavach tests
git diff --check
~~~

Module 8 adds deterministic tests for:

- bounded score and separate confidence
- breakdown components and weighted sums
- repeat-frequency evidence
- YAML policy loading and default severity
- cooldown and deduplication
- new incident creation after the deduplication window
- false-positive isolation
- review and lifecycle transitions
- rejection of out-of-order events

Latest validation:

~~~text
Ran 59 tests in 0.104s
OK
~~~

The package also compiled successfully with compileall.

## 11. Files changed

### Added

- kavach/risk/__init__.py
- kavach/risk/engine.py
- kavach/risk/config.yaml
- kavach/incidents/__init__.py
- kavach/incidents/models.py
- kavach/incidents/manager.py
- tests/test_risk_incidents.py
- docs/MODULE_08_RISK.md

### Updated

- kavach/__init__.py: exposes RiskEngine, RiskAssessment, Incident, and
  IncidentManager through the package-level API

No existing video, perception, tracking, memory, geometry, scene-graph,
behaviour detector, Streamlit, or dashboard implementation was redesigned.

## 12. Known limitations

This is an explainable policy layer, not a validated safety certification
system:

- event quality is limited by upstream detection, tracking, geometry, and
  behaviour false positives/false negatives
- default severities and thresholds are starting assumptions and need domain
  validation
- pixels and pixel speeds are not metres or metres/second
- no camera calibration or physical hazard model is used for risk
- confidence is not a calibrated probability
- no learned risk model or LLM is used
- incidents are lost when the Python process stops because persistence is not
  implemented yet
- entity fingerprints can split one real-world object after an upstream ID
  switch
- the manager only sees events supplied to it; it is not yet wired into the
  existing Streamlit processing path
- a score or category does not prove impact, damage, injury, intent, or legal
  responsibility

Module 8 ends at transparent risk assessment and in-memory incident
intelligence. Module 9 adds persistence and replay in a separate document.
