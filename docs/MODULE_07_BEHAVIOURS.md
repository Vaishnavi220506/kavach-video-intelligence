# KAVACH Module 7: Temporal Behaviour Intelligence

## Scope

Module 7 is the first KAVACH contribution that turns tracked object histories and scene-graph relations into named temporal events.

The implementation focuses on six behaviors, in this order:

1. Zone violation
2. Possible dragging
3. Possible drop
4. Pallet overhang
5. Unstable stack
6. Unsafe human-forklift proximity

These detectors consume structured KAVACH state:

~~~text
ObjectMemory + SceneGraph + Geometry + Zones
→ explicit temporal/spatial rules
→ debounced BehaviourEvent objects
~~~

They do not consume raw video whenever the structured state is available. No learned action model, risk engine, incident database, LLM, or dashboard redesign is included.

## 1. Frame-level versus temporal reasoning

A frame-level detector asks:

> What is visible in this one frame?

For example, a detector can see that a person bounding box is inside a polygon.

Temporal reasoning asks:

> What changed across several timestamped observations?

For example:

~~~text
t=10.0 s: carton is moving near a person
t=11.0 s: carton remains near the person
t=12.0 s: carton has moved horizontally while remaining near the image floor
→ POSSIBLE_DRAGGING
~~~

One frame can be noisy or ambiguous. A time sequence provides duration, direction, displacement, acceleration-like changes, and relation transitions.

## 2. Behaviour event contract

Every emitted event is a BehaviourEvent:

~~~python
{
    "type": "POSSIBLE_DRAGGING",
    "timestamp": 12.0,
    "entities": ["carton_7", "person_3"],
    "confidence": 0.72,
    "evidence": {
        "near_floor": True,
        "horizontal_displacement_px": 60.0,
        "vertical_displacement_px": 0.0,
        "duration_seconds": 2.0,
        "person_nearby": True
    }
}
~~~

The event contract contains:

- type: stable uppercase event name
- timestamp: source-video timestamp in seconds
- entities: graph node IDs involved in the event
- confidence: a rule-based heuristic score, not a calibrated probability
- evidence: measurements and conditions used by the detector

Evidence is deliberately retained so a supervisor, test, or later explanation layer can inspect why an event was emitted.

## 3. Repository architecture

~~~text
kavach/behaviours/
├── __init__.py          # Public Module 7 API
├── base.py              # Context, event, candidate, detector, debounce
├── registry.py          # Ordered detector registry and YAML loading
├── zone_violation.py    # Restricted polygon membership
├── dragging.py          # Near-floor horizontal movement
├── drop.py              # Possible move-separate-downward-settle pattern
├── overhang.py          # Low support ratio over pallet-like provider
├── stacking.py          # Multi-object stack with weak support link
├── proximity.py         # Moving human/forklift close pair
└── config.yaml          # Configurable thresholds and class names
~~~

Module 7 extends the existing visualization module with draw_scene_graph, which draws current object nodes, relation arrows, and configured zones for debugging.

## 4. State machine and debouncing

A detector condition may be true for many consecutive frames. The base BehaviourDetector turns a continuous condition into a rising-edge event:

~~~text
condition false
    ↓
condition becomes true
    ↓ debounce duration
emit one event
    ↓
condition remains true: emit nothing
    ↓
condition becomes false: reset active state
~~~

Each detector has:

- debounce_seconds: how long a condition must persist before emission
- cooldown_seconds: minimum time before the same entity key can emit again

Therefore a three-second dragging condition produces one POSSIBLE_DRAGGING event, not one event for every processed frame.

The registry calls every detector on every structured update, including frames where no matching object is present. That absence is what resets an active condition.

## 5. Behavior 1: zone violation

ZoneViolationDetector reads inside_zone edges from SceneGraph. The default target is RESTRICTED_ZONE, and the monitored classes are configurable.

Rule:

~~~text
object --inside_zone--> restricted_zone
→ ZONE_VIOLATION
~~~

The graph relation was created using the configured polygon and the tracked object's bottom-center point. Evidence includes the zone name, bottom-center coordinates, boundary distance, track ID, and detector confidence.

This is a geometric zone violation event. It does not decide whether the entry was authorized, whether a gate was open, or whether an injury occurred.

## 6. Behavior 2: possible dragging

DraggingDetector checks a recent ObjectMemory window for a draggable object whose:

- current bounding box is near the image floor
- horizontal displacement exceeds a configurable pixel threshold
- vertical displacement stays below a configurable limit
- movement lasts at least a configurable duration
- optional nearby person relation exists

Near-floor is defined using frame height when available:

~~~text
bottom_y >= frame_height × near_floor_fraction
~~~

If frame height is unavailable, the detector can use a configured absolute pixel fallback. The location is still an image-space approximation, not a measured ground height.

The event is named POSSIBLE_DRAGGING because a moving low box can also be caused by camera motion, detector jitter, occlusion, a pushed object, or a carried object viewed from an unusual angle.

Evidence includes horizontal and vertical displacement, duration, near-floor state, person proximity, and the history window.

## 7. Behavior 3: possible drop

PossibleDropDetector looks for a cautious four-state pattern:

~~~text
1. object was moving
2. a near/support/moving-with relation disappeared
3. object moved rapidly downward in image coordinates
4. object then became slow or stationary
~~~

The detector measures three consecutive motion intervals from four timestamped states:

- pre-motion speed
- downward velocity and displacement
- post-motion speed
- abrupt speed reduction

Image y increases downward, so positive vertical velocity means downward image motion.

The emitted event is POSSIBLE_DROP. It does not claim physical impact, damage, or a confirmed dropped load. Monocular image evidence cannot reliably distinguish all drops from camera movement, occlusion, detector-box changes, or an object passing behind another object.

The detector requires a disappeared relation by default. That relation may be a prior near, supported_by, or moving_with edge from SceneGraph.

## 8. Behavior 4: pallet overhang

OverhangDetector compares an object above a pallet-like support provider. The provider class list is configurable and defaults to pallet, platform, shelf, floor, and pallet_truck.

Rule:

~~~text
object is above provider
provider is within the configured vertical gap
support_ratio < minimum support ratio
→ PALLET_OVERHANG
~~~

The support ratio is the fraction of the top object's width horizontally covered by the provider. Evidence includes:

- support ratio
- horizontal overlap in pixels
- left and right overhang in pixels
- vertical gap in pixels
- widths of both boxes

This is an improper-support geometry cue. It does not prove that the object is physically resting on the pallet or that it will fall.

## 9. Behavior 5: unstable stack

UnstableStackDetector builds short vertical chains from adjacent stackable objects. For each adjacent pair it checks:

- upper object is above lower object
- vertical gap is small
- horizontal overlap is positive
- support ratio is below the stable-support threshold for at least one link

The default detector requires three stackable objects so it does not label every two-box overlap as a stack. The minimum stack height and class list are configurable.

Example:

~~~text
carton_1
    supported by a weak horizontal overlap with carton_2
carton_2
    supported by carton_3
carton_3
→ UNSTABLE_STACK
~~~

Evidence records stack height, each support ratio, each vertical gap, each horizontal overlap, and the number of weak links.

The event is a geometric warning cue. It does not estimate load weight, friction, center of gravity, or physical stability.

## 10. Behavior 6: unsafe human-forklift proximity

HumanForkliftProximityDetector searches current object nodes for configured human and forklift-like classes.

It requires:

- bottom-center distance below the configured threshold
- motion evidence by default: approaching relation or non-trivial recent speed

Evidence includes:

- bottom-center distance in pixels
- configured maximum distance
- human speed in pixels/second
- forklift speed in pixels/second
- approaching flag
- motion-present flag
- calibration availability flag

The default forklift class list is forklift, pallet_truck, and lift_truck. If the current detector model cannot recognize any of these classes, this behavior produces no event; configuring a class name does not create detections.

Without ground-plane calibration, the threshold is an image-space approximation. It is not a universal safe distance in metres.

## 11. Configuration

The thresholds and class vocabularies live in config.yaml. The ordered registry loads:

~~~python
from kavach.behaviours import build_default_registry

registry = build_default_registry()
events = registry.detect(context)
~~~

The requested evaluation order is preserved:

~~~text
ZONE_VIOLATION
POSSIBLE_DRAGGING
POSSIBLE_DROP
PALLET_OVERHANG
UNSTABLE_STACK
UNSAFE_HUMAN_FORKLIFT_PROXIMITY
~~~

A detector can be disabled by setting enabled: false in its YAML section. Thresholds are explicit so later experiments can change one assumption without rewriting detector logic.

## 12. False positives and false negatives

A false positive is an event emitted when the real behavior did not happen. Examples:

- a shadow or camera shake appears to move a carton
- two people pass near a forklift without an unsafe interaction
- a wide box visually overlaps a pallet but is not resting on it

A false negative is a real behavior that the rules fail to emit. Examples:

- a carton is dragged but its track disappears briefly
- a drop is hidden by occlusion
- a forklift class is not detected by the current model
- perspective changes the apparent distance or support ratio

Thresholds trade these errors against each other. Lower thresholds usually catch more cases but can increase false positives. Higher thresholds are more selective but may miss subtle cases.

## 13. Why rule-based temporal reasoning is useful

A rule-based first layer is valuable because it is:

- explainable: every event exposes its measurements
- deterministic: the same structured input produces the same output
- testable: synthetic trajectories can exercise edge cases
- tunable: thresholds and class lists are explicit
- modular: better detectors can later replace one rule without rewriting video ingestion

It is not a complete understanding of warehouse activity. It is a transparent foundation for later validation, labeling, and more advanced models.

## 14. Monocular-video limitations

A single ordinary camera provides a 2-D image projection. It does not directly reveal:

- true ground distance everywhere in the image
- object depth
- object height
- mass or load weight
- physical contact force
- whether a partially hidden object is actually carried
- whether an apparent downward motion is a real drop or camera/box motion

Pixel distances and pixel speeds are retained explicitly. No event in this module calls them metres or metres/second. Ground-plane calibration can improve distances for points on an approximately planar floor, but it does not recover general 3-D height.

## 15. Tests

Run the full suite:

~~~text
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v
.\\.venv\\Scripts\\python.exe -m compileall -q kavach tests
git diff --check
~~~

Latest validation: 49 tests passed in 0.130 seconds. Python bytecode
compilation completed successfully, and git diff --check reported no
whitespace errors.

The synthetic Module 7 tests cover each detector, structured event evidence, registry ordering, temporal trajectories, graph relation transitions, and the one-event-per-incident cooldown behavior.

Module 7 ends with six conservative, rule-based temporal behavior detectors.
Modules 8–10 build separate risk, storage, and grounded assistant layers on
these events.

## Integration extension: general activity and anomaly signals

The integration layer also exposes `ZONE_TRANSITION` and `OBJECT_ACTIVITY`
signals so a video is not reduced to restricted-zone incidents when its
detector does not contain warehouse-specific classes. It also exposes
`MOTION_ANOMALY`, an explainable image-space novelty signal based on a track's
recent speed and direction. These additions do not claim action recognition,
damage detection, or a learned industrial anomaly model. See
`docs/ACTIVITY_AND_ANOMALY.md` for interpretation and limitations.
