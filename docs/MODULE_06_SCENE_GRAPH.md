# KAVACH Module 6: Dynamic Spatio-Temporal Scene Graph

## Scope

Module 6 converts the outputs of Modules 2–5 into an inspectable graph of current objects, zones, and explainable relations.

The inputs are:

- frame-local detections
- persistent ByteTrack track IDs
- Module 4 timestamped ObjectMemory
- Module 5 geometry primitives
- Module 5 configurable polygon zones

The output is a current graph snapshot plus a bounded history of previous snapshots.

This is not a learned relation classifier. Relations are produced by explicit geometry and temporal rules, and every edge stores the numeric measurements that caused it.

Module 6 does not implement behavior classification, a complete scene graph ontology, risk reasoning, incident extraction, a database, an LLM, or a dashboard redesign. It also intentionally does not generate carried_by: a reliable carried-object relation needs stronger contact, persistence, visibility, and camera-geometry evidence than this module currently has.

## 1. What is a graph?

A graph is a collection of:

- nodes: the things being represented
- edges: connections between those things

For KAVACH, example nodes are:

~~~text
person_3
carton_7
pallet_2
forklift_1
staging_zone
~~~

An object node is identified by normalized class name plus track ID, for example carton_7. It stores a reference to the current TrackedObject state. That state contains class, confidence, bounding box, center, timestamp, and frame number.

A zone node is identified by the normalized configured zone name, for example staging_zone. It does not have an object track state; it references the configured Zone.

An edge is a directed fact:

~~~text
carton_7 --supported_by--> pallet_2
carton_7 --inside_zone--> staging_zone
~~~

Even a naturally symmetric fact such as near is represented as two directed edges, one in each direction. That keeps directed graph queries simple.

## 2. Scene graph versus dynamic scene graph

A scene graph describes what is present and how the entities relate at one moment.

A dynamic scene graph updates that description over time:

~~~text
time 0.0 s: person_3 --near--> forklift_1
time 1.0 s: person_3 --approaching--> forklift_1
time 2.0 s: person_3 --separating_from--> forklift_1
~~~

The current graph is replaced on each update. SceneGraph also retains a bounded snapshot history for debugging and later analysis. The source timestamp is preserved; wall-clock processing time is not substituted for video time.

For temporal relations, the caller should update ObjectMemory first and pass the current tracked objects to SceneGraph:

~~~python
memory.update(tracked_objects, packet.timestamp, packet.frame_number)
graph.update(tracked_objects, timestamp=packet.timestamp)
~~~

The graph reads the authoritative Module 4 history. It does not append a duplicate memory state.

## 3. Repository architecture

~~~text
kavach/intelligence/
├── relationships.py   # Relation names, thresholds, evidence, rule derivation
├── scene_graph.py     # Current nodes, edges, snapshots, queries, debug log
├── visualization.py   # Existing trajectories plus scene-graph debug overlay
├── object_memory.py   # Timestamped states used for temporal relations
├── geometry.py        # Box, point, overlap, and distance primitives
└── zones.py           # Configurable polygon zones
~~~

Main types:

- RelationThresholds: explicit thresholds for near, overlap, support, motion, and temporal change.
- RelationEdge: one directed relation with timestamp and numeric evidence.
- SceneNode: an object node with current state or a configured zone node.
- SceneSnapshot: immutable nodes and edges at one timestamp.
- SceneGraph: current graph, bounded snapshot history, queries, serialization, and logging.
- RelationshipError and SceneGraphError: input and update validation errors.

The Module 6 public API is exported from kavach.intelligence.

## 4. Relation vocabulary

The current rule-based vocabulary is:

- near
- inside_zone
- above
- below
- left_of
- right_of
- overlapping
- supported_by
- moving_with
- approaching
- separating_from

The inverse directional edges are emitted together. For example, if carton_7 is above pallet_2, the graph contains both:

~~~text
carton_7 --above--> pallet_2
pallet_2 --below--> carton_7
~~~

The same relation edge always includes a non-empty numeric evidence mapping. Examples include distance_px, iou, horizontal_overlap_ratio, vertical_gap_px, interval_seconds, and velocity_delta_px_per_second.

## 5. How numerical geometry becomes relations

### near

Two current object centers are measured with Euclidean distance. If:

~~~text
distance_px <= near_distance_px
~~~

the graph emits near in both directions. Evidence records the measured distance, configured threshold, and remaining margin.

This is image distance. It is not metres unless a separate valid ground-plane calibration is applied by a later consumer.

### inside_zone

The bottom-center of each object bounding box is tested against each configured polygon using the Module 5 OpenCV point-in-polygon primitive. If the point is inside or on the boundary:

~~~text
object --inside_zone--> zone
~~~

Evidence includes bottom-center coordinates, boundary distance, and inside=1.0.

The bottom-center is used because it is a rough image-plane contact point. It is still only an approximation.

### above, below, left_of, right_of

The current bounding-box centers are compared in image coordinates:

- smaller y means above
- larger y means below
- smaller x means left
- larger x means right

Evidence records the x and y center differences in pixels. Equal center coordinates do not produce a directional relation.

### overlapping

Module 5 IoU is calculated for each pair. If IoU is positive and meets the configured overlap threshold, overlapping is emitted in both directions.

Evidence records:

- IoU
- intersection width in pixels
- intersection height in pixels

Overlap is a 2-D image cue, not proof of physical contact.

### supported_by

This is intentionally conservative and class-aware. The proposed support object must have a configured provider class such as pallet, platform, shelf, floor, trolley, or pallet_truck.

For object A over provider B, the rule requires:

- A's center is above B's center
- the horizontal support ratio is at least the configured minimum
- the vertical gap from A's bottom to B's top is no more than the configured maximum

The resulting edge is:

~~~text
A --supported_by--> B
~~~

Evidence includes horizontal overlap in pixels, horizontal overlap ratio, and vertical gap in pixels.

This is a candidate geometric support relation. It does not prove physical resting, load contact, weight transfer, or stable stacking. Those require later temporal and physical reasoning.

### moving_with

If Module 4 provides current velocity estimates for both objects, the rule requires:

- the objects are within the configured maximum distance
- both speeds exceed the minimum motion speed
- the velocity-vector difference is below the configured tolerance

Evidence records distance, both pixel speeds, and velocity-vector difference in pixels/second.

No physical velocity is claimed without calibration.

### approaching and separating_from

For a pair with at least two historical states each, the graph compares the prior center distance with the current center distance:

~~~text
distance_change_px = current_distance_px - previous_distance_px
~~~

- a sufficiently negative change creates approaching
- a sufficiently positive change creates separating_from
- a smaller change creates neither relation

Evidence records previous distance, current distance, distance change, interval, and closing rate in pixels/second.

These are relative-motion relations. They are not action recognition and do not by themselves imply a collision, unsafe event, or intent.

## 6. Relation thresholds

Defaults are explicit in RelationThresholds:

~~~text
near distance:                         100 px
overlap IoU threshold:                 0.01
support horizontal overlap minimum:   0.50
support maximum vertical gap:          20 px
moving-with maximum distance:          120 px
moving-with velocity difference:       25 px/s
moving-with minimum speed:              2 px/s
minimum temporal distance change:        2 px
~~~

These are starting image-space thresholds, not universal warehouse truth. They should be tuned against camera resolution, perspective, detector quality, and the operational scene.

Support provider classes are configurable. carried_by is not in the current vocabulary and is not emitted.

## 7. Query API

SceneGraph provides:

~~~python
graph.get_relations(7)
graph.has_relation("carton_7", "supported_by", "pallet_2")
graph.get_objects_in_zone("STAGING_ZONE")
graph.get_nearby(7)
~~~

Query return values:

- get_relations returns current RelationEdge objects touching the requested object node.
- has_relation returns a boolean for one exact directed edge.
- get_objects_in_zone returns object node IDs connected to the zone.
- get_nearby returns object node IDs connected by near edges.

Track IDs may be supplied as integers or readable object node IDs. Zone names may be supplied in configured form such as STAGING_ZONE or normalized form such as staging_zone. Unknown query targets return an empty result or False.

For inspection and serialization:

~~~python
print(graph.debug_log())
payload = graph.to_dict()
snapshot = graph.update(tracked_objects, timestamp=timestamp)
~~~

debug_log prints each edge and all of its numeric evidence. to_dict returns JSON-friendly nodes and relations.

## 8. Complete example: raw detections to graph

### Step 1: raw detector outputs

YOLO operates on one frame and returns independent detections:

~~~python
raw_detections = [
    {
        "class_name": "carton",
        "class_id": 1,
        "confidence": 0.91,
        "bbox": (35.0, 20.0, 75.0, 40.0),
    },
    {
        "class_name": "pallet",
        "class_id": 2,
        "confidence": 0.88,
        "bbox": (35.0, 40.0, 75.0, 70.0),
    },
]
~~~

At this point there are boxes, but no persistent identity or semantic relation.

### Step 2: tracking adds identity

ByteTrack associates detections across frames:

~~~python
tracked_objects = [
    {
        "track_id": 7,
        "class_name": "carton",
        "confidence": 0.91,
        "bbox": (35.0, 20.0, 75.0, 40.0),
        "center": (55.0, 30.0),
        "timestamp": 12.0,
        "frame_number": 720,
    },
    {
        "track_id": 2,
        "class_name": "pallet",
        "confidence": 0.88,
        "bbox": (35.0, 40.0, 75.0, 70.0),
        "center": (55.0, 55.0),
        "timestamp": 12.0,
        "frame_number": 720,
    },
]
~~~

The actual implementation passes TrackedObject instances, not these illustrative dictionaries.

### Step 3: geometry computes evidence

For carton_7 and pallet_2:

~~~text
carton center:                 (55, 30)
pallet center:                 (55, 55)
center distance:               25 px
horizontal overlap:            40 px
carton width:                  40 px
support ratio:                 1.00
vertical gap:                  40 - 40 = 0 px
carton bottom-center:          (55, 40)
~~~

If the configured staging polygon contains (55, 40), the carton is also inside that zone.

### Step 4: graph stores meaning plus evidence

The resulting graph can contain:

~~~json
{
  "nodes": [
    {
      "node_id": "carton_7",
      "node_type": "object",
      "state": {
        "track_id": 7,
        "class_name": "carton",
        "center": [55.0, 30.0],
        "bbox": [35.0, 20.0, 75.0, 40.0],
        "timestamp": 12.0,
        "frame_number": 720
      }
    },
    {
      "node_id": "pallet_2",
      "node_type": "object"
    },
    {
      "node_id": "staging_zone",
      "node_type": "zone"
    }
  ],
  "relations": [
    {
      "subject": "carton_7",
      "relation": "supported_by",
      "object": "pallet_2",
      "timestamp": 12.0,
      "evidence": {
        "horizontal_overlap_px": 40.0,
        "horizontal_overlap_ratio": 1.0,
        "vertical_gap_px": 0.0
      }
    },
    {
      "subject": "carton_7",
      "relation": "inside_zone",
      "object": "staging_zone",
      "timestamp": 12.0,
      "evidence": {
        "inside": 1.0,
        "bottom_center_x": 55.0,
        "bottom_center_y": 40.0,
        "boundary_distance_px": 0.0
      }
    }
  ]
}
~~~

The real to_dict output includes the full current pallet state as well. The graph does not hide the measurements behind a classification label.

## 9. Debugging visualization

draw_scene_graph in intelligence/visualization.py:

- draws configured zones
- draws current object boxes and centers
- labels object nodes such as carton_7
- draws arrows between object nodes
- colors arrows by relation type
- labels relation names such as approaching or supported_by

inside_zone edges are represented by the zone polygon and object label rather than a line to an invisible point. debug_log provides the exact numeric evidence when the drawing is not sufficient.

## 10. Tests

Run the full suite:

~~~text
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v
.\\.venv\\Scripts\\python.exe -m compileall -q kavach tests
git diff --check
~~~

Latest validation: 41 tests passed in 0.112 seconds. Python bytecode
compilation completed successfully, and git diff --check reported no
whitespace errors.

The Module 6 test file covers:

- readable object and zone node IDs
- current tracked-state references
- near, directional, overlap, support, and zone relations
- numeric evidence on every emitted relation
- approaching and separating from ObjectMemory
- moving_with from similar timestamped velocity
- bounded snapshot updates
- query methods, serialization, and debug logging
- annotated-copy scene-graph visualization

## 11. Limitations

- Relations are thresholded geometric cues, not learned world understanding.
- Detection errors and ByteTrack ID switches propagate into the graph.
- The graph only knows currently supplied active tracks; a missing track has no new state.
- near and distance are in pixels unless a separate ground-plane calibration is applied.
- approaching and separating compare recent observed states and can be noisy with detector jitter.
- moving_with depends on image-space velocity and configurable tolerance.
- supported_by is a candidate relation, not proof of physical support.
- A single monocular view does not automatically provide depth or height.
- Zone membership depends on a camera-specific polygon and the chosen bottom-center approximation.
- Overlapping zones can create multiple inside_zone edges.
- No carried_by relation is emitted.
- No action recognition, behavior model, collision/risk engine, event database, LLM, or dashboard redesign is included.

Module 6 ends at a dynamic, explainable graph of current tracked objects and configured zones. The next module should consume these relations only after validating their evidence and limitations.
