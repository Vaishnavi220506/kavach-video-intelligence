# KAVACH Activity and Anomaly Signals

KAVACH is designed to accept different videos. It must not assume that every
video contains a forklift, carton, pallet, or a particular camera layout.
The selected perception model still determines which object classes are
available. The temporal layer therefore exposes general signals in addition
to warehouse-specific safety rules.

## Signals now available

| Signal | What it means | Evidence used |
| --- | --- | --- |
| `ZONE_TRANSITION` | A tracked object crossed into or out of a configured polygon | bottom-center point, polygon membership, boundary distance, track ID |
| `OBJECT_ACTIVITY` | A tracked object was observed moving or becoming stationary | timestamp-aware speed, direction, displacement, track age |
| `MOTION_ANOMALY` | A sufficiently established track showed an unusual speed spike or large direction change compared with its recent history | recent median speed, speed ratio, direction change, image-space acceleration |

The original six safety rules remain available: zone violation, possible
dragging, possible drop, pallet overhang, unstable stack, and unsafe
human–forklift proximity. Their class filters are deliberate. If the loaded
model does not expose `forklift` or `carton`, those rules do not invent those
objects.

## What “anomaly” means here

`MOTION_ANOMALY` is an explainable novelty heuristic. It compares the current
tracked motion with the same track's recent timestamped motion. It can flag a
sudden image-space speed change or direction reversal for review.

YOLO is used here for the visual object observation, not as a magic anomaly
classifier: YOLO finds the object box, ByteTrack preserves the short-term
track identity, and the temporal detector evaluates that track's motion. The
processed evidence video highlights the matching YOLO/ByteTrack box in red and
labels the track when a motion anomaly is active.

It is not a learned industrial anomaly model. It cannot prove a collision,
damage, unsafe intent, or a physical-world measurement. Camera shake, missed
detections, ID switches, zoom, perspective, and occlusion can all produce
false positives or hide real events. The UI labels these records as anomaly
signals so they are not confused with verified incidents.

## Why this is general rather than video-specific

- Zones are configured polygons; they are not tied to one filename or one
  frame.
- Motion uses every visible tracked class and source timestamps.
- Anomaly baselines are calculated per track, not from hardcoded coordinates.
- Warehouse-specific rules activate only when the model actually detects the
  required semantic classes.
- The website reports the model vocabulary and the limitation of its class
  coverage after each analysis.

## Debouncing

Signals are emitted on meaningful transitions or novelty points. The shared
behaviour base class prevents a condition that remains true for several
seconds from creating one database row per frame. Cooldown and incident
deduplication remain configurable in `kavach/behaviours/config.yaml` and the
Module 8 risk/incident layer.

## Interpreting the output

Treat `OBJECT_ACTIVITY` as context, `MOTION_ANOMALY` as a review lead, and the
warehouse safety behaviours as explainable rule outputs. All scores are policy
risk scores, not probabilities. Review the linked replay before making an
operational decision.
