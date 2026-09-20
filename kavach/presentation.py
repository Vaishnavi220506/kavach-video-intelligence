"""Shape stored incident rows for a client.

This is the single definition of the client-facing evidence contract. The
FastAPI service uses it to answer requests, and ``scripts/build_demo_dataset``
uses it to generate the static fixtures the published website reads, so the
two cannot drift into describing the same record differently.

Nothing here scores, detects or decides. It reads rows that Modules 7 and 8
already wrote and presents them: a display timestamp, the signal class, a
flattened evidence summary, verified artifact references, and the
entity-to-incident graph.
"""

from __future__ import annotations

from typing import Any

from .assistant.retrieval import format_timestamp
from .behaviours import (
    AISLE_OBSTRUCTION,
    COLLISION_RISK,
    IMPROPER_PLACEMENT,
    MOTION_ANOMALY,
    OBJECT_ACTIVITY,
    POSSIBLE_ROUGH_HANDLING,
    POSSIBLE_THROWING,
    ZONE_TRANSITION,
)
from .incidents import EvidenceError, EvidenceStore

#: Signals describing what an object did, without asserting a safety finding.
ACTIVITY_EVENT_TYPES = frozenset({ZONE_TRANSITION, OBJECT_ACTIVITY})

#: Explainable image-space novelty. Not a learned damage or defect classifier.
ANOMALY_EVENT_TYPES = frozenset({MOTION_ANOMALY})

#: Rule findings that a supervisor reviews as safety evidence.
SAFETY_EVENT_TYPES = frozenset(
    {
        "ZONE_VIOLATION",
        "POSSIBLE_DRAGGING",
        "POSSIBLE_DROP",
        "PALLET_OVERHANG",
        "UNSTABLE_STACK",
        "UNSAFE_HUMAN_FORKLIFT_PROXIMITY",
        POSSIBLE_THROWING,
        POSSIBLE_ROUGH_HANDLING,
        AISLE_OBSTRUCTION,
        IMPROPER_PLACEMENT,
        COLLISION_RISK,
    }
)

SUPPORTED_BEHAVIOURS = tuple(
    sorted(ACTIVITY_EVENT_TYPES | ANOMALY_EVENT_TYPES | SAFETY_EVENT_TYPES)
)

MODEL_SCOPE_NOTE = (
    "The default model detects only classes present in its trained "
    "vocabulary. Warehouse labels such as forklift or carton are not "
    "claimed unless the selected weights actually expose them."
)

ANOMALY_SCOPE_NOTE = (
    "MOTION_ANOMALY is an explainable image-space novelty signal based "
    "on tracked speed and direction changes; it is not a learned "
    "damage or defect classifier."
)


def signal_kind(behaviour: str) -> str:
    """Classify a behaviour as an activity, anomaly or safety signal."""

    if behaviour in ACTIVITY_EVENT_TYPES:
        return "activity"
    if behaviour in ANOMALY_EVENT_TYPES:
        return "anomaly"
    return "safety"


def event_payload(
    event: dict[str, Any],
    *,
    verify_artifacts: bool = True,
) -> dict[str, Any]:
    """Present one stored event row for a client.

    ``verify_artifacts`` re-hashes each referenced snapshot against the record.
    The API leaves it on so a tampered or missing file is visible. Static
    fixture generation turns it off, because the published site ships no
    snapshot files to verify and an unconditional ``False`` would read as a
    failed integrity check rather than an absent one.
    """

    payload = dict(event)
    timestamp = float(payload.get("timestamp", 0.0))
    payload["timestamp"] = timestamp
    payload["timestamp_display"] = format_timestamp(timestamp)

    event_id = str(payload.get("event_id", ""))
    payload["replay_url"] = f"/api/events/{event_id}/replay"
    payload["clip_url"] = f"/api/events/{event_id}/clip" if event_id else None
    payload["signal_kind"] = signal_kind(str(payload.get("behaviour", "")))

    raw_evidence = payload.get("evidence")
    evidence_summary: dict[str, Any] = {}
    if isinstance(raw_evidence, dict):
        for candidate_key in ("latest_event", "initial_event"):
            candidate = raw_evidence.get(candidate_key)
            if isinstance(candidate, dict) and isinstance(candidate.get("evidence"), dict):
                evidence_summary = dict(candidate["evidence"])
                break
        if not evidence_summary:
            evidence_summary = dict(raw_evidence)
    payload["evidence_summary"] = evidence_summary

    artifacts: list[dict[str, Any]] = []
    if isinstance(raw_evidence, dict):
        raw_artifacts = raw_evidence.get("evidence_artifacts")
        if isinstance(raw_artifacts, list):
            for index, artifact in enumerate(raw_artifacts):
                if not isinstance(artifact, dict):
                    continue
                item = dict(artifact)
                item["url"] = f"/api/events/{event_id}/snapshot/{index}"
                if verify_artifacts:
                    # A record missing its path or digest raises rather than
                    # returning False. That must not take down the response
                    # for the whole video: an artifact that cannot be checked
                    # is reported as unverified, which is the honest reading,
                    # and the rest of the record still reaches the reviewer.
                    try:
                        item["verified"] = EvidenceStore.verify(item)
                    except EvidenceError:
                        item["verified"] = False
                artifacts.append(item)
    payload["evidence_artifacts"] = artifacts

    payload["prevention"] = (
        raw_evidence.get("prevention", []) if isinstance(raw_evidence, dict) else []
    )
    payload["root_cause_category"] = (
        raw_evidence.get("root_cause_category") if isinstance(raw_evidence, dict) else None
    )
    return payload


def evidence_graph(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a transparent entity to incident graph from stored evidence.

    This is deliberately not presented as a persisted frame-level scene graph;
    Module 6 graph snapshots are in-memory. It only visualizes relationships
    explicitly retained in the incident records.
    """

    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    for event in events:
        event_id = str(event.get("event_id", "event"))
        risk = event.get("risk") if isinstance(event.get("risk"), dict) else {}
        event_node = f"event:{event_id}"
        nodes[event_node] = {
            "id": event_node,
            "label": str(event.get("behaviour", "Event")).replace("_", " "),
            "kind": "incident",
            "risk": str(risk.get("category", "UNKNOWN")),
            "timestamp": float(event.get("timestamp", 0.0)),
        }
        for raw_entity in event.get("entities", []):
            entity = str(raw_entity)
            entity_node = f"entity:{entity}"
            nodes.setdefault(
                entity_node,
                {
                    "id": entity_node,
                    "label": entity.replace("_", " "),
                    "kind": "entity",
                },
            )
            edges.append(
                {
                    "source": entity_node,
                    "target": event_node,
                    "label": "involved in",
                }
            )
    return {"nodes": list(nodes.values()), "edges": edges}
