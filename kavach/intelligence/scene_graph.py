"""Dynamic, inspectable scene graph for KAVACH Module 6."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass
import math

from ..perception.tracker import TrackedObject
from .object_memory import ObjectMemory, ObjectState
from .relationships import (
    INSIDE_ZONE,
    NEAR,
    RelationEdge,
    RelationThresholds,
    RelationshipError,
    build_relations,
    object_node_id,
    zone_node_id,
)
from .zones import ZoneManager


class SceneGraphError(RuntimeError):
    """Raised when a dynamic scene graph update or query is invalid."""


@dataclass(frozen=True)
class SceneNode:
    """A current object or configured zone in the scene graph."""

    node_id: str
    node_type: str
    state: TrackedObject | ObjectState | None = None
    zone_name: str | None = None

    def __post_init__(self) -> None:
        if not str(self.node_id).strip():
            raise SceneGraphError("node_id cannot be empty")
        if self.node_type not in {"object", "zone"}:
            raise SceneGraphError("node_type must be 'object' or 'zone'")
        if self.node_type == "object" and self.state is None:
            raise SceneGraphError("object nodes must reference a current state")
        if self.node_type == "zone" and self.zone_name is None:
            raise SceneGraphError("zone nodes must have a zone_name")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly node representation."""

        result: dict[str, object] = {
            "node_id": self.node_id,
            "node_type": self.node_type,
        }
        if self.zone_name is not None:
            result["zone_name"] = self.zone_name
        if self.state is not None:
            result["state"] = asdict(self.state)
        return result


@dataclass(frozen=True)
class SceneSnapshot:
    """One immutable view of graph nodes and relations at a source timestamp."""

    timestamp: float
    nodes: tuple[SceneNode, ...]
    relations: tuple[RelationEdge, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly graph snapshot."""

        return {
            "timestamp": self.timestamp,
            "nodes": [node.to_dict() for node in self.nodes],
            "relations": [relation.to_dict() for relation in self.relations],
        }


class SceneGraph:
    """Maintain the current explainable graph and bounded snapshot history.

    The graph itself does not update ObjectMemory. Callers should update the
    Module 4 memory first, then pass the same current tracked objects to this
    graph. That keeps one authoritative temporal history and avoids duplicate
    observations.
    """

    def __init__(
        self,
        *,
        memory: ObjectMemory | None = None,
        zones: ZoneManager | None = None,
        thresholds: RelationThresholds | None = None,
        max_snapshots: int = 120,
    ) -> None:
        if max_snapshots <= 0:
            raise SceneGraphError("max_snapshots must be positive")
        self.memory = memory
        self.zones = zones
        self.thresholds = thresholds or RelationThresholds()
        self.max_snapshots = int(max_snapshots)
        self._nodes: dict[str, SceneNode] = {}
        self._relations: tuple[RelationEdge, ...] = ()
        self._snapshots: deque[SceneSnapshot] = deque(maxlen=self.max_snapshots)
        self._last_timestamp: float | None = None

    @property
    def nodes(self) -> tuple[SceneNode, ...]:
        """Return current nodes in stable insertion order."""

        return tuple(self._nodes.values())

    @property
    def relations(self) -> tuple[RelationEdge, ...]:
        """Return the current directed, evidence-bearing edges."""

        return self._relations

    @property
    def snapshots(self) -> tuple[SceneSnapshot, ...]:
        """Return the bounded history of graph snapshots."""

        return tuple(self._snapshots)

    @property
    def timestamp(self) -> float | None:
        """Return the timestamp of the current graph, if it has been updated."""

        return self._last_timestamp

    def update(
        self,
        tracked_objects: Iterable[TrackedObject],
        *,
        timestamp: float | None = None,
        memory: ObjectMemory | None = None,
    ) -> SceneSnapshot:
        """Replace the current graph with relations for one source timestamp."""

        objects = tuple(tracked_objects)
        if timestamp is None:
            timestamp = 0.0 if not objects else float(objects[0].timestamp)
        timestamp_value = float(timestamp)
        if not math.isfinite(timestamp_value) or timestamp_value < 0.0:
            raise SceneGraphError("timestamp must be finite and non-negative")
        if (
            self._last_timestamp is not None
            and timestamp_value < self._last_timestamp
        ):
            raise SceneGraphError(
                f"timestamp {timestamp_value} is earlier than the previous graph "
                f"timestamp {self._last_timestamp}"
            )

        active_memory = self.memory if memory is None else memory
        try:
            relations = build_relations(
                objects,
                timestamp=timestamp_value,
                zones=self.zones,
                memory=active_memory,
                thresholds=self.thresholds,
            )
        except (RelationshipError, ValueError) as exc:
            raise SceneGraphError(str(exc)) from exc

        nodes: dict[str, SceneNode] = {}
        for tracked in objects:
            node_id = object_node_id(tracked.track_id, tracked.class_name)
            nodes[node_id] = SceneNode(
                node_id=node_id,
                node_type="object",
                state=tracked,
            )
        if self.zones is not None:
            for zone in self.zones.zones:
                node_id = zone_node_id(zone.name)
                nodes[node_id] = SceneNode(
                    node_id=node_id,
                    node_type="zone",
                    zone_name=zone.name,
                )

        snapshot = SceneSnapshot(
            timestamp=timestamp_value,
            nodes=tuple(nodes.values()),
            relations=tuple(relations),
        )
        self._nodes = nodes
        self._relations = snapshot.relations
        self._snapshots.append(snapshot)
        self._last_timestamp = timestamp_value
        return snapshot

    def _resolve_node_id(self, value: int | str) -> str | None:
        candidate = str(value)
        if candidate in self._nodes:
            return candidate
        if isinstance(value, int) or candidate.isdigit():
            track_id = int(value)
            for node in self._nodes.values():
                if node.state is not None and int(node.state.track_id) == track_id:
                    return node.node_id
        normalized_zone = zone_node_id(candidate)
        if normalized_zone in self._nodes and self._nodes[normalized_zone].node_type == "zone":
            return normalized_zone
        return None

    def get_relations(self, track_id: int | str) -> tuple[RelationEdge, ...]:
        """Return all current edges touching a track's object node."""

        node_id = self._resolve_node_id(track_id)
        if node_id is None:
            return ()
        return tuple(
            relation
            for relation in self._relations
            if relation.subject == node_id or relation.object == node_id
        )

    def has_relation(
        self,
        subject: int | str,
        relation: str,
        object_name: int | str,
    ) -> bool:
        """Return whether one exact directed relation is currently present."""

        subject_id = self._resolve_node_id(subject)
        object_id = self._resolve_node_id(object_name)
        if subject_id is None or object_id is None:
            return False
        return any(
            edge.subject == subject_id
            and edge.relation == str(relation)
            and edge.object == object_id
            for edge in self._relations
        )

    def get_objects_in_zone(self, zone: str) -> tuple[str, ...]:
        """Return current object node IDs connected by inside_zone edges."""

        zone_id = self._resolve_node_id(zone)
        if zone_id is None:
            return ()
        return tuple(
            edge.subject
            for edge in self._relations
            if edge.relation == INSIDE_ZONE and edge.object == zone_id
        )

    def get_nearby(self, track_id: int | str) -> tuple[str, ...]:
        """Return current object node IDs connected by near edges."""

        node_id = self._resolve_node_id(track_id)
        if node_id is None:
            return ()
        nearby: list[str] = []
        for edge in self._relations:
            if edge.relation != NEAR:
                continue
            other = edge.object if edge.subject == node_id else edge.subject
            if edge.subject == node_id or edge.object == node_id:
                if other not in nearby and other in self._nodes:
                    nearby.append(other)
        return tuple(nearby)

    def to_dict(self) -> dict[str, object]:
        """Return the current graph for logs, notebooks, or JSON serialization."""

        snapshot = SceneSnapshot(
            timestamp=0.0 if self._last_timestamp is None else self._last_timestamp,
            nodes=self.nodes,
            relations=self.relations,
        )
        return snapshot.to_dict()

    def debug_log(self) -> str:
        """Return a concise human-readable graph/evidence log."""

        current_timestamp = "-" if self._last_timestamp is None else f"{self._last_timestamp:.3f}s"
        lines = [
            f"SceneGraph timestamp={current_timestamp} nodes={len(self.nodes)} "
            f"relations={len(self.relations)}"
        ]
        for edge in self.relations:
            evidence = ", ".join(
                f"{key}={value:.3f}" for key, value in edge.evidence.items()
            )
            lines.append(
                f"{edge.subject} --{edge.relation}--> {edge.object} "
                f"[{evidence}]"
            )
        return "\n".join(lines)

    def clear(self) -> None:
        """Remove current nodes, edges, and bounded snapshot history."""

        self._nodes.clear()
        self._relations = ()
        self._snapshots.clear()
        self._last_timestamp = None

