"""WorldGraph container, query API, provenance wiring, and JSON snapshots."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Self

from .model import WorldEdge, WorldId, WorldNode
from .provenance import SemanticProvenance

SCHEMA_VERSION = 1


class WorldGraphError(Exception):
    """Base class for WorldGraph operation errors."""


class UnknownNodeError(WorldGraphError):
    """Raised when an edge endpoint references a node absent from the graph."""

    def __init__(self, node_id: WorldId | int) -> None:
        self.node_id = WorldId.coerce(node_id)
        super().__init__(f"unknown node {self.node_id!r}")


class WorldGraphSerializationError(WorldGraphError):
    """Raised when persisted graph JSON cannot be parsed or emitted."""


@dataclass(frozen=True)
class PrivacyRollup:
    """Result of recomputing privacy-limiting decisions for observes edges."""

    mode: str
    suppressed_nodes: tuple[WorldId, ...] = ()
    denied_pairs: tuple[tuple[WorldId, WorldId], ...] = ()
    allowed_pairs: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", str(self.mode))
        object.__setattr__(self, "suppressed_nodes", tuple(WorldId.coerce(node) for node in self.suppressed_nodes))
        object.__setattr__(
            self,
            "denied_pairs",
            tuple((WorldId.coerce(sensor), WorldId.coerce(target)) for sensor, target in self.denied_pairs),
        )
        object.__setattr__(self, "allowed_pairs", int(self.allowed_pairs))

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "suppressed_nodes": [int(node) for node in self.suppressed_nodes],
            "denied_pairs": [[int(sensor), int(target)] for sensor, target in self.denied_pairs],
            "allowed_pairs": self.allowed_pairs,
        }


@dataclass(frozen=True)
class WorldGraphSnapshot:
    """Serializable snapshot of a :class:`WorldGraph`."""

    schema_version: int
    registration: Mapping[str, Any]
    next_id: int
    nodes: tuple[WorldNode, ...] = ()
    edges: tuple[tuple[WorldId, WorldId, WorldEdge], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "registration", _jsonable(self.registration))
        object.__setattr__(self, "next_id", int(self.next_id))
        object.__setattr__(self, "nodes", tuple(WorldNode.coerce(node) for node in self.nodes))
        object.__setattr__(self, "edges", tuple(_coerce_edge_record(edge) for edge in self.edges))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "registration": _jsonable(self.registration),
            "next_id": self.next_id,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [[int(source), int(target), edge.to_dict()] for source, target, edge in self.edges],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        return cls(
            schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
            registration=_mapping_or_empty(payload.get("registration")),
            next_id=int(payload.get("next_id", 1)),
            nodes=tuple(WorldNode.coerce(node) for node in payload.get("nodes", ())),
            edges=tuple(_coerce_edge_record(edge) for edge in payload.get("edges", ())),
        )


@dataclass(frozen=True)
class _StoredEdge:
    source: WorldId
    target: WorldId
    edge: WorldEdge

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", WorldId.coerce(self.source))
        object.__setattr__(self, "target", WorldId.coerce(self.target))
        object.__setattr__(self, "edge", WorldEdge.coerce(self.edge))


class WorldGraph:
    """Typed environmental digital twin with stable WorldId-based queries."""

    def __init__(self, registration: Mapping[str, Any] | None = None) -> None:
        self._nodes: dict[WorldId, WorldNode] = {}
        self._edges: list[_StoredEdge] = []
        self._registration = _jsonable(registration or {})
        self._next_id = 1
        self._schema_version = SCHEMA_VERSION

    @property
    def registration(self) -> Mapping[str, Any]:
        """Installation geo-registration payload carried through snapshots."""

        return self._registration

    @property
    def schema_version(self) -> int:
        return self._schema_version

    @property
    def next_id(self) -> int:
        return self._next_id

    def node_count(self) -> int:
        """Number of live nodes."""

        return len(self._nodes)

    def edge_count(self) -> int:
        """Number of live directed edges."""

        return len(self._edges)

    def upsert_node(self, node: WorldNode | Mapping[str, Any]) -> WorldId:
        """Insert or replace a node, allocating an id for the unassigned sentinel."""

        current = WorldNode.coerce(node)
        node_id = current.id
        if node_id.is_unassigned():
            node_id = WorldId(self._next_id)
            current = current.with_id(node_id)
            self._next_id += 1
        else:
            self._next_id = max(self._next_id, int(node_id) + 1)

        self._nodes[node_id] = current
        return node_id

    def add_edge(self, source: WorldId | int, target: WorldId | int, edge: WorldEdge | Mapping[str, Any]) -> None:
        """Add a typed edge between two known nodes."""

        source_id = WorldId.coerce(source)
        target_id = WorldId.coerce(target)
        if source_id not in self._nodes:
            raise UnknownNodeError(source_id)
        if target_id not in self._nodes:
            raise UnknownNodeError(target_id)
        self._edges.append(_StoredEdge(source_id, target_id, WorldEdge.coerce(edge)))

    def node(self, node_id: WorldId | int) -> WorldNode | None:
        """Return a node by stable id, if it is live."""

        return self._nodes.get(WorldId.coerce(node_id))

    def remove_node(self, node_id: WorldId | int) -> WorldNode | None:
        """Remove a node and its incident edges."""

        stable_id = WorldId.coerce(node_id)
        removed = self._nodes.pop(stable_id, None)
        if removed is None:
            return None
        self._edges = [edge for edge in self._edges if edge.source != stable_id and edge.target != stable_id]
        return removed

    def nodes(self) -> tuple[WorldNode, ...]:
        """Return live nodes in insertion order."""

        return tuple(self._nodes.values())

    def neighbors(self, node_id: WorldId | int) -> list[tuple[WorldId, WorldEdge]]:
        """Outgoing neighbours of a node with their connecting edge."""

        stable_id = WorldId.coerce(node_id)
        if stable_id not in self._nodes:
            return []
        return [(edge.target, edge.edge) for edge in self._edges if edge.source == stable_id]

    def room_for_area(self, area_id: str) -> WorldId | None:
        """Resolve a HomeCore area id to its room node."""

        for node in self._nodes.values():
            if node.kind == "room" and node.area_id == area_id:
                return node.id
        return None

    def observed_by(self, sensor: WorldId | int) -> list[WorldId]:
        """Return nodes currently observed by a sensor or RF link."""

        return [target for target, edge in self.neighbors(sensor) if edge.rel == "observes"]

    def contents_of(self, container: WorldId | int) -> list[WorldId]:
        """Return nodes with incoming ``located_in`` edges to a room or zone."""

        container_id = WorldId.coerce(container)
        if container_id not in self._nodes:
            return []
        return [
            edge.source
            for edge in self._edges
            if edge.target == container_id and edge.edge.rel == "located_in"
        ]

    def add_semantic_state(
        self,
        statement: str,
        confidence: float,
        valid_from_unix_ms: int,
        provenance: SemanticProvenance | Mapping[str, Any],
        evidence_sources: Sequence[WorldId | int] = (),
    ) -> WorldId:
        """Insert a semantic state and connect existing evidence sources."""

        semantic_provenance = SemanticProvenance.coerce(provenance)
        node_id = self.upsert_node(
            WorldNode.semantic_state(
                statement=statement,
                confidence=confidence,
                valid_from_unix_ms=valid_from_unix_ms,
                provenance=semantic_provenance,
            )
        )
        for index, source in enumerate(evidence_sources):
            source_id = WorldId.coerce(source)
            if source_id not in self._nodes:
                continue
            evidence = semantic_provenance.evidence[index] if index < len(semantic_provenance.evidence) else ""
            self.add_edge(node_id, source_id, WorldEdge.derived_from(evidence))
        return node_id

    def add_contradiction(self, a: WorldId | int, b: WorldId | int, magnitude: float, flag: str) -> None:
        """Record a queryable contradiction between two still-live beliefs."""

        self.add_edge(a, b, WorldEdge.contradicts(magnitude=magnitude, flag=flag))

    def apply_privacy_mode(
        self,
        mode: str,
        action: str,
        policy: Callable[[str, str], bool],
    ) -> PrivacyRollup:
        """Append privacy-limiting decisions for current ``observes`` edges."""

        decisions: list[tuple[WorldId, WorldId, bool]] = []
        for stored in list(self._edges):
            if stored.edge.rel != "observes":
                continue
            sensor = self._nodes[stored.source]
            target = self._nodes[stored.target]
            allowed = bool(policy(sensor.kind, target.kind))
            decisions.append((stored.source, stored.target, allowed))

        denied_pairs: list[tuple[WorldId, WorldId]] = []
        suppressed_nodes: list[WorldId] = []
        allowed_pairs = 0
        for sensor, target, allowed in decisions:
            self.add_edge(sensor, target, WorldEdge.privacy_limited_by(mode=mode, action=action, allowed=allowed))
            if allowed:
                allowed_pairs += 1
                continue
            denied_pairs.append((sensor, target))
            if target not in suppressed_nodes:
                suppressed_nodes.append(target)

        return PrivacyRollup(
            mode=mode,
            suppressed_nodes=tuple(suppressed_nodes),
            denied_pairs=tuple(denied_pairs),
            allowed_pairs=allowed_pairs,
        )

    def snapshot(self) -> WorldGraphSnapshot:
        """Return a deterministic persistence snapshot."""

        return WorldGraphSnapshot(
            schema_version=self._schema_version,
            registration=self._registration,
            next_id=self._next_id,
            nodes=tuple(self._nodes.values()),
            edges=tuple((edge.source, edge.target, edge.edge) for edge in self._edges),
        )

    def to_json(self) -> str:
        """Serialize this graph to deterministic JSON."""

        try:
            return json.dumps(self.snapshot().to_dict(), sort_keys=True, separators=(",", ":"))
        except TypeError as exc:
            raise WorldGraphSerializationError(str(exc)) from exc

    def to_json_bytes(self) -> bytes:
        """Serialize this graph to deterministic UTF-8 JSON bytes."""

        return self.to_json().encode("utf-8")

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray | Mapping[str, Any] | WorldGraphSnapshot) -> Self:
        """Reconstruct a graph from deterministic JSON or a snapshot mapping."""

        if isinstance(payload, WorldGraphSnapshot):
            snapshot = payload
        else:
            if isinstance(payload, bytes | bytearray):
                payload = payload.decode("utf-8")
            if isinstance(payload, str):
                try:
                    parsed = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise WorldGraphSerializationError(str(exc)) from exc
            elif isinstance(payload, Mapping):
                parsed = payload
            else:
                raise TypeError(f"payload must be JSON, mapping, or snapshot, got {type(payload).__name__}")
            try:
                snapshot = WorldGraphSnapshot.from_dict(parsed)
            except (KeyError, TypeError, ValueError) as exc:
                raise WorldGraphSerializationError(str(exc)) from exc

        graph = cls(snapshot.registration)
        graph._schema_version = snapshot.schema_version
        for node in snapshot.nodes:
            graph.upsert_node(node)
        for source, target, edge in snapshot.edges:
            graph.add_edge(source, target, edge)
        graph._next_id = snapshot.next_id
        return graph


def _coerce_edge_record(value: Any) -> tuple[WorldId, WorldId, WorldEdge]:
    if isinstance(value, Mapping):
        source = value.get("source", value.get("from", value.get("from_id")))
        target = value.get("target", value.get("to", value.get("to_id")))
        edge = value["edge"]
        return WorldId.coerce(source), WorldId.coerce(target), WorldEdge.coerce(edge)
    if isinstance(value, Sequence) and not isinstance(value, str):
        if len(value) != 3:
            raise ValueError("edge record must contain source, target, and edge")
        return WorldId.coerce(value[0]), WorldId.coerce(value[1]), WorldEdge.coerce(value[2])
    raise TypeError(f"edge record must be a mapping or 3-item sequence, got {type(value).__name__}")


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    raise TypeError(f"registration must be a mapping, got {type(value).__name__}")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, WorldId):
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    return value


__all__ = [
    "PrivacyRollup",
    "SCHEMA_VERSION",
    "UnknownNodeError",
    "WorldGraph",
    "WorldGraphError",
    "WorldGraphSerializationError",
    "WorldGraphSnapshot",
]
