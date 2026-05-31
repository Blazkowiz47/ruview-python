"""Deterministic swarm research primitives.

The objects in this package are simulation-only helpers for topology,
formation, coverage planning, and multiview CSI fusion.  They intentionally do
not include live flight-control, radio, or networking side effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _validate_finite_triplet(values: tuple[float, float, float], name: str) -> None:
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError(f"{name} values must be finite")


@dataclass(frozen=True, order=True)
class NodeId:
    """Unique identifier for a simulated swarm node."""

    value: int

    def __post_init__(self) -> None:
        value = int(self.value)
        if value < 0:
            raise ValueError("node id must be non-negative")
        object.__setattr__(self, "value", value)

    def __int__(self) -> int:
        return self.value

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class Position3D:
    """3-D position in a local NED-like frame, in metres."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __post_init__(self) -> None:
        _validate_finite_triplet((self.x, self.y, self.z), "position")

    @classmethod
    def zero(cls) -> "Position3D":
        return cls()

    def distance_to(self, other: "Position3D") -> float:
        return math.sqrt(
            (self.x - other.x) * (self.x - other.x)
            + (self.y - other.y) * (self.y - other.y)
            + (self.z - other.z) * (self.z - other.z)
        )

    def as_tuple(self) -> tuple[float, float, float]:
        return (float(self.x), float(self.y), float(self.z))


@dataclass(frozen=True)
class Velocity3D:
    """Velocity in a local NED-like frame, in m/s."""

    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    def __post_init__(self) -> None:
        _validate_finite_triplet((self.vx, self.vy, self.vz), "velocity")

    @classmethod
    def zero(cls) -> "Velocity3D":
        return cls()

    def magnitude(self) -> float:
        return math.sqrt(self.vx * self.vx + self.vy * self.vy + self.vz * self.vz)

    def as_tuple(self) -> tuple[float, float, float]:
        return (float(self.vx), float(self.vy), float(self.vz))


@dataclass(frozen=True)
class DroneState:
    """Full kinematic state for one simulated drone node."""

    id: NodeId
    position: Position3D = Position3D()
    velocity: Velocity3D = Velocity3D()
    heading_rad: float = 0.0
    altitude_agl_m: float = 0.0
    battery_pct: float = 100.0
    link_quality: float = 1.0
    timestamp_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", coerce_node_id(self.id))
        if not math.isfinite(float(self.heading_rad)):
            raise ValueError("heading_rad must be finite")
        if not math.isfinite(float(self.altitude_agl_m)):
            raise ValueError("altitude_agl_m must be finite")
        if not 0.0 <= float(self.battery_pct) <= 100.0:
            raise ValueError("battery_pct must be in [0, 100]")
        if not 0.0 <= float(self.link_quality) <= 1.0:
            raise ValueError("link_quality must be in [0, 1]")
        if int(self.timestamp_ms) < 0:
            raise ValueError("timestamp_ms must be non-negative")
        object.__setattr__(self, "timestamp_ms", int(self.timestamp_ms))

    @classmethod
    def default_at_origin(cls, node_id: NodeId | int) -> "DroneState":
        return cls(id=coerce_node_id(node_id))


@dataclass(frozen=True)
class CsiDetection:
    """CSI victim-location report emitted by one simulated sensing payload."""

    drone_id: NodeId
    confidence: float
    victim_position: Position3D | None
    timestamp_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "drone_id", coerce_node_id(self.drone_id))
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if int(self.timestamp_ms) < 0:
            raise ValueError("timestamp_ms must be non-negative")
        object.__setattr__(self, "timestamp_ms", int(self.timestamp_ms))


def coerce_node_id(value: NodeId | int) -> NodeId:
    """Return ``value`` as a :class:`NodeId`."""

    if isinstance(value, NodeId):
        return value
    return NodeId(int(value))


from ruview.swarm.formation import (  # noqa: E402
    LeaderFollower,
    ReynoldsParams,
    VirtualStructure,
    formation_vector_to_target,
)
from ruview.swarm.planning import (  # noqa: E402
    CoveragePhase,
    CoverageStrategy,
    GridCell,
    MissionMetrics,
    PatternContext,
    ProbabilityGrid,
    RrtApfPlanner,
    Waypoint,
    coverage_fraction,
    compute_mission_metrics,
    next_serpentine_target,
)
from ruview.swarm.sensing import (  # noqa: E402
    FusedDetection,
    MultiViewFusion,
    PayloadConfig,
    gdop,
    geometric_diversity_index,
    mean_uncertainty,
    synthetic_payload_scan,
)
from ruview.swarm.topology import (  # noqa: E402
    AppendEntries,
    AppendEntriesAck,
    GossipState,
    LogEntry,
    MeshTopology,
    RaftConfig,
    RaftMessage,
    RaftNode,
    RaftRole,
    RequestVote,
    VoteGranted,
)

__all__ = [
    "CoveragePhase",
    "CoverageStrategy",
    "CsiDetection",
    "DroneState",
    "FusedDetection",
    "GossipState",
    "GridCell",
    "LeaderFollower",
    "LogEntry",
    "MeshTopology",
    "MissionMetrics",
    "MultiViewFusion",
    "NodeId",
    "PatternContext",
    "PayloadConfig",
    "Position3D",
    "ProbabilityGrid",
    "RaftConfig",
    "RaftMessage",
    "RaftNode",
    "RaftRole",
    "ReynoldsParams",
    "RequestVote",
    "RrtApfPlanner",
    "Velocity3D",
    "VirtualStructure",
    "VoteGranted",
    "Waypoint",
    "AppendEntries",
    "AppendEntriesAck",
    "coerce_node_id",
    "compute_mission_metrics",
    "coverage_fraction",
    "formation_vector_to_target",
    "gdop",
    "geometric_diversity_index",
    "mean_uncertainty",
    "next_serpentine_target",
    "synthetic_payload_scan",
]
