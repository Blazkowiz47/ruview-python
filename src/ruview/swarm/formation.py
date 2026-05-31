"""Simulation-only formation vector helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ruview.swarm import NodeId, Position3D, Velocity3D, coerce_node_id

PositionEntry = tuple[NodeId | int, Position3D]


@dataclass(frozen=True)
class ReynoldsParams:
    """Parameters for position-only Reynolds flocking."""

    separation_dist_m: float = 3.0
    separation_weight: float = 1.5
    alignment_weight: float = 1.0
    cohesion_weight: float = 0.8
    k_neighbors: int = 7

    def compute_velocity(self, node_id: NodeId | int, positions: Sequence[PositionEntry]) -> Velocity3D:
        """Compute the separation/cohesion velocity delta for ``node_id``."""

        nid = coerce_node_id(node_id)
        own_pos = _positions_dict(positions).get(nid)
        if own_pos is None:
            return Velocity3D.zero()

        neighbors = [
            (own_pos.distance_to(pos), int(other_id), pos)
            for other_id, pos in _positions_dict(positions).items()
            if other_id != nid
        ]
        neighbors.sort(key=lambda item: (item[0], item[1]))
        neighbors = neighbors[: max(0, int(self.k_neighbors))]
        if not neighbors:
            return Velocity3D.zero()

        sep_x = sep_y = sep_z = 0.0
        for distance, _, pos in neighbors:
            if 1e-6 < distance < self.separation_dist_m:
                factor = (self.separation_dist_m - distance) / self.separation_dist_m
                sep_x += (own_pos.x - pos.x) / distance * factor
                sep_y += (own_pos.y - pos.y) / distance * factor
                sep_z += (own_pos.z - pos.z) / distance * factor

        n = float(len(neighbors))
        avg_x = sum(pos.x for _, _, pos in neighbors) / n
        avg_y = sum(pos.y for _, _, pos in neighbors) / n
        avg_z = sum(pos.z for _, _, pos in neighbors) / n
        coh_x = avg_x - own_pos.x
        coh_y = avg_y - own_pos.y
        coh_z = avg_z - own_pos.z

        return Velocity3D(
            vx=self.separation_weight * sep_x + self.cohesion_weight * coh_x,
            vy=self.separation_weight * sep_y + self.cohesion_weight * coh_y,
            vz=self.separation_weight * sep_z + self.cohesion_weight * coh_z,
        )


@dataclass
class LeaderFollower:
    """Followers maintain fixed offsets from one leader."""

    leader_id: NodeId
    offsets: dict[NodeId, tuple[float, float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.leader_id = coerce_node_id(self.leader_id)

    def add_follower(self, follower: NodeId | int, offset: tuple[float, float, float]) -> None:
        self.offsets[coerce_node_id(follower)] = tuple(float(value) for value in offset)

    def target_position(self, node_id: NodeId | int, positions: Sequence[PositionEntry]) -> Position3D:
        """Return the node's desired position under the leader-follower layout."""

        nid = coerce_node_id(node_id)
        by_id = _positions_dict(positions)
        leader_pos = by_id.get(self.leader_id, Position3D.zero())
        if nid == self.leader_id:
            return leader_pos
        dx, dy, dz = self.offsets.get(nid, (0.0, 0.0, 0.0))
        return Position3D(leader_pos.x + dx, leader_pos.y + dy, leader_pos.z + dz)

    def formation_vector(
        self,
        node_id: NodeId | int,
        positions: Sequence[PositionEntry],
        *,
        gain: float = 1.0,
        max_magnitude: float | None = None,
    ) -> Velocity3D:
        """Return a velocity-like vector from current position to formation target."""

        nid = coerce_node_id(node_id)
        by_id = _positions_dict(positions)
        current = by_id.get(nid)
        if current is None:
            return Velocity3D.zero()
        return formation_vector_to_target(
            current,
            self.target_position(nid, positions),
            gain=gain,
            max_magnitude=max_magnitude,
        )


@dataclass(frozen=True)
class VirtualStructure:
    """Fixed offsets from a shared reference point."""

    offsets: Mapping[NodeId, tuple[float, float, float]]

    @classmethod
    def grid_formation(cls, n: int, spacing_m: float) -> "VirtualStructure":
        cols = max(1, math.ceil(math.sqrt(max(0, n))))
        offsets = {
            NodeId(index): (
                float(index // cols) * float(spacing_m),
                float(index % cols) * float(spacing_m),
                0.0,
            )
            for index in range(max(0, int(n)))
        }
        return cls(offsets)

    @classmethod
    def circle_formation(cls, n: int, radius_m: float) -> "VirtualStructure":
        count = max(0, int(n))
        if count == 0:
            return cls({})
        offsets = {}
        for index in range(count):
            angle = math.tau * float(index) / float(count)
            offsets[NodeId(index)] = (float(radius_m) * math.cos(angle), float(radius_m) * math.sin(angle), 0.0)
        return cls(offsets)

    def target_position(self, node_id: NodeId | int, reference: Position3D) -> Position3D:
        dx, dy, dz = self.offsets.get(coerce_node_id(node_id), (0.0, 0.0, 0.0))
        return Position3D(reference.x + dx, reference.y + dy, reference.z + dz)


def formation_vector_to_target(
    current: Position3D,
    target: Position3D,
    *,
    gain: float = 1.0,
    max_magnitude: float | None = None,
) -> Velocity3D:
    """Return a bounded vector pointing from ``current`` toward ``target``."""

    vx = (target.x - current.x) * float(gain)
    vy = (target.y - current.y) * float(gain)
    vz = (target.z - current.z) * float(gain)
    velocity = Velocity3D(vx, vy, vz)
    if max_magnitude is None:
        return velocity
    limit = float(max_magnitude)
    if limit < 0.0:
        raise ValueError("max_magnitude must be non-negative")
    magnitude = velocity.magnitude()
    if magnitude <= limit or magnitude <= 1e-12:
        return velocity
    scale = limit / magnitude
    return Velocity3D(velocity.vx * scale, velocity.vy * scale, velocity.vz * scale)


def _positions_dict(positions: Sequence[PositionEntry]) -> dict[NodeId, Position3D]:
    return {coerce_node_id(node_id): position for node_id, position in positions}


__all__ = [
    "LeaderFollower",
    "ReynoldsParams",
    "VirtualStructure",
    "formation_vector_to_target",
]
