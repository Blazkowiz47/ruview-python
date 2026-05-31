"""Deterministic coverage and path-planning research helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from ruview.swarm import DroneState, NodeId, Position3D, coerce_node_id


@dataclass
class GridCell:
    """A cell in a 2-D mission-area probability grid."""

    x_idx: int
    y_idx: int
    victim_probability: float = 0.5
    pheromone: float = 0.0
    last_scanned_ms: int = 0

    def __post_init__(self) -> None:
        self.x_idx = int(self.x_idx)
        self.y_idx = int(self.y_idx)
        self.victim_probability = _clamp01(float(self.victim_probability))
        self.pheromone = _clamp01(float(self.pheromone))
        self.last_scanned_ms = max(0, int(self.last_scanned_ms))

    @property
    def scanned(self) -> bool:
        return self.last_scanned_ms > 0


class ProbabilityGrid:
    """2-D posterior grid for deterministic simulation planning."""

    def __init__(self, width: int, height: int, cell_size_m: float) -> None:
        self.width = int(width)
        self.height = int(height)
        self.cell_size_m = float(cell_size_m)
        if self.width < 0 or self.height < 0:
            raise ValueError("grid dimensions must be non-negative")
        if self.cell_size_m <= 0.0 or not math.isfinite(self.cell_size_m):
            raise ValueError("cell_size_m must be positive and finite")
        self.cells = [
            [GridCell(x_idx=x, y_idx=y) for x in range(self.width)]
            for y in range(self.height)
        ]

    def in_bounds(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= int(x) < self.width and 0 <= int(y) < self.height

    def cell(self, x: int, y: int) -> GridCell:
        if not self.in_bounds((x, y)):
            raise IndexError("grid cell is out of bounds")
        return self.cells[int(y)][int(x)]

    def set_probability(self, cell: tuple[int, int], probability: float) -> None:
        self.cell(*cell).victim_probability = _clamp01(float(probability))

    def update_bayesian(self, cell: tuple[int, int], confidence: float, detected: bool) -> None:
        """Apply a simple likelihood-ratio Bayesian update to one cell."""

        if not self.in_bounds(cell):
            return
        confidence = _clamp01(float(confidence))
        target = self.cell(*cell)
        prior = target.victim_probability
        likelihood = confidence if detected else 1.0 - confidence
        denom = likelihood * prior + (1.0 - likelihood) * (1.0 - prior)
        if denom > 1e-9:
            target.victim_probability = _clamp01(likelihood * prior / denom)
        target.pheromone = min(1.0, target.pheromone + 0.1)

    def highest_priority_unscanned(self) -> tuple[int, int] | None:
        """Return the highest-value cell under probability and pheromone penalty."""

        best: tuple[float, int, int] | None = None
        for row in self.cells:
            for cell in row:
                scanned_weight = cell.pheromone if cell.scanned else 0.0
                score = cell.victim_probability * (1.0 - scanned_weight)
                candidate = (score, -cell.y_idx, -cell.x_idx)
                if best is None or candidate > best:
                    best = candidate
        if best is None:
            return None
        _, neg_y, neg_x = best
        return (-neg_x, -neg_y)

    def highest_priority_cell(self) -> tuple[int, int] | None:
        """Compatibility alias for :meth:`highest_priority_unscanned`."""

        return self.highest_priority_unscanned()

    def mark_scanned(self, cell: tuple[int, int], timestamp_ms: int = 1) -> bool:
        """Mark a cell as scanned and return whether this was the first scan."""

        if not self.in_bounds(cell):
            return False
        target = self.cell(*cell)
        first = not target.scanned
        target.last_scanned_ms = max(1, int(timestamp_ms))
        return first

    def coverage_fraction(self) -> float:
        """Fraction of grid cells scanned at least once."""

        return coverage_fraction(self)

    def coverage_pct(self) -> float:
        """Rust-compatible alias for :meth:`coverage_fraction`."""

        return self.coverage_fraction()

    def next_systematic_cell(self, state: DroneState | None = None) -> tuple[int, int] | None:
        """Next unscanned cell in boustrophedon row order."""

        _ = state
        for y in range(self.height):
            if y % 2 == 0:
                x_iter = range(self.width)
            else:
                x_iter = range(self.width - 1, -1, -1)
            for x in x_iter:
                if not self.cells[y][x].scanned:
                    return (x, y)
        return None

    def apply_gossip_update(self, remote: "ProbabilityGrid") -> None:
        """Merge remote probabilities by deterministic average over overlap."""

        for y in range(min(self.height, remote.height)):
            for x in range(min(self.width, remote.width)):
                local = self.cells[y][x]
                local.victim_probability = (local.victim_probability + remote.cells[y][x].victim_probability) / 2.0


class CoveragePhase(str, Enum):
    """Phase of a coverage mission."""

    SYSTEMATIC = "systematic"
    PROBABILISTIC_PURSUIT = "probabilistic_pursuit"
    CONVERGENCE = "convergence"


@dataclass
class CoverageStrategy:
    """Systematic sweep to probabilistic pursuit to convergence."""

    convergence_threshold: float = 0.7
    phase: CoveragePhase = CoveragePhase.SYSTEMATIC
    assignments: dict[NodeId, tuple[int, int]] = field(default_factory=dict)
    convergence_drones: tuple[NodeId, ...] = ()

    def assign(self, node_id: NodeId | int, cell: tuple[int, int]) -> None:
        self.assignments[coerce_node_id(node_id)] = (int(cell[0]), int(cell[1]))

    def next_waypoint(
        self,
        node_id: NodeId | int,
        state: DroneState,
        grid: ProbabilityGrid,
        flight_altitude_m: float,
    ) -> Position3D:
        """Return the next waypoint for a drone, using assigned convergence cells first."""

        nid = coerce_node_id(node_id)
        if self.phase == CoveragePhase.CONVERGENCE and nid in self.assignments:
            x, y = self.assignments[nid]
        else:
            target = grid.highest_priority_unscanned()
            if target is None:
                return state.position
            x, y = target
        return Position3D(x * grid.cell_size_m, y * grid.cell_size_m, -float(flight_altitude_m))

    def next_target(self, state: DroneState, grid: ProbabilityGrid) -> Position3D | None:
        """Return the next cell-centre target for an orchestrator step."""

        if self.phase == CoveragePhase.SYSTEMATIC:
            cell = grid.next_systematic_cell(state)
        else:
            cell = grid.highest_priority_unscanned()
        if cell is None:
            return None
        x, y = cell
        radius = grid.cell_size_m
        return Position3D(x * radius + radius / 2.0, y * radius + radius / 2.0, state.position.z)

    def phase_transition_with_threshold(self, grid: ProbabilityGrid, threshold: float) -> CoveragePhase:
        """Transition with an explicit Systematic->Pursuit threshold."""

        return self.phase_transition(grid, convergence_threshold=threshold)

    def phase_transition(
        self,
        grid: ProbabilityGrid,
        *,
        convergence_threshold: float | None = None,
        confirmation_threshold: float = 0.9,
        contributing_drones: Sequence[NodeId | int] = (),
    ) -> CoveragePhase:
        """Update and return the coverage phase from current grid probabilities."""

        max_p = max((cell.victim_probability for row in grid.cells for cell in row), default=0.0)
        threshold = self.convergence_threshold if convergence_threshold is None else float(convergence_threshold)
        if self.phase == CoveragePhase.SYSTEMATIC and max_p >= threshold:
            self.phase = CoveragePhase.PROBABILISTIC_PURSUIT
        elif self.phase == CoveragePhase.PROBABILISTIC_PURSUIT and max_p >= float(confirmation_threshold):
            self.phase = CoveragePhase.CONVERGENCE
            self.convergence_drones = tuple(coerce_node_id(node_id) for node_id in contributing_drones)
        return self.phase


@dataclass(frozen=True)
class PatternContext:
    """Inputs for deterministic serpentine pattern helpers."""

    drone_id: NodeId
    swarm_size: int
    current: Position3D
    area_w: float
    area_h: float
    altitude_z: float
    scan_width_m: float
    step: int


@dataclass(frozen=True)
class Waypoint:
    """A planned simulation waypoint with target speed."""

    position: Position3D
    speed_ms: float = 5.0


@dataclass
class RrtApfPlanner:
    """Deterministic APF component of an RRT-APF planner."""

    apf_repulsion_dist: float
    obstacle_cells: list[Position3D] = field(default_factory=list)
    step_size_m: float = 2.0

    def apf_force(self, pos: Position3D, neighbors: Sequence[Position3D] = ()) -> tuple[float, float, float]:
        fx = fy = fz = 0.0
        for obstacle in [*self.obstacle_cells, *neighbors]:
            distance = pos.distance_to(obstacle)
            if 1e-6 < distance < self.apf_repulsion_dist:
                strength = (self.apf_repulsion_dist - distance) / (distance * distance)
                fx += strength * (pos.x - obstacle.x)
                fy += strength * (pos.y - obstacle.y)
                fz += strength * (pos.z - obstacle.z)
        return (fx, fy, fz)

    def direct_path(self, start: Position3D, goal: Position3D) -> list[Waypoint]:
        """Return a deterministic fallback path without random sampling."""

        return [Waypoint(start, speed_ms=5.0), Waypoint(goal, speed_ms=5.0)]


@dataclass(frozen=True)
class MissionMetrics:
    """Compact mission/evaluation summary."""

    coverage_fraction: float
    mean_uncertainty_m: float
    detection_rate: float
    n_detections: int


def next_serpentine_target(ctx: PatternContext) -> Position3D:
    """Return a partitioned lawnmower target for ``ctx.step``."""

    swarm_size = max(1, int(ctx.swarm_size))
    drone_index = int(ctx.drone_id) % swarm_size
    strip_w = float(ctx.area_w) / float(swarm_size)
    x0 = drone_index * strip_w
    x1 = x0 + strip_w
    x, y = _serpentine_in_region(x0, x1, 0.0, float(ctx.area_h), float(ctx.scan_width_m), int(ctx.step))
    return Position3D(_clamp(x, 0.0, ctx.area_w), _clamp(y, 0.0, ctx.area_h), ctx.altitude_z)


def coverage_fraction(grid: ProbabilityGrid | Sequence[Sequence[GridCell | bool]]) -> float:
    """Return the fraction of scanned cells for a grid-like object."""

    if isinstance(grid, ProbabilityGrid):
        cells = [cell for row in grid.cells for cell in row]
        total = len(cells)
        if total == 0:
            return 1.0
        return sum(1 for cell in cells if cell.scanned) / total

    flat = [item for row in grid for item in row]
    if not flat:
        return 1.0
    scanned = 0
    for item in flat:
        if isinstance(item, GridCell):
            scanned += int(item.scanned)
        else:
            scanned += int(bool(item))
    return scanned / len(flat)


def compute_mission_metrics(
    grid: ProbabilityGrid,
    *,
    uncertainties_m: Sequence[float] = (),
    detected_count: int | None = None,
    opportunities: int | None = None,
) -> MissionMetrics:
    """Compute deterministic mission summary metrics."""

    finite_uncertainties = [float(value) for value in uncertainties_m if math.isfinite(float(value))]
    mean_uncertainty = sum(finite_uncertainties) / len(finite_uncertainties) if finite_uncertainties else 0.0
    detections = len(finite_uncertainties) if detected_count is None else max(0, int(detected_count))
    denom = detections if opportunities is None else max(0, int(opportunities))
    detection_rate = 0.0 if denom == 0 else min(1.0, detections / denom)
    return MissionMetrics(
        coverage_fraction=coverage_fraction(grid),
        mean_uncertainty_m=mean_uncertainty,
        detection_rate=detection_rate,
        n_detections=detections,
    )


def _serpentine_in_region(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    scan_width_m: float,
    step: int,
) -> tuple[float, float]:
    scan = max(1e-9, float(scan_width_m))
    strip_w = max(x1 - x0, scan)
    height = max(y1 - y0, scan)
    cols = max(1, math.ceil(strip_w / scan))
    rows = max(1, math.ceil(height / scan))
    s = int(step) % (cols * rows)
    row = s // cols
    col = s % cols
    along = col if row % 2 == 0 else cols - 1 - col
    x = min(x1, x0 + (along + 0.5) * scan)
    y = min(y1, y0 + (row + 0.5) * scan)
    return x, y


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(float(hi), max(float(lo), float(value)))


__all__ = [
    "CoveragePhase",
    "CoverageStrategy",
    "GridCell",
    "MissionMetrics",
    "PatternContext",
    "ProbabilityGrid",
    "RrtApfPlanner",
    "Waypoint",
    "compute_mission_metrics",
    "coverage_fraction",
    "next_serpentine_target",
]
