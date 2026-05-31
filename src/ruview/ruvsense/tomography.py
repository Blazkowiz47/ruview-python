"""Coarse RF tomography helpers and deterministic ridge inversion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


class TomographyError(ValueError):
    """Base error for RF tomography operations."""


@dataclass(frozen=True)
class Position3D:
    """A 3D point in meters."""

    x: float
    y: float
    z: float = 0.0

    def as_array(self) -> FloatArray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)


@dataclass(frozen=True)
class LinkGeometry:
    """Transmitter-receiver line segment used for tomography."""

    tx: Position3D | Sequence[float]
    rx: Position3D | Sequence[float]
    link_id: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "tx", _coerce_position(self.tx))
        object.__setattr__(self, "rx", _coerce_position(self.rx))
        object.__setattr__(self, "link_id", int(self.link_id))

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self.rx.as_array() - self.tx.as_array()))


@dataclass(frozen=True)
class TomographyConfig:
    """Voxel grid and ridge solver settings."""

    nx: int = 8
    ny: int = 8
    nz: int = 4
    bounds: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 6.0, 6.0, 3.0)
    ridge: float = 1e-2
    min_links: int = 1
    ray_radius: float | None = None
    wavelength_m: float = 0.06
    occupancy_threshold: float = 1e-2
    nonnegative: bool = True

    def __post_init__(self) -> None:
        if self.nx <= 0 or self.ny <= 0 or self.nz <= 0:
            raise TomographyError("grid dimensions must be positive")
        if len(self.bounds) != 6:
            raise TomographyError("bounds must contain six values")
        x0, y0, z0, x1, y1, z1 = (float(value) for value in self.bounds)
        if x1 <= x0 or y1 <= y0 or z1 <= z0:
            raise TomographyError("bounds maxima must exceed minima")
        if self.ridge < 0.0:
            raise TomographyError("ridge must be non-negative")
        if self.min_links <= 0:
            raise TomographyError("min_links must be positive")
        if self.ray_radius is not None and self.ray_radius <= 0.0:
            raise TomographyError("ray_radius must be positive when provided")
        if self.wavelength_m <= 0.0:
            raise TomographyError("wavelength_m must be positive")
        if self.occupancy_threshold < 0.0:
            raise TomographyError("occupancy_threshold must be non-negative")

        object.__setattr__(self, "nx", int(self.nx))
        object.__setattr__(self, "ny", int(self.ny))
        object.__setattr__(self, "nz", int(self.nz))
        object.__setattr__(self, "bounds", (x0, y0, z0, x1, y1, z1))
        object.__setattr__(self, "min_links", int(self.min_links))

    @property
    def n_voxels(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def voxel_size(self) -> tuple[float, float, float]:
        x0, y0, z0, x1, y1, z1 = self.bounds
        return ((x1 - x0) / self.nx, (y1 - y0) / self.ny, (z1 - z0) / self.nz)


@dataclass(frozen=True)
class VoxelWeight:
    """Weight assigned to one voxel for one link."""

    index: int
    ix: int
    iy: int
    iz: int
    weight: float


@dataclass(frozen=True)
class OccupancyVolume:
    """3D occupancy density volume produced by tomographic inversion."""

    densities: FloatArray
    config: TomographyConfig
    residual: float
    iterations: int = 1

    def __post_init__(self) -> None:
        values = np.asarray(self.densities, dtype=np.float64)
        expected = (self.config.nz, self.config.ny, self.config.nx)
        if values.shape == (self.config.n_voxels,):
            values = values.reshape(expected)
        if values.shape != expected:
            raise TomographyError(f"densities must have shape {expected} or flat voxel count")
        object.__setattr__(self, "densities", values)
        object.__setattr__(self, "residual", float(self.residual))
        object.__setattr__(self, "iterations", int(self.iterations))

    @property
    def total_voxels(self) -> int:
        return self.config.n_voxels

    @property
    def occupied_count(self) -> int:
        return int(np.count_nonzero(self.densities > self.config.occupancy_threshold))

    def get(self, ix: int, iy: int, iz: int) -> float | None:
        if 0 <= ix < self.config.nx and 0 <= iy < self.config.ny and 0 <= iz < self.config.nz:
            return float(self.densities[iz, iy, ix])
        return None

    def voxel_center(self, ix: int, iy: int, iz: int) -> Position3D:
        return voxel_center(ix, iy, iz, self.config)

    def voxel_size(self) -> tuple[float, float, float]:
        return self.config.voxel_size

    def heatmap(self, *, axis: str = "z", reduce: str = "max") -> FloatArray:
        """Collapse the 3D volume into a 2D occupancy heatmap."""

        reducer = np.max if reduce == "max" else np.sum if reduce == "sum" else None
        if reducer is None:
            raise TomographyError("reduce must be 'max' or 'sum'")
        if axis == "z":
            return reducer(self.densities, axis=0)
        if axis == "y":
            return reducer(self.densities, axis=1)
        if axis == "x":
            return reducer(self.densities, axis=2)
        raise TomographyError("axis must be 'x', 'y', or 'z'")

    def peak_index(self) -> tuple[int, int, int]:
        iz, iy, ix = np.unravel_index(int(np.argmax(self.densities)), self.densities.shape)
        return int(ix), int(iy), int(iz)

    def peak_position(self) -> Position3D:
        return self.voxel_center(*self.peak_index())


class RfTomographer:
    """Precomputed RF tomography inverse over a fixed link geometry."""

    def __init__(self, config: TomographyConfig, links: Iterable[LinkGeometry]) -> None:
        self.config = config
        self.links = tuple(LinkGeometry(link.tx, link.rx, link.link_id) for link in links)
        if len(self.links) < config.min_links:
            raise TomographyError(f"insufficient links: need >= {config.min_links}, got {len(self.links)}")
        self.weights = link_weight_matrix(self.links, config)
        if not np.any(self.weights):
            raise TomographyError("no voxels intersected by any link")

    @property
    def n_links(self) -> int:
        return len(self.links)

    @property
    def n_voxels(self) -> int:
        return self.config.n_voxels

    def reconstruct(self, attenuations: ArrayLike) -> OccupancyVolume:
        y = np.asarray(attenuations, dtype=np.float64)
        if y.shape != (self.n_links,):
            raise TomographyError(f"expected {self.n_links} attenuation values, got shape {y.shape}")
        if not np.all(np.isfinite(y)):
            raise TomographyError("attenuations must be finite")

        a = self.weights
        if self.config.ridge > 0.0:
            normal = a.T @ a
            normal.flat[:: normal.shape[0] + 1] += self.config.ridge
            rhs = a.T @ y
            try:
                density = np.linalg.solve(normal, rhs)
            except np.linalg.LinAlgError:
                density = np.linalg.lstsq(normal, rhs, rcond=None)[0]
        else:
            density = np.linalg.lstsq(a, y, rcond=None)[0]

        if self.config.nonnegative:
            density = np.maximum(density, 0.0)

        predicted = a @ density
        residual = float(np.sqrt(np.mean((predicted - y) ** 2))) if y.size else 0.0
        return OccupancyVolume(density, self.config, residual=residual, iterations=1)


def voxel_index(ix: int, iy: int, iz: int, config: TomographyConfig) -> int:
    if not (0 <= ix < config.nx and 0 <= iy < config.ny and 0 <= iz < config.nz):
        raise TomographyError(f"voxel index out of bounds: {(ix, iy, iz)}")
    return iz * config.ny * config.nx + iy * config.nx + ix


def unravel_voxel(index: int, config: TomographyConfig) -> tuple[int, int, int]:
    if not 0 <= index < config.n_voxels:
        raise TomographyError(f"voxel index out of bounds: {index}")
    iz, rem = divmod(int(index), config.nx * config.ny)
    iy, ix = divmod(rem, config.nx)
    return ix, iy, iz


def voxel_center(ix: int, iy: int, iz: int, config: TomographyConfig) -> Position3D:
    idx = voxel_index(ix, iy, iz, config)
    _ = idx
    vx, vy, vz = config.voxel_size
    x0, y0, z0, _x1, _y1, _z1 = config.bounds
    return Position3D(x0 + (ix + 0.5) * vx, y0 + (iy + 0.5) * vy, z0 + (iz + 0.5) * vz)


def rasterize_link(link: LinkGeometry, config: TomographyConfig) -> list[VoxelWeight]:
    """Return voxels inside the link Fresnel/ray tube."""

    start = link.tx.as_array()
    end = link.rx.as_array()
    link_distance = max(float(np.linalg.norm(end - start)), 1e-12)
    radius = _link_radius(link_distance, config)
    weights: list[VoxelWeight] = []

    for iz in range(config.nz):
        for iy in range(config.ny):
            for ix in range(config.nx):
                center = voxel_center(ix, iy, iz, config).as_array()
                distance = point_to_segment_distance(center, start, end)
                if distance <= radius:
                    weight = max(0.0, 1.0 - distance / radius)
                    if weight > 0.0:
                        weights.append(VoxelWeight(voxel_index(ix, iy, iz, config), ix, iy, iz, float(weight)))
    return weights


def link_weight_matrix(links: Sequence[LinkGeometry], config: TomographyConfig) -> FloatArray:
    matrix = np.zeros((len(links), config.n_voxels), dtype=np.float64)
    for row, link in enumerate(links):
        for voxel in rasterize_link(link, config):
            matrix[row, voxel.index] = voxel.weight
    return matrix


def point_to_segment_distance(point: ArrayLike, start: ArrayLike, end: ArrayLike) -> float:
    point_arr = np.asarray(point, dtype=np.float64)
    start_arr = np.asarray(start, dtype=np.float64)
    end_arr = np.asarray(end, dtype=np.float64)
    segment = end_arr - start_arr
    length_sq = float(segment @ segment)
    if length_sq <= 1e-24:
        return float(np.linalg.norm(point_arr - start_arr))
    t = float(((point_arr - start_arr) @ segment) / length_sq)
    t = min(1.0, max(0.0, t))
    closest = start_arr + t * segment
    return float(np.linalg.norm(point_arr - closest))


def _link_radius(link_distance: float, config: TomographyConfig) -> float:
    if config.ray_radius is not None:
        return float(config.ray_radius)
    vx, vy, vz = config.voxel_size
    fresnel = math.sqrt(config.wavelength_m * link_distance / 4.0)
    return max(fresnel, 0.55 * min(vx, vy, vz))


def _coerce_position(value: Position3D | Sequence[float]) -> Position3D:
    if isinstance(value, Position3D):
        return value
    coords = tuple(float(v) for v in value)
    if len(coords) == 2:
        return Position3D(coords[0], coords[1], 0.0)
    if len(coords) == 3:
        return Position3D(coords[0], coords[1], coords[2])
    raise TomographyError("positions must contain two or three coordinates")


__all__ = [
    "LinkGeometry",
    "OccupancyVolume",
    "Position3D",
    "RfTomographer",
    "TomographyConfig",
    "TomographyError",
    "VoxelWeight",
    "link_weight_matrix",
    "point_to_segment_distance",
    "rasterize_link",
    "unravel_voxel",
    "voxel_center",
    "voxel_index",
]
