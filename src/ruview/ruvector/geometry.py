"""RuVector-style Fresnel and viewpoint geometry helpers.

The Rust reference uses sparse solvers for tiny normal-equation systems.  This
module keeps the same behavior in pure NumPy: validate inputs, solve the small
systems directly, and preserve the Rust clamping/quality semantics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
NodeId = int


class GeometryError(ValueError):
    """Base error for RuVector geometry operations."""


@dataclass(frozen=True)
class FresnelGeometryResult:
    """Estimated TX-body and body-RX distances for one multistatic link."""

    d1: float
    d2: float
    total_distance: float
    residual: float
    n_observations: int

    @property
    def path_error(self) -> float:
        return abs((self.d1 + self.d2) - self.total_distance)

    @property
    def as_tuple(self) -> tuple[float, float]:
        return self.d1, self.d2

    def __iter__(self):
        yield self.d1
        yield self.d2


@dataclass(frozen=True)
class GeometricDiversityIndex:
    """Angular spread quality metric for a multistatic viewpoint array."""

    value: float
    n_effective: float
    worst_pair: tuple[NodeId, NodeId]
    n_physical: int

    def is_sufficient(self) -> bool:
        """Return whether the layout reaches half the uniform-spacing ideal."""

        if self.n_physical <= 0:
            return False
        ideal = 2.0 * math.pi / self.n_physical
        return self.value >= ideal * 0.5

    def efficiency(self) -> float:
        """Return effective viewpoints divided by physical viewpoints."""

        if self.n_physical <= 0:
            return 0.0
        return self.n_effective / self.n_physical


@dataclass(frozen=True)
class ViewpointPosition:
    """Sensor position and range-measurement noise used for CRB estimation."""

    x: float
    y: float
    noise_std: float = 1.0


@dataclass(frozen=True)
class CramerRaoBound:
    """Position-estimation Cramer-Rao bound and GDOP proxy."""

    crb_x: float
    crb_y: float
    rmse_lower_bound: float
    gdop: float


def solve_fresnel_geometry(
    observations: Sequence[tuple[float, float]] | ArrayLike,
    d_total: float,
    *,
    lambda_reg: float = 0.05,
    min_distance: float = 0.1,
) -> FresnelGeometryResult | None:
    """Estimate Fresnel path split distances from wavelength/amplitude samples.

    Returns ``None`` for the same under-determined case as the Rust reference:
    fewer than three observations.  Distances are clamped positive and adjusted
    so ``d1 + d2`` equals the known TX-RX distance.
    """

    values = _coerce_observations(observations)
    if values.shape[0] < 3:
        return None

    total = float(d_total)
    if not math.isfinite(total) or total <= 0.0:
        raise GeometryError("d_total must be a positive finite distance")
    if lambda_reg < 0.0:
        raise GeometryError("lambda_reg must be non-negative")
    if min_distance <= 0.0:
        raise GeometryError("min_distance must be positive")

    wavelengths = values[:, 0]
    amplitudes = values[:, 1]
    if np.any(wavelengths <= 0.0):
        raise GeometryError("observation wavelengths must be positive")

    sum_inv_w2 = float(np.sum(1.0 / (wavelengths * wavelengths)))
    rhs = float(np.sum(amplitudes / wavelengths))
    scale = lambda_reg + sum_inv_w2
    if scale <= 0.0:
        raise GeometryError("regularized system is singular")

    raw_d1 = abs(rhs / scale)
    lower, upper = _distance_bounds(total, min_distance)
    d1 = float(np.clip(raw_d1, lower, upper))
    d2 = float(total - d1)

    predicted = (d1 - d2) / wavelengths
    residual = float(np.sqrt(np.mean((predicted - amplitudes) ** 2)))
    return FresnelGeometryResult(
        d1=d1,
        d2=d2,
        total_distance=total,
        residual=residual,
        n_observations=int(values.shape[0]),
    )


def angular_distance(a: float, b: float) -> float:
    """Return the shortest distance between two angles in radians."""

    diff = abs(float(a) - float(b)) % (2.0 * math.pi)
    if diff > math.pi:
        return 2.0 * math.pi - diff
    return diff


def compute_effective_viewpoints(azimuths: ArrayLike, sigma: float = math.pi / 6.0) -> float:
    """Estimate independent viewpoint count under Gaussian angular correlation."""

    angles = _coerce_angles(azimuths)
    n = angles.size
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    if sigma <= 0.0 or not math.isfinite(float(sigma)):
        raise GeometryError("sigma must be a positive finite angle")

    two_sigma_sq = 2.0 * float(sigma) * float(sigma)
    corr = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            distance = angular_distance(float(angles[i]), float(angles[j]))
            rho = math.exp(-(distance * distance) / two_sigma_sq)
            corr[i, j] = rho
            corr[j, i] = rho

    trace_r = float(n)
    trace_r2 = float(np.sum(corr * corr))
    n_effective = (trace_r * trace_r) / max(trace_r2, np.finfo(np.float64).eps)
    return float(np.clip(n_effective, 1.0, float(n)))


def geometric_diversity_index(
    azimuths: ArrayLike,
    node_ids: Sequence[NodeId] | None = None,
) -> GeometricDiversityIndex | None:
    """Compute mean nearest-neighbor angular separation for viewpoints."""

    angles = _coerce_angles(azimuths)
    n = angles.size
    if n < 2:
        return None

    if node_ids is None:
        ids = tuple(range(n))
    else:
        ids = tuple(int(node_id) for node_id in node_ids)
    if len(ids) != n:
        return None

    min_separations = np.empty(n, dtype=np.float64)
    worst_sep = math.inf
    worst_i = 0
    worst_j = 1

    for i in range(n):
        min_sep = math.inf
        min_j = (i + 1) % n
        for j in range(n):
            if i == j:
                continue
            sep = angular_distance(float(angles[i]), float(angles[j]))
            if sep < min_sep:
                min_sep = sep
                min_j = j
        min_separations[i] = min_sep
        if min_sep < worst_sep:
            worst_sep = min_sep
            worst_i = i
            worst_j = min_j

    return GeometricDiversityIndex(
        value=float(np.mean(min_separations)),
        n_effective=compute_effective_viewpoints(angles),
        worst_pair=(ids[worst_i], ids[worst_j]),
        n_physical=int(n),
    )


def estimate_crb(
    target: tuple[float, float] | ArrayLike,
    viewpoints: Sequence[ViewpointPosition | tuple[float, float] | tuple[float, float, float]],
    *,
    regularization: float = 0.0,
) -> CramerRaoBound | None:
    """Estimate the 2D position CRB/GDOP for a target and viewpoint layout."""

    points = _coerce_viewpoints(viewpoints)
    if points.shape[0] < 3:
        return None
    if regularization < 0.0:
        raise GeometryError("regularization must be non-negative")

    target_xy = np.asarray(target, dtype=np.float64)
    if target_xy.shape != (2,) or not np.all(np.isfinite(target_xy)):
        raise GeometryError("target must be a finite 2D point")

    fim_00 = float(regularization)
    fim_01 = 0.0
    fim_11 = float(regularization)

    for x, y, noise_std in points:
        dx = float(target_xy[0] - x)
        dy = float(target_xy[1] - y)
        radius = max(math.hypot(dx, dy), 1e-6)
        cos_phi = dx / radius
        sin_phi = dy / radius
        inv_var = 1.0 / max(float(noise_std * noise_std), 1e-10)
        fim_00 += inv_var * cos_phi * cos_phi
        fim_01 += inv_var * cos_phi * sin_phi
        fim_11 += inv_var * sin_phi * sin_phi

    det = fim_00 * fim_11 - fim_01 * fim_01
    if abs(det) < 1e-12:
        return None

    crb_x = fim_11 / det
    crb_y = fim_00 / det
    rmse = math.sqrt(max(crb_x + crb_y, 0.0))
    return CramerRaoBound(
        crb_x=float(crb_x),
        crb_y=float(crb_y),
        rmse_lower_bound=float(rmse),
        gdop=float(rmse),
    )


def estimate_regularized_crb(
    target: tuple[float, float] | ArrayLike,
    viewpoints: Sequence[ViewpointPosition | tuple[float, float] | tuple[float, float, float]],
    regularization: float = 1e-4,
) -> CramerRaoBound | None:
    """Compatibility wrapper mirroring the Rust regularised CRB helper."""

    return estimate_crb(target, viewpoints, regularization=regularization)


def estimate_crb_gdop(
    target: tuple[float, float] | ArrayLike,
    viewpoints: Sequence[ViewpointPosition | tuple[float, float] | tuple[float, float, float]],
    *,
    regularization: float = 0.0,
) -> CramerRaoBound | None:
    """Alias for callers that care about the GDOP field explicitly."""

    return estimate_crb(target, viewpoints, regularization=regularization)


def _coerce_observations(observations: Sequence[tuple[float, float]] | ArrayLike) -> FloatArray:
    values = np.asarray(observations, dtype=np.float64)
    if values.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise GeometryError("observations must be shaped [n, 2]")
    if not np.all(np.isfinite(values)):
        raise GeometryError("observations must be finite")
    return values


def _coerce_angles(azimuths: ArrayLike) -> FloatArray:
    values = np.asarray(azimuths, dtype=np.float64)
    if values.ndim != 1:
        raise GeometryError("azimuths must be a one-dimensional sequence")
    if not np.all(np.isfinite(values)):
        raise GeometryError("azimuths must be finite")
    return values


def _coerce_viewpoints(
    viewpoints: Sequence[ViewpointPosition | tuple[float, float] | tuple[float, float, float]],
) -> FloatArray:
    rows: list[tuple[float, float, float]] = []
    for viewpoint in viewpoints:
        if isinstance(viewpoint, ViewpointPosition):
            rows.append((float(viewpoint.x), float(viewpoint.y), float(viewpoint.noise_std)))
            continue
        if len(viewpoint) == 2:
            x, y = viewpoint
            rows.append((float(x), float(y), 1.0))
            continue
        if len(viewpoint) == 3:
            x, y, noise_std = viewpoint
            rows.append((float(x), float(y), float(noise_std)))
            continue
        raise GeometryError("viewpoints must contain (x, y) or (x, y, noise_std)")

    values = np.asarray(rows, dtype=np.float64)
    if values.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
        raise GeometryError("viewpoints must be finite 2D positions")
    if np.any(values[:, 2] <= 0.0):
        raise GeometryError("viewpoint noise_std must be positive")
    return values


def _distance_bounds(total: float, min_distance: float) -> tuple[float, float]:
    lower = min(float(min_distance), total * 0.5)
    lower = max(lower, np.finfo(np.float64).tiny)
    upper = max(total - lower, lower)
    return lower, upper
