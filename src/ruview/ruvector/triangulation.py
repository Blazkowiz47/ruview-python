"""RuVector-style 2D TDoA triangulation helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
SPEED_OF_LIGHT_M_S = 3.0e8


class TriangulationError(ValueError):
    """Base error for TDoA triangulation operations."""


@dataclass(frozen=True)
class TdoaMeasurement:
    """Time-difference observation between two access points."""

    ap_i: int
    ap_j: int
    tdoa_s: float


@dataclass(frozen=True)
class TdoaResult:
    """Estimated 2D target position from TDoA observations."""

    x: float
    y: float
    residual_m: float
    iterations: int
    converged: bool
    n_measurements: int

    @property
    def position(self) -> tuple[float, float]:
        return self.x, self.y

    def __iter__(self):
        yield self.x
        yield self.y


def solve_triangulation(
    tdoa_measurements: Sequence[TdoaMeasurement | tuple[int, int, float]],
    ap_positions: Sequence[tuple[float, float]] | ArrayLike,
    *,
    speed_m_s: float = SPEED_OF_LIGHT_M_S,
    ridge: float = 1e-6,
    max_iterations: int = 100,
    tolerance: float = 1e-9,
    initial_position: tuple[float, float] | ArrayLike | None = None,
) -> TdoaResult | None:
    """Estimate a 2D position from TDoA measurements.

    Returns ``None`` for the under-determined Rust-reference case of fewer
    than three measurements.  Valid problems are initialized with the reference
    linearized system and refined with deterministic Levenberg-Marquardt steps
    on the true range-difference residuals.
    """

    if len(tdoa_measurements) < 3:
        return None
    if speed_m_s <= 0.0 or not math.isfinite(float(speed_m_s)):
        raise TriangulationError("speed_m_s must be positive and finite")
    if ridge < 0.0:
        raise TriangulationError("ridge must be non-negative")
    if max_iterations <= 0:
        raise TriangulationError("max_iterations must be positive")
    if tolerance <= 0.0:
        raise TriangulationError("tolerance must be positive")

    aps = _coerce_ap_positions(ap_positions)
    if aps.shape[0] < 3:
        return None
    measurements = _coerce_measurements(tdoa_measurements, aps, float(speed_m_s))
    if measurements.shape[0] < 3:
        return None

    candidates = [_centroid(aps)]
    linear = _linearized_initial(measurements, aps, float(speed_m_s), ridge=max(ridge, 1e-9))
    if linear is not None:
        candidates.append(linear)
    if initial_position is not None:
        initial = np.asarray(initial_position, dtype=np.float64)
        if initial.shape != (2,) or not np.all(np.isfinite(initial)):
            raise TriangulationError("initial_position must be a finite 2D point")
        candidates.append(initial)

    best_initial = min(candidates, key=lambda point: _rmse(point, measurements, aps, float(speed_m_s)))
    return _refine_tdoa(
        best_initial,
        measurements,
        aps,
        speed_m_s=float(speed_m_s),
        ridge=float(ridge),
        max_iterations=int(max_iterations),
        tolerance=float(tolerance),
    )


def _refine_tdoa(
    initial: FloatArray,
    measurements: FloatArray,
    aps: FloatArray,
    *,
    speed_m_s: float,
    ridge: float,
    max_iterations: int,
    tolerance: float,
) -> TdoaResult:
    position = np.asarray(initial, dtype=np.float64).copy()
    damping = max(ridge, 1e-8)
    residuals, jacobian = _residuals_and_jacobian(position, measurements, aps, speed_m_s)
    error = float(residuals @ residuals)
    converged = False
    iterations = 0

    for iteration in range(1, max_iterations + 1):
        normal = jacobian.T @ jacobian
        normal.flat[::3] += damping
        rhs = -(jacobian.T @ residuals)
        try:
            step = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(normal, rhs, rcond=None)[0]

        if not np.all(np.isfinite(step)):
            damping *= 10.0
            continue

        max_step = _layout_scale(aps) * 2.0
        step_norm = float(np.linalg.norm(step))
        if step_norm > max_step:
            step *= max_step / step_norm
            step_norm = max_step

        trial = position + step
        trial_residuals, trial_jacobian = _residuals_and_jacobian(trial, measurements, aps, speed_m_s)
        trial_error = float(trial_residuals @ trial_residuals)

        if trial_error <= error:
            position = trial
            residuals = trial_residuals
            jacobian = trial_jacobian
            improvement = error - trial_error
            error = trial_error
            damping = max(damping * 0.5, 1e-12)
            iterations = iteration
            if step_norm <= tolerance or improvement <= tolerance * max(1.0, error):
                converged = True
                break
        else:
            damping *= 10.0
            iterations = iteration

    residual_m = float(math.sqrt(error / measurements.shape[0]))
    return TdoaResult(
        x=float(position[0]),
        y=float(position[1]),
        residual_m=residual_m,
        iterations=iterations,
        converged=converged,
        n_measurements=int(measurements.shape[0]),
    )


def _residuals_and_jacobian(
    position: FloatArray,
    measurements: FloatArray,
    aps: FloatArray,
    speed_m_s: float,
) -> tuple[FloatArray, FloatArray]:
    residuals = np.empty(measurements.shape[0], dtype=np.float64)
    jacobian = np.empty((measurements.shape[0], 2), dtype=np.float64)
    for row, (ap_i, ap_j, tdoa_s) in enumerate(measurements):
        i = int(ap_i)
        j = int(ap_j)
        vec_i = position - aps[i]
        vec_j = position - aps[j]
        dist_i = max(float(np.linalg.norm(vec_i)), 1e-9)
        dist_j = max(float(np.linalg.norm(vec_j)), 1e-9)
        expected_delta = float(tdoa_s) * speed_m_s
        residuals[row] = (dist_i - dist_j) - expected_delta
        jacobian[row] = (vec_i / dist_i) - (vec_j / dist_j)
    return residuals, jacobian


def _linearized_initial(
    measurements: FloatArray,
    aps: FloatArray,
    speed_m_s: float,
    *,
    ridge: float,
) -> FloatArray | None:
    reference = aps[0]
    rows = []
    rhs = []
    for ap_i, ap_j, tdoa_s in measurements:
        i = int(ap_i)
        j = int(ap_j)
        pi = aps[i]
        pj = aps[j]
        delta = speed_m_s * float(tdoa_s)
        row = pi - pj
        value = (
            0.5 * delta
            + 0.5 * ((pi[0] * pi[0] - pj[0] * pj[0]) + (pi[1] * pi[1] - pj[1] * pj[1]))
            - reference[0] * row[0]
            - reference[1] * row[1]
        )
        rows.append(row)
        rhs.append(value)

    a = np.asarray(rows, dtype=np.float64)
    b = np.asarray(rhs, dtype=np.float64)
    normal = a.T @ a
    normal.flat[::3] += ridge
    try:
        estimate = np.linalg.solve(normal, a.T @ b)
    except np.linalg.LinAlgError:
        estimate = np.linalg.lstsq(normal, a.T @ b, rcond=None)[0]
    if estimate.shape != (2,) or not np.all(np.isfinite(estimate)):
        return None
    return estimate


def _rmse(position: FloatArray, measurements: FloatArray, aps: FloatArray, speed_m_s: float) -> float:
    residuals, _jacobian = _residuals_and_jacobian(position, measurements, aps, speed_m_s)
    return float(np.sqrt(np.mean(residuals * residuals)))


def _coerce_ap_positions(ap_positions: Sequence[tuple[float, float]] | ArrayLike) -> FloatArray:
    values = np.asarray(ap_positions, dtype=np.float64)
    if values.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise TriangulationError("ap_positions must be shaped [n, 2]")
    if not np.all(np.isfinite(values)):
        raise TriangulationError("ap_positions must be finite")
    if np.unique(values, axis=0).shape[0] != values.shape[0]:
        raise TriangulationError("ap_positions must not contain duplicates")
    return values


def _coerce_measurements(
    measurements: Sequence[TdoaMeasurement | tuple[int, int, float]],
    aps: FloatArray,
    speed_m_s: float,
) -> FloatArray:
    rows: list[tuple[int, int, float]] = []
    n_aps = aps.shape[0]
    for measurement in measurements:
        if isinstance(measurement, TdoaMeasurement):
            ap_i = measurement.ap_i
            ap_j = measurement.ap_j
            tdoa_s = measurement.tdoa_s
        else:
            if len(measurement) != 3:
                raise TriangulationError("TDoA measurements must contain (ap_i, ap_j, tdoa_s)")
            ap_i, ap_j, tdoa_s = measurement
        i = int(ap_i)
        j = int(ap_j)
        if i == j:
            raise TriangulationError("TDoA measurement AP indices must differ")
        if not (0 <= i < n_aps and 0 <= j < n_aps):
            raise TriangulationError("TDoA measurement AP index out of range")
        tdoa = float(tdoa_s)
        if not math.isfinite(tdoa):
            raise TriangulationError("TDoA values must be finite")
        max_delta = float(np.linalg.norm(aps[i] - aps[j]))
        if abs(tdoa * speed_m_s) > max_delta + 1e-6:
            raise TriangulationError("TDoA range difference exceeds AP pair separation")
        rows.append((i, j, tdoa))

    values = np.asarray(rows, dtype=np.float64)
    if values.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return values


def _centroid(aps: FloatArray) -> FloatArray:
    return np.mean(aps, axis=0)


def _layout_scale(aps: FloatArray) -> float:
    if aps.size == 0:
        return 1.0
    span = np.ptp(aps, axis=0)
    scale = float(np.linalg.norm(span))
    return max(scale, 1.0)
