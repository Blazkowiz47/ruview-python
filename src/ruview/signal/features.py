"""Small CSI feature and normalization primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ruview.core import CsiFrame
from ruview.signal.csi_processor import CsiWindow, complex_to_amplitude_phase


@dataclass(frozen=True)
class SignalFeatures:
    """Scalar summary features for one CSI frame or a temporal CSI window."""

    mean_amplitude: float
    amplitude_variance: float
    phase_variance: float
    motion_energy: float


def min_max_normalize(data: ArrayLike) -> NDArray[np.float64]:
    """Normalize values to ``[0, 1]``; constant arrays map to zeros."""

    values = np.asarray(data, dtype=np.float64)
    if values.size == 0:
        return values.copy()

    minimum = float(np.min(values))
    maximum = float(np.max(values))
    span = maximum - minimum
    if abs(span) < np.finfo(np.float64).eps:
        return np.zeros_like(values, dtype=np.float64)
    return ((values - minimum) / span).astype(np.float64, copy=False)


def zscore_normalize(data: ArrayLike) -> NDArray[np.float64]:
    """Normalize values by population mean and standard deviation."""

    values = np.asarray(data, dtype=np.float64)
    if values.size == 0:
        return values.copy()

    std = float(np.std(values))
    if abs(std) < np.finfo(np.float64).eps:
        return np.zeros_like(values, dtype=np.float64)
    return ((values - float(np.mean(values))) / std).astype(np.float64, copy=False)


def extract_frame_features(
    frame: CsiFrame | ArrayLike,
    phase: ArrayLike | None = None,
) -> SignalFeatures:
    """Extract scalar features from one CSI frame or one complex frame array."""

    amplitude, phase_array = _coerce_amplitude_phase(frame, phase)
    return _features_from_arrays(amplitude, phase_array, motion_energy=0.0)


def extract_window_features(
    window: CsiWindow | ArrayLike,
    phase: ArrayLike | None = None,
    *,
    time_axis: int = 0,
) -> SignalFeatures:
    """Extract scalar features from a CSI window tensor."""

    amplitude, phase_array = _coerce_amplitude_phase(window, phase)
    motion_energy = calculate_motion_energy(amplitude, time_axis=time_axis)
    return _features_from_arrays(amplitude, phase_array, motion_energy=motion_energy)


def calculate_motion_energy(amplitude: ArrayLike, *, time_axis: int = 0) -> float:
    """Return mean squared frame-to-frame amplitude change along ``time_axis``."""

    values = np.asarray(amplitude, dtype=np.float64)
    if values.size == 0 or values.ndim == 0:
        return 0.0

    axis = _normalize_axis(time_axis, values.ndim)
    if values.shape[axis] < 2:
        return 0.0

    deltas = np.diff(values, axis=axis)
    return float(np.mean(np.square(deltas)))


def _features_from_arrays(
    amplitude: NDArray[np.float64],
    phase: NDArray[np.float64],
    *,
    motion_energy: float,
) -> SignalFeatures:
    if amplitude.shape != phase.shape:
        raise ValueError(
            f"amplitude and phase shapes must match, got {amplitude.shape} and {phase.shape}"
        )

    return SignalFeatures(
        mean_amplitude=float(np.mean(amplitude)) if amplitude.size else 0.0,
        amplitude_variance=float(np.var(amplitude)) if amplitude.size else 0.0,
        phase_variance=float(np.var(phase)) if phase.size else 0.0,
        motion_energy=motion_energy,
    )


def _coerce_amplitude_phase(
    source: Any,
    phase: ArrayLike | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if phase is not None:
        amplitude = np.asarray(source, dtype=np.float64)
        phase_array = np.asarray(phase, dtype=np.float64)
        if amplitude.shape != phase_array.shape:
            raise ValueError(
                "amplitude and phase shapes must match, "
                f"got {amplitude.shape} and {phase_array.shape}"
            )
        return amplitude, phase_array

    if isinstance(source, CsiWindow):
        return source.amplitude, source.phase
    if isinstance(source, CsiFrame):
        return source.amplitude, source.phase

    return complex_to_amplitude_phase(source)


def _normalize_axis(axis: int, ndim: int) -> int:
    if not -ndim <= axis < ndim:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {ndim}")
    return axis % ndim
