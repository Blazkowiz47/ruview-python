"""Rolling CSI baseline helpers for presence and motion classifiers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ruview.signal.csi_processor import CsiWindow, complex_to_amplitude_phase
from ruview.signal.features import calculate_motion_energy
from ruview.signal.phase import unwrap_phase
from ruview.signal.subcarrier import subcarrier_variance


@dataclass(frozen=True)
class BaselineStats:
    """Scalar baseline measurements for one CSI window or rolling estimate."""

    mean_amplitude: float = 0.0
    amplitude_variance: float = 0.0
    temporal_delta: float = 0.0
    phase_variance: float = 0.0
    subcarrier_variance: float = 0.0
    motion_energy: float = 0.0
    frame_count: int = 0

    def blend(self, other: "BaselineStats", alpha: float) -> "BaselineStats":
        """Return an exponential blend where ``alpha`` weights ``other``."""

        old_weight = 1.0 - alpha
        return BaselineStats(
            mean_amplitude=old_weight * self.mean_amplitude + alpha * other.mean_amplitude,
            amplitude_variance=old_weight * self.amplitude_variance
            + alpha * other.amplitude_variance,
            temporal_delta=old_weight * self.temporal_delta + alpha * other.temporal_delta,
            phase_variance=old_weight * self.phase_variance + alpha * other.phase_variance,
            subcarrier_variance=old_weight * self.subcarrier_variance
            + alpha * other.subcarrier_variance,
            motion_energy=old_weight * self.motion_energy + alpha * other.motion_energy,
            frame_count=other.frame_count,
        )


@dataclass
class RollingBaseline:
    """Exponentially smoothed baseline with bounded raw-history retention."""

    alpha: float = 0.1
    window: int = 64
    stats: BaselineStats = field(default_factory=BaselineStats)
    sample_count: int = 0
    _history: deque[BaselineStats] = field(default_factory=deque, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if self.window <= 0:
            raise ValueError("window must be positive")

    @property
    def is_ready(self) -> bool:
        """Whether the baseline has observed at least one window."""

        return self.sample_count > 0

    @property
    def history(self) -> tuple[BaselineStats, ...]:
        """Raw per-update measurements retained for adaptive thresholds."""

        return tuple(self._history)

    def update(
        self,
        source: CsiWindow | ArrayLike,
        phase: ArrayLike | None = None,
        *,
        time_axis: int = 0,
        subcarrier_axis: int = -1,
    ) -> BaselineStats:
        """Update the rolling baseline from a CSI window and return the estimate."""

        observed = measure_baseline(
            source,
            phase,
            time_axis=time_axis,
            subcarrier_axis=subcarrier_axis,
        )
        self._history.append(observed)
        while len(self._history) > self.window:
            self._history.popleft()

        self.stats = observed if not self.is_ready else self.stats.blend(observed, self.alpha)
        self.sample_count += 1
        return self.stats

    def adaptive_threshold(
        self,
        field_name: str,
        *,
        std_multiplier: float = 1.0,
        minimum: float = 0.0,
        maximum: float = 1.0,
        default: float = 0.5,
    ) -> float:
        """Return ``mean + multiplier * std`` for a retained baseline field."""

        values = [float(getattr(item, field_name)) for item in self._history]
        return adaptive_threshold(
            values,
            std_multiplier=std_multiplier,
            minimum=minimum,
            maximum=maximum,
            default=default,
        )


@dataclass
class DetectionDebouncer:
    """Consecutive-hit debounce with an EMA score for boolean detections."""

    confirm_count: int = 2
    release_count: int = 2
    smoothing_factor: float = 0.5
    state: bool = False
    smoothed_score: float = 0.0
    _positive_streak: int = field(default=0, init=False, repr=False)
    _negative_streak: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.confirm_count <= 0:
            raise ValueError("confirm_count must be positive")
        if self.release_count <= 0:
            raise ValueError("release_count must be positive")
        if not 0.0 <= self.smoothing_factor < 1.0:
            raise ValueError("smoothing_factor must be in [0, 1)")
        self.smoothed_score = _clamp01(self.smoothed_score)

    def update(self, detected: bool, *, confidence: float | None = None) -> bool:
        """Update with one raw detection and return the debounced state."""

        raw_score = float(detected) if confidence is None else _clamp01(confidence)
        self.smoothed_score = (
            self.smoothing_factor * self.smoothed_score
            + (1.0 - self.smoothing_factor) * raw_score
        )

        if detected:
            self._positive_streak += 1
            self._negative_streak = 0
            if self._positive_streak >= self.confirm_count:
                self.state = True
        else:
            self._negative_streak += 1
            self._positive_streak = 0
            if self._negative_streak >= self.release_count:
                self.state = False

        return self.state


def measure_baseline(
    source: CsiWindow | ArrayLike,
    phase: ArrayLike | None = None,
    *,
    time_axis: int = 0,
    subcarrier_axis: int = -1,
) -> BaselineStats:
    """Measure amplitude, phase, and temporal CSI statistics for one window."""

    amplitude, phase_array = coerce_amplitude_phase(source, phase)
    if amplitude.shape != phase_array.shape:
        raise ValueError(
            "amplitude and phase shapes must match, "
            f"got {amplitude.shape} and {phase_array.shape}"
        )
    if amplitude.size == 0:
        return BaselineStats()

    time_axis = _normalize_axis(time_axis, amplitude.ndim)
    frame_count = int(amplitude.shape[time_axis])
    temporal_delta = _mean_abs_delta(amplitude, axis=time_axis)
    motion_energy = calculate_motion_energy(amplitude, time_axis=time_axis)
    phase_variance = _phase_temporal_variance(phase_array, axis=time_axis)
    subcarrier_var = _mean_subcarrier_variance(
        amplitude,
        time_axis=time_axis,
        subcarrier_axis=subcarrier_axis,
    )

    return BaselineStats(
        mean_amplitude=float(np.mean(amplitude)),
        amplitude_variance=float(np.var(amplitude)),
        temporal_delta=temporal_delta,
        phase_variance=phase_variance,
        subcarrier_variance=subcarrier_var,
        motion_energy=motion_energy,
        frame_count=frame_count,
    )


def coerce_amplitude_phase(
    source: CsiWindow | ArrayLike,
    phase: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return float amplitude and phase arrays from a CSI window or complex array."""

    if phase is not None:
        return np.asarray(source, dtype=np.float64), np.asarray(phase, dtype=np.float64)
    if isinstance(source, CsiWindow):
        return source.amplitude, source.phase
    return complex_to_amplitude_phase(source)


def adaptive_threshold(
    values: Iterable[float] | ArrayLike,
    *,
    std_multiplier: float = 1.0,
    minimum: float = 0.0,
    maximum: float = 1.0,
    default: float = 0.5,
) -> float:
    """Return a clamped ``mean + std_multiplier * std`` threshold."""

    if minimum > maximum:
        raise ValueError("minimum must be less than or equal to maximum")

    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float(np.clip(default, minimum, maximum))

    threshold = float(np.mean(finite) + std_multiplier * np.std(finite))
    return float(np.clip(threshold, minimum, maximum))


def relative_increase(
    value: float,
    reference: float,
    *,
    relative_scale: float = 1.0,
    absolute_scale: float = 1.0,
    epsilon: float = 1e-12,
) -> float:
    """Normalize positive increase over a baseline into ``[0, 1]``."""

    if relative_scale <= 0.0:
        raise ValueError("relative_scale must be positive")
    if absolute_scale <= 0.0:
        raise ValueError("absolute_scale must be positive")
    if abs(reference) > epsilon:
        raw = (value - reference) / (abs(reference) * relative_scale)
    else:
        raw = value / absolute_scale
    return _clamp01(raw)


def _mean_abs_delta(values: NDArray[np.float64], *, axis: int) -> float:
    if values.shape[axis] < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(values, axis=axis))))


def _phase_temporal_variance(values: NDArray[np.float64], *, axis: int) -> float:
    if values.shape[axis] < 2:
        return 0.0
    unwrapped = unwrap_phase(values, axis=axis)
    return float(np.mean(np.var(unwrapped, axis=axis)))


def _mean_subcarrier_variance(
    values: NDArray[np.float64],
    *,
    time_axis: int,
    subcarrier_axis: int,
) -> float:
    if values.ndim < 2 or values.shape[time_axis] < 2:
        return 0.0
    subcarrier_axis = _normalize_axis(subcarrier_axis, values.ndim)
    if subcarrier_axis == time_axis:
        return 0.0
    variances = subcarrier_variance(
        values,
        time_axis=time_axis,
        subcarrier_axis=subcarrier_axis,
        ddof=1,
        aggregate_other_axes=True,
    )
    return float(np.mean(variances)) if variances.size else 0.0


def _normalize_axis(axis: int, ndim: int) -> int:
    if not -ndim <= axis < ndim:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {ndim}")
    return axis % ndim


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))
