"""Heart-rate residual fusion and FFT-based cardiac estimation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ruview.vitals.quality import (
    VitalEstimate,
    VitalStatus,
    estimate_rate_from_samples,
    signal_quality_from_confidence,
)
from ruview.vitals.smoothing import BpmSmoothingBuffer, SampleBuffer

HEART_RATE_BAND_HZ = (0.8, 2.0)


def extract_heart_residual(
    residuals: ArrayLike,
    phases: ArrayLike | None = None,
) -> NDArray[np.float64] | float:
    """Fuse residuals with inter-subcarrier phase-coherence weights."""

    values = np.asarray(residuals, dtype=np.float64)
    if values.size == 0:
        return 0.0
    if values.ndim == 1:
        return float(_coherence_weighted_mean(values, phases))
    return np.asarray([_coherence_weighted_mean(row, _phase_row(phases, i)) for i, row in enumerate(values)])


@dataclass
class HeartRateEstimator:
    """Stateful heart-rate estimator for the 0.8-2.0 Hz cardiac band."""

    sample_rate_hz: float = 100.0
    window_seconds: float = 15.0
    n_subcarriers: int = 56
    min_duration_seconds: float = 5.0
    min_samples: int | None = None
    min_subcarriers: int = 4
    smoother: BpmSmoothingBuffer | None = field(
        default_factory=lambda: BpmSmoothingBuffer(mode="median", window=3)
    )

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive")
        if self.window_seconds <= 0.0:
            raise ValueError("window_seconds must be positive")
        if self.n_subcarriers <= 0:
            raise ValueError("n_subcarriers must be positive")
        if self.min_subcarriers <= 0:
            raise ValueError("min_subcarriers must be positive")
        capacity = max(int(round(self.sample_rate_hz * self.window_seconds)), 1)
        self._buffer = SampleBuffer(capacity)
        self._last_subcarrier_count = 0
        if self.min_samples is None:
            self.min_samples = max(int(round(self.sample_rate_hz * self.min_duration_seconds)), 1)

    @classmethod
    def esp32_default(cls) -> "HeartRateEstimator":
        """Return ESP32 defaults: 56 subcarriers, 100 Hz, 15 seconds."""

        return cls(sample_rate_hz=100.0, window_seconds=15.0, n_subcarriers=56)

    def update(self, residuals: ArrayLike, phases: ArrayLike | None = None) -> VitalEstimate:
        """Append one residual/phase frame and return the latest heart-rate estimate."""

        residual_arr = np.asarray(residuals, dtype=np.float64).reshape(-1)[: self.n_subcarriers]
        phase_arr = None if phases is None else np.asarray(phases, dtype=np.float64).reshape(-1)[: residual_arr.size]
        self._last_subcarrier_count = int(residual_arr.size)
        sample = extract_heart_residual(residual_arr, phase_arr)
        self._buffer.append(float(sample))
        return self.estimate()

    def estimate(self) -> VitalEstimate:
        """Estimate heart rate from the current cardiac residual buffer."""

        estimate = estimate_rate_from_samples(
            self._buffer.values(),
            sample_rate_hz=self.sample_rate_hz,
            band_hz=HEART_RATE_BAND_HZ,
            min_duration_seconds=self.min_duration_seconds,
            min_samples=int(self.min_samples or 1),
        )
        estimate = self._apply_subcarrier_confidence(estimate)
        return self.smoother.smooth_estimate(estimate) if self.smoother is not None else estimate

    def reset(self) -> None:
        self._buffer.clear()
        self._last_subcarrier_count = 0
        if self.smoother is not None:
            self.smoother.clear()

    @property
    def history_len(self) -> int:
        return len(self._buffer)

    @property
    def band_hz(self) -> tuple[float, float]:
        return HEART_RATE_BAND_HZ

    @property
    def samples(self) -> NDArray[np.float64]:
        return self._buffer.values()

    def _apply_subcarrier_confidence(self, estimate: VitalEstimate) -> VitalEstimate:
        if estimate.status is VitalStatus.UNAVAILABLE:
            return estimate
        subcarrier_factor = min(self._last_subcarrier_count / self.min_subcarriers, 1.0)
        confidence = float(np.clip(estimate.confidence * subcarrier_factor, 0.0, 1.0))
        if confidence >= 0.72 and self._last_subcarrier_count >= self.min_subcarriers:
            status = VitalStatus.VALID
        elif confidence >= 0.45:
            status = VitalStatus.DEGRADED
        elif confidence >= 0.18:
            status = VitalStatus.UNRELIABLE
        else:
            status = VitalStatus.UNAVAILABLE
        return VitalEstimate(
            value_bpm=estimate.value_bpm if status is not VitalStatus.UNAVAILABLE else 0.0,
            confidence=confidence,
            status=status,
            quality=signal_quality_from_confidence(confidence),
            peak_hz=estimate.peak_hz,
            peak_prominence=estimate.peak_prominence,
            in_band_energy_ratio=estimate.in_band_energy_ratio,
            duration_seconds=estimate.duration_seconds,
            sample_count=estimate.sample_count,
        )


HeartRateExtractor = HeartRateEstimator


def _coherence_weighted_mean(residuals: NDArray[np.float64], phases: ArrayLike | None) -> np.float64:
    values = np.asarray(residuals, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.float64(0.0)
    if phases is None:
        return np.float64(np.mean(values))

    phase_values = np.asarray(phases, dtype=np.float64).reshape(-1)[: values.size]
    if phase_values.size <= 1:
        return np.float64(np.mean(values[: phase_values.size or values.size]))

    weights = np.ones(phase_values.size, dtype=np.float64)
    for index in range(phase_values.size):
        if index + 1 < phase_values.size:
            diff = _wrapped_phase_diff(phase_values[index + 1], phase_values[index])
        else:
            diff = _wrapped_phase_diff(phase_values[index], phase_values[index - 1])
        weights[index] = np.exp(-abs(diff))
    total = float(np.sum(weights))
    if total <= 1e-12:
        return np.float64(np.mean(values[: phase_values.size]))
    return np.float64(np.sum(values[: phase_values.size] * weights) / total)


def _wrapped_phase_diff(a: float, b: float) -> float:
    return float(np.angle(np.exp(1j * (a - b))))


def _phase_row(phases: ArrayLike | None, index: int) -> NDArray[np.float64] | None:
    if phases is None:
        return None
    phase_values = np.asarray(phases, dtype=np.float64)
    if phase_values.ndim == 1:
        return phase_values
    return phase_values[index]
