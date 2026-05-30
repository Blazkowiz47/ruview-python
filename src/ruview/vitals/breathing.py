"""Respiratory residual fusion and FFT-based breathing-rate estimation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ruview.vitals.quality import VitalEstimate, estimate_rate_from_samples
from ruview.vitals.smoothing import BpmSmoothingBuffer, SampleBuffer

BREATHING_BAND_HZ = (0.1, 0.5)


def extract_breathing_residual(
    residuals: ArrayLike,
    weights: ArrayLike | None = None,
) -> NDArray[np.float64] | float:
    """Fuse per-subcarrier residuals into one breathing-sensitive signal."""

    values = np.asarray(residuals, dtype=np.float64)
    if values.size == 0:
        return 0.0
    if values.ndim == 1:
        return float(_weighted_mean(values, weights))
    return _weighted_mean(values, weights, axis=-1)


@dataclass
class BreathingRateEstimator:
    """Stateful breathing-rate estimator for the 0.1-0.5 Hz band."""

    sample_rate_hz: float = 100.0
    window_seconds: float = 30.0
    n_subcarriers: int = 56
    min_duration_seconds: float = 10.0
    min_samples: int | None = None
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
        capacity = max(int(round(self.sample_rate_hz * self.window_seconds)), 1)
        self._buffer = SampleBuffer(capacity)
        if self.min_samples is None:
            self.min_samples = max(int(round(self.sample_rate_hz * self.min_duration_seconds)), 1)

    @classmethod
    def esp32_default(cls) -> "BreathingRateEstimator":
        """Return ESP32 defaults: 56 subcarriers, 100 Hz, 30 seconds."""

        return cls(sample_rate_hz=100.0, window_seconds=30.0, n_subcarriers=56)

    def update(self, residuals: ArrayLike, weights: ArrayLike | None = None) -> VitalEstimate:
        """Append one residual frame and return the latest breathing estimate."""

        sample = extract_breathing_residual(np.asarray(residuals, dtype=np.float64)[: self.n_subcarriers], weights)
        self._buffer.append(float(sample))
        return self.estimate()

    def estimate(self) -> VitalEstimate:
        """Estimate breathing rate from the current residual buffer."""

        estimate = estimate_rate_from_samples(
            self._buffer.values(),
            sample_rate_hz=self.sample_rate_hz,
            band_hz=BREATHING_BAND_HZ,
            min_duration_seconds=self.min_duration_seconds,
            min_samples=int(self.min_samples or 1),
        )
        return self.smoother.smooth_estimate(estimate) if self.smoother is not None else estimate

    def reset(self) -> None:
        self._buffer.clear()
        if self.smoother is not None:
            self.smoother.clear()

    @property
    def history_len(self) -> int:
        return len(self._buffer)

    @property
    def band_hz(self) -> tuple[float, float]:
        return BREATHING_BAND_HZ

    @property
    def samples(self) -> NDArray[np.float64]:
        return self._buffer.values()


BreathingExtractor = BreathingRateEstimator


def _weighted_mean(
    values: NDArray[np.float64],
    weights: ArrayLike | None,
    *,
    axis: int | None = None,
) -> NDArray[np.float64] | np.float64:
    if weights is None:
        return np.mean(values, axis=axis)
    raw_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if raw_weights.size == 0:
        return np.mean(values, axis=axis)
    n = min(values.shape[-1], raw_weights.size)
    clipped_weights = raw_weights[:n].copy()
    clipped_weights[~np.isfinite(clipped_weights)] = 0.0
    weight_sum = float(np.sum(np.abs(clipped_weights)))
    if weight_sum <= 1e-12:
        return np.mean(values[..., :n], axis=axis)
    normalized = clipped_weights / weight_sum
    return np.sum(values[..., :n] * normalized, axis=-1)
