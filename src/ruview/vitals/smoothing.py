"""Bounded sample and BPM smoothing buffers for vital estimators."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ruview.vitals.quality import VitalEstimate, VitalStatus

SmoothingMode = Literal["ema", "mean", "median"]


@dataclass
class SampleBuffer:
    """Bounded FIFO buffer for scalar residual samples."""

    maxlen: int
    _values: deque[float] = field(default_factory=deque, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.maxlen <= 0:
            raise ValueError("maxlen must be positive")

    def append(self, value: float) -> None:
        if np.isfinite(value):
            self._values.append(float(value))
            while len(self._values) > self.maxlen:
                self._values.popleft()

    def extend(self, values: ArrayLike) -> None:
        for value in np.asarray(values, dtype=np.float64).reshape(-1):
            self.append(float(value))

    def clear(self) -> None:
        self._values.clear()

    @property
    def fill_fraction(self) -> float:
        return min(len(self._values) / self.maxlen, 1.0)

    def values(self) -> NDArray[np.float64]:
        return np.fromiter(self._values, dtype=np.float64, count=len(self._values))

    def __len__(self) -> int:
        return len(self._values)


@dataclass
class BpmSmoothingBuffer:
    """Smooth available BPM estimates with EMA, bounded mean, or median."""

    mode: SmoothingMode = "median"
    window: int = 5
    alpha: float = 0.35
    _values: deque[float] = field(default_factory=deque, init=False, repr=False)
    _ema: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in {"ema", "mean", "median"}:
            raise ValueError("mode must be 'ema', 'mean', or 'median'")
        if self.window <= 0:
            raise ValueError("window must be positive")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")

    def update(self, value_bpm: float) -> float:
        """Append one BPM value and return the smoothed value."""

        if not np.isfinite(value_bpm):
            return float(self._ema if self._ema is not None else 0.0)

        value = float(value_bpm)
        if self.mode == "ema":
            self._ema = value if self._ema is None else self.alpha * value + (1.0 - self.alpha) * self._ema
            return float(self._ema)

        self._values.append(value)
        while len(self._values) > self.window:
            self._values.popleft()
        arr = np.fromiter(self._values, dtype=np.float64, count=len(self._values))
        if self.mode == "mean":
            return float(np.mean(arr))
        return float(np.median(arr))

    def smooth_estimate(self, estimate: VitalEstimate) -> VitalEstimate:
        """Return ``estimate`` with a smoothed BPM while preserving diagnostics."""

        if estimate.status is VitalStatus.UNAVAILABLE:
            return estimate
        smoothed = self.update(estimate.value_bpm)
        return VitalEstimate(
            value_bpm=smoothed,
            confidence=estimate.confidence,
            status=estimate.status,
            quality=estimate.quality,
            peak_hz=estimate.peak_hz,
            peak_prominence=estimate.peak_prominence,
            in_band_energy_ratio=estimate.in_band_energy_ratio,
            duration_seconds=estimate.duration_seconds,
            sample_count=estimate.sample_count,
        )

    def clear(self) -> None:
        self._values.clear()
        self._ema = None

    @property
    def values(self) -> tuple[float, ...]:
        return tuple(self._values)

    def __len__(self) -> int:
        return len(self._values) if self.mode != "ema" else int(self._ema is not None)
