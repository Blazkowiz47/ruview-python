"""Long-running scalar trend and biomechanics drift summaries."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
from typing import Deque

import numpy as np


DAY_US = 86_400_000_000


class LongitudinalError(ValueError):
    """Base error for longitudinal monitoring operations."""


class BiomechanicsMetric(str, Enum):
    """Scalar RuvSense biomechanics metrics tracked over time."""

    GAIT_SYMMETRY = "gait_symmetry"
    STABILITY_INDEX = "stability_index"
    BREATHING_REGULARITY = "breathing_regularity"
    MICRO_TREMOR = "micro_tremor"
    ACTIVITY_LEVEL = "activity_level"

    @classmethod
    def from_value(cls, value: "BiomechanicsMetric | str") -> "BiomechanicsMetric":
        if isinstance(value, BiomechanicsMetric):
            return value
        normalized = str(value).strip().lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            valid = ", ".join(item.value for item in cls)
            raise LongitudinalError(f"unknown biomechanics metric {value!r}; expected one of {valid}") from exc


class TrendDirection(str, Enum):
    """Direction of a scalar trend."""

    STABLE = "stable"
    INCREASING = "increasing"
    DECREASING = "decreasing"


class DriftSeverity(str, Enum):
    """Severity label for sustained baseline deviation."""

    NONE = "none"
    WATCH = "watch"
    ALERT = "alert"


@dataclass(frozen=True)
class LongitudinalConfig:
    """Configuration for trend buffers and drift detection."""

    window_size: int = 90
    min_baseline_points: int = 7
    min_sustained_points: int = 3
    drift_z_threshold: float = 2.0
    alert_z_threshold: float = 3.0
    stable_slope_per_day: float = 1e-3
    min_std: float = 1e-6

    def __post_init__(self) -> None:
        if self.window_size <= 1:
            raise LongitudinalError("window_size must be greater than 1")
        if self.min_baseline_points <= 1:
            raise LongitudinalError("min_baseline_points must be greater than 1")
        if self.min_sustained_points <= 0:
            raise LongitudinalError("min_sustained_points must be positive")
        for name in ("drift_z_threshold", "alert_z_threshold", "stable_slope_per_day", "min_std"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise LongitudinalError(f"{name} must be a finite positive value")
        if self.alert_z_threshold < self.drift_z_threshold:
            raise LongitudinalError("alert_z_threshold must be >= drift_z_threshold")


@dataclass
class WelfordStats:
    """Numerically stable scalar mean and variance accumulator."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> "WelfordStats":
        value = _finite_float(value, "value")
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        return self

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return float(self.m2 / (self.count - 1))

    @property
    def std(self) -> float:
        return float(math.sqrt(max(self.variance, 0.0)))

    def z_score(self, value: float, *, min_std: float = 1e-6) -> float:
        return float((_finite_float(value, "value") - self.mean) / max(self.std, min_std))


@dataclass(frozen=True)
class TrendSummary:
    """Windowed trend summary for one scalar metric."""

    metric: BiomechanicsMetric
    count: int
    latest_value: float
    mean: float
    std: float
    minimum: float
    maximum: float
    slope_per_day: float
    direction: TrendDirection
    z_score: float


@dataclass(frozen=True)
class DriftReport:
    """Sustained deviation from a personal scalar baseline."""

    metric: BiomechanicsMetric
    direction: TrendDirection
    severity: DriftSeverity
    z_score: float
    current_value: float
    baseline_mean: float
    baseline_std: float
    sustained_points: int
    timestamp_us: int


@dataclass(frozen=True)
class MetricUpdate:
    """Return value for one longitudinal monitor update."""

    metric: BiomechanicsMetric
    summary: TrendSummary
    drift_report: DriftReport | None


@dataclass(frozen=True)
class BiomechanicsSummary:
    """Compact report over all metrics currently tracked by the monitor."""

    summaries: tuple[TrendSummary, ...]
    active_reports: tuple[DriftReport, ...]


class TrendBuffer:
    """Fixed-size scalar time series buffer with least-squares trend summaries."""

    def __init__(self, metric: BiomechanicsMetric | str, max_size: int = 90) -> None:
        if max_size <= 1:
            raise LongitudinalError("max_size must be greater than 1")
        self.metric = BiomechanicsMetric.from_value(metric)
        self._values: Deque[float] = deque(maxlen=int(max_size))
        self._timestamps_us: Deque[int] = deque(maxlen=int(max_size))

    @property
    def count(self) -> int:
        return len(self._values)

    @property
    def values(self) -> tuple[float, ...]:
        return tuple(self._values)

    def add(self, value: float, timestamp_us: int) -> None:
        self._values.append(_finite_float(value, self.metric.value))
        self._timestamps_us.append(int(timestamp_us))

    def summary(
        self,
        *,
        z_score: float = 0.0,
        stable_slope_per_day: float = 1e-3,
    ) -> TrendSummary:
        if not self._values:
            raise LongitudinalError("cannot summarize an empty trend buffer")

        values = np.asarray(self._values, dtype=np.float64)
        slope = _slope_per_day(values, np.asarray(self._timestamps_us, dtype=np.float64))
        if slope > stable_slope_per_day:
            direction = TrendDirection.INCREASING
        elif slope < -stable_slope_per_day:
            direction = TrendDirection.DECREASING
        else:
            direction = TrendDirection.STABLE
        std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        return TrendSummary(
            metric=self.metric,
            count=int(values.size),
            latest_value=float(values[-1]),
            mean=float(np.mean(values)),
            std=std,
            minimum=float(np.min(values)),
            maximum=float(np.max(values)),
            slope_per_day=float(slope),
            direction=direction,
            z_score=float(z_score),
        )


class LongitudinalMonitor:
    """Tracks per-metric trend buffers and sustained baseline drift."""

    def __init__(self, config: LongitudinalConfig | None = None) -> None:
        self.config = config if config is not None else LongitudinalConfig()
        self._buffers: dict[BiomechanicsMetric, TrendBuffer] = {}
        self._baselines: dict[BiomechanicsMetric, WelfordStats] = {}
        self._drift_counts: dict[BiomechanicsMetric, int] = {}
        self._active_reports: dict[BiomechanicsMetric, DriftReport] = {}

    def update(
        self,
        metric: BiomechanicsMetric | str,
        value: float,
        timestamp_us: int,
    ) -> MetricUpdate:
        metric_key = BiomechanicsMetric.from_value(metric)
        value = _finite_float(value, metric_key.value)
        baseline = self._baselines.setdefault(metric_key, WelfordStats())
        buffer = self._buffers.setdefault(
            metric_key,
            TrendBuffer(metric_key, max_size=self.config.window_size),
        )

        ready = baseline.count >= self.config.min_baseline_points
        z = baseline.z_score(value, min_std=self.config.min_std) if ready else 0.0
        is_drift = ready and abs(z) >= self.config.drift_z_threshold
        self._drift_counts[metric_key] = self._drift_counts.get(metric_key, 0) + 1 if is_drift else 0

        report: DriftReport | None = None
        if self._drift_counts[metric_key] >= self.config.min_sustained_points:
            direction = TrendDirection.INCREASING if z > 0.0 else TrendDirection.DECREASING
            severity = DriftSeverity.ALERT if abs(z) >= self.config.alert_z_threshold else DriftSeverity.WATCH
            report = DriftReport(
                metric=metric_key,
                direction=direction,
                severity=severity,
                z_score=float(z),
                current_value=float(value),
                baseline_mean=float(baseline.mean),
                baseline_std=float(baseline.std),
                sustained_points=int(self._drift_counts[metric_key]),
                timestamp_us=int(timestamp_us),
            )
            self._active_reports[metric_key] = report
        elif not is_drift:
            self._active_reports.pop(metric_key, None)

        buffer.add(value, timestamp_us)
        if not is_drift:
            baseline.update(value)

        summary = buffer.summary(
            z_score=z,
            stable_slope_per_day=self.config.stable_slope_per_day,
        )
        return MetricUpdate(metric=metric_key, summary=summary, drift_report=report)

    def baseline_for(self, metric: BiomechanicsMetric | str) -> WelfordStats:
        return self._baselines.setdefault(BiomechanicsMetric.from_value(metric), WelfordStats())

    def drift_points(self, metric: BiomechanicsMetric | str) -> int:
        return self._drift_counts.get(BiomechanicsMetric.from_value(metric), 0)

    def summarize_biomechanics(self) -> BiomechanicsSummary:
        summaries = tuple(
            buffer.summary(stable_slope_per_day=self.config.stable_slope_per_day)
            for _, buffer in sorted(self._buffers.items(), key=lambda item: item[0].value)
        )
        reports = tuple(
            report for _, report in sorted(self._active_reports.items(), key=lambda item: item[0].value)
        )
        return BiomechanicsSummary(summaries=summaries, active_reports=reports)


def _slope_per_day(values: np.ndarray, timestamps_us: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    days = (timestamps_us - timestamps_us[0]) / DAY_US
    if float(np.ptp(days)) <= 1e-12:
        return 0.0
    centered = days - float(np.mean(days))
    denom = float(np.dot(centered, centered))
    if denom <= 1e-18:
        return 0.0
    return float(np.dot(centered, values - float(np.mean(values))) / denom)


def _finite_float(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise LongitudinalError(f"{name} must be finite")
    return result


__all__ = [
    "BiomechanicsMetric",
    "BiomechanicsSummary",
    "DAY_US",
    "DriftReport",
    "DriftSeverity",
    "LongitudinalConfig",
    "LongitudinalError",
    "LongitudinalMonitor",
    "MetricUpdate",
    "TrendBuffer",
    "TrendDirection",
    "TrendSummary",
    "WelfordStats",
]
