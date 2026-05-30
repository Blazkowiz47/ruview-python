"""Componentized motion scoring for CSI windows."""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import ArrayLike

from ruview.signal.baseline import (
    BaselineStats,
    RollingBaseline,
    measure_baseline,
    relative_increase,
)
from ruview.signal.csi_processor import CsiWindow


@dataclass(frozen=True)
class MotionWeights:
    """Relative contribution of each motion score component."""

    temporal_delta: float = 0.35
    amplitude_variance: float = 0.30
    phase_variance: float = 0.20
    subcarrier_variance: float = 0.15

    def total_weight(self) -> float:
        return (
            self.temporal_delta
            + self.amplitude_variance
            + self.phase_variance
            + self.subcarrier_variance
        )


@dataclass(frozen=True)
class MotionScore:
    """Motion score with raw CSI measurements and normalized components."""

    total: float
    temporal_delta_component: float
    amplitude_variance_component: float
    phase_variance_component: float
    subcarrier_variance_component: float
    temporal_delta: float
    amplitude_variance: float
    phase_variance: float
    subcarrier_variance: float
    baseline_ready: bool

    def is_motion_detected(self, threshold: float = 0.45) -> bool:
        """Return whether the total score meets ``threshold``."""

        return self.total >= threshold


def calculate_motion_score(
    source: CsiWindow | ArrayLike,
    phase: ArrayLike | None = None,
    *,
    baseline: RollingBaseline | BaselineStats | None = None,
    weights: MotionWeights = MotionWeights(),
    time_axis: int = 0,
    subcarrier_axis: int = -1,
) -> MotionScore:
    """Return a weighted motion score for a CSI temporal window."""

    stats = measure_baseline(
        source,
        phase,
        time_axis=time_axis,
        subcarrier_axis=subcarrier_axis,
    )
    reference = _baseline_stats(baseline)

    if reference is None:
        temporal_component = _absolute_component(stats.temporal_delta, scale=0.04)
        amplitude_component = _absolute_component(stats.amplitude_variance, scale=0.03)
        phase_component = _absolute_component(stats.phase_variance, scale=0.004)
        subcarrier_component = _absolute_component(stats.subcarrier_variance, scale=0.015)
    else:
        temporal_component = relative_increase(
            stats.temporal_delta,
            reference.temporal_delta,
            relative_scale=1.0,
            absolute_scale=0.03,
        )
        amplitude_component = relative_increase(
            stats.amplitude_variance,
            reference.amplitude_variance,
            relative_scale=1.0,
            absolute_scale=0.02,
        )
        phase_component = relative_increase(
            stats.phase_variance,
            reference.phase_variance,
            relative_scale=1.0,
            absolute_scale=0.003,
        )
        subcarrier_component = relative_increase(
            stats.subcarrier_variance,
            reference.subcarrier_variance,
            relative_scale=1.0,
            absolute_scale=0.01,
        )

    total_weight = weights.total_weight()
    if total_weight <= 0.0:
        raise ValueError("motion weights must have positive total weight")

    total = (
        weights.temporal_delta * temporal_component
        + weights.amplitude_variance * amplitude_component
        + weights.phase_variance * phase_component
        + weights.subcarrier_variance * subcarrier_component
    ) / total_weight

    return MotionScore(
        total=_clamp01(total),
        temporal_delta_component=temporal_component,
        amplitude_variance_component=amplitude_component,
        phase_variance_component=phase_component,
        subcarrier_variance_component=subcarrier_component,
        temporal_delta=stats.temporal_delta,
        amplitude_variance=stats.amplitude_variance,
        phase_variance=stats.phase_variance,
        subcarrier_variance=stats.subcarrier_variance,
        baseline_ready=reference is not None,
    )


def _baseline_stats(baseline: RollingBaseline | BaselineStats | None) -> BaselineStats | None:
    if baseline is None:
        return None
    if isinstance(baseline, RollingBaseline):
        return baseline.stats if baseline.is_ready else None
    return baseline


def _absolute_component(value: float, *, scale: float) -> float:
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    return _clamp01(value / scale)


def _clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)
