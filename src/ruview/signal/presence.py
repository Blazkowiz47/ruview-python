"""Human presence and motion-state classification for CSI windows."""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import ArrayLike

from ruview.signal.baseline import (
    BaselineStats,
    DetectionDebouncer,
    RollingBaseline,
    measure_baseline,
    relative_increase,
)
from ruview.signal.csi_processor import CsiWindow
from ruview.signal.motion import MotionScore, calculate_motion_score


@dataclass(frozen=True)
class PresenceResult:
    """Presence classifier output for one CSI window."""

    present: bool
    raw_present: bool
    moving: bool
    state: str
    confidence: float
    presence_score: float
    motion_score: MotionScore
    threshold: float
    motion_threshold: float
    baseline_ready: bool


def classify_presence(
    source: CsiWindow | ArrayLike,
    phase: ArrayLike | None = None,
    *,
    baseline: RollingBaseline | BaselineStats | None = None,
    presence_threshold: float = 0.35,
    motion_threshold: float = 0.45,
    debouncer: DetectionDebouncer | None = None,
    update_baseline_when_empty: bool = False,
    time_axis: int = 0,
    subcarrier_axis: int = -1,
) -> PresenceResult:
    """Classify a CSI window as empty, still-present, or moving-present."""

    if not 0.0 <= presence_threshold <= 1.0:
        raise ValueError("presence_threshold must be in [0, 1]")
    if not 0.0 <= motion_threshold <= 1.0:
        raise ValueError("motion_threshold must be in [0, 1]")

    stats = measure_baseline(
        source,
        phase,
        time_axis=time_axis,
        subcarrier_axis=subcarrier_axis,
    )
    reference = _baseline_stats(baseline)
    motion = calculate_motion_score(
        source,
        phase,
        baseline=reference,
        time_axis=time_axis,
        subcarrier_axis=subcarrier_axis,
    )

    mean_component = _mean_amplitude_component(stats, reference)
    amplitude_component = (
        relative_increase(
            stats.amplitude_variance,
            reference.amplitude_variance,
            relative_scale=0.8,
            absolute_scale=0.02,
        )
        if reference is not None
        else _absolute_component(stats.amplitude_variance, scale=0.03)
    )

    presence_score = _clamp01(
        0.50 * mean_component
        + 0.25 * amplitude_component
        + 0.15 * motion.total
        + 0.10 * motion.subcarrier_variance_component
    )
    raw_present = presence_score >= presence_threshold
    present = (
        debouncer.update(raw_present, confidence=presence_score)
        if debouncer is not None
        else raw_present
    )
    moving = bool(present and motion.is_motion_detected(motion_threshold))
    state = "empty" if not present else "moving" if moving else "still"

    if (
        update_baseline_when_empty
        and isinstance(baseline, RollingBaseline)
        and not raw_present
    ):
        baseline.update(source, phase, time_axis=time_axis, subcarrier_axis=subcarrier_axis)

    return PresenceResult(
        present=present,
        raw_present=raw_present,
        moving=moving,
        state=state,
        confidence=presence_score,
        presence_score=presence_score,
        motion_score=motion,
        threshold=presence_threshold,
        motion_threshold=motion_threshold,
        baseline_ready=reference is not None,
    )


def _baseline_stats(baseline: RollingBaseline | BaselineStats | None) -> BaselineStats | None:
    if baseline is None:
        return None
    if isinstance(baseline, RollingBaseline):
        return baseline.stats if baseline.is_ready else None
    return baseline


def _mean_amplitude_component(
    stats: BaselineStats,
    reference: BaselineStats | None,
) -> float:
    if reference is None:
        return _absolute_component(stats.mean_amplitude - 1.05, scale=0.25)
    return relative_increase(
        stats.mean_amplitude,
        reference.mean_amplitude,
        relative_scale=0.15,
        absolute_scale=0.20,
    )


def _absolute_component(value: float, *, scale: float) -> float:
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    return _clamp01(value / scale)


def _clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)
