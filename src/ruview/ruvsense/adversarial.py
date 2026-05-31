"""Physical-impossibility checks for adversarial CSI signal anomalies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


class AdversarialError(ValueError):
    """Base error for adversarial signal checks."""


class Severity(str, Enum):
    """Severity labels for physical plausibility findings."""

    OK = "ok"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PhysicalAnomalyType(str, Enum):
    """Types of physical-impossibility anomalies."""

    NON_FINITE = "non_finite"
    AMPLITUDE_JUMP = "amplitude_jump"
    PHASE_VELOCITY = "phase_velocity"
    COHERENCE_CONFLICT = "coherence_conflict"


@dataclass(frozen=True)
class PhysicalCheckConfig:
    """Thresholds for frame-to-frame physical plausibility checks."""

    max_relative_amplitude_jump: float = 4.0
    max_absolute_amplitude_jump: float = 5.0
    max_phase_velocity_rad_s: float = 80.0
    min_coherence: float = 0.20
    max_coherence_conflict_fraction: float = 0.25
    min_amplitude_for_conflict: float = 1e-6

    def __post_init__(self) -> None:
        for name in (
            "max_relative_amplitude_jump",
            "max_absolute_amplitude_jump",
            "max_phase_velocity_rad_s",
            "min_coherence",
            "max_coherence_conflict_fraction",
            "min_amplitude_for_conflict",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise AdversarialError(f"{name} must be a finite non-negative value")
        if self.max_relative_amplitude_jump <= 0.0:
            raise AdversarialError("max_relative_amplitude_jump must be positive")
        if self.max_phase_velocity_rad_s <= 0.0:
            raise AdversarialError("max_phase_velocity_rad_s must be positive")
        if not 0.0 <= self.min_coherence <= 1.0:
            raise AdversarialError("min_coherence must be in [0, 1]")
        if not 0.0 <= self.max_coherence_conflict_fraction <= 1.0:
            raise AdversarialError("max_coherence_conflict_fraction must be in [0, 1]")


@dataclass(frozen=True)
class AnomalyFinding:
    """One failed physical plausibility check."""

    kind: PhysicalAnomalyType
    severity: Severity
    score: float
    message: str
    indices: tuple[int, ...]
    value: float


@dataclass(frozen=True)
class PhysicalPlausibilityResult:
    """Aggregated physical plausibility result for one CSI feature frame."""

    valid: bool
    anomaly_detected: bool
    severity: Severity
    score: float
    findings: tuple[AnomalyFinding, ...]
    timestamp_us: int


class PhysicalImpossibilityChecker:
    """Checks NaNs, amplitude discontinuities, phase speed, and coherence conflicts."""

    def __init__(self, config: PhysicalCheckConfig | None = None) -> None:
        self.config = config if config is not None else PhysicalCheckConfig()
        self._previous_amplitude: FloatArray | None = None
        self._previous_phase: FloatArray | None = None
        self._previous_timestamp_us: int | None = None
        self._frames_checked = 0
        self._anomalies = 0

    @property
    def frames_checked(self) -> int:
        return self._frames_checked

    @property
    def anomaly_count(self) -> int:
        return self._anomalies

    def reset(self) -> None:
        self._previous_amplitude = None
        self._previous_phase = None
        self._previous_timestamp_us = None
        self._frames_checked = 0
        self._anomalies = 0

    def check(
        self,
        amplitude: ArrayLike,
        phase: ArrayLike | None = None,
        coherence: ArrayLike | None = None,
        *,
        timestamp_us: int = 0,
    ) -> PhysicalPlausibilityResult:
        """Check one CSI-derived feature frame for impossible physics."""

        amp, phase_arr, coherence_arr = _coerce_frame(amplitude, phase, coherence)
        timestamp_us = int(timestamp_us)
        self._frames_checked += 1

        findings: list[AnomalyFinding] = []
        if not np.all(np.isfinite(amp)):
            findings.append(_finding(PhysicalAnomalyType.NON_FINITE, Severity.CRITICAL, 1.0, "amplitude contains NaN or infinity", amp))
        if phase_arr is not None and not np.all(np.isfinite(phase_arr)):
            findings.append(_finding(PhysicalAnomalyType.NON_FINITE, Severity.CRITICAL, 1.0, "phase contains NaN or infinity", phase_arr))
        if coherence_arr is not None and not np.all(np.isfinite(coherence_arr)):
            findings.append(_finding(PhysicalAnomalyType.NON_FINITE, Severity.CRITICAL, 1.0, "coherence contains NaN or infinity", coherence_arr))

        if findings:
            self._anomalies += 1
            return _result(findings, timestamp_us)

        if self._previous_amplitude is not None and self._previous_amplitude.shape == amp.shape:
            findings.extend(self._check_amplitude_jump(amp))

        if (
            phase_arr is not None
            and self._previous_phase is not None
            and self._previous_phase.shape == phase_arr.shape
            and self._previous_timestamp_us is not None
            and timestamp_us > self._previous_timestamp_us
        ):
            findings.extend(self._check_phase_velocity(phase_arr, timestamp_us))

        if coherence_arr is not None:
            findings.extend(self._check_coherence_conflict(amp, coherence_arr))

        self._previous_amplitude = np.array(amp, dtype=np.float64, copy=True)
        self._previous_phase = None if phase_arr is None else np.array(phase_arr, dtype=np.float64, copy=True)
        self._previous_timestamp_us = timestamp_us

        if findings:
            self._anomalies += 1
        return _result(findings, timestamp_us)

    def _check_amplitude_jump(self, amp: FloatArray) -> list[AnomalyFinding]:
        previous = self._previous_amplitude
        if previous is None:
            return []
        diff = np.abs(amp - previous)
        relative = diff / np.maximum(np.abs(previous), self.config.min_amplitude_for_conflict)
        max_rel = float(np.max(relative))
        max_abs = float(np.max(diff))
        if max_rel <= self.config.max_relative_amplitude_jump and max_abs <= self.config.max_absolute_amplitude_jump:
            return []
        severity = Severity.HIGH if max_rel >= 2.0 * self.config.max_relative_amplitude_jump else Severity.MEDIUM
        score = max(
            max_rel / self.config.max_relative_amplitude_jump,
            max_abs / max(self.config.max_absolute_amplitude_jump, 1e-12),
        )
        return [
            AnomalyFinding(
                kind=PhysicalAnomalyType.AMPLITUDE_JUMP,
                severity=severity,
                score=float(np.clip(score, 0.0, 1.0)),
                message="frame-to-frame amplitude jump exceeds physical threshold",
                indices=_top_indices(relative),
                value=max(max_rel, max_abs),
            )
        ]

    def _check_phase_velocity(self, phase: FloatArray, timestamp_us: int) -> list[AnomalyFinding]:
        previous = self._previous_phase
        previous_ts = self._previous_timestamp_us
        if previous is None or previous_ts is None:
            return []
        dt_s = (timestamp_us - previous_ts) / 1_000_000.0
        if dt_s <= 0.0:
            return []
        delta = np.angle(np.exp(1j * (phase - previous)))
        velocity = np.abs(delta) / dt_s
        max_velocity = float(np.max(velocity))
        if max_velocity <= self.config.max_phase_velocity_rad_s:
            return []
        severity = Severity.HIGH if max_velocity >= 2.0 * self.config.max_phase_velocity_rad_s else Severity.MEDIUM
        return [
            AnomalyFinding(
                kind=PhysicalAnomalyType.PHASE_VELOCITY,
                severity=severity,
                score=float(np.clip(max_velocity / self.config.max_phase_velocity_rad_s, 0.0, 1.0)),
                message="phase changed faster than the configured physical velocity bound",
                indices=_top_indices(velocity),
                value=max_velocity,
            )
        ]

    def _check_coherence_conflict(self, amp: FloatArray, coherence: FloatArray) -> list[AnomalyFinding]:
        conflict = (amp > self.config.min_amplitude_for_conflict) & (coherence < self.config.min_coherence)
        fraction = float(np.mean(conflict)) if conflict.size else 0.0
        if fraction <= self.config.max_coherence_conflict_fraction:
            return []
        severity = Severity.HIGH if fraction >= 2.0 * self.config.max_coherence_conflict_fraction else Severity.MEDIUM
        score = float(np.clip(fraction / max(self.config.max_coherence_conflict_fraction, 1e-12), 0.0, 1.0))
        return [
            AnomalyFinding(
                kind=PhysicalAnomalyType.COHERENCE_CONFLICT,
                severity=severity,
                score=score,
                message="strong amplitude is paired with implausibly low coherence",
                indices=tuple(int(i) for i in np.flatnonzero(conflict)[:8]),
                value=fraction,
            )
        ]


def _coerce_frame(
    amplitude: ArrayLike,
    phase: ArrayLike | None,
    coherence: ArrayLike | None,
) -> tuple[FloatArray, FloatArray | None, FloatArray | None]:
    raw_amp = np.asarray(amplitude)
    if np.iscomplexobj(raw_amp):
        amp = np.abs(raw_amp).astype(np.float64)
        inferred_phase: FloatArray | None = np.angle(raw_amp).astype(np.float64)
        if phase is not None:
            inferred_phase = _shape_like(phase, amp.shape, "phase")
    else:
        amp = raw_amp.astype(np.float64)
        inferred_phase = None if phase is None else _shape_like(phase, amp.shape, "phase")
    coherence_arr = None if coherence is None else _shape_like(coherence, amp.shape, "coherence")
    return amp, inferred_phase, coherence_arr


def _shape_like(value: ArrayLike, shape: tuple[int, ...], name: str) -> FloatArray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == ():
        return np.full(shape, float(arr), dtype=np.float64)
    if arr.shape != shape:
        raise AdversarialError(f"{name} shape mismatch: expected {shape}, got {arr.shape}")
    return np.array(arr, dtype=np.float64, copy=True)


def _finding(
    kind: PhysicalAnomalyType,
    severity: Severity,
    score: float,
    message: str,
    values: FloatArray,
) -> AnomalyFinding:
    bad = np.flatnonzero(~np.isfinite(values))
    return AnomalyFinding(
        kind=kind,
        severity=severity,
        score=float(np.clip(score, 0.0, 1.0)),
        message=message,
        indices=tuple(int(i) for i in bad[:8]),
        value=float("nan"),
    )


def _result(findings: list[AnomalyFinding], timestamp_us: int) -> PhysicalPlausibilityResult:
    severity = _max_severity((finding.severity for finding in findings), default=Severity.OK)
    score = max((finding.score for finding in findings), default=0.0)
    return PhysicalPlausibilityResult(
        valid=not findings,
        anomaly_detected=bool(findings),
        severity=severity,
        score=float(np.clip(score, 0.0, 1.0)),
        findings=tuple(findings),
        timestamp_us=int(timestamp_us),
    )


def _max_severity(severities, *, default: Severity) -> Severity:
    rank = {
        Severity.OK: 0,
        Severity.LOW: 1,
        Severity.MEDIUM: 2,
        Severity.HIGH: 3,
        Severity.CRITICAL: 4,
    }
    return max(severities, key=lambda item: rank[item], default=default)


def _top_indices(values: FloatArray, limit: int = 8) -> tuple[int, ...]:
    flat = np.ravel(values)
    if flat.size == 0:
        return ()
    count = min(limit, flat.size)
    indices = np.argpartition(flat, -count)[-count:]
    indices = indices[np.argsort(flat[indices])[::-1]]
    return tuple(int(index) for index in indices)


AdversarialConfig = PhysicalCheckConfig
AdversarialDetector = PhysicalImpossibilityChecker
AdversarialResult = PhysicalPlausibilityResult
AnomalyType = PhysicalAnomalyType


__all__ = [
    "AdversarialConfig",
    "AdversarialDetector",
    "AdversarialError",
    "AdversarialResult",
    "AnomalyFinding",
    "AnomalyType",
    "PhysicalAnomalyType",
    "PhysicalCheckConfig",
    "PhysicalImpossibilityChecker",
    "PhysicalPlausibilityResult",
    "Severity",
]
