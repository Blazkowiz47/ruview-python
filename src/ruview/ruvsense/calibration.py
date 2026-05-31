"""Empty-room CSI calibration and baseline drift scoring.

This module ports the calibration core from the Rust RuvSense reference into a
NumPy-first Python research API. JSON save/load helpers write a stable Python
research format, not the Rust ADR-135 little-endian binary ABI.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ruview.core import CsiFrame
from ruview.signal.csi_processor import (
    CsiWindow,
    amplitude_phase_to_complex,
    complex_to_amplitude_phase,
)


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

BASELINE_JSON_MAGIC = "ruview-python.ruvsense.baseline-calibration"
BASELINE_JSON_VERSION = 1
DEFAULT_MIN_AMPLITUDE_STD = 1e-6


class CalibrationError(ValueError):
    """Base error for calibration operations."""


class InsufficientFramesError(CalibrationError):
    """Raised when finalization is requested before enough frames were recorded."""

    def __init__(self, got: int, need: int) -> None:
        self.got = int(got)
        self.need = int(need)
        super().__init__(f"insufficient frames: have {self.got}, need {self.need}")


class SubcarrierMismatchError(CalibrationError):
    """Raised when an input does not match the baseline subcarrier layout."""

    def __init__(self, expected: int, got: int) -> None:
        self.expected = int(expected)
        self.got = int(got)
        super().__init__(f"subcarrier count mismatch: expected {self.expected}, got {self.got}")


class PhyTier(str, Enum):
    """802.11 PHY tier labels used to identify subcarrier layouts."""

    HT20 = "ht20"
    HT40 = "ht40"
    HE20 = "he20"
    HE40 = "he40"
    CUSTOM = "custom"

    @classmethod
    def from_value(cls, value: "PhyTier | str") -> "PhyTier":
        if isinstance(value, PhyTier):
            return value
        normalized = str(value).strip().lower().replace("-", "").replace("_", "")
        aliases = {
            "ht20": cls.HT20,
            "80211nht20": cls.HT20,
            "ht40": cls.HT40,
            "80211nht40": cls.HT40,
            "he20": cls.HE20,
            "80211axhe20": cls.HE20,
            "he40": cls.HE40,
            "80211axhe40": cls.HE40,
            "custom": cls.CUSTOM,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            valid = ", ".join(tier.value for tier in cls)
            raise CalibrationError(f"unknown PHY tier {value!r}; expected one of {valid}") from exc

    @property
    def total_subcarriers(self) -> int | None:
        return {
            PhyTier.HT20: 64,
            PhyTier.HT40: 128,
            PhyTier.HE20: 256,
            PhyTier.HE40: 512,
            PhyTier.CUSTOM: None,
        }[self]

    @property
    def active_subcarriers(self) -> int | None:
        return {
            PhyTier.HT20: 52,
            PhyTier.HT40: 114,
            PhyTier.HE20: 242,
            PhyTier.HE40: 484,
            PhyTier.CUSTOM: None,
        }[self]


@dataclass(frozen=True)
class CalibrationConfig:
    """Configuration for empty-room calibration capture."""

    tier: PhyTier | str = PhyTier.HT20
    num_subcarriers: int | None = None
    num_active: int | None = None
    min_frames: int = 600
    max_phase_dispersion: float = 0.3
    motion_amplitude_z_threshold: float = 2.0
    motion_phase_threshold: float = math.pi / 6.0
    drift_amplitude_z_threshold: float = 4.0
    drift_phase_threshold: float = math.pi / 3.0
    min_amplitude_std: float = DEFAULT_MIN_AMPLITUDE_STD

    def __post_init__(self) -> None:
        tier = PhyTier.from_value(self.tier)
        total = self.num_subcarriers if self.num_subcarriers is not None else tier.total_subcarriers
        active = self.num_active if self.num_active is not None else tier.active_subcarriers

        if total is None and active is None:
            raise CalibrationError("custom calibration requires num_subcarriers or num_active")
        if total is None:
            total = active
        if active is None:
            active = total
        if total is None or active is None:
            raise CalibrationError("calibration subcarrier counts could not be resolved")
        if total <= 0:
            raise CalibrationError("num_subcarriers must be positive")
        if active <= 0:
            raise CalibrationError("num_active must be positive")
        if active > total:
            raise CalibrationError("num_active must be less than or equal to num_subcarriers")
        if self.min_frames <= 0:
            raise CalibrationError("min_frames must be positive")
        if not 0.0 <= self.max_phase_dispersion <= 1.0:
            raise CalibrationError("max_phase_dispersion must be in [0, 1]")
        if self.motion_amplitude_z_threshold <= 0.0:
            raise CalibrationError("motion_amplitude_z_threshold must be positive")
        if self.motion_phase_threshold <= 0.0:
            raise CalibrationError("motion_phase_threshold must be positive")
        if self.drift_amplitude_z_threshold <= 0.0:
            raise CalibrationError("drift_amplitude_z_threshold must be positive")
        if self.drift_phase_threshold <= 0.0:
            raise CalibrationError("drift_phase_threshold must be positive")
        if self.min_amplitude_std <= 0.0:
            raise CalibrationError("min_amplitude_std must be positive")

        object.__setattr__(self, "tier", tier)
        object.__setattr__(self, "num_subcarriers", int(total))
        object.__setattr__(self, "num_active", int(active))

    @classmethod
    def ht20(cls, **kwargs: Any) -> "CalibrationConfig":
        return cls(tier=PhyTier.HT20, **kwargs)

    @classmethod
    def ht40(cls, **kwargs: Any) -> "CalibrationConfig":
        return cls(tier=PhyTier.HT40, **kwargs)

    @classmethod
    def he20(cls, **kwargs: Any) -> "CalibrationConfig":
        return cls(tier=PhyTier.HE20, **kwargs)

    @classmethod
    def he40(cls, **kwargs: Any) -> "CalibrationConfig":
        return cls(tier=PhyTier.HE40, **kwargs)

    @classmethod
    def custom(
        cls,
        num_subcarriers: int,
        *,
        num_active: int | None = None,
        **kwargs: Any,
    ) -> "CalibrationConfig":
        return cls(
            tier=PhyTier.CUSTOM,
            num_subcarriers=num_subcarriers,
            num_active=num_active if num_active is not None else num_subcarriers,
            **kwargs,
        )


@dataclass
class WelfordStats:
    """Numerically stable running mean and variance accumulator."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> "WelfordStats":
        value = float(value)
        if not math.isfinite(value):
            raise CalibrationError(f"WelfordStats cannot update with non-finite value {value!r}")
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        return self

    def update_many(self, values: ArrayLike) -> "WelfordStats":
        for value in np.asarray(values, dtype=np.float64).ravel():
            self.update(float(value))
        return self

    @property
    def variance(self) -> float:
        """Bessel-corrected sample variance."""

        if self.count < 2:
            return 0.0
        return float(self.m2 / (self.count - 1))

    @property
    def population_variance(self) -> float:
        if self.count == 0:
            return 0.0
        return float(self.m2 / self.count)

    @property
    def std(self) -> float:
        return float(math.sqrt(self.variance))

    def z_score(self, value: float, *, min_std: float = DEFAULT_MIN_AMPLITUDE_STD) -> float:
        std = max(self.std, min_std)
        return float((float(value) - self.mean) / std)


@dataclass(frozen=True)
class SubcarrierBaselineStats:
    """Final per-subcarrier amplitude and circular-phase statistics."""

    amplitude_mean: float
    amplitude_variance: float
    phase_mean: float
    phase_dispersion: float
    sample_count: int

    @property
    def amp_mean(self) -> float:
        return self.amplitude_mean

    @property
    def amp_variance(self) -> float:
        return self.amplitude_variance

    @property
    def amplitude_std(self) -> float:
        return float(math.sqrt(max(self.amplitude_variance, 0.0)))

    def to_dict(self) -> dict[str, float | int]:
        return {
            "amplitude_mean": float(self.amplitude_mean),
            "amplitude_variance": float(self.amplitude_variance),
            "phase_mean": float(self.phase_mean),
            "phase_dispersion": float(self.phase_dispersion),
            "sample_count": int(self.sample_count),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SubcarrierBaselineStats":
        return cls(
            amplitude_mean=float(payload["amplitude_mean"]),
            amplitude_variance=float(payload["amplitude_variance"]),
            phase_mean=float(payload["phase_mean"]),
            phase_dispersion=float(payload["phase_dispersion"]),
            sample_count=int(payload["sample_count"]),
        )


@dataclass
class _SubcarrierAccumulator:
    amplitude: WelfordStats = field(default_factory=WelfordStats)
    phase_sin_sum: float = 0.0
    phase_cos_sum: float = 0.0
    phase_count: int = 0

    def update(self, amplitude: float, phase: float) -> None:
        self.amplitude.update(float(amplitude))
        self.phase_sin_sum += math.sin(float(phase))
        self.phase_cos_sum += math.cos(float(phase))
        self.phase_count += 1

    def update_many(self, amplitudes: FloatArray, phases: FloatArray) -> None:
        for amplitude, phase in zip(amplitudes.ravel(), phases.ravel(), strict=True):
            self.update(float(amplitude), float(phase))

    @property
    def phase_mean(self) -> float:
        if self.phase_count == 0:
            return 0.0
        return float(math.atan2(self.phase_sin_sum, self.phase_cos_sum))

    @property
    def phase_dispersion(self) -> float:
        if self.phase_count == 0:
            return 1.0
        resultant = math.hypot(self.phase_sin_sum, self.phase_cos_sum) / self.phase_count
        return float(1.0 - min(resultant, 1.0))

    def finalize(self) -> SubcarrierBaselineStats:
        return SubcarrierBaselineStats(
            amplitude_mean=float(self.amplitude.mean),
            amplitude_variance=float(self.amplitude.variance),
            phase_mean=self.phase_mean,
            phase_dispersion=self.phase_dispersion,
            sample_count=int(self.amplitude.count),
        )


@dataclass(frozen=True)
class CalibrationDeviationScore:
    """Deviation metrics for one frame or window against a finalized baseline."""

    amplitude_z_median: float
    amplitude_z_max: float
    phase_drift_median: float
    motion_flagged: bool
    drift_flagged: bool
    sample_count: int = 0

    @property
    def is_empty_like(self) -> bool:
        return not self.motion_flagged and not self.drift_flagged


@dataclass(frozen=True)
class DriftDecision:
    """Decision result for a baseline drift trigger."""

    recalibration_recommended: bool
    drift_flagged: bool
    motion_flagged: bool
    reason: str
    deviation: CalibrationDeviationScore


@dataclass(frozen=True)
class BaselineCalibration:
    """Finalized empty-room baseline used for runtime subtraction and scoring."""

    tier: PhyTier | str
    captured_at_unix_s: int
    frame_count: int
    subcarriers: tuple[SubcarrierBaselineStats, ...]
    num_subcarriers: int | None = None
    motion_amplitude_z_threshold: float = 2.0
    motion_phase_threshold: float = math.pi / 6.0
    drift_amplitude_z_threshold: float = 4.0
    drift_phase_threshold: float = math.pi / 3.0
    min_amplitude_std: float = DEFAULT_MIN_AMPLITUDE_STD

    def __post_init__(self) -> None:
        tier = PhyTier.from_value(self.tier)
        subcarriers = tuple(self.subcarriers)
        if not subcarriers:
            raise CalibrationError("BaselineCalibration requires at least one subcarrier")

        total = self.num_subcarriers if self.num_subcarriers is not None else tier.total_subcarriers
        if total is None:
            total = len(subcarriers)
        if total < len(subcarriers):
            raise CalibrationError("num_subcarriers cannot be smaller than active subcarrier count")
        if self.frame_count < 0:
            raise CalibrationError("frame_count must be non-negative")
        if self.min_amplitude_std <= 0.0:
            raise CalibrationError("min_amplitude_std must be positive")

        object.__setattr__(self, "tier", tier)
        object.__setattr__(self, "subcarriers", subcarriers)
        object.__setattr__(self, "num_subcarriers", int(total))

    @property
    def num_active(self) -> int:
        return len(self.subcarriers)

    @property
    def amplitude_mean(self) -> FloatArray:
        return np.asarray([item.amplitude_mean for item in self.subcarriers], dtype=np.float64)

    @property
    def amplitude_variance(self) -> FloatArray:
        return np.asarray([item.amplitude_variance for item in self.subcarriers], dtype=np.float64)

    @property
    def phase_mean(self) -> FloatArray:
        return np.asarray([item.phase_mean for item in self.subcarriers], dtype=np.float64)

    @property
    def phase_dispersion(self) -> FloatArray:
        return np.asarray([item.phase_dispersion for item in self.subcarriers], dtype=np.float64)

    def deviation(
        self,
        source: CsiWindow | CsiFrame | ArrayLike,
        phase: ArrayLike | None = None,
        *,
        subcarrier_axis: int = -1,
        motion_amplitude_z_threshold: float | None = None,
        motion_phase_threshold: float | None = None,
        drift_amplitude_z_threshold: float | None = None,
        drift_phase_threshold: float | None = None,
    ) -> CalibrationDeviationScore:
        """Score amplitude and phase deviation from this baseline."""

        amplitude, phase_array, _ = _coerce_amplitude_phase(source, phase)
        amplitude, phase_array = _active_views(
            amplitude,
            phase_array,
            expected_active=self.num_active,
            expected_total=self.num_subcarriers,
            subcarrier_axis=subcarrier_axis,
        )

        std = np.sqrt(np.maximum(self.amplitude_variance, self.min_amplitude_std**2))
        amplitude_z_abs = np.abs((amplitude - self.amplitude_mean) / std)
        phase_drift = circular_distance(phase_array, self.phase_mean)

        amplitude_z_median = _finite_median(amplitude_z_abs)
        amplitude_z_max = _finite_max(amplitude_z_abs)
        phase_drift_median = _finite_median(phase_drift)

        motion_amp = (
            self.motion_amplitude_z_threshold
            if motion_amplitude_z_threshold is None
            else motion_amplitude_z_threshold
        )
        motion_phase = self.motion_phase_threshold if motion_phase_threshold is None else motion_phase_threshold
        drift_amp = (
            self.drift_amplitude_z_threshold
            if drift_amplitude_z_threshold is None
            else drift_amplitude_z_threshold
        )
        drift_phase = self.drift_phase_threshold if drift_phase_threshold is None else drift_phase_threshold

        motion_flagged = amplitude_z_median > motion_amp or phase_drift_median > motion_phase
        drift_flagged = amplitude_z_median > drift_amp or phase_drift_median > drift_phase
        return CalibrationDeviationScore(
            amplitude_z_median=amplitude_z_median,
            amplitude_z_max=amplitude_z_max,
            phase_drift_median=phase_drift_median,
            motion_flagged=bool(motion_flagged),
            drift_flagged=bool(drift_flagged),
            sample_count=int(amplitude.size),
        )

    def drift_decision(
        self,
        source: CsiWindow | CsiFrame | ArrayLike,
        phase: ArrayLike | None = None,
        *,
        subcarrier_axis: int = -1,
    ) -> DriftDecision:
        """Return a recalibration-oriented decision for the latest observation."""

        score = self.deviation(source, phase, subcarrier_axis=subcarrier_axis)
        if score.drift_flagged:
            reason = "drift threshold exceeded"
        elif score.motion_flagged:
            reason = "motion or person-like deviation detected"
        else:
            reason = "within empty-room baseline"
        return DriftDecision(
            recalibration_recommended=score.drift_flagged,
            drift_flagged=score.drift_flagged,
            motion_flagged=score.motion_flagged,
            reason=reason,
            deviation=score,
        )

    def subtract_baseline(
        self,
        source: CsiWindow | CsiFrame | ArrayLike,
        phase: ArrayLike | None = None,
        *,
        subcarrier_axis: int = -1,
        clip: bool = True,
    ) -> FloatArray | ComplexArray:
        """Return a baseline-subtracted copy without mutating the input."""

        amplitude, phase_array, has_phase = _coerce_amplitude_phase(source, phase)
        residual = _subtract_amplitude_baseline(
            amplitude,
            self.amplitude_mean,
            expected_active=self.num_active,
            expected_total=self.num_subcarriers,
            subcarrier_axis=subcarrier_axis,
            clip=clip,
        )
        if not has_phase:
            return residual
        return amplitude_phase_to_complex(residual, phase_array)

    def reference_csi_vector(self) -> ComplexArray:
        """Return ``amplitude_mean * exp(1j * phase_mean)`` per active subcarrier."""

        return amplitude_phase_to_complex(self.amplitude_mean, self.phase_mean)

    def to_dict(self) -> dict[str, Any]:
        return {
            "magic": BASELINE_JSON_MAGIC,
            "version": BASELINE_JSON_VERSION,
            "format_note": "Python research JSON format; not the Rust ADR-135 binary ABI.",
            "tier": self.tier.value,
            "captured_at_unix_s": int(self.captured_at_unix_s),
            "frame_count": int(self.frame_count),
            "num_subcarriers": int(self.num_subcarriers),
            "num_active": int(self.num_active),
            "thresholds": {
                "motion_amplitude_z": float(self.motion_amplitude_z_threshold),
                "motion_phase": float(self.motion_phase_threshold),
                "drift_amplitude_z": float(self.drift_amplitude_z_threshold),
                "drift_phase": float(self.drift_phase_threshold),
                "min_amplitude_std": float(self.min_amplitude_std),
            },
            "subcarriers": [item.to_dict() for item in self.subcarriers],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BaselineCalibration":
        if payload.get("magic") != BASELINE_JSON_MAGIC:
            raise CalibrationError(f"invalid baseline magic {payload.get('magic')!r}")
        if int(payload.get("version", -1)) != BASELINE_JSON_VERSION:
            raise CalibrationError(f"unsupported baseline JSON version {payload.get('version')!r}")
        thresholds = payload.get("thresholds", {})
        return cls(
            tier=PhyTier.from_value(payload["tier"]),
            captured_at_unix_s=int(payload["captured_at_unix_s"]),
            frame_count=int(payload["frame_count"]),
            num_subcarriers=int(payload["num_subcarriers"]),
            subcarriers=tuple(
                SubcarrierBaselineStats.from_dict(item) for item in payload["subcarriers"]
            ),
            motion_amplitude_z_threshold=float(
                thresholds.get("motion_amplitude_z", cls.motion_amplitude_z_threshold)
            ),
            motion_phase_threshold=float(thresholds.get("motion_phase", cls.motion_phase_threshold)),
            drift_amplitude_z_threshold=float(
                thresholds.get("drift_amplitude_z", cls.drift_amplitude_z_threshold)
            ),
            drift_phase_threshold=float(thresholds.get("drift_phase", cls.drift_phase_threshold)),
            min_amplitude_std=float(thresholds.get("min_amplitude_std", DEFAULT_MIN_AMPLITUDE_STD)),
        )

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load_json(cls, path: str | Path) -> "BaselineCalibration":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class CalibrationRecorder:
    """Accumulate empty-room frames/windows into a finalized baseline."""

    config: CalibrationConfig = field(default_factory=CalibrationConfig)
    captured_at_unix_s: int = field(default_factory=lambda: int(time.time()))
    frame_count: int = 0
    _stats: list[_SubcarrierAccumulator] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.config, CalibrationConfig):
            self.config = CalibrationConfig(tier=self.config)
        self._stats = [_SubcarrierAccumulator() for _ in range(self.config.num_active)]

    @property
    def frames_recorded(self) -> int:
        return self.frame_count

    def record(
        self,
        source: CsiWindow | CsiFrame | ArrayLike,
        phase: ArrayLike | None = None,
        *,
        subcarrier_axis: int = -1,
        frame_axis: int | None = None,
    ) -> CalibrationDeviationScore:
        """Record empty-room CSI data and return a partial-baseline deviation score."""

        amplitude, phase_array, _ = _coerce_amplitude_phase(source, phase)
        active_amplitude, active_phase = _active_views(
            amplitude,
            phase_array,
            expected_active=self.config.num_active,
            expected_total=self.config.num_subcarriers,
            subcarrier_axis=subcarrier_axis,
        )
        flat_amplitude = active_amplitude.reshape(-1, self.config.num_active)
        flat_phase = active_phase.reshape(-1, self.config.num_active)
        for subcarrier_index, stats in enumerate(self._stats):
            stats.update_many(flat_amplitude[:, subcarrier_index], flat_phase[:, subcarrier_index])

        self.frame_count += _recorded_frame_count(source, amplitude, frame_axis=frame_axis)
        return self._partial_baseline().deviation(
            source,
            phase,
            subcarrier_axis=subcarrier_axis,
            motion_amplitude_z_threshold=self.config.motion_amplitude_z_threshold,
            motion_phase_threshold=self.config.motion_phase_threshold,
            drift_amplitude_z_threshold=self.config.drift_amplitude_z_threshold,
            drift_phase_threshold=self.config.drift_phase_threshold,
        )

    def finalize(self) -> BaselineCalibration:
        if self.frame_count < self.config.min_frames:
            raise InsufficientFramesError(self.frame_count, self.config.min_frames)
        return self._partial_baseline()

    def _partial_baseline(self) -> BaselineCalibration:
        return BaselineCalibration(
            tier=self.config.tier,
            captured_at_unix_s=self.captured_at_unix_s,
            frame_count=self.frame_count,
            num_subcarriers=self.config.num_subcarriers,
            subcarriers=tuple(stats.finalize() for stats in self._stats),
            motion_amplitude_z_threshold=self.config.motion_amplitude_z_threshold,
            motion_phase_threshold=self.config.motion_phase_threshold,
            drift_amplitude_z_threshold=self.config.drift_amplitude_z_threshold,
            drift_phase_threshold=self.config.drift_phase_threshold,
            min_amplitude_std=self.config.min_amplitude_std,
        )


def subtract_baseline(
    source: CsiWindow | CsiFrame | ArrayLike,
    baseline: BaselineCalibration,
    phase: ArrayLike | None = None,
    *,
    subcarrier_axis: int = -1,
    clip: bool = True,
) -> FloatArray | ComplexArray:
    """Return ``source`` with the amplitude baseline removed."""

    return baseline.subtract_baseline(
        source,
        phase,
        subcarrier_axis=subcarrier_axis,
        clip=clip,
    )


def save_baseline(baseline: BaselineCalibration, path: str | Path) -> Path:
    """Save a baseline in the Python research JSON format."""

    return baseline.save_json(path)


def load_baseline(path: str | Path) -> BaselineCalibration:
    """Load a baseline saved by :func:`save_baseline`."""

    return BaselineCalibration.load_json(path)


def circular_distance(a: ArrayLike, b: ArrayLike) -> FloatArray:
    """Unsigned circular distance in radians, wrapped to ``[0, pi]``."""

    delta = np.abs(np.angle(np.exp(1j * (np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))))
    return np.asarray(delta, dtype=np.float64)


def _coerce_amplitude_phase(
    source: CsiWindow | CsiFrame | ArrayLike,
    phase: ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray, bool]:
    if phase is not None:
        amplitude = np.asarray(source, dtype=np.float64)
        phase_array = np.asarray(phase, dtype=np.float64)
        if amplitude.shape != phase_array.shape:
            raise CalibrationError(
                f"amplitude and phase shapes must match, got {amplitude.shape} and {phase_array.shape}"
            )
        return amplitude, phase_array, True
    if isinstance(source, CsiWindow):
        return (
            np.asarray(source.amplitude, dtype=np.float64),
            np.asarray(source.phase, dtype=np.float64),
            True,
        )
    if isinstance(source, CsiFrame):
        return (
            np.asarray(source.amplitude, dtype=np.float64),
            np.asarray(source.phase, dtype=np.float64),
            True,
        )

    array = np.asarray(source)
    if np.iscomplexobj(array):
        amplitude, phase_array = complex_to_amplitude_phase(array)
        return amplitude, phase_array, True

    amplitude = np.asarray(array, dtype=np.float64)
    phase_array = np.zeros_like(amplitude, dtype=np.float64)
    return amplitude, phase_array, False


def _active_views(
    amplitude: FloatArray,
    phase: FloatArray,
    *,
    expected_active: int,
    expected_total: int | None,
    subcarrier_axis: int,
) -> tuple[FloatArray, FloatArray]:
    axis = _normalize_axis(subcarrier_axis, amplitude.ndim)
    amplitude_last = np.moveaxis(amplitude, axis, -1)
    phase_last = np.moveaxis(phase, axis, -1)
    got = int(amplitude_last.shape[-1])
    if got == expected_active:
        return amplitude_last, phase_last
    if expected_total is not None and got == expected_total and expected_active <= got:
        return amplitude_last[..., :expected_active], phase_last[..., :expected_active]
    raise SubcarrierMismatchError(expected_active, got)


def _subtract_amplitude_baseline(
    amplitude: FloatArray,
    baseline_mean: FloatArray,
    *,
    expected_active: int,
    expected_total: int | None,
    subcarrier_axis: int,
    clip: bool,
) -> FloatArray:
    axis = _normalize_axis(subcarrier_axis, amplitude.ndim)
    moved = np.moveaxis(np.asarray(amplitude, dtype=np.float64), axis, -1)
    got = int(moved.shape[-1])
    residual = np.array(moved, dtype=np.float64, copy=True)
    if got == expected_active:
        residual = residual - baseline_mean
    elif expected_total is not None and got == expected_total and expected_active <= got:
        residual[..., :expected_active] = residual[..., :expected_active] - baseline_mean
    else:
        raise SubcarrierMismatchError(expected_active, got)
    if clip:
        residual = np.maximum(residual, 0.0)
    return np.moveaxis(residual, -1, axis)


def _recorded_frame_count(
    source: CsiWindow | CsiFrame | ArrayLike,
    amplitude: FloatArray,
    *,
    frame_axis: int | None,
) -> int:
    if isinstance(source, CsiWindow):
        return len(source)
    if isinstance(source, CsiFrame):
        return 1
    if frame_axis is not None:
        axis = _normalize_axis(frame_axis, amplitude.ndim)
        return int(amplitude.shape[axis])
    if amplitude.ndim >= 3:
        return int(amplitude.shape[0])
    if amplitude.ndim == 2 and not np.iscomplexobj(np.asarray(source)):
        return int(amplitude.shape[0])
    return 1


def _normalize_axis(axis: int, ndim: int) -> int:
    if ndim == 0:
        raise CalibrationError("CSI input must have at least one dimension")
    if not -ndim <= axis < ndim:
        raise CalibrationError(f"axis {axis} is out of bounds for array of dimension {ndim}")
    return axis % ndim


def _finite_median(values: ArrayLike) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return 0.0
    return float(np.median(finite))


def _finite_max(values: ArrayLike) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return 0.0
    return float(np.max(finite))
