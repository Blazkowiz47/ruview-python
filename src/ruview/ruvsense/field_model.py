"""Persistent RF field normal model for RuvSense research experiments.

The model learns an empty-room amplitude baseline for each link and extracts
low-rank environmental modes from the calibration covariance. Runtime frames
are decomposed into baseline, environmental drift, and residual body energy.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


class FieldModelError(ValueError):
    """Base error for field model operations."""


class CalibrationStatus(str, Enum):
    """Lifecycle state for an empty-room field calibration."""

    UNCALIBRATED = "uncalibrated"
    COLLECTING = "collecting"
    FRESH = "fresh"
    STALE = "stale"
    EXPIRED = "expired"


@dataclass(frozen=True)
class FieldModelConfig:
    """Configuration for empty-room baseline learning."""

    n_links: int
    n_subcarriers: int
    n_modes: int = 3
    min_calibration_frames: int = 600
    baseline_expiry_s: float = 86_400.0
    min_mode_energy: float = 1e-12

    def __post_init__(self) -> None:
        if self.n_links <= 0:
            raise FieldModelError("n_links must be positive")
        if self.n_subcarriers <= 0:
            raise FieldModelError("n_subcarriers must be positive")
        if self.n_modes < 0:
            raise FieldModelError("n_modes must be non-negative")
        if self.min_calibration_frames <= 0:
            raise FieldModelError("min_calibration_frames must be positive")
        if self.baseline_expiry_s <= 0.0:
            raise FieldModelError("baseline_expiry_s must be positive")
        if self.min_mode_energy < 0.0:
            raise FieldModelError("min_mode_energy must be non-negative")

        object.__setattr__(self, "n_links", int(self.n_links))
        object.__setattr__(self, "n_subcarriers", int(self.n_subcarriers))
        object.__setattr__(self, "n_modes", int(self.n_modes))
        object.__setattr__(self, "min_calibration_frames", int(self.min_calibration_frames))

    @property
    def n_features(self) -> int:
        return self.n_links * self.n_subcarriers


@dataclass
class LinkBaselineStats:
    """Vector Welford accumulator for one TX-RX link."""

    n_subcarriers: int
    count: int = 0
    mean: FloatArray = field(init=False, repr=False)
    m2: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_subcarriers <= 0:
            raise FieldModelError("n_subcarriers must be positive")
        self.mean = np.zeros(int(self.n_subcarriers), dtype=np.float64)
        self.m2 = np.zeros(int(self.n_subcarriers), dtype=np.float64)

    def update(self, amplitudes: ArrayLike) -> None:
        values = _coerce_link_amplitudes(amplitudes, self.n_subcarriers)
        self.count += 1
        delta = values - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (values - self.mean)

    @property
    def variance(self) -> FloatArray:
        """Bessel-corrected per-subcarrier sample variance."""

        if self.count < 2:
            return np.zeros_like(self.mean)
        return self.m2 / (self.count - 1)

    @property
    def population_variance(self) -> FloatArray:
        if self.count == 0:
            return np.zeros_like(self.mean)
        return self.m2 / self.count


@dataclass(frozen=True)
class FieldNormalMode:
    """Learned empty-room baseline and environmental eigenspace."""

    baseline: FloatArray
    environmental_modes: FloatArray
    mode_energies: FloatArray
    variance_explained: float
    calibrated_at_s: float
    geometry_hash: int = 0
    baseline_eigenvalue_count: int = 0

    @property
    def n_modes(self) -> int:
        return int(self.environmental_modes.shape[0])

    @property
    def n_links(self) -> int:
        return int(self.baseline.shape[0])

    @property
    def n_subcarriers(self) -> int:
        return int(self.baseline.shape[1])


@dataclass(frozen=True)
class BodyPerturbation:
    """Runtime residual after subtracting baseline and environmental modes."""

    residuals: FloatArray
    energies: FloatArray
    total_energy: float
    environmental_projections: FloatArray
    mode_coefficients: FloatArray

    @property
    def body_energy(self) -> float:
        return self.total_energy


class FieldModel:
    """Learn and apply a room-level RF field normal model."""

    def __init__(self, config: FieldModelConfig) -> None:
        self.config = config
        self.link_stats = [LinkBaselineStats(config.n_subcarriers) for _ in range(config.n_links)]
        self.status = CalibrationStatus.UNCALIBRATED
        self.field_mode: FieldNormalMode | None = None
        self._sum_outer = np.zeros((config.n_features, config.n_features), dtype=np.float64)
        self._calibration_count = 0
        self._last_calibration_s = 0.0

    @property
    def calibration_frame_count(self) -> int:
        return self._calibration_count

    @property
    def modes(self) -> FieldNormalMode | None:
        return self.field_mode

    @classmethod
    def fit(
        cls,
        frames: ArrayLike,
        config: FieldModelConfig,
        *,
        calibrated_at_s: float | None = None,
        geometry_hash: int = 0,
    ) -> "FieldModel":
        model = cls(config)
        model.feed_calibration_frames(frames)
        model.finalize_calibration(calibrated_at_s=calibrated_at_s, geometry_hash=geometry_hash)
        return model

    def feed_calibration_frames(self, frames: ArrayLike) -> None:
        """Feed one frame or a stack shaped ``[frames, links, subcarriers]``."""

        values = np.asarray(frames, dtype=np.float64)
        if values.ndim == 2:
            self.feed_calibration(values)
            return
        if values.ndim != 3:
            raise FieldModelError("calibration frames must be 2D or 3D amplitude arrays")
        for frame in values:
            self.feed_calibration(frame)

    def feed_calibration(self, observations: ArrayLike) -> None:
        """Record one empty-room amplitude frame shaped ``[links, subcarriers]``."""

        frame = _coerce_frame(observations, self.config)
        for stats, amplitudes in zip(self.link_stats, frame, strict=True):
            stats.update(amplitudes)

        flattened = frame.reshape(-1)
        self._sum_outer += np.outer(flattened, flattened)
        self._calibration_count += 1
        if self.status == CalibrationStatus.UNCALIBRATED:
            self.status = CalibrationStatus.COLLECTING

    def finalize_calibration(
        self,
        *,
        calibrated_at_s: float | None = None,
        geometry_hash: int = 0,
    ) -> FieldNormalMode:
        """Compute the SVD environmental modes from accumulated calibration data."""

        if self._calibration_count < self.config.min_calibration_frames:
            raise FieldModelError(
                "insufficient calibration frames: "
                f"need {self.config.min_calibration_frames}, got {self._calibration_count}"
            )

        baseline = np.vstack([stats.mean for stats in self.link_stats]).astype(np.float64)
        mean_flat = baseline.reshape(-1)
        covariance = self._covariance(mean_flat)

        n_modes = min(self.config.n_modes, self.config.n_features)
        if n_modes == 0:
            mode_energies = np.zeros(0, dtype=np.float64)
            modes = np.zeros((0, self.config.n_links, self.config.n_subcarriers), dtype=np.float64)
        else:
            _u, singular_values, vh = np.linalg.svd(covariance, full_matrices=False, hermitian=True)
            keep = singular_values[:n_modes] > self.config.min_mode_energy
            mode_energies = singular_values[:n_modes][keep].astype(np.float64, copy=True)
            flat_modes = vh[:n_modes][keep].astype(np.float64, copy=True)
            flat_modes = _normalize_rows(flat_modes)
            modes = flat_modes.reshape((-1, self.config.n_links, self.config.n_subcarriers))

        total_variance = float(np.trace(covariance))
        explained = float(mode_energies.sum() / total_variance) if total_variance > 0.0 else 0.0
        eigen_count = _marcenko_pastur_count(
            mode_energies,
            n_features=self.config.n_features,
            n_samples=self._calibration_count,
        )

        timestamp = float(time.time() if calibrated_at_s is None else calibrated_at_s)
        self.field_mode = FieldNormalMode(
            baseline=baseline,
            environmental_modes=modes,
            mode_energies=mode_energies,
            variance_explained=explained,
            calibrated_at_s=timestamp,
            geometry_hash=int(geometry_hash),
            baseline_eigenvalue_count=eigen_count,
        )
        self._last_calibration_s = timestamp
        self.status = CalibrationStatus.FRESH
        return self.field_mode

    def extract_perturbation(self, observations: ArrayLike) -> BodyPerturbation:
        """Return body residuals after baseline subtraction and mode projection."""

        if self.field_mode is None:
            raise FieldModelError("field model has not been calibrated")

        frame = _coerce_frame(observations, self.config)
        residual_flat = (frame - self.field_mode.baseline).reshape(-1)
        mode_flat = self.field_mode.environmental_modes.reshape((self.field_mode.n_modes, -1))

        if mode_flat.size:
            coefficients = mode_flat @ residual_flat
            environmental_flat = coefficients @ mode_flat
            residual_flat = residual_flat - environmental_flat
        else:
            coefficients = np.zeros(0, dtype=np.float64)
            environmental_flat = np.zeros_like(residual_flat)

        residuals = residual_flat.reshape(self.config.n_links, self.config.n_subcarriers)
        environmental = environmental_flat.reshape(self.config.n_links, self.config.n_subcarriers)
        energies = np.linalg.norm(residuals, axis=1)
        environmental_projections = np.linalg.norm(environmental, axis=1)
        return BodyPerturbation(
            residuals=residuals,
            energies=energies.astype(np.float64),
            total_energy=float(energies.sum()),
            environmental_projections=environmental_projections.astype(np.float64),
            mode_coefficients=coefficients.astype(np.float64),
        )

    def body_energy(self, observations: ArrayLike) -> float:
        return self.extract_perturbation(observations).total_energy

    def subtract_baseline(self, observations: ArrayLike) -> FloatArray:
        if self.field_mode is None:
            raise FieldModelError("field model has not been calibrated")
        return _coerce_frame(observations, self.config) - self.field_mode.baseline

    def check_freshness(self, current_s: float | None = None) -> CalibrationStatus:
        if self.field_mode is None:
            return CalibrationStatus.UNCALIBRATED
        now = float(time.time() if current_s is None else current_s)
        elapsed = max(0.0, now - self._last_calibration_s)
        if elapsed > self.config.baseline_expiry_s:
            return CalibrationStatus.EXPIRED
        if elapsed > self.config.baseline_expiry_s * 0.5:
            return CalibrationStatus.STALE
        return CalibrationStatus.FRESH

    def reset_calibration(self) -> None:
        self.link_stats = [LinkBaselineStats(self.config.n_subcarriers) for _ in range(self.config.n_links)]
        self.field_mode = None
        self._sum_outer.fill(0.0)
        self._calibration_count = 0
        self._last_calibration_s = 0.0
        self.status = CalibrationStatus.UNCALIBRATED

    def _covariance(self, mean_flat: FloatArray) -> FloatArray:
        if self._calibration_count < 2:
            return np.zeros_like(self._sum_outer)
        centered_outer = self._sum_outer - self._calibration_count * np.outer(mean_flat, mean_flat)
        covariance = centered_outer / (self._calibration_count - 1)
        covariance = (covariance + covariance.T) * 0.5
        return np.nan_to_num(covariance, copy=False)


def learn_field_model(
    frames: ArrayLike,
    config: FieldModelConfig,
    *,
    calibrated_at_s: float | None = None,
    geometry_hash: int = 0,
) -> FieldModel:
    """Convenience wrapper around :meth:`FieldModel.fit`."""

    return FieldModel.fit(
        frames,
        config,
        calibrated_at_s=calibrated_at_s,
        geometry_hash=geometry_hash,
    )


def _coerce_link_amplitudes(values: ArrayLike, n_subcarriers: int) -> FloatArray:
    amplitudes = np.asarray(values, dtype=np.float64)
    if amplitudes.shape != (n_subcarriers,):
        raise FieldModelError(f"expected {n_subcarriers} subcarriers, got shape {amplitudes.shape}")
    if not np.all(np.isfinite(amplitudes)):
        raise FieldModelError("amplitudes must be finite")
    return amplitudes


def _coerce_frame(values: ArrayLike, config: FieldModelConfig) -> FloatArray:
    frame = np.asarray(values, dtype=np.float64)
    expected = (config.n_links, config.n_subcarriers)
    if frame.shape != expected:
        raise FieldModelError(f"expected amplitude frame shape {expected}, got {frame.shape}")
    if not np.all(np.isfinite(frame)):
        raise FieldModelError("amplitude frame must contain only finite values")
    return frame


def _normalize_rows(values: FloatArray) -> FloatArray:
    if values.size == 0:
        return values
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-15)


def _marcenko_pastur_count(eigenvalues: FloatArray, *, n_features: int, n_samples: int) -> int:
    positive = np.asarray(eigenvalues, dtype=np.float64)
    positive = positive[positive > 1e-12]
    if positive.size == 0:
        return 0
    sorted_values = np.sort(positive)
    lower_half = sorted_values[: max(1, sorted_values.size // 2)]
    noise = float(np.mean(lower_half))
    ratio = n_features / max(1, n_samples)
    threshold = noise * (1.0 + math.sqrt(ratio)) ** 2
    return int(np.count_nonzero(positive > threshold))


__all__ = [
    "BodyPerturbation",
    "CalibrationStatus",
    "FieldModel",
    "FieldModelConfig",
    "FieldModelError",
    "FieldNormalMode",
    "LinkBaselineStats",
    "learn_field_model",
]
