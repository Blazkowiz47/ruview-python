"""LO phase alignment helpers for complex CSI vectors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class PhaseAlignmentConfig:
    """Configuration for common LO phase offset estimation."""

    min_magnitude: float = 1e-12
    min_inner_product_quality: float = 1e-9

    def __post_init__(self) -> None:
        if self.min_magnitude <= 0.0:
            raise ValueError("min_magnitude must be positive")
        if not 0.0 <= self.min_inner_product_quality <= 1.0:
            raise ValueError("min_inner_product_quality must be in [0, 1]")


@dataclass(frozen=True)
class PhaseOffsetEstimate:
    """Estimated common phase rotation from a target CSI vector to a reference."""

    offset_radians: float
    quality: float
    method: str
    sample_count: int


@dataclass(frozen=True)
class PhaseAlignmentResult:
    """Target CSI vector after removing an estimated common LO phase offset."""

    aligned_csi: ComplexArray
    offset_radians: float
    quality: float
    method: str
    sample_count: int

    @property
    def aligned(self) -> ComplexArray:
        """Alias for callers that prefer concise result names."""

        return self.aligned_csi


def wrap_phase(phase: ArrayLike | float) -> FloatArray | float:
    """Wrap phase angle(s) to the principal ``[-pi, pi]`` interval."""

    phase_array = np.asarray(phase, dtype=np.float64)
    wrapped = np.angle(np.exp(1j * phase_array)).astype(np.float64, copy=False)
    if np.isscalar(phase):
        return float(wrapped)
    return wrapped


def circular_mean(phases: ArrayLike, weights: ArrayLike | None = None) -> tuple[float, float]:
    """Return ``(mean_angle, resultant_length)`` for wrapped phase samples."""

    phase_array = np.asarray(phases, dtype=np.float64).ravel()
    if phase_array.size == 0:
        return 0.0, 0.0

    valid = np.isfinite(phase_array)
    if weights is None:
        weight_array = np.ones_like(phase_array, dtype=np.float64)
    else:
        weight_array = np.asarray(weights, dtype=np.float64).ravel()
        if weight_array.shape != phase_array.shape:
            raise ValueError("weights must match the phase sample shape")
        valid &= np.isfinite(weight_array) & (weight_array > 0.0)

    phase_array = phase_array[valid]
    weight_array = weight_array[valid]
    if phase_array.size == 0:
        return 0.0, 0.0

    phasors = weight_array * np.exp(1j * phase_array)
    resultant = np.sum(phasors)
    weight_sum = float(np.sum(weight_array))
    if weight_sum <= 0.0:
        return 0.0, 0.0

    quality = min(1.0, max(0.0, float(np.abs(resultant) / weight_sum)))
    return float(np.angle(resultant)), quality


def estimate_lo_phase_offset(
    reference: ArrayLike,
    target: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    config: PhaseAlignmentConfig | None = None,
) -> PhaseOffsetEstimate:
    """Estimate the common phase rotation between two complex CSI vectors.

    The primary estimator uses the angle of the conjugate inner product,
    ``vdot(reference, target)``. If the vectors have too little energy or the
    normalized inner-product coherence is unusable, the estimator falls back to
    a circular mean of per-subcarrier phase differences.
    """

    cfg = config or PhaseAlignmentConfig()
    ref = np.asarray(reference, dtype=np.complex128).ravel()
    tgt = np.asarray(target, dtype=np.complex128).ravel()
    if ref.shape != tgt.shape:
        raise ValueError("reference and target CSI vectors must have the same length")
    if ref.size == 0:
        raise ValueError("phase offset estimation requires at least one CSI sample")

    valid = np.isfinite(ref.real) & np.isfinite(ref.imag) & np.isfinite(tgt.real) & np.isfinite(tgt.imag)
    valid &= (np.abs(ref) > cfg.min_magnitude) & (np.abs(tgt) > cfg.min_magnitude)
    if weights is not None:
        weight_array = np.asarray(weights, dtype=np.float64).ravel()
        if weight_array.shape != ref.shape:
            raise ValueError("weights must match the CSI vector shape")
        valid &= np.isfinite(weight_array) & (weight_array > 0.0)
    else:
        weight_array = None

    sample_count = int(np.count_nonzero(valid))
    if sample_count == 0:
        return PhaseOffsetEstimate(0.0, 0.0, "empty", 0)

    ref_valid = ref[valid]
    tgt_valid = tgt[valid]
    if weight_array is None:
        weighted_ref = ref_valid
        weighted_tgt = tgt_valid
    else:
        sample_weights = np.sqrt(weight_array[valid])
        weighted_ref = ref_valid * sample_weights
        weighted_tgt = tgt_valid * sample_weights

    inner = np.vdot(weighted_ref, weighted_tgt)
    denom = float(np.linalg.norm(weighted_ref) * np.linalg.norm(weighted_tgt))
    if denom > cfg.min_magnitude and np.isfinite(inner.real) and np.isfinite(inner.imag):
        quality = min(1.0, max(0.0, float(np.abs(inner) / denom)))
        if quality >= cfg.min_inner_product_quality:
            return PhaseOffsetEstimate(float(np.angle(inner)), quality, "inner_product", sample_count)

    phase_delta = np.angle(tgt_valid) - np.angle(ref_valid)
    if weight_array is None:
        circular_weights = np.abs(ref_valid) * np.abs(tgt_valid)
    else:
        circular_weights = weight_array[valid]
    offset, quality = circular_mean(phase_delta, circular_weights)
    return PhaseOffsetEstimate(offset, quality, "circular_mean", sample_count)


def apply_lo_phase_offset(csi: ArrayLike, offset_radians: float) -> ComplexArray:
    """Remove a common LO phase offset from complex CSI samples."""

    csi_array = np.asarray(csi, dtype=np.complex128)
    return (csi_array * np.exp(-1j * float(offset_radians))).astype(np.complex128, copy=False)


def align_lo_phase(
    reference: ArrayLike,
    target: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    config: PhaseAlignmentConfig | None = None,
) -> PhaseAlignmentResult:
    """Estimate and remove the target vector's common LO phase offset."""

    estimate = estimate_lo_phase_offset(reference, target, weights=weights, config=config)
    aligned = apply_lo_phase_offset(target, estimate.offset_radians)
    return PhaseAlignmentResult(
        aligned_csi=aligned,
        offset_radians=estimate.offset_radians,
        quality=estimate.quality,
        method=estimate.method,
        sample_count=estimate.sample_count,
    )


def estimate_common_lo_phase_offset(
    reference: ArrayLike,
    target: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    config: PhaseAlignmentConfig | None = None,
) -> PhaseOffsetEstimate:
    """Compatibility alias for :func:`estimate_lo_phase_offset`."""

    return estimate_lo_phase_offset(reference, target, weights=weights, config=config)


def estimate_phase_offset(
    reference: ArrayLike,
    target: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    config: PhaseAlignmentConfig | None = None,
) -> PhaseOffsetEstimate:
    """Compatibility alias for :func:`estimate_lo_phase_offset`."""

    return estimate_lo_phase_offset(reference, target, weights=weights, config=config)


def apply_phase_offset(csi: ArrayLike, offset_radians: float) -> ComplexArray:
    """Compatibility alias for :func:`apply_lo_phase_offset`."""

    return apply_lo_phase_offset(csi, offset_radians)


def align_phase(
    reference: ArrayLike,
    target: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    config: PhaseAlignmentConfig | None = None,
) -> PhaseAlignmentResult:
    """Compatibility alias for :func:`align_lo_phase`."""

    return align_lo_phase(reference, target, weights=weights, config=config)


PhaseAlignConfig = PhaseAlignmentConfig
PhaseAlignResult = PhaseAlignmentResult
PhaseEstimate = PhaseOffsetEstimate
apply_phase_correction = apply_lo_phase_offset
estimate_common_offset = estimate_lo_phase_offset


__all__ = [
    "PhaseAlignConfig",
    "PhaseAlignResult",
    "PhaseAlignmentConfig",
    "PhaseAlignmentResult",
    "PhaseEstimate",
    "PhaseOffsetEstimate",
    "align_lo_phase",
    "align_phase",
    "apply_lo_phase_offset",
    "apply_phase_offset",
    "apply_phase_correction",
    "circular_mean",
    "estimate_common_lo_phase_offset",
    "estimate_common_offset",
    "estimate_lo_phase_offset",
    "estimate_phase_offset",
    "wrap_phase",
]
