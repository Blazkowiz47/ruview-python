"""Multi-band CSI vector fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ruview.ruvsense.phase_align import estimate_lo_phase_offset, wrap_phase


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
LengthStrategy = Literal["interpolate", "truncate"]


@dataclass(frozen=True)
class MultiBandFusionConfig:
    """Configuration for aligning and fusing per-band CSI observations."""

    target_length: int | None = None
    length_strategy: LengthStrategy = "interpolate"
    normalize: bool = True
    phase_align: bool = True
    min_norm: float = 1e-12

    def __post_init__(self) -> None:
        if self.target_length is not None and self.target_length <= 0:
            raise ValueError("target_length must be positive")
        if self.length_strategy not in {"interpolate", "truncate"}:
            raise ValueError("length_strategy must be 'interpolate' or 'truncate'")
        if self.min_norm <= 0.0:
            raise ValueError("min_norm must be positive")


@dataclass(frozen=True)
class MultiBandObservation:
    """One CSI vector captured on a frequency band or WiFi channel."""

    csi: ArrayLike
    band_id: str = ""
    frequency_mhz: float | None = None
    weight: float = 1.0
    quality: float = 1.0


@dataclass(frozen=True)
class MultiBandContribution:
    """A band's normalized, length-aligned contribution to the fused vector."""

    band_id: str
    weight: float
    quality: float
    input_length: int
    aligned_length: int
    normalization_scale: float
    phase_offset_radians: float
    contribution_vector: ComplexArray

    @property
    def contribution(self) -> ComplexArray:
        """Alias for the weighted contribution vector."""

        return self.contribution_vector


@dataclass(frozen=True)
class MultiBandFusionResult:
    """Fused multi-band CSI vector and per-band contribution audit data."""

    fused_vector: ComplexArray
    contributions: tuple[MultiBandContribution, ...]
    weights: FloatArray
    coherence: float
    target_length: int

    @property
    def fused(self) -> ComplexArray:
        """Alias for callers that prefer concise result names."""

        return self.fused_vector


BandObservation = MultiBandObservation
BandContribution = MultiBandContribution
MultiBandConfig = MultiBandFusionConfig
MultiBandResult = MultiBandFusionResult


def fuse_multiband(
    observations: Sequence[MultiBandObservation | ArrayLike],
    *,
    weights: Sequence[float] | None = None,
    qualities: Sequence[float] | None = None,
    band_ids: Sequence[str] | None = None,
    frequencies_mhz: Sequence[float | None] | None = None,
    config: MultiBandFusionConfig | None = None,
) -> MultiBandFusionResult:
    """Normalize, align, and fuse multiple per-band complex CSI vectors."""

    cfg = config or MultiBandFusionConfig()
    obs = _coerce_observations(observations, weights, qualities, band_ids, frequencies_mhz)
    if not obs:
        raise ValueError("multi-band fusion requires at least one observation")

    arrays = [np.asarray(item.csi, dtype=np.complex128).ravel() for item in obs]
    if any(array.size == 0 for array in arrays):
        raise ValueError("multi-band observations must not be empty")

    target_length = _resolve_target_length(arrays, cfg)
    aligned = [_align_vector_length(array, target_length, cfg.length_strategy) for array in arrays]
    normalized, scales = zip(*[_normalize_vector(array, cfg) for array in aligned], strict=True)

    phase_offsets = [0.0] * len(normalized)
    phase_aligned = list(normalized)
    if cfg.phase_align and len(phase_aligned) > 1:
        reference = phase_aligned[0]
        for index in range(1, len(phase_aligned)):
            estimate = estimate_lo_phase_offset(reference, phase_aligned[index])
            phase_offsets[index] = estimate.offset_radians
            phase_aligned[index] = phase_aligned[index] * np.exp(-1j * estimate.offset_radians)

    normalized_weights = normalize_weights([item.weight * item.quality for item in obs])
    fused = np.zeros(target_length, dtype=np.complex128)
    contributions: list[MultiBandContribution] = []
    for index, (item, vector, scale, phase_offset, weight) in enumerate(
        zip(obs, phase_aligned, scales, phase_offsets, normalized_weights, strict=True)
    ):
        contribution = (float(weight) * vector).astype(np.complex128, copy=False)
        fused += contribution
        contributions.append(
            MultiBandContribution(
                band_id=item.band_id or f"band-{index}",
                weight=float(weight),
                quality=float(item.quality),
                input_length=int(arrays[index].size),
                aligned_length=int(target_length),
                normalization_scale=float(scale),
                phase_offset_radians=float(phase_offset),
                contribution_vector=contribution,
            )
        )

    return MultiBandFusionResult(
        fused_vector=fused.astype(np.complex128, copy=False),
        contributions=tuple(contributions),
        weights=normalized_weights,
        coherence=_cross_band_coherence(phase_aligned),
        target_length=int(target_length),
    )


def fuse_multi_band(
    observations: Sequence[MultiBandObservation | ArrayLike],
    *,
    weights: Sequence[float] | None = None,
    qualities: Sequence[float] | None = None,
    band_ids: Sequence[str] | None = None,
    frequencies_mhz: Sequence[float | None] | None = None,
    config: MultiBandFusionConfig | None = None,
) -> MultiBandFusionResult:
    """Compatibility alias for :func:`fuse_multiband`."""

    return fuse_multiband(
        observations,
        weights=weights,
        qualities=qualities,
        band_ids=band_ids,
        frequencies_mhz=frequencies_mhz,
        config=config,
    )


def normalize_weights(weights: Sequence[float]) -> FloatArray:
    """Clip negative numerical noise and normalize weights to sum to one."""

    weight_array = np.asarray(weights, dtype=np.float64)
    if weight_array.ndim != 1:
        raise ValueError("weights must be a one-dimensional sequence")
    if weight_array.size == 0:
        raise ValueError("at least one weight is required")
    if not np.all(np.isfinite(weight_array)):
        raise ValueError("weights must be finite")
    if np.any(weight_array < 0.0):
        raise ValueError("weights must be non-negative")

    total = float(np.sum(weight_array))
    if total <= 0.0:
        return np.full(weight_array.shape, 1.0 / weight_array.size, dtype=np.float64)
    return (weight_array / total).astype(np.float64, copy=False)


def align_subcarrier_length(
    csi: ArrayLike,
    target_length: int,
    *,
    strategy: LengthStrategy = "interpolate",
) -> ComplexArray:
    """Align one complex CSI vector to a target subcarrier count."""

    if target_length <= 0:
        raise ValueError("target_length must be positive")
    return _align_vector_length(np.asarray(csi, dtype=np.complex128).ravel(), target_length, strategy)


def _coerce_observations(
    observations: Sequence[MultiBandObservation | ArrayLike],
    weights: Sequence[float] | None,
    qualities: Sequence[float] | None,
    band_ids: Sequence[str] | None,
    frequencies_mhz: Sequence[float | None] | None,
) -> list[MultiBandObservation]:
    obs_list = list(observations)
    count = len(obs_list)
    _validate_optional_length(weights, count, "weights")
    _validate_optional_length(qualities, count, "qualities")
    _validate_optional_length(band_ids, count, "band_ids")
    _validate_optional_length(frequencies_mhz, count, "frequencies_mhz")

    coerced: list[MultiBandObservation] = []
    for index, item in enumerate(obs_list):
        if isinstance(item, MultiBandObservation):
            weight = item.weight if weights is None else float(weights[index])
            quality = item.quality if qualities is None else float(qualities[index])
            band_id = item.band_id if band_ids is None else str(band_ids[index])
            frequency = item.frequency_mhz if frequencies_mhz is None else frequencies_mhz[index]
            csi = item.csi
        else:
            weight = 1.0 if weights is None else float(weights[index])
            quality = 1.0 if qualities is None else float(qualities[index])
            band_id = f"band-{index}" if band_ids is None else str(band_ids[index])
            frequency = None if frequencies_mhz is None else frequencies_mhz[index]
            csi = item

        if weight < 0.0 or not np.isfinite(weight):
            raise ValueError("observation weights must be finite and non-negative")
        if quality < 0.0 or not np.isfinite(quality):
            raise ValueError("observation qualities must be finite and non-negative")
        coerced.append(
            MultiBandObservation(
                csi=csi,
                band_id=band_id,
                frequency_mhz=None if frequency is None else float(frequency),
                weight=float(weight),
                quality=float(quality),
            )
        )
    return coerced


def _validate_optional_length(values: Sequence[object] | None, count: int, name: str) -> None:
    if values is not None and len(values) != count:
        raise ValueError(f"{name} length must match the number of observations")


def _resolve_target_length(arrays: Sequence[ComplexArray], config: MultiBandFusionConfig) -> int:
    if config.target_length is not None:
        return int(config.target_length)
    lengths = [array.size for array in arrays]
    if config.length_strategy == "interpolate":
        return int(max(lengths))
    return int(min(lengths))


def _align_vector_length(csi: ComplexArray, target_length: int, strategy: LengthStrategy) -> ComplexArray:
    if csi.size == target_length:
        return csi.astype(np.complex128, copy=True)
    if strategy == "interpolate":
        if csi.size == 1:
            return np.full(target_length, csi[0], dtype=np.complex128)
        old_x = np.linspace(0.0, 1.0, csi.size)
        new_x = np.linspace(0.0, 1.0, target_length)
        real = np.interp(new_x, old_x, csi.real)
        imag = np.interp(new_x, old_x, csi.imag)
        return (real + 1j * imag).astype(np.complex128, copy=False)
    if strategy == "truncate":
        if csi.size < target_length:
            raise ValueError("truncate strategy cannot expand a shorter CSI vector")
        start = (csi.size - target_length) // 2
        return csi[start : start + target_length].astype(np.complex128, copy=True)
    raise ValueError("length_strategy must be 'interpolate' or 'truncate'")


def _normalize_vector(csi: ComplexArray, config: MultiBandFusionConfig) -> tuple[ComplexArray, float]:
    if not config.normalize:
        return csi.astype(np.complex128, copy=True), 1.0
    scale = float(np.sqrt(np.mean(np.abs(csi) ** 2)))
    if scale <= config.min_norm or not np.isfinite(scale):
        return np.zeros_like(csi, dtype=np.complex128), 0.0
    return (csi / scale).astype(np.complex128, copy=False), scale


def _cross_band_coherence(vectors: Sequence[ComplexArray]) -> float:
    if len(vectors) <= 1:
        return 1.0

    scores: list[float] = []
    for left_index in range(len(vectors)):
        left = vectors[left_index]
        left_norm = float(np.linalg.norm(left))
        for right in vectors[left_index + 1 :]:
            denom = left_norm * float(np.linalg.norm(right))
            if denom <= 1e-12:
                scores.append(0.0)
                continue
            inner = np.vdot(left, right)
            score = float(np.abs(inner) / denom)
            scores.append(min(1.0, max(0.0, score)))
    if not scores:
        return 1.0
    return float(np.mean(scores))


__all__ = [
    "BandObservation",
    "BandContribution",
    "MultiBandConfig",
    "MultiBandContribution",
    "MultiBandFusionConfig",
    "MultiBandFusionResult",
    "MultiBandObservation",
    "MultiBandResult",
    "align_subcarrier_length",
    "fuse_multi_band",
    "fuse_multiband",
    "normalize_weights",
    "wrap_phase",
]
