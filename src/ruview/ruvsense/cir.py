"""Channel impulse response helpers for RuvSense research experiments.

The Rust reference uses an ISTA sparse solver over a sub-DFT sensing matrix.
This Python port keeps the research API light by using active-subcarrier
placement plus an oversampled IFFT, then selecting sparse taps by magnitude.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ruview.core import CsiFrame
from ruview.signal.csi_processor import CsiWindow


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

SPEED_OF_LIGHT_M_S = 299_792_458.0
DEFAULT_BANDWIDTH_HZ = 20e6


class CirError(ValueError):
    """Raised when CIR estimation input or configuration is invalid."""


class WindowFunction(str, Enum):
    """Frequency-domain window used before the IFFT."""

    RECTANGULAR = "rectangular"
    HANN = "hann"
    HAMMING = "hamming"
    BLACKMAN = "blackman"

    @classmethod
    def from_value(cls, value: "WindowFunction | str | None") -> "WindowFunction":
        if value is None:
            return cls.RECTANGULAR
        if isinstance(value, WindowFunction):
            return value
        normalized = str(value).strip().lower().replace("_", "-")
        aliases = {
            "none": cls.RECTANGULAR,
            "rect": cls.RECTANGULAR,
            "rectangular": cls.RECTANGULAR,
            "boxcar": cls.RECTANGULAR,
            "hann": cls.HANN,
            "hanning": cls.HANN,
            "hamming": cls.HAMMING,
            "blackman": cls.BLACKMAN,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            valid = ", ".join(item.value for item in cls)
            raise CirError(f"unknown CIR window {value!r}; expected one of {valid}") from exc


@dataclass(frozen=True)
class CirConfig:
    """Configuration for CSI-to-CIR estimation."""

    bandwidth_hz: float = DEFAULT_BANDWIDTH_HZ
    num_subcarriers: int = 64
    num_active: int = 52
    delay_bins: int | None = None
    window: WindowFunction | str | None = WindowFunction.RECTANGULAR
    tap_threshold_ratio: float = 0.05
    top_k: int | None = None
    noise_floor_multiplier: float | None = 4.0
    minimum_separation_bins: int = 1
    ranging_min_bandwidth_hz: float = 40e6
    dominant_ratio_threshold: float = 0.3

    def __post_init__(self) -> None:
        if self.bandwidth_hz <= 0.0 or not math.isfinite(self.bandwidth_hz):
            raise CirError("bandwidth_hz must be positive and finite")
        if self.num_subcarriers <= 0:
            raise CirError("num_subcarriers must be positive")
        if self.num_active <= 0:
            raise CirError("num_active must be positive")
        if self.num_active > self.num_subcarriers:
            raise CirError("num_active must be less than or equal to num_subcarriers")

        delay_bins = self.delay_bins if self.delay_bins is not None else 3 * self.num_active
        if delay_bins <= 0:
            raise CirError("delay_bins must be positive")
        if delay_bins < self.num_active:
            raise CirError("delay_bins must be at least num_active")
        if self.tap_threshold_ratio < 0.0:
            raise CirError("tap_threshold_ratio must be non-negative")
        if self.top_k is not None and self.top_k <= 0:
            raise CirError("top_k must be positive when provided")
        if self.noise_floor_multiplier is not None and self.noise_floor_multiplier < 0.0:
            raise CirError("noise_floor_multiplier must be non-negative when provided")
        if self.minimum_separation_bins < 0:
            raise CirError("minimum_separation_bins must be non-negative")
        if self.ranging_min_bandwidth_hz < 0.0:
            raise CirError("ranging_min_bandwidth_hz must be non-negative")
        if not 0.0 <= self.dominant_ratio_threshold <= 1.0:
            raise CirError("dominant_ratio_threshold must be in [0, 1]")

        object.__setattr__(self, "delay_bins", int(delay_bins))
        object.__setattr__(self, "num_subcarriers", int(self.num_subcarriers))
        object.__setattr__(self, "num_active", int(self.num_active))
        object.__setattr__(self, "window", WindowFunction.from_value(self.window))

    @classmethod
    def ht20(cls, **kwargs: Any) -> "CirConfig":
        return cls(bandwidth_hz=20e6, num_subcarriers=64, num_active=52, **kwargs)

    @classmethod
    def ht40(cls, **kwargs: Any) -> "CirConfig":
        return cls(bandwidth_hz=40e6, num_subcarriers=128, num_active=114, **kwargs)

    @classmethod
    def he20(cls, **kwargs: Any) -> "CirConfig":
        return cls(bandwidth_hz=20e6, num_subcarriers=256, num_active=242, **kwargs)

    @classmethod
    def he40(cls, **kwargs: Any) -> "CirConfig":
        return cls(bandwidth_hz=40e6, num_subcarriers=512, num_active=484, **kwargs)

    @classmethod
    def custom(
        cls,
        num_subcarriers: int,
        *,
        num_active: int | None = None,
        bandwidth_hz: float = DEFAULT_BANDWIDTH_HZ,
        **kwargs: Any,
    ) -> "CirConfig":
        return cls(
            bandwidth_hz=bandwidth_hz,
            num_subcarriers=num_subcarriers,
            num_active=num_active if num_active is not None else num_subcarriers,
            **kwargs,
        )

    @property
    def subcarrier_spacing_hz(self) -> float:
        return float(self.bandwidth_hz / self.num_subcarriers)

    @property
    def tap_spacing_s(self) -> float:
        return float(1.0 / (self.delay_bins * self.subcarrier_spacing_hz))


@dataclass(frozen=True)
class CirTap:
    """One selected delay-domain CIR tap."""

    bin_index: int
    delay_s: float
    amplitude: float
    phase_rad: float
    power: float
    value: complex

    @property
    def bin(self) -> int:
        return self.bin_index

    @property
    def delay_ns(self) -> float:
        return float(self.delay_s * 1e9)

    @property
    def distance_m(self) -> float:
        return float(self.delay_s * SPEED_OF_LIGHT_M_S)


@dataclass(frozen=True)
class CirEstimate:
    """Estimated delay-domain CIR plus sparse tap diagnostics."""

    taps: ComplexArray
    selected_taps: tuple[CirTap, ...]
    bandwidth_hz: float
    tap_spacing_s: float
    dominant_tap: CirTap | None
    dominant_tap_ratio: float
    active_tap_count: int
    rms_delay_spread_s: float
    ranging_valid: bool
    threshold_amplitude: float
    noise_floor_amplitude: float

    def top_k_taps(self, k: int) -> tuple[CirTap, ...]:
        """Return the largest ``k`` taps sorted by descending amplitude."""

        if k <= 0:
            return ()
        return tuple(
            _tap_from_value(index, value, self.tap_spacing_s)
            for index, value in _sorted_tap_indices(self.taps, top_k=k)
        )

    @property
    def dominant_tap_idx(self) -> int | None:
        return None if self.dominant_tap is None else self.dominant_tap.bin_index

    def dominant_delay_s(self) -> float | None:
        return None if self.dominant_tap is None else self.dominant_tap.delay_s

    def dominant_distance_m(self) -> float | None:
        return None if self.dominant_tap is None else self.dominant_tap.distance_m


def active_subcarrier_indices(num_active: int) -> NDArray[np.int64]:
    """Return signed active subcarrier indices in negative-then-positive order."""

    if num_active <= 0:
        raise CirError("num_active must be positive")
    negative = num_active // 2
    positive = num_active - negative
    return np.concatenate(
        [
            np.arange(-negative, 0, dtype=np.int64),
            np.arange(1, positive + 1, dtype=np.int64),
        ]
    )


def subtract_reference(
    source: CsiWindow | CsiFrame | ArrayLike,
    reference: CsiWindow | CsiFrame | ArrayLike,
    *,
    subcarrier_axis: int = -1,
) -> ComplexArray:
    """Return CSI with a reference vector or window subtracted."""

    current = _coerce_csi_vector(source, subcarrier_axis=subcarrier_axis)
    baseline = _coerce_csi_vector(reference, subcarrier_axis=subcarrier_axis)
    try:
        baseline = np.broadcast_to(baseline, current.shape).astype(np.complex128, copy=False)
    except ValueError as exc:
        raise CirError(
            f"reference shape {baseline.shape} cannot be broadcast to CSI shape {current.shape}"
        ) from exc
    return (current - baseline).astype(np.complex128, copy=False)


def csi_to_cir(
    source: CsiWindow | CsiFrame | ArrayLike,
    config: CirConfig | None = None,
    *,
    reference: CsiWindow | CsiFrame | ArrayLike | None = None,
    active_indices: ArrayLike | None = None,
    subcarrier_axis: int = -1,
) -> ComplexArray:
    """Convert CSI samples to an oversampled delay-domain CIR using an IFFT."""

    vector = _coerce_csi_vector(source, subcarrier_axis=subcarrier_axis)
    if reference is not None:
        vector = subtract_reference(vector, reference, subcarrier_axis=-1)
    if config is None:
        config = CirConfig.custom(num_subcarriers=vector.size, num_active=vector.size)

    frequency_grid, coherent_gain = _frequency_grid(vector, config, active_indices=active_indices)
    if coherent_gain <= 1e-12:
        return np.zeros(config.delay_bins, dtype=np.complex128)
    return (np.fft.ifft(frequency_grid, n=config.delay_bins) * (config.delay_bins / coherent_gain)).astype(
        np.complex128,
        copy=False,
    )


def estimate_cir(
    source: CsiWindow | CsiFrame | ArrayLike,
    config: CirConfig | None = None,
    *,
    reference: CsiWindow | CsiFrame | ArrayLike | None = None,
    active_indices: ArrayLike | None = None,
    subcarrier_axis: int = -1,
) -> CirEstimate:
    """Estimate a CIR and select a sparse set of delay taps."""

    vector = _coerce_csi_vector(source, subcarrier_axis=subcarrier_axis)
    if config is None:
        config = CirConfig.custom(num_subcarriers=vector.size, num_active=vector.size)

    cir = csi_to_cir(
        vector,
        config,
        reference=reference,
        active_indices=active_indices,
        subcarrier_axis=-1,
    )
    magnitudes = np.abs(cir)
    dominant_idx = int(np.argmax(magnitudes)) if magnitudes.size else 0
    dominant_amplitude = float(magnitudes[dominant_idx]) if magnitudes.size else 0.0
    magnitude_sum = float(np.sum(magnitudes))
    dominant_ratio = dominant_amplitude / magnitude_sum if magnitude_sum > 1e-12 else 0.0

    threshold, noise_floor = sparse_tap_threshold(
        cir,
        threshold_ratio=config.tap_threshold_ratio,
        noise_floor_multiplier=config.noise_floor_multiplier,
    )
    selected = select_sparse_taps(
        cir,
        threshold_ratio=config.tap_threshold_ratio,
        top_k=config.top_k,
        noise_floor_multiplier=config.noise_floor_multiplier,
        minimum_separation_bins=config.minimum_separation_bins,
        tap_spacing_s=config.tap_spacing_s,
    )
    dominant = (
        _tap_from_value(dominant_idx, cir[dominant_idx], config.tap_spacing_s)
        if dominant_amplitude > 1e-12
        else None
    )

    active_cutoff = dominant_amplitude * 0.01
    active_count = int(np.count_nonzero(magnitudes >= active_cutoff)) if dominant_amplitude > 0.0 else 0
    rms_delay = rms_delay_spread(cir, tap_spacing_s=config.tap_spacing_s)
    ranging_valid = (
        config.bandwidth_hz >= config.ranging_min_bandwidth_hz
        and dominant_ratio >= config.dominant_ratio_threshold
    )

    return CirEstimate(
        taps=cir,
        selected_taps=selected,
        bandwidth_hz=float(config.bandwidth_hz),
        tap_spacing_s=config.tap_spacing_s,
        dominant_tap=dominant,
        dominant_tap_ratio=float(np.clip(dominant_ratio, 0.0, 1.0)),
        active_tap_count=active_count,
        rms_delay_spread_s=rms_delay,
        ranging_valid=bool(ranging_valid),
        threshold_amplitude=threshold,
        noise_floor_amplitude=noise_floor,
    )


def select_sparse_taps(
    cir: ArrayLike,
    *,
    threshold_ratio: float = 0.05,
    min_amplitude: float | None = None,
    top_k: int | None = None,
    noise_floor_multiplier: float | None = None,
    minimum_separation_bins: int = 0,
    tap_spacing_s: float = 1.0,
) -> tuple[CirTap, ...]:
    """Select sparse CIR taps by dominant-relative and optional absolute thresholds."""

    if threshold_ratio < 0.0:
        raise CirError("threshold_ratio must be non-negative")
    if min_amplitude is not None and min_amplitude < 0.0:
        raise CirError("min_amplitude must be non-negative when provided")
    if top_k is not None and top_k <= 0:
        raise CirError("top_k must be positive when provided")
    if minimum_separation_bins < 0:
        raise CirError("minimum_separation_bins must be non-negative")

    taps = np.asarray(cir, dtype=np.complex128).ravel()
    if taps.size == 0:
        return ()
    threshold, _ = sparse_tap_threshold(
        taps,
        threshold_ratio=threshold_ratio,
        min_amplitude=min_amplitude,
        noise_floor_multiplier=noise_floor_multiplier,
    )
    indices = _sorted_tap_indices(
        taps,
        min_amplitude=threshold,
        top_k=top_k,
        minimum_separation_bins=minimum_separation_bins,
    )
    return tuple(_tap_from_value(index, taps[index], tap_spacing_s) for index in indices)


def sparse_tap_threshold(
    cir: ArrayLike,
    *,
    threshold_ratio: float = 0.05,
    min_amplitude: float | None = None,
    noise_floor_multiplier: float | None = None,
) -> tuple[float, float]:
    """Return ``(threshold_amplitude, noise_floor_amplitude)`` for tap selection."""

    if threshold_ratio < 0.0:
        raise CirError("threshold_ratio must be non-negative")
    taps = np.asarray(cir, dtype=np.complex128).ravel()
    if taps.size == 0:
        return 0.0, 0.0
    magnitudes = np.abs(taps)
    dominant = float(np.max(magnitudes))
    threshold = dominant * threshold_ratio
    if min_amplitude is not None:
        if min_amplitude < 0.0:
            raise CirError("min_amplitude must be non-negative when provided")
        threshold = max(threshold, float(min_amplitude))

    noise_floor = _robust_noise_floor(magnitudes)
    if noise_floor_multiplier is not None:
        if noise_floor_multiplier < 0.0:
            raise CirError("noise_floor_multiplier must be non-negative when provided")
        threshold = max(threshold, noise_floor * noise_floor_multiplier)
    return float(threshold), float(noise_floor)


def rms_delay_spread(cir: ArrayLike, *, tap_spacing_s: float = 1.0) -> float:
    """Return RMS delay spread in seconds from the power-delay profile."""

    taps = np.asarray(cir, dtype=np.complex128).ravel()
    if taps.size == 0:
        return 0.0
    power = np.square(np.abs(taps), dtype=np.float64)
    total_power = float(np.sum(power))
    if total_power <= 1e-24:
        return 0.0
    delay = np.arange(taps.size, dtype=np.float64) * float(tap_spacing_s)
    mean_delay = float(np.sum(delay * power) / total_power)
    variance = float(np.sum(np.square(delay - mean_delay) * power) / total_power)
    return float(math.sqrt(max(variance, 0.0)))


def phase_variance(source: CsiWindow | CsiFrame | ArrayLike, *, subcarrier_axis: int = -1) -> float:
    """Return variance of wrapped phase angles across the CSI vector."""

    vector = _coerce_csi_vector(source, subcarrier_axis=subcarrier_axis)
    if vector.size < 2:
        return 0.0
    phases = np.angle(vector)
    return float(np.var(phases))


def _coerce_csi_vector(
    source: CsiWindow | CsiFrame | ArrayLike,
    *,
    subcarrier_axis: int = -1,
) -> ComplexArray:
    if isinstance(source, CsiFrame):
        data = source.data
        subcarrier_axis = -1
    elif isinstance(source, CsiWindow):
        data = source.data
        subcarrier_axis = -1
    else:
        data = source

    array = np.asarray(data, dtype=np.complex128)
    if array.size == 0:
        raise CirError("CSI input must not be empty")
    if array.ndim == 0:
        array = array.reshape(1)
    if not np.all(np.isfinite(array.real) & np.isfinite(array.imag)):
        raise CirError("CSI input contains non-finite values")

    axis = int(subcarrier_axis)
    if axis < 0:
        axis += array.ndim
    if axis < 0 or axis >= array.ndim:
        raise CirError(f"subcarrier_axis {subcarrier_axis} is out of bounds for {array.ndim}D input")

    moved = np.moveaxis(array, axis, -1)
    if moved.ndim == 1:
        return moved.astype(np.complex128, copy=True)
    flattened = moved.reshape(-1, moved.shape[-1])
    return np.mean(flattened, axis=0).astype(np.complex128, copy=False)


def _frequency_grid(
    vector: ComplexArray,
    config: CirConfig,
    *,
    active_indices: ArrayLike | None,
) -> tuple[ComplexArray, float]:
    observed = np.asarray(vector, dtype=np.complex128).ravel()
    if observed.size == 0:
        raise CirError("CSI vector must not be empty")
    if observed.size not in {config.num_active, config.num_subcarriers} and active_indices is None:
        raise CirError(
            "CSI vector length must match config.num_active or config.num_subcarriers "
            f"(got {observed.size}, expected {config.num_active} or {config.num_subcarriers})"
        )

    weights = _window_weights(observed.size, config.window)
    windowed = observed * weights
    coherent_gain = float(np.sum(weights))
    grid = np.zeros(config.delay_bins, dtype=np.complex128)

    if active_indices is not None:
        indices = np.asarray(active_indices, dtype=np.int64).ravel()
        if indices.size != observed.size:
            raise CirError(
                f"active_indices length {indices.size} must match CSI vector length {observed.size}"
            )
        grid[np.mod(indices, config.delay_bins)] = windowed
        return grid, coherent_gain

    if observed.size == config.num_active:
        indices = active_subcarrier_indices(config.num_active)
        grid[np.mod(indices, config.delay_bins)] = windowed
        return grid, coherent_gain

    offsets = np.fft.fftfreq(observed.size) * observed.size
    grid[np.mod(offsets.astype(np.int64), config.delay_bins)] = windowed
    return grid, coherent_gain


def _window_weights(length: int, window: WindowFunction) -> FloatArray:
    if length <= 0:
        raise CirError("window length must be positive")
    if window == WindowFunction.RECTANGULAR:
        weights = np.ones(length, dtype=np.float64)
    elif window == WindowFunction.HANN:
        weights = np.hanning(length).astype(np.float64)
    elif window == WindowFunction.HAMMING:
        weights = np.hamming(length).astype(np.float64)
    elif window == WindowFunction.BLACKMAN:
        weights = np.blackman(length).astype(np.float64)
    else:
        raise CirError(f"unsupported window {window!r}")
    if not np.any(weights):
        weights = np.ones(length, dtype=np.float64)
    return weights


def _sorted_tap_indices(
    taps: ComplexArray,
    *,
    min_amplitude: float = 0.0,
    top_k: int | None = None,
    minimum_separation_bins: int = 0,
) -> list[int]:
    magnitudes = np.abs(np.asarray(taps, dtype=np.complex128).ravel())
    if magnitudes.size == 0:
        return []
    indices = np.flatnonzero(magnitudes >= float(min_amplitude))
    ordered = sorted(indices.tolist(), key=lambda index: (-float(magnitudes[index]), int(index)))
    if minimum_separation_bins > 0:
        selected: list[int] = []
        for index in ordered:
            if all(abs(int(index) - prior) > minimum_separation_bins for prior in selected):
                selected.append(int(index))
        ordered = selected
    if top_k is not None:
        ordered = ordered[:top_k]
    return [int(index) for index in ordered]


def _tap_from_value(index: int, value: complex | np.complexfloating, tap_spacing_s: float) -> CirTap:
    complex_value = complex(value)
    amplitude = abs(complex_value)
    return CirTap(
        bin_index=int(index),
        delay_s=float(index * tap_spacing_s),
        amplitude=float(amplitude),
        phase_rad=float(math.atan2(complex_value.imag, complex_value.real)),
        power=float(amplitude * amplitude),
        value=complex_value,
    )


def _robust_noise_floor(magnitudes: FloatArray) -> float:
    finite = np.asarray(magnitudes, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return median + 1.4826 * mad


__all__ = [
    "CirConfig",
    "CirError",
    "CirEstimate",
    "CirTap",
    "WindowFunction",
    "active_subcarrier_indices",
    "csi_to_cir",
    "estimate_cir",
    "phase_variance",
    "rms_delay_spread",
    "select_sparse_taps",
    "sparse_tap_threshold",
    "subtract_reference",
]
