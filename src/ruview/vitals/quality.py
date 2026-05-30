"""Vital-sign status, confidence, and NumPy-only spectral scoring.

The Rust reference mixes streaming IIR/FIR filters, zero crossings, and
autocorrelation. This Python port intentionally uses compact FFT/PSD helpers
for the first lightweight implementation so core installs only need NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


class VitalStatus(str, Enum):
    """Status of one vital-sign estimate."""

    VALID = "valid"
    DEGRADED = "degraded"
    UNRELIABLE = "unreliable"
    UNAVAILABLE = "unavailable"


class SignalQuality(str, Enum):
    """Human-readable signal-quality label."""

    VALID = "valid"
    DEGRADED = "degraded"
    UNRELIABLE = "unreliable"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class VitalEstimate:
    """A single vital-sign estimate in beats or breaths per minute."""

    value_bpm: float
    confidence: float
    status: VitalStatus
    quality: SignalQuality
    peak_hz: float | None = None
    peak_prominence: float = 0.0
    in_band_energy_ratio: float = 0.0
    duration_seconds: float = 0.0
    sample_count: int = 0

    @classmethod
    def unavailable(
        cls,
        *,
        duration_seconds: float = 0.0,
        sample_count: int = 0,
    ) -> "VitalEstimate":
        """Return an unavailable estimate with Rust-compatible ``0.0`` BPM."""

        return cls(
            value_bpm=0.0,
            confidence=0.0,
            status=VitalStatus.UNAVAILABLE,
            quality=SignalQuality.UNAVAILABLE,
            duration_seconds=float(max(duration_seconds, 0.0)),
            sample_count=max(int(sample_count), 0),
        )

    @property
    def is_available(self) -> bool:
        """Whether this estimate contains a usable BPM value."""

        return self.status is not VitalStatus.UNAVAILABLE


@dataclass(frozen=True)
class VitalReading:
    """Combined respiratory and heart-rate reading."""

    respiratory_rate: VitalEstimate
    heart_rate: VitalEstimate
    subcarrier_count: int
    signal_quality: float
    quality: SignalQuality
    timestamp_secs: float = 0.0


@dataclass(frozen=True)
class SpectralPeak:
    """Diagnostic details for a band-limited spectral peak."""

    frequency_hz: float
    bpm: float
    power: float
    prominence: float
    in_band_energy_ratio: float
    band_power: float
    total_power: float


def frequency_domain_bandpass(
    samples: ArrayLike,
    sample_rate_hz: float,
    low_hz: float,
    high_hz: float,
) -> NDArray[np.float64]:
    """Return ``samples`` with FFT bins outside ``[low_hz, high_hz]`` removed."""

    values = _finite_array(samples)
    if values.size < 3 or sample_rate_hz <= 0.0:
        return values.copy()
    low_hz, high_hz = _ordered_band(low_hz, high_hz)
    nyquist = sample_rate_hz / 2.0
    if low_hz >= nyquist or high_hz <= 0.0 or low_hz >= high_hz:
        return np.zeros_like(values)

    centered = values - float(np.mean(values))
    spectrum = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(centered.size, d=1.0 / sample_rate_hz)
    keep = (freqs >= low_hz) & (freqs <= min(high_hz, nyquist))
    spectrum[~keep] = 0.0
    return np.fft.irfft(spectrum, n=centered.size).astype(np.float64, copy=False)


def estimate_rate_from_samples(
    samples: ArrayLike,
    *,
    sample_rate_hz: float,
    band_hz: tuple[float, float],
    min_duration_seconds: float,
    min_samples: int,
) -> VitalEstimate:
    """Estimate a BPM peak and confidence from a scalar residual time series."""

    values = _finite_array(samples)
    sample_count = int(values.size)
    duration_seconds = sample_count / sample_rate_hz if sample_rate_hz > 0.0 else 0.0
    if (
        sample_rate_hz <= 0.0
        or sample_count < max(int(min_samples), 1)
        or duration_seconds < min_duration_seconds
    ):
        return VitalEstimate.unavailable(
            duration_seconds=duration_seconds,
            sample_count=sample_count,
        )

    centered = values - float(np.mean(values))
    if float(np.var(centered)) < 1e-14:
        return VitalEstimate.unavailable(
            duration_seconds=duration_seconds,
            sample_count=sample_count,
        )

    peak = _spectral_peak(centered, sample_rate_hz=sample_rate_hz, band_hz=band_hz)
    if peak is None:
        return VitalEstimate.unavailable(
            duration_seconds=duration_seconds,
            sample_count=sample_count,
        )

    duration_score = _clamp01(duration_seconds / max(min_duration_seconds, 1e-12))
    prominence_score = _clamp01((np.log2(max(peak.prominence, 1.0)) - 0.25) / 3.0)
    energy_score = _clamp01(peak.in_band_energy_ratio / 0.35)
    band_support = _clamp01(peak.in_band_energy_ratio / 0.25)
    confidence = _clamp01(
        duration_score * (0.65 * prominence_score + 0.35 * energy_score) * band_support
    )
    status = vital_status_from_confidence(confidence)
    quality = signal_quality_from_confidence(confidence)

    return VitalEstimate(
        value_bpm=float(peak.bpm),
        confidence=confidence,
        status=status,
        quality=quality,
        peak_hz=float(peak.frequency_hz),
        peak_prominence=float(peak.prominence),
        in_band_energy_ratio=float(peak.in_band_energy_ratio),
        duration_seconds=float(duration_seconds),
        sample_count=sample_count,
    )


def signal_quality_from_amplitude(
    amplitudes: ArrayLike,
    *,
    history_fill_fraction: float = 1.0,
) -> tuple[float, SignalQuality]:
    """Score instantaneous CSI amplitude quality using coefficient of variation."""

    values = _finite_array(amplitudes)
    if values.size == 0:
        return 0.0, SignalQuality.UNAVAILABLE
    mean = float(np.mean(np.abs(values)))
    if mean <= 1e-12:
        return 0.0, SignalQuality.UNAVAILABLE

    cv = float(np.std(values) / mean)
    if cv < 0.01:
        quality = cv / 0.01 * 0.3
    elif cv < 0.3:
        quality = 0.3 + 0.7 * max(0.0, 1.0 - abs((cv - 0.15) / 0.15))
    else:
        quality = float(np.clip(1.0 - (cv - 0.3) / 0.7, 0.1, 0.5))

    fill = _clamp01(history_fill_fraction)
    score = _clamp01(quality * (0.3 + 0.7 * fill))
    return score, signal_quality_from_confidence(score)


def vital_status_from_confidence(confidence: float) -> VitalStatus:
    """Map a confidence score to a vital status."""

    value = _clamp01(confidence)
    if value >= 0.72:
        return VitalStatus.VALID
    if value >= 0.45:
        return VitalStatus.DEGRADED
    if value >= 0.18:
        return VitalStatus.UNRELIABLE
    return VitalStatus.UNAVAILABLE


def signal_quality_from_confidence(confidence: float) -> SignalQuality:
    """Map a confidence score to a signal-quality label."""

    value = _clamp01(confidence)
    if value >= 0.72:
        return SignalQuality.VALID
    if value >= 0.45:
        return SignalQuality.DEGRADED
    if value >= 0.18:
        return SignalQuality.UNRELIABLE
    return SignalQuality.UNAVAILABLE


def _spectral_peak(
    values: NDArray[np.float64],
    *,
    sample_rate_hz: float,
    band_hz: tuple[float, float],
) -> SpectralPeak | None:
    low_hz, high_hz = _ordered_band(*band_hz)
    nyquist = sample_rate_hz / 2.0
    if low_hz >= high_hz or low_hz >= nyquist:
        return None

    n_fft = _next_power_of_two(max(values.size, 4))
    window = np.hanning(values.size) if values.size > 1 else np.ones(values.size)
    padded = np.zeros(n_fft, dtype=np.float64)
    padded[: values.size] = values * window
    spectrum = np.fft.rfft(padded)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate_hz)
    power = (spectrum.real * spectrum.real + spectrum.imag * spectrum.imag).astype(np.float64)
    if power.size:
        power[0] = 0.0

    total_mask = freqs > 0.0
    total_power = float(np.sum(power[total_mask]))
    band_mask = (freqs >= low_hz) & (freqs <= min(high_hz, nyquist))
    band_indices = np.flatnonzero(band_mask)
    if band_indices.size == 0 or total_power <= 1e-20:
        return None

    band_power_values = power[band_indices]
    band_power = float(np.sum(band_power_values))
    if band_power <= 1e-20:
        return None

    local_peak = int(np.argmax(band_power_values))
    peak_index = int(band_indices[local_peak])
    peak_power = float(power[peak_index])
    peak_freq = _parabolic_frequency(power, peak_index, sample_rate_hz / n_fft)
    peak_freq = float(np.clip(peak_freq, low_hz, high_hz))

    background_values = band_power_values[band_power_values < peak_power]
    if background_values.size:
        background = float(np.median(background_values))
    else:
        background = float(np.mean(band_power_values))
    prominence = peak_power / max(background, 1e-20)

    return SpectralPeak(
        frequency_hz=peak_freq,
        bpm=peak_freq * 60.0,
        power=peak_power,
        prominence=float(prominence),
        in_band_energy_ratio=_clamp01(band_power / total_power),
        band_power=band_power,
        total_power=total_power,
    )


def _parabolic_frequency(power: NDArray[np.float64], peak_index: int, bin_width_hz: float) -> float:
    if peak_index <= 0 or peak_index >= power.size - 1:
        return peak_index * bin_width_hz
    left = float(power[peak_index - 1])
    center = float(power[peak_index])
    right = float(power[peak_index + 1])
    denom = left - 2.0 * center + right
    if abs(denom) <= 1e-20:
        return peak_index * bin_width_hz
    offset = float(np.clip(0.5 * (left - right) / denom, -1.0, 1.0))
    return (peak_index + offset) * bin_width_hz


def _finite_array(values: ArrayLike | Iterable[float]) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def _ordered_band(low_hz: float, high_hz: float) -> tuple[float, float]:
    low = float(low_hz)
    high = float(high_hz)
    if not isfinite(low) or not isfinite(high):
        return 0.0, 0.0
    return (low, high) if low <= high else (high, low)


def _next_power_of_two(value: int) -> int:
    return 1 << (max(int(value), 1) - 1).bit_length()


def _clamp01(value: float) -> float:
    if not isfinite(float(value)):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))
