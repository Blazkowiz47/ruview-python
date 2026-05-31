"""NumPy-based MAT vital and movement detection primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ruview.mat.domain import (
    BreathingPattern,
    BreathingType,
    HeartbeatSignature,
    MovementActivity,
    MovementType,
    SignalStrength,
    SurvivorCondition,
    VitalSignsReading,
)


BREATHING_RATE_BPM = (4.0, 40.0)
HEART_RATE_BPM = (30.0, 200.0)


@dataclass(frozen=True)
class SpectralPeak:
    """Diagnostic description of a detected spectral peak."""

    frequency_hz: float
    rate_bpm: float
    amplitude: float
    regularity: float
    confidence: float


@dataclass
class BreathingDetectorConfig:
    """Configuration for FFT-based breathing detection."""

    min_rate_bpm: float = BREATHING_RATE_BPM[0]
    max_rate_bpm: float = BREATHING_RATE_BPM[1]
    min_amplitude: float = 0.1
    window_size: int = 512
    confidence_threshold: float = 0.3

    def __post_init__(self) -> None:
        self.min_rate_bpm = max(float(self.min_rate_bpm), 0.0)
        self.max_rate_bpm = max(float(self.max_rate_bpm), self.min_rate_bpm)
        self.min_amplitude = max(float(self.min_amplitude), 0.0)
        self.window_size = max(int(self.window_size), 4)
        self.confidence_threshold = _clamp01(self.confidence_threshold)


class BreathingDetector:
    """Detect breathing patterns from amplitude variation."""

    def __init__(self, config: BreathingDetectorConfig | None = None) -> None:
        self.config = config or BreathingDetectorConfig()

    @classmethod
    def with_defaults(cls) -> "BreathingDetector":
        """Return a detector with default thresholds."""

        return cls(BreathingDetectorConfig())

    def detect(
        self,
        csi_amplitudes: ArrayLike,
        sample_rate_hz: float,
    ) -> BreathingPattern | None:
        """Detect a breathing pattern from a scalar or sample-by-subcarrier signal."""

        samples = _coerce_signal(csi_amplitudes)
        if sample_rate_hz <= 0.0 or samples.size < self.config.window_size:
            return None

        peak = self._spectral_peak(
            samples,
            sample_rate_hz=sample_rate_hz,
            min_rate_bpm=self.config.min_rate_bpm,
            max_rate_bpm=self.config.max_rate_bpm,
            window="hann",
        )
        if peak is None or peak.amplitude < self.config.min_amplitude:
            return None

        pattern_type = self._classify_pattern(peak.rate_bpm, peak.regularity)
        if peak.confidence < self.config.confidence_threshold:
            return None
        return BreathingPattern(
            rate_bpm=peak.rate_bpm,
            amplitude=peak.amplitude,
            regularity=peak.regularity,
            pattern_type=pattern_type,
        )

    def _spectral_peak(
        self,
        samples: NDArray[np.float64],
        *,
        sample_rate_hz: float,
        min_rate_bpm: float,
        max_rate_bpm: float,
        window: str,
    ) -> SpectralPeak | None:
        freqs, amplitude = _one_sided_amplitude_spectrum(samples, sample_rate_hz, window=window)
        min_freq = min_rate_bpm / 60.0
        max_freq = max_rate_bpm / 60.0
        return _peak_in_band(freqs, amplitude, min_freq, max_freq)

    def _classify_pattern(self, rate_bpm: float, regularity: float) -> BreathingType:
        if rate_bpm < 6.0:
            return BreathingType.AGONAL if regularity < 0.3 else BreathingType.SHALLOW
        if rate_bpm < 10.0:
            return BreathingType.SHALLOW
        if rate_bpm > 30.0:
            return BreathingType.LABORED
        if regularity < 0.4:
            return BreathingType.IRREGULAR
        return BreathingType.NORMAL


@dataclass
class HeartbeatDetectorConfig:
    """Configuration for FFT-based heartbeat detection."""

    min_rate_bpm: float = HEART_RATE_BPM[0]
    max_rate_bpm: float = HEART_RATE_BPM[1]
    min_signal_strength: float = 0.05
    window_size: int = 1024
    confidence_threshold: float = 0.4

    def __post_init__(self) -> None:
        self.min_rate_bpm = max(float(self.min_rate_bpm), 0.0)
        self.max_rate_bpm = max(float(self.max_rate_bpm), self.min_rate_bpm)
        self.min_signal_strength = max(float(self.min_signal_strength), 0.0)
        self.window_size = max(int(self.window_size), 4)
        self.confidence_threshold = _clamp01(self.confidence_threshold)


class HeartbeatDetector:
    """Detect heartbeat signatures from CSI phase variation."""

    def __init__(self, config: HeartbeatDetectorConfig | None = None) -> None:
        self.config = config or HeartbeatDetectorConfig()

    @classmethod
    def with_defaults(cls) -> "HeartbeatDetector":
        """Return a detector with default thresholds."""

        return cls(HeartbeatDetectorConfig())

    def detect(
        self,
        csi_phase: ArrayLike,
        sample_rate_hz: float,
        breathing_rate_bpm: float | None = None,
    ) -> HeartbeatSignature | None:
        """Detect a heartbeat signature from phase data."""

        samples = _coerce_signal(csi_phase)
        if sample_rate_hz <= 0.0 or samples.size < self.config.window_size:
            return None

        if breathing_rate_bpm is not None:
            filtered = self._remove_breathing_component(samples, sample_rate_hz, breathing_rate_bpm)
        else:
            filtered = _highpass_filter(samples, sample_rate_hz, cutoff_hz=0.8)

        freqs, amplitude = _one_sided_amplitude_spectrum(filtered, sample_rate_hz, window="blackman")
        peak = _peak_in_band(
            freqs,
            amplitude,
            self.config.min_rate_bpm / 60.0,
            self.config.max_rate_bpm / 60.0,
            require_local_maximum=True,
        )
        if peak is None or peak.amplitude < self.config.min_signal_strength:
            return None

        variability = _estimate_peak_width_variability(freqs, amplitude, peak.frequency_hz)
        strength = self._categorize_strength(peak.amplitude)
        confidence = self._confidence(peak.amplitude, variability)
        if confidence < self.config.confidence_threshold:
            return None
        return HeartbeatSignature(
            rate_bpm=peak.rate_bpm,
            variability=variability,
            strength=strength,
        )

    def _remove_breathing_component(
        self,
        samples: NDArray[np.float64],
        sample_rate_hz: float,
        breathing_rate_bpm: float,
    ) -> NDArray[np.float64]:
        filtered = samples
        breathing_hz = max(float(breathing_rate_bpm) / 60.0, 0.0)
        for harmonic in (1, 2, 3):
            notch_hz = breathing_hz * harmonic
            if 0.0 < notch_hz < sample_rate_hz / 2.0:
                filtered = _notch_filter(filtered, sample_rate_hz, notch_hz, bandwidth_hz=0.05)
        return filtered

    def _categorize_strength(self, strength: float) -> SignalStrength:
        if strength > 0.5:
            return SignalStrength.STRONG
        if strength > 0.2:
            return SignalStrength.MODERATE
        if strength > 0.1:
            return SignalStrength.WEAK
        return SignalStrength.VERY_WEAK

    def _confidence(self, strength: float, variability: float) -> float:
        strength_score = min(strength / 0.5, 1.0)
        variability_score = 1.0 if 0.05 < variability < 0.5 else 0.5
        return _clamp01(strength_score * 0.7 + variability_score * 0.3)


@dataclass
class MovementClassifierConfig:
    """Configuration for variance/autocorrelation movement classification."""

    movement_threshold: float = 0.1
    gross_movement_threshold: float = 0.5
    window_size: int = 100
    periodicity_threshold: float = 0.3

    def __post_init__(self) -> None:
        self.movement_threshold = max(float(self.movement_threshold), 0.0)
        self.gross_movement_threshold = max(float(self.gross_movement_threshold), 1e-12)
        self.window_size = max(int(self.window_size), 3)
        self.periodicity_threshold = _clamp01(self.periodicity_threshold)


class MovementClassifier:
    """Classify movement from CSI amplitude variation."""

    def __init__(self, config: MovementClassifierConfig | None = None) -> None:
        self.config = config or MovementClassifierConfig()

    @classmethod
    def with_defaults(cls) -> "MovementClassifier":
        """Return a classifier with default thresholds."""

        return cls(MovementClassifierConfig())

    def classify(self, csi_signal: ArrayLike, sample_rate_hz: float) -> MovementActivity:
        """Classify movement type, intensity, and frequency."""

        samples = _coerce_signal(csi_signal)
        if sample_rate_hz <= 0.0 or samples.size < self.config.window_size:
            return MovementActivity()
        variance = float(np.var(samples))
        max_change = float(np.max(np.abs(np.diff(samples)))) if samples.size > 1 else 0.0
        periodicity = _autocorrelation_periodicity(samples)
        movement_type, is_voluntary = self._determine_movement_type(
            variance,
            max_change,
            periodicity,
        )
        return MovementActivity(
            movement_type=movement_type,
            intensity=self._intensity(variance, max_change),
            frequency=_zero_crossing_frequency(samples, sample_rate_hz),
            is_voluntary=is_voluntary,
        )

    def _determine_movement_type(
        self,
        variance: float,
        max_change: float,
        periodicity: float,
    ) -> tuple[MovementType, bool]:
        if (
            variance < self.config.movement_threshold * 0.5
            and max_change < self.config.movement_threshold
        ):
            return MovementType.NONE, False
        if (
            max_change > self.config.gross_movement_threshold
            and variance > self.config.movement_threshold
        ):
            return MovementType.GROSS, periodicity < self.config.periodicity_threshold
        if periodicity > self.config.periodicity_threshold:
            if variance < self.config.movement_threshold * 2.0:
                return MovementType.PERIODIC, False
            return MovementType.TREMOR, False
        if variance > self.config.movement_threshold * 0.5:
            return MovementType.FINE, periodicity < 0.2
        return MovementType.NONE, False

    def _intensity(self, variance: float, max_change: float) -> float:
        variance_score = min(variance / (self.config.gross_movement_threshold * 2.0), 1.0)
        change_score = min(max_change / self.config.gross_movement_threshold, 1.0)
        return _clamp01(variance_score * 0.6 + change_score * 0.4)


@dataclass
class EnsembleConfig:
    """Weights for combining vital and movement confidences."""

    breathing_weight: float = 0.50
    heartbeat_weight: float = 0.30
    movement_weight: float = 0.20
    min_ensemble_confidence: float = 0.3

    def __post_init__(self) -> None:
        self.breathing_weight = max(float(self.breathing_weight), 0.0)
        self.heartbeat_weight = max(float(self.heartbeat_weight), 0.0)
        self.movement_weight = max(float(self.movement_weight), 0.0)
        self.min_ensemble_confidence = _clamp01(self.min_ensemble_confidence)


@dataclass(frozen=True)
class SignalConfidences:
    """Individual confidence scores used by the ensemble."""

    breathing: float = 0.0
    heartbeat: float = 0.0
    movement: float = 0.0


@dataclass(frozen=True)
class EnsembleResult:
    """Weighted vital-sign ensemble result."""

    confidence: float
    recommended_condition: SurvivorCondition
    breathing_detected: bool
    heartbeat_detected: bool
    movement_detected: bool
    signal_confidences: SignalConfidences


class EnsembleClassifier:
    """Combine breathing, heartbeat, and movement into one confidence."""

    def __init__(self, config: EnsembleConfig | None = None) -> None:
        self.config = config or EnsembleConfig()

    def classify(self, reading: VitalSignsReading) -> EnsembleResult:
        """Classify a vital reading with weighted detector confidence."""

        breathing_confidence = reading.breathing.confidence() if reading.breathing else 0.0
        heartbeat_confidence = reading.heartbeat.confidence() if reading.heartbeat else 0.0
        movement_confidence = reading.movement.confidence() if reading.has_movement() else 0.0
        total_weight = (
            self.config.breathing_weight
            + self.config.heartbeat_weight
            + self.config.movement_weight
        )
        confidence = 0.0
        if total_weight > 0.0:
            confidence = (
                breathing_confidence * self.config.breathing_weight
                + heartbeat_confidence * self.config.heartbeat_weight
                + movement_confidence * self.config.movement_weight
            ) / total_weight
        condition = self._condition(reading, confidence)
        return EnsembleResult(
            confidence=_clamp01(confidence),
            recommended_condition=condition,
            breathing_detected=reading.has_breathing(),
            heartbeat_detected=reading.has_heartbeat(),
            movement_detected=reading.has_movement(),
            signal_confidences=SignalConfidences(
                breathing=breathing_confidence,
                heartbeat=heartbeat_confidence,
                movement=movement_confidence,
            ),
        )

    def _condition(self, reading: VitalSignsReading, confidence: float) -> SurvivorCondition:
        if reading.breathing is not None:
            breathing = reading.breathing
            if breathing.pattern_type in {BreathingType.AGONAL, BreathingType.APNEA}:
                return SurvivorCondition.IMMEDIATE
            if breathing.rate_bpm < 10.0 or breathing.rate_bpm > 30.0:
                return SurvivorCondition.IMMEDIATE
        if confidence < self.config.min_ensemble_confidence:
            return SurvivorCondition.UNKNOWN
        if not reading.has_vitals():
            return SurvivorCondition.DECEASED
        if not reading.has_breathing() and (reading.has_heartbeat() or reading.has_movement()):
            return SurvivorCondition.IMMEDIATE
        if reading.breathing is None:
            return SurvivorCondition.UNKNOWN
        if not 12.0 <= reading.breathing.rate_bpm <= 24.0:
            return SurvivorCondition.DELAYED if reading.has_movement() else SurvivorCondition.IMMEDIATE
        return SurvivorCondition.MINOR if reading.has_movement() else SurvivorCondition.DELAYED


@dataclass
class DetectionConfig:
    """Configuration for the full detector pipeline."""

    breathing: BreathingDetectorConfig = field(default_factory=BreathingDetectorConfig)
    heartbeat: HeartbeatDetectorConfig = field(default_factory=HeartbeatDetectorConfig)
    movement: MovementClassifierConfig = field(default_factory=MovementClassifierConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    sample_rate_hz: float = 1000.0
    enable_heartbeat: bool = False
    min_confidence: float = 0.3
    max_buffer_seconds: float = 30.0

    def __post_init__(self) -> None:
        self.sample_rate_hz = max(float(self.sample_rate_hz), 1e-12)
        self.min_confidence = _clamp01(self.min_confidence)
        self.max_buffer_seconds = max(float(self.max_buffer_seconds), 1e-12)


@dataclass
class CsiDataBuffer:
    """Flat CSI amplitude/phase sample buffer."""

    sample_rate_hz: float = 1000.0
    amplitudes: list[float] = field(default_factory=list)
    phases: list[float] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)

    def add_samples(
        self,
        amplitudes: ArrayLike,
        phases: ArrayLike | None = None,
    ) -> None:
        """Append samples and deterministic timestamps."""

        amp = _coerce_signal(amplitudes)
        phase = np.zeros_like(amp) if phases is None else _coerce_signal(phases)
        n = min(amp.size, phase.size)
        if n <= 0:
            return
        amp = amp[:n]
        phase = phase[:n]
        start = self.timestamps[-1] if self.timestamps else 0.0
        step = 1.0 / self.sample_rate_hz
        self.amplitudes.extend(float(value) for value in amp)
        self.phases.extend(float(value) for value in phase)
        self.timestamps.extend(start + (index + 1) * step for index in range(n))

    def clear(self) -> None:
        """Clear all buffered samples."""

        self.amplitudes.clear()
        self.phases.clear()
        self.timestamps.clear()

    def duration(self) -> float:
        """Return buffered duration in seconds."""

        return len(self.amplitudes) / self.sample_rate_hz

    def has_sufficient_data(self, min_duration_seconds: float) -> bool:
        """Return whether enough samples are buffered."""

        return self.duration() >= min_duration_seconds


@dataclass(frozen=True)
class PipelineDetection:
    """Pipeline output containing the reading and ensemble summary."""

    reading: VitalSignsReading
    ensemble: EnsembleResult


class DetectionPipeline:
    """Synchronous research pipeline combining MAT detectors."""

    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()
        self.breathing_detector = BreathingDetector(self.config.breathing)
        self.heartbeat_detector = HeartbeatDetector(self.config.heartbeat)
        self.movement_classifier = MovementClassifier(self.config.movement)
        self.ensemble_classifier = EnsembleClassifier(self.config.ensemble)
        self.data_buffer = CsiDataBuffer(sample_rate_hz=self.config.sample_rate_hz)

    def add_data(self, amplitudes: ArrayLike, phases: ArrayLike | None = None) -> None:
        """Append CSI data to the internal buffer."""

        self.data_buffer.add_samples(amplitudes, phases)
        max_samples = int(round(self.config.max_buffer_seconds * self.config.sample_rate_hz))
        overflow = len(self.data_buffer.amplitudes) - max_samples
        if overflow > 0:
            del self.data_buffer.amplitudes[:overflow]
            del self.data_buffer.phases[:overflow]
            del self.data_buffer.timestamps[:overflow]

    def clear_buffer(self) -> None:
        """Clear the internal data buffer."""

        self.data_buffer.clear()

    def detect(self, buffer: CsiDataBuffer | None = None) -> PipelineDetection | None:
        """Detect vital signs from a buffer and return ensemble confidence."""

        source = buffer or self.data_buffer
        breathing = self.breathing_detector.detect(source.amplitudes, source.sample_rate_hz)
        heartbeat = None
        if self.config.enable_heartbeat:
            breathing_rate = breathing.rate_bpm if breathing is not None else None
            heartbeat = self.heartbeat_detector.detect(
                source.phases,
                source.sample_rate_hz,
                breathing_rate,
            )
        movement = self.movement_classifier.classify(source.amplitudes, source.sample_rate_hz)
        if breathing is None and heartbeat is None and movement.movement_type is MovementType.NONE:
            return None
        reading = VitalSignsReading(breathing=breathing, heartbeat=heartbeat, movement=movement)
        ensemble = self.ensemble_classifier.classify(reading)
        if ensemble.confidence < self.config.min_confidence:
            return None
        return PipelineDetection(reading=reading, ensemble=ensemble)

    def process(
        self,
        amplitudes: ArrayLike,
        phases: ArrayLike | None = None,
    ) -> PipelineDetection | None:
        """Detect from a one-shot amplitude/phase window without mutating the buffer."""

        buffer = CsiDataBuffer(sample_rate_hz=self.config.sample_rate_hz)
        buffer.add_samples(amplitudes, phases)
        return self.detect(buffer)


def _one_sided_amplitude_spectrum(
    samples: NDArray[np.float64],
    sample_rate_hz: float,
    *,
    window: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    values = samples.astype(np.float64, copy=False)
    values = values - float(np.mean(values))
    n_fft = _next_power_of_two(max(values.size, 4))
    window_values = _window(values.size, window)
    padded = np.zeros(n_fft, dtype=np.float64)
    padded[: values.size] = values * window_values
    spectrum = np.fft.rfft(padded)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate_hz)
    coherent_gain = max(float(np.sum(window_values)) / 2.0, 1e-12)
    amplitude = np.abs(spectrum) / coherent_gain
    if amplitude.size:
        amplitude[0] = 0.0
    return freqs, amplitude.astype(np.float64, copy=False)


def _peak_in_band(
    freqs: NDArray[np.float64],
    amplitude: NDArray[np.float64],
    min_freq_hz: float,
    max_freq_hz: float,
    *,
    require_local_maximum: bool = False,
) -> SpectralPeak | None:
    if amplitude.size == 0 or min_freq_hz >= max_freq_hz:
        return None
    mask = (freqs >= min_freq_hz) & (freqs <= max_freq_hz)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return None
    peak_index = int(indices[np.argmax(amplitude[indices])])
    if require_local_maximum and 0 < peak_index < amplitude.size - 1:
        if amplitude[peak_index] <= amplitude[peak_index - 1] or amplitude[peak_index] <= amplitude[peak_index + 1]:
            return None
    frequency = _interpolated_frequency(freqs, amplitude, peak_index)
    peak_amplitude = float(amplitude[peak_index])
    power = amplitude * amplitude
    total_power = float(np.sum(power[freqs > 0.0]))
    if total_power <= 1e-18:
        return None
    harmonic_power = 0.0
    for multiplier in (2, 3):
        harmonic_frequency = frequency * multiplier
        if harmonic_frequency <= freqs[-1]:
            harmonic_index = int(np.argmin(np.abs(freqs - harmonic_frequency)))
            harmonic_power += float(power[harmonic_index])
    regularity = _clamp01((float(power[peak_index]) + 0.5 * harmonic_power) / total_power * 3.0)
    confidence = _clamp01(min(peak_amplitude, 1.0) * 0.4 + regularity * 0.6)
    return SpectralPeak(
        frequency_hz=frequency,
        rate_bpm=frequency * 60.0,
        amplitude=peak_amplitude,
        regularity=regularity,
        confidence=confidence,
    )


def _estimate_peak_width_variability(
    freqs: NDArray[np.float64],
    amplitude: NDArray[np.float64],
    peak_frequency_hz: float,
) -> float:
    if amplitude.size == 0:
        return 0.0
    peak_index = int(np.argmin(np.abs(freqs - peak_frequency_hz)))
    peak_power = float(amplitude[peak_index] ** 2)
    if peak_power <= 0.0:
        return 0.0
    half_power = peak_power / 2.0
    power = amplitude * amplitude
    left = peak_index
    right = peak_index
    while left > 0 and power[left] > half_power:
        left -= 1
    while right < power.size - 1 and power[right] > half_power:
        right += 1
    bandwidth_hz = float(freqs[right] - freqs[left])
    return _clamp01((bandwidth_hz * 60.0) / 20.0)


def _highpass_filter(
    samples: NDArray[np.float64],
    sample_rate_hz: float,
    *,
    cutoff_hz: float,
) -> NDArray[np.float64]:
    if samples.size == 0:
        return samples.copy()
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    dt = 1.0 / sample_rate_hz
    alpha = rc / (rc + dt)
    output = np.zeros_like(samples, dtype=np.float64)
    output[0] = samples[0]
    for index in range(1, samples.size):
        output[index] = alpha * (output[index - 1] + samples[index] - samples[index - 1])
    return output


def _notch_filter(
    samples: NDArray[np.float64],
    sample_rate_hz: float,
    center_hz: float,
    bandwidth_hz: float,
) -> NDArray[np.float64]:
    w0 = 2.0 * math.pi * center_hz / sample_rate_hz
    bw = 2.0 * math.pi * bandwidth_hz / sample_rate_hz
    radius = 1.0 - bw / 2.0
    cos_w0 = math.cos(w0)
    b0 = 1.0
    b1 = -2.0 * cos_w0
    b2 = 1.0
    a1 = -2.0 * radius * cos_w0
    a2 = radius * radius
    output = np.zeros_like(samples, dtype=np.float64)
    x1 = x2 = y1 = y2 = 0.0
    for index, sample in enumerate(samples):
        y = b0 * sample + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        output[index] = y
        x2, x1 = x1, float(sample)
        y2, y1 = y1, float(y)
    return output


def _autocorrelation_periodicity(samples: NDArray[np.float64]) -> float:
    if samples.size < 3:
        return 0.0
    centered = samples - float(np.mean(samples))
    variance = float(np.dot(centered, centered))
    if variance <= 1e-18:
        return 0.0
    max_lag = samples.size // 2
    best = 0.0
    for lag in range(1, max_lag):
        corr = float(np.dot(centered[:-lag], centered[lag:]) / variance)
        if corr > best:
            best = corr
    return _clamp01(best)


def _zero_crossing_frequency(samples: NDArray[np.float64], sample_rate_hz: float) -> float:
    if samples.size < 3 or sample_rate_hz <= 0.0:
        return 0.0
    centered = samples - float(np.mean(samples))
    crossings = int(np.count_nonzero((centered[:-1] >= 0.0) != (centered[1:] >= 0.0)))
    duration = samples.size / sample_rate_hz
    if duration <= 0.0:
        return 0.0
    return crossings / (2.0 * duration)


def _coerce_signal(source: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(source, dtype=np.float64)
    if values.size == 0:
        return np.asarray([], dtype=np.float64)
    if values.ndim > 1:
        axes = tuple(range(1, values.ndim))
        values = np.mean(values, axis=axes)
    values = values.reshape(-1).astype(np.float64, copy=False)
    if np.all(np.isfinite(values)):
        return values
    finite = values[np.isfinite(values)]
    fill = float(np.mean(finite)) if finite.size else 0.0
    return np.nan_to_num(values, nan=fill, posinf=fill, neginf=fill)


def _window(size: int, name: str) -> NDArray[np.float64]:
    if size <= 1:
        return np.ones(max(size, 1), dtype=np.float64)
    normalized = name.strip().lower()
    if normalized == "blackman":
        return np.blackman(size).astype(np.float64, copy=False)
    if normalized in {"hann", "hanning"}:
        return np.hanning(size).astype(np.float64, copy=False)
    return np.ones(size, dtype=np.float64)


def _interpolated_frequency(
    freqs: NDArray[np.float64],
    amplitude: NDArray[np.float64],
    peak_index: int,
) -> float:
    if peak_index <= 0 or peak_index >= amplitude.size - 1:
        return float(freqs[peak_index])
    left = float(amplitude[peak_index - 1])
    center = float(amplitude[peak_index])
    right = float(amplitude[peak_index + 1])
    denominator = left - 2.0 * center + right
    if abs(denominator) <= 1e-18:
        return float(freqs[peak_index])
    offset = 0.5 * (left - right) / denominator
    bin_width = float(freqs[1] - freqs[0]) if freqs.size > 1 else 0.0
    return max(float(freqs[peak_index] + offset * bin_width), 0.0)


def _next_power_of_two(value: int) -> int:
    return 1 << (max(int(value), 1) - 1).bit_length()


def _clamp01(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return min(max(float(value), 0.0), 1.0)


__all__ = [
    "BREATHING_RATE_BPM",
    "HEART_RATE_BPM",
    "BreathingDetector",
    "BreathingDetectorConfig",
    "CsiDataBuffer",
    "DetectionConfig",
    "DetectionPipeline",
    "EnsembleClassifier",
    "EnsembleConfig",
    "EnsembleResult",
    "HeartbeatDetector",
    "HeartbeatDetectorConfig",
    "MovementClassifier",
    "MovementClassifierConfig",
    "PipelineDetection",
    "SignalConfidences",
    "SpectralPeak",
]
