from __future__ import annotations

import numpy as np
import pytest

from ruview.vitals import (
    BREATHING_BAND_HZ,
    HEART_RATE_BAND_HZ,
    BpmSmoothingBuffer,
    BreathingRateEstimator,
    CsiVitalFrame,
    CsiVitalPreprocessor,
    HeartRateEstimator,
    SignalQuality,
    VitalStatus,
    extract_breathing_residual,
    extract_heart_residual,
    frequency_domain_bandpass,
    signal_quality_from_amplitude,
)


def test_csi_vital_frame_validates_matching_phase_shape() -> None:
    frame = CsiVitalFrame([1.0, 2.0], phases=[0.0, 0.1], sample_index=7, sample_rate_hz=50.0)

    assert frame.n_subcarriers == 2
    assert frame.sample_index == 7
    np.testing.assert_allclose(frame.amplitudes, [1.0, 2.0])

    with pytest.raises(ValueError, match="matching lengths"):
        CsiVitalFrame([1.0, 2.0], phases=[0.0])


def test_preprocessor_suppresses_static_components_and_resets() -> None:
    preprocessor = CsiVitalPreprocessor(n_subcarriers=2, alpha=0.1)
    first = preprocessor.process(CsiVitalFrame([5.0, 10.0]))
    assert first is not None
    np.testing.assert_allclose(first, [0.0, 0.0])

    last = np.array([0.0, 0.0])
    for _ in range(100):
        last = preprocessor.process([5.0, 10.0])

    np.testing.assert_allclose(last, [0.0, 0.0], atol=1e-9)
    residual = preprocessor.process([6.0, 8.0])
    assert residual is not None
    np.testing.assert_allclose(residual, [1.0, -2.0], atol=1e-2)

    preprocessor.set_alpha(-5.0)
    assert preprocessor.alpha == 0.001
    preprocessor.reset()
    reset_residual = preprocessor.process([6.0, 8.0])
    np.testing.assert_allclose(reset_residual, [0.0, 0.0])


def test_esp32_default_preprocessor_tracks_56_subcarriers() -> None:
    assert CsiVitalPreprocessor.esp32_default().n_subcarriers == 56


def test_frequency_domain_bandpass_rejects_out_of_band_energy() -> None:
    sample_rate = 20.0
    t = np.arange(int(sample_rate * 20.0)) / sample_rate
    in_band = np.sin(2.0 * np.pi * 0.3 * t)
    out_of_band = 0.8 * np.sin(2.0 * np.pi * 3.0 * t)
    filtered = frequency_domain_bandpass(in_band + out_of_band, sample_rate, *BREATHING_BAND_HZ)

    assert np.mean(filtered**2) > 0.35
    assert np.mean((filtered - in_band) ** 2) < np.mean(out_of_band**2) * 0.05


def test_breathing_estimator_detects_18_bpm_sine() -> None:
    sample_rate = 20.0
    estimator = BreathingRateEstimator(sample_rate_hz=sample_rate, window_seconds=30.0, smoother=None)
    target_hz = 18.0 / 60.0
    weights = np.ones(8) / 8.0

    estimate = None
    for i in range(int(sample_rate * 30.0)):
        t = i / sample_rate
        signal = np.sin(2.0 * np.pi * target_hz * t)
        residuals = signal * np.linspace(0.8, 1.2, 8)
        estimate = estimator.update(residuals, weights)

    assert estimate is not None
    assert estimate.status is VitalStatus.VALID
    assert estimate.quality is SignalQuality.VALID
    assert estimate.value_bpm == pytest.approx(18.0, abs=1.5)
    assert estimate.confidence >= 0.72
    assert estimate.peak_prominence > 1.0
    assert estimate.in_band_energy_ratio > 0.8


def test_heart_rate_estimator_detects_72_bpm_sine() -> None:
    sample_rate = 50.0
    estimator = HeartRateEstimator(sample_rate_hz=sample_rate, window_seconds=15.0, smoother=None)
    target_hz = 72.0 / 60.0
    phases = np.linspace(0.0, 0.03, 8)

    estimate = None
    for i in range(int(sample_rate * 15.0)):
        t = i / sample_rate
        signal = np.sin(2.0 * np.pi * target_hz * t)
        residuals = signal * np.array([0.08, 0.10, 0.09, 0.11, 0.10, 0.12, 0.09, 0.10])
        estimate = estimator.update(residuals, phases)

    assert estimate is not None
    assert estimate.status is VitalStatus.VALID
    assert estimate.value_bpm == pytest.approx(72.0, abs=2.0)
    assert estimate.confidence >= 0.72
    assert estimate.in_band_energy_ratio > 0.8


def test_short_static_and_noisy_inputs_are_not_reported_as_valid() -> None:
    sample_rate = 20.0
    short = BreathingRateEstimator(sample_rate_hz=sample_rate, window_seconds=30.0, smoother=None)
    for _ in range(int(sample_rate * 3.0)):
        estimate = short.update([1.0, 1.0, 1.0, 1.0])
    assert estimate.status is VitalStatus.UNAVAILABLE

    static = BreathingRateEstimator(sample_rate_hz=sample_rate, window_seconds=30.0, smoother=None)
    for _ in range(int(sample_rate * 12.0)):
        estimate = static.update([0.0, 0.0, 0.0, 0.0])
    assert estimate.status is VitalStatus.UNAVAILABLE

    rng = np.random.default_rng(7)
    noisy = HeartRateEstimator(sample_rate_hz=sample_rate, window_seconds=15.0, smoother=None)
    for _ in range(int(sample_rate * 15.0)):
        estimate = noisy.update(rng.normal(0.0, 1.0, size=8), rng.normal(0.0, np.pi, size=8))
    assert estimate.status in {
        VitalStatus.DEGRADED,
        VitalStatus.UNRELIABLE,
        VitalStatus.UNAVAILABLE,
    }


def test_residual_fusion_helpers_handle_weights_and_wrapped_phase() -> None:
    residuals = np.array([[1.0, 3.0], [2.0, 4.0]])

    np.testing.assert_allclose(extract_breathing_residual(residuals, [0.25, 0.75]), [2.5, 3.5])
    coherent = extract_heart_residual([1.0, 1.0], [np.pi - 0.001, -np.pi + 0.001])
    assert coherent == pytest.approx(1.0, abs=1e-6)


def test_bpm_smoothing_buffers_support_median_mean_and_ema() -> None:
    median = BpmSmoothingBuffer(mode="median", window=3)
    assert median.update(60.0) == 60.0
    assert median.update(90.0) == 75.0
    assert median.update(300.0) == 90.0

    mean = BpmSmoothingBuffer(mode="mean", window=2)
    assert mean.update(60.0) == 60.0
    assert mean.update(90.0) == 75.0
    assert mean.update(120.0) == 105.0

    ema = BpmSmoothingBuffer(mode="ema", alpha=0.5)
    assert ema.update(100.0) == 100.0
    assert ema.update(80.0) == 90.0


def test_signal_quality_from_amplitude_labels_static_as_unavailable_or_unreliable() -> None:
    static_score, static_label = signal_quality_from_amplitude(np.ones(56), history_fill_fraction=1.0)
    useful_score, useful_label = signal_quality_from_amplitude(
        10.0 + 1.5 * np.sin(np.linspace(0.0, 2.0 * np.pi, 56)),
        history_fill_fraction=1.0,
    )

    assert static_score < 0.18
    assert static_label is SignalQuality.UNAVAILABLE
    assert useful_score > static_score
    assert useful_label in {SignalQuality.DEGRADED, SignalQuality.VALID}


def test_heart_estimator_single_subcarrier_lowers_confidence() -> None:
    sample_rate = 50.0
    estimator = HeartRateEstimator(
        sample_rate_hz=sample_rate,
        window_seconds=15.0,
        min_subcarriers=4,
        smoother=None,
    )
    target_hz = 72.0 / 60.0
    for i in range(int(sample_rate * 15.0)):
        t = i / sample_rate
        estimator.update([np.sin(2.0 * np.pi * target_hz * t)], [0.0])

    estimate = estimator.estimate()
    assert estimate.status in {VitalStatus.UNRELIABLE, VitalStatus.UNAVAILABLE}
    assert estimate.confidence < 0.45
    assert estimator.band_hz == HEART_RATE_BAND_HZ
