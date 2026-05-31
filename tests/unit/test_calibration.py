from __future__ import annotations

import json
import math

import numpy as np
import pytest

from ruview.hardware import SyntheticCsiConfig, generate_synthetic_sequence
from ruview.ruvsense import (
    BASELINE_JSON_MAGIC,
    BaselineCalibration,
    CalibrationConfig,
    CalibrationRecorder,
    InsufficientFramesError,
    WelfordStats,
    load_baseline,
)
from ruview.signal import stack_csi_frames


def test_welford_variance_uses_stable_sample_estimate() -> None:
    stats = WelfordStats()
    stats.update_many([1.0, 2.0, 3.0, 4.0])

    assert stats.count == 4
    assert np.isclose(stats.mean, 2.5)
    assert np.isclose(stats.variance, np.var([1.0, 2.0, 3.0, 4.0], ddof=1))
    assert np.isclose(stats.population_variance, 1.25)


def test_circular_phase_mean_handles_pi_wraparound() -> None:
    amplitude = np.ones((4, 1), dtype=np.float64)
    phase = np.array(
        [[math.pi - 0.03], [-math.pi + 0.03], [math.pi - 0.02], [-math.pi + 0.02]],
        dtype=np.float64,
    )
    recorder = CalibrationRecorder(CalibrationConfig.custom(1, min_frames=4))

    recorder.record(amplitude, phase, frame_axis=0)
    baseline = recorder.finalize()
    subcarrier = baseline.subcarriers[0]

    assert abs(abs(subcarrier.phase_mean) - math.pi) < 0.04
    assert subcarrier.phase_dispersion < 0.001


def test_finalize_rejects_insufficient_frames() -> None:
    recorder = CalibrationRecorder(CalibrationConfig.custom(4, min_frames=3))
    recorder.record(np.ones((1, 4), dtype=np.complex128))

    with pytest.raises(InsufficientFramesError) as exc_info:
        recorder.finalize()

    assert exc_info.value.got == 1
    assert exc_info.value.need == 3


def test_finalization_from_amplitude_phase_arrays() -> None:
    amplitude = np.array(
        [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]],
        dtype=np.float64,
    )
    phase = np.zeros_like(amplitude)
    recorder = CalibrationRecorder(CalibrationConfig.custom(3, min_frames=3))

    score = recorder.record(amplitude, phase, frame_axis=0)
    baseline = recorder.finalize()

    assert score.sample_count == 9
    assert baseline.frame_count == 3
    np.testing.assert_allclose(baseline.amplitude_mean, [2.0, 3.0, 4.0])
    np.testing.assert_allclose(baseline.amplitude_variance, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(baseline.phase_mean, [0.0, 0.0, 0.0])


def test_baseline_json_roundtrip(tmp_path) -> None:
    baseline = _simple_baseline()
    path = baseline.save_json(tmp_path / "baseline.json")

    payload = json.loads(path.read_text(encoding="utf-8"))
    loaded = load_baseline(path)

    assert payload["magic"] == BASELINE_JSON_MAGIC
    assert isinstance(loaded, BaselineCalibration)
    assert loaded.tier == baseline.tier
    assert loaded.frame_count == baseline.frame_count
    assert loaded.num_subcarriers == baseline.num_subcarriers
    np.testing.assert_allclose(loaded.amplitude_mean, baseline.amplitude_mean)
    np.testing.assert_allclose(loaded.amplitude_variance, baseline.amplitude_variance)
    np.testing.assert_allclose(loaded.phase_mean, baseline.phase_mean)
    np.testing.assert_allclose(loaded.phase_dispersion, baseline.phase_dispersion)


def test_deviation_distinguishes_empty_like_from_person_like_window() -> None:
    config = SyntheticCsiConfig(seed=2468, frames=96, streams=3, subcarriers=32)
    empty = _window("empty_room", config)
    present = _window("person_present", config)
    drifted = empty.data * 1.25 * np.exp(1j * 0.8)
    recorder = CalibrationRecorder(CalibrationConfig.custom(32, min_frames=config.frames))
    recorder.record(empty)
    baseline = recorder.finalize()

    empty_score = baseline.deviation(empty)
    present_score = baseline.deviation(present)
    drift_score = baseline.deviation(drifted)
    decision = baseline.drift_decision(drifted)

    assert empty_score.is_empty_like
    assert empty_score.amplitude_z_median < 2.0
    assert present_score.amplitude_z_median > empty_score.amplitude_z_median
    assert present_score.motion_flagged
    assert drift_score.phase_drift_median > empty_score.phase_drift_median
    assert drift_score.motion_flagged
    assert present_score.drift_flagged
    assert drift_score.drift_flagged
    assert decision.recalibration_recommended


def test_subtract_baseline_reduces_mean_amplitude_without_mutating_source() -> None:
    config = SyntheticCsiConfig(seed=1357, frames=80, streams=2, subcarriers=24)
    empty = _window("empty_room", config)
    present = _window("person_present", config)
    recorder = CalibrationRecorder(CalibrationConfig.custom(24, min_frames=config.frames))
    recorder.record(empty)
    baseline = recorder.finalize()
    before = np.array(present.data, copy=True)

    residual = baseline.subtract_baseline(present)

    assert np.iscomplexobj(residual)
    assert float(np.mean(np.abs(residual))) < float(np.mean(np.abs(present.data)))
    np.testing.assert_allclose(present.data, before)


def _simple_baseline() -> BaselineCalibration:
    amplitude = np.array(
        [[1.0, 1.5, 2.0], [1.1, 1.6, 2.1], [0.9, 1.4, 1.9]],
        dtype=np.float64,
    )
    phase = np.array(
        [[0.0, 0.1, -0.1], [0.02, 0.12, -0.08], [-0.02, 0.08, -0.12]],
        dtype=np.float64,
    )
    recorder = CalibrationRecorder(CalibrationConfig.custom(3, min_frames=3))
    recorder.record(amplitude, phase, frame_axis=0)
    return recorder.finalize()


def _window(scenario: str, config: SyntheticCsiConfig):
    return stack_csi_frames(generate_synthetic_sequence(scenario, config))
