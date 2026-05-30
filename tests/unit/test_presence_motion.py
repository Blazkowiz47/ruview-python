from __future__ import annotations

import numpy as np

from ruview.hardware import SyntheticCsiConfig, generate_synthetic_sequence
from ruview.signal import (
    DetectionDebouncer,
    RollingBaseline,
    adaptive_threshold,
    calculate_motion_score,
    classify_presence,
    measure_baseline,
    stack_csi_frames,
)


def test_motion_score_orders_synthetic_empty_present_and_walking() -> None:
    config = SyntheticCsiConfig(seed=123, frames=96, streams=3, subcarriers=32)
    empty = _window("empty_room", config)
    present = _window("person_present", config)
    walking = _window("walking", config)
    baseline = RollingBaseline(alpha=1.0, window=8)
    baseline.update(empty)

    empty_score = calculate_motion_score(empty, baseline=baseline)
    present_score = calculate_motion_score(present, baseline=baseline)
    walking_score = calculate_motion_score(walking, baseline=baseline)

    assert empty_score.total < present_score.total < walking_score.total
    assert present_score.amplitude_variance_component > 0.1
    assert walking_score.temporal_delta_component > present_score.temporal_delta_component
    assert walking_score.phase_variance_component > present_score.phase_variance_component
    assert not present_score.is_motion_detected(0.45)
    assert walking_score.is_motion_detected(0.45)


def test_presence_classifier_distinguishes_empty_still_and_moving() -> None:
    config = SyntheticCsiConfig(seed=456, frames=96, streams=3, subcarriers=32)
    empty = _window("empty_room", config)
    present = _window("person_present", config)
    stillness = _window("stillness", config)
    walking = _window("walking", config)
    baseline = RollingBaseline(alpha=1.0, window=8)
    baseline.update(empty)

    empty_result = classify_presence(empty, baseline=baseline)
    present_result = classify_presence(present, baseline=baseline)
    stillness_result = classify_presence(stillness, baseline=baseline)
    walking_result = classify_presence(walking, baseline=baseline)

    assert empty_result.state == "empty"
    assert not empty_result.present
    assert present_result.state == "still"
    assert present_result.present
    assert not present_result.moving
    assert stillness_result.state == "still"
    assert stillness_result.present
    assert not stillness_result.moving
    assert walking_result.state == "moving"
    assert walking_result.present
    assert walking_result.moving
    assert empty_result.presence_score < present_result.presence_score < walking_result.presence_score


def test_rolling_baseline_ema_history_and_adaptive_threshold() -> None:
    config = SyntheticCsiConfig(seed=789, frames=48, streams=2, subcarriers=24)
    empty = _window("empty_room", config)
    present = _window("person_present", config)
    walking = _window("walking", config)
    empty_stats = measure_baseline(empty)
    present_stats = measure_baseline(present)
    walking_stats = measure_baseline(walking)
    baseline = RollingBaseline(alpha=0.25, window=2)

    first = baseline.update(empty)
    second = baseline.update(present)

    assert first == empty_stats
    assert baseline.sample_count == 2
    assert len(baseline.history) == 2
    expected_mean = 0.75 * empty_stats.mean_amplitude + 0.25 * present_stats.mean_amplitude
    assert np.isclose(second.mean_amplitude, expected_mean)

    baseline.update(walking)

    assert baseline.sample_count == 3
    assert len(baseline.history) == 2
    assert baseline.history[0] == present_stats
    assert baseline.history[1] == walking_stats
    expected_threshold = np.mean(
        [present_stats.amplitude_variance, walking_stats.amplitude_variance]
    )
    assert np.isclose(
        baseline.adaptive_threshold("amplitude_variance", std_multiplier=0.0),
        expected_threshold,
    )
    assert np.isclose(
        adaptive_threshold([0.2, 0.4], std_multiplier=0.0, minimum=0.0, maximum=1.0),
        0.3,
    )


def test_detection_debouncer_requires_confirmation_and_release() -> None:
    debouncer = DetectionDebouncer(confirm_count=2, release_count=2, smoothing_factor=0.5)

    assert not debouncer.update(False, confidence=0.1)
    assert not debouncer.update(True, confidence=0.9)
    assert debouncer.update(True, confidence=0.9)
    assert debouncer.state
    assert 0.0 < debouncer.smoothed_score < 1.0

    assert debouncer.update(False, confidence=0.1)
    assert not debouncer.update(False, confidence=0.1)
    assert not debouncer.state


def _window(scenario: str, config: SyntheticCsiConfig):
    return stack_csi_frames(generate_synthetic_sequence(scenario, config))
