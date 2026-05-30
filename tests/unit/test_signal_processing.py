from __future__ import annotations

import numpy as np
import pytest

from ruview.core import CsiFrame, CsiMetadata, Timestamp
from ruview.signal import (
    CsiWindow,
    amplitude_phase_to_complex,
    complex_amplitude,
    complex_phase,
    complex_to_amplitude_phase,
    extract_frame_features,
    extract_window_features,
    hampel_filter,
    min_max_normalize,
    stack_csi_frames,
    subcarrier_variance,
    unwrap_phase,
    zscore_normalize,
)


def test_complex_amplitude_phase_roundtrip() -> None:
    data = np.array([[1 + 1j, -3 + 4j], [0 - 2j, 1 + 0j]], dtype=np.complex128)

    amplitude, phase = complex_to_amplitude_phase(data)

    np.testing.assert_allclose(complex_amplitude(data), np.abs(data))
    np.testing.assert_allclose(complex_phase(data), np.angle(data))
    np.testing.assert_allclose(amplitude, np.array([[np.sqrt(2), 5.0], [2.0, 1.0]]))
    np.testing.assert_allclose(amplitude_phase_to_complex(amplitude, phase), data, atol=1e-12)


def test_unwrap_phase_along_axis() -> None:
    wrapped = np.array(
        [
            [0.0, 0.75 * np.pi, -0.75 * np.pi, -0.5 * np.pi],
            [0.0, -0.75 * np.pi, 0.75 * np.pi, 0.5 * np.pi],
        ]
    )

    unwrapped = unwrap_phase(wrapped, axis=1)

    np.testing.assert_allclose(unwrapped[0], [0.0, 0.75 * np.pi, 1.25 * np.pi, 1.5 * np.pi])
    np.testing.assert_allclose(unwrapped[1], [0.0, -0.75 * np.pi, -1.25 * np.pi, -1.5 * np.pi])


def test_hampel_filter_replaces_spike_and_reports_local_stats() -> None:
    result = hampel_filter([1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0], half_window=2, threshold=3.0)

    np.testing.assert_allclose(result.filtered, np.ones(7))
    assert result.outlier_indices.tolist() == [3]
    np.testing.assert_allclose(result.medians, np.ones(7))
    assert result.sigma_estimates[3] == 0.0


def test_normalization_handles_range_zscore_and_constants() -> None:
    data = np.array([2.0, 4.0, 6.0])

    np.testing.assert_allclose(min_max_normalize(data), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(zscore_normalize(data), (data - np.mean(data)) / np.std(data))
    np.testing.assert_allclose(min_max_normalize([5.0, 5.0]), [0.0, 0.0])
    np.testing.assert_allclose(zscore_normalize([5.0, 5.0]), [0.0, 0.0])


def test_frame_and_window_feature_extraction() -> None:
    amplitude = np.array([[1.0, 2.0], [3.0, 4.0]])
    phase = np.array([[0.0, 0.5], [1.0, 1.5]])

    frame_features = extract_frame_features(amplitude, phase)

    assert frame_features.mean_amplitude == 2.5
    assert frame_features.amplitude_variance == np.var(amplitude)
    assert frame_features.phase_variance == np.var(phase)
    assert frame_features.motion_energy == 0.0

    window_amplitude = np.array([[[1.0, 2.0]], [[2.0, 4.0]], [[4.0, 8.0]]])
    window_phase = np.zeros_like(window_amplitude)
    window_features = extract_window_features(window_amplitude, window_phase)

    assert window_features.mean_amplitude == np.mean(window_amplitude)
    assert window_features.motion_energy == 6.25


def test_subcarrier_variance_uses_sample_variance_across_time() -> None:
    data = np.array([[1.0, 1.0, 10.0], [2.0, 1.0, 12.0], [4.0, 1.0, 14.0]])

    np.testing.assert_allclose(subcarrier_variance(data), np.var(data, axis=0, ddof=1))

    tensor = np.stack([data, data * np.array([2.0, 1.0, 0.5])], axis=1)
    expected = np.mean(np.var(tensor, axis=0, ddof=1), axis=0)
    np.testing.assert_allclose(subcarrier_variance(tensor), expected)


def test_csi_window_stacks_csi_frames_for_signal_features() -> None:
    metadata = CsiMetadata(
        "node-1",
        "band_5_ghz",
        36,
        timestamp=Timestamp(1_700_000_000, 0),
        sequence_number=7,
    )
    frame_1 = CsiFrame(metadata, np.array([[1 + 0j, 0 + 1j], [3 + 4j, -1 - 1j]]))
    metadata.sequence_number = 8
    metadata.timestamp = Timestamp(1_700_000_000, 1)
    frame_2 = CsiFrame(metadata, np.array([[2 + 0j, 0 + 2j], [6 + 8j, -2 - 2j]]))

    window = stack_csi_frames([frame_1, frame_2])

    assert isinstance(window, CsiWindow)
    assert window.shape == (2, 2, 2)
    np.testing.assert_allclose(window.amplitude[0], frame_1.amplitude)
    np.testing.assert_allclose(window.phase[1], frame_2.phase)
    np.testing.assert_array_equal(window.sequence_numbers, [7, 8])
    assert extract_window_features(window).motion_energy > 0.0
    np.testing.assert_allclose(
        subcarrier_variance(window.amplitude),
        np.var(window.amplitude, axis=0, ddof=1).mean(axis=0),
    )


def test_csi_window_rejects_mismatched_frame_shapes() -> None:
    metadata = CsiMetadata("node-1", "band_5_ghz", 36)
    frame_1 = CsiFrame(metadata, np.ones((1, 2), dtype=np.complex128))
    frame_2 = CsiFrame(metadata, np.ones((1, 3), dtype=np.complex128))

    with pytest.raises(ValueError, match="matching data shapes"):
        CsiWindow.from_frames([frame_1, frame_2])
