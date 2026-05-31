from __future__ import annotations

import numpy as np

from ruview.ruvector.bvp import attention_weighted_bvp, bvp_attention_weights
from ruview.ruvector.spectrogram import gate_spectrogram
from ruview.ruvector.subcarrier import (
    mincut_subcarrier_partition,
    subcarrier_importance_weights,
)


def test_subcarrier_partition_covers_indices_and_names_higher_mean_sensitive() -> None:
    sensitivity = np.array([0.04, 0.05, 0.90, 0.82, 0.03, 0.06])

    sensitive, insensitive = mincut_subcarrier_partition(sensitivity)

    combined = sorted([*sensitive, *insensitive])
    assert combined == list(range(sensitivity.size))
    assert set(sensitive).isdisjoint(insensitive)
    assert np.mean(sensitivity[sensitive]) > np.mean(sensitivity[insensitive])
    assert set(sensitive) == {2, 3}


def test_subcarrier_partition_handles_empty_single_and_all_equal_inputs() -> None:
    assert mincut_subcarrier_partition([]) == ([], [])
    assert mincut_subcarrier_partition([0.5]) == ([0], [])

    sensitivity = np.ones(6)
    sensitive, insensitive = mincut_subcarrier_partition(sensitivity)

    assert sorted([*sensitive, *insensitive]) == list(range(6))
    assert sensitive
    assert insensitive
    assert np.mean(sensitivity[sensitive]) >= np.mean(sensitivity[insensitive])


def test_subcarrier_importance_weights_emphasize_sensitive_indices() -> None:
    sensitivity = np.array([0.04, 0.05, 0.90, 0.82, 0.03, 0.06])

    weights = subcarrier_importance_weights(sensitivity)

    assert weights.shape == sensitivity.shape
    assert np.all(np.isfinite(weights))
    assert np.mean(weights[[2, 3]]) > np.mean(weights[[0, 1, 4, 5]])
    assert subcarrier_importance_weights([]).size == 0
    assert subcarrier_importance_weights([1.0]).tolist() == [2.0]


def test_subcarrier_importance_weights_handle_all_equal_without_nan() -> None:
    weights = subcarrier_importance_weights(np.ones(4))

    assert weights.shape == (4,)
    assert np.all(np.isfinite(weights))
    assert np.min(weights) >= 0.5
    assert np.max(weights) <= 2.0


def test_gate_spectrogram_suppresses_low_energy_and_amplifies_motion_frames() -> None:
    spectrogram = np.array(
        [
            [0.10, 0.10, 0.10, 0.10],
            [0.20, 0.20, 0.20, 0.20],
            [5.00, 5.00, 5.00, 5.00],
            [0.25, 0.25, 0.25, 0.25],
            [0.10, 0.10, 0.10, 0.10],
        ],
        dtype=np.float64,
    )

    gated = gate_spectrogram(spectrogram, lambda_=0.4, tau=1)

    assert gated.shape == spectrogram.shape
    assert np.linalg.norm(gated[0]) < np.linalg.norm(spectrogram[0])
    assert np.linalg.norm(gated[2]) > np.linalg.norm(spectrogram[2])
    np.testing.assert_allclose(gated[1], spectrogram[1], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gated[3], spectrogram[3], rtol=1e-12, atol=1e-12)


def test_gate_spectrogram_accepts_flat_and_column_time_layouts() -> None:
    spectrogram = np.array(
        [
            [0.10, 0.10, 0.10],
            [3.00, 3.00, 3.00],
            [0.10, 0.10, 0.10],
        ],
        dtype=np.float64,
    )

    row_time = gate_spectrogram(spectrogram, lambda_=0.5, tau=0)
    flat = gate_spectrogram(spectrogram.ravel(), n_freq=3, n_time=3, lambda_=0.5, tau=0)
    column_time = gate_spectrogram(spectrogram.T, lambda_=0.5, tau=0, time_axis=1)

    np.testing.assert_allclose(flat.reshape(spectrogram.shape), row_time)
    np.testing.assert_allclose(column_time.T, row_time)


def test_attention_weighted_bvp_prefers_sensitivity_seeded_motion_row() -> None:
    rows = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    sensitivity = np.array([0.1, 3.0, 0.1])

    weights = bvp_attention_weights(rows, sensitivity)
    result = attention_weighted_bvp(rows, sensitivity)

    assert weights.shape == (3,)
    assert np.isclose(np.sum(weights), 1.0)
    assert weights[1] > weights[0]
    assert weights[1] > weights[2]
    assert result.shape == (4,)
    assert int(np.argmax(result)) == 2
    assert result[2] > 3.8


def test_attention_weighted_bvp_empty_input_returns_requested_zero_bins() -> None:
    result = attention_weighted_bvp([], [], n_velocity_bins=5)

    np.testing.assert_allclose(result, np.zeros(5))


def test_attention_weighted_bvp_truncates_to_requested_velocity_bins() -> None:
    rows = np.array([[1.0, 2.0, 9.0], [3.0, 4.0, 9.0]])

    result = attention_weighted_bvp(rows, [1.0, 1.0], n_velocity_bins=2)

    assert result.shape == (2,)
    assert np.all(result < 4.1)
