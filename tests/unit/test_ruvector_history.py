from __future__ import annotations

import numpy as np
import pytest

from ruview.ruvector.history import (
    CompressedBreathingBuffer,
    CompressedBreathingHistory,
    CompressedHeartbeatSpectrogram,
    RuVectorHistoryError,
    TierPolicy,
)


def test_breathing_history_frame_count_reconstruction_and_diagnostics() -> None:
    policy = TierPolicy(hot_frames=2, warm_frames=2, warm_bits=5, cold_bits=3)
    history = CompressedBreathingHistory(
        n_subcarriers=4,
        capacity_frames=6,
        policy=policy,
    )
    frames = [np.linspace(i, i + 0.3, 4) for i in range(3)]

    for frame in frames:
        history.push_frame(frame)

    assert history.frame_count == 3
    assert history.frame_count() == 3
    assert history.stored_frame_count == 3
    assert history.reconstruct().shape == (3, 4)
    np.testing.assert_allclose(history.reconstruct()[-1], frames[-1], atol=0.002)

    diagnostics = history.diagnostics()
    assert diagnostics.raw_bytes == 3 * 4 * 4
    assert diagnostics.encoded_bytes == history.byte_size()
    assert diagnostics.payload_bytes > 0
    assert diagnostics.compression_ratio > 0.0
    assert sum(diagnostics.tier_counts.values()) == 3
    assert all(item.payload_bytes > 0 for item in history.quantization_metadata())


def test_breathing_history_ring_overwrite_keeps_recent_frames() -> None:
    history = CompressedBreathingBuffer(n_subcarriers=4, capacity_frames=3)

    for value in range(5):
        history.push_frame(np.full(4, float(value)))

    assert history.frame_count == 5
    assert history.stored_frame_count == 3
    assert history.sequence_numbers == (2, 3, 4)
    np.testing.assert_allclose(
        history.to_array(),
        np.array(
            [
                [2.0, 2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0, 4.0],
            ]
        ),
    )
    assert len(history.to_vec()) == 12


def test_breathing_history_quantization_metadata_tracks_tiers() -> None:
    history = CompressedBreathingHistory(
        n_subcarriers=3,
        capacity_frames=4,
        policy=TierPolicy(hot_frames=1, warm_frames=1, hot_bits=8, warm_bits=5, cold_bits=3),
    )

    for value in range(4):
        history.push_frame(np.array([value, value + 1.0, value + 2.0]))

    metadata = history.quantization_metadata()
    assert [item.sequence_number for item in metadata] == [0, 1, 2, 3]
    assert [item.tier for item in metadata] == ["cold", "cold", "warm", "hot"]
    assert [item.bits for item in metadata] == [3, 3, 5, 8]


def test_breathing_history_rejects_invalid_shapes_and_values() -> None:
    history = CompressedBreathingHistory(n_subcarriers=4, capacity_frames=3)

    with pytest.raises(RuVectorHistoryError, match="shape"):
        history.push_frame(np.ones(3))
    with pytest.raises(RuVectorHistoryError, match="shape"):
        history.push_frame(np.ones((1, 4)))
    with pytest.raises(RuVectorHistoryError, match="finite"):
        history.push_frame(np.array([1.0, np.nan, 2.0, 3.0]))


def test_heartbeat_spectrogram_reconstruction_and_band_power() -> None:
    spectrogram = CompressedHeartbeatSpectrogram(
        n_freq_bins=4,
        capacity_columns=5,
        recent_window=2,
    )

    spectrogram.push_column(np.array([0.0, 1.0, 0.0, 0.0]))
    spectrogram.push_column(np.array([0.0, 2.0, 0.0, 0.0]))
    spectrogram.push_column(np.array([0.0, 3.0, 4.0, 0.0]))

    assert spectrogram.frame_count == 3
    assert spectrogram.reconstruct().shape == (3, 4)
    power = spectrogram.band_power(1, 2)
    assert np.isfinite(power)
    assert power > 0.0
    assert spectrogram.band_power(4, 6) == 0.0
    assert spectrogram.band_power(2, 1) == 0.0


def test_heartbeat_spectrogram_ring_overwrite_and_nonnegative_power() -> None:
    spectrogram = CompressedHeartbeatSpectrogram(
        n_freq_bins=3,
        capacity_columns=2,
        recent_window=10,
    )

    for value in range(4):
        spectrogram.push_column(np.full(3, float(value)))

    assert spectrogram.frame_count == 4
    assert spectrogram.stored_column_count == 2
    assert spectrogram.sequence_numbers == (2, 3)
    np.testing.assert_allclose(
        spectrogram.to_array(),
        np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]),
    )
    assert spectrogram.band_power(0, 2) == pytest.approx(6.5)
    assert spectrogram.band_power(1, 1) >= 0.0


def test_heartbeat_spectrogram_rejects_invalid_shapes_and_windows() -> None:
    spectrogram = CompressedHeartbeatSpectrogram(n_freq_bins=3, capacity_columns=2)

    with pytest.raises(RuVectorHistoryError, match="shape"):
        spectrogram.push_column(np.ones(2))
    with pytest.raises(RuVectorHistoryError, match="shape"):
        spectrogram.push_column(np.ones((3, 1)))
    with pytest.raises(RuVectorHistoryError, match="non-negative"):
        spectrogram.band_power(-1, 2)
    with pytest.raises(RuVectorHistoryError, match="positive"):
        spectrogram.band_power(0, 2, window_columns=0)
