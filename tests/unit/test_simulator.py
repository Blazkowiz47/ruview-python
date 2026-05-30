from __future__ import annotations

import numpy as np

from ruview.core import CsiFrame, FrequencyBand, Timestamp
from ruview.hardware import (
    SyntheticCsiConfig,
    generate_synthetic_frame,
    generate_synthetic_sequence,
    generate_synthetic_window,
    load_synthetic_fixture,
    save_synthetic_fixture,
)


def test_synthetic_csi_is_deterministic_by_seed() -> None:
    config = SyntheticCsiConfig(seed=7, frames=6, streams=2, subcarriers=12)

    first = generate_synthetic_sequence("motion", config)
    second = generate_synthetic_sequence("walking", config)

    for left, right in zip(first, second, strict=True):
        np.testing.assert_allclose(left.data, right.data)
        assert str(left.id) == str(right.id)
        assert left.witness_hash() == right.witness_hash()
        assert left.metadata.timestamp == right.metadata.timestamp

    changed_seed = generate_synthetic_sequence("walking", SyntheticCsiConfig(seed=8, frames=6))
    assert not np.allclose(first[0].data, changed_seed[0].data[:2, :12])


def test_synthetic_frame_shape_and_metadata() -> None:
    config = SyntheticCsiConfig(
        seed=11,
        streams=4,
        subcarriers=18,
        frames=5,
        sample_rate_hz=10.0,
        device_id="sim-test-node",
        channel=44,
        start_time_seconds=1_800_000_000,
    )

    frame = generate_synthetic_frame("empty_room", config, frame_index=3)

    assert isinstance(frame, CsiFrame)
    assert frame.data.shape == (4, 18)
    assert frame.num_spatial_streams() == 4
    assert frame.num_subcarriers() == 18
    assert str(frame.metadata.device_id) == "sim-test-node"
    assert frame.metadata.frequency_band == FrequencyBand.BAND_5_GHZ
    assert frame.metadata.channel == 44
    assert frame.metadata.sequence_number == 3
    assert frame.metadata.timestamp == Timestamp(1_800_000_000, 300_000_000)
    assert frame.metadata.antenna_config.spatial_streams() == 4
    assert frame.metadata.rssi_dbm == config.rssi_dbm - 4


def test_scenarios_have_expected_amplitude_and_motion_differences() -> None:
    config = SyntheticCsiConfig(
        seed=21,
        frames=96,
        streams=3,
        subcarriers=32,
        noise_std=0.0,
        phase_noise_std=0.0,
    )

    empty = generate_synthetic_window("empty", config)
    present = generate_synthetic_window("person_present", config)
    stillness = generate_synthetic_window("stillness", config)
    walking = generate_synthetic_window("walking", config)

    empty_mean = float(np.mean(np.abs(empty)))
    present_mean = float(np.mean(np.abs(present)))
    assert present_mean > empty_mean * 1.18

    empty_motion = _temporal_amplitude_variance(empty)
    stillness_motion = _temporal_amplitude_variance(stillness)
    walking_motion = _temporal_amplitude_variance(walking)
    assert stillness_motion > empty_motion * 5.0
    assert walking_motion > stillness_motion * 10.0


def test_synthetic_fixture_npz_roundtrip(tmp_path) -> None:
    config = SyntheticCsiConfig(seed=99, frames=4, streams=2, subcarriers=10)
    frames = generate_synthetic_sequence("still", config)
    path = tmp_path / "stillness_fixture.npz"

    saved_path = save_synthetic_fixture(path, frames, scenario="stillness", config=config)
    loaded = load_synthetic_fixture(saved_path)

    assert saved_path == path
    assert loaded.metadata["schema"] == "ruview.synthetic_csi.v1"
    assert loaded.metadata["scenario"] == "stillness"
    assert loaded.window().shape == (4, 2, 10)
    for original, roundtripped in zip(frames, loaded.frames, strict=True):
        np.testing.assert_allclose(original.data, roundtripped.data)
        assert str(original.id) == str(roundtripped.id)
        assert original.metadata.sequence_number == roundtripped.metadata.sequence_number
        assert original.metadata.timestamp == roundtripped.metadata.timestamp


def _temporal_amplitude_variance(window: np.ndarray) -> float:
    return float(np.mean(np.var(np.abs(window), axis=0)))
