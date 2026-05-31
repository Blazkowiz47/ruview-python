from __future__ import annotations

import numpy as np

import ruview.ruvector as rv


def test_milestone_9_public_exports_are_available() -> None:
    sensitive, insensitive = rv.mincut_subcarrier_partition([0.1, 0.9, 0.2])

    assert sorted([*sensitive, *insensitive]) == [0, 1, 2]
    assert rv.gate_spectrogram(np.ones((2, 3))).shape == (2, 3)
    assert rv.attention_weighted_bvp([[0.0, 1.0]], [1.0]).shape == (2,)
    assert rv.solve_fresnel_geometry([(0.12, 1.0), (0.13, 0.8), (0.14, 0.6)], 5.0) is not None
    assert rv.solve_triangulation([], [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]) is None
    assert rv.CompressedBreathingHistory(n_subcarriers=4, capacity_frames=2).frame_count == 0
    assert "CompressedHeartbeatSpectrogram" in rv.__all__
