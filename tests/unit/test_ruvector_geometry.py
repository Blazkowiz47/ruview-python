from __future__ import annotations

import math

import numpy as np
import pytest

from ruview.ruvector.geometry import (
    ViewpointPosition,
    angular_distance,
    compute_effective_viewpoints,
    estimate_crb,
    geometric_diversity_index,
    solve_fresnel_geometry,
)
from ruview.ruvector.triangulation import SPEED_OF_LIGHT_M_S, TriangulationError, solve_triangulation


def test_fresnel_geometry_requires_three_observations() -> None:
    assert solve_fresnel_geometry([(0.125, 0.3), (0.130, 0.25)], 5.0) is None


def test_fresnel_geometry_conserves_total_path_and_clamps_positive() -> None:
    d_total = 6.0
    wavelengths = np.array([0.115, 0.120, 0.125, 0.130, 0.135], dtype=np.float64)
    contrast_m = -2.0
    observations = [(float(wavelength), float(contrast_m / wavelength)) for wavelength in wavelengths]

    result = solve_fresnel_geometry(observations, d_total)

    assert result is not None
    assert result.n_observations == len(observations)
    assert result.d1 == pytest.approx(2.0, abs=2e-3)
    assert result.d2 == pytest.approx(4.0, abs=2e-3)
    assert result.d1 + result.d2 == pytest.approx(d_total, abs=1e-9)
    assert result.d1 > 0.0
    assert result.d2 > 0.0


def test_fresnel_geometry_clamps_extreme_solution_inside_link() -> None:
    result = solve_fresnel_geometry([(0.12, 1e6), (0.13, 1e6), (0.14, 1e6)], 5.0)

    assert result is not None
    assert result.d1 == pytest.approx(4.9)
    assert result.d2 == pytest.approx(0.1)
    assert result.path_error < 1e-12


def test_viewpoint_angular_distance_wraps_across_zero() -> None:
    assert angular_distance(0.1, 2.0 * math.pi - 0.1) == pytest.approx(0.2)


def test_geometric_diversity_uniform_layout_is_sufficient() -> None:
    azimuths = [0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0]

    gdi = geometric_diversity_index(azimuths, node_ids=[10, 20, 30, 40])

    assert gdi is not None
    assert gdi.value == pytest.approx(math.pi / 2.0)
    assert gdi.n_physical == 4
    assert gdi.is_sufficient()
    assert 0.0 < gdi.efficiency() <= 1.0


def test_geometric_diversity_flags_clustered_worst_pair_and_effective_count() -> None:
    gdi = geometric_diversity_index([0.0, 0.1, math.pi, 1.5 * math.pi], node_ids=[10, 20, 30, 40])

    assert gdi is not None
    assert gdi.value < math.pi / 2.0
    assert gdi.worst_pair in {(10, 20), (20, 10)}
    assert compute_effective_viewpoints([0.0, 0.0, 0.0, 0.0]) == pytest.approx(1.0, abs=0.1)


def test_crb_decreases_with_more_balanced_viewpoints() -> None:
    target = (0.0, 0.0)
    three = [
        ViewpointPosition(5.0 * math.cos(2.0 * math.pi * i / 3.0), 5.0 * math.sin(2.0 * math.pi * i / 3.0), 0.1)
        for i in range(3)
    ]
    six = [
        ViewpointPosition(5.0 * math.cos(2.0 * math.pi * i / 6.0), 5.0 * math.sin(2.0 * math.pi * i / 6.0), 0.1)
        for i in range(6)
    ]

    crb_three = estimate_crb(target, three)
    crb_six = estimate_crb(target, six)

    assert crb_three is not None
    assert crb_six is not None
    assert crb_six.rmse_lower_bound < crb_three.rmse_lower_bound
    assert crb_six.gdop == pytest.approx(crb_six.rmse_lower_bound)


def test_tdoa_triangulation_estimates_known_target() -> None:
    ap_positions = np.array(
        [
            [0.0, 0.0],
            [8.0, 0.0],
            [8.0, 6.0],
            [0.0, 6.0],
            [4.0, 10.0],
        ],
        dtype=np.float64,
    )
    target = np.array([3.2, 4.1], dtype=np.float64)
    measurements = _tdoa_pairs(ap_positions, target, [(1, 0), (2, 0), (3, 0), (4, 0), (2, 1), (3, 4)])

    result = solve_triangulation(measurements, ap_positions)

    assert result is not None
    assert result.converged
    assert result.n_measurements == len(measurements)
    assert np.linalg.norm(np.array(result.position) - target) < 0.05
    assert result.residual_m < 1e-5


def test_tdoa_triangulation_returns_none_for_too_few_measurements() -> None:
    ap_positions = [(0.0, 0.0), (5.0, 0.0), (0.0, 5.0)]

    assert solve_triangulation([(1, 0, 1e-9), (2, 0, 1e-9)], ap_positions) is None


def test_tdoa_triangulation_rejects_impossible_range_difference() -> None:
    ap_positions = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    impossible_tdoa = 2.0 / SPEED_OF_LIGHT_M_S

    with pytest.raises(TriangulationError):
        solve_triangulation([(1, 0, impossible_tdoa), (2, 0, 0.0), (2, 1, 0.0)], ap_positions)


def _tdoa_pairs(
    ap_positions: np.ndarray,
    target: np.ndarray,
    pairs: list[tuple[int, int]],
) -> list[tuple[int, int, float]]:
    distances = np.linalg.norm(ap_positions - target, axis=1)
    return [(i, j, float((distances[i] - distances[j]) / SPEED_OF_LIGHT_M_S)) for i, j in pairs]
