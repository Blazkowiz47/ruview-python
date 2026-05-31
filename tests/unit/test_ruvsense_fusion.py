from __future__ import annotations

import numpy as np

from ruview.ruvsense.multiband import (
    MultiBandFusionConfig,
    MultiBandObservation,
    fuse_multiband,
)
from ruview.ruvsense.multistatic import (
    MultistaticFusionConfig,
    MultistaticObservation,
    fuse_multistatic,
)
from ruview.ruvsense.phase_align import align_lo_phase, estimate_lo_phase_offset, wrap_phase


def test_phase_offset_recovery_with_conjugate_inner_product() -> None:
    base_phase = np.linspace(-1.2, 1.1, 64)
    reference = (1.0 + np.linspace(0.0, 0.4, 64)) * np.exp(1j * base_phase)
    offset = 0.73
    target = reference * np.exp(1j * offset)

    estimate = estimate_lo_phase_offset(reference, target)
    aligned = align_lo_phase(reference, target)

    assert estimate.method == "inner_product"
    assert estimate.quality > 0.999
    assert abs(wrap_phase(estimate.offset_radians - offset)) < 1e-10
    np.testing.assert_allclose(aligned.aligned_csi, reference, atol=1e-10)


def test_phase_offset_falls_back_to_circular_mean_for_zero_energy() -> None:
    reference = np.zeros(8, dtype=np.complex128)
    target = np.ones(8, dtype=np.complex128)

    estimate = estimate_lo_phase_offset(reference, target)

    assert estimate.method == "empty"
    assert estimate.quality == 0.0
    assert estimate.offset_radians == 0.0


def test_multiband_weighted_fusion_aligns_lengths_and_normalizes() -> None:
    phase = np.linspace(0.0, np.pi / 2.0, 8)
    band_a = 2.0 * np.exp(1j * phase)
    band_b = 10.0 * np.exp(1j * np.linspace(0.0, np.pi / 2.0, 4))

    result = fuse_multiband(
        [
            MultiBandObservation(band_a, band_id="ch1", weight=1.0),
            MultiBandObservation(band_b, band_id="ch6", weight=3.0),
        ],
        config=MultiBandFusionConfig(target_length=8, normalize=True, phase_align=False),
    )

    assert result.fused_vector.shape == (8,)
    np.testing.assert_allclose(result.weights, [0.25, 0.75])
    assert [item.band_id for item in result.contributions] == ["ch1", "ch6"]
    assert all(item.aligned_length == 8 for item in result.contributions)
    assert np.isclose(np.sqrt(np.mean(np.abs(result.contributions[0].contribution / 0.25) ** 2)), 1.0)
    assert np.isclose(np.sqrt(np.mean(np.abs(result.contributions[1].contribution / 0.75) ** 2)), 1.0)


def test_multiband_truncate_strategy_uses_min_length_by_default() -> None:
    result = fuse_multiband(
        [np.arange(6, dtype=np.float64), np.arange(4, dtype=np.float64)],
        config=MultiBandFusionConfig(length_strategy="truncate", normalize=False, phase_align=False),
    )

    assert result.fused_vector.shape == (4,)
    np.testing.assert_allclose(result.weights, [0.5, 0.5])


def test_multistatic_attention_weights_sum_and_fuse_features() -> None:
    observations = [
        MultistaticObservation([1.0, 3.0], node_id="a", quality=0.9, coherence=0.8, distance_m=1.0),
        MultistaticObservation([5.0, 7.0], node_id="b", quality=0.3, coherence=0.7, distance_m=4.0),
        MultistaticObservation([9.0, 11.0], node_id="c", quality=0.1, coherence=0.2, distance_m=1.0),
    ]

    result = fuse_multistatic(observations, config=MultistaticFusionConfig(distance_reference_m=1.0))

    assert result.weights.shape == (3,)
    assert np.isclose(np.sum(result.weights), 1.0)
    assert result.weights[0] > result.weights[1] > result.weights[2]
    expected = sum(weight * np.asarray(obs.features) for weight, obs in zip(result.weights, observations, strict=True))
    np.testing.assert_allclose(result.fused_field, expected)
    assert result.weight_entropy < 1.0


def test_multistatic_zero_quality_links_do_not_create_nan_weights() -> None:
    result = fuse_multistatic(
        [
            MultistaticObservation([1.0, 1.0], node_id="good", quality=1.0, coherence=1.0),
            MultistaticObservation([100.0, 100.0], node_id="zero", quality=0.0, coherence=1.0),
            MultistaticObservation([-100.0, -100.0], node_id="low", quality=1e-6, coherence=1e-6),
        ]
    )

    assert np.isfinite(result.weights).all()
    assert np.isclose(np.sum(result.weights), 1.0)
    assert result.weights[0] > 0.999999
    assert result.weights[1] == 0.0
    np.testing.assert_allclose(result.fused_field, [1.0, 1.0], atol=3e-4)


def test_multistatic_all_zero_quality_falls_back_to_uniform_weights() -> None:
    result = fuse_multistatic(
        [
            MultistaticObservation([1.0, 2.0], quality=0.0, coherence=0.0),
            MultistaticObservation([3.0, 4.0], quality=0.0, coherence=0.0),
        ]
    )

    np.testing.assert_allclose(result.weights, [0.5, 0.5])
    np.testing.assert_allclose(result.fused_features, [2.0, 3.0])
