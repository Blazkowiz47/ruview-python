from __future__ import annotations

import math

import numpy as np

from ruview.ruvsense.cir import (
    CirConfig,
    active_subcarrier_indices,
    estimate_cir,
    select_sparse_taps,
)
from ruview.ruvsense.coherence import (
    coherence_components,
    coherence_score,
    consecutive_coherence,
)
from ruview.ruvsense.coherence_gate import (
    CoherenceGate,
    CoherenceGateConfig,
    GateDecisionType,
    adaptive_noise_multiplier,
    decide_gate,
)


def test_cir_recovers_synthetic_single_tap() -> None:
    config = _test_cir_config(tap_threshold_ratio=0.20, top_k=3)
    csi = _project_taps(config, [(9, 1.0 + 0.0j)])

    estimate = estimate_cir(csi, config)

    assert estimate.dominant_tap is not None
    assert abs(estimate.dominant_tap.bin - 9) <= 1
    assert estimate.dominant_tap.amplitude > 0.8
    assert estimate.selected_taps[0].bin == estimate.dominant_tap.bin
    assert estimate.tap_spacing_s > 0.0
    assert 0.0 <= estimate.dominant_tap_ratio <= 1.0


def test_cir_recovers_synthetic_multiple_taps() -> None:
    config = _test_cir_config(tap_threshold_ratio=0.18, top_k=5)
    tap_specs = [
        (7, 1.0 * np.exp(1j * 0.1)),
        (26, 0.62 * np.exp(-1j * 0.6)),
        (61, 0.42 * np.exp(1j * 1.1)),
    ]
    csi = _project_taps(config, tap_specs)

    estimate = estimate_cir(csi, config)
    selected_bins = [tap.bin for tap in estimate.selected_taps]

    assert abs(estimate.dominant_tap_idx - 7) <= 1
    assert _contains_bin(selected_bins, 7)
    assert _contains_bin(selected_bins, 26)
    assert _contains_bin(selected_bins, 61)
    assert estimate.rms_delay_spread_s > 0.0


def test_cir_rejects_weak_ghost_and_noise_taps() -> None:
    config = _test_cir_config(tap_threshold_ratio=0.25, top_k=4, noise_floor_multiplier=6.0)
    csi = _project_taps(
        config,
        [
            (12, 1.0 + 0.0j),
            (49, 0.04 * np.exp(1j * 0.8)),
        ],
        noise_std=0.01,
    )

    estimate = estimate_cir(csi, config)
    selected_bins = [tap.bin for tap in estimate.selected_taps]

    assert _contains_bin(selected_bins, 12)
    assert not _contains_bin(selected_bins, 49)
    assert all(tap.amplitude >= estimate.threshold_amplitude for tap in estimate.selected_taps)


def test_select_sparse_taps_honors_top_k_and_threshold() -> None:
    cir = np.zeros(16, dtype=np.complex128)
    cir[2] = 1.0
    cir[5] = 0.5j
    cir[9] = 0.1

    selected = select_sparse_taps(cir, threshold_ratio=0.2, top_k=1, tap_spacing_s=2e-9)

    assert len(selected) == 1
    assert selected[0].bin == 2
    assert math.isclose(selected[0].delay_ns, 4.0)
    assert math.isclose(selected[0].power, 1.0)


def test_coherence_orders_high_and_low_similarity() -> None:
    reference = _project_taps(_test_cir_config(), [(5, 1.0), (19, 0.5j)])
    high = reference * (1.01 * np.exp(1j * 0.02))
    rng = np.random.default_rng(123)
    low = rng.standard_normal(reference.size) + 1j * rng.standard_normal(reference.size)

    high_components = coherence_components(high, reference)
    low_score = coherence_score(low, reference)

    assert high_components.score > 0.95
    assert high_components.correlation > 0.99
    assert low_score < high_components.score
    assert low_score < 0.80


def test_coherence_handles_zero_inputs_without_nan() -> None:
    zeros = np.zeros(8, dtype=np.complex128)
    nonzero = np.ones(8, dtype=np.complex128)

    assert coherence_score(zeros, zeros) == 1.0
    assert coherence_score(zeros, nonzero) == 0.0


def test_consecutive_coherence_scores_windows() -> None:
    config = _test_cir_config()
    first = _project_taps(config, [(4, 1.0), (17, 0.3j)])
    second = first * np.exp(1j * 0.01)
    third = _project_taps(config, [(40, 1.0)])

    scores = consecutive_coherence((first, second, third))

    assert len(scores) == 2
    assert scores[0].score > scores[1].score


def test_gate_accept_predict_reject_and_recalibrate_decisions() -> None:
    gate = CoherenceGate(CoherenceGateConfig(max_stale_frames=3))

    accept = gate.evaluate(0.94, novelty_score=0.05, calibration_drift=0.1)
    predict = gate.evaluate(0.70, novelty_score=0.20)
    reject = gate.evaluate(0.30, novelty_score=0.20)
    recalibrate = gate.evaluate(0.93, calibration_drift=0.95)

    assert accept.action == GateDecisionType.ACCEPT
    assert accept.allows_update
    assert accept.noise_multiplier == 1.0
    assert predict.action == GateDecisionType.PREDICT_ONLY
    assert predict.noise_multiplier >= 3.0
    assert reject.action == GateDecisionType.REJECT
    assert reject.is_rejected
    assert recalibrate.action == GateDecisionType.RECALIBRATE
    assert recalibrate.requires_recalibration


def test_gate_recalibrates_after_consecutive_stale_frames() -> None:
    gate = CoherenceGate(CoherenceGateConfig(max_stale_frames=3))

    first = gate.evaluate(0.2)
    second = gate.evaluate(0.2)
    third = gate.evaluate(0.2)

    assert first.action == GateDecisionType.REJECT
    assert second.action == GateDecisionType.REJECT
    assert third.action == GateDecisionType.RECALIBRATE
    assert third.stale_frames == 3


def test_stateless_gate_and_adaptive_noise_helpers() -> None:
    decision = decide_gate(0.92, novelty_score=0.8)
    midpoint_noise = adaptive_noise_multiplier(0.675, accept=0.85, reject=0.5, max_inflation=3.0)

    assert decision.action == GateDecisionType.REJECT
    assert math.isclose(midpoint_noise, 2.0)


def _test_cir_config(**kwargs) -> CirConfig:
    defaults = {
        "delay_bins": 156,
        "window": "rectangular",
        "tap_threshold_ratio": 0.08,
        "top_k": 8,
        "noise_floor_multiplier": 4.0,
    }
    defaults.update(kwargs)
    return CirConfig.ht20(**defaults)


def _project_taps(
    config: CirConfig,
    taps: list[tuple[int, complex]],
    *,
    noise_std: float = 0.0,
) -> np.ndarray:
    indices = active_subcarrier_indices(config.num_active)
    csi = np.zeros(config.num_active, dtype=np.complex128)
    for bin_index, amplitude in taps:
        csi += complex(amplitude) * np.exp(
            -1j * 2.0 * math.pi * indices * int(bin_index) / config.delay_bins
        )
    if noise_std > 0.0:
        rng = np.random.default_rng(42)
        csi += noise_std * (
            rng.standard_normal(config.num_active) + 1j * rng.standard_normal(config.num_active)
        )
    return csi


def _contains_bin(bins: list[int], target: int, *, tolerance: int = 1) -> bool:
    return any(abs(bin_index - target) <= tolerance for bin_index in bins)
