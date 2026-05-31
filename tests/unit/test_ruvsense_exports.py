from __future__ import annotations

import ruview.ruvsense as rs


def test_milestone_8_public_exports_are_available() -> None:
    assert rs.CirConfig.ht20().num_active == 52
    assert rs.CoherenceGateConfig().accept_threshold > rs.CoherenceGateConfig().reject_threshold
    assert rs.MultiBandFusionConfig().length_strategy == "interpolate"
    assert rs.MultistaticFusionConfig().distance_reference_m == 1.0
    assert rs.FieldModelConfig(n_links=1, n_subcarriers=4).n_features == 4
    assert rs.NUM_KEYPOINTS == 17
    assert rs.GestureType.WAVE.value == "wave"
    assert rs.IntentLabel.LIKELY.value == "likely"
    assert rs.PhysicalAnomalyType.NON_FINITE.value == "non_finite"
    assert "LongitudinalWelfordStats" in rs.__all__
