from __future__ import annotations

import math

import numpy as np

from ruview.nvsim import (
    FLAG_ADC_SATURATED,
    FLAG_HEAVY_ATTENUATION,
    FLAG_NOISE_DISABLED,
    FLAG_SATURATION_NEAR_FIELD,
    MAG_FRAME_BYTES,
    MU_0,
    CurrentLoop,
    DipoleSource,
    FerrousObject,
    LosSegment,
    MagFrame,
    Material,
    Pipeline,
    PipelineConfig,
    Scene,
    attenuate_field,
    dipole_field,
)


def test_scene_canonical_json_roundtrip_and_source_count() -> None:
    scene = Scene(
        dipoles=(DipoleSource((0.0, 0.0, 0.5), (0.0, 0.0, 1.0e-3)),),
        loops=(CurrentLoop((0.0, 0.1, 0.3), (0.0, 1.0, 0.0), 0.05, 0.25),),
        ferrous=(FerrousObject.steel((0.2, 0.0, 0.0), 1.0e-4),),
        sensors=((0.0, 0.0, 0.0), (0.1, 0.0, 0.0)),
        ambient_field=(1.0e-6, 0.0, 0.0),
    )

    payload = scene.to_canonical_json()
    roundtripped = Scene.from_canonical_json(payload)

    assert roundtripped == scene
    assert roundtripped.source_count == 3
    assert roundtripped.n_sources() == 3
    assert roundtripped.to_canonical_json() == payload


def test_material_attenuation_and_heavy_flag() -> None:
    field = (1.0, 2.0, -3.0)

    concrete, concrete_heavy = attenuate_field(
        field,
        (LosSegment(Material.CONCRETE_DRY, 2.0),),
    )
    expected_scale = 10.0 ** (-1.0 / 20.0)
    np.testing.assert_allclose(concrete, np.asarray(field) * expected_scale)
    assert not concrete_heavy

    reinforced, reinforced_heavy = attenuate_field(
        field,
        (LosSegment("reinforced_concrete", 0.2),),
    )
    expected_reinforced_scale = 10.0 ** (-4.0 / 20.0)
    np.testing.assert_allclose(reinforced, np.asarray(field) * expected_reinforced_scale)
    assert reinforced_heavy


def test_dipole_field_direction_and_magnitude_sanity() -> None:
    moment = 1.0e-3
    distance = 0.5
    dipole = DipoleSource((0.0, 0.0, 0.0), (0.0, 0.0, moment))

    on_axis, on_axis_near = dipole_field(dipole, (0.0, 0.0, distance))
    equatorial, equatorial_near = dipole_field(dipole, (distance, 0.0, 0.0))

    expected_on_axis_z = MU_0 * moment / (2.0 * math.pi * distance**3)
    expected_equatorial_z = -MU_0 * moment / (4.0 * math.pi * distance**3)
    assert not on_axis_near
    assert not equatorial_near
    np.testing.assert_allclose(on_axis, (0.0, 0.0, expected_on_axis_z), rtol=1.0e-12)
    np.testing.assert_allclose(
        equatorial,
        (0.0, 0.0, expected_equatorial_z),
        rtol=1.0e-12,
    )
    assert on_axis[2] > 0.0
    assert equatorial[2] < 0.0
    assert abs(on_axis[2] / equatorial[2]) == 2.0


def test_pipeline_witness_is_seed_deterministic() -> None:
    scene = Scene(
        dipoles=(DipoleSource((0.0, 0.0, 0.5), (0.0, 0.0, 1.0e-3)),),
        sensors=((0.0, 0.0, 0.0),),
    )
    config = PipelineConfig(noise_std_t=2.0e-9)

    _, first = Pipeline(scene, config, seed=42).run_with_witness(16)
    _, second = Pipeline(scene, config, seed=42).run_with_witness(16)
    _, changed = Pipeline(scene, config, seed=43).run_with_witness(16)

    assert first == second
    assert first != changed
    assert len(first) == 32


def test_pipeline_frame_count_and_core_flags() -> None:
    scene = Scene(
        dipoles=(DipoleSource((0.0, 0.0, 0.005), (0.0, 0.0, 1.0)),),
        sensors=((0.0, 0.0, 0.0), (0.1, 0.0, 0.0)),
    )
    config = PipelineConfig(
        noise_enabled=False,
        los_segments=(LosSegment(Material.REINFORCED_CONCRETE, 0.1),),
    )

    frames = Pipeline(scene, config, seed=1).run(3)

    assert len(frames) == 6
    assert {frame.sensor_id for frame in frames} == {0, 1}
    assert all(frame.has_flag(FLAG_NOISE_DISABLED) for frame in frames)
    assert all(frame.has_flag(FLAG_HEAVY_ATTENUATION) for frame in frames)
    assert any(frame.has_flag(FLAG_ADC_SATURATED) for frame in frames)
    assert len(frames[0].to_canonical_bytes()) == MAG_FRAME_BYTES
    assert MagFrame.from_canonical_bytes(frames[0].to_canonical_bytes()) == frames[0]


def test_near_field_flag_clamps_unstable_dipole_model() -> None:
    scene = Scene(
        dipoles=(DipoleSource((0.0, 0.0, 5.0e-4), (0.0, 0.0, 1.0)),),
        sensors=((0.0, 0.0, 0.0),),
    )
    config = PipelineConfig(noise_enabled=False)

    frame = Pipeline(scene, config, seed=0).run(1)[0]

    assert frame.has_flag(FLAG_SATURATION_NEAR_FIELD)
    assert frame.has_flag(FLAG_NOISE_DISABLED)
    assert not frame.has_flag(FLAG_ADC_SATURATED)
    assert frame.b_pt == (0.0, 0.0, 0.0)
