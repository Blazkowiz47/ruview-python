from __future__ import annotations

import numpy as np

from ruview.ruvsense.field_model import FieldModel, FieldModelConfig
from ruview.ruvsense.pose_tracker import NUM_KEYPOINTS, PoseTracker, PoseTrackerConfig, TrackStatus
from ruview.ruvsense.tomography import LinkGeometry, RfTomographer, TomographyConfig, voxel_index


def test_field_model_residual_energy_increases_for_person_like_perturbation() -> None:
    rng = np.random.default_rng(1234)
    config = FieldModelConfig(n_links=3, n_subcarriers=8, n_modes=1, min_calibration_frames=48)
    base = _field_base(config)
    environmental = rng.normal(size=base.shape)
    environmental /= np.linalg.norm(environmental)

    frames = np.stack([base + 0.18 * np.sin(i * 0.37) * environmental for i in range(64)])
    model = FieldModel.fit(frames, config, calibrated_at_s=100.0)

    empty_like = base + 0.14 * environmental
    person_like = empty_like.copy()
    person_like[1, 5] += 1.0
    person_like[2, 2] -= 0.7

    empty = model.extract_perturbation(empty_like)
    person = model.extract_perturbation(person_like)

    assert model.modes is not None
    assert model.modes.n_modes == 1
    assert empty.total_energy < 0.05
    assert person.total_energy > empty.total_energy + 1.0
    assert person.energies[1] > empty.energies[1]


def test_pose_tracker_smooths_measurements_and_predicts_missing_frames() -> None:
    config = PoseTrackerConfig(
        birth_hits=1,
        assignment_max_distance=3.0,
        default_dt=1.0,
        min_keypoint_confidence=0.4,
    )
    tracker = PoseTracker(config)
    pose0 = _pose(0.0)
    pose1 = _pose(1.0)

    track = tracker.update([pose0], dt=1.0)[0]
    assert track.status == TrackStatus.ACTIVE
    first_centroid = track.centroid().copy()

    track = tracker.update([pose1], dt=1.0)[0]
    smoothed_centroid = track.centroid()
    raw_centroid = pose1[:, :3].mean(axis=0)

    assert first_centroid[0] < smoothed_centroid[0] < raw_centroid[0]
    assert float(np.mean(track.velocities[:, 0])) > 0.0

    predicted = tracker.update([], dt=1.0)[0]
    assert predicted.misses == 1
    assert predicted.centroid()[0] > smoothed_centroid[0]


def test_pose_tracker_ignores_low_confidence_keypoint_updates() -> None:
    tracker = PoseTracker(PoseTrackerConfig(birth_hits=1, assignment_max_distance=4.0))
    track = tracker.update([_pose(0.0)], dt=1.0)[0]
    before = track.positions[0].copy()

    noisy = _pose(0.1)
    noisy[0, :3] = [50.0, 50.0, 50.0]
    noisy[0, 3] = 0.05
    track = tracker.update([noisy], dt=1.0)[0]

    np.testing.assert_allclose(track.positions[0], before, atol=1e-6)
    assert track.confidences[0] < 1.0


def test_tomography_localizes_synthetic_attenuation_cell() -> None:
    config = TomographyConfig(
        nx=5,
        ny=5,
        nz=1,
        bounds=(0.0, 0.0, 0.0, 5.0, 5.0, 1.0),
        ray_radius=0.48,
        ridge=1e-4,
        min_links=8,
    )
    links = _perimeter_links()
    tomographer = RfTomographer(config, links)
    target = voxel_index(3, 2, 0, config)
    attenuations = tomographer.weights[:, target] * 2.0

    volume = tomographer.reconstruct(attenuations)

    assert volume.peak_index() == (3, 2, 0)
    assert volume.get(3, 2, 0) is not None
    assert volume.heatmap().shape == (5, 5)
    assert volume.residual < 0.05


def _field_base(config: FieldModelConfig) -> np.ndarray:
    links = np.arange(config.n_links, dtype=np.float64)[:, None]
    subcarriers = np.arange(config.n_subcarriers, dtype=np.float64)[None, :]
    return 4.0 + 0.2 * links + 0.03 * subcarriers


def _pose(offset_x: float) -> np.ndarray:
    keypoints = np.zeros((NUM_KEYPOINTS, 4), dtype=np.float64)
    keypoints[:, 0] = offset_x + np.linspace(0.0, 0.32, NUM_KEYPOINTS)
    keypoints[:, 1] = np.linspace(0.0, 0.16, NUM_KEYPOINTS)
    keypoints[:, 2] = 1.0
    keypoints[:, 3] = 1.0
    return keypoints


def _perimeter_links() -> list[LinkGeometry]:
    z = 0.5
    nodes = [
        (-0.5, 0.5, z),
        (-0.5, 2.5, z),
        (-0.5, 4.5, z),
        (5.5, 0.5, z),
        (5.5, 2.5, z),
        (5.5, 4.5, z),
        (0.5, -0.5, z),
        (2.5, -0.5, z),
        (4.5, -0.5, z),
        (0.5, 5.5, z),
        (2.5, 5.5, z),
        (4.5, 5.5, z),
    ]
    links: list[LinkGeometry] = []
    link_id = 0
    for i, tx in enumerate(nodes):
        for j, rx in enumerate(nodes):
            if i == j:
                continue
            links.append(LinkGeometry(tx, rx, link_id=link_id))
            link_id += 1
    return links
