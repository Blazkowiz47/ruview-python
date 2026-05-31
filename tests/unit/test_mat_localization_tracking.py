from __future__ import annotations

import math

import numpy as np
import pytest

from ruview.mat.localization import (
    DistanceEstimate,
    EstimateSource,
    LocationUncertainty,
    PositionEstimate,
    PositionFuser,
    RangeConstraint,
    RangeConstraintFuser,
    SensorPosition,
    TriangulationConfig,
    Triangulator,
    rssi_to_distance,
    toa_to_distance,
    validate_range_constraints,
)
from ruview.mat.tracking import (
    CsiFingerprint,
    DetectionObservation,
    KalmanState,
    SurvivorTracker,
    TrackState,
    TrackerConfig,
    fingerprint_similarity,
)


def test_distance_conversion_helpers() -> None:
    assert rssi_to_distance(-30.0) == pytest.approx(1.0)
    assert rssi_to_distance(-60.0) > 1.0

    toa_ns = 2.0 * 5.0 / 299_792_458.0 * 1e9
    assert toa_to_distance(toa_ns) == pytest.approx(5.0)


def test_triangulation_estimates_known_point() -> None:
    sensors = _sensors()
    target = np.array([3.0, 4.0], dtype=np.float64)
    distances = [
        DistanceEstimate(sensor.id, float(np.linalg.norm(target - np.array(sensor.xy))), uncertainty_m=0.15)
        for sensor in sensors
    ]

    estimate = Triangulator().trilaterate(sensors, distances)

    assert estimate is not None
    assert np.linalg.norm(np.array([estimate.x, estimate.y]) - target) < 0.05
    assert estimate.uncertainty.horizontal_error < 0.05
    assert estimate.uncertainty.gdop >= 1.0


def test_triangulation_rejects_insufficient_sensors_and_impossible_uncertainty() -> None:
    sensors = _sensors()
    triangulator = Triangulator(TriangulationConfig(max_uncertainty_m=0.5))

    assert triangulator.trilaterate(sensors[:2], {"s1": 1.0, "s2": 1.0}) is None

    impossible = [
        DistanceEstimate("s1", 1.0),
        DistanceEstimate("s2", 1.0),
        DistanceEstimate("s3", 20.0),
        DistanceEstimate("s4", 20.0),
    ]
    assert triangulator.trilaterate(sensors, impossible) is None


def test_range_constraint_validation_and_refinement() -> None:
    truth = np.array([2.0, 2.0, 0.0], dtype=np.float64)
    anchors = [
        np.array([0.0, 0.0, 0.0]),
        np.array([4.0, 0.0, 0.0]),
        np.array([0.0, 4.0, 0.0]),
    ]
    constraints = [
        RangeConstraint(index, anchor, float(np.linalg.norm(truth - anchor)), uncertainty_m=0.3)
        for index, anchor in enumerate(anchors)
    ]
    constraints.append(RangeConstraint(99, [0.0, 0.0, 0.0], 50.0, uncertainty_m=0.2))

    validation = validate_range_constraints(truth, constraints, gate_sigma=3.0)
    assert not validation.consistent
    assert 99 in validation.rejected_anchor_ids
    assert constraints[0].is_consistent(truth)
    assert not constraints[-1].is_consistent(truth)

    result = RangeConstraintFuser().refine([1.5, 1.5, 0.0], constraints)
    assert np.linalg.norm(np.array(result.position) - truth) < 0.08
    assert 99 in result.rejected_anchors


def test_position_fusion_weights_lower_uncertainty_estimate() -> None:
    truth = np.array([2.0, 2.0, 0.0], dtype=np.float64)
    good = PositionEstimate(
        2.1,
        1.9,
        0.0,
        uncertainty=LocationUncertainty(0.25, 0.3, 0.95),
        source=EstimateSource.TIME_OF_ARRIVAL,
    )
    noisy = PositionEstimate(
        5.0,
        5.0,
        0.0,
        uncertainty=LocationUncertainty(4.0, 4.0, 0.5),
        source=EstimateSource.RSSI_TRIANGULATION,
    )

    fused = PositionFuser().fuse([good, noisy])

    assert fused is not None
    assert fused.distance_to(truth) < noisy.distance_to(truth)
    assert fused.distance_to(truth) < 0.5
    assert fused.uncertainty.horizontal_error < noisy.uncertainty.horizontal_error


def test_kalman_predict_update_tracks_constant_velocity() -> None:
    state = KalmanState([0.0, 0.0, 0.0], process_noise_var=0.1, obs_noise_var=0.5)

    for step in range(1, 6):
        state.predict(1.0)
        state.update([float(step), 0.0, 0.0])

    assert state.position[0] == pytest.approx(5.0, abs=0.7)
    assert state.velocity[0] > 0.4
    near = state.mahalanobis_distance_sq([5.0, 0.0, 0.0])
    far = state.mahalanobis_distance_sq([30.0, 0.0, 0.0])
    assert near < far


def test_tracker_creates_updates_loses_and_drops_tracks() -> None:
    config = TrackerConfig(
        birth_hits_required=1,
        max_active_misses=1,
        max_lost_age_s=0.5,
        gate_mahalanobis_sq=16.0,
        obs_noise_var=0.5,
        drop_terminal=True,
    )
    tracker = SurvivorTracker(config)

    first = tracker.update([_obs([0.0, 0.0, 0.0])], dt_s=1.0)
    assert len(first.born_track_ids) == 1
    track_id = first.born_track_ids[0]
    assert tracker.get_track(track_id).state == TrackState.ACTIVE

    second = tracker.update([_obs([0.8, 0.0, 0.0])], dt_s=1.0)
    assert second.matched_track_ids == (track_id,)
    assert tracker.get_track(track_id).position[0] > 0.0

    lost = tracker.update([], dt_s=1.0)
    assert lost.lost_track_ids == (track_id,)
    assert tracker.get_track(track_id).state == TrackState.LOST

    dropped = tracker.update([], dt_s=1.0)
    assert track_id in dropped.terminated_track_ids
    assert track_id in dropped.dropped_track_ids
    assert tracker.get_track(track_id) is None


def test_fingerprint_similarity_orders_close_before_far() -> None:
    base = CsiFingerprint(16.0, 0.7, 72.0, [1.0, 2.0, 0.0])
    close = CsiFingerprint(16.5, 0.68, 73.0, [1.2, 2.1, 0.0])
    medium = CsiFingerprint(20.0, 0.8, 80.0, [3.0, 4.0, 0.0])
    far = CsiFingerprint(8.0, 0.2, None, [15.0, 10.0, 0.0])

    assert fingerprint_similarity(base, close) > fingerprint_similarity(base, medium)
    assert fingerprint_similarity(base, medium) > fingerprint_similarity(base, far)
    assert base.distance(close) < base.distance(far)


def _sensors() -> list[SensorPosition]:
    return [
        SensorPosition("s1", 0.0, 0.0, 1.5),
        SensorPosition("s2", 8.0, 0.0, 1.5),
        SensorPosition("s3", 0.0, 6.0, 1.5),
        SensorPosition("s4", 8.0, 6.0, 1.5),
    ]


def _obs(position: list[float]) -> DetectionObservation:
    fingerprint = CsiFingerprint(16.0, 0.7, 72.0, position)
    return DetectionObservation(position=position, fingerprint=fingerprint, confidence=0.9, zone_id="zone-a")
