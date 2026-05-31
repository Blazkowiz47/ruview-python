from __future__ import annotations

import numpy as np

from ruview.ruvsense.adversarial import (
    PhysicalAnomalyType,
    PhysicalCheckConfig,
    PhysicalImpossibilityChecker,
    Severity,
)
from ruview.ruvsense.cross_room import (
    CrossRoomConfig,
    CrossRoomTracker,
    EntryEvent,
    ExitEvent,
    RoomFingerprint,
)
from ruview.ruvsense.gesture import (
    GestureClassifier,
    GestureConfig,
    GestureTemplate,
    GestureType,
    dtw_distance,
)
from ruview.ruvsense.intention import (
    IntentLabel,
    IntentionConfig,
    IntentionDetector,
)
from ruview.ruvsense.longitudinal import (
    DAY_US,
    BiomechanicsMetric,
    DriftSeverity,
    LongitudinalConfig,
    LongitudinalMonitor,
    TrendDirection,
)


def test_dtw_template_classifier_recognizes_wave() -> None:
    frames = np.linspace(0.0, 2.0 * np.pi, 12)
    wave = np.column_stack([np.sin(frames), np.cos(frames)])
    push = np.column_stack([np.linspace(0.0, 1.0, 12), np.zeros(12)])
    query = wave + np.array([0.015, -0.01])

    classifier = GestureClassifier(
        GestureConfig(feature_dim=2, min_sequence_len=6, max_distance=1.0, band_width=3)
    )
    classifier.add_template(GestureTemplate("wave_template", GestureType.WAVE, wave))
    classifier.add_template(GestureTemplate("push_template", GestureType.PUSH, push))

    result = classifier.classify(query, person_id=7, timestamp_us=123_000)

    assert result.recognized
    assert result.gesture_type is GestureType.WAVE
    assert result.template_name == "wave_template"
    assert result.person_id == 7
    assert result.confidence > 0.5
    assert dtw_distance(wave, wave, band_width=2) == 0.0
    assert dtw_distance(wave, push, band_width=3) > result.distance


def test_intent_score_rises_before_motion_onset() -> None:
    detector = IntentionDetector(
        IntentionConfig(
            slope_threshold=0.05,
            energy_threshold=0.10,
            phase_lead_threshold=0.10,
            motion_onset_energy=0.50,
            min_sustained_frames=3,
        )
    )

    lead_signals = [
        detector.update(slope=0.02, energy=0.03, phase_lead=0.02, timestamp_us=0),
        detector.update(slope=0.12, energy=0.18, phase_lead=0.18, timestamp_us=50_000),
        detector.update(slope=0.14, energy=0.22, phase_lead=0.20, timestamp_us=100_000),
        detector.update(slope=0.16, energy=0.28, phase_lead=0.22, timestamp_us=150_000),
    ]

    assert lead_signals[0].label is IntentLabel.NONE
    assert lead_signals[-1].detected
    assert lead_signals[-1].label is IntentLabel.LIKELY
    assert lead_signals[-1].estimated_lead_time_s > 0.0

    motion = detector.update(slope=0.03, energy=0.70, phase_lead=0.05, timestamp_us=200_000)
    assert motion.label is IntentLabel.MOTION
    assert not motion.detected


def test_cross_room_fingerprint_and_transition_match() -> None:
    tracker = CrossRoomTracker(CrossRoomConfig(embedding_dim=4, min_similarity=0.85))
    tracker.register_room(RoomFingerprint("kitchen", [1.0, 0.0, 0.0, 0.0], node_count=2))
    tracker.register_room(RoomFingerprint("hall", [0.0, 1.0, 0.0, 0.0], node_count=2))

    room = tracker.match_room([0.02, 0.98, 0.0, 0.0])
    assert room.matched
    assert room.room_id == "hall"

    tracker.record_exit(
        ExitEvent("kitchen", track_id=101, embedding=[0.9, 0.1, 0.0, 0.0], timestamp_us=1_000_000)
    )
    result = tracker.match_entry(
        EntryEvent("hall", track_id=202, embedding=[0.88, 0.12, 0.01, 0.0], timestamp_us=6_000_000)
    )

    assert result.matched
    assert result.transition is not None
    assert result.transition.from_room == "kitchen"
    assert result.transition.to_room == "hall"
    assert result.transition.gap_s == 5.0
    assert result.best_similarity >= 0.85
    assert tracker.pending_exit_count == 0
    assert tracker.transition_count == 1


def test_longitudinal_monitor_detects_sustained_upward_trend() -> None:
    monitor = LongitudinalMonitor(
        LongitudinalConfig(
            min_baseline_points=7,
            min_sustained_points=3,
            drift_z_threshold=2.0,
            stable_slope_per_day=0.01,
        )
    )

    for day in range(7):
        update = monitor.update(BiomechanicsMetric.GAIT_SYMMETRY, 0.10, day * DAY_US)
        assert update.drift_report is None

    report = None
    for day, value in enumerate([0.50, 0.60, 0.70], start=7):
        update = monitor.update(BiomechanicsMetric.GAIT_SYMMETRY, value, day * DAY_US)
        report = update.drift_report

    assert report is not None
    assert report.metric is BiomechanicsMetric.GAIT_SYMMETRY
    assert report.direction is TrendDirection.INCREASING
    assert report.severity is DriftSeverity.ALERT
    assert report.sustained_points == 3
    assert update.summary.direction is TrendDirection.INCREASING
    assert update.summary.slope_per_day > 0.01


def test_adversarial_checker_flags_physical_impossibilities() -> None:
    config = PhysicalCheckConfig(
        max_relative_amplitude_jump=3.0,
        max_absolute_amplitude_jump=4.0,
        max_phase_velocity_rad_s=40.0,
        min_coherence=0.25,
        max_coherence_conflict_fraction=0.25,
    )

    checker = PhysicalImpossibilityChecker(config)
    clean = checker.check(np.ones(4), np.zeros(4), np.ones(4) * 0.95, timestamp_us=0)
    jump = checker.check(np.array([15.0, 1.0, 1.0, 1.0]), np.zeros(4), np.ones(4), timestamp_us=50_000)

    assert clean.valid
    assert not clean.anomaly_detected
    assert any(f.kind is PhysicalAnomalyType.AMPLITUDE_JUMP for f in jump.findings)
    assert jump.severity in {Severity.MEDIUM, Severity.HIGH}

    phase_checker = PhysicalImpossibilityChecker(config)
    phase_checker.check(np.ones(4), np.zeros(4), timestamp_us=0)
    phase = phase_checker.check(np.ones(4), np.ones(4) * np.pi, timestamp_us=10_000)
    assert any(f.kind is PhysicalAnomalyType.PHASE_VELOCITY for f in phase.findings)

    nonfinite = checker.check(np.array([1.0, np.nan, 1.0, 1.0]), timestamp_us=100_000)
    assert nonfinite.severity is Severity.CRITICAL
    assert any(f.kind is PhysicalAnomalyType.NON_FINITE for f in nonfinite.findings)

    coherence = PhysicalImpossibilityChecker(config).check(
        np.ones(4),
        np.zeros(4),
        np.array([0.0, 0.1, 0.2, 0.1]),
        timestamp_us=0,
    )
    assert any(f.kind is PhysicalAnomalyType.COHERENCE_CONFLICT for f in coherence.findings)
