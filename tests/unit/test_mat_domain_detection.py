from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from ruview.mat.detection import (
    BreathingDetector,
    BreathingDetectorConfig,
    DetectionConfig,
    DetectionPipeline,
    EnsembleClassifier,
    EnsembleConfig,
    HeartbeatDetector,
    HeartbeatDetectorConfig,
    MovementClassifier,
)
from ruview.mat.domain import (
    BreathingPattern,
    BreathingType,
    Coordinates3D,
    DisasterEvent,
    DisasterType,
    EventStatus,
    HeartbeatSignature,
    MovementActivity,
    MovementType,
    ScanZone,
    SignalStrength,
    Survivor,
    SurvivorCondition,
    SurvivorStatus,
    VitalSignsReading,
    ZoneBounds,
    ZoneStatus,
)


def test_zone_containment_area_and_progress() -> None:
    zone = ScanZone(
        name="North void",
        bounds=ZoneBounds.rectangle(0.0, 0.0, 10.0, 5.0),
        id="zone-north",
        expected_scans=4,
    )

    assert zone.area() == pytest.approx(50.0)
    assert zone.contains_point(3.0, 4.0)
    assert zone.contains_point(10.0, 5.0)
    assert not zone.contains_point(11.0, 1.0)
    assert zone.progress_fraction() == 0.0

    scan_time = datetime(2026, 5, 31, 9, 30, tzinfo=UTC)
    zone.record_scan(found_detections=2, at=scan_time)

    assert zone.last_scan == scan_time
    assert zone.scan_count == 1
    assert zone.detections_count == 2
    assert zone.progress_fraction() == pytest.approx(0.25)

    zone.complete()
    assert zone.status is ZoneStatus.COMPLETE
    assert zone.progress_fraction() == 1.0


def test_survivor_vital_updates_refresh_condition_and_confidence() -> None:
    first = _reading(
        breathing=BreathingPattern(16.0, amplitude=0.8, regularity=0.9),
        movement=MovementActivity(MovementType.PERIODIC, intensity=0.2, frequency=0.25),
    )
    survivor = Survivor(
        zone_id="zone-a",
        initial_vitals=first,
        location=Coordinates3D(1.0, 2.0, -0.5),
        id="survivor-a",
    )

    assert survivor.status is SurvivorStatus.ACTIVE
    assert survivor.condition is SurvivorCondition.MINOR
    assert len(survivor.vital_signs) == 1
    assert survivor.confidence == pytest.approx(first.confidence.value)
    assert survivor.location is not None and survivor.location.is_buried()

    second = _reading(
        breathing=BreathingPattern(
            8.0,
            amplitude=0.5,
            regularity=0.8,
            pattern_type=BreathingType.SHALLOW,
        )
    )
    survivor.update_vitals(second)
    survivor.update_location(Coordinates3D(1.5, 2.0, -1.0))

    assert survivor.condition is SurvivorCondition.IMMEDIATE
    assert len(survivor.vital_signs) == 2
    assert survivor.vital_signs.latest() is second
    assert survivor.confidence == pytest.approx(
        (first.confidence.value + second.confidence.value) / 2.0
    )
    assert survivor.location is not None
    assert survivor.location.depth == pytest.approx(1.0)


def test_disaster_event_aggregate_adds_zones_and_updates_nearby_survivors() -> None:
    event = DisasterEvent(
        event_type=DisasterType.EARTHQUAKE,
        location=(10.0, 59.0),
        description="deterministic MAT test",
        id="event-a",
    )
    zone = ScanZone("Zone A", ZoneBounds.circle(0.0, 0.0, 5.0), id="zone-a")

    event.add_zone(zone)

    assert event.status is EventStatus.ACTIVE
    assert event.zone_count == 1
    assert event.get_zone("zone-a") is zone

    first = event.record_detection(
        "zone-a",
        _reading(breathing=BreathingPattern(16.0, 0.9, 0.9)),
        Coordinates3D(1.0, 1.0, -0.5),
    )
    second = event.record_detection(
        "zone-a",
        _reading(breathing=BreathingPattern(18.0, 0.8, 0.9)),
        Coordinates3D(1.4, 1.0, -0.6),
    )
    far = event.record_detection(
        "zone-a",
        _reading(movement=MovementActivity(MovementType.GROSS, intensity=0.7, frequency=0.1)),
        Coordinates3D(4.5, 0.0, -0.2),
    )
    far.mark_rescued()

    assert first.id == second.id
    assert event.survivor_count == 2
    assert len(first.vital_signs) == 2
    assert zone.detections_count == 2
    counts = event.survivor_counts()
    assert counts.active == 1
    assert counts.rescued == 1
    assert counts.living == 2

    event.close()
    assert event.status is EventStatus.CLOSED


def test_breathing_detector_detects_deterministic_sine_wave() -> None:
    sample_rate = 50.0
    duration = 24.0
    target_bpm = 18.0
    times = np.arange(int(sample_rate * duration)) / sample_rate
    signal = np.sin(2.0 * np.pi * (target_bpm / 60.0) * times)
    detector = BreathingDetector(
        BreathingDetectorConfig(
            window_size=512,
            min_amplitude=0.05,
            confidence_threshold=0.25,
        )
    )

    pattern = detector.detect(signal, sample_rate)

    assert pattern is not None
    assert pattern.rate_bpm == pytest.approx(target_bpm, abs=1.0)
    assert pattern.pattern_type is BreathingType.NORMAL
    assert pattern.regularity > 0.75
    assert pattern.confidence() > 0.65


def test_heartbeat_detector_detects_deterministic_phase_sine_wave() -> None:
    sample_rate = 100.0
    duration = 12.0
    target_bpm = 72.0
    times = np.arange(int(sample_rate * duration)) / sample_rate
    phase = 0.45 * np.sin(2.0 * np.pi * (target_bpm / 60.0) * times)
    phase += 0.06 * np.sin(4.0 * np.pi * (target_bpm / 60.0) * times)
    detector = HeartbeatDetector(
        HeartbeatDetectorConfig(
            window_size=512,
            min_signal_strength=0.03,
            confidence_threshold=0.35,
        )
    )

    signature = detector.detect(phase, sample_rate)

    assert signature is not None
    assert signature.rate_bpm == pytest.approx(target_bpm, abs=2.0)
    assert signature.is_normal_rate()
    assert signature.strength in {SignalStrength.MODERATE, SignalStrength.STRONG}


def test_movement_classifier_labels_static_gross_and_periodic_motion() -> None:
    classifier = MovementClassifier.with_defaults()
    sample_rate = 100.0

    static = classifier.classify(np.ones(200), sample_rate)
    assert static.movement_type is MovementType.NONE
    assert static.intensity == 0.0

    gross_signal = np.zeros(240)
    gross_signal[60:120] = 2.0
    gross_signal[170:210] = -1.5
    gross = classifier.classify(gross_signal, sample_rate)
    assert gross.movement_type is MovementType.GROSS
    assert gross.intensity > static.intensity

    times = np.arange(500) / sample_rate
    periodic_signal = 0.35 * np.sin(2.0 * np.pi * 0.4 * times)
    periodic = classifier.classify(periodic_signal, sample_rate)
    assert periodic.movement_type is MovementType.PERIODIC
    assert periodic.frequency == pytest.approx(0.4, abs=0.1)


def test_ensemble_confidence_and_condition_are_weighted() -> None:
    reading = _reading(
        breathing=BreathingPattern(16.0, amplitude=0.9, regularity=0.9),
        heartbeat=HeartbeatSignature(72.0, variability=0.1, strength=SignalStrength.MODERATE),
        movement=MovementActivity(MovementType.PERIODIC, intensity=0.4, frequency=0.3),
    )
    classifier = EnsembleClassifier(
        EnsembleConfig(
            breathing_weight=0.5,
            heartbeat_weight=0.3,
            movement_weight=0.2,
            min_ensemble_confidence=0.0,
        )
    )

    result = classifier.classify(reading)

    assert result.confidence == pytest.approx(0.81 * 0.5 + 0.7 * 0.3 + 0.5 * 0.2)
    assert result.recommended_condition is SurvivorCondition.MINOR
    assert result.breathing_detected
    assert result.heartbeat_detected
    assert result.movement_detected
    assert result.signal_confidences.breathing == pytest.approx(0.81)


def test_detection_pipeline_combines_detector_outputs() -> None:
    sample_rate = 100.0
    times = np.arange(int(sample_rate * 12.0)) / sample_rate
    amplitudes = np.sin(2.0 * np.pi * (18.0 / 60.0) * times)
    phases = 0.4 * np.sin(2.0 * np.pi * (72.0 / 60.0) * times)
    pipeline = DetectionPipeline(
        DetectionConfig(
            sample_rate_hz=sample_rate,
            enable_heartbeat=True,
            min_confidence=0.2,
            breathing=BreathingDetectorConfig(window_size=512, min_amplitude=0.05),
            heartbeat=HeartbeatDetectorConfig(
                window_size=512,
                min_signal_strength=0.03,
                confidence_threshold=0.35,
            ),
        )
    )

    result = pipeline.process(amplitudes, phases)

    assert result is not None
    assert result.reading.has_breathing()
    assert result.reading.has_heartbeat()
    assert result.ensemble.confidence >= pipeline.config.min_confidence


def _reading(
    *,
    breathing: BreathingPattern | None = None,
    heartbeat: HeartbeatSignature | None = None,
    movement: MovementActivity | None = None,
) -> VitalSignsReading:
    return VitalSignsReading(
        breathing=breathing,
        heartbeat=heartbeat,
        movement=movement or MovementActivity(),
    )
