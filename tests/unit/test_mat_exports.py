from __future__ import annotations

import pytest

from ruview.mat import (
    AlertGenerator,
    BreathingPattern,
    Coordinates3D,
    DistanceEstimate,
    DomainLocationUncertainty,
    LocalizationSensorPosition,
    Priority,
    TriageInput,
    TriageStatus,
    rssi_to_distance,
)


def test_mat_package_exports_domain_localization_triage_and_alert_helpers() -> None:
    location = Coordinates3D(
        1.0,
        2.0,
        -0.5,
        uncertainty=DomainLocationUncertainty(horizontal_error=1.0, vertical_error=0.5),
    )
    sensor = LocalizationSensorPosition("sensor-a", 0.0, 0.0, 1.0)
    estimate = DistanceEstimate(sensor.id, rssi_to_distance(-30.0), uncertainty_m=0.25)
    triage_input = TriageInput(
        breathing=BreathingPattern(34.0, amplitude=0.8, regularity=0.7).rate_bpm,
        heart=112.0,
        movement="minimal",
        location=(location.x, location.y, location.z),
        confidence=0.9,
    )

    alert = AlertGenerator().generate(
        {
            "survivor_id": "s-1",
            "zone_id": "z-1",
            "vitals": {
                "breathing": triage_input.breathing,
                "heart": triage_input.heart,
                "movement": triage_input.movement,
                "confidence": triage_input.confidence,
            },
        }
    )

    assert estimate.distance_m == pytest.approx(1.0)
    assert alert.payload.triage_status is TriageStatus.IMMEDIATE
    assert alert.priority is Priority.CRITICAL
