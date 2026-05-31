from __future__ import annotations

from dataclasses import dataclass

import pytest

from ruview.mat.alerts import (
    AlertDispatcher,
    AlertGenerator,
    AlertStatus,
    ResolutionType,
)
from ruview.mat.triage import (
    Priority,
    TriageCalculator,
    TriageInput,
    TriageService,
    TriageStatus,
)


def test_priority_ordering_and_triage_mapping() -> None:
    assert Priority.CRITICAL < Priority.HIGH < Priority.MEDIUM < Priority.LOW
    assert Priority.from_triage(TriageStatus.IMMEDIATE) is Priority.CRITICAL
    assert Priority.from_triage(TriageStatus.DELAYED) is Priority.HIGH
    assert Priority.from_triage(TriageStatus.MINOR) is Priority.MEDIUM
    assert Priority.from_triage(TriageStatus.DECEASED) is Priority.LOW
    assert TriageStatus.UNKNOWN.priority > TriageStatus.DECEASED.priority


@pytest.mark.parametrize(
    "triage_input,expected_phrase",
    [
        (TriageInput(breathing=35.0, heart=86.0, movement="minimal", confidence=0.9), "above START"),
        (TriageInput(breathing=16.0, heart=28.0, movement="moving", confidence=0.9), "critically low"),
        (
            TriageInput(
                breathing=8.0,
                heart=70.0,
                movement="gross",
                confidence=0.9,
                movement_is_voluntary=True,
            ),
            "below START",
        ),
    ],
)
def test_critical_vital_signs_are_immediate(triage_input: TriageInput, expected_phrase: str) -> None:
    result = TriageCalculator.calculate(triage_input)

    assert result.status is TriageStatus.IMMEDIATE
    assert result.priority is Priority.CRITICAL
    assert any(expected_phrase in reason for reason in result.rationale)


def test_deceased_no_vitals_case() -> None:
    result = TriageCalculator.calculate(
        TriageInput(breathing=0.0, heart=0.0, movement="none", location=(4.0, 2.0, -1.5), confidence=0.92)
    )

    assert result.status is TriageStatus.DECEASED
    assert result.priority is Priority.LOW
    assert result.triage_input.location == (4.0, 2.0, -1.5)
    assert any("no breathing" in reason for reason in result.rationale)


def test_unknown_low_confidence_case() -> None:
    result = TriageCalculator.calculate(TriageInput(breathing=18.0, heart=72.0, movement="walking", confidence=0.12))

    assert result.status is TriageStatus.UNKNOWN
    assert result.priority is Priority.MEDIUM
    assert result.is_actionable is False
    assert any("confidence" in reason for reason in result.rationale)


def test_alert_generation_from_survivor_mapping_is_local_object() -> None:
    generator = AlertGenerator()
    generator.register_zone("zone-a", "North void")

    alert = generator.generate(
        {
            "survivor_id": "s-001",
            "zone_id": "zone-a",
            "vitals": {
                "breathing_rate_bpm": 34.0,
                "heart_rate_bpm": 118.0,
                "movement": "minimal",
                "confidence": 0.88,
            },
            "location": (1.0, 2.0, -0.5),
        }
    )

    assert alert.id
    assert alert.survivor_id == "s-001"
    assert alert.status is AlertStatus.PENDING
    assert alert.priority is Priority.CRITICAL
    assert alert.payload.triage_status is TriageStatus.IMMEDIATE
    assert alert.payload.location == (1.0, 2.0, -0.5)
    assert "North void" in alert.payload.title
    assert "Immediate rescue" in alert.payload.recommended_action


def test_local_dispatcher_acknowledge_resolve_lifecycle() -> None:
    dispatcher = AlertDispatcher()
    alert = dispatcher.generate_and_dispatch(
        {
            "survivor_id": "s-ack",
            "breathing": 20.0,
            "heart": 75.0,
            "movement": "fine",
            "confidence": 0.86,
        }
    )

    assert dispatcher.pending_count() == 1
    assert dispatcher.records[-1].action == "dispatch"

    acknowledged = dispatcher.acknowledge(alert.id, by="team-1")
    assert acknowledged.status is AlertStatus.ACKNOWLEDGED
    assert acknowledged.acknowledged_by == "team-1"
    assert dispatcher.pending_count() == 0
    assert dispatcher.open_alerts() == [acknowledged]

    resolved = dispatcher.resolve(alert.id, ResolutionType.RESCUED, by="team-1", notes="extracted")
    assert resolved.status is AlertStatus.RESOLVED
    assert resolved.resolution is not None
    assert resolved.resolution.resolution_type is ResolutionType.RESCUED
    assert dispatcher.open_alerts() == []
    assert [record.action for record in dispatcher.records] == ["dispatch", "acknowledge", "resolve"]


@dataclass(frozen=True)
class SurvivorLike:
    survivor_id: str
    zone_id: str
    breathing_rate_bpm: float | None
    heart_rate_bpm: float | None
    movement: str
    confidence: float


def test_triage_service_batch_sorting_uses_triage_order() -> None:
    service = TriageService()
    assessments = service.triage_batch(
        [
            {
                "survivor_id": "unknown",
                "breathing": 18.0,
                "heart": 72.0,
                "movement": "walking",
                "confidence": 0.1,
            },
            SurvivorLike("delayed", "z", 18.0, 80.0, "fine", 0.9),
            {"survivor_id": "deceased", "breathing": 0.0, "heart": 0.0, "movement": "none", "confidence": 0.9},
            {"survivor_id": "minor", "breathing": 18.0, "heart": 72.0, "movement": "walking", "confidence": 0.9},
            {"survivor_id": "immediate", "breathing": 36.0, "heart": 110.0, "movement": "minimal", "confidence": 0.9},
        ]
    )

    assert [assessment.survivor_id for assessment in assessments] == [
        "immediate",
        "delayed",
        "minor",
        "deceased",
        "unknown",
    ]
    assert [assessment.status for assessment in assessments] == [
        TriageStatus.IMMEDIATE,
        TriageStatus.DELAYED,
        TriageStatus.MINOR,
        TriageStatus.DECEASED,
        TriageStatus.UNKNOWN,
    ]
