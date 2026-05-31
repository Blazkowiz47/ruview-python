"""Local-only MAT alert objects and in-memory dispatch recording."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
import math
from typing import Any
from uuid import uuid4

from ruview.mat.triage import (
    Priority,
    SurvivorTriage,
    TriageResult,
    TriageService,
    TriageStatus,
    coerce_survivor_snapshot,
)


class AlertError(ValueError):
    """Raised for invalid local alert lifecycle operations."""


class AlertStatus(str, Enum):
    """Lifecycle state for a local alert record."""

    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

    @property
    def is_open(self) -> bool:
        return self in {AlertStatus.PENDING, AlertStatus.ACKNOWLEDGED, AlertStatus.IN_PROGRESS}


class ResolutionType(str, Enum):
    """Local resolution labels for closed alerts."""

    RESCUED = "rescued"
    FALSE_POSITIVE = "false_positive"
    DECEASED = "deceased"
    SUPERSEDED = "superseded"
    TIMED_OUT = "timed_out"
    OTHER = "other"


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class AlertPayload:
    """Human-readable alert details derived from local triage state."""

    title: str
    message: str
    triage_status: TriageStatus
    location: Sequence[float] | None = None
    recommended_action: str = ""
    rationale: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    deadline: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "triage_status", _coerce_triage_status(self.triage_status))
        object.__setattr__(self, "location", _coerce_location(self.location))
        object.__setattr__(self, "rationale", tuple(str(item) for item in self.rationale))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        object.__setattr__(self, "deadline", _coerce_datetime(self.deadline))

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "title": self.title,
            "message": self.message,
            "triage_status": self.triage_status.value,
            "recommended_action": self.recommended_action,
            "rationale": list(self.rationale),
            "metadata": dict(self.metadata),
        }
        if self.location is not None:
            payload["location"] = list(self.location)
        if self.deadline is not None:
            payload["deadline"] = self.deadline.isoformat()
        return payload


@dataclass(frozen=True)
class AlertResolution:
    """Resolution details for a closed local alert."""

    resolution_type: ResolutionType = ResolutionType.OTHER
    notes: str = ""
    resolved_by: str | None = None
    resolved_at: datetime = field(default_factory=_utcnow)

    def __post_init__(self) -> None:
        object.__setattr__(self, "resolution_type", _coerce_resolution_type(self.resolution_type))
        object.__setattr__(self, "resolved_at", _coerce_datetime(self.resolved_at) or _utcnow())

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolution_type": self.resolution_type.value,
            "notes": self.notes,
            "resolved_by": self.resolved_by,
            "resolved_at": self.resolved_at.isoformat(),
        }


@dataclass(frozen=True)
class AlertDispatchRecord:
    """Audit event recorded by the local dispatcher."""

    alert_id: str
    action: str
    status: AlertStatus
    timestamp: datetime = field(default_factory=_utcnow)
    actor: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _coerce_alert_status(self.status))
        object.__setattr__(self, "timestamp", _coerce_datetime(self.timestamp) or _utcnow())


@dataclass
class Alert:
    """Local rescue alert record.

    The object intentionally has no networking, paging, SMS, MQTT, or external
    delivery hooks; dispatchers in this module only retain in-memory records.
    """

    survivor_id: str
    priority: Priority
    payload: AlertPayload
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=_utcnow)
    status: AlertStatus = AlertStatus.PENDING
    acknowledged_at: datetime | None = None
    acknowledged_by: str | None = None
    resolution: AlertResolution | None = None
    escalation_count: int = 0

    def __post_init__(self) -> None:
        self.survivor_id = str(self.survivor_id)
        self.priority = Priority(self.priority)
        self.status = _coerce_alert_status(self.status)
        self.timestamp = _coerce_datetime(self.timestamp) or _utcnow()
        self.acknowledged_at = _coerce_datetime(self.acknowledged_at)
        self.escalation_count = int(self.escalation_count)
        if self.escalation_count < 0:
            raise AlertError("escalation_count must be non-negative")

    @property
    def triage_status(self) -> TriageStatus:
        return self.payload.triage_status

    @property
    def is_pending(self) -> bool:
        return self.status is AlertStatus.PENDING

    @property
    def is_open(self) -> bool:
        return self.status.is_open

    def acknowledge(self, by: str = "local") -> None:
        if not self.status.is_open:
            raise AlertError(f"cannot acknowledge {self.status.value} alert")
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_at = _utcnow()
        self.acknowledged_by = str(by)

    def start(self, by: str | None = None) -> None:
        if not self.status.is_open:
            raise AlertError(f"cannot start {self.status.value} alert")
        if self.status is AlertStatus.PENDING:
            self.acknowledge(by or "local")
        self.status = AlertStatus.IN_PROGRESS

    def resolve(
        self,
        resolution_type: ResolutionType | str = ResolutionType.OTHER,
        *,
        by: str | None = None,
        notes: str = "",
    ) -> None:
        if not self.status.is_open:
            raise AlertError(f"cannot resolve {self.status.value} alert")
        self.status = AlertStatus.RESOLVED
        self.resolution = AlertResolution(
            resolution_type=_coerce_resolution_type(resolution_type),
            notes=notes,
            resolved_by=by,
        )

    def cancel(self, *, by: str | None = None, notes: str = "") -> None:
        if not self.status.is_open:
            raise AlertError(f"cannot cancel {self.status.value} alert")
        self.status = AlertStatus.CANCELLED
        self.resolution = AlertResolution(
            resolution_type=ResolutionType.SUPERSEDED,
            notes=notes,
            resolved_by=by,
        )

    def escalate(self) -> None:
        self.escalation_count += 1
        if self.priority > Priority.CRITICAL:
            self.priority = Priority(int(self.priority) - 1)

    def needs_escalation(self, timeout_seconds: float, *, now: datetime | None = None) -> bool:
        if not self.status.is_open:
            return False
        deadline = self.timestamp + timedelta(seconds=float(timeout_seconds))
        return (_coerce_datetime(now) or _utcnow()) >= deadline

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "survivor_id": self.survivor_id,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.name.lower(),
            "priority_value": int(self.priority),
            "status": self.status.value,
            "payload": self.payload.to_dict(),
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
            "resolution": self.resolution.to_dict() if self.resolution else None,
            "escalation_count": self.escalation_count,
        }


class AlertGenerator:
    """Generate local alert objects from survivor triage assessments."""

    def __init__(self, *, triage_service: TriageService | None = None) -> None:
        self.triage_service = triage_service or TriageService()
        self.zone_names: dict[str, str] = {}

    def register_zone(self, zone_id: str, name: str) -> None:
        self.zone_names[str(zone_id)] = str(name)

    def generate(
        self,
        survivor: SurvivorTriage | TriageResult | Mapping[str, Any] | Any,
        *,
        survivor_id: str | None = None,
        zone_id: str | None = None,
    ) -> Alert:
        assessment = self._assessment_for(survivor, survivor_id=survivor_id, zone_id=zone_id)
        payload = self.create_payload(assessment)
        return Alert(
            survivor_id=assessment.survivor_id,
            priority=assessment.priority,
            payload=payload,
        )

    def generate_escalation(
        self,
        survivor: SurvivorTriage | TriageResult | Mapping[str, Any] | Any,
        reason: str,
        *,
        survivor_id: str | None = None,
        zone_id: str | None = None,
    ) -> Alert:
        assessment = self._assessment_for(survivor, survivor_id=survivor_id, zone_id=zone_id)
        payload = self.create_payload(assessment)
        payload = AlertPayload(
            title=f"ESCALATED: {payload.title}",
            message=f"{payload.message}\n\nEscalation reason: {reason}",
            triage_status=payload.triage_status,
            location=payload.location,
            recommended_action=payload.recommended_action,
            rationale=payload.rationale,
            metadata=payload.metadata,
            deadline=payload.deadline,
        )
        priority = Priority.CRITICAL if assessment.status is TriageStatus.IMMEDIATE else Priority.HIGH
        return Alert(survivor_id=assessment.survivor_id, priority=priority, payload=payload)

    def create_payload(self, assessment: SurvivorTriage) -> AlertPayload:
        status = assessment.status
        zone_name = self.zone_names.get(str(assessment.zone_id), str(assessment.zone_id or "unknown zone"))
        title = f"{status.value.upper()} survivor detected - {zone_name}"
        location_line = _format_location(assessment.location)
        vital_line = _format_vitals(assessment.triage.triage_input)
        rationale = tuple(assessment.triage.rationale)
        rationale_text = "; ".join(rationale) if rationale else "no rationale recorded"
        message = (
            f"Survivor ID: {assessment.survivor_id}\n"
            f"Zone: {zone_name}\n"
            f"Triage: {status.value} ({status.color})\n"
            f"Confidence: {assessment.confidence:.0%}\n"
            f"Vitals: {vital_line}\n"
            f"Location: {location_line}\n"
            f"Rationale: {rationale_text}"
        )
        metadata = {
            "survivor_id": assessment.survivor_id,
            "triage_status": status.value,
            "confidence": f"{assessment.confidence:.3f}",
        }
        if assessment.zone_id is not None:
            metadata["zone_id"] = assessment.zone_id
        metadata.update({str(key): str(value) for key, value in assessment.metadata.items()})

        return AlertPayload(
            title=title,
            message=message,
            triage_status=status,
            location=assessment.location,
            recommended_action=recommended_action(status),
            rationale=rationale,
            metadata=metadata,
        )

    def _assessment_for(
        self,
        survivor: SurvivorTriage | TriageResult | Mapping[str, Any] | Any,
        *,
        survivor_id: str | None,
        zone_id: str | None,
    ) -> SurvivorTriage:
        if isinstance(survivor, SurvivorTriage):
            return survivor
        if isinstance(survivor, TriageResult):
            return SurvivorTriage(
                survivor_id=str(survivor_id or "unknown"),
                triage=survivor,
                zone_id=str(zone_id) if zone_id is not None else None,
            )

        snapshot = coerce_survivor_snapshot(survivor)
        if survivor_id is not None or zone_id is not None:
            snapshot = type(snapshot)(
                survivor_id=str(survivor_id or snapshot.survivor_id),
                triage_input=snapshot.triage_input,
                zone_id=str(zone_id or snapshot.zone_id) if zone_id is not None or snapshot.zone_id is not None else None,
                metadata=snapshot.metadata,
            )
        return self.triage_service.triage_survivor(snapshot)


class AlertDispatcher:
    """In-memory dispatcher that records alerts without external delivery."""

    def __init__(self, *, generator: AlertGenerator | None = None) -> None:
        self.generator = generator or AlertGenerator()
        self._alerts: dict[str, Alert] = {}
        self._records: list[AlertDispatchRecord] = []

    @property
    def records(self) -> tuple[AlertDispatchRecord, ...]:
        return tuple(self._records)

    @property
    def alerts(self) -> tuple[Alert, ...]:
        return tuple(sorted(self._alerts.values(), key=lambda alert: (int(alert.priority), alert.timestamp, alert.id)))

    def dispatch(self, alert: Alert) -> Alert:
        if not isinstance(alert, Alert):
            raise AlertError("dispatch expects an Alert")
        self._alerts[alert.id] = alert
        self._record(alert, "dispatch")
        return alert

    def generate_and_dispatch(
        self,
        survivor: SurvivorTriage | TriageResult | Mapping[str, Any] | Any,
        *,
        survivor_id: str | None = None,
        zone_id: str | None = None,
    ) -> Alert:
        return self.dispatch(self.generator.generate(survivor, survivor_id=survivor_id, zone_id=zone_id))

    def acknowledge(self, alert_id: str, by: str = "local") -> Alert:
        alert = self.get(alert_id)
        alert.acknowledge(by)
        self._record(alert, "acknowledge", actor=by)
        return alert

    def start(self, alert_id: str, by: str | None = None) -> Alert:
        alert = self.get(alert_id)
        alert.start(by=by)
        self._record(alert, "start", actor=by)
        return alert

    def resolve(
        self,
        alert_id: str,
        resolution_type: ResolutionType | str = ResolutionType.OTHER,
        *,
        by: str | None = None,
        notes: str = "",
    ) -> Alert:
        alert = self.get(alert_id)
        alert.resolve(resolution_type, by=by, notes=notes)
        self._record(alert, "resolve", actor=by, notes=notes)
        return alert

    def get(self, alert_id: str) -> Alert:
        try:
            return self._alerts[str(alert_id)]
        except KeyError as exc:
            raise AlertError(f"alert {alert_id!r} not found") from exc

    def pending(self) -> list[Alert]:
        return [alert for alert in self.alerts if alert.status is AlertStatus.PENDING]

    def open_alerts(self) -> list[Alert]:
        return [alert for alert in self.alerts if alert.status.is_open]

    def pending_by_priority(self, priority: Priority | int) -> list[Alert]:
        priority = Priority(priority)
        return [alert for alert in self.pending() if alert.priority is priority]

    def pending_count(self) -> int:
        return len(self.pending())

    def escalate_timed_out(self, timeout_seconds: float, *, now: datetime | None = None) -> int:
        escalated = 0
        for alert in self.open_alerts():
            if alert.needs_escalation(timeout_seconds, now=now):
                alert.escalate()
                self._record(alert, "escalate")
                escalated += 1
        return escalated

    def _record(
        self,
        alert: Alert,
        action: str,
        *,
        actor: str | None = None,
        notes: str = "",
    ) -> None:
        self._records.append(
            AlertDispatchRecord(
                alert_id=alert.id,
                action=action,
                status=alert.status,
                actor=actor,
                notes=notes,
            )
        )


def recommended_action(status: TriageStatus | str) -> str:
    """Return the local recommended action text for a triage status."""

    status = _coerce_triage_status(status)
    return {
        TriageStatus.IMMEDIATE: (
            "Immediate rescue required. Prioritize extraction and prepare airway/critical-care support."
        ),
        TriageStatus.DELAYED: (
            "Rescue team required. Mark location and monitor for deterioration while immediate cases are handled."
        ),
        TriageStatus.MINOR: "Lower priority. Guide to extraction if conscious and mobile.",
        TriageStatus.DECEASED: "Mark location for recovery and document the finding.",
        TriageStatus.UNKNOWN: "Repeat assessment with additional sensing coverage before escalation.",
    }[status]


def _format_vitals(triage_input: Any) -> str:
    breathing = "not detected" if triage_input.breathing is None else f"{triage_input.breathing:.1f} BPM breathing"
    heart = "not detected" if triage_input.heart is None else f"{triage_input.heart:.1f} BPM heart"
    movement = triage_input.movement or "unknown movement"
    return f"{breathing}, {heart}, {movement}"


def _format_location(location: Sequence[float] | None) -> str:
    if location is None:
        return "not determined"
    return "(" + ", ".join(f"{value:.2f}" for value in location) + ")"


def _coerce_triage_status(status: TriageStatus | str) -> TriageStatus:
    if isinstance(status, TriageStatus):
        return status
    try:
        return TriageStatus(str(status).strip().lower())
    except ValueError as exc:
        raise AlertError(f"unknown triage status {status!r}") from exc


def _coerce_alert_status(status: AlertStatus | str) -> AlertStatus:
    if isinstance(status, AlertStatus):
        return status
    try:
        return AlertStatus(str(status).strip().lower())
    except ValueError as exc:
        raise AlertError(f"unknown alert status {status!r}") from exc


def _coerce_resolution_type(resolution_type: ResolutionType | str) -> ResolutionType:
    if isinstance(resolution_type, ResolutionType):
        return resolution_type
    try:
        return ResolutionType(str(resolution_type).strip().lower())
    except ValueError as exc:
        raise AlertError(f"unknown resolution type {resolution_type!r}") from exc


def _coerce_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, datetime):
        raise AlertError("timestamp values must be datetime objects")
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _coerce_location(location: Sequence[float] | None) -> tuple[float, ...] | None:
    if location is None:
        return None
    values = tuple(float(value) for value in location)
    if len(values) not in {2, 3}:
        raise AlertError("location must contain two or three coordinates")
    if not all(math.isfinite(value) for value in values):
        raise AlertError("location coordinates must be finite")
    return values


__all__ = [
    "Alert",
    "AlertDispatchRecord",
    "AlertDispatcher",
    "AlertError",
    "AlertGenerator",
    "AlertPayload",
    "AlertResolution",
    "AlertStatus",
    "Priority",
    "ResolutionType",
    "recommended_action",
]
