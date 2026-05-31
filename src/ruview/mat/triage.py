"""START-like triage scoring for MAT research pipelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass, field, is_dataclass
from enum import Enum, IntEnum
import math
from typing import Any


LOW_CONFIDENCE_THRESHOLD = 0.35


class TriageError(ValueError):
    """Raised when triage input cannot be interpreted."""


class TriageStatus(str, Enum):
    """START-style triage categories ordered by treatment priority."""

    IMMEDIATE = "immediate"
    DELAYED = "delayed"
    MINOR = "minor"
    DECEASED = "deceased"
    UNKNOWN = "unknown"

    @property
    def priority(self) -> int:
        """Return treatment ordering where lower means more urgent."""

        return {
            TriageStatus.IMMEDIATE: 1,
            TriageStatus.DELAYED: 2,
            TriageStatus.MINOR: 3,
            TriageStatus.DECEASED: 4,
            TriageStatus.UNKNOWN: 5,
        }[self]

    @property
    def color(self) -> str:
        """Return a conventional triage color name."""

        return {
            TriageStatus.IMMEDIATE: "red",
            TriageStatus.DELAYED: "yellow",
            TriageStatus.MINOR: "green",
            TriageStatus.DECEASED: "black",
            TriageStatus.UNKNOWN: "gray",
        }[self]

    @property
    def description(self) -> str:
        """Return a compact human-readable status description."""

        return {
            TriageStatus.IMMEDIATE: "Requires immediate life-saving intervention",
            TriageStatus.DELAYED: "Serious but can wait for treatment",
            TriageStatus.MINOR: "Minor injuries or responsive walking wounded",
            TriageStatus.DECEASED: "No vital signs detected",
            TriageStatus.UNKNOWN: "Insufficient confidence or data for classification",
        }[self]

    @property
    def is_urgent(self) -> bool:
        return self in {TriageStatus.IMMEDIATE, TriageStatus.DELAYED}


class Priority(IntEnum):
    """Local alert priority where lower numeric values are more urgent."""

    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

    @classmethod
    def from_triage(cls, status: TriageStatus | str) -> "Priority":
        status = _coerce_triage_status(status)
        return {
            TriageStatus.IMMEDIATE: cls.CRITICAL,
            TriageStatus.DELAYED: cls.HIGH,
            TriageStatus.MINOR: cls.MEDIUM,
            TriageStatus.DECEASED: cls.LOW,
            TriageStatus.UNKNOWN: cls.MEDIUM,
        }[status]

    @property
    def color(self) -> str:
        return {
            Priority.CRITICAL: "red",
            Priority.HIGH: "orange",
            Priority.MEDIUM: "yellow",
            Priority.LOW: "blue",
        }[self]

    @property
    def audio_pattern(self) -> str:
        return {
            Priority.CRITICAL: "rapid_beep",
            Priority.HIGH: "double_beep",
            Priority.MEDIUM: "single_beep",
            Priority.LOW: "soft_tone",
        }[self]


@dataclass(frozen=True)
class TriageInput:
    """Minimal vital/movement observation used by the MAT triage scorer."""

    breathing: float | None = None
    heart: float | None = None
    movement: str | None = None
    location: Sequence[float] | None = None
    confidence: float = 1.0
    breathing_pattern: str | None = None
    movement_is_voluntary: bool | None = None
    metadata: Mapping[str, Any] | None = None
    breathing_rate_bpm: InitVar[float | None] = None
    heart_rate_bpm: InitVar[float | None] = None

    def __post_init__(
        self,
        breathing_rate_bpm: float | None,
        heart_rate_bpm: float | None,
    ) -> None:
        breathing = _coerce_optional_float("breathing", self.breathing)
        heart = _coerce_optional_float("heart", self.heart)
        alias_breathing = _coerce_optional_float("breathing_rate_bpm", breathing_rate_bpm)
        alias_heart = _coerce_optional_float("heart_rate_bpm", heart_rate_bpm)

        breathing = _merge_alias("breathing", breathing, alias_breathing)
        heart = _merge_alias("heart", heart, alias_heart)
        confidence = _coerce_float("confidence", self.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise TriageError("confidence must be in [0, 1]")

        movement = str(self.movement).strip().lower() if self.movement is not None else None
        breathing_pattern = (
            str(self.breathing_pattern).strip().lower() if self.breathing_pattern is not None else None
        )
        location = _coerce_location(self.location)

        object.__setattr__(self, "breathing", breathing)
        object.__setattr__(self, "heart", heart)
        object.__setattr__(self, "movement", movement)
        object.__setattr__(self, "location", location)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "breathing_pattern", breathing_pattern)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def breathing_rate_bpm_value(self) -> float | None:
        """Alias for callers that use the explicit rate naming."""

        return self.breathing

    @property
    def heart_rate_bpm_value(self) -> float | None:
        """Alias for callers that use the explicit rate naming."""

        return self.heart

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TriageInput":
        """Build triage input from survivor/vital-sign style dictionaries."""

        if not isinstance(payload, Mapping):
            raise TriageError("TriageInput.from_mapping expects a mapping")

        vitals = _mapping_or_empty(payload.get("vitals") or payload.get("vital_signs"))
        merged = {**vitals, **payload}
        metadata = dict(_mapping_or_empty(merged.get("metadata")))
        for key, value in merged.items():
            if key not in _TRIAGE_INPUT_KEYS:
                metadata.setdefault(str(key), value)

        return cls(
            breathing=_first_present(
                merged,
                "breathing",
                "breathing_rate_bpm",
                "respiration",
                "respiration_rate_bpm",
                "respiratory_rate_bpm",
            ),
            heart=_first_present(
                merged,
                "heart",
                "heart_rate_bpm",
                "heartbeat",
                "pulse",
                "pulse_rate_bpm",
            ),
            movement=_first_present(merged, "movement", "movement_type", "motion", "responsiveness"),
            location=_first_present(merged, "location", "position", "coordinates"),
            confidence=_first_present(merged, "confidence", "triage_confidence", default=1.0),
            breathing_pattern=_first_present(merged, "breathing_pattern", "pattern_type"),
            movement_is_voluntary=_first_present(merged, "movement_is_voluntary", "is_voluntary"),
            metadata=metadata,
        )


@dataclass(frozen=True)
class TriageResult:
    """Result of scoring a single MAT observation."""

    status: TriageStatus
    priority: Priority
    confidence: float
    rationale: tuple[str, ...]
    triage_input: TriageInput

    @property
    def is_actionable(self) -> bool:
        return self.status is not TriageStatus.UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "priority": self.priority.name.lower(),
            "priority_value": int(self.priority),
            "confidence": float(self.confidence),
            "rationale": list(self.rationale),
            "location": list(self.triage_input.location) if self.triage_input.location is not None else None,
        }


@dataclass(frozen=True)
class SurvivorSnapshot:
    """Small survivor-like record accepted by the local MAT helpers."""

    survivor_id: str
    triage_input: TriageInput
    zone_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def location(self) -> tuple[float, ...] | None:
        return self.triage_input.location

    @property
    def confidence(self) -> float:
        return self.triage_input.confidence


@dataclass(frozen=True)
class SurvivorTriage:
    """Triage assessment attached to a survivor identifier."""

    survivor_id: str
    triage: TriageResult
    zone_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> TriageStatus:
        return self.triage.status

    @property
    def priority(self) -> Priority:
        return self.triage.priority

    @property
    def confidence(self) -> float:
        return self.triage.confidence

    @property
    def location(self) -> tuple[float, ...] | None:
        return self.triage.triage_input.location


class TriageCalculator:
    """START-like remote-sensing triage calculator."""

    @classmethod
    def calculate(cls, observation: TriageInput | Mapping[str, Any] | Any) -> TriageResult:
        triage_input = coerce_triage_input(observation)
        status, rationale = cls._status_and_rationale(triage_input)
        return TriageResult(
            status=status,
            priority=Priority.from_triage(status),
            confidence=triage_input.confidence,
            rationale=tuple(rationale),
            triage_input=triage_input,
        )

    @staticmethod
    def should_upgrade(current: TriageStatus | str, *, is_deteriorating: bool) -> bool:
        if not is_deteriorating:
            return False
        current = _coerce_triage_status(current)
        return current in {TriageStatus.DELAYED, TriageStatus.MINOR}

    @staticmethod
    def upgrade(current: TriageStatus | str) -> TriageStatus:
        current = _coerce_triage_status(current)
        if current is TriageStatus.MINOR:
            return TriageStatus.DELAYED
        if current is TriageStatus.DELAYED:
            return TriageStatus.IMMEDIATE
        return current

    @classmethod
    def _status_and_rationale(cls, triage_input: TriageInput) -> tuple[TriageStatus, list[str]]:
        rationale: list[str] = []
        breathing = triage_input.breathing
        heart = triage_input.heart
        movement = _movement_category(triage_input)

        if triage_input.confidence < LOW_CONFIDENCE_THRESHOLD:
            rationale.append(
                f"confidence {triage_input.confidence:.2f} below {LOW_CONFIDENCE_THRESHOLD:.2f} threshold"
            )
            return TriageStatus.UNKNOWN, rationale

        if breathing is None and heart is None and triage_input.movement is None:
            rationale.append("no breathing, heart, or movement observation provided")
            return TriageStatus.UNKNOWN, rationale

        if cls._no_vitals_detected(triage_input, movement):
            rationale.append("no breathing, heartbeat, or movement detected with adequate confidence")
            return TriageStatus.DECEASED, rationale

        if triage_input.breathing_pattern == "agonal":
            rationale.append("agonal breathing pattern")
            return TriageStatus.IMMEDIATE, rationale

        if breathing is not None:
            if breathing <= 0.0:
                rationale.append("breathing absent but other signs may be present")
                return TriageStatus.IMMEDIATE, rationale
            if breathing < 10.0:
                rationale.append(f"respiration {breathing:.1f} BPM below START lower bound")
                return TriageStatus.IMMEDIATE, rationale
            if breathing > 30.0:
                rationale.append(f"respiration {breathing:.1f} BPM above START upper bound")
                return TriageStatus.IMMEDIATE, rationale
            rationale.append(f"respiration {breathing:.1f} BPM within START range")

        if heart is not None:
            if heart <= 0.0:
                rationale.append("heartbeat absent")
                return TriageStatus.IMMEDIATE, rationale
            if heart < 40.0:
                rationale.append(f"heart rate {heart:.1f} BPM critically low")
                return TriageStatus.IMMEDIATE, rationale
            if heart > 130.0:
                rationale.append(f"heart rate {heart:.1f} BPM critically high")
                return TriageStatus.IMMEDIATE, rationale
            rationale.append(f"heart rate {heart:.1f} BPM present")

        if movement == "none" and breathing is not None:
            rationale.append("breathing present but no movement detected")
            return TriageStatus.IMMEDIATE, rationale

        if movement == "involuntary":
            rationale.append("only involuntary movement detected")
            return TriageStatus.IMMEDIATE, rationale

        if movement == "responsive":
            rationale.append("responsive or voluntary gross movement detected")
            return TriageStatus.MINOR, rationale

        if movement in {"moving", "minimal"}:
            rationale.append(f"{movement} movement detected without immediate vital-sign trigger")
            return TriageStatus.DELAYED, rationale

        if breathing is not None or heart is not None:
            rationale.append("vitals present but responsiveness is unclear")
            return TriageStatus.DELAYED, rationale

        rationale.append("insufficient evidence after START screening")
        return TriageStatus.UNKNOWN, rationale

    @staticmethod
    def _no_vitals_detected(triage_input: TriageInput, movement: str) -> bool:
        breathing_absent = triage_input.breathing is not None and triage_input.breathing <= 0.0
        heart_absent = triage_input.heart is not None and triage_input.heart <= 0.0
        no_movement = movement == "none"
        return no_movement and breathing_absent and (heart_absent or triage_input.heart is None)


class TriageService:
    """Convenience service for triaging survivor-like objects and batches."""

    def calculate_triage(self, observation: TriageInput | Mapping[str, Any] | Any) -> TriageResult:
        return TriageCalculator.calculate(observation)

    def triage_survivor(self, survivor: SurvivorSnapshot | Mapping[str, Any] | Any) -> SurvivorTriage:
        snapshot = coerce_survivor_snapshot(survivor)
        return SurvivorTriage(
            survivor_id=snapshot.survivor_id,
            triage=TriageCalculator.calculate(snapshot.triage_input),
            zone_id=snapshot.zone_id,
            metadata=dict(snapshot.metadata),
        )

    def triage_batch(self, survivors: Sequence[SurvivorSnapshot | Mapping[str, Any] | Any]) -> list[SurvivorTriage]:
        assessments = [self.triage_survivor(survivor) for survivor in survivors]
        return sorted(
            assessments,
            key=lambda assessment: (
                assessment.status.priority,
                -assessment.confidence,
                assessment.survivor_id,
            ),
        )

    @staticmethod
    def should_upgrade(current: TriageStatus | str, *, is_deteriorating: bool) -> bool:
        return TriageCalculator.should_upgrade(current, is_deteriorating=is_deteriorating)

    @staticmethod
    def upgrade_status(current: TriageStatus | str) -> TriageStatus:
        return TriageCalculator.upgrade(current)


def calculate_triage(observation: TriageInput | Mapping[str, Any] | Any) -> TriageResult:
    """Score one observation with the default MAT triage calculator."""

    return TriageCalculator.calculate(observation)


def coerce_triage_input(observation: TriageInput | Mapping[str, Any] | Any) -> TriageInput:
    """Coerce mappings/dataclasses/objects into a :class:`TriageInput`."""

    if isinstance(observation, TriageInput):
        return observation
    if isinstance(observation, Mapping):
        return TriageInput.from_mapping(observation)
    if is_dataclass(observation):
        return TriageInput.from_mapping({name: getattr(observation, name) for name in _dataclass_field_names(observation)})

    attrs = {
        key: getattr(observation, key)
        for key in _TRIAGE_INPUT_KEYS
        if hasattr(observation, key)
    }
    if attrs:
        return TriageInput.from_mapping(attrs)
    raise TriageError(f"cannot coerce {type(observation).__name__} to TriageInput")


def coerce_survivor_snapshot(survivor: SurvivorSnapshot | Mapping[str, Any] | Any) -> SurvivorSnapshot:
    """Coerce survivor-like mappings/dataclasses into a stable snapshot."""

    if isinstance(survivor, SurvivorSnapshot):
        return survivor

    payload: dict[str, Any]
    if isinstance(survivor, Mapping):
        payload = dict(survivor)
    elif is_dataclass(survivor):
        payload = {name: getattr(survivor, name) for name in _dataclass_field_names(survivor)}
    else:
        payload = {
            key: getattr(survivor, key)
            for key in _SURVIVOR_KEYS
            if hasattr(survivor, key)
        }

    if not payload:
        raise TriageError(f"cannot coerce {type(survivor).__name__} to SurvivorSnapshot")

    survivor_id = _first_present(payload, "survivor_id", "id", "track_id", "person_id", default="unknown")
    zone_id = _first_present(payload, "zone_id", "scan_zone_id", "zone", default=None)
    metadata = dict(_mapping_or_empty(payload.get("metadata")))

    if isinstance(payload.get("triage_input"), TriageInput):
        triage_input = payload["triage_input"]
    else:
        triage_payload = dict(_mapping_or_empty(payload.get("triage_input")))
        vitals = _mapping_or_empty(payload.get("vitals") or payload.get("vital_signs"))
        triage_payload.update(vitals)
        for key in _TRIAGE_INPUT_KEYS:
            if key in payload:
                triage_payload[key] = payload[key]
        if "location" not in triage_payload:
            triage_payload["location"] = _first_present(payload, "location", "position", "coordinates")
        if "confidence" not in triage_payload:
            triage_payload["confidence"] = _first_present(payload, "confidence", default=1.0)
        triage_input = TriageInput.from_mapping(triage_payload)

    for key, value in payload.items():
        if key not in _SURVIVOR_KEYS:
            metadata.setdefault(str(key), value)

    return SurvivorSnapshot(
        survivor_id=str(survivor_id),
        zone_id=str(zone_id) if zone_id is not None else None,
        triage_input=triage_input,
        metadata=metadata,
    )


def _movement_category(triage_input: TriageInput) -> str:
    movement = triage_input.movement
    if triage_input.movement_is_voluntary is True:
        return "responsive"
    if movement is None:
        return "unknown"

    if movement in {"none", "absent", "still", "immobile", "unresponsive", "no_movement"}:
        return "none"
    if movement in {"tremor", "seizure", "involuntary", "agonal"}:
        return "involuntary"
    if movement in {"fine", "minimal", "periodic", "small"}:
        return "minimal"
    if movement in {"gross", "moving", "motion"}:
        return "moving" if triage_input.movement_is_voluntary is not True else "responsive"
    if movement in {"responsive", "voluntary", "purposeful", "walking", "walk", "ambulatory"}:
        return "responsive"
    return "unknown"


def _coerce_triage_status(status: TriageStatus | str) -> TriageStatus:
    if isinstance(status, TriageStatus):
        return status
    try:
        return TriageStatus(str(status).strip().lower())
    except ValueError as exc:
        raise TriageError(f"unknown triage status {status!r}") from exc


def _coerce_optional_float(name: str, value: Any) -> float | None:
    if value is None:
        return None
    return _coerce_float(name, value)


def _coerce_float(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TriageError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise TriageError(f"{name} must be finite")
    return result


def _merge_alias(name: str, primary: float | None, alias: float | None) -> float | None:
    if primary is None:
        return alias
    if alias is not None and not math.isclose(primary, alias, rel_tol=1e-9, abs_tol=1e-9):
        raise TriageError(f"{name} and {name}_rate_bpm disagree")
    return primary


def _coerce_location(location: Sequence[float] | None) -> tuple[float, ...] | None:
    if location is None:
        return None
    values = tuple(float(value) for value in location)
    if len(values) not in {2, 3}:
        raise TriageError("location must contain two or three coordinates")
    if not all(math.isfinite(value) for value in values):
        raise TriageError("location coordinates must be finite")
    return values


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_present(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _dataclass_field_names(instance: Any) -> tuple[str, ...]:
    return tuple(instance.__dataclass_fields__.keys())


_TRIAGE_INPUT_KEYS = {
    "breathing",
    "breathing_rate_bpm",
    "respiration",
    "respiration_rate_bpm",
    "respiratory_rate_bpm",
    "heart",
    "heart_rate_bpm",
    "heartbeat",
    "pulse",
    "pulse_rate_bpm",
    "movement",
    "movement_type",
    "motion",
    "responsiveness",
    "location",
    "position",
    "coordinates",
    "confidence",
    "triage_confidence",
    "breathing_pattern",
    "pattern_type",
    "movement_is_voluntary",
    "is_voluntary",
    "metadata",
    "vitals",
    "vital_signs",
}

_SURVIVOR_KEYS = _TRIAGE_INPUT_KEYS | {
    "survivor_id",
    "id",
    "track_id",
    "person_id",
    "zone_id",
    "scan_zone_id",
    "zone",
    "triage_input",
}


__all__ = [
    "LOW_CONFIDENCE_THRESHOLD",
    "Priority",
    "SurvivorSnapshot",
    "SurvivorTriage",
    "TriageCalculator",
    "TriageError",
    "TriageInput",
    "TriageResult",
    "TriageService",
    "TriageStatus",
    "calculate_triage",
    "coerce_survivor_snapshot",
    "coerce_triage_input",
]
