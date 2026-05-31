"""Provenance records for WorldGraph semantic beliefs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self


@dataclass(frozen=True)
class SemanticProvenance:
    """Evidence, model, calibration, and privacy lineage for a semantic state."""

    evidence: tuple[str, ...] = ()
    model_version: str = ""
    calibration_version: str = ""
    privacy_decision: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", tuple(str(item) for item in self.evidence))
        object.__setattr__(self, "model_version", str(self.model_version))
        object.__setattr__(self, "calibration_version", str(self.calibration_version))
        object.__setattr__(self, "privacy_decision", str(self.privacy_decision))

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic JSON shape used in WorldGraph snapshots."""

        return {
            "evidence": list(self.evidence),
            "model_version": self.model_version,
            "calibration_version": self.calibration_version,
            "privacy_decision": self.privacy_decision,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Build provenance from a JSON-style mapping."""

        return cls(
            evidence=_string_sequence(payload.get("evidence", ())),
            model_version=str(payload.get("model_version", "")),
            calibration_version=str(payload.get("calibration_version", "")),
            privacy_decision=str(payload.get("privacy_decision", "")),
        )

    @classmethod
    def coerce(cls, value: "SemanticProvenance | Mapping[str, Any]") -> "SemanticProvenance":
        """Return ``value`` as a :class:`SemanticProvenance`."""

        if isinstance(value, SemanticProvenance):
            return value
        if isinstance(value, Mapping):
            return cls.from_dict(value)
        raise TypeError(f"provenance must be SemanticProvenance or mapping, got {type(value).__name__}")


def _string_sequence(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    raise TypeError(f"evidence must be a sequence of strings, got {type(value).__name__}")


__all__ = ["SemanticProvenance"]
