"""Plain dataclass schemas for local sensing-server messages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any

SENSING_UPDATE_TYPE = "sensing_update"


@dataclass(frozen=True)
class NodeInfo:
    """JSON-friendly summary for one sensing node."""

    node_id: str | int
    rssi_dbm: float = 0.0
    position: Sequence[float] = (0.0, 0.0, 0.0)
    amplitude: Sequence[float] = ()
    subcarrier_count: int = 0
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        position = tuple(float(value) for value in self.position)
        if len(position) != 3:
            raise ValueError("position must contain exactly three coordinates")

        amplitude = tuple(float(value) for value in self.amplitude)
        subcarrier_count = self.subcarrier_count or len(amplitude)
        if subcarrier_count < 0:
            raise ValueError("subcarrier_count must be non-negative")

        object.__setattr__(self, "position", position)
        object.__setattr__(self, "amplitude", amplitude)
        object.__setattr__(self, "subcarrier_count", int(subcarrier_count))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "node_id": _to_plain(self.node_id),
            "rssi_dbm": float(self.rssi_dbm),
            "position": list(self.position),
            "amplitude": list(self.amplitude),
            "subcarrier_count": int(self.subcarrier_count),
        }
        if self.metadata:
            result["metadata"] = _to_plain(self.metadata)
        return result


@dataclass(frozen=True)
class FeatureSummary:
    """Scalar feature summary for one sensing tick."""

    mean_rssi: float = 0.0
    variance: float = 0.0
    motion_band_power: float = 0.0
    breathing_band_power: float = 0.0
    dominant_freq_hz: float = 0.0
    change_points: int = 0
    spectral_power: float = 0.0
    mean_amplitude: float = 0.0
    temporal_delta: float = 0.0
    phase_variance: float = 0.0
    subcarrier_variance: float = 0.0
    motion_energy: float = 0.0
    motion_score: float = 0.0
    frame_count: int = 0
    scenario: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _without_none(_to_plain_dataclass(self))


@dataclass(frozen=True)
class ClassificationSummary:
    """Presence and motion classification for one sensing tick."""

    motion_level: str = "absent"
    presence: bool = False
    confidence: float = 0.0
    state: str = "empty"
    moving: bool = False
    presence_score: float = 0.0
    motion_score: float = 0.0
    baseline_ready: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _to_plain_dataclass(self)


@dataclass(frozen=True)
class SignalFieldSummary:
    """Compact signal-field vector for browser or notebook inspection."""

    grid_size: Sequence[int] = (0, 0, 0)
    values: Sequence[float] = ()

    def __post_init__(self) -> None:
        grid_size = tuple(int(value) for value in self.grid_size)
        if len(grid_size) != 3:
            raise ValueError("grid_size must contain exactly three dimensions")
        values = tuple(float(value) for value in self.values)
        object.__setattr__(self, "grid_size", grid_size)
        object.__setattr__(self, "values", values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid_size": list(self.grid_size),
            "values": list(self.values),
        }


@dataclass(frozen=True)
class VitalSignsSummary:
    """Optional vital-sign summary for a sensing update."""

    breathing_rate_bpm: float | None = None
    heart_rate_bpm: float | None = None
    breathing_confidence: float | None = None
    heart_confidence: float | None = None
    signal_quality: float | None = None
    status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _without_none(_to_plain_dataclass(self))


@dataclass(frozen=True)
class SensingUpdate:
    """Top-level WebSocket/REST payload for one sensing tick."""

    source: str
    tick: int
    nodes: Sequence[NodeInfo | Mapping[str, Any]] = ()
    features: FeatureSummary | Mapping[str, Any] | None = None
    classification: ClassificationSummary | Mapping[str, Any] | None = None
    signal_field: SignalFieldSummary | Mapping[str, Any] | None = None
    vital_signs: VitalSignsSummary | Mapping[str, Any] | None = None
    estimated_persons: int = 0
    msg_type: str = SENSING_UPDATE_TYPE

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("source must not be empty")
        if int(self.tick) < 0:
            raise ValueError("tick must be non-negative")
        if self.msg_type != SENSING_UPDATE_TYPE:
            raise ValueError(f"msg_type must be {SENSING_UPDATE_TYPE!r}")
        if int(self.estimated_persons) < 0:
            raise ValueError("estimated_persons must be non-negative")

        object.__setattr__(self, "tick", int(self.tick))
        object.__setattr__(self, "estimated_persons", int(self.estimated_persons))
        object.__setattr__(self, "nodes", tuple(_coerce_node(node) for node in self.nodes))
        object.__setattr__(self, "features", self.features or {})
        object.__setattr__(self, "classification", self.classification or {})
        object.__setattr__(self, "signal_field", self.signal_field or {})
        object.__setattr__(self, "vital_signs", self.vital_signs or {})

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        default_source: str = "replay",
        default_tick: int = 0,
    ) -> "SensingUpdate":
        """Build an update from a JSON-style mapping, filling missing plan fields."""

        return cls(
            source=str(payload.get("source", default_source)),
            tick=int(payload.get("tick", default_tick)),
            nodes=tuple(_coerce_node(node) for node in payload.get("nodes", ())),
            features=_mapping_or_empty(payload.get("features")),
            classification=_mapping_or_empty(payload.get("classification")),
            signal_field=_mapping_or_empty(payload.get("signal_field")),
            vital_signs=_mapping_or_empty(payload.get("vital_signs")),
            estimated_persons=int(payload.get("estimated_persons", 0)),
            msg_type=str(payload.get("type", SENSING_UPDATE_TYPE)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the compact Milestone 7 sensing-update shape."""

        return {
            "type": self.msg_type,
            "source": self.source,
            "tick": int(self.tick),
            "nodes": [node.to_dict() for node in self.nodes],
            "features": _summary_to_dict(self.features),
            "classification": _summary_to_dict(self.classification),
            "signal_field": _summary_to_dict(self.signal_field),
            "vital_signs": _summary_to_dict(self.vital_signs),
            "estimated_persons": int(self.estimated_persons),
        }


def sensing_update_to_dict(update: SensingUpdate | Mapping[str, Any]) -> dict[str, Any]:
    """Return a plain dict for an update or update-like mapping."""

    if isinstance(update, SensingUpdate):
        return update.to_dict()
    return SensingUpdate.from_dict(update).to_dict()


def _coerce_node(node: NodeInfo | Mapping[str, Any]) -> NodeInfo:
    if isinstance(node, NodeInfo):
        return node
    if not isinstance(node, Mapping):
        raise TypeError(f"node must be NodeInfo or mapping, got {type(node).__name__}")

    amplitude = node.get("amplitude", node.get("amplitudes", ()))
    known = {"node_id", "id", "rssi_dbm", "rssi", "position", "amplitude", "amplitudes", "subcarrier_count", "metadata"}
    metadata = dict(_mapping_or_empty(node.get("metadata")))
    for key, value in node.items():
        if key not in known:
            metadata.setdefault(str(key), value)

    return NodeInfo(
        node_id=node.get("node_id", node.get("id", "node")),
        rssi_dbm=float(node.get("rssi_dbm", node.get("rssi", 0.0))),
        position=node.get("position", (0.0, 0.0, 0.0)),
        amplitude=amplitude,
        subcarrier_count=int(node.get("subcarrier_count", len(amplitude))),
        metadata=metadata,
    )


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _summary_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, Mapping):
        return _to_plain(value)
    return _to_plain(value)


def _to_plain(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if is_dataclass(value) and not isinstance(value, type):
        return _to_plain_dataclass(value)
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    return value


def _to_plain_dataclass(value: Any) -> dict[str, Any]:
    return {field.name: _to_plain(getattr(value, field.name)) for field in fields(value)}


def _without_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


__all__ = [
    "SENSING_UPDATE_TYPE",
    "ClassificationSummary",
    "FeatureSummary",
    "NodeInfo",
    "SensingUpdate",
    "SignalFieldSummary",
    "VitalSignsSummary",
    "sensing_update_to_dict",
]
