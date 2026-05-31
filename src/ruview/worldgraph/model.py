"""Typed WorldGraph node and edge model primitives."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, ClassVar, Self

from .provenance import SemanticProvenance


@dataclass(frozen=True, order=True)
class WorldId:
    """Stable monotonic identity for a world entity."""

    value: int = 0

    UNASSIGNED: ClassVar["WorldId"]

    def __post_init__(self) -> None:
        value = int(self.value)
        if value < 0:
            raise ValueError("world id must be non-negative")
        object.__setattr__(self, "value", value)

    def is_unassigned(self) -> bool:
        """Whether this id is the allocation sentinel."""

        return self.value == 0

    def to_json_value(self) -> int:
        """Return the JSON scalar representation."""

        return self.value

    @classmethod
    def coerce(cls, value: "WorldId | int") -> "WorldId":
        """Return ``value`` as a :class:`WorldId`."""

        if isinstance(value, WorldId):
            return value
        return cls(int(value))

    def __int__(self) -> int:
        return self.value

    def __repr__(self) -> str:
        return f"WorldId({self.value})"


WorldId.UNASSIGNED = WorldId(0)


@dataclass(frozen=True)
class EnuPoint:
    """Local ENU coordinate in metres relative to the installation origin."""

    east_m: float
    north_m: float
    up_m: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "east_m", float(self.east_m))
        object.__setattr__(self, "north_m", float(self.north_m))
        object.__setattr__(self, "up_m", float(self.up_m))

    def to_dict(self) -> dict[str, float]:
        return {
            "east_m": self.east_m,
            "north_m": self.north_m,
            "up_m": self.up_m,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        return cls(
            east_m=float(payload.get("east_m", 0.0)),
            north_m=float(payload.get("north_m", 0.0)),
            up_m=float(payload.get("up_m", 0.0)),
        )

    @classmethod
    def coerce(cls, value: "EnuPoint | Mapping[str, Any] | Sequence[float]") -> "EnuPoint":
        if isinstance(value, EnuPoint):
            return value
        if isinstance(value, Mapping):
            return cls.from_dict(value)
        if isinstance(value, Sequence) and not isinstance(value, str):
            if len(value) != 3:
                raise ValueError("ENU point sequence must contain exactly three coordinates")
            return cls(float(value[0]), float(value[1]), float(value[2]))
        raise TypeError(f"ENU point must be EnuPoint, mapping, or sequence, got {type(value).__name__}")


@dataclass(frozen=True)
class ZoneBoundsEnu:
    """MAT zone bounds reprojected into the installation ENU frame."""

    shape: str
    min_e: float | None = None
    min_n: float | None = None
    max_e: float | None = None
    max_n: float | None = None
    center_e: float | None = None
    center_n: float | None = None
    radius_m: float | None = None
    vertices: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        shape = str(self.shape)
        if shape not in {"rectangle", "circle", "polygon"}:
            raise ValueError(f"unknown zone bounds shape {shape!r}")
        object.__setattr__(self, "shape", shape)

        for name in ("min_e", "min_n", "max_e", "max_n", "center_e", "center_n", "radius_m"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, float(value))
        object.__setattr__(self, "vertices", _coerce_vertices(self.vertices))

        if shape == "rectangle" and None in (self.min_e, self.min_n, self.max_e, self.max_n):
            raise ValueError("rectangle bounds require min_e, min_n, max_e, and max_n")
        if shape == "circle" and None in (self.center_e, self.center_n, self.radius_m):
            raise ValueError("circle bounds require center_e, center_n, and radius_m")
        if shape == "polygon" and len(self.vertices) < 3:
            raise ValueError("polygon bounds require at least three vertices")

    @classmethod
    def rectangle(cls, min_e: float, min_n: float, max_e: float, max_n: float) -> Self:
        return cls("rectangle", min_e=min_e, min_n=min_n, max_e=max_e, max_n=max_n)

    @classmethod
    def circle(cls, center_e: float, center_n: float, radius_m: float) -> Self:
        return cls("circle", center_e=center_e, center_n=center_n, radius_m=radius_m)

    @classmethod
    def polygon(cls, vertices: Sequence[Sequence[float]]) -> Self:
        return cls("polygon", vertices=_coerce_vertices(vertices))

    Rectangle = rectangle
    Circle = circle
    Polygon = polygon

    def contains(self, point: EnuPoint | Mapping[str, Any] | Sequence[float]) -> bool:
        """Whether an ENU point lies within these bounds, ignoring ``up_m``."""

        p = EnuPoint.coerce(point)
        if self.shape == "rectangle":
            return bool(
                self.min_e <= p.east_m <= self.max_e
                and self.min_n <= p.north_m <= self.max_n
            )
        if self.shape == "circle":
            de = p.east_m - self.center_e
            dn = p.north_m - self.center_n
            return math.hypot(de, dn) <= self.radius_m
        return _point_in_polygon(p.east_m, p.north_m, self.vertices)

    def to_dict(self) -> dict[str, Any]:
        if self.shape == "rectangle":
            return {
                "shape": self.shape,
                "min_e": self.min_e,
                "min_n": self.min_n,
                "max_e": self.max_e,
                "max_n": self.max_n,
            }
        if self.shape == "circle":
            return {
                "shape": self.shape,
                "center_e": self.center_e,
                "center_n": self.center_n,
                "radius_m": self.radius_m,
            }
        return {
            "shape": self.shape,
            "vertices": [[east, north] for east, north in self.vertices],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        shape = str(payload.get("shape", ""))
        if shape == "rectangle":
            return cls.rectangle(payload["min_e"], payload["min_n"], payload["max_e"], payload["max_n"])
        if shape == "circle":
            return cls.circle(payload["center_e"], payload["center_n"], payload["radius_m"])
        if shape == "polygon":
            return cls.polygon(payload.get("vertices", ()))
        raise ValueError(f"unknown zone bounds shape {shape!r}")

    @classmethod
    def coerce(cls, value: "ZoneBoundsEnu | Mapping[str, Any]") -> "ZoneBoundsEnu":
        if isinstance(value, ZoneBoundsEnu):
            return value
        if isinstance(value, Mapping):
            return cls.from_dict(value)
        raise TypeError(f"zone bounds must be ZoneBoundsEnu or mapping, got {type(value).__name__}")


class SensorModality(str, Enum):
    """Sensing modality of a physical device placement."""

    WIFI_CSI = "wifi_csi"
    MM_WAVE = "mm_wave"
    UWB = "uwb"
    PRESENCE = "presence"


class AnchorKind(str, Enum):
    """Kind of persistent static anchor."""

    REFLECTOR = "reflector"
    FURNITURE = "furniture"
    UWB_BEACON = "uwb_beacon"


@dataclass(frozen=True)
class WorldNode:
    """Variant-like WorldGraph node with a stable id and snake-case kind tag."""

    kind: str
    id: WorldId = field(default_factory=lambda: WorldId.UNASSIGNED)
    area_id: str | None = None
    name: str = ""
    bounds_enu: ZoneBoundsEnu | None = None
    floor: int = 0
    parent_room: WorldId | None = None
    a: EnuPoint | None = None
    b: EnuPoint | None = None
    rf_attenuation_db: float = 0.0
    center: EnuPoint | None = None
    width_m: float = 0.0
    device_id: str = ""
    position: EnuPoint | None = None
    modality: SensorModality | None = None
    tx: WorldId | None = None
    rx: WorldId | None = None
    link_group_id: str | None = None
    center_freq_mhz: int = 0
    track_id: int = 0
    last_position: EnuPoint | None = None
    reid_embedding_ref: str | None = None
    anchor_kind: AnchorKind | None = None
    confidence: float = 0.0
    event_type: str = ""
    at_unix_ms: int = 0
    located_in: WorldId | None = None
    statement: str = ""
    provenance: SemanticProvenance | None = None
    valid_from_unix_ms: int = 0

    def __post_init__(self) -> None:
        kind = str(self.kind)
        if kind not in _NODE_FIELDS:
            raise ValueError(f"unknown node kind {kind!r}")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "id", WorldId.coerce(self.id))
        object.__setattr__(self, "floor", int(self.floor))
        object.__setattr__(self, "center_freq_mhz", int(self.center_freq_mhz))
        object.__setattr__(self, "track_id", int(self.track_id))
        object.__setattr__(self, "at_unix_ms", int(self.at_unix_ms))
        object.__setattr__(self, "valid_from_unix_ms", int(self.valid_from_unix_ms))
        object.__setattr__(self, "rf_attenuation_db", float(self.rf_attenuation_db))
        object.__setattr__(self, "width_m", float(self.width_m))
        object.__setattr__(self, "confidence", float(self.confidence))

        for attr in ("parent_room", "tx", "rx", "located_in"):
            value = getattr(self, attr)
            if value is not None:
                object.__setattr__(self, attr, WorldId.coerce(value))
        for attr in ("a", "b", "center", "position", "last_position"):
            value = getattr(self, attr)
            if value is not None:
                object.__setattr__(self, attr, EnuPoint.coerce(value))
        if self.bounds_enu is not None:
            object.__setattr__(self, "bounds_enu", ZoneBoundsEnu.coerce(self.bounds_enu))
        if self.modality is not None:
            object.__setattr__(self, "modality", _coerce_enum(SensorModality, self.modality))
        if self.anchor_kind is not None:
            object.__setattr__(self, "anchor_kind", _coerce_enum(AnchorKind, self.anchor_kind))
        if self.provenance is not None:
            object.__setattr__(self, "provenance", SemanticProvenance.coerce(self.provenance))
        if self.kind == "semantic_state" and self.provenance is None:
            raise ValueError("semantic_state nodes require provenance")

    @classmethod
    def room(
        cls,
        *,
        id: WorldId | int = WorldId.UNASSIGNED,
        area_id: str | None = None,
        name: str,
        bounds_enu: ZoneBoundsEnu | Mapping[str, Any],
        floor: int = 0,
    ) -> Self:
        return cls("room", id=WorldId.coerce(id), area_id=area_id, name=name, bounds_enu=ZoneBoundsEnu.coerce(bounds_enu), floor=floor)

    @classmethod
    def zone(
        cls,
        *,
        id: WorldId | int = WorldId.UNASSIGNED,
        parent_room: WorldId | int,
        name: str,
        bounds_enu: ZoneBoundsEnu | Mapping[str, Any],
    ) -> Self:
        return cls(
            "zone",
            id=WorldId.coerce(id),
            parent_room=WorldId.coerce(parent_room),
            name=name,
            bounds_enu=ZoneBoundsEnu.coerce(bounds_enu),
        )

    @classmethod
    def wall(
        cls,
        *,
        id: WorldId | int = WorldId.UNASSIGNED,
        a: EnuPoint | Mapping[str, Any] | Sequence[float],
        b: EnuPoint | Mapping[str, Any] | Sequence[float],
        rf_attenuation_db: float,
    ) -> Self:
        return cls("wall", id=WorldId.coerce(id), a=EnuPoint.coerce(a), b=EnuPoint.coerce(b), rf_attenuation_db=rf_attenuation_db)

    @classmethod
    def doorway(
        cls,
        *,
        id: WorldId | int = WorldId.UNASSIGNED,
        center: EnuPoint | Mapping[str, Any] | Sequence[float],
        width_m: float,
    ) -> Self:
        return cls("doorway", id=WorldId.coerce(id), center=EnuPoint.coerce(center), width_m=width_m)

    @classmethod
    def sensor(
        cls,
        *,
        id: WorldId | int = WorldId.UNASSIGNED,
        device_id: str,
        position: EnuPoint | Mapping[str, Any] | Sequence[float],
        modality: SensorModality | str,
    ) -> Self:
        return cls("sensor", id=WorldId.coerce(id), device_id=device_id, position=EnuPoint.coerce(position), modality=_coerce_enum(SensorModality, modality))

    @classmethod
    def rf_link(
        cls,
        *,
        id: WorldId | int = WorldId.UNASSIGNED,
        tx: WorldId | int,
        rx: WorldId | int,
        link_group_id: str | None = None,
        center_freq_mhz: int,
    ) -> Self:
        return cls(
            "rf_link",
            id=WorldId.coerce(id),
            tx=WorldId.coerce(tx),
            rx=WorldId.coerce(rx),
            link_group_id=link_group_id,
            center_freq_mhz=center_freq_mhz,
        )

    @classmethod
    def person_track(
        cls,
        *,
        id: WorldId | int = WorldId.UNASSIGNED,
        track_id: int,
        last_position: EnuPoint | Mapping[str, Any] | Sequence[float],
        reid_embedding_ref: str | None = None,
    ) -> Self:
        return cls(
            "person_track",
            id=WorldId.coerce(id),
            track_id=track_id,
            last_position=EnuPoint.coerce(last_position),
            reid_embedding_ref=reid_embedding_ref,
        )

    @classmethod
    def object_anchor(
        cls,
        *,
        id: WorldId | int = WorldId.UNASSIGNED,
        position: EnuPoint | Mapping[str, Any] | Sequence[float],
        anchor_kind: AnchorKind | str,
        confidence: float,
    ) -> Self:
        return cls(
            "object_anchor",
            id=WorldId.coerce(id),
            position=EnuPoint.coerce(position),
            anchor_kind=_coerce_enum(AnchorKind, anchor_kind),
            confidence=confidence,
        )

    @classmethod
    def event(
        cls,
        *,
        id: WorldId | int = WorldId.UNASSIGNED,
        event_type: str,
        at_unix_ms: int,
        located_in: WorldId | int | None = None,
    ) -> Self:
        return cls(
            "event",
            id=WorldId.coerce(id),
            event_type=event_type,
            at_unix_ms=at_unix_ms,
            located_in=None if located_in is None else WorldId.coerce(located_in),
        )

    @classmethod
    def semantic_state(
        cls,
        *,
        id: WorldId | int = WorldId.UNASSIGNED,
        statement: str,
        confidence: float,
        provenance: SemanticProvenance | Mapping[str, Any],
        valid_from_unix_ms: int,
    ) -> Self:
        return cls(
            "semantic_state",
            id=WorldId.coerce(id),
            statement=statement,
            confidence=confidence,
            provenance=SemanticProvenance.coerce(provenance),
            valid_from_unix_ms=valid_from_unix_ms,
        )

    Room = room
    Zone = zone
    Wall = wall
    Doorway = doorway
    Sensor = sensor
    RfLink = rf_link
    PersonTrack = person_track
    ObjectAnchor = object_anchor
    Event = event
    SemanticState = semantic_state

    def with_id(self, new_id: WorldId | int) -> "WorldNode":
        """Return this node with a different embedded stable id."""

        return replace(self, id=WorldId.coerce(new_id))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind, "id": int(self.id)}
        for name in _NODE_FIELDS[self.kind]:
            result[name] = _to_plain(getattr(self, name))
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        kind = str(payload["kind"])
        if kind not in _NODE_FIELDS:
            raise ValueError(f"unknown node kind {kind!r}")
        kwargs = {"kind": kind, "id": WorldId.coerce(payload.get("id", 0))}
        for name in _NODE_FIELDS[kind]:
            if name in payload:
                kwargs[name] = payload[name]
        return cls(**kwargs)

    @classmethod
    def coerce(cls, value: "WorldNode | Mapping[str, Any]") -> "WorldNode":
        if isinstance(value, WorldNode):
            return value
        if isinstance(value, Mapping):
            return cls.from_dict(value)
        raise TypeError(f"node must be WorldNode or mapping, got {type(value).__name__}")


@dataclass(frozen=True)
class WorldEdge:
    """Variant-like directed WorldGraph edge with a snake-case relation tag."""

    rel: str
    quality: float = 0.0
    last_seen_unix_ms: int = 0
    since_unix_ms: int = 0
    via_doorway: WorldId | None = None
    strength: float = 0.0
    magnitude: float = 0.0
    flag: str = ""
    evidence: str = ""
    mode: str = ""
    action: str = ""
    allowed: bool = True

    def __post_init__(self) -> None:
        rel = str(self.rel)
        if rel not in _EDGE_FIELDS:
            raise ValueError(f"unknown edge relation {rel!r}")
        object.__setattr__(self, "rel", rel)
        object.__setattr__(self, "quality", float(self.quality))
        object.__setattr__(self, "last_seen_unix_ms", int(self.last_seen_unix_ms))
        object.__setattr__(self, "since_unix_ms", int(self.since_unix_ms))
        object.__setattr__(self, "strength", float(self.strength))
        object.__setattr__(self, "magnitude", float(self.magnitude))
        object.__setattr__(self, "allowed", bool(self.allowed))
        if self.via_doorway is not None:
            object.__setattr__(self, "via_doorway", WorldId.coerce(self.via_doorway))

    @classmethod
    def observes(cls, quality: float, last_seen_unix_ms: int) -> Self:
        return cls("observes", quality=quality, last_seen_unix_ms=last_seen_unix_ms)

    @classmethod
    def located_in(cls, since_unix_ms: int) -> Self:
        return cls("located_in", since_unix_ms=since_unix_ms)

    @classmethod
    def adjacent_to(cls, via_doorway: WorldId | int) -> Self:
        return cls("adjacent_to", via_doorway=WorldId.coerce(via_doorway))

    @classmethod
    def supports(cls, strength: float) -> Self:
        return cls("supports", strength=strength)

    @classmethod
    def contradicts(cls, magnitude: float, flag: str) -> Self:
        return cls("contradicts", magnitude=magnitude, flag=flag)

    @classmethod
    def derived_from(cls, evidence: str) -> Self:
        return cls("derived_from", evidence=evidence)

    @classmethod
    def privacy_limited_by(cls, mode: str, action: str, allowed: bool) -> Self:
        return cls("privacy_limited_by", mode=mode, action=action, allowed=allowed)

    Observes = observes
    LocatedIn = located_in
    AdjacentTo = adjacent_to
    Supports = supports
    Contradicts = contradicts
    DerivedFrom = derived_from
    PrivacyLimitedBy = privacy_limited_by

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"rel": self.rel}
        for name in _EDGE_FIELDS[self.rel]:
            result[name] = _to_plain(getattr(self, name))
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        rel = str(payload["rel"])
        if rel not in _EDGE_FIELDS:
            raise ValueError(f"unknown edge relation {rel!r}")
        kwargs = {"rel": rel}
        for name in _EDGE_FIELDS[rel]:
            if name in payload:
                kwargs[name] = payload[name]
        return cls(**kwargs)

    @classmethod
    def coerce(cls, value: "WorldEdge | Mapping[str, Any]") -> "WorldEdge":
        if isinstance(value, WorldEdge):
            return value
        if isinstance(value, Mapping):
            return cls.from_dict(value)
        raise TypeError(f"edge must be WorldEdge or mapping, got {type(value).__name__}")


_NODE_FIELDS: dict[str, tuple[str, ...]] = {
    "room": ("area_id", "name", "bounds_enu", "floor"),
    "zone": ("parent_room", "name", "bounds_enu"),
    "wall": ("a", "b", "rf_attenuation_db"),
    "doorway": ("center", "width_m"),
    "sensor": ("device_id", "position", "modality"),
    "rf_link": ("tx", "rx", "link_group_id", "center_freq_mhz"),
    "person_track": ("track_id", "last_position", "reid_embedding_ref"),
    "object_anchor": ("position", "anchor_kind", "confidence"),
    "event": ("event_type", "at_unix_ms", "located_in"),
    "semantic_state": ("statement", "confidence", "provenance", "valid_from_unix_ms"),
}

_EDGE_FIELDS: dict[str, tuple[str, ...]] = {
    "observes": ("quality", "last_seen_unix_ms"),
    "located_in": ("since_unix_ms",),
    "adjacent_to": ("via_doorway",),
    "supports": ("strength",),
    "contradicts": ("magnitude", "flag"),
    "derived_from": ("evidence",),
    "privacy_limited_by": ("mode", "action", "allowed"),
}


def _coerce_vertices(vertices: Sequence[Sequence[float]]) -> tuple[tuple[float, float], ...]:
    coerced: list[tuple[float, float]] = []
    for vertex in vertices:
        if len(vertex) != 2:
            raise ValueError("polygon vertices must contain exactly east and north")
        coerced.append((float(vertex[0]), float(vertex[1])))
    return tuple(coerced)


def _point_in_polygon(px: float, py: float, vertices: Sequence[tuple[float, float]]) -> bool:
    inside = False
    j = len(vertices) - 1
    for i, (xi, yi) in enumerate(vertices):
        xj, yj = vertices[j]
        if (yi > py) != (yj > py):
            x_intersection = (xj - xi) * (py - yi) / (yj - yi) + xi
            if px < x_intersection:
                inside = not inside
        j = i
    return inside


def _coerce_enum(enum_type: type[Enum], value: Enum | str) -> Enum:
    if isinstance(value, enum_type):
        return value
    return enum_type(str(value))


def _to_plain(value: Any) -> Any:
    if isinstance(value, WorldId):
        return int(value)
    if isinstance(value, EnuPoint):
        return value.to_dict()
    if isinstance(value, ZoneBoundsEnu):
        return value.to_dict()
    if isinstance(value, SemanticProvenance):
        return value.to_dict()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_to_plain(item) for item in value]
    return value


__all__ = [
    "AnchorKind",
    "EnuPoint",
    "SemanticProvenance",
    "SensorModality",
    "WorldEdge",
    "WorldId",
    "WorldNode",
    "ZoneBoundsEnu",
]
