"""Deterministic magnetic scene primitives for the NV simulator subset."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

Vec3 = tuple[float, float, float]


def coerce_vec3(value: ArrayLike, *, name: str = "vec3") -> Vec3:
    """Return ``value`` as a finite 3-vector of Python floats."""

    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def vec3_to_list(value: ArrayLike) -> list[float]:
    """Return a JSON-ready list representation of a 3-vector."""

    return list(coerce_vec3(value))


@dataclass(frozen=True)
class DipoleSource:
    """A point magnetic dipole in SI units."""

    position: Vec3
    moment: Vec3

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", coerce_vec3(self.position, name="position"))
        object.__setattr__(self, "moment", coerce_vec3(self.moment, name="moment"))

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical JSON-compatible mapping."""

        return {
            "position": vec3_to_list(self.position),
            "moment": vec3_to_list(self.moment),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DipoleSource":
        """Build a dipole from JSON-compatible data."""

        return cls(position=data["position"], moment=data["moment"])


@dataclass(frozen=True)
class CurrentLoop:
    """A circular current loop used as a deterministic Biot-Savart source."""

    centre: Vec3
    normal: Vec3
    radius: float
    current: float
    n_segments: int = 64

    def __post_init__(self) -> None:
        object.__setattr__(self, "centre", coerce_vec3(self.centre, name="centre"))
        object.__setattr__(self, "normal", coerce_vec3(self.normal, name="normal"))
        object.__setattr__(self, "radius", float(self.radius))
        object.__setattr__(self, "current", float(self.current))
        object.__setattr__(self, "n_segments", int(self.n_segments))
        if self.radius < 0.0:
            raise ValueError("radius must be non-negative")
        if self.n_segments <= 0:
            raise ValueError("n_segments must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical JSON-compatible mapping."""

        return {
            "centre": vec3_to_list(self.centre),
            "normal": vec3_to_list(self.normal),
            "radius": self.radius,
            "current": self.current,
            "n_segments": self.n_segments,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CurrentLoop":
        """Build a loop from JSON-compatible data."""

        return cls(
            centre=data["centre"],
            normal=data["normal"],
            radius=data["radius"],
            current=data["current"],
            n_segments=data.get("n_segments", 64),
        )


@dataclass(frozen=True)
class FerrousObject:
    """A linearly induced ferrous object, re-radiated as a dipole."""

    position: Vec3
    volume: float
    susceptibility: float = 5000.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", coerce_vec3(self.position, name="position"))
        object.__setattr__(self, "volume", float(self.volume))
        object.__setattr__(self, "susceptibility", float(self.susceptibility))
        if self.volume < 0.0:
            raise ValueError("volume must be non-negative")

    @classmethod
    def steel(cls, position: ArrayLike, volume: float) -> "FerrousObject":
        """Return a low-carbon-steel default ferrous object."""

        return cls(position=coerce_vec3(position, name="position"), volume=volume)

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical JSON-compatible mapping."""

        return {
            "position": vec3_to_list(self.position),
            "volume": self.volume,
            "susceptibility": self.susceptibility,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FerrousObject":
        """Build a ferrous object from JSON-compatible data."""

        return cls(
            position=data["position"],
            volume=data["volume"],
            susceptibility=data.get("susceptibility", 5000.0),
        )


@dataclass(frozen=True)
class Scene:
    """Ground-truth magnetic primitives and sensor positions."""

    dipoles: tuple[DipoleSource, ...] = ()
    loops: tuple[CurrentLoop, ...] = ()
    ferrous: tuple[FerrousObject, ...] = ()
    sensors: tuple[Vec3, ...] = ()
    ambient_field: Vec3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dipoles",
            _coerce_objects(self.dipoles, DipoleSource, "dipoles"),
        )
        object.__setattr__(
            self,
            "loops",
            _coerce_objects(self.loops, CurrentLoop, "loops"),
        )
        object.__setattr__(
            self,
            "ferrous",
            _coerce_objects(self.ferrous, FerrousObject, "ferrous"),
        )
        object.__setattr__(
            self,
            "sensors",
            tuple(coerce_vec3(sensor, name="sensor") for sensor in self.sensors),
        )
        object.__setattr__(
            self,
            "ambient_field",
            coerce_vec3(self.ambient_field, name="ambient_field"),
        )

    @classmethod
    def new(cls) -> "Scene":
        """Return an empty scene."""

        return cls()

    @property
    def source_count(self) -> int:
        """Return the total count of behavior-level source primitives."""

        return len(self.dipoles) + len(self.loops) + len(self.ferrous)

    def n_sources(self) -> int:
        """Rust-compatible source-count helper."""

        return self.source_count

    def with_dipole(self, dipole: DipoleSource) -> "Scene":
        """Return a copy with ``dipole`` appended."""

        return replace(self, dipoles=self.dipoles + (dipole,))

    def with_loop(self, loop: CurrentLoop) -> "Scene":
        """Return a copy with ``loop`` appended."""

        return replace(self, loops=self.loops + (loop,))

    def with_ferrous(self, ferrous: FerrousObject) -> "Scene":
        """Return a copy with ``ferrous`` appended."""

        return replace(self, ferrous=self.ferrous + (ferrous,))

    def with_sensor(self, position: ArrayLike) -> "Scene":
        """Return a copy with a sensor position appended."""

        return replace(self, sensors=self.sensors + (coerce_vec3(position),))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping in the simulator schema."""

        return {
            "ambient_field": vec3_to_list(self.ambient_field),
            "dipoles": [dipole.to_dict() for dipole in self.dipoles],
            "ferrous": [obj.to_dict() for obj in self.ferrous],
            "loops": [loop.to_dict() for loop in self.loops],
            "sensors": [vec3_to_list(sensor) for sensor in self.sensors],
        }

    def to_canonical_json(self) -> str:
        """Return deterministic, whitespace-free JSON for content addressing."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Scene":
        """Build a scene from JSON-compatible data."""

        return cls(
            dipoles=tuple(DipoleSource.from_dict(item) for item in data.get("dipoles", ())),
            loops=tuple(CurrentLoop.from_dict(item) for item in data.get("loops", ())),
            ferrous=tuple(
                FerrousObject.from_dict(item) for item in data.get("ferrous", ())
            ),
            sensors=tuple(data.get("sensors", ())),
            ambient_field=data.get("ambient_field", (0.0, 0.0, 0.0)),
        )

    @classmethod
    def from_canonical_json(cls, payload: str) -> "Scene":
        """Parse a canonical scene JSON string."""

        return cls.from_dict(json.loads(payload))


def _coerce_objects(
    values: Sequence[Any],
    cls: type[DipoleSource] | type[CurrentLoop] | type[FerrousObject],
    name: str,
) -> tuple[Any, ...]:
    coerced = []
    for item in values:
        if isinstance(item, cls):
            coerced.append(item)
        elif isinstance(item, Mapping):
            coerced.append(cls.from_dict(item))
        else:
            raise TypeError(f"{name} entries must be {cls.__name__} or mappings")
    return tuple(coerced)
