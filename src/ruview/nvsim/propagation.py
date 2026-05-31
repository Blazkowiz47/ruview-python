"""Magnetic field synthesis and material attenuation for the NV simulator."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ruview.nvsim.scene import (
    CurrentLoop,
    DipoleSource,
    FerrousObject,
    Scene,
    Vec3,
    coerce_vec3,
)

MU_0 = 4.0 * math.pi * 1.0e-7
R_MIN_M = 1.0e-3


class Material(str, Enum):
    """Material categories with behavior-level attenuation defaults."""

    AIR = "air"
    DRYWALL = "drywall"
    BRICK = "brick"
    CONCRETE_DRY = "concrete_dry"
    REINFORCED_CONCRETE = "reinforced_concrete"
    SHEET_STEEL = "sheet_steel"

    @classmethod
    def from_value(cls, value: "Material | str") -> "Material":
        """Return a material enum from an enum instance or value/name string."""

        if isinstance(value, cls):
            return value
        lowered = str(value).lower()
        try:
            return cls(lowered)
        except ValueError:
            return cls[lowered.upper()]


@dataclass(frozen=True)
class LosSegment:
    """One material slab along a source-to-sensor line of sight."""

    material: Material
    path_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "material", Material.from_value(self.material))
        object.__setattr__(self, "path_m", float(self.path_m))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""

        return {"material": self.material.value, "path_m": self.path_m}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LosSegment":
        """Build a segment from JSON-compatible data."""

        return cls(material=Material.from_value(data["material"]), path_m=data["path_m"])


def material_loss_db_per_m(material: Material | str) -> float:
    """Return additional per-metre magnetic attenuation in decibels."""

    material = Material.from_value(material)
    if material in {Material.AIR, Material.DRYWALL, Material.BRICK}:
        return 0.0
    if material is Material.CONCRETE_DRY:
        return 0.5
    if material is Material.REINFORCED_CONCRETE:
        return 20.0
    if material is Material.SHEET_STEEL:
        return 100.0
    raise ValueError(f"unknown material {material!r}")


def material_is_heavy(material: Material | str) -> bool:
    """Return true for low-confidence or strong attenuation materials."""

    material = Material.from_value(material)
    return material in {Material.REINFORCED_CONCRETE, Material.SHEET_STEEL}


def attenuate_field(
    b_in: ArrayLike,
    segments: Sequence[LosSegment | Mapping[str, Any]],
) -> tuple[Vec3, bool]:
    """Apply line-of-sight material attenuation to a magnetic field vector."""

    total_db = 0.0
    heavy = False
    for segment_like in segments:
        segment = (
            segment_like
            if isinstance(segment_like, LosSegment)
            else LosSegment.from_dict(segment_like)
        )
        if not math.isfinite(segment.path_m) or segment.path_m <= 0.0:
            continue
        total_db += segment.path_m * material_loss_db_per_m(segment.material)
        heavy = heavy or material_is_heavy(segment.material)
    scale = 10.0 ** (-total_db / 20.0)
    return coerce_vec3(_as_array3(b_in) * scale), heavy


def dipole_field(dipole: DipoleSource, sensor_pos: ArrayLike) -> tuple[Vec3, bool]:
    """Return the field from a dipole at ``sensor_pos`` and a near-field flag."""

    r = _as_array3(sensor_pos) - _as_array3(dipole.position)
    r_norm = float(np.linalg.norm(r))
    if r_norm < R_MIN_M:
        return (0.0, 0.0, 0.0), True

    r_hat = r / r_norm
    moment = _as_array3(dipole.moment)
    bracket = 3.0 * float(np.dot(moment, r_hat)) * r_hat - moment
    coefficient = MU_0 / (4.0 * math.pi * r_norm**3)
    return coerce_vec3(bracket * coefficient), False


def current_loop_field(loop: CurrentLoop, sensor_pos: ArrayLike) -> tuple[Vec3, bool]:
    """Return the Biot-Savart field from a discretised circular current loop."""

    if loop.radius == 0.0 or loop.current == 0.0:
        return (0.0, 0.0, 0.0), False

    normal = _normalise(_as_array3(loop.normal))
    u, v = _orthonormal_basis(normal)
    centre = _as_array3(loop.centre)
    sensor = _as_array3(sensor_pos)
    n_segments = max(8, int(loop.n_segments))
    total = np.zeros(3, dtype=float)
    near_field = False
    two_pi = 2.0 * math.pi

    for index in range(n_segments):
        theta_a = (index / n_segments) * two_pi
        theta_b = ((index + 1) / n_segments) * two_pi
        p_a = centre + loop.radius * (math.cos(theta_a) * u + math.sin(theta_a) * v)
        p_b = centre + loop.radius * (math.cos(theta_b) * u + math.sin(theta_b) * v)
        midpoint = 0.5 * (p_a + p_b)
        dl = p_b - p_a
        r = sensor - midpoint
        r_norm = float(np.linalg.norm(r))
        if r_norm < R_MIN_M:
            near_field = True
            continue
        r_hat = r / r_norm
        coefficient = MU_0 * loop.current / (4.0 * math.pi * r_norm**2)
        total += coefficient * np.cross(dl, r_hat)

    return coerce_vec3(total), near_field


def ferrous_field(
    obj: FerrousObject,
    ambient_b: ArrayLike,
    sensor_pos: ArrayLike,
) -> tuple[Vec3, bool]:
    """Return the field from a linearly induced ferrous dipole."""

    h_ambient = _as_array3(ambient_b) / MU_0
    induced_moment = h_ambient * obj.susceptibility * obj.volume
    induced = DipoleSource(position=obj.position, moment=coerce_vec3(induced_moment))
    return dipole_field(induced, sensor_pos)


def scene_field_at(scene: Scene, sensor_pos: ArrayLike) -> tuple[Vec3, bool]:
    """Return the total field from all scene sources at one sensor position."""

    total = np.zeros(3, dtype=float)
    near_field = False
    for dipole in scene.dipoles:
        field, near = dipole_field(dipole, sensor_pos)
        total += _as_array3(field)
        near_field = near_field or near
    for loop in scene.loops:
        field, near = current_loop_field(loop, sensor_pos)
        total += _as_array3(field)
        near_field = near_field or near
    for obj in scene.ferrous:
        field, near = ferrous_field(obj, scene.ambient_field, sensor_pos)
        total += _as_array3(field)
        near_field = near_field or near
    return coerce_vec3(total), near_field


def scene_field_at_sensors(scene: Scene) -> list[tuple[Vec3, bool]]:
    """Return total fields for every sensor in scene order."""

    return [scene_field_at(scene, sensor) for sensor in scene.sensors]


def _as_array3(value: ArrayLike) -> np.ndarray:
    return np.asarray(coerce_vec3(value), dtype=float)


def _normalise(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm < 1.0e-20:
        return np.asarray((0.0, 0.0, 1.0), dtype=float)
    return value / norm


def _orthonormal_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pick = np.asarray((1.0, 0.0, 0.0) if abs(normal[0]) < 0.9 else (0.0, 1.0, 0.0))
    u = _normalise(np.cross(pick, normal))
    v = np.cross(normal, u)
    return u, v
