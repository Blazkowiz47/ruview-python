"""Deterministic NV simulator pipeline, frame bytes, and SHA-256 witness."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ruview.nvsim.propagation import LosSegment, attenuate_field, scene_field_at
from ruview.nvsim.scene import Scene, Vec3, coerce_vec3

MAG_FRAME_MAGIC = 0xC51A_6E70
MAG_FRAME_VERSION = 1
_MAG_FRAME_STRUCT = struct.Struct("<IHHHHQffffffff8x")
MAG_FRAME_BYTES = _MAG_FRAME_STRUCT.size

FLAG_SATURATION_NEAR_FIELD = 1 << 0
FLAG_ADC_SATURATED = 1 << 1
FLAG_HEAVY_ATTENUATION = 1 << 2
FLAG_NOISE_DISABLED = 1 << 3

ADC_FULL_SCALE_T = 10.0e-6
ADC_BITS = 16
DEFAULT_SAMPLE_RATE_HZ = 10_000.0
DEFAULT_NOISE_STD_T = 5.0e-10


def adc_lsb_t(full_scale_t: float = ADC_FULL_SCALE_T, bits: int = ADC_BITS) -> float:
    """Return one signed ADC least significant bit in tesla."""

    max_code = (1 << (int(bits) - 1)) - 1
    return float(full_scale_t) / max_code


def adc_quantise(
    b_in_t: float,
    *,
    full_scale_t: float = ADC_FULL_SCALE_T,
    bits: int = ADC_BITS,
) -> tuple[int, bool]:
    """Quantise one magnetic-field sample to a signed ADC code."""

    max_code = (1 << (int(bits) - 1)) - 1
    min_code = -max_code
    code = int(np.rint(float(b_in_t) / adc_lsb_t(full_scale_t, bits)))
    if code >= max_code:
        return max_code, True
    if code <= min_code:
        return min_code, True
    return code, False


def adc_dequantise(
    code: int,
    *,
    full_scale_t: float = ADC_FULL_SCALE_T,
    bits: int = ADC_BITS,
) -> float:
    """Convert a signed ADC code back to tesla."""

    return int(code) * adc_lsb_t(full_scale_t, bits)


@dataclass(frozen=True)
class MagFrame:
    """Fixed-layout magnetic frame emitted per sensor per timestep."""

    flags: int
    sensor_id: int
    t_us: int
    b_pt: Vec3
    sigma_pt: Vec3 = (0.0, 0.0, 0.0)
    noise_floor_pt_sqrt_hz: float = 0.0
    temperature_k: float = 295.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "flags", int(self.flags) & 0xFFFF)
        object.__setattr__(self, "sensor_id", int(self.sensor_id) & 0xFFFF)
        object.__setattr__(self, "t_us", int(self.t_us))
        object.__setattr__(self, "b_pt", coerce_vec3(self.b_pt, name="b_pt"))
        object.__setattr__(self, "sigma_pt", coerce_vec3(self.sigma_pt, name="sigma_pt"))
        object.__setattr__(
            self,
            "noise_floor_pt_sqrt_hz",
            float(self.noise_floor_pt_sqrt_hz),
        )
        object.__setattr__(self, "temperature_k", float(self.temperature_k))
        if self.t_us < 0:
            raise ValueError("t_us must be non-negative")

    def has_flag(self, flag_bit: int) -> bool:
        """Return true when ``flag_bit`` is set."""

        return bool(self.flags & int(flag_bit))

    def to_canonical_bytes(self) -> bytes:
        """Return deterministic little-endian 60-byte frame bytes."""

        return _MAG_FRAME_STRUCT.pack(
            MAG_FRAME_MAGIC,
            MAG_FRAME_VERSION,
            self.flags,
            self.sensor_id,
            0,
            self.t_us,
            float(self.b_pt[0]),
            float(self.b_pt[1]),
            float(self.b_pt[2]),
            float(self.sigma_pt[0]),
            float(self.sigma_pt[1]),
            float(self.sigma_pt[2]),
            float(self.noise_floor_pt_sqrt_hz),
            float(self.temperature_k),
        )

    def witness_hash(self) -> bytes:
        """Return the SHA-256 hash of this frame's canonical bytes."""

        return hashlib.sha256(self.to_canonical_bytes()).digest()

    @classmethod
    def from_canonical_bytes(cls, payload: bytes) -> "MagFrame":
        """Decode one v1 magnetic frame."""

        if len(payload) != MAG_FRAME_BYTES:
            raise ValueError(f"frame must be {MAG_FRAME_BYTES} bytes")
        unpacked = _MAG_FRAME_STRUCT.unpack(payload)
        magic, version = unpacked[0], unpacked[1]
        if magic != MAG_FRAME_MAGIC:
            raise ValueError(f"bad magic 0x{magic:08x}")
        if version != MAG_FRAME_VERSION:
            raise ValueError(f"unsupported frame version {version}")
        return cls(
            flags=unpacked[2],
            sensor_id=unpacked[3],
            t_us=unpacked[5],
            b_pt=(unpacked[6], unpacked[7], unpacked[8]),
            sigma_pt=(unpacked[9], unpacked[10], unpacked[11]),
            noise_floor_pt_sqrt_hz=unpacked[12],
            temperature_k=unpacked[13],
        )


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the compact behavior-level simulator pipeline."""

    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ
    dt_s: float | None = None
    adc_full_scale_t: float = ADC_FULL_SCALE_T
    adc_bits: int = ADC_BITS
    noise_std_t: float = DEFAULT_NOISE_STD_T
    noise_enabled: bool = True
    los_segments: tuple[LosSegment, ...] = ()
    temperature_k: float = 295.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_rate_hz", float(self.sample_rate_hz))
        object.__setattr__(self, "adc_full_scale_t", float(self.adc_full_scale_t))
        object.__setattr__(self, "adc_bits", int(self.adc_bits))
        object.__setattr__(self, "noise_std_t", float(self.noise_std_t))
        object.__setattr__(self, "noise_enabled", bool(self.noise_enabled))
        object.__setattr__(self, "temperature_k", float(self.temperature_k))
        if self.dt_s is not None:
            object.__setattr__(self, "dt_s", float(self.dt_s))
        object.__setattr__(self, "los_segments", _coerce_los_segments(self.los_segments))
        if self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive")
        if self.dt_s is not None and self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if self.adc_full_scale_t <= 0.0:
            raise ValueError("adc_full_scale_t must be positive")
        if self.adc_bits < 2:
            raise ValueError("adc_bits must be at least 2")
        if self.noise_std_t < 0.0:
            raise ValueError("noise_std_t must be non-negative")

    @property
    def effective_dt_s(self) -> float:
        """Return the integration period used for frame timestamps."""

        return self.dt_s if self.dt_s is not None else 1.0 / self.sample_rate_hz


class Pipeline:
    """Forward-only deterministic simulator over a scene and fixed config."""

    def __init__(
        self,
        scene: Scene,
        config: PipelineConfig | None = None,
        *,
        seed: int = 0,
    ) -> None:
        self.scene = scene if isinstance(scene, Scene) else Scene.from_dict(scene)
        self.config = config or PipelineConfig()
        self.seed = int(seed) & 0xFFFF_FFFF_FFFF_FFFF

    def run(self, n_samples: int) -> list[MagFrame]:
        """Run the pipeline for ``n_samples`` across all scene sensors."""

        n_samples = int(n_samples)
        if n_samples < 0:
            raise ValueError("n_samples must be non-negative")

        rng = np.random.default_rng(self.seed)
        dt_us = int(round(self.config.effective_dt_s * 1.0e6))
        frames: list[MagFrame] = []

        for sample_index in range(n_samples):
            t_us = sample_index * dt_us
            for sensor_id, sensor_pos in enumerate(self.scene.sensors):
                field_t, near_field = scene_field_at(self.scene, sensor_pos)
                attenuated_t, heavy = attenuate_field(field_t, self.config.los_segments)
                measured_t = np.asarray(attenuated_t, dtype=float)

                noise_disabled = (
                    not self.config.noise_enabled or self.config.noise_std_t == 0.0
                )
                sigma_t = 0.0 if noise_disabled else self.config.noise_std_t
                if not noise_disabled:
                    measured_t = measured_t + rng.normal(0.0, sigma_t, size=3)

                b_pt, adc_saturated = self._quantise_to_pt(measured_t)
                flags = 0
                if near_field:
                    flags |= FLAG_SATURATION_NEAR_FIELD
                if adc_saturated:
                    flags |= FLAG_ADC_SATURATED
                if heavy:
                    flags |= FLAG_HEAVY_ATTENUATION
                if noise_disabled:
                    flags |= FLAG_NOISE_DISABLED

                sigma_pt = (sigma_t * 1.0e12, sigma_t * 1.0e12, sigma_t * 1.0e12)
                frames.append(
                    MagFrame(
                        flags=flags,
                        sensor_id=sensor_id,
                        t_us=t_us,
                        b_pt=b_pt,
                        sigma_pt=sigma_pt,
                        noise_floor_pt_sqrt_hz=sigma_t * 1.0e12,
                        temperature_k=self.config.temperature_k,
                    )
                )

        return frames

    def run_with_witness(self, n_samples: int) -> tuple[list[MagFrame], bytes]:
        """Return frames and the SHA-256 witness over their canonical bytes."""

        frames = self.run(n_samples)
        return frames, sha256_witness(frames)

    def _quantise_to_pt(self, measured_t: np.ndarray) -> tuple[Vec3, bool]:
        values_pt = []
        saturated = False
        for component_t in measured_t:
            code, did_saturate = adc_quantise(
                float(component_t),
                full_scale_t=self.config.adc_full_scale_t,
                bits=self.config.adc_bits,
            )
            saturated = saturated or did_saturate
            values_pt.append(
                adc_dequantise(
                    code,
                    full_scale_t=self.config.adc_full_scale_t,
                    bits=self.config.adc_bits,
                )
                * 1.0e12
            )
        return coerce_vec3(values_pt), saturated


def sha256_witness(frames: Sequence[MagFrame]) -> bytes:
    """Return SHA-256 over the concatenated canonical bytes of ``frames``."""

    hasher = hashlib.sha256()
    for frame in frames:
        hasher.update(frame.to_canonical_bytes())
    return hasher.digest()


def _coerce_los_segments(
    segments: Sequence[LosSegment | Mapping[str, Any]],
) -> tuple[LosSegment, ...]:
    return tuple(
        segment if isinstance(segment, LosSegment) else LosSegment.from_dict(segment)
        for segment in segments
    )
