"""CSI frame contracts ported from ``wifi-densepose-core``."""

from __future__ import annotations

import math
import struct
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import IntEnum
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from ruview.core.canonical import witness_hash_bytes
from ruview.core.errors import ValidationError


@dataclass(frozen=True)
class ComplexSample:
    """Canonical complex sample encoded as little-endian ``f64 re || f64 im``."""

    re: float
    im: float

    @classmethod
    def from_complex(cls, value: complex | np.complexfloating) -> "ComplexSample":
        z = complex(value)
        return cls(float(z.real), float(z.imag))

    @classmethod
    def from_le_bytes(cls, payload: bytes) -> "ComplexSample":
        if len(payload) != 16:
            raise ValidationError(f"ComplexSample requires 16 bytes, got {len(payload)}")
        re, im = struct.unpack("<dd", payload)
        return cls(re, im)

    def to_le_bytes(self) -> bytes:
        return struct.pack("<dd", self.re, self.im)

    def norm(self) -> float:
        return math.hypot(self.re, self.im)

    def arg(self) -> float:
        return math.atan2(self.im, self.re)

    def as_complex(self) -> complex:
        return complex(self.re, self.im)


@dataclass(frozen=True)
class FrameId:
    """Unique identifier for a CSI frame or pose estimate."""

    value: uuid.UUID = field(default_factory=uuid.uuid4)

    @classmethod
    def new(cls) -> "FrameId":
        return cls()

    @classmethod
    def from_uuid(cls, value: uuid.UUID | str) -> "FrameId":
        return cls(value if isinstance(value, uuid.UUID) else uuid.UUID(str(value)))

    @property
    def bytes(self) -> bytes:
        return self.value.bytes

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class DeviceId:
    """Stable identifier for a CSI capture device."""

    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValidationError("DeviceId must not be empty")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, order=True)
class Timestamp:
    """Unix timestamp with nanosecond precision."""

    seconds: int
    nanos: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.nanos < 1_000_000_000:
            raise ValidationError(f"Timestamp nanos must be in [0, 1e9), got {self.nanos}")

    @classmethod
    def now(cls) -> "Timestamp":
        nanos = time.time_ns()
        return cls(nanos // 1_000_000_000, nanos % 1_000_000_000)

    @classmethod
    def from_datetime(cls, value: datetime) -> "Timestamp":
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        value = value.astimezone(UTC)
        return cls(int(value.timestamp()), value.microsecond * 1_000)

    def to_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.seconds + self.nanos / 1_000_000_000, tz=UTC)

    def as_nanos(self) -> int:
        return self.seconds * 1_000_000_000 + self.nanos

    def duration_since(self, earlier: "Timestamp") -> float:
        return (self.as_nanos() - earlier.as_nanos()) / 1_000_000_000.0


class FrequencyBand(IntEnum):
    """WiFi frequency band."""

    BAND_2_4_GHZ = 0
    BAND_5_GHZ = 1
    BAND_6_GHZ = 2

    @classmethod
    def from_value(cls, value: "FrequencyBand | str | int") -> "FrequencyBand":
        if isinstance(value, FrequencyBand):
            return value
        if isinstance(value, str):
            normalized = value.lower().replace("-", "_").replace(".", "_")
            aliases = {
                "band2_4ghz": cls.BAND_2_4_GHZ,
                "band_2_4_ghz": cls.BAND_2_4_GHZ,
                "2_4ghz": cls.BAND_2_4_GHZ,
                "band5ghz": cls.BAND_5_GHZ,
                "band_5_ghz": cls.BAND_5_GHZ,
                "5ghz": cls.BAND_5_GHZ,
                "band6ghz": cls.BAND_6_GHZ,
                "band_6_ghz": cls.BAND_6_GHZ,
                "6ghz": cls.BAND_6_GHZ,
            }
            if normalized in aliases:
                return aliases[normalized]
        try:
            return cls(int(value))
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"Invalid frequency band: {value!r}") from exc

    def center_frequency_mhz(self) -> int:
        return {
            FrequencyBand.BAND_2_4_GHZ: 2437,
            FrequencyBand.BAND_5_GHZ: 5180,
            FrequencyBand.BAND_6_GHZ: 5975,
        }[self]

    def typical_subcarriers(self) -> int:
        return {
            FrequencyBand.BAND_2_4_GHZ: 56,
            FrequencyBand.BAND_5_GHZ: 114,
            FrequencyBand.BAND_6_GHZ: 234,
        }[self]


@dataclass(frozen=True)
class AntennaConfig:
    """Antenna layout for MIMO/SIMO CSI streams."""

    tx_antennas: int
    rx_antennas: int
    spacing_mm: float | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.tx_antennas <= 255:
            raise ValidationError("tx_antennas must be in [1, 255]")
        if not 1 <= self.rx_antennas <= 255:
            raise ValidationError("rx_antennas must be in [1, 255]")

    @classmethod
    def simo_1x3(cls) -> "AntennaConfig":
        return cls(1, 3)

    @classmethod
    def mimo_2x2(cls) -> "AntennaConfig":
        return cls(2, 2)

    @classmethod
    def mimo_3x3(cls) -> "AntennaConfig":
        return cls(3, 3)

    def with_spacing(self, spacing_mm: float) -> "AntennaConfig":
        return AntennaConfig(self.tx_antennas, self.rx_antennas, spacing_mm)

    def spatial_streams(self) -> int:
        return self.tx_antennas * self.rx_antennas


@dataclass
class CsiMetadata:
    """Metadata associated with one CSI frame."""

    device_id: DeviceId | str
    frequency_band: FrequencyBand | str | int
    channel: int
    timestamp: Timestamp = field(default_factory=Timestamp.now)
    bandwidth_mhz: int = 20
    antenna_config: AntennaConfig = field(default_factory=AntennaConfig.simo_1x3)
    rssi_dbm: int = -50
    noise_floor_dbm: int = -90
    sequence_number: int = 0
    calibration_id: uuid.UUID | None = None
    model_id: int = 0
    model_version: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.device_id, str):
            self.device_id = DeviceId(self.device_id)
        self.frequency_band = FrequencyBand.from_value(self.frequency_band)
        if not 0 <= self.channel <= 255:
            raise ValidationError("channel must fit in u8")
        if not 0 <= self.bandwidth_mhz <= 65_535:
            raise ValidationError("bandwidth_mhz must fit in u16")
        if not -128 <= self.rssi_dbm <= 127:
            raise ValidationError("rssi_dbm must fit in i8")
        if not -128 <= self.noise_floor_dbm <= 127:
            raise ValidationError("noise_floor_dbm must fit in i8")
        if not 0 <= self.sequence_number <= 4_294_967_295:
            raise ValidationError("sequence_number must fit in u32")
        if isinstance(self.calibration_id, str):
            self.calibration_id = uuid.UUID(self.calibration_id)
        if not 0 <= self.model_id <= 65_535:
            raise ValidationError("model_id must fit in u16")
        if not 0 <= self.model_version <= 65_535:
            raise ValidationError("model_version must fit in u16")

    def set_calibration(self, calibration_id: uuid.UUID | str) -> None:
        self.calibration_id = (
            calibration_id if isinstance(calibration_id, uuid.UUID) else uuid.UUID(str(calibration_id))
        )

    def set_model(self, model_id: int, model_version: int) -> None:
        if not 0 <= model_id <= 65_535:
            raise ValidationError("model_id must fit in u16")
        if not 0 <= model_version <= 65_535:
            raise ValidationError("model_version must fit in u16")
        self.model_id = model_id
        self.model_version = model_version

    def snr_db(self) -> float:
        return float(self.rssi_dbm - self.noise_floor_dbm)


@dataclass
class CsiFrame:
    """One CSI frame with complex data shaped ``[spatial_streams, subcarriers]``."""

    metadata: CsiMetadata
    data: NDArray[np.complexfloating]
    id: FrameId = field(default_factory=FrameId.new)
    amplitude: NDArray[np.float64] = field(init=False, repr=False)
    phase: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.metadata = deepcopy(self.metadata)
        data = np.asarray(self.data, dtype=np.complex128)
        if data.ndim != 2:
            raise ValidationError(f"CsiFrame data must be 2D, got shape {data.shape}")
        self.data = data
        self.amplitude = np.abs(data)
        self.phase = np.angle(data)

    def num_spatial_streams(self) -> int:
        return int(self.data.shape[0])

    def num_subcarriers(self) -> int:
        return int(self.data.shape[1])

    def mean_amplitude(self) -> float:
        return float(np.mean(self.amplitude)) if self.amplitude.size else 0.0

    def amplitude_variance(self) -> float:
        return float(np.var(self.amplitude)) if self.amplitude.size else 0.0

    def data_complex_samples(self) -> Iterator[ComplexSample]:
        for sample in np.ravel(self.data, order="C"):
            yield ComplexSample.from_complex(sample)

    def to_canonical_bytes(self) -> bytes:
        """Return bytes matching the ADR-136 Rust ``CanonicalFrame`` layout."""

        metadata = self.metadata
        chunks: list[bytes] = [
            self.id.bytes,
            struct.pack("<q", metadata.timestamp.seconds),
            struct.pack("<I", metadata.timestamp.nanos),
        ]

        device = str(metadata.device_id).encode("utf-8")
        chunks.extend(
            [
                struct.pack("<I", len(device)),
                device,
                struct.pack("<B", int(metadata.frequency_band)),
                struct.pack("<B", metadata.channel),
                struct.pack("<H", metadata.bandwidth_mhz),
                struct.pack("<B", metadata.antenna_config.tx_antennas),
                struct.pack("<B", metadata.antenna_config.rx_antennas),
            ]
        )

        if metadata.antenna_config.spacing_mm is None:
            chunks.extend([b"\x00", b"\x00\x00\x00\x00"])
        else:
            chunks.extend([b"\x01", struct.pack("<f", metadata.antenna_config.spacing_mm)])

        chunks.extend(
            [
                struct.pack("<b", metadata.rssi_dbm),
                struct.pack("<b", metadata.noise_floor_dbm),
                struct.pack("<I", metadata.sequence_number),
                metadata.calibration_id.bytes if metadata.calibration_id else b"\x00" * 16,
                struct.pack("<H", metadata.model_id),
                struct.pack("<H", metadata.model_version),
                struct.pack("<I", self.num_spatial_streams()),
                struct.pack("<I", self.num_subcarriers()),
            ]
        )

        chunks.extend(sample.to_le_bytes() for sample in self.data_complex_samples())
        return b"".join(chunks)

    def witness_hash(self) -> bytes:
        return witness_hash_bytes(self.to_canonical_bytes())
