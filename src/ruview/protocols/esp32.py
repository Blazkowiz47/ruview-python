"""ESP32 host-side binary packet parsers.

The layouts mirror the ESP32 firmware under
``firmware/esp32-csi-node/main`` in the RuView reference repository:

* ADR-018 raw CSI packets from ``csi_collector.c``
* Edge vitals packets from ``edge_processing.h``
* WASM output packets from ``wasm_runtime.h``
* ADR-110 sync packets from ``csi_collector.c``
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import TypeAlias

RAW_CSI_MAGIC = 0xC511_0001
EDGE_VITALS_MAGIC = 0xC511_0002
WASM_EVENT_MAGIC = 0xC511_0004
SYNC_MAGIC = 0xC511_A110

CSI_HEADER_SIZE = 20
EDGE_VITALS_PACKET_SIZE = 32
SYNC_PACKET_SIZE = 32
WASM_OUTPUT_HEADER_SIZE = 8
WASM_EVENT_SIZE = 5
WASM_MAX_EVENTS = 16

_MAGIC = struct.Struct("<I")
_RAW_CSI_HEADER = struct.Struct("<IBBHIIbbBB")
_EDGE_VITALS = struct.Struct("<IBBHIbB2sffII")
_WASM_OUTPUT_HEADER = struct.Struct("<IBBH")
_WASM_EVENT = struct.Struct("<Bf")
_SYNC = struct.Struct("<IBBBBQQII")


class Esp32ProtocolError(ValueError):
    """Base class for ESP32 packet parse failures."""


class UnknownPacketMagic(Esp32ProtocolError):
    """Raised when a packet's first four bytes are not a supported magic."""

    def __init__(self, magic: int) -> None:
        self.magic = magic
        super().__init__(f"unknown ESP32 packet magic 0x{magic:08X}")


class TruncatedPacket(Esp32ProtocolError):
    """Raised when a packet is shorter than its declared fixed layout."""

    def __init__(self, *, expected: int, actual: int, context: str) -> None:
        self.expected = expected
        self.actual = actual
        self.context = context
        super().__init__(f"{context} requires at least {expected} bytes, got {actual}")


class InvalidPacket(Esp32ProtocolError):
    """Raised when a packet is well-sized but semantically invalid."""


@dataclass(frozen=True)
class RawCsiPacket:
    """ADR-018 raw CSI packet emitted by ``csi_serialize_frame``."""

    magic: int
    node_id: int
    n_antennas: int
    n_subcarriers: int
    frequency_mhz: int
    sequence_number: int
    rssi_dbm: int
    noise_floor_dbm: int
    ppdu_type: int
    flags: int
    iq_bytes: bytes
    trailing_bytes: bytes = b""

    @property
    def payload(self) -> bytes:
        """All bytes after the 20-byte header."""

        return self.iq_bytes + self.trailing_bytes

    @property
    def expected_iq_bytes(self) -> int:
        return self.n_antennas * self.n_subcarriers * 2

    @property
    def bandwidth_40mhz(self) -> bool:
        """ADR-018 byte 19 bit 0: 40 MHz channel width when set."""

        return bool(self.flags & 0x01)

    @property
    def stbc(self) -> bool:
        """ADR-018 byte 19 bit 2: STBC tagging from legacy ESP32 targets."""

        return bool(self.flags & 0x04)

    @property
    def sync_valid(self) -> bool:
        """ADR-018 byte 19 bit 4: cross-node sync transport is valid."""

        return bool(self.flags & 0x10)

    def iq_pairs(self) -> tuple[tuple[int, int], ...]:
        return decode_iq_pairs(self.iq_bytes)

    def amplitudes(self) -> tuple[float, ...]:
        return iq_amplitudes(self.iq_bytes)

    def phases(self) -> tuple[float, ...]:
        return iq_phases(self.iq_bytes)


@dataclass(frozen=True)
class EdgeVitalsPacket:
    """32-byte edge vitals packet emitted by ``edge_processing``."""

    magic: int
    node_id: int
    flags: int
    breathing_rate_raw: int
    heartrate_raw: int
    rssi_dbm: int
    n_persons: int
    reserved: bytes
    motion_energy: float
    presence_score: float
    timestamp_ms: int
    reserved2: int
    trailing_bytes: bytes = b""

    @property
    def presence(self) -> bool:
        return bool(self.flags & 0x01)

    @property
    def fall_detected(self) -> bool:
        return bool(self.flags & 0x02)

    @property
    def motion(self) -> bool:
        return bool(self.flags & 0x04)

    @property
    def breathing_rate_bpm(self) -> float:
        return self.breathing_rate_raw / 100.0

    @property
    def heartrate_bpm(self) -> float:
        return self.heartrate_raw / 10_000.0


@dataclass(frozen=True)
class WasmEvent:
    """One packed WASM output event: ``u8 event_type`` plus ``f32 value``."""

    event_type: int
    value: float
    raw_bytes: bytes


@dataclass(frozen=True)
class WasmEventPacket:
    """WASM output packet emitted by ``wasm_runtime``."""

    magic: int
    node_id: int
    module_id: int
    event_count: int
    events: tuple[WasmEvent, ...]
    event_payload: bytes
    trailing_bytes: bytes = b""


@dataclass(frozen=True)
class SyncPacket:
    """ADR-110 sync packet emitted periodically alongside raw CSI."""

    magic: int
    node_id: int
    protocol_version: int
    flags: int
    reserved: int
    local_us: int
    epoch_us: int
    sequence_high_water: int
    reserved2: int
    trailing_bytes: bytes = b""

    @property
    def is_leader(self) -> bool:
        return bool(self.flags & 0x01)

    @property
    def sync_valid(self) -> bool:
        return bool(self.flags & 0x02)

    @property
    def smoothed_offset_valid(self) -> bool:
        return bool(self.flags & 0x04)


Esp32Packet: TypeAlias = RawCsiPacket | EdgeVitalsPacket | WasmEventPacket | SyncPacket


def parse_packet(data: bytes) -> Esp32Packet:
    """Parse any supported ESP32 host-side packet by magic dispatch."""

    magic = packet_magic(data)
    if magic == RAW_CSI_MAGIC:
        return parse_raw_csi_packet(data)
    if magic == EDGE_VITALS_MAGIC:
        return parse_edge_vitals_packet(data)
    if magic == WASM_EVENT_MAGIC:
        return parse_wasm_event_packet(data)
    if magic == SYNC_MAGIC:
        return parse_sync_packet(data)
    raise UnknownPacketMagic(magic)


def parse_raw_csi_packet(data: bytes) -> RawCsiPacket:
    """Parse an ADR-018 raw CSI packet.

    Current firmware layout:
    ``magic, node_id, n_antennas, n_subcarriers, frequency_mhz,
    sequence, rssi, noise_floor, ppdu_type, flags`` followed by signed
    I/Q byte pairs.
    """

    _require_magic(data, RAW_CSI_MAGIC, "raw CSI packet")
    _require_len(data, CSI_HEADER_SIZE, "raw CSI header")
    (
        magic,
        node_id,
        n_antennas,
        n_subcarriers,
        frequency_mhz,
        sequence_number,
        rssi_dbm,
        noise_floor_dbm,
        ppdu_type,
        flags,
    ) = _RAW_CSI_HEADER.unpack_from(data)

    if n_antennas == 0:
        raise InvalidPacket("raw CSI packet declares zero antennas")
    if n_subcarriers == 0:
        raise InvalidPacket("raw CSI packet declares zero subcarriers")

    iq_len = n_antennas * n_subcarriers * 2
    expected_len = CSI_HEADER_SIZE + iq_len
    _require_len(data, expected_len, "raw CSI payload")

    return RawCsiPacket(
        magic=magic,
        node_id=node_id,
        n_antennas=n_antennas,
        n_subcarriers=n_subcarriers,
        frequency_mhz=frequency_mhz,
        sequence_number=sequence_number,
        rssi_dbm=rssi_dbm,
        noise_floor_dbm=noise_floor_dbm,
        ppdu_type=ppdu_type,
        flags=flags,
        iq_bytes=data[CSI_HEADER_SIZE:expected_len],
        trailing_bytes=data[expected_len:],
    )


def parse_edge_vitals_packet(data: bytes) -> EdgeVitalsPacket:
    """Parse a 32-byte edge vitals packet."""

    _require_magic(data, EDGE_VITALS_MAGIC, "edge vitals packet")
    _require_len(data, EDGE_VITALS_PACKET_SIZE, "edge vitals packet")
    (
        magic,
        node_id,
        flags,
        breathing_rate_raw,
        heartrate_raw,
        rssi_dbm,
        n_persons,
        reserved,
        motion_energy,
        presence_score,
        timestamp_ms,
        reserved2,
    ) = _EDGE_VITALS.unpack_from(data)

    return EdgeVitalsPacket(
        magic=magic,
        node_id=node_id,
        flags=flags,
        breathing_rate_raw=breathing_rate_raw,
        heartrate_raw=heartrate_raw,
        rssi_dbm=rssi_dbm,
        n_persons=n_persons,
        reserved=reserved,
        motion_energy=motion_energy,
        presence_score=presence_score,
        timestamp_ms=timestamp_ms,
        reserved2=reserved2,
        trailing_bytes=data[EDGE_VITALS_PACKET_SIZE:],
    )


def parse_wasm_event_packet(data: bytes) -> WasmEventPacket:
    """Parse a WASM output packet with packed event records."""

    _require_magic(data, WASM_EVENT_MAGIC, "WASM event packet")
    _require_len(data, WASM_OUTPUT_HEADER_SIZE, "WASM event header")
    magic, node_id, module_id, event_count = _WASM_OUTPUT_HEADER.unpack_from(data)

    if event_count > WASM_MAX_EVENTS:
        raise InvalidPacket(f"WASM event packet declares {event_count} events; max is {WASM_MAX_EVENTS}")

    payload_len = event_count * WASM_EVENT_SIZE
    expected_len = WASM_OUTPUT_HEADER_SIZE + payload_len
    _require_len(data, expected_len, "WASM event payload")

    events: list[WasmEvent] = []
    offset = WASM_OUTPUT_HEADER_SIZE
    for _ in range(event_count):
        raw = data[offset : offset + WASM_EVENT_SIZE]
        event_type, value = _WASM_EVENT.unpack(raw)
        events.append(WasmEvent(event_type=event_type, value=value, raw_bytes=raw))
        offset += WASM_EVENT_SIZE

    return WasmEventPacket(
        magic=magic,
        node_id=node_id,
        module_id=module_id,
        event_count=event_count,
        events=tuple(events),
        event_payload=data[WASM_OUTPUT_HEADER_SIZE:expected_len],
        trailing_bytes=data[expected_len:],
    )


def parse_sync_packet(data: bytes) -> SyncPacket:
    """Parse an ADR-110 sync packet."""

    _require_magic(data, SYNC_MAGIC, "sync packet")
    _require_len(data, SYNC_PACKET_SIZE, "sync packet")
    (
        magic,
        node_id,
        protocol_version,
        flags,
        reserved,
        local_us,
        epoch_us,
        sequence_high_water,
        reserved2,
    ) = _SYNC.unpack_from(data)

    return SyncPacket(
        magic=magic,
        node_id=node_id,
        protocol_version=protocol_version,
        flags=flags,
        reserved=reserved,
        local_us=local_us,
        epoch_us=epoch_us,
        sequence_high_water=sequence_high_water,
        reserved2=reserved2,
        trailing_bytes=data[SYNC_PACKET_SIZE:],
    )


def parse_csi_packet(data: bytes) -> RawCsiPacket:
    """Compatibility alias for raw CSI packet parsing."""

    return parse_raw_csi_packet(data)


def parse_wasm_output_packet(data: bytes) -> WasmEventPacket:
    """Compatibility alias for WASM output packet parsing."""

    return parse_wasm_event_packet(data)


def packet_magic(data: bytes) -> int:
    """Return the little-endian u32 magic from a packet header."""

    _require_len(data, _MAGIC.size, "ESP32 packet magic")
    return _MAGIC.unpack_from(data)[0]


def decode_iq_pairs(iq_bytes: bytes) -> tuple[tuple[int, int], ...]:
    """Decode interleaved signed I/Q byte pairs into Python integers."""

    _require_even_iq(iq_bytes)
    return tuple((_u8_to_i8(iq_bytes[i]), _u8_to_i8(iq_bytes[i + 1])) for i in range(0, len(iq_bytes), 2))


def iq_amplitudes(iq_bytes: bytes) -> tuple[float, ...]:
    """Return ``sqrt(I^2 + Q^2)`` for signed I/Q byte pairs."""

    return tuple(math.hypot(i_val, q_val) for i_val, q_val in decode_iq_pairs(iq_bytes))


def iq_phases(iq_bytes: bytes) -> tuple[float, ...]:
    """Return ``atan2(Q, I)`` in radians for signed I/Q byte pairs."""

    return tuple(math.atan2(q_val, i_val) for i_val, q_val in decode_iq_pairs(iq_bytes))


def _require_magic(data: bytes, expected: int, context: str) -> None:
    magic = packet_magic(data)
    if magic != expected:
        raise UnknownPacketMagic(magic)


def _require_len(data: bytes, expected: int, context: str) -> None:
    if len(data) < expected:
        raise TruncatedPacket(expected=expected, actual=len(data), context=context)


def _require_even_iq(iq_bytes: bytes) -> None:
    if len(iq_bytes) % 2:
        raise InvalidPacket(f"I/Q payload must contain even byte count, got {len(iq_bytes)}")


def _u8_to_i8(value: int) -> int:
    return value - 256 if value >= 128 else value


__all__ = [
    "CSI_HEADER_SIZE",
    "EDGE_VITALS_MAGIC",
    "EDGE_VITALS_PACKET_SIZE",
    "Esp32Packet",
    "Esp32ProtocolError",
    "InvalidPacket",
    "RAW_CSI_MAGIC",
    "RawCsiPacket",
    "SYNC_MAGIC",
    "SYNC_PACKET_SIZE",
    "SyncPacket",
    "TruncatedPacket",
    "UnknownPacketMagic",
    "WASM_EVENT_MAGIC",
    "WASM_EVENT_SIZE",
    "WASM_MAX_EVENTS",
    "WASM_OUTPUT_HEADER_SIZE",
    "EdgeVitalsPacket",
    "WasmEvent",
    "WasmEventPacket",
    "decode_iq_pairs",
    "iq_amplitudes",
    "iq_phases",
    "packet_magic",
    "parse_csi_packet",
    "parse_edge_vitals_packet",
    "parse_packet",
    "parse_raw_csi_packet",
    "parse_sync_packet",
    "parse_wasm_event_packet",
    "parse_wasm_output_packet",
]
