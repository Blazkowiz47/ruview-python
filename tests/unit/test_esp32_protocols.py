from __future__ import annotations

import math
import struct

import pytest

from ruview.protocols import (
    CSI_HEADER_SIZE,
    EDGE_VITALS_MAGIC,
    RAW_CSI_MAGIC,
    SYNC_MAGIC,
    WASM_EVENT_MAGIC,
    EdgeVitalsPacket,
    Esp32ProtocolError,
    InvalidPacket,
    RawCsiPacket,
    SyncPacket,
    TruncatedPacket,
    UnknownPacketMagic,
    WasmEventPacket,
    decode_iq_pairs,
    iq_amplitudes,
    iq_phases,
    parse_edge_vitals_packet,
    parse_packet,
    parse_raw_csi_packet,
    parse_wasm_event_packet,
)


def raw_csi_packet_bytes() -> bytes:
    iq = bytes([3, 4, 0xFF, 0, 0, 0x80, 5, 12, 9, 40, 0, 1])
    header = struct.pack(
        "<IBBHIIbbBB",
        RAW_CSI_MAGIC,
        7,
        2,
        3,
        2412,
        42,
        -55,
        -95,
        1,
        0x11,
    )
    return header + iq + b"extra"


def test_parse_raw_csi_packet_uses_current_firmware_header_layout() -> None:
    packet = parse_packet(raw_csi_packet_bytes())

    assert isinstance(packet, RawCsiPacket)
    assert packet.magic == RAW_CSI_MAGIC
    assert packet.node_id == 7
    assert packet.n_antennas == 2
    assert packet.n_subcarriers == 3
    assert packet.frequency_mhz == 2412
    assert packet.sequence_number == 42
    assert packet.rssi_dbm == -55
    assert packet.noise_floor_dbm == -95
    assert packet.ppdu_type == 1
    assert packet.bandwidth_40mhz
    assert packet.sync_valid
    assert packet.expected_iq_bytes == 12
    assert packet.iq_bytes == raw_csi_packet_bytes()[CSI_HEADER_SIZE : CSI_HEADER_SIZE + 12]
    assert packet.trailing_bytes == b"extra"
    assert packet.iq_pairs()[:3] == ((3, 4), (-1, 0), (0, -128))


def test_parse_edge_vitals_packet_fixed_point_fields_and_flags() -> None:
    data = struct.pack(
        "<IBBHIbB2sffII",
        EDGE_VITALS_MAGIC,
        3,
        0x07,
        1842,
        725000,
        -42,
        2,
        b"\xAA\xBB",
        1.25,
        2.5,
        123456,
        0xDEADBEEF,
    )

    packet = parse_packet(data)

    assert isinstance(packet, EdgeVitalsPacket)
    assert packet.node_id == 3
    assert packet.flags == 0x07
    assert packet.presence
    assert packet.fall_detected
    assert packet.motion
    assert packet.breathing_rate_bpm == 18.42
    assert packet.heartrate_bpm == 72.5
    assert packet.rssi_dbm == -42
    assert packet.n_persons == 2
    assert packet.reserved == b"\xAA\xBB"
    assert packet.motion_energy == pytest.approx(1.25)
    assert packet.presence_score == pytest.approx(2.5)
    assert packet.timestamp_ms == 123456
    assert packet.reserved2 == 0xDEADBEEF


def test_parse_wasm_event_packet_preserves_event_payload() -> None:
    events = struct.pack("<Bf", 2, 1.5) + struct.pack("<Bf", 9, -0.25)
    data = struct.pack("<IBBH", WASM_EVENT_MAGIC, 4, 1, 2) + events + b"\x99"

    packet = parse_packet(data)

    assert isinstance(packet, WasmEventPacket)
    assert packet.node_id == 4
    assert packet.module_id == 1
    assert packet.event_count == 2
    assert packet.event_payload == events
    assert packet.trailing_bytes == b"\x99"
    assert packet.events[0].event_type == 2
    assert packet.events[0].value == pytest.approx(1.5)
    assert packet.events[0].raw_bytes == events[:5]
    assert packet.events[1].event_type == 9
    assert packet.events[1].value == pytest.approx(-0.25)


def test_parse_sync_packet_flags_and_epoch_fields() -> None:
    data = struct.pack(
        "<IBBBBQQII",
        SYNC_MAGIC,
        8,
        1,
        0x07,
        0,
        1_000_000,
        1_234_567,
        77,
        0,
    )

    packet = parse_packet(data)

    assert isinstance(packet, SyncPacket)
    assert packet.node_id == 8
    assert packet.protocol_version == 1
    assert packet.is_leader
    assert packet.sync_valid
    assert packet.smoothed_offset_valid
    assert packet.local_us == 1_000_000
    assert packet.epoch_us == 1_234_567
    assert packet.sequence_high_water == 77


def test_iq_helpers_decode_signed_bytes_amplitude_and_phase() -> None:
    iq = bytes([3, 4, 0xFF, 0, 0, 0x80])

    assert decode_iq_pairs(iq) == ((3, 4), (-1, 0), (0, -128))
    assert iq_amplitudes(iq) == pytest.approx((5.0, 1.0, 128.0))
    assert iq_phases(iq) == pytest.approx((math.atan2(4, 3), math.pi, -math.pi / 2))


def test_invalid_magic_and_truncated_packets_raise_protocol_errors() -> None:
    with pytest.raises(TruncatedPacket):
        parse_packet(b"\x01\x02\x03")

    with pytest.raises(UnknownPacketMagic):
        parse_packet(struct.pack("<I", 0x1234_5678) + b"\x00" * 8)

    with pytest.raises(TruncatedPacket):
        parse_raw_csi_packet(raw_csi_packet_bytes()[: CSI_HEADER_SIZE + 2])

    with pytest.raises(TruncatedPacket):
        parse_edge_vitals_packet(struct.pack("<I", EDGE_VITALS_MAGIC) + b"\x00" * 10)

    wasm_header = struct.pack("<IBBH", WASM_EVENT_MAGIC, 1, 2, 2)
    with pytest.raises(TruncatedPacket):
        parse_wasm_event_packet(wasm_header + struct.pack("<Bf", 1, 2.0))

    with pytest.raises(Esp32ProtocolError):
        parse_raw_csi_packet(struct.pack("<I", EDGE_VITALS_MAGIC) + b"\x00" * 24)


def test_iq_helpers_reject_odd_byte_payloads() -> None:
    with pytest.raises(InvalidPacket):
        decode_iq_pairs(b"\x01")


def test_raw_csi_rejects_zero_declared_dimensions() -> None:
    zero_antennas = struct.pack("<IBBHIIbbBB", RAW_CSI_MAGIC, 1, 0, 1, 2412, 1, -50, -90, 0, 0)
    with pytest.raises(InvalidPacket):
        parse_raw_csi_packet(zero_antennas)

    zero_subcarriers = struct.pack("<IBBHIIbbBB", RAW_CSI_MAGIC, 1, 1, 0, 2412, 1, -50, -90, 0, 0)
    with pytest.raises(InvalidPacket):
        parse_raw_csi_packet(zero_subcarriers)
