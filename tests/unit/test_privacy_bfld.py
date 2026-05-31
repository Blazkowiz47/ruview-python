from __future__ import annotations

import math
import struct
from dataclasses import replace

import pytest

from ruview.privacy.bfld import (
    BFLD_HEADER_SIZE,
    BFLD_MAGIC,
    BFLD_VERSION,
    EMBEDDING_DIM,
    HAS_CSI_DELTA,
    PRIVACY_MODE,
    RF_SIGNATURE_LEN,
    RISK_FACTOR_BYTES,
    SITE_SALT_LEN,
    BfldFrame,
    BfldFrameHeader,
    BfldPayload,
    CrcMismatch,
    IdentityFeatures,
    InvalidDemote,
    PrivacyAction,
    PrivacyAttestationProof,
    PrivacyClass,
    PrivacyMode,
    PrivacyModeRegistry,
    SignatureHasher,
    crc32_of_payload,
)
from ruview.privacy.identity_risk import (
    PREDICT_ONLY_THRESHOLD,
    RECALIBRATE_THRESHOLD,
    REJECT_THRESHOLD,
    GateAction,
    identity_risk_score,
)
from ruview.privacy.privacy_gate import PrivacyGate


def _salt(seed: int) -> bytes:
    return bytes((seed + i) & 0xFF for i in range(SITE_SALT_LEN))


def _payload(with_csi: bool = True) -> BfldPayload:
    return BfldPayload(
        compressed_angle_matrix=bytes([0x11]) * 32,
        amplitude_proxy=bytes([0x22]) * 16,
        phase_proxy=bytes([0x33]) * 16,
        snr_vector=bytes([0x44]) * 8,
        csi_delta=bytes([0x55]) * 24 if with_csi else None,
        vendor_extension=b"\xAA",
    )


def _header(class_: PrivacyClass = PrivacyClass.Derived) -> BfldFrameHeader:
    return BfldFrameHeader(
        flags=PRIVACY_MODE,
        timestamp_ns=123456789,
        ap_hash=bytes(range(16)),
        sta_hash=bytes(range(16, 32)),
        session_id=bytes(range(32, 48)),
        channel=36,
        bandwidth_mhz=80,
        rssi_dbm=-42,
        noise_floor_dbm=-93,
        n_subcarriers=234,
        n_tx=2,
        n_rx=3,
        quantization=1,
        privacy_class=class_.as_u8(),
    )


def _frame(class_: PrivacyClass = PrivacyClass.Derived, with_csi: bool = True) -> BfldFrame:
    return BfldFrame.from_payload(_header(class_), _payload(with_csi))


def test_privacy_class_network_and_matter_eligibility() -> None:
    assert not PrivacyClass.Raw.allows_network()
    assert PrivacyClass.Derived.allows_network()
    assert not PrivacyClass.Derived.allows_matter()
    assert PrivacyClass.Anonymous.allows_matter()
    assert PrivacyClass.Restricted.matter_eligible


def test_header_roundtrip_is_little_endian_and_86_bytes() -> None:
    header = _header()
    header.payload_len = 99
    header.payload_crc32 = 0xAABBCCDD

    data = header.to_le_bytes()
    parsed = BfldFrameHeader.from_le_bytes(data)

    assert len(data) == BFLD_HEADER_SIZE == 86
    assert data[:8] == struct.pack("<IHH", BFLD_MAGIC, BFLD_VERSION, PRIVACY_MODE)
    assert parsed == header
    assert parsed.rssi_dbm == -42
    assert parsed.noise_floor_dbm == -93


def test_payload_and_frame_crc_roundtrip() -> None:
    payload = _payload(with_csi=True)
    frame = BfldFrame.from_payload(_header(), payload)
    wire = frame.to_bytes()

    parsed = BfldFrame.from_bytes(wire + b"trailer")

    assert parsed.header.payload_len == len(parsed.payload)
    assert parsed.header.payload_crc32 == crc32_of_payload(parsed.payload)
    assert parsed.header.flags & HAS_CSI_DELTA
    assert parsed.parse_payload() == payload

    tampered = bytearray(wire)
    tampered[BFLD_HEADER_SIZE + 10] ^= 0xFF
    with pytest.raises(CrcMismatch):
        BfldFrame.from_bytes(tampered)


def test_payload_section_encoding_detects_flag_skew() -> None:
    payload = _payload(with_csi=True)
    data = payload.to_bytes(include_csi_delta=True)

    parsed = BfldPayload.from_bytes(data, expect_csi_delta=True)
    assert parsed == payload
    assert payload.wire_len(True) == len(data)

    with pytest.raises(ValueError, match="trailing"):
        BfldPayload.from_bytes(data, expect_csi_delta=False)


def test_identity_risk_score_thresholds_and_gate_actions() -> None:
    assert identity_risk_score(0.8, 0.9, 0.85, 0.95) == pytest.approx(0.8 * 0.9 * 0.85 * 0.95)
    assert identity_risk_score(-1.0, 1.0, 1.0, 1.0) == 0.0
    assert identity_risk_score(math.nan, 1.0, 1.0, 1.0) == 0.0
    assert PREDICT_ONLY_THRESHOLD == pytest.approx(0.5)
    assert REJECT_THRESHOLD == pytest.approx(0.7)
    assert RECALIBRATE_THRESHOLD == pytest.approx(0.9)

    assert GateAction.from_score(0.49) is GateAction.Accept
    assert GateAction.from_score(0.5) is GateAction.PredictOnly
    assert GateAction.from_score(0.7) is GateAction.Reject
    assert GateAction.from_score(0.9) is GateAction.Recalibrate
    assert GateAction.from_score(math.nan) is GateAction.Accept
    assert GateAction.PredictOnly.allows_publish()
    assert GateAction.Reject.drops_event()
    assert GateAction.Recalibrate.requires_recalibrate()


def test_signature_hash_daily_rotation_and_site_separation() -> None:
    features = bytes(range(64))
    site_a = SignatureHasher(_salt(1))
    site_b = SignatureHasher(_salt(2))

    same_a = site_a.compute(42, features)
    same_b = site_a.compute(42, features)
    other_day = site_a.compute(43, features)
    other_site = site_b.compute(42, features)

    assert same_a == same_b
    assert len(same_a) == RF_SIGNATURE_LEN == 32
    assert same_a != other_day
    assert same_a != other_site
    assert SignatureHasher.day_epoch_from_unix_secs(86_399) == 0
    assert SignatureHasher.day_epoch_from_unix_secs(86_400) == 1
    assert site_a.compute_at(86_400 * 42, features) == same_a


def test_identity_features_canonical_bytes_and_hash() -> None:
    risk = IdentityFeatures.from_risk_factors(0.1, 0.2, 0.3, 0.4)
    expected = struct.pack("<ffff", 0.1, 0.2, 0.3, 0.4)
    assert risk.canonical_byte_len() == RISK_FACTOR_BYTES == 16
    assert risk.canonical_bytes() == expected

    embedding_values = [0.5 + i * 0.001 for i in range(EMBEDDING_DIM)]
    embedding = IdentityFeatures.from_embedding(embedding_values)
    assert embedding.canonical_byte_len() == EMBEDDING_DIM * 4

    hasher = SignatureHasher(bytes([42]) * SITE_SALT_LEN)
    assert risk.compute_hash(hasher, 7) == hasher.compute(7, expected)
    assert embedding.compute_hash(hasher, 7) != risk.compute_hash(hasher, 7)


def test_privacy_mode_registry_hash_chain_verifies_and_detects_tamper() -> None:
    registry = PrivacyModeRegistry(PrivacyMode.RawResearch)

    genesis = registry.latest_proof
    p1 = registry.set_mode(PrivacyMode.PrivateHome)
    p2 = registry.set_mode(PrivacyMode.StrictNoIdentity)

    assert genesis.prev_hash == bytes(32)
    assert p1.prev_hash == genesis.hash
    assert p2.prev_hash == p1.hash
    assert registry.active_class is PrivacyClass.Restricted
    assert registry.is_action_enforced(PrivacyAction.AggregateOnly)
    assert registry.verify_chain()

    registry._audit_log[1] = replace(p1, mode=PrivacyMode.CareWithConsent)  # noqa: SLF001
    assert not registry.verify_chain()
    recomputed = PrivacyAttestationProof.compute(PrivacyMode.PrivateHome, genesis.hash)
    assert recomputed.hash == p1.hash


def test_privacy_gate_demotion_is_monotonic_and_strips_sections() -> None:
    derived = _frame(PrivacyClass.Derived, with_csi=True)

    anonymous = PrivacyGate.demote(derived, PrivacyClass.Anonymous)
    anon_payload = anonymous.parse_payload()
    assert anonymous.header.privacy_class == PrivacyClass.Anonymous.as_u8()
    assert not (anonymous.header.flags & HAS_CSI_DELTA)
    assert anon_payload.compressed_angle_matrix == b""
    assert anon_payload.csi_delta is None
    assert anon_payload.amplitude_proxy == bytes([0x22]) * 16
    assert BfldFrame.from_bytes(anonymous.to_bytes()).parse_payload() == anon_payload

    restricted = PrivacyGate.demote(anonymous, PrivacyClass.Restricted)
    restricted_payload = restricted.parse_payload()
    assert restricted.header.privacy_class == PrivacyClass.Restricted.as_u8()
    assert restricted_payload.compressed_angle_matrix == b""
    assert restricted_payload.csi_delta is None
    assert restricted_payload.amplitude_proxy == b""
    assert restricted_payload.phase_proxy == b""
    assert restricted_payload.snr_vector == bytes([0x44]) * 8
    assert restricted_payload.vendor_extension == b"\xAA"
    assert BfldFrame.from_bytes(restricted.to_bytes()).header.payload_crc32 == restricted.header.payload_crc32

    with pytest.raises(InvalidDemote):
        PrivacyGate.demote(restricted, PrivacyClass.Derived)
