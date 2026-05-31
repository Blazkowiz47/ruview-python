"""BFLD privacy and wire-format research primitives.

The layout mirrors the Rust ``wifi-densepose-bfld`` reference: an 86-byte
little-endian header, a length-prefixed section payload, and BLAKE3-based
privacy attestations and daily RF signature hashes.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Iterable

import blake3

BFLD_MAGIC = 0xBF1D_0001
BFLD_VERSION = 1
BFLD_HEADER_SIZE = 86

HAS_CSI_DELTA = 1 << 0
PRIVACY_MODE = 1 << 1
SELF_ONLY = 1 << 3
KNOWN_FLAGS_MASK = HAS_CSI_DELTA | PRIVACY_MODE | SELF_ONLY
RESERVED_FLAGS_MASK = 0xFFFF ^ KNOWN_FLAGS_MASK

SECTION_PREFIX_LEN = 4
SECONDS_PER_DAY = 86_400
SITE_SALT_LEN = 32
RF_SIGNATURE_LEN = 32
EMBEDDING_DIM = 128
RISK_FACTOR_BYTES = 16

_HEADER = struct.Struct("<IHHQ16s16s16sHHhhHBBBBII")
assert _HEADER.size == BFLD_HEADER_SIZE


class BfldError(ValueError):
    """Base class for BFLD parse, privacy, and wire-format errors."""


class InvalidMagic(BfldError):
    """Raised when a BFLD header magic value does not match."""

    def __init__(self, actual: int) -> None:
        self.expected = BFLD_MAGIC
        self.actual = actual
        super().__init__(f"invalid BFLD magic: expected 0x{BFLD_MAGIC:08X}, got 0x{actual:08X}")


class UnsupportedVersion(BfldError):
    """Raised when a BFLD header version is unsupported."""

    def __init__(self, version: int) -> None:
        self.version = version
        super().__init__(f"unsupported BFLD version: {version}")


class CrcMismatch(BfldError):
    """Raised when the payload CRC32 does not match the header."""

    def __init__(self, *, expected: int, actual: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(f"payload CRC mismatch: expected 0x{expected:08X}, got 0x{actual:08X}")


class InvalidPrivacyClass(BfldError):
    """Raised when a privacy class byte is outside 0..=3."""

    def __init__(self, value: int) -> None:
        self.value = value
        super().__init__(f"invalid PrivacyClass byte: {value}")


class TruncatedFrame(BfldError):
    """Raised when a buffer is shorter than the declared frame layout."""

    def __init__(self, *, got: int, need: int) -> None:
        self.got = got
        self.need = need
        super().__init__(f"truncated frame: got {got} bytes, need at least {need}")


class MalformedSection(BfldError):
    """Raised when payload section length-prefix decoding fails."""

    def __init__(self, *, offset: int, reason: str) -> None:
        self.offset = offset
        self.reason = reason
        super().__init__(f"malformed payload section at offset {offset}: {reason}")


class InvalidDemote(BfldError):
    """Raised when a privacy gate transition would promote information."""

    def __init__(self, *, from_class: int, to_class: int) -> None:
        self.from_class = from_class
        self.to_class = to_class
        super().__init__(f"invalid demote: cannot move from class {from_class} to class {to_class}")


class PrivacyClass(IntEnum):
    """Byte-level privacy class carried by every BFLD frame."""

    Raw = 0
    Derived = 1
    Anonymous = 2
    Restricted = 3

    def allows_network(self) -> bool:
        """Return whether frames of this class may cross a network sink."""

        return self is not PrivacyClass.Raw

    def allows_matter(self) -> bool:
        """Return whether frames of this class may cross the Matter boundary."""

        return self in {PrivacyClass.Anonymous, PrivacyClass.Restricted}

    @property
    def network_eligible(self) -> bool:
        return self.allows_network()

    @property
    def matter_eligible(self) -> bool:
        return self.allows_matter()

    def as_u8(self) -> int:
        return int(self)

    @classmethod
    def from_u8(cls, value: int) -> "PrivacyClass":
        try:
            return cls(int(value))
        except ValueError as exc:
            raise InvalidPrivacyClass(int(value)) from exc


class PrivacyAction(IntEnum):
    """Concrete enforcement actions attached to a privacy mode."""

    Allow = 0
    SuppressIdentity = 1
    ReduceResolution = 2
    DropRaw = 3
    AggregateOnly = 4


PRIVACY_ACTIONS = (
    PrivacyAction.Allow,
    PrivacyAction.SuppressIdentity,
    PrivacyAction.ReduceResolution,
    PrivacyAction.DropRaw,
    PrivacyAction.AggregateOnly,
)
PrivacyAction.ALL = PRIVACY_ACTIONS  # type: ignore[attr-defined]


class PrivacyMode(IntEnum):
    """Operator-facing privacy posture layered over ``PrivacyClass``."""

    RawResearch = 0
    PrivateHome = 1
    EnterpriseAnonymous = 2
    CareWithConsent = 3
    StrictNoIdentity = 4

    def target_class(self) -> PrivacyClass:
        if self is PrivacyMode.RawResearch:
            return PrivacyClass.Raw
        if self in {PrivacyMode.PrivateHome, PrivacyMode.EnterpriseAnonymous}:
            return PrivacyClass.Anonymous
        if self is PrivacyMode.CareWithConsent:
            return PrivacyClass.Derived
        return PrivacyClass.Restricted

    def soul_signature_enabled(self) -> bool:
        return self in {PrivacyMode.RawResearch, PrivacyMode.CareWithConsent}

    def action_bits(self) -> int:
        suppress = 1 << PrivacyAction.SuppressIdentity
        reduce = 1 << PrivacyAction.ReduceResolution
        drop_raw = 1 << PrivacyAction.DropRaw
        aggregate = 1 << PrivacyAction.AggregateOnly

        if self in {PrivacyMode.RawResearch, PrivacyMode.CareWithConsent}:
            return 1 << PrivacyAction.Allow
        if self is PrivacyMode.PrivateHome:
            return suppress | drop_raw
        if self is PrivacyMode.EnterpriseAnonymous:
            return suppress | drop_raw | aggregate
        return suppress | reduce | drop_raw | aggregate

    def enforces(self, action: PrivacyAction) -> bool:
        action = PrivacyAction(action)
        return bool(self.action_bits() & (1 << action))

    def as_u8(self) -> int:
        return int(self)


@dataclass(frozen=True, slots=True)
class PrivacyAttestationProof:
    """BLAKE3 hash-chained proof that a privacy mode was active."""

    mode: PrivacyMode
    action_bits: int
    class_byte: int
    prev_hash: bytes
    hash: bytes

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", PrivacyMode(self.mode))
        object.__setattr__(self, "prev_hash", _fixed_bytes("prev_hash", self.prev_hash, 32))
        object.__setattr__(self, "hash", _fixed_bytes("hash", self.hash, 32))
        object.__setattr__(self, "action_bits", int(self.action_bits) & 0xFF)
        object.__setattr__(self, "class_byte", int(self.class_byte) & 0xFF)

    @classmethod
    def compute(cls, mode: PrivacyMode, prev_hash: bytes = bytes(32)) -> "PrivacyAttestationProof":
        mode = PrivacyMode(mode)
        prev = _fixed_bytes("prev_hash", prev_hash, 32)
        action_bits = mode.action_bits()
        class_byte = mode.target_class().as_u8()
        digest = blake3.blake3(prev + bytes([mode.as_u8(), action_bits, class_byte])).digest()
        return cls(mode=mode, action_bits=action_bits, class_byte=class_byte, prev_hash=prev, hash=digest)


class PrivacyModeRegistry:
    """Active privacy-mode source of truth with a verifiable audit chain."""

    def __init__(self, initial: PrivacyMode) -> None:
        self._active = PrivacyMode(initial)
        self._audit_log = [PrivacyAttestationProof.compute(self._active)]

    @classmethod
    def new(cls, initial: PrivacyMode) -> "PrivacyModeRegistry":
        return cls(initial)

    @property
    def active_mode(self) -> PrivacyMode:
        return self._active

    @property
    def active_class(self) -> PrivacyClass:
        return self._active.target_class()

    def is_action_enforced(self, action: PrivacyAction) -> bool:
        return self._active.enforces(action)

    def set_mode(self, mode: PrivacyMode) -> PrivacyAttestationProof:
        mode = PrivacyMode(mode)
        prev_hash = self._audit_log[-1].hash if self._audit_log else bytes(32)
        proof = PrivacyAttestationProof.compute(mode, prev_hash)
        self._active = mode
        self._audit_log.append(proof)
        return proof

    @property
    def latest_proof(self) -> PrivacyAttestationProof:
        return self._audit_log[-1]

    @property
    def audit_log(self) -> tuple[PrivacyAttestationProof, ...]:
        return tuple(self._audit_log)

    def verify_chain(self) -> bool:
        expected_prev = bytes(32)
        for proof in self._audit_log:
            if proof.prev_hash != expected_prev:
                return False
            recomputed = PrivacyAttestationProof.compute(proof.mode, proof.prev_hash)
            if (
                recomputed.action_bits != proof.action_bits
                or recomputed.class_byte != proof.class_byte
                or recomputed.hash != proof.hash
            ):
                return False
            expected_prev = proof.hash
        return True


@dataclass(slots=True)
class BfldFrameHeader:
    """On-the-wire BFLD frame header, serialized as 86 little-endian bytes."""

    magic: int = BFLD_MAGIC
    version: int = BFLD_VERSION
    flags: int = 0
    timestamp_ns: int = 0
    ap_hash: bytes = field(default_factory=lambda: bytes(16))
    sta_hash: bytes = field(default_factory=lambda: bytes(16))
    session_id: bytes = field(default_factory=lambda: bytes(16))
    channel: int = 0
    bandwidth_mhz: int = 0
    rssi_dbm: int = 0
    noise_floor_dbm: int = 0
    n_subcarriers: int = 0
    n_tx: int = 0
    n_rx: int = 0
    quantization: int = 0
    privacy_class: int = 0
    payload_len: int = 0
    payload_crc32: int = 0

    def __post_init__(self) -> None:
        self.ap_hash = _fixed_bytes("ap_hash", self.ap_hash, 16)
        self.sta_hash = _fixed_bytes("sta_hash", self.sta_hash, 16)
        self.session_id = _fixed_bytes("session_id", self.session_id, 16)

    @classmethod
    def empty(cls) -> "BfldFrameHeader":
        return cls()

    def to_le_bytes(self) -> bytes:
        """Serialize to the canonical 86-byte little-endian wire form."""

        return _HEADER.pack(
            int(self.magic),
            int(self.version),
            int(self.flags),
            int(self.timestamp_ns),
            self.ap_hash,
            self.sta_hash,
            self.session_id,
            int(self.channel),
            int(self.bandwidth_mhz),
            int(self.rssi_dbm),
            int(self.noise_floor_dbm),
            int(self.n_subcarriers),
            int(self.n_tx),
            int(self.n_rx),
            int(self.quantization),
            int(self.privacy_class),
            int(self.payload_len),
            int(self.payload_crc32),
        )

    @classmethod
    def from_le_bytes(cls, data: bytes | bytearray | memoryview) -> "BfldFrameHeader":
        """Parse a BFLD header from its little-endian wire form."""

        raw = bytes(data)
        if len(raw) < BFLD_HEADER_SIZE:
            raise TruncatedFrame(got=len(raw), need=BFLD_HEADER_SIZE)
        unpacked = _HEADER.unpack(raw[:BFLD_HEADER_SIZE])
        magic = unpacked[0]
        if magic != BFLD_MAGIC:
            raise InvalidMagic(magic)
        version = unpacked[1]
        if version != BFLD_VERSION:
            raise UnsupportedVersion(version)
        return cls(
            magic=magic,
            version=version,
            flags=unpacked[2],
            timestamp_ns=unpacked[3],
            ap_hash=unpacked[4],
            sta_hash=unpacked[5],
            session_id=unpacked[6],
            channel=unpacked[7],
            bandwidth_mhz=unpacked[8],
            rssi_dbm=unpacked[9],
            noise_floor_dbm=unpacked[10],
            n_subcarriers=unpacked[11],
            n_tx=unpacked[12],
            n_rx=unpacked[13],
            quantization=unpacked[14],
            privacy_class=unpacked[15],
            payload_len=unpacked[16],
            payload_crc32=unpacked[17],
        )


@dataclass(slots=True)
class BfldPayload:
    """Typed BFLD payload sections in canonical wire order."""

    compressed_angle_matrix: bytes = b""
    amplitude_proxy: bytes = b""
    phase_proxy: bytes = b""
    snr_vector: bytes = b""
    csi_delta: bytes | None = None
    vendor_extension: bytes = b""

    def __post_init__(self) -> None:
        self.compressed_angle_matrix = bytes(self.compressed_angle_matrix)
        self.amplitude_proxy = bytes(self.amplitude_proxy)
        self.phase_proxy = bytes(self.phase_proxy)
        self.snr_vector = bytes(self.snr_vector)
        self.csi_delta = None if self.csi_delta is None else bytes(self.csi_delta)
        self.vendor_extension = bytes(self.vendor_extension)

    @classmethod
    def default(cls) -> "BfldPayload":
        return cls()

    def to_bytes(self, include_csi_delta: bool | None = None) -> bytes:
        """Serialize as length-prefixed sections."""

        include_csi = self.csi_delta is not None if include_csi_delta is None else include_csi_delta
        out = bytearray()
        _push_section(out, self.compressed_angle_matrix)
        _push_section(out, self.amplitude_proxy)
        _push_section(out, self.phase_proxy)
        _push_section(out, self.snr_vector)
        if include_csi:
            _push_section(out, self.csi_delta or b"")
        _push_section(out, self.vendor_extension)
        return bytes(out)

    def wire_len(self, include_csi_delta: bool | None = None) -> int:
        include_csi = self.csi_delta is not None if include_csi_delta is None else include_csi_delta
        total = (
            SECTION_PREFIX_LEN * 5
            + len(self.compressed_angle_matrix)
            + len(self.amplitude_proxy)
            + len(self.phase_proxy)
            + len(self.snr_vector)
            + len(self.vendor_extension)
        )
        if include_csi:
            total += SECTION_PREFIX_LEN + len(self.csi_delta or b"")
        return total

    @classmethod
    def from_bytes(cls, data: bytes | bytearray | memoryview, expect_csi_delta: bool) -> "BfldPayload":
        raw = bytes(data)
        cursor = 0
        compressed_angle_matrix, cursor = _read_section(raw, cursor)
        amplitude_proxy, cursor = _read_section(raw, cursor)
        phase_proxy, cursor = _read_section(raw, cursor)
        snr_vector, cursor = _read_section(raw, cursor)
        csi_delta = None
        if expect_csi_delta:
            csi_delta, cursor = _read_section(raw, cursor)
        vendor_extension, cursor = _read_section(raw, cursor)
        if cursor != len(raw):
            raise MalformedSection(offset=cursor, reason="trailing bytes after vendor_extension")
        return cls(
            compressed_angle_matrix=compressed_angle_matrix,
            amplitude_proxy=amplitude_proxy,
            phase_proxy=phase_proxy,
            snr_vector=snr_vector,
            csi_delta=csi_delta,
            vendor_extension=vendor_extension,
        )


@dataclass(slots=True)
class BfldFrame:
    """Complete BFLD frame: header plus CRC-protected payload bytes."""

    header: BfldFrameHeader
    payload: bytes = b""

    def __post_init__(self) -> None:
        self.header = replace(self.header)
        self.payload = bytes(self.payload)
        self.resync_payload_metadata()

    @classmethod
    def new(cls, header: BfldFrameHeader, payload: bytes | bytearray | memoryview) -> "BfldFrame":
        return cls(header=header, payload=bytes(payload))

    @classmethod
    def from_payload(cls, header: BfldFrameHeader, payload: BfldPayload) -> "BfldFrame":
        header = replace(header)
        if payload.csi_delta is None:
            header.flags &= ~HAS_CSI_DELTA
        else:
            header.flags |= HAS_CSI_DELTA
        return cls(header=header, payload=payload.to_bytes(payload.csi_delta is not None))

    def parse_payload(self) -> BfldPayload:
        expect_csi_delta = bool(self.header.flags & HAS_CSI_DELTA)
        return BfldPayload.from_bytes(self.payload, expect_csi_delta)

    def resync_payload_metadata(self) -> None:
        if len(self.payload) > 0xFFFF_FFFF:
            raise BfldError("BFLD payload exceeds u32 payload_len")
        self.header.payload_len = len(self.payload)
        self.header.payload_crc32 = crc32_of_payload(self.payload)

    def to_bytes(self) -> bytes:
        self.resync_payload_metadata()
        return self.header.to_le_bytes() + self.payload

    @classmethod
    def from_bytes(cls, data: bytes | bytearray | memoryview) -> "BfldFrame":
        raw = bytes(data)
        if len(raw) < BFLD_HEADER_SIZE:
            raise TruncatedFrame(got=len(raw), need=BFLD_HEADER_SIZE)
        header = BfldFrameHeader.from_le_bytes(raw[:BFLD_HEADER_SIZE])
        expected_total = BFLD_HEADER_SIZE + int(header.payload_len)
        if len(raw) < expected_total:
            raise TruncatedFrame(got=len(raw), need=expected_total)
        payload = raw[BFLD_HEADER_SIZE:expected_total]
        actual_crc = crc32_of_payload(payload)
        if actual_crc != header.payload_crc32:
            raise CrcMismatch(expected=header.payload_crc32, actual=actual_crc)
        return cls(header=header, payload=payload)


def crc32_of_payload(payload: bytes | bytearray | memoryview) -> int:
    """Compute CRC-32/ISO-HDLC over payload bytes only."""

    return zlib.crc32(bytes(payload)) & 0xFFFF_FFFF


@dataclass(frozen=True, slots=True)
class SignatureHasher:
    """BLAKE3 keyed hasher for per-site, per-day RF signature hashes."""

    site_salt: bytes

    def __post_init__(self) -> None:
        object.__setattr__(self, "site_salt", _fixed_bytes("site_salt", self.site_salt, SITE_SALT_LEN))

    @classmethod
    def new(cls, site_salt: bytes | bytearray | memoryview) -> "SignatureHasher":
        return cls(bytes(site_salt))

    @staticmethod
    def day_epoch_from_unix_secs(unix_secs: int) -> int:
        return int(unix_secs) // SECONDS_PER_DAY

    def compute(self, day_epoch: int, features: bytes | bytearray | memoryview) -> bytes:
        hasher = blake3.blake3(key=self.site_salt)
        hasher.update(int(day_epoch).to_bytes(4, "little", signed=False))
        hasher.update(bytes(features))
        return hasher.digest(length=RF_SIGNATURE_LEN)

    def compute_at(self, unix_secs: int, features: bytes | bytearray | memoryview) -> bytes:
        return self.compute(self.day_epoch_from_unix_secs(unix_secs), features)


@dataclass(frozen=True, slots=True)
class IdentityFeatures:
    """Canonical little-endian f32 feature encoder for ``SignatureHasher``."""

    kind: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        values = tuple(float(v) for v in self.values)
        object.__setattr__(self, "values", values)
        if self.kind == "embedding" and len(values) != EMBEDDING_DIM:
            raise ValueError(f"embedding must contain {EMBEDDING_DIM} floats, got {len(values)}")
        if self.kind == "risk_factors" and len(values) != 4:
            raise ValueError(f"risk_factors must contain 4 floats, got {len(values)}")
        if self.kind not in {"embedding", "risk_factors"}:
            raise ValueError(f"unknown identity feature kind: {self.kind}")

    @classmethod
    def from_embedding(cls, embedding: Iterable[float] | object) -> "IdentityFeatures":
        source = embedding.as_slice() if hasattr(embedding, "as_slice") else embedding
        values = tuple(float(v) for v in source)  # type: ignore[union-attr]
        if len(values) != EMBEDDING_DIM:
            raise ValueError(f"embedding must contain {EMBEDDING_DIM} floats, got {len(values)}")
        return cls(kind="embedding", values=values)

    @classmethod
    def from_risk_factors(cls, sep: float, stab: float, consist: float, conf: float) -> "IdentityFeatures":
        return cls(kind="risk_factors", values=(float(sep), float(stab), float(consist), float(conf)))

    def canonical_byte_len(self) -> int:
        if self.kind == "embedding":
            return EMBEDDING_DIM * 4
        if self.kind == "risk_factors":
            return RISK_FACTOR_BYTES
        raise ValueError(f"unknown identity feature kind: {self.kind}")

    def write_canonical_bytes(self, out: bytearray) -> None:
        out.extend(self.canonical_bytes())

    def canonical_bytes(self) -> bytes:
        self.canonical_byte_len()
        return struct.pack("<" + "f" * len(self.values), *self.values)

    def compute_hash(self, hasher: SignatureHasher, day_epoch: int) -> bytes:
        return hasher.compute(day_epoch, self.canonical_bytes())


def _fixed_bytes(name: str, value: bytes | bytearray | memoryview, length: int) -> bytes:
    raw = bytes(value)
    if len(raw) != length:
        raise ValueError(f"{name} must be exactly {length} bytes, got {len(raw)}")
    return raw


def _push_section(out: bytearray, section: bytes) -> None:
    if len(section) > 0xFFFF_FFFF:
        raise BfldError("BFLD payload section exceeds u32 length")
    out.extend(len(section).to_bytes(SECTION_PREFIX_LEN, "little"))
    out.extend(section)


def _read_section(data: bytes, cursor: int) -> tuple[bytes, int]:
    start = cursor
    if start + SECTION_PREFIX_LEN > len(data):
        raise MalformedSection(offset=start, reason="section length prefix runs past buffer end")
    length = int.from_bytes(data[start : start + SECTION_PREFIX_LEN], "little")
    data_start = start + SECTION_PREFIX_LEN
    data_end = data_start + length
    if data_end > len(data):
        raise MalformedSection(offset=start, reason="section body runs past buffer end")
    return data[data_start:data_end], data_end
