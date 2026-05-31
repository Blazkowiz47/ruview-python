"""Compressed RuVector-style temporal histories for vital-sign features."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


class RuVectorHistoryError(ValueError):
    """Raised when a compressed history receives invalid input."""


class _CallableInt(int):
    """Small compatibility helper: behaves like an int and can be called."""

    def __new__(cls, value: int) -> "_CallableInt":
        return int.__new__(cls, int(value))

    def __call__(self) -> int:
        return int(self)


@dataclass(frozen=True)
class TierPolicy:
    """Age-based quantization tiers for a compressed temporal history."""

    hot_frames: int = 10
    warm_frames: int = 300
    hot_bits: int = 8
    warm_bits: int = 6
    cold_bits: int = 3

    def __post_init__(self) -> None:
        for name in ("hot_frames", "warm_frames"):
            if int(getattr(self, name)) < 0:
                raise RuVectorHistoryError(f"{name} must be non-negative")
        for name in ("hot_bits", "warm_bits", "cold_bits"):
            value = int(getattr(self, name))
            if value < 1 or value > 8:
                raise RuVectorHistoryError(f"{name} must be in the range 1..8")

    @property
    def max_bits(self) -> int:
        return max(int(self.hot_bits), int(self.warm_bits), int(self.cold_bits))

    def tier_for_age(self, age: int) -> str:
        age = max(int(age), 0)
        if age < self.hot_frames:
            return "hot"
        if age < self.hot_frames + self.warm_frames:
            return "warm"
        return "cold"

    def bits_for_age(self, age: int) -> int:
        tier = self.tier_for_age(age)
        if tier == "hot":
            return int(self.hot_bits)
        if tier == "warm":
            return int(self.warm_bits)
        return int(self.cold_bits)


@dataclass(frozen=True)
class FrameQuantizationMetadata:
    """Per-vector metadata needed to reconstruct one compressed history item."""

    sequence_number: int
    tier: str
    bits: int
    minimum: float
    scale: float
    payload_bytes: int


@dataclass(frozen=True)
class HistoryDiagnostics:
    """Byte-size and compression diagnostics for a compressed history."""

    frame_count: int
    stored_frame_count: int
    capacity: int
    vector_length: int
    raw_bytes: int
    payload_bytes: int
    metadata_bytes: int
    encoded_bytes: int
    compression_ratio: float
    tier_counts: Mapping[str, int]

    @property
    def raw_size_bytes(self) -> int:
        return self.raw_bytes

    @property
    def compressed_size_bytes(self) -> int:
        return self.encoded_bytes


class _CompressedVectorRing:
    """Ring buffer of independently quantized one-dimensional vectors."""

    def __init__(
        self,
        *,
        vector_length: int,
        capacity: int,
        policy: TierPolicy | None = None,
        vector_name: str = "vector",
    ) -> None:
        if int(vector_length) <= 0:
            raise RuVectorHistoryError("vector_length must be positive")
        if int(capacity) <= 0:
            raise RuVectorHistoryError("capacity must be positive")

        self.vector_length = int(vector_length)
        self.capacity = int(capacity)
        self.policy = policy if policy is not None else TierPolicy()
        self._vector_name = vector_name
        self._max_payload_bytes = _packed_byte_count(self.vector_length, self.policy.max_bits)
        self._payload = np.zeros((self.capacity, self._max_payload_bytes), dtype=np.uint8)
        self._payload_lengths = np.zeros(self.capacity, dtype=np.uint32)
        self._minimums = np.zeros(self.capacity, dtype=np.float32)
        self._scales = np.zeros(self.capacity, dtype=np.float32)
        self._bits = np.zeros(self.capacity, dtype=np.uint8)
        self._sequence_numbers = np.full(self.capacity, -1, dtype=np.int64)
        self._filled = np.zeros(self.capacity, dtype=np.bool_)
        self._frame_count = 0

    @property
    def frame_count(self) -> _CallableInt:
        """Total number of vectors pushed, including overwritten ones."""

        return _CallableInt(self._frame_count)

    @property
    def stored_frame_count(self) -> int:
        """Number of vectors currently retained in the ring."""

        return int(np.count_nonzero(self._filled))

    @property
    def is_full(self) -> bool:
        return self.stored_frame_count == self.capacity

    @property
    def sequence_numbers(self) -> tuple[int, ...]:
        return tuple(int(self._sequence_numbers[slot]) for slot in self._ordered_slots())

    @property
    def encoded_byte_size(self) -> int:
        return self.diagnostics().encoded_bytes

    @property
    def compression_ratio(self) -> float:
        return self.diagnostics().compression_ratio

    def push(self, values: ArrayLike) -> None:
        """Append one vector, overwriting the oldest vector when full."""

        vector = self._coerce_vector(values)
        sequence_number = self._frame_count
        slot = sequence_number % self.capacity
        self._write_slot(slot, vector, sequence_number, self.policy.bits_for_age(0))
        self._frame_count += 1
        self._retier()

    def reconstruct(self) -> FloatArray:
        """Return retained vectors in chronological order as ``float64``."""

        slots = self._ordered_slots()
        out = np.empty((slots.size, self.vector_length), dtype=np.float64)
        for index, slot in enumerate(slots):
            out[index] = self._decode_slot(int(slot))
        return out

    def to_array(self) -> FloatArray:
        """Alias for :meth:`reconstruct`."""

        return self.reconstruct()

    def decoded(self) -> FloatArray:
        """Alias for :meth:`reconstruct`."""

        return self.reconstruct()

    def to_vec(self) -> list[float]:
        """Return retained vectors flattened in chronological order."""

        return self.reconstruct().reshape(-1).tolist()

    def quantization_metadata(self) -> tuple[FrameQuantizationMetadata, ...]:
        """Return reconstruction metadata for retained vectors in order."""

        latest = self._frame_count - 1
        metadata: list[FrameQuantizationMetadata] = []
        for slot in self._ordered_slots():
            sequence_number = int(self._sequence_numbers[slot])
            age = latest - sequence_number
            metadata.append(
                FrameQuantizationMetadata(
                    sequence_number=sequence_number,
                    tier=self.policy.tier_for_age(age),
                    bits=int(self._bits[slot]),
                    minimum=float(self._minimums[slot]),
                    scale=float(self._scales[slot]),
                    payload_bytes=int(self._payload_lengths[slot]),
                )
            )
        return tuple(metadata)

    def diagnostics(self) -> HistoryDiagnostics:
        """Return byte-size and tier summaries for retained vectors."""

        stored = self.stored_frame_count
        raw_bytes = stored * self.vector_length * np.dtype(np.float32).itemsize
        payload_bytes = int(np.sum(self._payload_lengths[self._filled], dtype=np.uint64))
        metadata_per_frame = (
            self._minimums.dtype.itemsize
            + self._scales.dtype.itemsize
            + self._bits.dtype.itemsize
            + self._payload_lengths.dtype.itemsize
            + self._sequence_numbers.dtype.itemsize
        )
        metadata_bytes = stored * metadata_per_frame
        encoded_bytes = payload_bytes + metadata_bytes
        if raw_bytes == 0:
            compression_ratio = 1.0
        else:
            compression_ratio = raw_bytes / encoded_bytes if encoded_bytes > 0 else math.inf

        tier_counts = {"hot": 0, "warm": 0, "cold": 0}
        for item in self.quantization_metadata():
            tier_counts[item.tier] = tier_counts.get(item.tier, 0) + 1

        return HistoryDiagnostics(
            frame_count=int(self._frame_count),
            stored_frame_count=stored,
            capacity=int(self.capacity),
            vector_length=int(self.vector_length),
            raw_bytes=int(raw_bytes),
            payload_bytes=int(payload_bytes),
            metadata_bytes=int(metadata_bytes),
            encoded_bytes=int(encoded_bytes),
            compression_ratio=float(compression_ratio),
            tier_counts=tier_counts,
        )

    def byte_size(self) -> int:
        """Return encoded payload plus reconstruction metadata bytes."""

        return self.diagnostics().encoded_bytes

    def raw_byte_size(self) -> int:
        """Return the equivalent retained ``float32`` byte size."""

        return self.diagnostics().raw_bytes

    def clear(self) -> None:
        """Drop all retained vectors and reset the total frame counter."""

        self._payload.fill(0)
        self._payload_lengths.fill(0)
        self._minimums.fill(0.0)
        self._scales.fill(0.0)
        self._bits.fill(0)
        self._sequence_numbers.fill(-1)
        self._filled.fill(False)
        self._frame_count = 0

    def _coerce_vector(self, values: ArrayLike) -> FloatArray:
        vector = np.asarray(values, dtype=np.float64)
        expected_shape = (self.vector_length,)
        if vector.shape != expected_shape:
            raise RuVectorHistoryError(
                f"{self._vector_name} must have shape {expected_shape}, got {vector.shape}"
            )
        if not bool(np.all(np.isfinite(vector))):
            raise RuVectorHistoryError(f"{self._vector_name} must contain only finite values")
        return vector

    def _ordered_slots(self) -> NDArray[np.int64]:
        slots = np.flatnonzero(self._filled)
        if slots.size == 0:
            return slots.astype(np.int64)
        order = np.argsort(self._sequence_numbers[slots], kind="stable")
        return slots[order].astype(np.int64)

    def _write_slot(self, slot: int, values: FloatArray, sequence_number: int, bits: int) -> None:
        codes, minimum, scale = _quantize(values, bits)
        payload = _pack_quantized(codes, bits)
        payload_len = int(payload.size)
        self._payload[slot].fill(0)
        self._payload[slot, :payload_len] = payload
        self._payload_lengths[slot] = payload_len
        self._minimums[slot] = np.float32(minimum)
        self._scales[slot] = np.float32(scale)
        self._bits[slot] = np.uint8(bits)
        self._sequence_numbers[slot] = int(sequence_number)
        self._filled[slot] = True

    def _decode_slot(self, slot: int) -> FloatArray:
        if not bool(self._filled[slot]):
            raise RuVectorHistoryError("cannot decode an empty ring slot")
        bits = int(self._bits[slot])
        length = int(self._payload_lengths[slot])
        payload = self._payload[slot, :length]
        codes = _unpack_quantized(payload, self.vector_length, bits)
        return _dequantize(codes, bits, float(self._minimums[slot]), float(self._scales[slot]))

    def _retier(self) -> None:
        latest = self._frame_count - 1
        if latest < 0:
            return
        for slot in np.flatnonzero(self._filled):
            sequence_number = int(self._sequence_numbers[slot])
            target_bits = self.policy.bits_for_age(latest - sequence_number)
            if int(self._bits[slot]) != target_bits:
                values = self._decode_slot(int(slot))
                self._write_slot(int(slot), values, sequence_number, target_bits)


class CompressedBreathingHistory(_CompressedVectorRing):
    """Compressed ring buffer for per-frame subcarrier amplitudes."""

    def __init__(
        self,
        n_subcarriers: int = 56,
        zone_id: int = 0,
        *,
        capacity_frames: int = 6_000,
        policy: TierPolicy | None = None,
    ) -> None:
        self.n_subcarriers = int(n_subcarriers)
        self.zone_id = int(zone_id)
        super().__init__(
            vector_length=self.n_subcarriers,
            capacity=capacity_frames,
            policy=policy,
            vector_name="amplitudes",
        )

    @property
    def capacity_frames(self) -> int:
        return self.capacity

    def push_frame(self, amplitudes: ArrayLike) -> None:
        """Append one subcarrier-amplitude frame."""

        self.push(amplitudes)


class CompressedHeartbeatSpectrogram(_CompressedVectorRing):
    """Compressed rolling heartbeat spectrogram column history."""

    def __init__(
        self,
        n_freq_bins: int = 128,
        *,
        capacity_columns: int = 1_500,
        recent_window: int = 100,
        policy: TierPolicy | None = None,
    ) -> None:
        if int(recent_window) <= 0:
            raise RuVectorHistoryError("recent_window must be positive")
        self.n_freq_bins = int(n_freq_bins)
        self.recent_window = int(recent_window)
        super().__init__(
            vector_length=self.n_freq_bins,
            capacity=capacity_columns,
            policy=policy,
            vector_name="spectrogram column",
        )

    @property
    def capacity_columns(self) -> int:
        return self.capacity

    @property
    def stored_column_count(self) -> int:
        return self.stored_frame_count

    def push_column(self, column: ArrayLike) -> None:
        """Append one spectrogram time column."""

        self.push(column)

    def band_power(
        self,
        low_bin: int,
        high_bin: int,
        *,
        window_columns: int | None = None,
    ) -> float:
        """Return mean squared power for a frequency-bin band over recent columns."""

        low = int(low_bin)
        high = int(high_bin)
        window = self.recent_window if window_columns is None else int(window_columns)
        if low < 0 or high < 0:
            raise RuVectorHistoryError("band indices must be non-negative")
        if window <= 0:
            raise RuVectorHistoryError("window_columns must be positive")
        if low > high or low >= self.n_freq_bins or self.stored_frame_count == 0:
            return 0.0
        high = min(high, self.n_freq_bins - 1)

        columns = self.reconstruct()
        recent = columns[-min(window, columns.shape[0]) :, low : high + 1]
        if recent.size == 0:
            return 0.0
        return float(np.mean(np.square(recent, dtype=np.float64)))


CompressedBreathingBuffer = CompressedBreathingHistory
CompressedHeartbeatHistory = CompressedHeartbeatSpectrogram


def _quantize(values: FloatArray, bits: int) -> tuple[NDArray[np.uint8], float, float]:
    _validate_bits(bits)
    levels = (1 << int(bits)) - 1
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    span = maximum - minimum
    if span <= 0.0:
        return np.zeros(values.shape, dtype=np.uint8), minimum, 0.0
    scale = span / levels
    codes = np.rint((values - minimum) / scale)
    codes = np.clip(codes, 0, levels).astype(np.uint8)
    return codes, minimum, float(scale)


def _dequantize(codes: NDArray[np.uint8], bits: int, minimum: float, scale: float) -> FloatArray:
    _validate_bits(bits)
    if scale <= 0.0:
        return np.full(codes.shape, float(minimum), dtype=np.float64)
    levels = (1 << int(bits)) - 1
    clipped = np.clip(codes.astype(np.uint16), 0, levels).astype(np.float64)
    return float(minimum) + clipped * float(scale)


def _pack_quantized(codes: NDArray[np.uint8], bits: int) -> NDArray[np.uint8]:
    _validate_bits(bits)
    values = np.asarray(codes, dtype=np.uint8).reshape(-1)
    if int(bits) == 8:
        return values.copy()

    out = np.zeros(_packed_byte_count(values.size, bits), dtype=np.uint8)
    bit_position = 0
    for value in values:
        raw = int(value)
        for offset in range(int(bits)):
            if raw & (1 << offset):
                byte_index = bit_position >> 3
                bit_index = bit_position & 7
                out[byte_index] = np.uint8(int(out[byte_index]) | (1 << bit_index))
            bit_position += 1
    return out


def _unpack_quantized(payload: NDArray[np.uint8], count: int, bits: int) -> NDArray[np.uint8]:
    _validate_bits(bits)
    expected_bytes = _packed_byte_count(count, bits)
    if payload.size < expected_bytes:
        raise RuVectorHistoryError(
            f"packed payload requires {expected_bytes} bytes, got {payload.size}"
        )
    if int(bits) == 8:
        return np.asarray(payload[:count], dtype=np.uint8).copy()

    out = np.zeros(int(count), dtype=np.uint8)
    bit_position = 0
    for index in range(int(count)):
        raw = 0
        for offset in range(int(bits)):
            byte_index = bit_position >> 3
            bit_index = bit_position & 7
            if int(payload[byte_index]) & (1 << bit_index):
                raw |= 1 << offset
            bit_position += 1
        out[index] = np.uint8(raw)
    return out


def _packed_byte_count(count: int, bits: int) -> int:
    _validate_bits(bits)
    return int(math.ceil(int(count) * int(bits) / 8))


def _validate_bits(bits: int) -> None:
    if int(bits) < 1 or int(bits) > 8:
        raise RuVectorHistoryError("quantization bits must be in the range 1..8")


__all__ = [
    "CompressedBreathingBuffer",
    "CompressedBreathingHistory",
    "CompressedHeartbeatHistory",
    "CompressedHeartbeatSpectrogram",
    "FrameQuantizationMetadata",
    "HistoryDiagnostics",
    "RuVectorHistoryError",
    "TierPolicy",
]
