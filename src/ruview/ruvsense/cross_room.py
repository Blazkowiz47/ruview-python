"""Room fingerprint matching and cross-room transition detection."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Hashable

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
RoomId = Hashable


class CrossRoomError(ValueError):
    """Base error for cross-room tracking operations."""


@dataclass(frozen=True)
class CrossRoomConfig:
    """Configuration for room and transition matching."""

    embedding_dim: int = 128
    min_similarity: float = 0.80
    room_min_similarity: float = 0.75
    max_gap_s: float = 60.0
    max_rooms: int = 100
    max_pending_exits: int = 200

    def __post_init__(self) -> None:
        if self.embedding_dim <= 0:
            raise CrossRoomError("embedding_dim must be positive")
        if self.max_rooms <= 0 or self.max_pending_exits <= 0:
            raise CrossRoomError("max_rooms and max_pending_exits must be positive")
        if self.max_gap_s <= 0.0 or not math.isfinite(self.max_gap_s):
            raise CrossRoomError("max_gap_s must be a finite positive value")
        for name in ("min_similarity", "room_min_similarity"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise CrossRoomError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class RoomFingerprint:
    """Static electromagnetic fingerprint for a room."""

    room_id: RoomId
    embedding: ArrayLike
    computed_at_us: int = 0
    node_count: int = 1

    def __post_init__(self) -> None:
        if self.node_count <= 0:
            raise CrossRoomError("node_count must be positive")
        object.__setattr__(self, "embedding", _as_embedding(self.embedding))
        object.__setattr__(self, "computed_at_us", int(self.computed_at_us))


@dataclass(frozen=True)
class RoomMatchResult:
    """Nearest room fingerprint result."""

    matched: bool
    room_id: RoomId | None
    similarity: float
    fingerprint: RoomFingerprint | None


@dataclass(frozen=True)
class ExitEvent:
    """A person track leaving one room."""

    room_id: RoomId
    track_id: int
    embedding: ArrayLike
    timestamp_us: int
    matched: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "embedding", _as_embedding(self.embedding))
        object.__setattr__(self, "track_id", int(self.track_id))
        object.__setattr__(self, "timestamp_us", int(self.timestamp_us))


@dataclass(frozen=True)
class EntryEvent:
    """A person track appearing in a room."""

    room_id: RoomId
    track_id: int
    embedding: ArrayLike
    timestamp_us: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "embedding", _as_embedding(self.embedding))
        object.__setattr__(self, "track_id", int(self.track_id))
        object.__setattr__(self, "timestamp_us", int(self.timestamp_us))


@dataclass(frozen=True)
class TransitionEvent:
    """Immutable cross-room identity continuity record."""

    person_id: int
    from_room: RoomId
    to_room: RoomId
    exit_track_id: int
    entry_track_id: int
    similarity: float
    gap_s: float
    timestamp_us: int


@dataclass(frozen=True)
class MatchResult:
    """Result of matching an entry against pending exits."""

    matched: bool
    transition: TransitionEvent | None
    candidates_checked: int
    best_similarity: float


class CrossRoomTracker:
    """Maintains room fingerprints, pending exits, and transition records."""

    def __init__(self, config: CrossRoomConfig | None = None) -> None:
        self.config = config if config is not None else CrossRoomConfig()
        self._rooms: dict[RoomId, RoomFingerprint] = {}
        self._pending_exits: list[ExitEvent] = []
        self._transitions: list[TransitionEvent] = []
        self._next_person_id = 1

    @property
    def room_count(self) -> int:
        return len(self._rooms)

    @property
    def pending_exit_count(self) -> int:
        return sum(1 for event in self._pending_exits if not event.matched)

    @property
    def transition_count(self) -> int:
        return len(self._transitions)

    @property
    def transitions(self) -> tuple[TransitionEvent, ...]:
        return tuple(self._transitions)

    def register_room(self, fingerprint: RoomFingerprint) -> None:
        self._validate_dim(fingerprint.embedding)
        is_new = fingerprint.room_id not in self._rooms
        if is_new and len(self._rooms) >= self.config.max_rooms:
            raise CrossRoomError(f"maximum rooms exceeded: limit is {self.config.max_rooms}")
        self._rooms[fingerprint.room_id] = fingerprint

    def room_fingerprint(self, room_id: RoomId) -> RoomFingerprint | None:
        return self._rooms.get(room_id)

    def match_room(self, embedding: ArrayLike) -> RoomMatchResult:
        """Match a static CSI embedding to the nearest registered room."""

        query = _as_embedding(embedding)
        self._validate_dim(query)
        best_room: RoomFingerprint | None = None
        best_similarity = 0.0
        for fingerprint in self._rooms.values():
            similarity = cosine_similarity(query, fingerprint.embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_room = fingerprint
        matched = best_room is not None and best_similarity >= self.config.room_min_similarity
        return RoomMatchResult(
            matched=matched,
            room_id=best_room.room_id if matched and best_room is not None else None,
            similarity=float(best_similarity),
            fingerprint=best_room if matched else None,
        )

    def record_exit(self, event: ExitEvent) -> None:
        self._validate_dim(event.embedding)
        if len(self._pending_exits) >= self.config.max_pending_exits:
            self._pending_exits.pop(0)
        self._pending_exits.append(event)

    def match_entry(self, entry: EntryEvent) -> MatchResult:
        """Try to link an entry event to the best unmatched cross-room exit."""

        self._validate_dim(entry.embedding)
        best_index: int | None = None
        best_similarity = 0.0
        candidates_checked = 0

        for index, exit_event in enumerate(self._pending_exits):
            if exit_event.matched or exit_event.room_id == entry.room_id:
                continue
            gap_s = max(0.0, (entry.timestamp_us - exit_event.timestamp_us) / 1_000_000.0)
            if gap_s > self.config.max_gap_s:
                continue

            candidates_checked += 1
            similarity = cosine_similarity(exit_event.embedding, entry.embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_index = index

        if best_index is None or best_similarity < self.config.min_similarity:
            return MatchResult(
                matched=False,
                transition=None,
                candidates_checked=candidates_checked,
                best_similarity=float(best_similarity),
            )

        exit_event = self._pending_exits[best_index]
        gap_s = max(0.0, (entry.timestamp_us - exit_event.timestamp_us) / 1_000_000.0)
        transition = TransitionEvent(
            person_id=self._next_person_id,
            from_room=exit_event.room_id,
            to_room=entry.room_id,
            exit_track_id=exit_event.track_id,
            entry_track_id=entry.track_id,
            similarity=float(best_similarity),
            gap_s=float(gap_s),
            timestamp_us=entry.timestamp_us,
        )
        self._next_person_id += 1
        self._pending_exits[best_index] = replace(exit_event, matched=True)
        self._transitions.append(transition)
        return MatchResult(
            matched=True,
            transition=transition,
            candidates_checked=candidates_checked,
            best_similarity=float(best_similarity),
        )

    def expire_exits(self, current_us: int) -> None:
        max_gap_us = int(self.config.max_gap_s * 1_000_000.0)
        now = int(current_us)
        self._pending_exits = [
            event
            for event in self._pending_exits
            if not event.matched and now - event.timestamp_us <= max_gap_us
        ]

    def transitions_between(self, from_room: RoomId, to_room: RoomId) -> tuple[TransitionEvent, ...]:
        return tuple(t for t in self._transitions if t.from_room == from_room and t.to_room == to_room)

    def _validate_dim(self, embedding: FloatArray) -> None:
        if embedding.shape != (self.config.embedding_dim,):
            raise CrossRoomError(
                f"embedding dimension mismatch: expected {self.config.embedding_dim}, "
                f"got {embedding.shape[0]}"
            )


def cosine_similarity(a: ArrayLike, b: ArrayLike) -> float:
    """Cosine similarity with a zero-vector guard."""

    left = _as_embedding(a)
    right = _as_embedding(b)
    if left.shape != right.shape:
        raise CrossRoomError(f"embedding dimension mismatch: expected {left.shape[0]}, got {right.shape[0]}")
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(left, right) / denom)


def _as_embedding(value: ArrayLike) -> FloatArray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise CrossRoomError("embedding must be a one-dimensional vector")
    if arr.size == 0:
        raise CrossRoomError("embedding must not be empty")
    if not np.all(np.isfinite(arr)):
        raise CrossRoomError("embedding contains non-finite values")
    return np.array(arr, dtype=np.float64, copy=True)


__all__ = [
    "CrossRoomConfig",
    "CrossRoomError",
    "CrossRoomTracker",
    "EntryEvent",
    "ExitEvent",
    "MatchResult",
    "RoomFingerprint",
    "RoomId",
    "RoomMatchResult",
    "TransitionEvent",
    "cosine_similarity",
]
