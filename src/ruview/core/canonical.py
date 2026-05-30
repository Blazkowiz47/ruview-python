"""Deterministic canonical frame serialization and witness hashing."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class CanonicalFrame(Protocol):
    """Protocol for frames with architecture-independent bytes."""

    def to_canonical_bytes(self) -> bytes:
        """Return deterministic little-endian frame bytes."""

    def witness_hash(self) -> bytes:
        """Return the BLAKE3-256 hash of the canonical bytes."""


def witness_hash(frame: CanonicalFrame) -> bytes:
    """Return the BLAKE3-256 witness hash for a canonical frame."""

    return witness_hash_bytes(frame.to_canonical_bytes())


def witness_hash_bytes(payload: bytes) -> bytes:
    """Return the BLAKE3-256 digest for ``payload``."""

    import blake3

    return blake3.blake3(payload).digest()

