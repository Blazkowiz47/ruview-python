"""Trust-throughline helpers for semantic WorldGraph records."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import blake3

from ruview.privacy.bfld import PrivacyAction, PrivacyClass, PrivacyModeRegistry
from ruview.worldgraph.graph import PrivacyRollup, WorldGraph
from ruview.worldgraph.model import WorldId
from ruview.worldgraph.provenance import SemanticProvenance


@dataclass(frozen=True)
class TrustedSemanticState:
    """Auditable result of appending one semantic state to a WorldGraph."""

    semantic_id: WorldId
    effective_class: PrivacyClass
    demoted: bool
    provenance: SemanticProvenance
    witness: bytes

    @property
    def witness_hex(self) -> str:
        return self.witness.hex()


def demote_one(privacy_class: PrivacyClass) -> PrivacyClass:
    """Demote a privacy class by one stricter step, clamped at Restricted."""

    current = PrivacyClass(privacy_class)
    return PrivacyClass(min(current.as_u8() + 1, PrivacyClass.Restricted.as_u8()))


def witness_of(provenance: SemanticProvenance, privacy_class: PrivacyClass) -> bytes:
    """Return the deterministic BLAKE3 witness for a trust decision."""

    p = SemanticProvenance.coerce(provenance)
    h = blake3.blake3()
    for evidence in p.evidence:
        h.update(evidence.encode("utf-8"))
        h.update(b"\x1f")
    h.update(p.model_version.encode("utf-8"))
    h.update(p.calibration_version.encode("utf-8"))
    h.update(p.privacy_decision.encode("utf-8"))
    h.update(bytes([PrivacyClass(privacy_class).as_u8()]))
    return h.digest()


def record_trusted_semantic_state(
    graph: WorldGraph,
    *,
    statement: str,
    confidence: float,
    evidence: Sequence[str],
    model_version: str,
    calibration_version: str | None,
    privacy: PrivacyModeRegistry,
    valid_from_unix_ms: int,
    evidence_sources: Sequence[WorldId | int] = (),
    force_demote: bool = False,
) -> TrustedSemanticState:
    """Append a semantic state with provenance, privacy decision, and witness."""

    base_class = privacy.active_class
    demoted = bool(force_demote or calibration_version is None)
    effective_class = demote_one(base_class) if demoted else base_class
    provenance = SemanticProvenance(
        evidence=tuple(str(item) for item in evidence),
        model_version=str(model_version),
        calibration_version=str(calibration_version) if calibration_version is not None else "cal:none",
        privacy_decision=f"{privacy.active_mode.name}/{effective_class.name}",
    )
    semantic_id = graph.add_semantic_state(
        statement=str(statement),
        confidence=float(confidence),
        valid_from_unix_ms=int(valid_from_unix_ms),
        provenance=provenance,
        evidence_sources=evidence_sources,
    )
    return TrustedSemanticState(
        semantic_id=semantic_id,
        effective_class=effective_class,
        demoted=demoted,
        provenance=provenance,
        witness=witness_of(provenance, effective_class),
    )


def apply_active_privacy_mode(graph: WorldGraph, privacy: PrivacyModeRegistry) -> PrivacyRollup:
    """Materialize privacy-limiting edges for the registry's active mode."""

    suppress_identity = privacy.is_action_enforced(PrivacyAction.SuppressIdentity)
    return graph.apply_privacy_mode(
        mode=privacy.active_mode.name,
        action=PrivacyAction.SuppressIdentity.name,
        policy=lambda _sensor_kind, node_kind: not (suppress_identity and node_kind == "person_track"),
    )


__all__ = [
    "TrustedSemanticState",
    "apply_active_privacy_mode",
    "demote_one",
    "record_trusted_semantic_state",
    "witness_of",
]
