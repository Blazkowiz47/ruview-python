# WorldGraph, Trust, And Privacy Porting Notes

Milestone 12 ports the trust-traceable graph and privacy workflow from the
Rust RuView reference into Python research equivalents. The Python scope is a
local, deterministic visual and notebook workflow first; full graph APIs,
privacy gates, and witness replay can follow once the research contracts settle.

## Source References

- `ruview-python/plan.md`
- `RuView/v2/crates/wifi-densepose-worldgraph/src/model.rs`
- `RuView/v2/crates/wifi-densepose-worldgraph/src/graph.rs`
- `RuView/v2/crates/wifi-densepose-engine/src/lib.rs`
- `RuView/v2/crates/wifi-densepose-bfld/README.md`
- `RuView/v2/crates/wifi-densepose-bfld/src/privacy_mode.rs`
- `RuView/v2/crates/wifi-densepose-bfld/src/privacy_gate.rs`
- `RuView/v2/crates/wifi-densepose-bfld/src/identity_risk.rs`
- `RuView/v2/crates/wifi-densepose-bfld/src/frame.rs`

## Deliverable Map

| Rust reference | Python research equivalent | Behavior to preserve |
|---|---|---|
| `WorldId`, `WorldNode`, `WorldEdge`, `SemanticProvenance` | `ruview.worldgraph` typed dataclasses or enums for stable ids, room/zone/sensor/link/person/object/event/semantic nodes, relation records, and provenance bundles | Stable graph ids survive persistence; semantic states always name evidence handles, model version, calibration version, and privacy decision. |
| `WorldGraph` and `WorldGraphSnapshot` | A small directed graph container with deterministic JSON snapshots | Upsert-by-stable-id, typed edges, observation/location queries, room lookup by `area_id`, append-only semantic beliefs, and deterministic round trips. |
| `add_semantic_state`, `add_contradiction`, `DerivedFrom`, `Contradicts` | Provenance inspection helpers and notebook graph overlays | Beliefs are retained even when contradictory, and provenance stays queryable instead of being flattened into text. |
| `apply_privacy_mode` and `PrivacyRollup` | Local privacy rollup helpers for active-mode impact summaries | Observation policy returns allowed/denied pairs, suppressed nodes, and count of still-observable pairs; privacy edges record the mode/action/allowed decision. |
| `StreamingEngine::process_cycle*` | Later composition helper connecting fusion quality, calibration, privacy demotion, semantic state, and witness hash | A tolerated contradiction or calibration mismatch can only demote privacy class; the emitted semantic state carries the trust tuple and graph anchor. |
| `PrivacyMode`, `PrivacyAction`, `PrivacyModeRegistry`, `PrivacyAttestationProof` | `ruview.privacy` mode registry plus audit-chain helpers | Modes map to target classes and enforced actions; mode changes append a hash-chained attestation over previous hash, mode byte, action bits, and class byte. |
| `PrivacyGate::demote` | One-way BFLD frame demotion helper | Demotion never increases information density; identity-leaky payload sections are zeroed/removed and payload CRC metadata is resynced. |
| `identity_risk::score` and `GateAction::from_score` | Identity-risk score and gate-threshold helpers | Risk is `sep * stab * consist * conf` after clamping; thresholds are `<0.5 Accept`, `0.5..0.7 PredictOnly`, `0.7..0.9 Reject`, and `>=0.9 Recalibrate`. |
| `BfldFrameHeader` | Research metadata record for BFLD notebook fixtures | Keep header intent visible: privacy class byte, flags, timestamp, channel metadata, payload length, and payload CRC, without emitting raw RF frames. |

## Notebook Intent

- `notebooks/13_worldgraph_privacy_provenance.ipynb` is the lightweight visual
  lab for this milestone.
- The fixture is deterministic and covers one room, two WiFi CSI sensors, one
  person track, a provenance-bearing semantic state, privacy-mode rollups, BFLD
  frame demotion effects, and identity-risk gate thresholds.
- The notebook probes `ruview.worldgraph` and `ruview.privacy` with guarded
  imports. Until those packages expose real APIs, local NumPy/Matplotlib
  helpers keep the notebook JSON-valid and smoke-testable.

## Research-Only Boundary

- WorldGraph snapshots in this Python port are local research artifacts. They
  should not contain raw BFI, raw CSI payloads, identity embeddings, MAC
  addresses, or cross-site identifiers.
- Trust and privacy proofs are local-only diagnostics unless a later milestone
  explicitly adds signed export formats. The notebook may visualize witness-like
  hashes, but it must not imply remote attestation or compliance guarantees.
- BFLD privacy classes and risk thresholds are shown to inspect behavior on
  synthetic fixtures. They are not user-consent UX, regulatory controls, or
  production privacy policy.

## Intentional Deviations

- Python should prefer readable dataclasses and deterministic JSON over
  line-by-line `petgraph` parity. NetworkX may be useful for visualization, but
  core tests should not require graph-layout randomness.
- The first notebook uses local dictionaries for nodes, edges, privacy modes,
  and payload sections because `src/ruview/worldgraph` and `src/ruview/privacy`
  are currently namespace placeholders.
- The notebook rollup policy is diagnostic: identity-suppressing modes hide
  `person_track` observations, and aggregate-only modes also suppress
  per-entity semantic observations. The Rust engine's current composed helper
  demonstrates the narrower `person_track` suppression path.
- Witness hashing can use Python `blake3` later, but this docs/notebook slice
  records only the tuple that must feed the witness: evidence, model,
  calibration, privacy decision, and effective class.
- The BFLD payload-demotion plot visualizes section removal at a high level; it
  does not claim byte-for-byte `BfldPayload` or CRC parity.

## Verification

Initial verification should include:

- JSON validity for `notebooks/13_worldgraph_privacy_provenance.ipynb`.
- The selected notebook scaffolding test in `tests/unit/test_notebook_json.py`.
- A smoke execution of notebook code cells with `MPLBACKEND=Agg`.
- Future unit tests for stable-id upsert, deterministic graph snapshots,
  `DerivedFrom` and `Contradicts` edges, privacy rollup suppression, mode
  attestation-chain verification, one-way frame demotion, identity-risk
  threshold boundaries, and local witness determinism.

## Run Path

```bash
uv sync --extra research
uv run jupyter lab notebooks
```

Open `notebooks/13_worldgraph_privacy_provenance.ipynb` and choose Run All. The
notebook should run without radio hardware, captured packets, live identity
data, or completed `ruview.worldgraph` / `ruview.privacy` implementations.
