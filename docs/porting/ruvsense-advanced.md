# RuvSense Advanced Porting Notes

Milestone 8 covers the advanced RuvSense signal modules and the first
multistatic visual notebook. The Python port should keep the Rust research
semantics visible while using deterministic fixtures until live ESP32 meshes and
captured replay data are available.

## Source References

- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-signal/src/ruvsense/multistatic.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-signal/src/ruvsense/phase_align.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-signal/src/ruvsense/multiband.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-signal/src/ruvsense/coherence.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-signal/src/ruvsense/coherence_gate.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-signal/src/ruvsense/cir.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-signal/src/ruvsense/field_model.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-signal/src/ruvsense/tomography.rs`

The multistatic Rust reference fuses the first channel from each
`MultiBandCsiFrame`, computes attention weights from amplitude agreement with a
consensus vector, and reports normalized weight entropy as cross-node
coherence. Related modules provide phase-offset removal, multi-band
channel-ordering/coherence, coherence gates, CIR dominant-tap checks, and later
field/tomography outputs.

## Module Map

| Rust reference | Python target | Notebook role |
|---|---|---|
| `cir.rs` | `ruview.ruvsense.cir` | Future CIR dominant-tap quality gate; documented as out of the fallback path. |
| `coherence.rs` | `ruview.ruvsense.coherence` | Guarded import target for scalar coherence helpers. |
| `coherence_gate.rs` | `ruview.ruvsense.coherence_gate` | Future accept/predict-only/reject/recalibrate gate policy. |
| `multiband.rs` | `ruview.ruvsense.multiband` | Guarded import target for per-node multi-channel fusion and channel coherence. |
| `phase_align.rs` | `ruview.ruvsense.phase_align` | Guarded import target for local-oscillator phase offset removal. |
| `multistatic.rs` | `ruview.ruvsense.multistatic` | Guarded import target for attention weights and fused sensing frames. |
| `field_model.rs` / `tomography.rs` | `ruview.ruvsense.field_model`, `ruview.ruvsense.tomography` | Future room-field and RF tomography backends. |

## Notebook Intent

- `notebooks/07_multistatic_node_comparison.ipynb` replaces the placeholder with
  a valid unexecuted visual lab.
- The notebook fixture creates four synthetic nodes around a small room, three
  channel observations per node, deterministic phase offsets, RSSI and packet
  health metadata, and one intentionally low-quality node with an incoherent
  multipath artifact.
- It compares single-node weights, naive all-node attention, and quality-gated
  multistatic fusion.
- It plots node quality/coherence, fusion weights, room-field heatmaps, fused
  subcarrier residuals, and target peak-error bars.
- Guarded imports probe future `ruview.ruvsense.multistatic`, `phase_align`,
  `multiband`, and `coherence` modules. Local NumPy helpers remain the runnable
  fallback while the Milestone 8 APIs settle.

## Intentional Deviations

- The fixture is deterministic and visual. It is not a channel model, a parity
  fixture, or a replacement for captured ESP32 mesh recordings.
- The fallback phase alignment uses static-subcarrier circular means instead of
  the Rust Neumann-style refinement loop.
- The fallback multi-band path averages channel amplitudes and circular phase
  after alignment. It does not preserve full `CanonicalCsiFrame` metadata.
- The fallback multistatic path adds an explicit quality gate to make low-quality
  node handling visible. The Rust reference separates attention entropy,
  scored fusion evidence, tolerated contradictions, and optional CIR blending.
- The room-field heatmap is an illustrative 2D Gaussian field. It is not the
  ADR-030 field-model SVD path or the RF tomography inverse solver.
- No generated plot output, hardware packet capture, or binary fixture is
  committed.

## Run Path

```bash
uv sync --extra research
uv run jupyter lab notebooks
```

Open `notebooks/07_multistatic_node_comparison.ipynb` and choose Run All. The
notebook should run without radio hardware because it uses inline deterministic
synthetic data and local fallback helpers.
