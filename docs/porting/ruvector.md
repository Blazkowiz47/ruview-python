# RuVector Porting Notes

Milestone 9 ports the published RuVector integration behavior into Python
research equivalents. This is not dependency-internal parity with the Rust
`ruvector-*` crates: the Python side should expose readable, deterministic
signal and geometry behavior for experiments, notebooks, and tests while leaving
exact graph-cut, attention kernel, sparse-solver, and temporal-tensor internals
to the Rust reference.

## Source References

- `/Users/sushrutpatwardhan/1Projects/ruview-python/plan.md`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-ruvector/README.md`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-ruvector/src/signal/subcarrier.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-ruvector/src/signal/spectrogram.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-ruvector/src/signal/bvp.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-ruvector/src/signal/fresnel.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-ruvector/src/viewpoint/attention.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-ruvector/src/viewpoint/geometry.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-ruvector/src/viewpoint/fusion.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-ruvector/src/viewpoint/coherence.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-ruvector/src/mat/triangulation.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-ruvector/src/mat/breathing.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-ruvector/src/mat/heartbeat.rs`

## Deliverable Map

| Rust reference | Python deliverable | Behavior to preserve |
|---|---|---|
| `signal/subcarrier.rs` | `ruview.ruvector.signal.partition_subcarriers` and `subcarrier_importance_weights` | Split subcarriers into sensitive and insensitive groups from body-motion sensitivity scores; every index is covered exactly once; higher-sensitivity group receives larger downstream weights. |
| `signal/spectrogram.rs` | `ruview.ruvector.signal.gate_spectrogram` | Accept a time-frequency spectrogram, score body-motion frames, suppress noise-dominated periods, and keep the output shape unchanged for DensePose-style feature heads. |
| `signal/bvp.rs` | `ruview.ruvector.signal.attention_weighted_bvp` | Aggregate per-subcarrier STFT rows into one body velocity profile using sensitivity-biased attention rather than an unweighted mean. |
| `signal/fresnel.rs` | `ruview.ruvector.signal.solve_fresnel_geometry` | Estimate TX-body and body-RX path lengths from multi-subcarrier wavelength/amplitude observations, with the invariant that `d1 + d2` remains close to the known TX-RX baseline. |
| `mat/triangulation.rs` | `ruview.ruvector.mat.solve_tdoa_triangulation` | Estimate a 2-D survivor or target position from at least three TDoA measurements across AP pairs, returning `None` or a diagnostic when geometry is underdetermined. |
| `mat/breathing.rs` | `ruview.ruvector.mat.CompressedBreathingHistory` | Store streaming subcarrier breathing frames in a bounded, quantized history that preserves recent detail and exposes frame count plus decoded/demo views for analysis. |
| `mat/heartbeat.rs` | `ruview.ruvector.mat.CompressedHeartbeatHistory` | Store heartbeat spectrogram columns with per-frequency-bin history and expose band-power extraction over selected frequency bins. |
| `viewpoint/attention.rs`, `geometry.rs`, `fusion.rs`, `coherence.rs` | Shared helpers under `ruview.ruvector.viewpoint` or thin utilities reused by signal/MAT modules | Keep geometric bias, array diversity, coherence gating, and attention-weighted fusion semantics available for multistatic experiments without requiring Rust DDD event plumbing. |

## Notebook Intent

- `notebooks/11_ruvector_signal_geometry.ipynb` is the lightweight visual lab for
  this milestone.
- The fixture is deterministic and covers subcarrier sensitivity, gated
  spectrogram energy, BVP aggregation, Fresnel path-length estimation, TDoA
  triangulation, and compressed breathing/heartbeat histories.
- The notebook first probes `ruview.ruvector` with guarded imports. While code
  workers are still shaping the package, local NumPy fallbacks keep the notebook
  JSON-valid and runnable with only NumPy and Matplotlib.

## Intentional Deviations

- Python ports should prefer clear research APIs and typed NumPy arrays over
  line-by-line Rust translation.
- Graph cuts may be implemented with NetworkX or deterministic thresholded
  fallbacks; they do not need to reproduce `ruvector-mincut` residual-graph
  internals.
- Spectrogram and BVP attention may use PyTorch or NumPy softmax formulations;
  tests should verify shape, monotonic gating behavior, and sensitivity bias
  before claiming numeric parity.
- Fresnel and TDoA solvers may use SciPy or small closed-form least squares.
  Verification should focus on invariants, conditioning diagnostics, and known
  synthetic positions rather than Neumann-series iteration parity.
- Compressed temporal histories should document memory tiers and decoded
  analysis views, but they do not need byte-for-byte parity with
  `ruvector-temporal-tensor` segments.
- No notebook plot output, captured packet data, or hardware calibration state is
  committed as part of this milestone.

## Verification

Initial verification should include:

- JSON validity for `notebooks/11_ruvector_signal_geometry.ipynb`.
- The selected notebook scaffolding test in `tests/unit/test_notebook_json.py`.
- A smoke execution of notebook code cells with `MPLBACKEND=Agg` when feasible.
- Focused unit tests for future `ruview.ruvector` APIs using deterministic
  synthetic sensitivity vectors, spectrograms, STFT rows, Fresnel observations,
  AP layouts, and temporal-history inputs.
- Optional Rust-to-Python golden fixtures only after the parent process exports
  stable behavior traces from the Rust reference.

## Run Path

```bash
uv sync --extra research
uv run jupyter lab notebooks
```

Open `notebooks/11_ruvector_signal_geometry.ipynb` and choose Run All. The
notebook should run without radio hardware because it uses inline deterministic
synthetic data and local fallback helpers.
