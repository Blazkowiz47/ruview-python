# Learnings

Durable findings from this project. Keep this compact and useful for future work.

## Confirmed

- `plan.md` defines this as a research-first Python port of RuView / WiFi-DensePose, not a commercial product rewrite.
- The port should proceed by capability boundary and keep hardware-dependent behavior as protocol parsers, replay tools, and host-side simulations.
- Each ported module should include a fixture, test, or reproducible notebook experiment.
- The ADR-136 canonical CSI frame layout is fixed-width little-endian metadata plus stream-major complex payload encoded as `f64 re || f64 im`; the Python fixed witness vector for the first synthetic parity frame is `9ac4fdb9b6b9b2cca62b6ac6751a50400d49146d16435a7d3aed8350e01d983e`.
- Python frame constructors should copy boundary metadata when emulating Rust-owned frame contracts, otherwise later mutation of the original metadata object can silently change witness bytes.
- ESP32 ADR-018 raw CSI packets should follow the firmware and `wifi-densepose-hardware` layout, not the stale sensing-server raw parser offsets: `u16 n_subcarriers` at bytes 6-7, `u32 freq_mhz` at 8-11, `u32 sequence` at 12-15, RSSI/noise at 16-17, PPDU/flags at 18-19.
- ESP32 magic `0xC5110004` is ambiguous in the reference tree: `edge_processing.h` uses it for fused vitals while `wasm_runtime.h` uses it for WASM output. The current Python Milestone 2 parser implements the plan-requested WASM output shape and documents the collision.
- The first Python signal API deliberately favors compact NumPy primitives over the full Rust PSD/Doppler/correlation feature stack: `CsiWindow` uses `[time, stream, subcarrier]`, motion energy is mean squared temporal amplitude delta, and `subcarrier_variance` averages non-subcarrier axes by default.
- Synthetic CSI scenarios now provide deterministic empty-room, person-present, stillness, and walking windows. They are visual/debug fixtures, not calibrated RF channel models.
- Milestone 4 presence/motion uses deterministic NumPy heuristics rather than a trained detector: rolling baseline stats, positive relative increases, weighted motion components, and debounce are enough to separate the current synthetic empty/still/walking scenarios.
- Milestone 5 vitals ports the Rust ADR-021 pipeline conceptually with NumPy FFT/PSD helpers instead of exact streaming Rust IIR/FIR/autocorrelation internals. Unit fixtures estimate 18 BPM breathing and 72 BPM heart rate, while notebook confidence remains a tuning signal rather than a calibrated clinical result.
- Milestone 6 calibration preserves the ADR-135 statistical shape: Welford amplitude mean/variance, circular phase mean/dispersion, median amplitude z-score, median phase drift, and non-mutating amplitude baseline subtraction. The Python save/load path is JSON research format, not Rust binary ABI parity yet.
- Milestone 7 should stay local-first: source schemas and replay/simulation tests are deterministic, FastAPI is exposed through an app factory, and hardware UDP remains optional so `uv run pytest -q` never depends on radio packets.
- Milestone 8 advanced RuvSense APIs are intentionally research primitives: CIR uses active-subcarrier placement plus oversampled IFFT/top-k sparse taps rather than Rust's full ISTA solver, fusion uses deterministic NumPy weights, and temporal/adversarial modules use synthetic-test calibrated thresholds pending real CSI captures.
- Notebook smoke checks that execute matplotlib cells should set `MPLBACKEND=Agg` and close figures after each cell; this keeps automated `uv run --extra research` notebook verification non-interactive.
- Milestone 9 RuVector equivalents port behavior, not dependency internals: graph/min-cut becomes deterministic sensitivity-gap partitioning, solver calls become tiny NumPy least-squares/refinement routines, attention becomes explicit softmax weighting, and temporal tensor compression becomes quantized ring buffers.
- Milestone 10 keeps the default neural/training stack NumPy-only while exposing optional PyTorch modules behind the `nn` extra; tests use `pytest.importorskip("torch")` so the base `uv run pytest -q` path stays lightweight.
- Milestone 11 MAT stays research/local-only: disaster/survivor/vitals/domain objects and range/tracking/triage/alert behavior are ported as deterministic Python primitives, and alert dispatch is intentionally an in-memory lifecycle rather than SMS/MQTT/pager integration.
- Milestone 12 separates graph/provenance from privacy/BFLD: WorldGraph stays JSON/dataclass based, BFLD preserves the 86-byte little-endian header and sectioned payload behavior, and the engine-style trust path is a small witness helper rather than a full streaming-engine port.
- Milestone 13 optional tracks are deliberately pure research helpers: HOMECORE has local state/automation semantics, nvsim is a deterministic magnetic-scene simulator, swarm is simulation-only, and browser/desktop helpers build data plans without live browser, serial, flashing, sidecar, or network side effects.
- The full `plan.md` milestone stack is implemented through Milestone 13; the remaining high-value work is validation and tuning against real captures/reference behavior, not initial capability porting.

## Likely But Needs Verification

- Real ESP32 captures will be needed to tune Milestone 4 thresholds beyond the synthetic simulator.
- Real ESP32 captures will also be needed to tune vitals confidence thresholds, especially the heart-rate path where synthetic visual fixtures can detect the right peak but still report low confidence.
- Rust ADR-135 little-endian baseline serialization remains a future parity target if interchange with the Rust tools becomes important.
- The Python server currently normalizes simple JSONL replay rows and compact sensing updates, not the full Rust recording/session management API.
- Real multistatic node captures are needed to tune Milestone 8 quality gates, attention weights, and physically impossible signal thresholds beyond deterministic synthetic fixtures.
- Real RuVector-style multiband captures are needed to validate M9 subcarrier partitions, spectrogram gates, BVP attention weights, Fresnel path splits, and TDoA residual thresholds.
- Milestone 10 synthetic datasets and notebooks verify shapes and APIs, not training quality; real CSI/pose datasets are required before any model-performance claims.
- MAT thresholds and localization confidence are synthetic-fixture calibrated only; real rubble/debris CSI or UWB-style captures are needed before using M11 outputs as operational rescue evidence.
- M12 privacy and trust fixtures validate deterministic mechanics, not regulatory compliance; real deployment policy review is still needed before interpreting privacy modes as production guarantees.
- M13 HOMECORE/nvsim/swarm/browser/desktop helpers are synthetic-fixture verified only; live device behavior, browser rendering, and drone/serial/server integration remain out of scope until explicit hardware-backed experiments are added.

## Failed Approaches

-

## Reusable Ideas

- Preserve wire formats and data contracts where they matter, but prefer clear NumPy/SciPy/PyTorch implementations over opaque wrappers.
- Keep the default Python install small and place heavier research tools in optional extras so low-level parity tests remain quick to run.
- Keep advanced RuvSense modules importable through `ruview.ruvsense` with explicit exports; a small export smoke test catches missing public symbols after parallel worker slices land.
- Keep behavior-level ports in their own package namespace (`ruview.ruvector`) so later neural/training code can depend on compact signal and geometry helpers without pulling in external RuVector internals.
- Export optional neural components through `ruview.nn` with placeholders that explain the `uv sync --extra nn` path, while keeping NumPy-safe tensor, contrastive, dataset, loss, metric, checkpoint, and export helpers usable in the base environment.
- For namespaces with duplicate domain/localization concepts, export explicit aliases such as `DomainLocationUncertainty` and `LocalizationLocationUncertainty` to avoid accidental public-symbol overwrites after parallel worker slices land.
- Keep public package exports and tiny cross-module smoke tests as the parent integration layer after parallel workers; this caught the M12 composition boundary between WorldGraph provenance and BFLD privacy classes.
- Keep optional later-track modules side-effect free by default; deterministic data builders and simulation objects are easier to test, document, and compose with notebooks than live-control adapters.
