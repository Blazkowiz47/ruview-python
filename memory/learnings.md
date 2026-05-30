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

## Likely But Needs Verification

- Milestone 2 should begin with small handcrafted binary packet fixtures before adding live UDP capture.

## Failed Approaches

-

## Reusable Ideas

- Preserve wire formats and data contracts where they matter, but prefer clear NumPy/SciPy/PyTorch implementations over opaque wrappers.
- Keep the default Python install small and place heavier research tools in optional extras so low-level parity tests remain quick to run.
