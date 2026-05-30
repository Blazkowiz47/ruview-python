# Learnings

Durable findings from this project. Keep this compact and useful for future work.

## Confirmed

- `plan.md` defines this as a research-first Python port of RuView / WiFi-DensePose, not a commercial product rewrite.
- The port should proceed by capability boundary and keep hardware-dependent behavior as protocol parsers, replay tools, and host-side simulations.
- Each ported module should include a fixture, test, or reproducible notebook experiment.
- The ADR-136 canonical CSI frame layout is fixed-width little-endian metadata plus stream-major complex payload encoded as `f64 re || f64 im`; the Python fixed witness vector for the first synthetic parity frame is `9ac4fdb9b6b9b2cca62b6ac6751a50400d49146d16435a7d3aed8350e01d983e`.
- Python frame constructors should copy boundary metadata when emulating Rust-owned frame contracts, otherwise later mutation of the original metadata object can silently change witness bytes.

## Likely But Needs Verification

- Milestone 2 should begin with small handcrafted binary packet fixtures before adding live UDP capture.

## Failed Approaches

-

## Reusable Ideas

- Preserve wire formats and data contracts where they matter, but prefer clear NumPy/SciPy/PyTorch implementations over opaque wrappers.
- Keep the default Python install small and place heavier research tools in optional extras so low-level parity tests remain quick to run.
