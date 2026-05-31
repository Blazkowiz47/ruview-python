# 2026-05-31 - sushruts-macbook-pro-m3 - signal primitives

## Session

- Node: sushruts-macbook-pro-m3
- Device/server: local macOS workspace
- Repo path: `ruview-python`
- Branch/base: `master`, starting from `fbf21db Record milestone 2 completion`

## Source mappings

- `wifi-densepose-core/src/utils.rs`: complex magnitude/phase, phase unwrap, min-max, z-score parity.
- `wifi-densepose-signal/src/hampel.rs`: Hampel median/MAD algorithm and zero-MAD outlier behavior.
- `wifi-densepose-signal/src/features.rs`: amplitude/phase variance feature intent.
- `wifi-densepose-signal/src/subcarrier_selection.rs`: per-subcarrier sample variance across temporal samples.
- `wifi-densepose-sensing-server/src/csi.rs`: checked for CSI integration context; Python work uses existing `CsiFrame` contracts.

## Work log

- Implemented NumPy-first signal primitives in `src/ruview/signal/`: complex amplitude/phase conversion, arbitrary-axis phase unwrap, 1D Hampel filter, min-max and z-score normalization, scalar frame/window feature extraction, subcarrier variance, and `CsiWindow` frame stacking.
- Added deterministic unit coverage in `tests/unit/test_signal_processing.py`, including `CsiFrame` integration and window tensors shaped `[time, spatial_stream, subcarrier]`.

## Command/config

- Verification command: `uv run pytest -q`

## Result

- `uv run pytest -q` completed successfully: 32 tests passed.
- Signal primitives are scoped to the assigned signal package, unit test, and this dedicated memory note.

## Next action

- Downstream Milestone 3 workers can import the new primitives from `ruview.signal` and build higher-level processing flows on top of `CsiWindow`.
