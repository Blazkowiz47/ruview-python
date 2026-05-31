# Presence And Motion Porting Notes

Milestone 4 ports the first presence and motion experiments from the Rust RuView codebase into lightweight Python research notebooks.

## Source References

- `RuView/v2/crates/wifi-densepose-signal/src/motion.rs`
- `RuView/v2/crates/wifi-densepose-sensing-server/src/csi.rs`

The Rust signal module combines variance, correlation, phase, and optional Doppler components into a normalized motion score. The sensing server adds frame-history features, baseline adjustment, smoothing, debounce, and simple motion labels using thresholds around absent, still, moving, and active states.

## Python Notebook Intent

- `notebooks/02_motion_vs_stillness.ipynb` compares deterministic simulator captures for `empty_room`, `stillness`, `walking`, and `person_present`.
- The notebook plots empty baseline behavior, standing-still vs walking motion scores, rolling variance, a synthetic state transition, a threshold sweep, and illustrative false-positive / false-negative frames.
- The notebook has guarded imports for future `ruview.signal.motion` scorer APIs and falls back to local helper cells while the classifier modules are still in progress.

## Intentional Deviations

- The notebook scorer is intentionally small: it uses rolling amplitude variance, frame-to-frame decorrelation, and phase-step energy with Rust-inspired weights.
- Calibration is visual and supervised by the deterministic synthetic empty/walking captures. It is not a production thresholding strategy.
- The notebook stays unexecuted in Git and does not commit binary fixtures or generated plot outputs.
- Real hardware effects such as packet loss, antenna geometry, carrier-frequency offset, calibration drift, and environmental motion are outside this notebook.

## Run Path

```bash
uv sync --extra research
uv run jupyter lab notebooks
```

Open `notebooks/02_motion_vs_stillness.ipynb` and choose Run All. The notebook should run without radio hardware because it uses deterministic synthetic CSI from `ruview.hardware.simulator`.
