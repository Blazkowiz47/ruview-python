# Calibration Baseline Drift Porting Notes

Milestone 6 adds a Python research notebook for empty-room baseline calibration and baseline drift inspection.

## Source References

- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-signal/src/ruvsense/calibration.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-signal/src/ruvsense/field_model.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/docs/adr/ADR-135-empty-room-baseline-calibration.md`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-signal/tests/calibration_synthetic.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-signal/tests/calibration_drift.rs`

The Rust calibration module captures a per-subcarrier empty-room baseline with Welford amplitude mean/variance and circular phase mean/dispersion. Runtime scoring compares live CSI to that baseline using median amplitude z-scores and median circular phase drift, while the longer drift path watches rolling squared z-score energy for sustained environmental changes.

## Python Notebook Intent

- `notebooks/06_calibration_baseline_drift.ipynb` replaces the placeholder with a valid unexecuted visual lab for deterministic synthetic CSI.
- The fixture uses an HT20-style 52-subcarrier stream with 600 empty-room calibration frames, gradual post-calibration amplitude/phase drift, and a localized person/motion perturbation event.
- The notebook plots baseline amplitude distribution, deviation score over time, drift/event markers, before/after amplitude baseline subtraction, and a per-subcarrier z-score heatmap.
- Guarded imports probe `ruview.ruvsense.calibration` when it exists. Local NumPy helpers remain the runnable fallback so the notebook is usable while the Milestone 6 core module API is still being reconciled.

## Intentional Deviations

- The notebook is not a binary ABI or persistence port. It does not write the Rust little-endian baseline format, TOML host files, or ESP32 NVS keys.
- The fallback helper path uses arrays rather than Rust `CsiFrame` metadata, so it does not enforce PHY tier mismatch errors or stamp calibration provenance.
- The phase path assumes already-sanitized synthetic phase. It does not run the Rust `phase_sanitizer.rs` or `phase_align.rs` preprocessing stages.
- ADR-030 SVD field modeling is referenced but not reproduced. The notebook focuses on the lower-level ADR-135 statistical baseline that feeds those later field-model and CIR stages.
- The drift score is visual and deterministic; it is not a production recalibration policy and does not attempt to prove that a room is empty before recalibration.

## Run Path

```bash
uv sync --extra research
uv run jupyter lab notebooks
```

Open `notebooks/06_calibration_baseline_drift.ipynb` and choose Run All. The notebook should run without radio hardware because it uses inline deterministic synthetic CSI and local fallback calibration helpers.
