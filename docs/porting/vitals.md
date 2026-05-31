# Vitals Porting Notes

Milestone 5 covers the first Python research notebook for breathing and heart-rate frequency-band inspection.

## Source References

- `RuView/v2/crates/wifi-densepose-vitals`
- `RuView/v2/crates/wifi-densepose-vitals/src/preprocessor.rs`
- `RuView/v2/crates/wifi-densepose-vitals/src/breathing.rs`
- `RuView/v2/crates/wifi-densepose-vitals/src/heartrate.rs`
- `RuView/v2/crates/wifi-densepose-sensing-server/src/vital_signs.rs`
- `RuView/firmware/esp32-csi-node/README.md`

The Rust vitals crate uses EMA static suppression to produce per-subcarrier residuals, breathing extraction in the 0.1-0.5 Hz band, heart-rate extraction in the 0.8-2.0 Hz band, confidence/status values, and historical storage. The sensing-server reference also documents FFT-based spectral peak extraction and signal-quality heuristics. The ESP32 firmware README records the edge Tier 2 bands and caveats for breathing, heart rate, presence, fall detection, and vitals packets.

## Python Notebook Intent

- `notebooks/03_breathing_and_heart_rate_bands.ipynb` replaces the placeholder with a valid unexecuted visual lab for deterministic synthetic CSI amplitude/phase arrays.
- The clean fixture embeds breathing around 0.3 Hz / 18 bpm and heart motion around 1.2 Hz / 72 bpm.
- Additional fixtures cover a noisy-motion case and a static/no-person case so confidence and quality labels can be inspected.
- The notebook plots raw EMA residuals, breathing-band and heart-band filtered signals, Welch PSD with detected peak markers, rolling BPM estimates, rolling confidence, and quality labels.
- Guarded imports probe in-repo `ruview.vitals` classes and fall back to local helpers if the API is absent or changes shape.

## Intentional Deviations

- The notebook is an exploratory visualization, not a line-by-line port of the Rust crate or firmware.
- Local helper cells use readable NumPy/SciPy offline filtering and Welch PSD. The Rust references use EMA preprocessing, IIR/FIR filtering, zero-crossing, autocorrelation, FFT peak analysis, and edge-oriented confidence logic.
- The synthetic fixture is deterministic and compact. It does not model real multipath geometry, antenna placement, packet loss, ESP32 quantization, phase unwrap failures, or validated clinical thresholds.
- The notebook stays unexecuted in Git and does not commit generated plot outputs or binary fixtures.

## Run Path

```bash
uv sync --extra research
uv run jupyter lab notebooks
```

Open `notebooks/03_breathing_and_heart_rate_bands.ipynb` and choose Run All. The notebook should run without radio hardware because it uses inline deterministic synthetic data, guarded `ruview.vitals` probes, and local fallback helpers.
