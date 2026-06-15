# 2026-06-15 - RuView detection signatures

Node: sushruts-macbook-pro
Repo path: /Users/sushrutpatwardhan/1Projects/ruview-python
Reference path: /Users/sushrutpatwardhan/1Projects/RuView

## Analysis

- Compared the Python port against original `RuView` for pose detection, heart rate, breathing rate, and person detection signatures.
- Original v2 live server publishes these through `SensingUpdate`: raw/derived node amplitudes, `FeatureInfo`, `ClassificationInfo`, optional `VitalSigns`, optional `pose_keypoints`, optional `persons`, and `estimated_persons`.
- Original live pose is derived from sensing features and vitals into COCO-style keypoints, then tracker-smoothed; it is not the primary trained DensePose path in the sensing server.
- Original vitals have two active paths: server-side amplitude/phase buffers with bandpass + FFT peak detection, and ESP32 edge packets with top-K phase subcarriers, biquad filters, zero-crossing BPM, motion/presence, and `n_persons`.
- Python port mirrors the contracts and research primitives: `CsiFrame`, `PersonPose`/COCO 17 keypoints, `SensingUpdate`, presence/motion heuristics, and FFT-based breathing/heart-rate estimators.
- Follow-up: detection is not strictly limited to direct line-of-sight. CSI changes arise in the Fresnel zone and through multipath reflections, but a single TX-RX link cannot uniquely localize a person in 3D. Original server `signal_field` is a 20x20 top-down visualization derived from subcarrier variance, while true position requires multiple calibrated links, node positions, and tomography/trilateration-style reconstruction.
- Breathing deep dive: original `wifi-densepose-vitals` uses EMA amplitude residuals, weighted subcarrier fusion, a 0.1-0.5 Hz IIR bandpass, zero crossings, and SNR confidence. Original live sensing-server instead buffers mean amplitude, applies a FIR bandpass, and finds an FFT peak. Python port mirrors the residual-fusion contract but uses NumPy FFT peak/prominence/in-band-energy scoring rather than streaming zero crossings.
- Added `notebooks/07_recording_presence_breathing_triage.ipynb` for local repo recordings only. It auto-discovers `data/recordings/**/*_csi.csv`, runs the current windowed presence heuristic, compares dominant-link versus all-packets breathing spectra, and documents why multiple breathers cannot be reliably separated from one scalar CSI waveform.
- Smoke result on `data/recordings/esp32_csi/dummy-test/rx01_desk_csi.csv`: presence heuristic predicts `moving` with mean presence score 0.540 and max 0.634; dominant-link breathing diagnostic estimates about 6.4 BPM with caveats, while the mixed all-packets diagnostic estimates about 14.5 BPM degraded confidence.
- Notebook cleanup: pruned old milestone/backlog notebooks and kept only active scope notebooks for person detection, motion detection, breathing/heart rate, signal visualization, pose estimation, local recording triage, three-receiver position triangulation, and an end-to-end capture/visualize/predict demo.
- Added `notebooks/README.md` as the scope map, `notebooks/08_position_triangulation_three_receivers.ipynb` for the triangulation idea, and `notebooks/00_end_to_end_capture_visualization_prediction.ipynb` as the single demo flow.
- Added a recording source switch to every notebook: `USE_RECORDING` plus `RECORDING_CSV` at the top for single-capture notebooks, and `RECEIVER_RECORDINGS` for the three-receiver triangulation notebook. Default synthetic fixtures remain runnable; `00` and `07` are recording-backed by default.
- Verification: `uv run pytest tests/unit/test_notebook_json.py -q` passed (`3 passed`); all nine notebooks execute with `MPLBACKEND=Agg`; recording-mode smoke passed for notebooks `01`-`06`; forced three-file triangulation smoke passed using the same sample CSV only as a wiring check.
- Added `docs/detection-workings.md` with Mermaid diagrams and first-pass explanations for the recording pipeline, presence/motion, breathing, heart rate, pose, line-of-sight vs multipath, and three-receiver triangulation. Linked it from `README.md`.
- Verification after doc change: `uv run pytest tests/unit/test_notebook_json.py -q` passed (`4 passed`).

## Next Action

- Use the original server `SensingUpdate` and ESP32 edge-vitals packet as the compatibility targets when documenting or aligning the Python API.
- Use notebook `07_recording_presence_breathing_triage.ipynb` for quick local capture triage before deeper per-link/multistatic analysis.
- Keep future notebooks aligned to the scope table in `notebooks/README.md`; put end-to-end user demos in `notebooks/00_end_to_end_capture_visualization_prediction.ipynb` unless a new scope is needed.
