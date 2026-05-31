# RuView Python Research Port Plan

## Purpose

This project is a pure Python research port of the RuView / WiFi-DensePose
repository at:

```text
RuView
```

The goal is to port the functionality one capability at a time into a readable,
experiment-friendly Python codebase. This is not a commercial application and
should not optimize for product packaging, app-store flows, cloud distribution,
or smart-home polish unless those pieces directly support research.

## Guiding Principles

- Research-first, not product-first.
- Prefer clear NumPy/SciPy/PyTorch implementations over opaque wrappers.
- Port by capability boundary, not by line-by-line file translation.
- Preserve wire formats and data contracts where they matter.
- Add notebooks for visual inspection of every major signal-processing stage.
- Every ported module should include a small fixture, test, or reproducible
  notebook experiment.
- Keep hardware-dependent firmware behavior as protocol parsers, replay tools,
  and host-side simulations in Python. Do not try to replace ESP-IDF CSI capture
  with normal desktop Python.

## Proposed Project Layout

```text
ruview-python/
  pyproject.toml
  plan.md
  src/ruview/
    core/
      frames.py
      pose.py
      confidence.py
      errors.py
      canonical.py
    protocols/
      esp32.py
      sync.py
      rvf.py
      witness.py
    signal/
      csi_processor.py
      phase.py
      hampel.py
      features.py
      motion.py
      spectrogram.py
      subcarrier.py
      fresnel.py
      bvp.py
    vitals/
      preprocessing.py
      breathing.py
      heartrate.py
      smoothing.py
      quality.py
    ruvsense/
      calibration.py
      cir.py
      coherence.py
      coherence_gate.py
      multiband.py
      phase_align.py
      multistatic.py
      pose_tracker.py
      field_model.py
      tomography.py
      gesture.py
      intention.py
      cross_room.py
      adversarial.py
    hardware/
      udp_receiver.py
      replay.py
      simulator.py
      recordings.py
    server/
      app.py
      schemas.py
      websocket.py
      sources.py
    training/
      datasets.py
      embeddings.py
      contrastive.py
      trainer.py
      checkpoints.py
    nn/
      densepose.py
      translator.py
      rf_encoder.py
      inference.py
    mat/
      domain.py
      detection.py
      localization.py
      tracking.py
      triage.py
    worldgraph/
      graph.py
      model.py
      provenance.py
    privacy/
      bfld.py
      privacy_gate.py
      identity_risk.py
    swarm/
      topology.py
      formation.py
      planning.py
      sensing.py
    cli/
      main.py

  notebooks/
    00_signal_playground.ipynb
    01_empty_room_vs_person_present.ipynb
    02_motion_vs_stillness.ipynb
    03_breathing_and_heart_rate_bands.ipynb
    04_phase_and_amplitude_visualization.ipynb
    05_subcarrier_heatmaps.ipynb
    06_calibration_baseline_drift.ipynb
    07_multistatic_node_comparison.ipynb
    08_csi_to_pose_experiment.ipynb
    09_dataset_replay_lab.ipynb
    10_model_embedding_visualization.ipynb

  examples/
    simulate_empty_vs_present.py
    replay_recording.py
    live_udp_viewer.py

  data/
    fixtures/
    synthetic/
    recordings/

  tests/
    unit/
    parity/
    fixtures/
```

## Non-Commercial Scope Boundary

De-prioritize or keep as reference-only:

- app-store or cog marketplace UX
- product onboarding and sales/demo pages
- commercial cloud registry flows
- telemetry and production analytics
- Docker Hub or release packaging polish
- HomeKit/Matter production polish unless needed for a research experiment
- desktop app polish beyond useful hardware/research controls

Keep and port where useful:

- local signal experiments
- notebooks and visualizations
- hardware packet parsing
- replayable recordings
- deterministic fixtures
- model training experiments
- offline evaluation
- privacy and witness research primitives
- graph/provenance research

## Porting Method

For each capability:

1. Identify the source files in the reference repository.
2. Write a small Python API with typed dataclasses or Pydantic models.
3. Add a synthetic fixture or captured packet fixture.
4. Add a unit or parity test.
5. Add or update a notebook that visualizes the behavior.
6. Record any intentional deviations from the Rust/C implementation.

For numeric DSP parity, use tolerances:

```python
import numpy as np

np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)
```

## Milestone 0: Project Scaffold

Create the Python research package:

- `pyproject.toml`
- `src/ruview/`
- `tests/`
- `notebooks/`
- `examples/`
- `data/fixtures/`
- `data/synthetic/`
- `data/recordings/`

Recommended Python stack:

- NumPy
- SciPy
- pandas
- matplotlib
- plotly
- seaborn
- scikit-learn
- PyTorch
- FastAPI
- websockets or FastAPI WebSocket support
- pytest
- hypothesis, optional
- networkx
- jupyterlab
- ipywidgets

## Milestone 1: Core Data Contracts

Reference source:

- `v2/crates/wifi-densepose-core`

Port:

- `CsiFrame`
- `CsiMetadata`
- `ComplexSample`
- `Timestamp`
- `DeviceId`
- `FrameId`
- `Confidence`
- `Keypoint`
- `PersonPose`
- `PoseEstimate`
- canonical frame byte encoding
- witness hash helpers

Deliverables:

- `src/ruview/core/`
- unit tests for construction, validation, and canonical bytes
- notebook cells that create synthetic CSI frames and inspect their shape

## Milestone 2: ESP32 Protocol Parsers

Reference source:

- `firmware/esp32-csi-node/main/csi_collector.c`
- `firmware/esp32-csi-node/main/stream_sender.c`
- `v2/crates/wifi-densepose-sensing-server/src/csi.rs`
- `v2/crates/wifi-densepose-hardware`

Port host-side formats:

- raw CSI packet, magic `0xC511_0001`
- edge vitals packet, magic `0xC511_0002`
- WASM event packet, magic `0xC511_0004`
- sync packet, magic `0xC511_A110`
- amplitude and phase extraction from I/Q bytes

Do not port:

- ESP-IDF WiFi CSI callback itself
- FreeRTOS task logic
- embedded socket stack internals

Deliverables:

- `src/ruview/protocols/esp32.py`
- `src/ruview/hardware/udp_receiver.py`
- binary packet fixtures
- parser unit tests
- packet inspection notebook cells

## Milestone 3: Signal Visual Lab

Reference source:

- `v2/crates/wifi-densepose-signal`
- `v2/crates/wifi-densepose-sensing-server/src/csi.rs`

Port:

- amplitude and phase conversion
- CSI preprocessing
- phase unwrap
- Hampel filter
- feature extraction
- variance and motion energy
- subcarrier variance
- signal-field generation
- simple synthetic CSI generator

Deliver notebooks:

- `00_signal_playground.ipynb`
- `01_empty_room_vs_person_present.ipynb`
- `05_subcarrier_heatmaps.ipynb`

Primary visualizations:

- amplitude over time
- phase over time
- I/Q scatter
- subcarrier heatmap
- variance by subcarrier
- empty-room vs person-present overlay
- motion energy over time
- normalized vs raw signal

This milestone should produce the first clear visual answer to:

```text
What changes in CSI when a person is present?
```

## Milestone 4: Presence And Motion Experiments

Reference source:

- `v2/crates/wifi-densepose-signal/src/motion.rs`
- `v2/crates/wifi-densepose-sensing-server/src/csi.rs`

Port:

- motion score
- presence classification
- moving vs still heuristics
- rolling baseline
- smoothing and debounce
- simple adaptive threshold experiments

Deliver notebooks:

- `02_motion_vs_stillness.ipynb`
- updates to `01_empty_room_vs_person_present.ipynb`

Visualizations:

- empty room baseline
- person standing still
- person walking
- transition plots
- threshold sweep plots
- false positive and false negative examples

## Milestone 5: Vitals

Reference source:

- `v2/crates/wifi-densepose-vitals`
- `v2/crates/wifi-densepose-sensing-server/src/vital_signs.rs`

Port:

- CSI vital preprocessor
- breathing residual extraction
- breathing rate estimator
- heart-rate estimator
- bandpass filters
- confidence scoring
- signal quality labels
- smoothing buffers

Deliver notebook:

- `03_breathing_and_heart_rate_bands.ipynb`

Visualizations:

- raw residual signal
- breathing band `0.1-0.5 Hz`
- heart band `0.8-2.0 Hz`
- FFT or Welch PSD
- detected peak frequency
- BPM over time
- confidence over time

## Milestone 6: Calibration And Baseline Drift

Reference source:

- `v2/crates/wifi-densepose-signal/src/ruvsense/calibration.rs`
- `v2/crates/wifi-densepose-signal/src/ruvsense/field_model.rs`

Port:

- empty-room baseline recording
- Welford running stats
- amplitude baseline
- phase baseline
- calibration deviation score
- drift trigger
- baseline save/load

Deliver notebook:

- `06_calibration_baseline_drift.ipynb`

Visualizations:

- baseline amplitude distribution
- deviation score over time
- drift event marker
- before/after calibration comparison
- per-subcarrier z-score heatmap

## Milestone 7: Python Research Sensing Server

Reference source:

- `v2/crates/wifi-densepose-sensing-server`

Build:

- FastAPI server
- UDP source
- simulated source
- recording replay source
- WebSocket `/ws/sensing`
- REST `/api/v1/sensing/latest`
- REST `/api/v1/vital-signs`
- local-only config

Output message shape:

```json
{
  "type": "sensing_update",
  "source": "simulated|esp32|replay",
  "tick": 1,
  "nodes": [],
  "features": {},
  "classification": {},
  "signal_field": {},
  "vital_signs": {},
  "estimated_persons": 1
}
```

Deliver examples:

- `examples/simulate_empty_vs_present.py`
- `examples/replay_recording.py`
- `examples/live_udp_viewer.py`

## Milestone 8: RuvSense Advanced Signal Modules

Reference source:

- `v2/crates/wifi-densepose-signal/src/ruvsense`

Port in this order:

1. `cir.py`: CSI to CIR sparse recovery
2. `coherence.py`: coherence scoring
3. `coherence_gate.py`: accept, predict-only, reject, recalibrate decisions
4. `multiband.py`: multi-band CSI fusion
5. `phase_align.py`: LO phase offset estimation
6. `multistatic.py`: attention-weighted fusion
7. `pose_tracker.py`: 17-keypoint Kalman tracker
8. `field_model.py`: SVD room eigenstructure
9. `tomography.py`: RF tomography
10. `longitudinal.py`: drift and biomechanics trends
11. `intention.py`: pre-movement signals
12. `cross_room.py`: room fingerprint transitions
13. `gesture.py`: DTW gestures
14. `adversarial.py`: physically impossible signal checks

Deliver notebooks:

- `07_multistatic_node_comparison.ipynb`
- additional focused notebooks as needed

## Milestone 9: RuVector Equivalents In Python

Reference source:

- `v2/crates/wifi-densepose-ruvector`
- published RuVector behavior from the Rust reference

Python equivalents:

- NetworkX for graph/min-cut style experiments
- SciPy sparse solvers for interpolation and inverse problems
- PyTorch attention for attention-weighted fusion
- NumPy ring buffers and compressed temporal tensors

Port behavior, not dependency internals.

Deliverables:

- subcarrier partitioning
- attention-gated spectrogram
- BVP aggregation
- Fresnel geometry solver
- TDoA triangulation
- compressed breathing and heartbeat histories

## Milestone 10: Neural And Training Research

Reference source:

- `v2/crates/wifi-densepose-nn`
- `v2/crates/wifi-densepose-train`
- sensing-server training helpers

Port:

- PyTorch tensor pipeline
- DensePose-style head
- CSI to pose transformer
- RF encoder
- contrastive embeddings
- projection head
- dataset loaders
- checkpoints
- simple RVF-compatible or JSON model export, if useful

Deliver notebooks:

- `08_csi_to_pose_experiment.ipynb`
- `09_dataset_replay_lab.ipynb`
- `10_model_embedding_visualization.ipynb`

Visualizations:

- embedding PCA
- embedding UMAP or t-SNE
- empty vs present clusters
- motion vs still clusters
- per-room fingerprint clusters
- training loss curves
- keypoint confidence plots

## Milestone 11: MAT Research Pipeline

Reference source:

- `v2/crates/wifi-densepose-mat`

Port:

- disaster event models
- scan zones
- survivor model
- vital-sign survivor detection
- localization
- triangulation
- tracking
- triage scoring
- alert objects

Research-only boundary:

- Keep alert dispatch local and test-oriented.
- Do not build production emergency notification integrations.

## Milestone 12: WorldGraph, Trust, And Privacy Research

Reference source:

- `v2/crates/wifi-densepose-engine`
- `v2/crates/wifi-densepose-worldgraph`
- `v2/crates/wifi-densepose-bfld`

Port:

- WorldGraph nodes and edges
- semantic state records
- provenance
- privacy class
- privacy mode registry
- BFLD frame metadata
- identity risk scoring
- witness hash
- privacy demotion logic

Deliverables:

- graph serialization
- local graph visualizations
- provenance inspection notebook section

## Milestone 13: Optional Later Tracks

Only after the core sensing and notebook workflow is useful:

- HOMECORE research subset
- nvsim Python simulator
- ruview-swarm research models
- browser visualization helpers
- desktop hardware tooling equivalent

These are optional because they are less central to the pure Python signal
research goal.

## Notebook Experiment Requirements

Each notebook should include:

- short purpose statement
- fixture or simulated data source
- one-click run path
- plots with labeled axes
- expected interpretation
- notes about limitations

Recommended plotting stack:

- matplotlib for simple static plots
- seaborn for distributions
- plotly for interactive heatmaps and 3D
- ipywidgets for threshold sliders

## Initial Notebook Details

### `00_signal_playground.ipynb`

Purpose:

Explore synthetic and recorded CSI frames.

Plots:

- amplitude vector
- phase vector
- I/Q scatter
- subcarrier index vs amplitude
- subcarrier index vs phase

### `01_empty_room_vs_person_present.ipynb`

Purpose:

Compare empty-room CSI with person-present CSI.

Plots:

- raw amplitude overlay
- normalized amplitude overlay
- phase overlay
- variance by subcarrier
- motion energy timeline
- spectrogram comparison

### `02_motion_vs_stillness.ipynb`

Purpose:

Show how walking differs from still presence.

Plots:

- motion score over time
- threshold bands
- still vs moving variance
- false-positive examples

### `03_breathing_and_heart_rate_bands.ipynb`

Purpose:

Inspect vital-sign frequency bands.

Plots:

- residual waveform
- bandpass filtered breathing signal
- bandpass filtered heart signal
- PSD
- BPM estimate over time
- confidence score over time

### `05_subcarrier_heatmaps.ipynb`

Purpose:

Make occupancy effects visually obvious across subcarriers.

Plots:

- time x subcarrier amplitude heatmap
- time x subcarrier phase heatmap
- time x subcarrier z-score heatmap
- empty vs present difference heatmap

## Definition Of Done For A Ported Capability

A capability is considered ported when:

- Python API exists under `src/ruview/`
- it accepts documented inputs
- it has at least one unit test
- it has a synthetic or captured fixture
- it has notebook coverage if the output is visual or signal-related
- intentional differences from the reference repo are documented

## First Concrete Target

Build Milestone 1 through Milestone 3 first:

```text
core types
ESP32 protocol parsers
synthetic CSI generator
basic feature extraction
empty vs present visualization notebook
subcarrier heatmap notebook
```

That creates the minimum useful research loop:

```text
CSI packet or simulation
  -> Python parser
  -> amplitude and phase arrays
  -> feature extraction
  -> visual notebook comparison
```

