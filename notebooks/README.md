# RuView Notebook Scope

This folder is now organized around the active sensing questions rather than every historical milestone.

## Core Scope

| Scope | Primary notebook | Supporting notebooks |
|---|---|---|
| End-to-end demo | `00_end_to_end_capture_visualization_prediction.ipynb` | `07_recording_presence_breathing_triage.ipynb` |
| Signal visualization | `01_signal_amplitude_phase_visualization.ipynb` | `02_signal_subcarrier_heatmaps.ipynb` |
| Person detection / presence | `03_person_presence_detection.ipynb` | `07_recording_presence_breathing_triage.ipynb` |
| Motion detection | `04_motion_detection.ipynb` | `01_signal_amplitude_phase_visualization.ipynb`, `02_signal_subcarrier_heatmaps.ipynb` |
| Breathing rate | `05_breathing_heart_rate_detection.ipynb` | `07_recording_presence_breathing_triage.ipynb` |
| Heart rate | `05_breathing_heart_rate_detection.ipynb` | `02_signal_subcarrier_heatmaps.ipynb` |
| Pose estimation | `06_pose_estimation.ipynb` |  |
| Position detection / triangulation | `08_position_triangulation_three_receivers.ipynb` |  |

## Local Recording Triage

Use `07_recording_presence_breathing_triage.ipynb` for quick checks against local files under `data/recordings/`.
Use `00_end_to_end_capture_visualization_prediction.ipynb` when you want a single demo flow that covers capture planning, loading a local recording, signal visualization, and prediction summaries.

## Recording-Backed Charts

Every notebook now starts with a recording source switch. Set `USE_RECORDING = True` and either leave `RECORDING_CSV = None` to auto-pick the first `*_csi.csv` under `data/recordings/`, or point `RECORDING_CSV` at a specific capture file.

The scoped notebooks keep synthetic fixtures as the default so they run without hardware data. The end-to-end and recording-triage notebooks are recording-backed by default. The triangulation notebook needs three receiver recordings in `RECEIVER_RECORDINGS`; one receiver recording can visualize signal/range behavior, but it cannot produce a 2D person location by triangulation.

## Pruned Backlog

Older milestone and optional-track notebooks were removed from this directory to keep the working surface focused. The reusable implementation still lives in `src/ruview/`; new exploratory notebooks should be added only when they fit the scope table above.
