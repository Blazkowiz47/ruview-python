---
date: 2026-05-31
work_date: 2026-05-31
project: ruview-python
node: sushruts-macbook-pro
node_type: laptop
device: Sushrut's MacBook Pro
timezone: Europe/Oslo
repo_path: ruview-python
branch: master
commit:
sync_status: draft
source_format: node-specific
tags: [phd, research, ruview, wifi-densepose, milestone-10, neural, training, dataset]
---

# 2026-05-31 Milestone 10 Dataset Pipeline

## Intent

- Port the Rust neural tensor, training config, synthetic dataset, replay helpers, and deterministic batching behavior into owned Python files.

## Work Done

- Added a NumPy-only tensor utility module with `TensorShape`, `TensorSpec`, broadcasting validation, `Tensor`, stats, activations, stacking/splitting, and optional `to_torch`/`from_torch` hooks.
- Added `TrainingConfig` defaults mirroring the Rust training config spirit: 56 model subcarriers, 114 native subcarriers, 3x3 antennas, 100-frame windows, 17 keypoints, 24 parts, optimization/checkpoint/log settings, validation, presets, and JSON roundtrip helpers.
- Added `CsiSample`, deterministic `SyntheticCsiDataset`, server-style `ReplayCsiDataset`/`JsonlCsiDataset`, subcarrier interpolation, deterministic Xorshift dataloader shuffle, sample-list batches, stacked `CsiBatch` batches, and optional torch conversion hooks.
- Added focused unit coverage in `tests/unit/test_training_dataset.py`.

## Verification

- Command/config: `uv run pytest -q tests/unit/test_training_dataset.py`
- Dataset: deterministic synthetic CSI samples and hand-written replay JSONL rows.
- Output path: `tests/unit/test_training_dataset.py`
- Result: passed (`6 passed, 1 skipped`); skip is the optional torch conversion hook when PyTorch is absent.
- Command/config: `uv run pytest -q`
- Dataset: full unit/parity suite plus the new synthetic/replay dataset tests.
- Output path: full test suite
- Result: passed (`132 passed, 5 skipped, 1 warning`); warning is the existing FastAPI/Starlette TestClient deprecation.

## Next

- Parent worker should reconcile package exports and continue Milestone 10 model/head/checkpoint work.
