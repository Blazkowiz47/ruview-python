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
sync_status: draft
source_format: node-specific
tags: [phd, research, ruview, wifi-densepose, python, training]
---

# 2026-05-31 - Milestone 10 Training Utilities

## Work Done

- Started the Milestone 10 training-research slice for losses, metrics, checkpoint manifests, export manifests, and a tiny deterministic trainer.
- Added NumPy-first loss utilities with optional torch tensor support when torch tensors are supplied.
- Added PCK/OKS metrics, heatmap-to-keypoint conversion, checkpoint top-k selection, BLAKE3 model export provenance, and deterministic TrainingRun history.
- Added focused unit tests for heatmaps/loss masking, PCK/OKS ordering, checkpoint ranking, export manifests, and trainer history.

## References

- `RuView/v2/crates/wifi-densepose-train/src/losses.rs`
- `RuView/v2/crates/wifi-densepose-train/src/metrics.rs`
- `RuView/v2/crates/wifi-densepose-train/src/trainer.rs`
- `RuView/v2/crates/wifi-densepose-nn/src/onnx.rs`

## Next

- Commit only the owned training utility files after verification.

## Verification

- Command/config: `uv run pytest -q tests/unit/test_training_utils.py`
- Result: `5 passed in 0.19s`.
- Command/config: `uv run pytest -q`
- Result: `137 passed, 5 skipped, 1 existing FastAPI/Starlette warning in 1.03s`.
