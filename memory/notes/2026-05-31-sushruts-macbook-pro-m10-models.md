---
date: 2026-05-31
work_date: 2026-05-31
project: ruview-python
node: sushruts-macbook-pro
node_type: laptop
device: Sushrut's MacBook Pro
timezone: Europe/Oslo
repo_path: /Users/sushrutpatwardhan/1Projects/ruview-python
branch: master
commit: pending
sync_status: draft
source_format: node-specific
tags: [milestone-10, neural, pytorch, densepose, contrastive, csi-to-pose]
---

# 2026-05-31 Milestone 10 Neural Model Modules

## Work Done

- Added optional-PyTorch neural research modules under `src/ruview/nn/` without editing public exports.
- Ported RF embedding concepts from the Rust references into `encoders.py`: amplitude/phase CSI validation, phase sanitization, compact Conv2D encoder, optional transformer token mixer, L2 embeddings, cosine logits, calibration robustness loss, triplet loss, and deterministic contrastive triplet sampling.
- Added `heads.py` with a projection head, DensePose-style map head, and lightweight uncertainty-aware multi-task heads for pose, presence, count, activity, vitals, gait, and identity embedding.
- Added `transformer.py` with a CSI-to-pose transformer encoder over antenna-path tokens and an end-to-end pose/DensePose output wrapper.
- Added focused tests in `tests/unit/test_nn_models.py`; torch-dependent tests use `pytest.importorskip("torch")` and NumPy helper tests run in the default environment.

## Verification

- Command/config: `uv run pytest -q tests/unit/test_nn_models.py`
- Dataset/input: tiny synthetic CSI amplitude/phase tensors and deterministic NumPy embeddings.
- Result: `2 passed, 4 skipped`; skipped tests require torch, which is not installed in the default uv environment.
- Command/config: `uv run pytest -q`
- Result: `126 passed, 4 skipped, 1 warning`; warning is the existing FastAPI/Starlette TestClient deprecation.

## Notes

- The base package remains lightweight: importing these modules does not require torch, while constructing PyTorch classes raises a clear optional-dependency error if torch is missing.
- Parent integration still needs to decide public exports from `ruview.nn.__init__` and reconcile milestone-level memory.
