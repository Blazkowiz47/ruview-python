---
date: 2026-05-31
work_date: 2026-05-31
project: ruview-python
node: sushruts-macbook-pro
repo_path: /Users/sushrutpatwardhan/1Projects/ruview-python
branch: master
source_format: node-specific
tags: [milestone-11, mat, domain, detection, numpy]
---

# 2026-05-31 MAT Domain/Detection

## Work Done

- Added `src/ruview/mat/domain.py` with MAT research dataclasses/enums for coordinates, uncertainty, debris profiles, scan zones, survivor vital histories, survivor status/condition, vital sign readings, and the `DisasterEvent` aggregate.
- Added `src/ruview/mat/detection.py` with deterministic NumPy FFT/variance primitives for breathing, heartbeat, movement, ensemble confidence, and a synchronous detection pipeline.
- Added `tests/unit/test_mat_domain_detection.py` covering zone containment/progress, survivor vital updates, event aggregation, deterministic sine-wave breathing/heartbeat detection, movement classification, ensemble confidence, and pipeline output.

## Verification

- `uv run pytest -q tests/unit/test_mat_domain_detection.py` -> `8 passed`.
- `uv run pytest -q` -> `164 passed, 5 skipped, 1 existing Starlette deprecation warning`.

## Notes

- Ported behavior from Rust MAT references without production API internals or async/ML surfaces.
- Kept detector behavior deterministic and NumPy-only; thresholds remain research defaults pending real CSI calibration.
- Parallel untracked localization/tracking worker files were present and left untouched.
