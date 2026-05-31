# 2026-05-31 sushruts-macbook-pro M9 signal

- Node: sushruts-macbook-pro
- Repo: `/Users/sushrutpatwardhan/1Projects/ruview-python`
- Branch: `master`
- Scope: Milestone 9 RuVector signal equivalents; owned files only.

## Session log

- Added `ruview.ruvector.subcarrier` with deterministic NumPy subcarrier partitioning, largest-gap graph-cut behavior, higher-mean sensitive labeling, and finite importance weights for empty, single, clustered, and all-equal sensitivity inputs.
- Added `ruview.ruvector.spectrogram` with flat Rust-layout and 2D spectrogram gating, lambda thresholding, and tau temporal support that suppresses low-energy frames while preserving or amplifying motion frames.
- Added `ruview.ruvector.bvp` with scaled dot-product BVP aggregation seeded by sensitivity priors, attention-weight inspection, requested velocity-bin output lengths, and zero output for empty input.
- Added `tests/unit/test_ruvector_signal.py` covering partition coverage/order, edge-case weights, spectrogram shape/layout behavior, motion-frame gating, and BVP attention ordering.

## Verification

- Passed: `uv run pytest -q tests/unit/test_ruvector_signal.py` (9 tests)
- Passed: `uv run pytest -q` (116 tests, 1 existing FastAPI/Starlette deprecation warning)
