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
commit: worker commit `Add CIR and coherence primitives`
sync_status: draft
source_format: node-specific
tags: [phd, research, ruview, wifi-densepose, python, ruvsense, cir, coherence]
---

# Milestone 8 CIR and Coherence Core

## Work Done

- Added NumPy-only CIR helpers in `src/ruview/ruvsense/cir.py`.
- Added `CirConfig`, `CirTap`, `CirEstimate`, active-subcarrier indexing, CSI/reference subtraction, oversampled IFFT conversion, sparse tap thresholding, tap selection with local separation, RMS delay spread, and phase variance diagnostics.
- Added coherence scoring in `src/ruview/ruvsense/coherence.py` using normalized complex correlation, magnitude similarity, phase stability, combined score, consecutive-window scoring, and a small EMA tracker.
- Added coherence gate decisions in `src/ruview/ruvsense/coherence_gate.py` with accept, predict-only, reject, and recalibrate actions based on coherence, novelty/deviation, calibration drift, and stale-frame count.
- Added focused tests in `tests/unit/test_ruvsense_cir_coherence.py` covering synthetic single/multiple taps, weak ghost/noise rejection, high-vs-low coherence ordering, zero-input stability, consecutive scoring, and gate decisions.

## Source Reference

- Reference repo: `/Users/sushrutpatwardhan/1Projects/RuView`
- Rust sources read:
  - `v2/crates/wifi-densepose-signal/src/ruvsense/cir.rs`
  - `v2/crates/wifi-densepose-signal/src/ruvsense/coherence.rs`
  - `v2/crates/wifi-densepose-signal/src/ruvsense/coherence_gate.rs`
  - CIR tests under `v2/crates/wifi-densepose-signal/tests/cir_*.rs`

## Verification

- Command/config: `uv run pytest -q tests/unit/test_ruvsense_cir_coherence.py`
- Result: `10 passed`
- Command/config: `uv run pytest -q`
- Result: `92 passed, 1 warning`; warning is existing Starlette/FastAPI TestClient deprecation noise.

## Notes

- CIR estimation intentionally uses active-subcarrier placement plus oversampled IFFT rather than porting the Rust ISTA/Neumann sparse solver line-for-line.
- Sparse tap selection uses magnitude thresholds, optional top-k, robust noise floor, and local bin separation to avoid selecting adjacent sidelobes as independent physical taps.
- Coherence scores are clipped to `[0, 1]` and handle all-zero inputs deterministically.
- `src/ruview/ruvsense/__init__.py` was left untouched for parent export reconciliation.

## Next

- Parent Milestone 8 reconciliation can export these modules and coordinate with the other RuvSense workers.
