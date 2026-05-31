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
commit: worker commit `Add calibration baseline primitives`
sync_status: draft
source_format: node-specific
tags: [phd, research, ruview, wifi-densepose, python, ruvsense, calibration]
---

# Milestone 6 Calibration Core

## Work Done

- Ported Milestone 6 calibration primitives into `src/ruview/ruvsense/calibration.py`.
- Added `PhyTier`, `CalibrationConfig`, `WelfordStats`, finalized per-subcarrier baseline stats, `CalibrationRecorder`, `BaselineCalibration`, deviation scoring, drift decision, baseline subtraction, and JSON save/load helpers.
- Exported the calibration API from `src/ruview/ruvsense/__init__.py`.
- Added unit tests in `tests/unit/test_calibration.py` for Welford variance, circular phase wraparound, insufficient frames, finalization, JSON roundtrip, deviation scoring, and non-mutating baseline subtraction.

## Source Reference

- Reference repo: `/Users/sushrutpatwardhan/1Projects/RuView`
- Rust sources read: `v2/crates/wifi-densepose-signal/src/ruvsense/calibration.rs` and `field_model.rs`.

## Verification

- Command/config: `uv run pytest -q tests/unit/test_calibration.py`
- Result: `7 passed`
- Command/config: `uv run pytest -q`
- Result: `59 passed`

## Notes

- JSON persistence is intentionally a Python research format with magic/version metadata; it is not the Rust ADR-135 binary ABI yet.
- Baseline subtraction returns a new array and does not mutate the input.
- Deviation thresholds follow the Rust heuristic shape: motion around amplitude median z-score `> 2` or phase drift `> pi/6`; drift thresholds are configurable and higher by default.

## Next

- Coordinate with the notebook/docs Milestone 6 worker for visual calibration drift notebook coverage without editing their files.
