# 2026-05-31 - sushruts-macbook-pro - Milestone 6 Calibration Notebook

## Context

- Node: sushruts-macbook-pro
- Device/server: local macOS workspace
- Repo path: `/Users/sushrutpatwardhan/1Projects/ruview-python`
- Branch: `master`
- Scope: notebook/docs worker only; avoided `src/ruview/ruvsense/*` while a separate worker owns core calibration modules.

## Log

- Replaced `notebooks/06_calibration_baseline_drift.ipynb` placeholder with an unexecuted deterministic visual lab for ADR-135-style empty-room baseline calibration.
- Added guarded `ruview.ruvsense.calibration` import probe plus local NumPy fallback helpers for Welford-like amplitude statistics, circular phase means, per-frame deviation scoring, drift score, and amplitude baseline subtraction.
- Synthetic fixture: 52 active HT20-style subcarriers, 600 empty-room calibration frames, gradual post-calibration drift, and a localized person/motion perturbation event.
- Added required plots: baseline amplitude distribution, deviation score over time with drift/event markers, before/after calibration comparison, and per-subcarrier z-score heatmap.
- Added `docs/porting/calibration-baseline-drift.md` with Rust/ADR source references, Python notebook intent, intentional deviations from binary ABI/SVD field model, and run path.
- Extended notebook JSON scaffold coverage to include the Milestone 6 notebook.

## Verification

- `uv run python -m json.tool notebooks/06_calibration_baseline_drift.ipynb` passed.
- Notebook code-cell smoke check under `MPLBACKEND=Agg` executed 7 code cells; drift marker appeared at about 39.85 s.
- `uv run pytest -q` passed: 52 tests.

## Next

- Reconcile notebook guarded imports with the final `ruview.ruvsense.calibration` Python API after the core calibration worker lands.
