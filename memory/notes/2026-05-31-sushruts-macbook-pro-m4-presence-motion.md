# 2026-05-31 - sushruts-macbook-pro-m4 - presence and motion

## Session

- Node: sushruts-macbook-pro-m4
- Device/server: local macOS workspace
- Repo path: `/Users/sushrutpatwardhan/1Projects/ruview-python`
- Branch/base: `master`, Milestone 3 completion commit `6956689`
- Worker: G, Milestone 4 presence/motion slice

## Source mappings

- `wifi-densepose-signal/src/motion.rs`: weighted motion score components, baseline variance calibration, smoothing/debounce intent, adaptive threshold shape.
- `wifi-densepose-sensing-server/src/csi.rs`: CSI amplitude/phase and presence/motion packet context.
- Python Milestone 3: `CsiWindow`, scalar signal features, subcarrier variance, phase unwrap, and synthetic CSI scenarios.

## Work log

- Added `src/ruview/signal/baseline.py` with `BaselineStats`, `RollingBaseline`, adaptive thresholding, relative-increase normalization, and `DetectionDebouncer`.
- Added `src/ruview/signal/motion.py` with `MotionScore`, `MotionWeights`, and motion scoring from temporal amplitude deltas, amplitude variance delta from baseline, phase variance, and subcarrier variance.
- Added `src/ruview/signal/presence.py` with `PresenceResult` and empty/still/moving classification heuristics using baseline mean-amplitude shift plus motion components.
- Exported the new APIs from `ruview.signal`.
- Added `tests/unit/test_presence_motion.py` using `SyntheticCsiConfig` and `generate_synthetic_sequence` for empty, person-present, stillness, and walking behavior.

## Command/config

- Focused verification: `uv run pytest -q tests/unit/test_presence_motion.py`
- Full verification: `uv run pytest -q`

## Result

- Focused tests passed: 4 tests.
- Full suite passed: 41 tests.
- Synthetic behavior verified: empty scores below person-present, person-present remains still, stillness remains present/still, and walking crosses the moving threshold.

## Intentional limitations

- Classifiers are deterministic heuristics for the Python port baseline; they are not trained models.
- Baseline updates are opt-in during classification (`update_baseline_when_empty`) to avoid silently learning occupied-room windows.

## Next action

- Downstream Milestone 4 workers can integrate these APIs into streaming/server surfaces and tune thresholds against real CSI captures.
