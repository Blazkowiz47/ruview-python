# 2026-05-31 sushruts-macbook-pro M11 localization/tracking

- Node: sushruts-macbook-pro
- Repo: /Users/sushrutpatwardhan/1Projects/ruview-python
- Branch: master
- Scope: MAT localization and tracking primitives in owned files only.
- Started port from Rust MAT references for triangulation, range constraints, position fusion, depth estimation, Kalman tracking, lifecycle, nearest-neighbor association, and CSI fingerprint matching.
- Implemented `src/ruview/mat/localization.py` and `src/ruview/mat/tracking.py` with focused unit coverage in `tests/unit/test_mat_localization_tracking.py`.
- Verification: `uv run pytest -q tests/unit/test_mat_localization_tracking.py` passes (`8 passed`).
- Full suite attempt: `uv run pytest -q` currently stops during collection in non-owned `src/ruview/mat/domain.py` because `_now_utc` is referenced before definition by `ScanZone`.
- Next action: parent/owner should reconcile MAT exports and the unrelated domain/detection collection failure before using full-suite status for Milestone 11.
