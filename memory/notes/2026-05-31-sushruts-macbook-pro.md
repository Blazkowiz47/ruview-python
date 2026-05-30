---
date: 2026-05-31
work_date: 2026-05-31
project: ruview-python
node: sushruts-macbook-pro
node_type: laptop
device: Sushrut's MacBook Pro
server:
timezone: Europe/Oslo
repo_path: /Users/sushrutpatwardhan/1Projects/ruview-python
branch:
commit:
sync_status: draft
source_format: node-specific
tags: [phd, research, ruview, wifi-densepose, python, csi, signal-processing]
---

# 2026-05-31

## Intent

- Initialize durable project memory for the `ruview-python` research port.
- Capture the current port plan and next recovery actions.

## Work Done

- Created the `memory/` structure for this repo and linked it to the main knowledge-base workstream.
- Created project `AGENTS.md` containing the Sushrut memory block.
- Created project-local portable memory command specs under `memory/commands/`.
- Captured the current repo state as not a Git repository; the only discovered project file is `plan.md`.
- Read `plan.md` and checked the Rust reference core crate at `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-core`.
- Created Milestone 0 scaffold: `pyproject.toml`, `src/ruview/`, `tests/`, `notebooks/`, `examples/`, and `data/` fixture folders.
- Added initial Milestone 1 core-contract port: `ComplexSample`, `FrameId`, `DeviceId`, `Timestamp`, `Confidence`, `FrequencyBand`, `AntennaConfig`, `CsiMetadata`, `CsiFrame`, `Keypoint`, `PersonPose`, `PoseEstimate`, canonical bytes, and BLAKE3 witness hashing.
- Added unit/parity tests, a fixed ADR-136 witness vector, and `examples/simulate_empty_vs_present.py`.
- Added `.gitignore` and a local `.venv` for verification because Homebrew Python is externally managed.
- Removed the pip-created `.venv`, recreated it with `uv sync --extra dev`, generated `uv.lock`, and updated `README.md` quick-check commands to use `uv`.
- Expanded `README.md` to describe the Python research port, project goals, layout, current status, and acknowledgements to upstream `ruvnet/RuView` / rUv with MIT license notice.
- After user ran `git init`, expanded `.gitignore` for Python caches, local virtualenvs, `uv` scratch, notebook checkpoints, editor files, logs, and local large recordings while keeping `data/recordings/.gitkeep` trackable.
- Set repo-local `user.name` to `Sushrut Patwardhan` so the first commit can use the existing configured email.
- Completed Milestone 4 via two clean worker commits:
  - `600cd74` adds `src/ruview/signal/baseline.py`, `motion.py`, `presence.py`, exports, and presence/motion tests.
  - `589a1eb` updates `notebooks/02_motion_vs_stillness.ipynb`, adds `docs/porting/presence-motion.md`, notebook JSON coverage, and a worker memory note.

## Experiments / Runs

- Command/config: `.venv/bin/python -m pytest -q`
- Dataset: synthetic in-test CSI frame fixtures
- Output path: `tests/unit/test_core_contracts.py`, `tests/parity/test_adr136_canonical.py`
- Result: `9 passed`; fixed ADR-136 Python witness vector is `9ac4fdb9b6b9b2cca62b6ac6751a50400d49146d16435a7d3aed8350e01d983e`.
- Command/config: `.venv/bin/python examples/simulate_empty_vs_present.py`
- Dataset: synthetic 3-stream, 56-subcarrier CSI frames
- Output path: stdout
- Result: empty mean amplitude `1.008`, present mean amplitude `1.358`, example witness `d38bfb2309a59a0378769bf4aa8ea85093eae7187a355004718269b2824c543d`.
- Command/config: `uv sync --extra dev`; `uv run python --version && uv run pytest -q`
- Dataset: synthetic in-test CSI frame fixtures
- Output path: `.venv/`, `uv.lock`
- Result: `uv 0.10.4` created a CPython `3.12.12` environment and tests passed (`9 passed`).
- Command/config: `uv run pytest -q`
- Dataset: synthetic in-test CSI frame fixtures
- Output path: README-only doc change verification
- Result: tests still pass (`9 passed`) after README acknowledgement update.
- Command/config: `uv run pytest -q`
- Dataset: synthetic in-test CSI frame fixtures
- Output path: pre-commit verification
- Result: tests still pass (`9 passed`) after `.gitignore` updates.
- Command/config: `uv run pytest -q`
- Dataset: deterministic synthetic CSI empty-room, person-present, stillness, and walking windows
- Output path: Milestone 4 classifier and notebook verification
- Result: tests pass (`41 passed`) after classifier and motion notebook commits.
- Next action: Start Milestone 5 vitals with breathing/heart-rate band estimators and notebook `03_breathing_and_heart_rate_bands.ipynb`.

## Analysis Results

- `plan.md` frames this as a pure Python research port of RuView / WiFi-DensePose from `/Users/sushrutpatwardhan/1Projects/RuView`.
- The port should prioritize readable implementations, fixtures, tests, and notebooks over product packaging or commercial polish.
- Rust `wifi-densepose-core` canonical frame layout uses UUID bytes, fixed little-endian metadata fields, length-prefixed UTF-8 device id, 16 zero bytes for missing calibration id, `(nrows, ncols)` as `u32`, and stream-major complex samples as `f64 re || f64 im`.
- Python `CsiFrame` deep-copies metadata on construction to better match Rust ownership/move behavior and prevent accidental witness-hash changes from later external metadata mutation.
- Milestone 4 intentionally ports the Rust motion detector conceptually: weighted variance, temporal delta, phase variance, and subcarrier variance components with baseline-relative thresholds, plus debounce for human-readable empty/still/moving states.

## Learnings

- Memory should track capability-level port progress, source-reference mappings, intentional deviations from Rust/C behavior, and parity-test outcomes.
- Keep core install light (`numpy`, `blake3`) and put heavy research dependencies behind optional extras so parity tests stay fast.
- Use `uv sync --extra dev` and `uv run pytest -q` as the default local workflow.
- Synthetic presence/motion thresholds are useful for visual lab progress, but real ESP32 captures are still needed before treating scores as calibrated.

## Decisions

- Use in-repo memory as the operational record and the main knowledge-base workstream as the compact cross-project summary.

## Blockers

- No current blocker for Milestone 5; vitals modules need to be ported from the Rust reference.

## Next

- Start Milestone 5 by mapping `v2/crates/wifi-densepose-vitals` and `sensing-server/src/vital_signs.rs` into Python vital preprocessing, bandpass/PSD estimators, confidence labels, tests, docs, and `03_breathing_and_heart_rate_bands.ipynb`.
