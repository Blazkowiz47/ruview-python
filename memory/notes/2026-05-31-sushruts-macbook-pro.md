---
date: 2026-05-31
work_date: 2026-05-31
project: ruview-python
node: sushruts-macbook-pro
node_type: laptop
device: Sushrut's MacBook Pro
server:
timezone: Europe/Oslo
repo_path: ruview-python
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
- Read `plan.md` and checked the Rust reference core crate at `RuView/v2/crates/wifi-densepose-core`.
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
- Completed Milestone 5 via two clean worker commits:
  - `cd5f90a` adds `src/ruview/vitals/` preprocessing, breathing, heart-rate, quality, smoothing modules, exports, and vitals tests.
  - `6750d38` updates `notebooks/03_breathing_and_heart_rate_bands.ipynb`, adds `docs/porting/vitals.md`, notebook JSON coverage, and a worker memory note.
- Completed Milestone 6 via two clean worker commits:
  - `1b34432` updates `notebooks/06_calibration_baseline_drift.ipynb`, adds `docs/porting/calibration-baseline-drift.md`, notebook JSON coverage, and a worker memory note.
  - `6f72a73` adds `src/ruview/ruvsense/calibration.py`, RuvSense exports, calibration tests, and a worker memory note.
- Completed Milestone 7 in three commits:
  - `97c2f5d` replaces the replay example, adds sensing-server porting docs, and records an example worker note.
  - `0fb072b` adds sensing update schemas, simulated/replay/UDP sources, latest-state buffer, exports, and source tests.
  - `59841e1` adds the FastAPI app factory, REST/latest/vitals endpoints, WebSocket stream, uvicorn research extra, app tests, and run-path docs.
- Completed Milestone 8 in worker commits and a parent export integration:
  - `69bfbb1` adds multiband CSI fusion, LO phase alignment, multistatic attention fusion, tests, and a worker memory note.
  - `adaeaee` adds CIR sparse-tap estimation, coherence scoring, coherence gate decisions, tests, and a worker memory note.
  - `eae6b19` updates `notebooks/07_multistatic_node_comparison.ipynb`, adds `docs/porting/ruvsense-advanced.md`, notebook JSON coverage, and a worker memory note.
  - `785d47f` adds field model, pose tracker, tomography primitives, tests, and a worker memory note.
  - `101c15e` adds gesture, intention, cross-room, longitudinal, adversarial primitives, tests, and a worker memory note.
  - Parent integration exports Milestone 8 public APIs through `ruview.ruvsense` and adds a public-export smoke test.
- Completed Milestone 9 in worker commits and a parent export integration:
  - `a3041f9` adds RuVector Fresnel geometry, TDoA triangulation, GDI/effective-viewpoint, CRB/GDOP helpers, tests, and a worker memory note.
  - `42ff578` adds RuVector subcarrier partitioning, attention-gated spectrograms, BVP aggregation, tests, and a worker memory note.
  - `ce289e9` adds `docs/porting/ruvector.md`, `notebooks/11_ruvector_signal_geometry.ipynb`, notebook JSON coverage, and a worker memory note.
  - `886376a` adds compressed breathing and heartbeat histories, tests, and a worker memory note.
  - Parent integration exports Milestone 9 public APIs through `ruview.ruvector` and adds a public-export smoke test.
- Completed Milestone 10 in worker commits and a parent export integration:
  - `9a4e4bb` adds optional PyTorch RF encoder, projection heads, DensePose-style head, CSI-to-pose transformer, NumPy contrastive helpers, tests, and a worker memory note.
  - `72b7ad0` updates notebooks `08`-`10`, adds `docs/porting/neural-training.md`, notebook JSON coverage, and a worker memory note.
  - `34ceb63` adds NumPy tensor helpers, training config, synthetic/replay datasets, dataloader, tests, and a worker memory note.
  - `3d0e5ab` adds losses, metrics, checkpoint manifests, model export manifests, tiny trainer, tests, and a worker memory note.
  - Parent integration exports public APIs through `ruview.nn` and `ruview.training` and adds a public-export smoke test.
- Completed Milestone 11 in worker commits and a parent export integration:
  - `9e7c901` adds `docs/porting/mat.md`, `notebooks/12_mat_research_pipeline.ipynb`, notebook JSON coverage, and a worker memory note.
  - `2195498` adds START-style triage scoring, local-only alert payloads/dispatcher lifecycle, tests, and a worker memory note.
  - `c929e4d` adds MAT range localization, depth/fusion helpers, Kalman survivor tracking, fingerprints, tests, and a worker memory note.
  - `698e5fe` adds MAT disaster/domain/survivor/vital models, breathing/heartbeat/movement/ensemble detection, tests, and a worker memory note.
  - Parent integration exports public APIs through `ruview.mat` with explicit domain/localization aliases and adds a public-export smoke test.
- Completed Milestone 12 in worker commits and a parent export/trust integration:
  - `851cc02` adds `docs/porting/worldgraph-privacy.md`, `notebooks/13_worldgraph_privacy_provenance.ipynb`, notebook JSON coverage, and a worker memory note.
  - `05b5d9b` adds WorldGraph dataclass nodes/edges, graph snapshot/query/provenance/rollup helpers, tests, and a worker memory note.
  - `65922dd` adds BFLD header/payload/CRC primitives, privacy modes, attestation chain, identity-risk/signature helpers, privacy demotion, tests, and a worker memory note.
  - Parent integration exports public APIs through `ruview.worldgraph` and `ruview.privacy`, adds trust-throughline witness helpers, and adds a public composition smoke test.
- Completed Milestone 13 optional later tracks in worker commits and a parent export integration:
  - `3fc81b9` adds the deterministic `ruview.nvsim` magnetic scene, propagation, canonical frame, pipeline, tests, and worker memory note.
  - `35640e9` adds browser visualization payload/fusion helpers, desktop hardware-planning helpers, tests, and worker memory note.
  - `de0b808` adds `docs/porting/optional-tracks.md`, `notebooks/14_optional_tracks_research_overview.ipynb`, notebook JSON coverage, and worker memory note.
  - `b5aeb1d` adds HOMECORE state and automation research primitives, tests, and worker memory note.
  - `a4bbd5e` adds swarm topology, formation, planning, sensing/fusion research primitives, tests, and worker memory note.
  - Parent integration exports desktop hardware helpers through `ruview.hardware` and adds `tests/unit/test_optional_exports.py` as a cross-package smoke test.
- Cleaned Markdown documentation and memory notes to remove local `1Projects` absolute-path prefixes while preserving useful repo-relative references.
- Filled `notebooks/04_phase_and_amplitude_visualization.ipynb` with a deterministic synthetic CSI phase/amplitude fixture, amplitude and phase plots, expected interpretation, and limitations. Added notebook `04` to the visual-lab scaffolding test.
- Stripped saved execution counts and outputs from notebooks `00` through `14`; notebooks now carry runnable dummy/synthetic data in code cells without committed output blobs.
- Updated `README.md` so it describes the completed Milestone 0-13 port, expanded implemented module layout, `uv` checks, and the synthetic/dummy fixture status of the notebooks.
- Replaced the completed porting `plan.md` with a new Android low-level CSI driver feasibility plan covering approval/safety, chipset recon, Android API limits, root recon, kernel/user export paths, firmware patch feasibility, controlled collection, and decision gates.

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
- Command/config: `uv run pytest -q`
- Dataset: deterministic sine residuals and synthetic vital notebook fixtures
- Output path: Milestone 5 vitals API and notebook verification
- Result: tests pass (`52 passed`); unit tests estimate breathing near 18 BPM and heart near 72 BPM as valid.
- Command/config: `uv run pytest -q`; `MPLBACKEND=Agg uv run --extra research python <notebook smoke>`
- Dataset: deterministic empty-room, drift, and localized person/event calibration fixtures
- Output path: Milestone 6 calibration API and notebook verification
- Result: tests pass (`59 passed`); calibration notebook smoke executed 7 code cells with only noninteractive matplotlib warnings.
- Command/config: `uv run pytest -q`; `uv run --extra research python -m ruview.server.app --help`; `uv run python examples/replay_recording.py <fixture> --limit 2`
- Dataset: deterministic synthetic CSI sources and a hand-written JSONL replay fixture
- Output path: Milestone 7 server schemas/sources/app/examples verification
- Result: tests pass (`70 passed`); app help prints cleanly; replay fixture printed two normalized updates.
- Command/config: `uv run pytest -q tests/unit/test_ruvsense_exports.py tests/unit/test_ruvsense_cir_coherence.py tests/unit/test_ruvsense_fusion.py tests/unit/test_ruvsense_field_pose.py tests/unit/test_ruvsense_temporal.py`
- Dataset: deterministic synthetic CIR, coherence, fusion, field, pose, tomography, gesture, drift, and adversarial fixtures
- Output path: Milestone 8 focused verification
- Result: focused RuvSense tests pass (`27 passed`).
- Command/config: `uv run pytest -q`; `uv run python -m json.tool notebooks/07_multistatic_node_comparison.ipynb`; `MPLBACKEND=Agg uv run --extra research python <notebook smoke>`
- Dataset: full unit/parity suite and synthetic multistatic notebook fixture
- Output path: parent Milestone 8 verification
- Result: full tests pass (`97 passed`, 1 existing FastAPI/Starlette warning); notebook JSON valid; notebook smoke executed 6 code cells with only noninteractive matplotlib warnings.
- Command/config: `uv run pytest -q tests/unit/test_ruvector_exports.py tests/unit/test_ruvector_signal.py tests/unit/test_ruvector_geometry.py tests/unit/test_ruvector_history.py`
- Dataset: deterministic RuVector signal, Fresnel, TDoA, viewpoint, and compressed-history fixtures
- Output path: Milestone 9 focused verification
- Result: focused RuVector tests pass (`27 passed`).
- Command/config: `uv run pytest -q`; `uv run python -m json.tool notebooks/11_ruvector_signal_geometry.ipynb`; `MPLBACKEND=Agg uv run --extra research python <notebook smoke>`
- Dataset: full unit/parity suite and synthetic RuVector signal/geometry notebook fixture
- Output path: parent Milestone 9 verification
- Result: full tests pass (`124 passed`, 1 existing FastAPI/Starlette warning); notebook JSON valid; notebook smoke executed 5 code cells.
- Command/config: `uv run pytest -q tests/unit/test_neural_training_exports.py tests/unit/test_nn_models.py tests/unit/test_training_dataset.py tests/unit/test_training_utils.py`
- Dataset: deterministic synthetic CSI, pose labels, heatmaps, metrics, checkpoint/export, and optional torch model-shape fixtures
- Output path: Milestone 10 focused verification
- Result: focused M10 tests pass (`15 passed`, `5 skipped` for optional torch).
- Command/config: `uv run pytest -q`; JSON checks for notebooks `08`-`10`; `MPLBACKEND=Agg uv run --extra research python <notebook smoke>`
- Dataset: full unit/parity suite and synthetic CSI-to-pose, dataset replay, and embedding notebook fixtures
- Output path: parent Milestone 10 verification
- Result: full tests pass (`139 passed`, `5 skipped`, 1 existing FastAPI/Starlette warning); notebooks JSON valid; notebook smoke executed 9 code cells with only noninteractive matplotlib warnings.
- Command/config: `uv run pytest -q tests/unit/test_mat_exports.py tests/unit/test_mat_domain_detection.py tests/unit/test_mat_localization_tracking.py tests/unit/test_mat_triage_alerts.py`
- Dataset: deterministic MAT domain, survivor/vitals, localization/tracking, triage, and alert fixtures
- Output path: Milestone 11 focused verification
- Result: focused MAT tests pass (`26 passed`).
- Command/config: `uv run pytest -q`; `uv run python -m json.tool notebooks/12_mat_research_pipeline.ipynb`; `MPLBACKEND=Agg uv run --extra research python <notebook smoke>`
- Dataset: full unit/parity suite and synthetic MAT rescue-zone notebook fixture
- Output path: parent Milestone 11 verification
- Result: full tests pass (`165 passed`, `5 skipped`, 1 existing FastAPI/Starlette warning); notebook JSON valid; notebook smoke executed 5 code cells with only noninteractive matplotlib warning.
- Command/config: `uv run pytest -q tests/unit/test_worldgraph_graph.py tests/unit/test_privacy_bfld.py tests/unit/test_worldgraph_privacy_exports.py`
- Dataset: deterministic room/sensor/person WorldGraph, semantic provenance, privacy mode rollup, BFLD payload, risk, signature, and demotion fixtures
- Output path: Milestone 12 focused verification
- Result: focused M12 tests pass (`16 passed`).
- Command/config: `uv run pytest -q`; `uv run python -m json.tool notebooks/13_worldgraph_privacy_provenance.ipynb`; `MPLBACKEND=Agg uv run --extra research python <notebook smoke>`
- Dataset: full unit/parity suite and synthetic WorldGraph privacy provenance notebook fixture
- Output path: parent Milestone 12 verification
- Result: full tests pass (`181 passed`, `5 skipped`, 1 existing FastAPI/Starlette warning); notebook JSON valid; notebook smoke executed 5 code cells with only noninteractive matplotlib warning.
- Command/config: `uv run pytest -q tests/unit/test_homecore_research.py tests/unit/test_nvsim_research.py tests/unit/test_swarm_research.py tests/unit/test_desktop_browser_helpers.py tests/unit/test_optional_exports.py`
- Dataset: deterministic HOMECORE state/automation, nvsim magnetic scenes, swarm topology/fusion, browser pose/fusion payloads, and desktop provisioning fixtures
- Output path: Milestone 13 focused verification
- Result: focused M13 tests pass (`31 passed`).
- Command/config: `uv run python -m json.tool notebooks/14_optional_tracks_research_overview.ipynb`; `MPLBACKEND=Agg uv run --extra research python <notebook smoke>`
- Dataset: synthetic optional-track overview notebook fixtures
- Output path: notebook `14` verification
- Result: notebook JSON valid; smoke executed 5 code cells with only noninteractive matplotlib warnings.
- Command/config: `uv run pytest -q`
- Dataset: full unit/parity suite after all `plan.md` milestones
- Output path: parent Milestone 13 verification
- Result: full tests pass (`212 passed`, `5 skipped`, 1 existing FastAPI/Starlette warning).
- Command/config: `rg` scan for local `1Projects` absolute-path prefixes in Markdown/notebook-style docs
- Dataset: docs, porting notes, project memory notes, README, and plan
- Output path: documentation cleanup verification
- Result: no remaining matches after replacing those prefixes with repo-name references.
- Command/config: `uv run python -m json.tool notebooks/04_phase_and_amplitude_visualization.ipynb`; `MPLBACKEND=Agg uv run --extra research python <notebook 04 smoke>`; `uv run pytest -q tests/unit/test_notebook_json.py`
- Dataset: deterministic synthetic phase/amplitude CSI fixture and all committed notebooks
- Output path: notebook `04` and notebook hygiene verification
- Result: notebook `04` JSON valid; smoke executed 2 code cells; notebook JSON/scaffolding tests pass (`2 passed`); every notebook has a `Fixture / simulated source:` line.
- Command/config: `uv run pytest -q tests/unit/test_notebook_json.py`
- Dataset: committed notebook JSON/scaffolding after README update
- Output path: README-adjacent verification
- Result: notebook JSON/scaffolding tests pass (`2 passed`).
- Command/config: manual `plan.md` rewrite
- Dataset: Android CSI feasibility planning around Pixel/rooted-device path and ESP32 baseline collection
- Output path: `plan.md`
- Result: old completed porting roadmap replaced with Android low-level CSI driver research plan; no Pixel modification should happen before supervisor approval.
- Next action: Optional post-port audit with real RuView captures/reference APIs and threshold tuning.

## Analysis Results

- `plan.md` frames this as a pure Python research port of RuView / WiFi-DensePose from `RuView`.
- The port should prioritize readable implementations, fixtures, tests, and notebooks over product packaging or commercial polish.
- Rust `wifi-densepose-core` canonical frame layout uses UUID bytes, fixed little-endian metadata fields, length-prefixed UTF-8 device id, 16 zero bytes for missing calibration id, `(nrows, ncols)` as `u32`, and stream-major complex samples as `f64 re || f64 im`.
- Python `CsiFrame` deep-copies metadata on construction to better match Rust ownership/move behavior and prevent accidental witness-hash changes from later external metadata mutation.
- Milestone 4 intentionally ports the Rust motion detector conceptually: weighted variance, temporal delta, phase variance, and subcarrier variance components with baseline-relative thresholds, plus debounce for human-readable empty/still/moving states.
- Milestone 5 intentionally uses compact NumPy FFT/PSD peak scoring rather than line-by-line streaming Rust filters; this keeps the core install light while preserving the ADR-021 breathing and heart-rate bands.
- Milestone 6 intentionally uses Python JSON baseline persistence with magic/version metadata, not the Rust ADR-135 little-endian binary ABI. The statistical behavior is ported first; binary parity can be added later if cross-tool interchange is needed.
- Milestone 7 intentionally ports the local research server surface, not the full Rust Axum production server. Host validation, auth, MQTT, Matter, edge registry, and static UI serving remain out of scope unless a later milestone needs them.
- Milestone 8 ports the advanced RuvSense surface as deterministic NumPy research primitives rather than exact Rust solver internals: CIR uses oversampled IFFT/top-k taps, fusion uses explicit quality/coherence/distance weights, and temporal/adversarial detectors use compact thresholded models.
- Milestone 9 ports RuVector behavior rather than crate internals: min-cut, attention, sparse solver, and temporal tensor concepts are represented as deterministic NumPy research helpers with tests and notebook coverage.
- Milestone 10 ports neural/training behavior as a base NumPy research API plus optional PyTorch modules; base tests skip torch-dependent model checks when the `nn` extra is not installed.
- Milestone 11 ports MAT behavior as local research primitives: disaster events, zones, survivors, vital detection, localization, tracking, triage, and alerts are deterministic Python APIs with no external emergency dispatch integration.
- Milestone 12 ports WorldGraph/BFLD behavior as deterministic graph, provenance, privacy, and witness primitives; it does not port the full streaming engine or production privacy control plane.
- Milestone 13 ports the optional later tracks as side-effect-free research helpers: HOMECORE state/automation, nvsim magnetic-scene simulation, swarm simulation models, browser payload builders, and desktop hardware-planning descriptors. It does not open browsers, serial ports, network sockets, sidecars, or live flight-control links.
- The original `plan.md` milestone list is now implemented through Milestone 13; remaining work should be framed as validation, parity hardening, real-data calibration, or new scope rather than unfinished initial porting.

## Learnings

- Memory should track capability-level port progress, source-reference mappings, intentional deviations from Rust/C behavior, and parity-test outcomes.
- Keep core install light (`numpy`, `blake3`) and put heavy research dependencies behind optional extras so parity tests stay fast.
- Use `uv sync --extra dev` and `uv run pytest -q` as the default local workflow.
- Synthetic presence/motion thresholds are useful for visual lab progress, but real ESP32 captures are still needed before treating scores as calibrated.
- Synthetic vitals fixtures are enough for API smoke tests and visual notebooks, but confidence/status thresholds remain uncalibrated until real CSI captures are available.
- Calibration deviation tests confirm empty-like vs person/drift-like synthetic windows, but drift trigger thresholds still need real-room validation.
- Server tests should use simulated/replay sources and FastAPI TestClient; UDP construction/timeout is tested without requiring live ESP32 hardware.
- For notebook smoke tests that execute plotting cells, set `MPLBACKEND=Agg` and close figures after each cell so automated checks stay non-interactive.
- Keep `ruview.ruvector` as the stable public namespace for behavior-level RuVector equivalents; downstream training modules can use these helpers without depending on external Rust/RuVector crates.
- Keep `ruview.nn` and `ruview.training` importable without torch so dataset/loss/metric/checkpoint/export research can run in the default `uv sync --extra dev` environment.
- Use explicit aliases for colliding MAT concepts (`DomainLocationUncertainty`, `LocalizationLocationUncertainty`, etc.) so the public namespace remains predictable.
- Keep `ruview.worldgraph` and `ruview.privacy` separately useful, with `ruview.worldgraph.trust` as the small integration layer for provenance + class demotion + witness hashing.
- Keep optional later-track modules pure and deterministic by default so notebooks and tests can compose them without hardware, browser, drone, or sidecar dependencies.

## Decisions

- Use in-repo memory as the operational record and the main knowledge-base workstream as the compact cross-project summary.

## Blockers

- No current blocker.

## Next

- Wait for supervisor approval before any Pixel root/modification.
- Prepare ESP32 CSI collection as the safe baseline and use the new `plan.md` checklist for Android chipset reconnaissance.
