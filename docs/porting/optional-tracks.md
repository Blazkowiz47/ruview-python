# Optional Track Porting Notes

Milestone 13 ties together the optional HOMECORE, nvsim, swarm, browser
visualization, and desktop hardware-tooling subsets. The Python scope is a
deterministic research notebook and documentation bridge, not a live controller
or a parity port of every Rust subsystem.

## Source References

- `/Users/sushrutpatwardhan/1Projects/ruview-python/plan.md`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/homecore/README.md`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/homecore/src/state.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/homecore/src/bus.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/homecore-automation/README.md`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/homecore-automation/src/trigger.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/homecore-automation/src/engine.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/nvsim/README.md`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/nvsim/src/pipeline.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/nvsim/src/proof.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/nvsim/src/wasm.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/ruview-swarm/README.md`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/ruview-swarm/src/planning/probability_grid.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/ruview-swarm/src/sensing/multiview.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/ruview-swarm/evals/RESULTS.md`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-desktop/src/lib.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-desktop/src/commands/discovery.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-desktop/src/commands/flash.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-desktop/src/commands/ota.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-desktop/src/commands/provision.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-desktop/src/commands/server.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-desktop/src/commands/wasm.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-desktop/ui/src/App.tsx`

## Deliverable Map

| Rust reference | Python research equivalent | Behavior to preserve |
|---|---|---|
| `homecore::StateMachine`, `State`, `EntityId` | Small optional HOMECORE state/event fixtures or later `ruview.optional_tracks.homecore` helpers | Strict `domain.entity` ids, state snapshots, no-op write suppression, state-change event ordering, and readable state transition tables. |
| `homecore::EventBus` | Deterministic event-log records for notebook plots | Preserve the split between typed system events and domain/integration events; lag or replay mechanics can be summarized, not simulated as Tokio broadcast channels. |
| `homecore-automation` triggers, conditions, actions, engine | A local trigger->condition->action trace and event-log visualization | Preserve state/numeric/event trigger semantics, condition pass/fail outcomes, and action intent. Do not execute real services or mutate Home Assistant state. |
| `nvsim::Scene`, source synthesis, attenuation, `NvSensor` | A Python forward-simulator research fixture or later `ruview.optional_tracks.nvsim` helper | Same fixture and seed produce the same magnetic-field trace; plots should expose axes in pT and seconds; approximations remain explicit. |
| `nvsim::Pipeline`, `MagFrame`, `Proof` | Frame-like time series plus SHA-256 witness summary | Keep deterministic witness hashing over quantized synthetic readings. The Python notebook records proof shape, not byte-for-byte Rust frame parity. |
| `nvsim` WASM bindings | Browser-safe simulator boundary notes | Preserve the idea that browser demos use deterministic inputs and caller-supplied seeds, with no filesystem, time, or hardware dependencies. |
| `ruview-swarm::ProbabilityGrid` | Deterministic probability-grid heatmap | Preserve Bayesian-style probability updates, coverage/scanned-cell intuition, and cell-indexed coordinates. |
| `ruview-swarm::MultiViewFusion` | Confidence-weighted detection fusion plot | Preserve minimum-viewpoint behavior, confidence thresholding, contributing-drone list, and uncertainty shrinking with viewpoint diversity. |
| ADR-149 eval artifacts | Notebook summary cues for seeded evaluation | Keep seed/episode matrix, IQM/CI language, and Stage-1 kinematic caveat visible; do not imply Gazebo/PX4 or flight physics validation. |
| Desktop discovery, flash, OTA, provisioning, server, WASM commands | Local dry-run helper-flow diagram and future audit-only command models | Preserve command shape, hash/status/progress concepts, and UI flow. The Python docs/notebook must not scan networks, open serial ports, flash firmware, upload OTA/WASM modules, or start live services. |
| Browser visualization helpers | Notebook/browser flow fixture | Preserve browser-preview and WASM-demo boundaries using synthetic data and static diagrams before any in-app visualization helper is wired. |

## Notebook Intent

- `notebooks/14_optional_tracks_research_overview.ipynb` is the deterministic
  visual lab for this optional milestone.
- The notebook first probes for future `ruview.optional_tracks` modules with
  guarded imports. If they are absent or incomplete, local NumPy/Matplotlib
  fixtures run instead.
- The fixture covers HOMECORE state transitions and automation events, nvsim
  magnetic-field traces and witness summaries, swarm probability grids and
  detection fusion, and browser/desktop helper flow.
- All code cells are unexecuted in git, use deterministic values, and can be
  smoke-tested with `MPLBACKEND=Agg uv run --extra research python`.

## Intentional Deviations

- Python should favor readable fixtures, dataclasses, arrays, and deterministic
  JSON over line-by-line Tokio, DashMap, Tauri, or Rust enum parity.
- HOMECORE automation is represented as a local event trace. A future Python
  helper may evaluate rules, but this docs/notebook slice does not load YAML or
  call services.
- nvsim is represented by a deterministic pT-scale forward trace and witness
  hash. It does not claim full Biot-Savart, ODMR, digitiser, or MagFrame byte
  compatibility until golden fixtures are exported from Rust.
- Swarm behavior is restricted to local probability and fusion research. It
  does not model formation control, Raft, MAPPO training, MAVLink, Remote ID,
  geofencing, or ITAR-gated behavior.
- Browser and desktop tooling are visualized as helper flows, not driven. The
  notebook may show command phases and hashes, but it must not discover nodes,
  upload firmware, provision NVS, control WASM modules, or manage a sensing
  server process.

## Research-Only Boundary

- Milestone 13 artifacts are local, synthetic, and hardware-free.
- No network broadcast, mDNS, HTTP OTA, serial flashing, NVS provisioning,
  Tauri command invocation, browser telemetry, drone flight control, or live
  Home Assistant control belongs in the notebook.
- Any future helper that can touch hardware should expose an explicit dry-run
  mode before being called from notebooks or tests.
- Witness hashes in the notebook are reproducibility diagnostics only. They are
  not compliance attestations or signed hardware proofs.
- Swarm plots are kinematic research sketches. They should not be described as
  safety evidence, field readiness, or autonomous-flight validation.

## Verification

Initial verification should include:

- JSON validity for `notebooks/14_optional_tracks_research_overview.ipynb`.
- The selected notebook scaffolding test in `tests/unit/test_notebook_json.py`.
- A smoke execution of notebook code cells with `MPLBACKEND=Agg`.
- Future focused tests only after optional `ruview.optional_tracks` helpers
  exist, using deterministic fixtures and dry-run-only command surfaces.

## Run Path

```bash
uv sync --extra research
uv run jupyter lab notebooks
```

Open `notebooks/14_optional_tracks_research_overview.ipynb` and choose Run All.
The notebook should run without radio hardware, discovered ESP32 nodes, drone
controllers, Tauri APIs, browser automation, Home Assistant services, or
completed Milestone 13 Python modules.
