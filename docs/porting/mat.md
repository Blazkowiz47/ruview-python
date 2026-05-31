# MAT Porting Notes

Milestone 11 ports the Mass Casualty Assessment Tool reference into Python
research equivalents for deterministic experiments, notebooks, and later API
work. The Python side should preserve the domain concepts and verification
invariants without claiming emergency-response production readiness.

## Source References

- `/Users/sushrutpatwardhan/1Projects/ruview-python/plan.md`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-mat/README.md`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-mat/src/lib.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-mat/src/domain/disaster_event.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-mat/src/domain/scan_zone.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-mat/src/domain/survivor.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-mat/src/domain/vital_signs.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-mat/src/domain/coordinates.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-mat/src/domain/triage.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-mat/src/domain/alert.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-mat/src/domain/events.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-mat/src/detection/breathing.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-mat/src/detection/heartbeat.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-mat/src/detection/movement.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-mat/src/detection/ensemble.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-mat/src/detection/pipeline.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-mat/src/localization/triangulation.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-mat/src/localization/depth.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-mat/src/localization/fusion.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-mat/src/localization/range_constraint.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-mat/src/tracking/tracker.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-mat/src/tracking/lifecycle.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-mat/src/tracking/fingerprint.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-mat/src/alerting/generator.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-mat/src/alerting/dispatcher.rs`
- `/Users/sushrutpatwardhan/1Projects/RuView/v2/crates/wifi-densepose-mat/src/alerting/triage_service.rs`

## Deliverable Map

| Rust reference | Python deliverable | Behavior to preserve |
|---|---|---|
| `domain/disaster_event.rs`, `domain/events.rs` | `ruview.mat` disaster/event records and event helpers | Capture disaster type, status, start metadata, domain event names, timestamps, and detection/zone/alert/tracking event categories for reproducible research logs. |
| `domain/scan_zone.rs` | `ruview.mat.ScanZone`, `ZoneBounds`, scan parameters, and sensor-position records | Rectangle/circle/polygon containment and area, zone status transitions, scan counters, operational sensor filtering, and the invariant that at least three operational sensors are needed for 2-D localization. |
| `domain/survivor.rs`, `domain/vital_signs.rs` | Survivor aggregate, vital-sign readings, confidence scores, breathing/heartbeat/movement value objects | Preserve confidence clamping, vital history, stale/lost/rescued/false-positive states, alert eligibility, deterioration hints, and the distinction between no signal and detected but low-confidence vitals. |
| `detection/breathing.rs`, `detection/heartbeat.rs`, `detection/movement.rs`, `detection/ensemble.rs`, `detection/pipeline.rs` | Lightweight vital-survivor detection pipeline over NumPy arrays, with optional later hooks into `ruview.vitals`, `ruview.ruvsense`, and learned models | Keep breathing-rate bands, heartbeat micro-motion intent, movement categories, ensemble confidence, sample-rate/window checks, and ML enhancement as research hooks rather than mandatory dependencies. |
| `domain/coordinates.rs`, `localization/depth.rs`, `localization/triangulation.rs`, `localization/fusion.rs`, `localization/range_constraint.rs` | Coordinate, uncertainty, debris-profile, depth, range-constraint, trilateration/TDoA, and position-fusion helpers | Preserve meters-based 3-D coordinates, below-surface depth sign convention, actionable uncertainty thresholds, RSSI/ToA/TDoA solver shapes, geometric conditioning diagnostics, and weighted fusion reducing uncertainty when estimates agree. |
| `tracking/tracker.rs`, `tracking/lifecycle.rs`, `tracking/fingerprint.rs`, `tracking/kalman.rs` | Survivor tracking and re-identification helpers | Keep stable track IDs, tentative-to-active promotion, active-to-lost misses, lost re-ID windows, Kalman-style prediction/update, fingerprint matching from vitals/location, and deterministic assignment for small research fixtures. |
| `domain/triage.rs`, `alerting/triage_service.rs` | START-style triage scoring and mass-casualty summaries | Preserve `Immediate`, `Delayed`, `Minor`, `Deceased`, and `Unknown` categories; priority ordering; breathing-rate thresholds; movement as a responsiveness proxy; and upgrade behavior when vitals deteriorate. |
| `domain/alert.rs`, `alerting/generator.rs`, `alerting/dispatcher.rs` | Local alert objects, local queues, and notebook/test alert summaries | Preserve priority-from-triage, payload fields, acknowledgement/resolution status, escalation counters, and handler abstractions only for local console/test flows. |

## Notebook Intent

- `notebooks/12_mat_research_pipeline.ipynb` is the lightweight visual lab for
  this milestone.
- The fixture is deterministic and covers one rubble-zone survivor scenario
  with synthetic breathing, heart-rate, movement, localization, tracking,
  triage, and local-only alert summaries.
- The notebook first probes `ruview.mat` with guarded imports. While code
  workers are still shaping that package, local NumPy helpers keep the notebook
  JSON-valid and smoke-testable with only NumPy and Matplotlib.

## Research-Only Boundary

- Alerts are local/test-only. Do not add production emergency notification
  integrations, dispatch integrations, municipal service hooks, paging systems,
  SMS/email gateways, or responder workflow automation in this milestone.
- Notebook labels and triage colors are diagnostics for research fixtures, not
  clinical or operational rescue recommendations.
- Synthetic survivors, vitals, and coordinates are deliberately fictional and
  must not be mixed with live incident data.

## Intentional Deviations

- Python should use clear dataclasses and NumPy arrays instead of line-by-line
  Rust DDD/event-sourcing internals.
- UUID generation, async event stores, Axum APIs, WebSocket streaming, and
  distributed/drone feature flags are outside this research-notebook scope.
- Breathing and heartbeat helpers may start as deterministic spectral fallbacks;
  numeric parity with Rust FFT/notch-filter details should wait for exported
  golden fixtures.
- TDoA and range solvers may use compact least-squares or grid-search fallbacks;
  tests should emphasize known synthetic positions, finite residuals, and
  graceful underdetermined failures.
- Tracking can begin with deterministic nearest-neighbor or exponential
  smoothing before full Kalman/re-ID parity.
- Alert dispatch remains an in-memory queue or printed summary. The only
  acceptable handlers for this scope are local diagnostics used by tests or
  notebooks.

## Verification

Initial verification should include:

- JSON validity for `notebooks/12_mat_research_pipeline.ipynb`.
- The selected notebook scaffolding test in `tests/unit/test_notebook_json.py`.
- A smoke execution of notebook code cells with `MPLBACKEND=Agg` when feasible.
- Focused future unit tests for zone containment, confidence clamping, triage
  categories, location uncertainty, TDoA/trilateration synthetic positions,
  track lifecycle transitions, and local alert queue semantics.
- Optional Rust-to-Python parity fixtures only after the parent process exports
  stable behavior traces from the Rust reference.

## Run Path

```bash
uv sync --extra research
uv run jupyter lab notebooks
```

Open `notebooks/12_mat_research_pipeline.ipynb` and choose Run All. The
notebook should run without radio hardware because it uses inline deterministic
synthetic data and local fallback helpers.
