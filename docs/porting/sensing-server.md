# Sensing Server Porting Notes

Milestone 7 ports the operational shape of the Rust sensing server into a local
Python research surface. The Python version is intentionally small at first: it
standardizes sources, message shape, examples, and local-only HTTP/WebSocket
contracts before taking on production features from the Axum server.

## Source References

| Source | Why it matters |
|---|---|
| `RuView/v2/crates/wifi-densepose-sensing-server/README.md` | High-level Rust server architecture and default ports. |
| `RuView/v2/crates/wifi-densepose-sensing-server/src/main.rs` | Axum routes, WebSocket broadcast path, simulated source, UDP ingest loop, and latest/vitals endpoints. |
| `RuView/v2/crates/wifi-densepose-sensing-server/src/types.rs` | `SensingUpdate`, node, feature, classification, field, and vital-sign structures. |
| `RuView/v2/crates/wifi-densepose-sensing-server/src/recording.rs` | JSONL recording rows written from processed CSI frames. |
| `RuView/v2/crates/wifi-densepose-hardware/src/esp32_parser.rs` | Rust ESP32 CSI parser reference used by the live UDP path. |
| `ruview-python/src/ruview/protocols/esp32.py` | Current Python ESP32 UDP packet parser. |

## Python Server Intent

The Python server is a research and notebook-adjacent sensing surface. It should
make simulator, replay, and live ESP32 UDP sources look alike, publish a compact
`sensing_update` stream, and expose enough local REST/WebSocket API for UI and
agent experiments to consume current state.

The intended local routes mirror the Milestone 7 plan:

- `GET /api/v1/sensing/latest`
- `GET /api/v1/vital-signs`
- `WS /ws/sensing`

Run the local research server:

```bash
uv sync --extra research
uv run --extra research python -m ruview.server.app --source simulated --host 127.0.0.1 --port 8080
```

The Python source layer should support:

- `simulated` for deterministic synthetic CSI from `ruview.hardware.simulator`;
- `replay` for JSONL recordings and saved update rows;
- `esp32` for live UDP packets parsed by `ruview.protocols.esp32`.

## Local-Only Scope

Bind local development servers to `127.0.0.1` by default. The Python port is not
the production Axum server and should not expose LAN-facing control surfaces
without an explicit follow-up design for host validation, authentication, CORS,
and deployment hardening.

The examples in this milestone are safe to run without starting a web server:
they print summaries to stdout and use stderr for operational status.

## Message Shape

The shared update shape is intentionally close to the Milestone 7 plan while
allowing optional richer fields later:

```json
{
  "type": "sensing_update",
  "source": "simulated|esp32|replay",
  "tick": 1,
  "nodes": [],
  "features": {},
  "classification": {},
  "signal_field": {},
  "vital_signs": {},
  "estimated_persons": 1
}
```

Recommended field meanings:

| Field | Meaning |
|---|---|
| `type` | Always `sensing_update` for WebSocket/update rows. Python may also accept Rust-style `msg_type` and normalize it at the edge. |
| `source` | One of `simulated`, `esp32`, or `replay`; suffixes may be added for diagnostics, but examples keep the base label. |
| `tick` | Monotonic source-local counter, recording sequence, or JSONL row index. |
| `nodes` | Per-node summaries such as `node_id`, `rssi_dbm`, `subcarrier_count`, and optional amplitude vectors. |
| `features` | Scalar signal features such as variance, motion energy, breathing-band power, and mean amplitude. |
| `classification` | Presence and motion classification with confidence. |
| `signal_field` | Optional room/grid representation for UI experiments. |
| `vital_signs` | Optional breathing and heart-rate estimates. |
| `estimated_persons` | Privacy-preserving count estimate. |

## Run Paths

Simulated CSI summary:

```bash
uv run python examples/simulate_empty_vs_present.py --frames 128 --seed 42
```

Replay a JSONL recording:

```bash
uv run python examples/replay_recording.py data/recordings/session.csi.jsonl --limit 20
```

Create a tiny replay fixture by hand:

```bash
printf '%s\n' \
  '{"timestamp": 1.0, "subcarriers": [1.0, 1.2, 0.9], "rssi": -52, "features": {"presence_score": 0.8, "motion_energy": 0.2}}' \
  '{"type": "sensing_update", "source": "replay", "tick": 2, "nodes": [], "features": {}, "classification": {"presence": true, "motion_level": "still", "confidence": 0.7}, "signal_field": {}, "vital_signs": {}, "estimated_persons": 1}' \
  > /tmp/ruview-replay.jsonl
uv run python examples/replay_recording.py /tmp/ruview-replay.jsonl
```

The replay example tries `ruview.server.sources.ReplaySensingSource` when it is
available. Until that API exists, or when `--source-mode local` is passed, it
uses its built-in JSONL fallback parser.

Live ESP32 UDP packet summary:

```bash
uv run python examples/live_udp_viewer.py --host 0.0.0.0 --port 5005
```

Use `--limit N` during smoke tests so the command exits after receiving N parsed
packets. Parser errors are surfaced as CLI errors because malformed UDP packets
usually indicate a firmware/protocol mismatch.

## Intentional Deviations From Rust Axum

- No production static-file hosting is required for the Python research port.
- No LAN binding, bearer auth, Host-header policy, or CORS policy is implied by
  the examples.
- The replay path accepts simple JSONL updates and recording rows instead of the
  full Rust recording/session management API.
- The UDP example prints parsed packet summaries; it does not run the full
  server pipeline or maintain shared state.
- The Python server should prefer deterministic simulator/replay behavior for
  tests. Hardware availability should not be required for `uv run pytest -q`.
- Rich Rust features such as RVF model loading, SONA adaptation, MQTT, Matter,
  edge registry endpoints, and static UI serving remain out of scope unless a
  later milestone explicitly ports them.

## Caveats

Milestone 7 server schemas and source classes may land in parallel under
`src/ruview/server/*`. Examples should keep guarded imports and clear CLI
messages so they remain useful while those APIs settle.
