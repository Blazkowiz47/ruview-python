# 2026-05-31 - sushruts-macbook-pro-m2-docs

- Node: sushruts-macbook-pro-m2-docs
- Device/server: local macOS workspace
- Repo path: `/Users/sushrutpatwardhan/1Projects/ruview-python`
- Branch: `master`

## Worker C - Milestone 2 ESP32 protocol docs

- Scope: documentation/notebook/source-map lane only.
- Changed paths: `docs/porting/esp32-protocols.md`, `notebooks/09_dataset_replay_lab.ipynb`, `tests/unit/test_notebook_json.py`, this memory note.
- Source mappings inspected:
  - Raw CSI: `firmware/esp32-csi-node/main/csi_collector.{c,h}`; authoritative 20-byte ADR-018 layout, PPDU byte 18, flags byte 19, and payload length rules.
  - Sync: `csi_collector.c` sync packet construction plus `v2/crates/wifi-densepose-hardware/src/sync_packet.rs` canonical decoder and parity bytes.
  - Edge vitals: `edge_processing.{c,h}` packed 32-byte vitals layout, fused-vitals sibling, feature-vector/compressed siblings.
  - WASM events: `wasm_runtime.{c,h}` variable-length `8 + event_count * 5` output packet.
  - Host references: `v2/crates/wifi-densepose-hardware/src/esp32_parser.rs` is current parser authority; older sensing-server/pointcloud parsers have stale offsets and should not drive the Python port.
- Command/config:
  - `uv run python -m json.tool notebooks/09_dataset_replay_lab.ipynb > /dev/null`
  - `uv run pytest -q tests/unit/test_notebook_json.py` -> `1 passed`
  - `uv run pytest -q` -> `24 passed`
- Result:
  - Added exact byte offsets and endian notes for raw CSI and ADR-110 sync packets.
  - Documented edge vitals and WASM output packets with TODOs for ambiguous dispatch.
  - Added a notebook packet-inspection cell that imports future `ruview.protocols.esp32` parser entry points when present and otherwise falls back to canonical ADR-110 bytes.
  - Added lightweight JSON validity coverage for all notebooks.
- Format uncertainties:
  - `0xC5110004` collides between fused vitals and WASM output in current firmware; parser worker needs a length/field heuristic or firmware-side magic fix.
  - Edge/WASM floats are assumed ESP32 little-endian IEEE-754 and should be fixture-tested.
  - Raw ADR-018 CSI has no per-frame local timestamp; mesh alignment requires sync-packet sequence interpolation.
- Next action: parser worker should implement magic dispatch from the doc, pin canonical raw/sync fixtures, and explicitly handle or reject the `0xC5110004` collision until firmware resolves it.
