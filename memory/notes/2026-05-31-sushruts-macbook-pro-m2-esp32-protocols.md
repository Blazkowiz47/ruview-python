---
date: 2026-05-31
work_date: 2026-05-31
project: ruview-python
node: sushruts-macbook-pro-m2
node_type: laptop
device: Sushrut's MacBook Pro M2
timezone: Europe/Oslo
repo_path: ruview-python
branch: master
commit:
sync_status: committed
source_format: node-specific
tags: [ruview-python, milestone-2, esp32, protocol, csi]
---

# 2026-05-31 ESP32 Protocol Parsers

## Intent

- Worker A Milestone 2 scope: implement host-side ESP32 binary packet parsers and focused tests.

## Source Mappings

- Raw CSI magic `0xC5110001`: `firmware/esp32-csi-node/main/csi_collector.c`, `csi_serialize_frame`; current 20-byte header uses `u16 n_subcarriers`, `u32 frequency_mhz`, `u32 sequence`, signed RSSI/noise, byte 18 PPDU type, byte 19 flags.
- Edge vitals magic `0xC5110002`: `firmware/esp32-csi-node/main/edge_processing.h`, `edge_vitals_pkt_t`, 32 bytes packed.
- WASM event magic `0xC5110004`: `firmware/esp32-csi-node/main/wasm_runtime.h`, `wasm_output_pkt_t`, 8-byte header plus 5-byte packed events.
- Sync magic `0xC511A110`: `firmware/esp32-csi-node/main/csi_collector.c`, ADR-110 sync packet emitted every configured N CSI frames, 32 bytes.
- Rust reference checked: `v2/crates/wifi-densepose-sensing-server/src/csi.rs` parses vitals/WASM and an older raw CSI layout; Python follows the newer firmware source comments for raw CSI.

## Work Log

- Added `src/ruview/protocols/esp32.py` with dataclasses, little-endian `struct` unpacking, dispatch by magic, parse aliases, packet exceptions, and signed I/Q amplitude/phase helpers.
- Updated `src/ruview/protocols/__init__.py` exports.
- Added `tests/unit/test_esp32_protocols.py` with deterministic handcrafted bytes for raw CSI, edge vitals, WASM events, sync, invalid magic, truncation, odd I/Q payloads, and zero dimensions.

## Experiments / Runs

- Command/config: `uv run pytest -q`
- Dataset/input: deterministic handcrafted ESP32 packets in `tests/unit/test_esp32_protocols.py` plus existing core/parity tests.
- Output path: pytest stdout.
- Result: `23 passed in 0.10s`.

## Result

- ESP32 host-side packet parsers are implemented for raw CSI, edge vitals, WASM events, and ADR-110 sync packets.
- Parser behavior follows current firmware for raw CSI byte offsets; this intentionally differs from the older Rust sensing-server raw parser that treated some header fields as narrower/shifted.

## Next

- Hand off source-format uncertainty around magic `0xC5110004`: firmware currently uses it for both WASM output (`wasm_runtime.h`) and ADR-063 fused vitals (`edge_processing.h`); Worker A implemented the plan-requested WASM event parser.
