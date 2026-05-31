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
branch: master
commit: pending
sync_status: draft
source_format: node-specific
tags: [phd, research, ruview, wifi-densepose, python, esp32, udp, csi]
---

# 2026-05-31 UDP Receiver

## Work Done

- Added `src/ruview/hardware/udp_receiver.py` with a synchronous `UdpReceiver` for host-side UDP datagrams, parser injection, lazy default parser lookup, timeout handling, close/idempotent context-manager behavior, and packet envelopes.
- Added replay helpers `iter_packet_bytes()` and `replay_packets()` for parsing packet files or in-memory byte iterables without opening a socket.
- Exported the receiver and replay helpers from `src/ruview/hardware/__init__.py`.
- Replaced `examples/live_udp_viewer.py` placeholder with a small CLI that binds a UDP host/port, consumes the default ESP32 parser, and prints compact parsed frame summaries.
- Added `tests/unit/test_udp_receiver.py` covering localhost UDP receive, parser calls, timeout, close/context-manager behavior, replay from paths and bytes, and monkeypatched lazy default parser resolution.

## Source Mappings

- Python receiver surface maps to the host-side receive loops in `RuView/v2/crates/wifi-densepose-hardware/src/aggregator/mod.rs` and `RuView/v2/crates/wifi-densepose-hardware/src/bin/aggregator.rs`.
- Timeout and parse-accounting behavior was cross-checked against `RuView/v2/crates/wifi-densepose-mat/src/integration/csi_receiver.rs`.
- Deliberate Python deviation: no background thread or channel; `UdpReceiver.receive()` is a simple blocking call returning one parsed packet or `None` on timeout.

## Command / Config

- Command: `uv run pytest -q`
- Dataset/input: synthetic unit-test byte datagrams plus localhost UDP socket delivery on an ephemeral port.
- Output paths: `src/ruview/hardware/udp_receiver.py`, `tests/unit/test_udp_receiver.py`, `examples/live_udp_viewer.py`
- Result: `15 passed in 0.13s`.

## Assumptions

- Worker A will provide `ruview.protocols.esp32.parse_packet(bytes)`; until then callers can pass an explicit parser or monkeypatch the module in tests.
- Parser errors are surfaced to the caller, while receive timeouts return `None`.

## Next

- After Worker A lands the ESP32 parser, run replay/live smoke tests against real ADR-018/ESP32 fixture bytes and adjust only the viewer summary if the parsed object shape differs from core `CsiFrame`.
