---
date: 2026-05-31
work_date: 2026-05-31
project: ruview-python
node: sushruts-macbook-pro
node_type: laptop
device: Sushrut's MacBook Pro
timezone: Europe/Oslo
repo_path: /Users/sushrutpatwardhan/1Projects/ruview-python
branch: master
commit: worker commit `Document sensing server examples`
sync_status: draft
source_format: node-specific
tags: [phd, research, ruview, wifi-densepose, python, sensing-server, milestone-7]
---

# Milestone 7 Examples And Docs

## Work Done

- Replaced the `examples/replay_recording.py` placeholder with a runnable JSONL replay example.
- Added guarded support for future `ruview.server.sources.ReplaySensingSource`, with local fallback parsing for full `sensing_update` rows and simple recording rows containing `timestamp`, `subcarriers`, `rssi`, `noise_floor`, and `features`.
- Kept `examples/live_udp_viewer.py` unchanged after confirming its help path still works with the existing UDP receiver/parser surface.
- Added `docs/porting/sensing-server.md` with reference sources, Python server intent, local-only scope, update message shape, simulated/replay/live UDP run paths, and intentional deviations from the Rust Axum server.

## Source Reference

- Reference repo: `/Users/sushrutpatwardhan/1Projects/RuView`
- Rust sources read: `v2/crates/wifi-densepose-sensing-server/README.md`, `src/main.rs`, `src/types.rs`, and `src/recording.rs`.
- Python context read: `src/ruview/server/__init__.py`, `src/ruview/hardware/udp_receiver.py`, `src/ruview/protocols/esp32.py`, and existing Milestone 7 examples.

## Verification

- Command/config: `uv run python examples/replay_recording.py --help`
- Result: help rendered successfully.
- Command/config: `uv run python examples/live_udp_viewer.py --help`
- Result: help rendered successfully.
- Command/config: `uv run pytest -q`
- Result: `59 passed`

## Notes

- `src/ruview/server/*` was treated as read-only because another Milestone 7 worker owns schemas/sources there.
- `ReplaySensingSource` is not present in this checkout yet, so replay uses the local JSONL parser unless a future server source becomes importable at runtime.
- The replay fallback defaults missing ticks to one-based row order and prints compact summaries; `--json` emits normalized compact update JSON.

## Next

- When the server source worker lands `ReplaySensingSource`, run `uv run python examples/replay_recording.py <recording> --source-mode server --limit 5` against a small fixture and tighten the duck-typed adapter if needed.
