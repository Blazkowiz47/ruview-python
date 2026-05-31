---
date: 2026-05-31
work_date: 2026-05-31
project: ruview-python
node: sushruts-macbook-pro
node_type: laptop
device: Sushrut's MacBook Pro
timezone: Europe/Oslo
repo_path: ruview-python
branch: master
commit: worker commit `Add sensing server sources`
sync_status: draft
source_format: node-specific
tags: [phd, research, ruview, wifi-densepose, python, sensing-server]
---

# Milestone 7 Server Schemas And Sources

## Work Done

- Added plain dataclass server schemas for `NodeInfo`, `SensingUpdate`, feature/classification summaries, signal fields, and vital-sign summaries.
- Added deterministic `SimulatedSensingSource` using the synthetic CSI simulator plus existing presence/motion classifiers.
- Added `ReplaySensingSource` for JSONL files containing full sensing updates or simple CSI-ish amplitude records.
- Added `UdpSensingSource` around `ruview.hardware.UdpReceiver` with timeout-aware `next_update`/`iter_updates` behavior and packet summaries for raw CSI, edge vitals, sync, WASM, and generic parsed packets.
- Added `LatestState` as a small in-memory latest update and bounded history helper for later app integration.
- Exported the server schema/source API from `src/ruview/server/__init__.py`.
- Added focused unit tests in `tests/unit/test_server_sources.py`.

## Source Reference

- Plan: `ruview-python/plan.md` Milestone 7.
- Reference repo: `RuView`
- Rust sources sampled: `v2/crates/wifi-densepose-sensing-server/src/types.rs`, `csi.rs`, and `recording.rs`.

## Verification

- Command/config: `uv run pytest -q tests/unit/test_server_sources.py`
- Result: `6 passed`
- Command/config: `uv run pytest -q`
- Result: `65 passed`

## Notes

- `schemas.py` and `sources.py` intentionally avoid FastAPI imports.
- Simulated classification is baseline-backed against the deterministic `empty_room` synthetic scenario.
- Replay JSONL overrides recorded source to `replay` by default while preserving recorded ticks when present.
- UDP tests avoid network flakiness by using localhost ephemeral bind with a short timeout and a fake receiver for packet conversion.

## Next

- App/WebSocket workers can import `LatestState` and the source classes without creating a FastAPI dependency in the schema/source layer.
