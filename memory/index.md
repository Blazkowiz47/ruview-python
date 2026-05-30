# ruview-python Memory

## Context Card

Status: milestone-2-complete
Domain: phd
Tags: phd, research, ruview, wifi-densepose, python, csi, signal-processing
Project path: /Users/sushrutpatwardhan/1Projects/ruview-python
Main brain workstream: /Users/sushrutpatwardhan/sushrut/wiki/workstreams/ruview-python/index.md
Devices/servers: sushruts-macbook-pro
Latest useful result: Milestone 2 is implemented: ESP32 packet parsers, UDP/replay helpers, packet-format docs, and notebook JSON checks are committed. `uv run pytest -q` passes (`24 passed`).
Current blocker: Milestone 3 signal-processing modules and visual lab notebooks have not started.
Next action: Start Milestone 3 with signal primitives (amplitude/phase conversion, phase unwrap, Hampel filtering, feature extraction, subcarrier variance, synthetic CSI generator) and update notebooks `00`, `01`, and `05`.

## Active Threads

- Build a readable, experiment-friendly Python research port of RuView / WiFi-DensePose from `/Users/sushrutpatwardhan/1Projects/RuView`.
- Port by capability boundary rather than line-by-line translation.
- Preserve data contracts and wire formats where they matter, with fixtures, tests, and notebooks for inspection.

## Recent Work

- 2026-05-31: Initialized project memory from the knowledge base and linked it to the main workstream.
- 2026-05-31: Captured the current repo state as not a Git repository with `plan.md` as the only discovered project file.
- 2026-05-31: Created the Python package scaffold, first core data contracts, synthetic notebook placeholders, examples, and tests. Verified with `.venv/bin/python -m pytest -q` (`9 passed`) and a synthetic empty-vs-present example.
- 2026-05-31: Replaced the pip-created `.venv` with a `uv`-managed environment via `uv sync --extra dev`; verified with `uv run pytest -q` (`9 passed`).
- 2026-05-31: Expanded `README.md` with project goals, layout, `uv` quick check, and explicit acknowledgement of upstream `ruvnet/RuView` and its MIT license notice.
- 2026-05-31: User initialized Git; expanded `.gitignore` for Python, `uv`, notebook, editor, and local recording artifacts; verified with `uv run pytest -q` before first commit.
- 2026-05-31: Completed Milestone 2 in three commits: `dfb9e92` UDP receiver/replay helpers, `8b8e803` ESP32 packet parsers, `d123164` ESP32 packet docs/notebook checks. Parent verification: `uv run pytest -q` (`24 passed`) and 11 notebooks validated as JSON.

## Recent Runs

- See `runs.md`.

## Durable Learnings

- See `learnings.md`.

## Decisions

- See `decisions.md`.

## Commands

- See `commands/index.md`.

## Links

- Main workstream: `/Users/sushrutpatwardhan/sushrut/wiki/workstreams/ruview-python/index.md`
- Port plan: `/Users/sushrutpatwardhan/1Projects/ruview-python/plan.md`
- Reference repository: `/Users/sushrutpatwardhan/1Projects/RuView`
