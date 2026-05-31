# ruview-python Memory

## Context Card

Status: milestone-12-complete
Domain: phd
Tags: phd, research, ruview, wifi-densepose, python, csi, signal-processing
Project path: /Users/sushrutpatwardhan/1Projects/ruview-python
Main brain workstream: /Users/sushrutpatwardhan/sushrut/wiki/workstreams/ruview-python/index.md
Devices/servers: sushruts-macbook-pro
Latest useful result: Milestone 12 is implemented: WorldGraph node/edge/provenance models, deterministic graph snapshots, privacy rollups, BFLD header/payload/CRC primitives, privacy modes and attestation chain, identity-risk gate, signature hashes, monotonic privacy demotion, trust-throughline witness helpers, public `ruview.worldgraph`/`ruview.privacy` exports, docs, and notebook `13`. Parent verification: `uv run pytest -q` (`181 passed`, `5 skipped`, 1 existing FastAPI/Starlette warning) plus notebook JSON and `MPLBACKEND=Agg` notebook smoke.
Current blocker: No current blocker.
Next action: Start Milestone 13 optional later tracks with scoped HOMECORE, nvsim, swarm, browser visualization, and desktop hardware-tooling research subsets.

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
- 2026-05-31: Completed Milestone 3 in three commits: `0ab43fe` signal primitives, `efd73d5` synthetic CSI simulator, `58c783d` visual lab notebooks. Parent verification: `uv run pytest -q` (`37 passed`) and synthetic example run with seed 42.
- 2026-05-31: Completed Milestone 4 in two worker commits: `600cd74` presence/motion classifiers and `589a1eb` motion visual notebook/docs. Parent verification: `uv run pytest -q` (`41 passed`).
- 2026-05-31: Completed Milestone 5 in two worker commits: `cd5f90a` vital sign estimators and `6750d38` vital signs visual notebook/docs. Parent verification: `uv run pytest -q` (`52 passed`).
- 2026-05-31: Completed Milestone 6 in two worker commits: `1b34432` calibration drift notebook/docs and `6f72a73` calibration baseline primitives. Parent verification: `uv run pytest -q` (`59 passed`) and notebook smoke executed 7 code cells.
- 2026-05-31: Completed Milestone 7 in three commits: `97c2f5d` sensing examples/docs, `0fb072b` server schemas/sources, and `59841e1` FastAPI app integration. Parent verification: `uv run pytest -q` (`70 passed`) plus replay/server help smoke checks.
- 2026-05-31: Completed Milestone 8 in six commits: `69bfbb1` multiband/multistatic fusion, `adaeaee` CIR/coherence, `eae6b19` multistatic notebook/docs, `785d47f` field/pose/tomography, `101c15e` temporal RuvSense primitives, and parent public-export integration. Parent verification: `uv run pytest -q` (`97 passed`) plus notebook JSON and `MPLBACKEND=Agg` notebook smoke.
- 2026-05-31: Completed Milestone 9 in five commits: `a3041f9` RuVector geometry solvers, `42ff578` RuVector signal primitives, `ce289e9` docs/notebook, `886376a` compressed histories, and parent public-export integration. Parent verification: `uv run pytest -q` (`124 passed`) plus notebook JSON and `MPLBACKEND=Agg` notebook smoke.
- 2026-05-31: Completed Milestone 10 in five commits: `9a4e4bb` neural model modules, `72b7ad0` neural notebooks/docs, `34ceb63` tensor/dataset pipeline, `3d0e5ab` training utilities/export helpers, and parent public-export integration. Parent verification: `uv run pytest -q` (`139 passed`, `5 skipped`) plus notebook JSON and `MPLBACKEND=Agg` notebook smoke for `08`-`10`.
- 2026-05-31: Completed Milestone 11 in five commits: `9e7c901` MAT docs/notebook, `2195498` triage/local alerts, `c929e4d` localization/tracking, `698e5fe` domain/detection, and parent public-export integration. Parent verification: `uv run pytest -q` (`165 passed`, `5 skipped`) plus notebook JSON and `MPLBACKEND=Agg` notebook smoke for notebook `12`.
- 2026-05-31: Completed Milestone 12 in four commits: `851cc02` docs/notebook, `05b5d9b` WorldGraph graph/provenance, `65922dd` BFLD privacy primitives, and parent public-export/trust-throughline integration. Parent verification: `uv run pytest -q` (`181 passed`, `5 skipped`) plus notebook JSON and `MPLBACKEND=Agg` notebook smoke for notebook `13`.

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
