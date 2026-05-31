# 2026-05-31 - sushruts-macbook-pro - Milestone 3 synthetic CSI

- Node: sushruts-macbook-pro
- Repo: `ruview-python`
- Branch/base: `master` at `fbf21db Record milestone 2 completion`

## Session Log

- Worker E scope: synthetic CSI generation and local fixture support only.
- Implemented deterministic NumPy CSI simulation for `empty_room`, `person_present`, `walking`/`motion`, and `stillness` in `src/ruview/hardware/simulator.py`.
- Simulator returns core `CsiFrame` objects with deterministic timestamps, sequence numbers, frame IDs, metadata, and shaped complex arrays.
- Added `.npz` save/load helpers and documented `data/synthetic/` as metadata/local-fixture space without committing generated binary fixtures.
- Added CLI summary example in `examples/simulate_empty_vs_present.py` and unit coverage for determinism, shape/metadata, scenario amplitude/motion differences, and fixture roundtrip.

## Verification

- `uv run pytest -q` -> 36 passed.
- `uv run python examples/simulate_empty_vs_present.py --frames 16 --seed 42` -> printed deterministic summary stats for empty, present, stillness, and walking scenarios.
