# 2026-05-31 - sushruts-macbook-pro - m13 nvsim

- Node: sushruts-macbook-pro
- Repo: `ruview-python`
- Branch: `master`
- Start commit: `d146a90`
- Task: Port compact deterministic nvsim Python research subset from Rust behavior-level primitives.
- Files owned: `src/ruview/nvsim/__init__.py`, `src/ruview/nvsim/scene.py`, `src/ruview/nvsim/propagation.py`, `src/ruview/nvsim/pipeline.py`, `tests/unit/test_nvsim_research.py`, this note.
- Commands/results: `uv run --frozen pytest -q tests/unit/test_nvsim_research.py` -> 6 passed in 0.28s.
- Result summary: Added stdlib/NumPy nvsim scene primitives, canonical scene JSON, material attenuation, dipole/current-loop/ferrous field synthesis, deterministic magnetic frame bytes, ADC/noise/flag pipeline, and SHA-256 witness tests.
- Next action: Commit owned changes with `Add nvsim research simulator`; broader memory index intentionally untouched due file ownership restriction.
