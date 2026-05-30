# Runs

Track experiments, long-running jobs, evaluations, and important development runs.

| Date | Device/server | Branch/commit | Command/config | Dataset | Output path | Result | Next |
|---|---|---|---|---|---|---|---|
| 2026-05-31 | sushruts-macbook-pro | none / no git repo | `.venv/bin/python -m pytest -q` | synthetic CSI and pose fixtures | `tests/unit/test_core_contracts.py`, `tests/parity/test_adr136_canonical.py` | `9 passed`; core contracts and ADR-136 witness vector verified | Start Milestone 2 packet parser fixtures |
| 2026-05-31 | sushruts-macbook-pro | none / no git repo | `.venv/bin/python examples/simulate_empty_vs_present.py` | synthetic empty/present CSI frames | stdout | empty mean amplitude `1.008`, present mean amplitude `1.358`, witness `d38bfb2309a59a0378769bf4aa8ea85093eae7187a355004718269b2824c543d` | Replace example with signal helpers in Milestone 3 |
| 2026-05-31 | sushruts-macbook-pro | none / no git repo | `uv sync --extra dev`; `uv run pytest -q` | synthetic CSI and pose fixtures | `.venv/`, `uv.lock`, tests | `uv 0.10.4` created CPython `3.12.12` env; `9 passed` | Continue using `uv run` for checks |
| 2026-05-31 | sushruts-macbook-pro | main / pre-initial-commit | `uv run pytest -q` | synthetic CSI and pose fixtures | pre-commit verification | `9 passed` after `.gitignore` expansion | Commit initial scaffold |

## Notes

- Record parser parity tests, DSP fixture tests, and notebook experiments here as the port progresses.
