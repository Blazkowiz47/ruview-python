# Decisions

Record meaningful project decisions and why they were made.

| Date | Decision | Reason | Consequence | Revisit? |
|---|---|---|---|---|
| 2026-05-31 | Keep operational memory inside `memory/` and keep the knowledge-base workstream as the compact cross-project summary. | The port will accumulate capability-level experiments, fixtures, notebooks, and parity decisions that need local traceability. | Project-local notes own detailed progress; the knowledge base tracks durable state and next actions. | Revisit if the project splits into separate data, model, or hardware repos. |
| 2026-05-31 | Use a light default package dependency set (`numpy`, `blake3`) and move heavier research libraries to optional extras. | Core contract and parity tests should run quickly in a fresh environment; PyTorch/Jupyter/FastAPI are not needed for every milestone. | `.venv/bin/python -m pytest -q` can verify Milestone 1 without installing the full research stack. | Revisit when notebooks or server milestones become the active work. |
| 2026-05-31 | Deep-copy `CsiMetadata` inside `CsiFrame` construction. | Rust `CsiFrame::new` takes ownership of metadata; Python references would otherwise let external metadata mutation alter witness hashes. | Frame canonical bytes are stable unless the frame's own metadata is intentionally changed. | Revisit if immutable/frozen metadata models replace mutable dataclasses. |
| 2026-05-31 | Use `uv` for local environment management. | The system Python is externally managed and the user requested a `uv` initialized environment. | `.venv` is recreated by `uv sync --extra dev`, `uv.lock` records resolved dependencies, and tests run via `uv run pytest -q`. | Revisit if the project later adopts dependency groups or a different Python version pin. |

## Notes

-
