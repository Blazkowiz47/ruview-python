# Command: /check-initialisation

Verify and align this project's memory structure.

## Steps

1. Confirm required memory files exist:
   - `memory/index.md`
   - `memory/devices.md`
   - `memory/runs.md`
   - `memory/learnings.md`
   - `memory/decisions.md`
   - `memory/notes/`
   - `memory/commands/`
2. Confirm `AGENTS.md` contains exactly one Sushrut memory block.
3. Confirm `CLAUDE.md` is a relative symlink whose target is exactly `AGENTS.md`, unless the user explicitly asked to skip it.
4. Confirm project command specs exist for `/remember`, `/log`, `/run`, `/decision`, `/learned`, `/status`, and `/check-initialisation`.
5. Align missing memory infrastructure when it is safe to do so, preserving all existing project instructions and memory content.
6. If existing instructions conflict or preservation is ambiguous, stop and ask the user.
7. Do not stage, commit, or push unless the user explicitly asks from this project repository.

## Report

Report whether initialization is correct, what was aligned, and any remaining issue.
