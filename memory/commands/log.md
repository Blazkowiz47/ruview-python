# Command: /log

Record session work or a meaningful project change.

## Steps

1. Identify today's local date and active node.
2. If the stable `<node>` name is not known, ask the user for it before creating or writing a new node-specific note.
3. Prefer `memory/notes/YYYY-MM-DD-<node>.md` for new notes.
4. Continue writing `memory/notes/YYYY-MM-DD.md` only when that legacy file is already the active note for the target date or the user explicitly asks to keep the legacy convention.
5. Record the work done, why it mattered, and the next action.
6. Include branch, commit, command/config, dataset, output path, and node/device/server when relevant.
7. If project status, blocker, latest useful result, or next action changed, update `memory/index.md`.
8. Keep source-code details brief; link to files, commits, or outputs when needed.
