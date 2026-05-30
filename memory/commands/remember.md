# Command: /remember

Add a compact note to today's project memory.

## Steps

1. Identify today's local date and the active node.
2. If the stable `<node>` name is not known, ask the user for it before creating or writing a new node-specific note.
3. Prefer `memory/notes/YYYY-MM-DD-<node>.md` for new notes.
4. Continue writing `memory/notes/YYYY-MM-DD.md` only when that legacy file is already the active note for the target date or the user explicitly asks to keep the legacy convention.
5. Treat `YYYY-MM-DD.md` as the legacy/default-node note and `YYYY-MM-DD-<node>.md` as an explicit-node note.
6. Append the note under the most relevant section, such as `Work Done`, `Analysis Results`, `Learnings`, `Blockers`, or `Next`.
7. If the note is durable, also update `memory/learnings.md`, `memory/decisions.md`, `memory/runs.md`, or `memory/index.md` as appropriate.
8. Keep the note compact. Link to evidence paths instead of pasting bulky output.
