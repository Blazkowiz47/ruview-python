# Command: /organise-scratch

Route project-local scratch notes into the right memory files.

## Steps

1. Read `memory/scratch/index.md` and the relevant scratch files.
2. For each scratch file, decide whether it should be promoted, kept open, or deleted.
3. Promote durable outcomes into the appropriate place:
   - dated work into today's note under `memory/notes/`
   - experiment or evaluation state into `memory/runs.md`
   - durable findings into `memory/learnings.md`
   - decisions into `memory/decisions.md`
   - project status, blocker, latest useful result, or next action into `memory/index.md`
4. If a scratch file remains open, update its next action and keep it compact.
5. If a scratch file is fully routed, delete it or mark it closed according to the project's preference.
6. Do not stage, commit, or push unless the user explicitly asks from this project repository.
