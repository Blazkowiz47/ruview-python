# 2026-05-31 - sushruts-macbook-pro - Milestone 13 HOMECORE

- Node: sushruts-macbook-pro
- Device/server: local macOS workspace
- Repo path: `ruview-python`
- Branch: `master`
- Commit: pending at note write for `Add HOMECORE research primitives`

## Log

- Ported a small deterministic Python HOMECORE research subset from the Rust state/event/service and automation crates.
- Added ASCII `domain.name` `EntityId` validation, immutable `State` snapshots, `last_changed`/`last_updated` semantics, no-op event suppression, domain filtering, removals, and synchronous in-memory event logs/subscribers.
- Added service call/domain event context records plus deterministic automation triggers, conditions, actions, and local callback/service recording.
- Ignored unrelated concurrent swarm worktree changes outside this milestone's owned files.

## Verification

- Passed: `uv run pytest -q tests/unit/test_homecore_research.py` (`6 passed in 0.16s`)

## Next

- Commit owned HOMECORE files and this note, then report the resulting commit hash.
