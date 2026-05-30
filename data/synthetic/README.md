# Synthetic CSI Fixtures

This directory is reserved for small synthetic CSI fixture metadata and generated
`.npz` files used during local experiments.

The committed source of truth is the deterministic simulator in
`src/ruview/hardware/simulator.py`; large binary CSI fixtures should not be
committed. To generate a compact local fixture, use
`save_synthetic_fixture(...)` with frames from `generate_synthetic_sequence(...)`.

Supported scenarios:

- `empty_room`
- `person_present`
- `walking` (also accepts `motion`)
- `stillness`
