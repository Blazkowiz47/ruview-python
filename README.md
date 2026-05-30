# RuView Python

RuView Python is a pure Python research port of
[ruvnet/RuView](https://github.com/ruvnet/RuView), the RuView /
WiFi-DensePose project.

The upstream RuView project explores camera-free spatial sensing with commodity
WiFi: Channel State Information (CSI) from low-cost ESP32-style nodes is used to
study presence, occupancy, movement, room state, breathing, heart-rate signals,
and related radio-frequency perception tasks. This repository takes that broad
system and ports it capability by capability into a readable, experiment-first
Python codebase.

```text
/Users/sushrutpatwardhan/1Projects/RuView
```

This is not a product rewrite. It favors clear NumPy/SciPy/PyTorch
implementations, deterministic fixtures, parity tests, and notebooks over app
store flows, cloud distribution, smart-home polish, or commercial packaging. See
[plan.md](plan.md) for the capability-by-capability porting roadmap.

## Project Goals

- Preserve important RuView / WiFi-DensePose data contracts and wire formats.
- Reimplement core signal-processing and research logic in idiomatic Python.
- Make every major stage inspectable through tests, fixtures, or notebooks.
- Keep firmware-specific behavior as host-side parsers, replay tools, and
  simulators rather than replacing ESP-IDF capture code.
- Record intentional deviations from the upstream Rust/C/Python reference as
  the port grows.

## Current Slice

- Milestone 0 scaffold is present.
- Milestone 1 has an initial core-contract port for CSI frames, confidence,
  timestamps, IDs, pose keypoints, canonical frame bytes, and BLAKE3 witness
  hashes.
- Development environments are managed with `uv`.

## Layout

```text
src/ruview/core/        Core CSI, pose, timestamp, confidence, and witness types
src/ruview/protocols/   Host-side protocol parsers, planned
src/ruview/signal/      Signal-processing modules, planned
src/ruview/vitals/      Breathing and heart-rate research modules, planned
notebooks/              Reproducible visual inspection notebooks
tests/                  Unit and parity tests
examples/               Small runnable experiments
data/                   Fixtures, synthetic data, and recordings
```

## Quick Check

```bash
uv sync --extra dev
uv run pytest
```

## Acknowledgements

This project is based on and deeply indebted to
[ruvnet/RuView](https://github.com/ruvnet/RuView), created by
[ruvnet](https://github.com/ruvnet) / rUv. RuView provides the original system
architecture, research direction, data-contract ideas, firmware references, and
many of the algorithms and module boundaries this Python port is studying.

The upstream repository is distributed under the MIT License with copyright
notice `Copyright (c) 2024 rUv`. Any code or behavior ported from RuView should
preserve applicable license notices and make the source mapping explicit in the
relevant module, test, notebook, or documentation.

RuView Python is an independent research port in this local workspace; it is not
the upstream RuView repository and should not be presented as an official
replacement for it.
