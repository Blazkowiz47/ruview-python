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
RuView
```

This is not a product rewrite. It favors clear NumPy/SciPy/PyTorch
implementations, deterministic fixtures, parity tests, and notebooks over app
store flows, cloud distribution, smart-home polish, or commercial packaging.
See [plan.md](plan.md) for the completed capability-by-capability porting
roadmap.

## Project Goals

- Preserve important RuView / WiFi-DensePose data contracts and wire formats.
- Reimplement core signal-processing and research logic in idiomatic Python.
- Make every major stage inspectable through tests, fixtures, or notebooks.
- Keep firmware-specific behavior as host-side parsers, replay tools, and
  simulators rather than replacing ESP-IDF capture code.
- Record intentional deviations from the upstream Rust/C/Python reference as
  the port grows.

## Current Status

All milestones in [plan.md](plan.md), Milestone 0 through Milestone 13, are
implemented. The repository now includes:

- core CSI, pose, timestamp, confidence, canonical-frame, and witness types
- ESP32 packet parsers, UDP replay helpers, and host-side hardware simulators
- signal-processing, presence, motion, calibration, RuvSense, and RuVector
  research primitives
- breathing and heart-rate research estimators
- local FastAPI sensing-server research app and replay/simulation examples
- neural/training research helpers with optional PyTorch model modules
- MAT, WorldGraph, trust/provenance, privacy/BFLD, HOMECORE, nvsim, swarm,
  browser-visualization, and desktop hardware-tooling research subsets
- deterministic unit/parity tests and runnable visual notebooks

The notebooks currently use deterministic synthetic or dummy fixtures. They are
intended for inspection, shape checks, and workflow smoke tests before replacing
the fixtures with recorded CSI captures.

## Layout

```text
src/ruview/core/                  Core CSI, pose, timestamp, confidence, and witness types
src/ruview/protocols/             ESP32, sync, RVF, and witness protocol helpers
src/ruview/hardware/              UDP replay, synthetic CSI, desktop helper, and simulator tools
src/ruview/signal/                CSI processing, phase, motion, presence, and feature helpers
src/ruview/vitals/                Breathing and heart-rate research estimators
src/ruview/ruvsense/              Advanced RuvSense signal, fusion, field, and temporal primitives
src/ruview/ruvector/              RuVector-style signal, geometry, and history helpers
src/ruview/server/                Local research sensing server app and sources
src/ruview/nn/                    Optional neural model modules
src/ruview/training/              Dataset, metric, checkpoint, export, and trainer helpers
src/ruview/mat/                   Mass casualty assessment research pipeline primitives
src/ruview/worldgraph/            Graph, provenance, and trust helpers
src/ruview/privacy/               BFLD, privacy mode, identity risk, and privacy gate helpers
src/ruview/homecore/              Local state-machine and automation research subset
src/ruview/nvsim/                 Deterministic magnetic-scene simulator
src/ruview/swarm/                 Simulation-only swarm topology, planning, and sensing models
notebooks/                        Reproducible visual inspection notebooks with synthetic fixtures
tests/                            Unit and parity tests
examples/                         Small runnable experiments
data/                             Fixtures, synthetic data, and recordings
```

## Quick Check

```bash
uv sync --extra dev
uv run pytest -q
```

Validate notebook JSON and scaffolding with:

```bash
uv run pytest -q tests/unit/test_notebook_json.py
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
