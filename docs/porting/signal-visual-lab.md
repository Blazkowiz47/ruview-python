# Signal Visual Lab

Milestone 3 turns the first notebook placeholders into runnable visual fixtures for CSI signal exploration.

## Notebooks

- `notebooks/00_signal_playground.ipynb` creates one deterministic complex CSI frame, plots stream-wise amplitude and phase, and wraps the array in `ruview.core.CsiFrame` when that import is available.
- `notebooks/01_empty_room_vs_person_present.ipynb` compares synthetic empty-room and person-present captures, with a guarded hook for future simulator fixtures.
- `notebooks/05_subcarrier_heatmaps.ipynb` renders amplitude, phase, and rolling-variance heatmaps from an inline synthetic capture, with a guarded hook for future visualization helpers.

## Run Path

Install the optional research dependencies, then run all cells in the notebooks:

```bash
uv sync --extra research
uv run jupyter lab notebooks
```

The notebooks are intentionally unexecuted in Git. They keep deterministic inline fallbacks so the lab stays readable before hardware replay, simulator, or signal visualization APIs land.

## Limitations

The fallback fixtures are visual debugging aids only. They do not model calibrated RF channels, ESP32 packet timing, missing packets, antenna geometry, carrier-frequency offset, or real-room movement.
