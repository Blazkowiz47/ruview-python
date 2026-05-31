"""Lightweight checks that committed notebooks remain valid JSON."""

from __future__ import annotations

import json
from pathlib import Path


VISUAL_LAB_NOTEBOOKS = {
    "00_signal_playground.ipynb",
    "01_empty_room_vs_person_present.ipynb",
    "02_motion_vs_stillness.ipynb",
    "03_breathing_and_heart_rate_bands.ipynb",
    "05_subcarrier_heatmaps.ipynb",
    "06_calibration_baseline_drift.ipynb",
    "07_multistatic_node_comparison.ipynb",
    "08_csi_to_pose_experiment.ipynb",
    "09_dataset_replay_lab.ipynb",
    "10_model_embedding_visualization.ipynb",
    "11_ruvector_signal_geometry.ipynb",
    "12_mat_research_pipeline.ipynb",
    "13_worldgraph_privacy_provenance.ipynb",
    "14_optional_tracks_research_overview.ipynb",
}

VISUAL_LAB_PHRASES = (
    "Purpose:",
    "Run path:",
    "Fixture / simulated source:",
    "Expected interpretation:",
    "Limitations:",
)


def _cell_source(cell: dict[str, object]) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    return str(source)


def test_notebooks_are_valid_json() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    notebooks = sorted((repo_root / "notebooks").glob("*.ipynb"))

    assert notebooks, "expected at least one notebook"

    for notebook in notebooks:
        data = json.loads(notebook.read_text(encoding="utf-8"))
        assert data["nbformat"] == 4
        assert isinstance(data["cells"], list), notebook
        for cell in data["cells"]:
            if cell.get("cell_type") == "code":
                assert cell.get("execution_count") is None, notebook
                assert cell.get("outputs") == [], notebook


def test_visual_lab_notebooks_have_expected_scaffolding() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    for name in VISUAL_LAB_NOTEBOOKS:
        notebook = repo_root / "notebooks" / name
        data = json.loads(notebook.read_text(encoding="utf-8"))
        cells = data["cells"]
        text = "\n".join(_cell_source(cell) for cell in cells)

        assert any(cell.get("cell_type") == "code" for cell in cells), notebook
        assert "matplotlib" in text, notebook
        assert "try:" in text and "except Exception" in text, notebook

        for phrase in VISUAL_LAB_PHRASES:
            assert phrase in text, notebook
