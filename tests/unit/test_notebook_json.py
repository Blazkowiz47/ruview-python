"""Lightweight checks that committed notebooks remain valid JSON."""

from __future__ import annotations

import json
from pathlib import Path


VISUAL_LAB_NOTEBOOKS = {
    "00_end_to_end_capture_visualization_prediction.ipynb",
    "01_signal_amplitude_phase_visualization.ipynb",
    "02_signal_subcarrier_heatmaps.ipynb",
    "03_person_presence_detection.ipynb",
    "04_motion_detection.ipynb",
    "05_breathing_heart_rate_detection.ipynb",
    "06_pose_estimation.ipynb",
    "07_recording_presence_breathing_triage.ipynb",
    "08_position_triangulation_three_receivers.ipynb",
}

SCOPE_NOTEBOOKS = {
    "end-to-end demo": "00_end_to_end_capture_visualization_prediction.ipynb",
    "signal visualization": "01_signal_amplitude_phase_visualization.ipynb",
    "person detection": "03_person_presence_detection.ipynb",
    "motion detection": "04_motion_detection.ipynb",
    "breathing rate": "05_breathing_heart_rate_detection.ipynb",
    "heart rate": "05_breathing_heart_rate_detection.ipynb",
    "pose estimation": "06_pose_estimation.ipynb",
    "position detection": "08_position_triangulation_three_receivers.ipynb",
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


def test_notebook_scope_readme_covers_active_scope() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    readme = repo_root / "notebooks" / "README.md"
    text = readme.read_text(encoding="utf-8").lower()

    for scope, notebook_name in SCOPE_NOTEBOOKS.items():
        assert scope in text
        assert notebook_name in text


def test_notebooks_expose_recording_source_switch() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    notebooks_dir = repo_root / "notebooks"

    for notebook_name in VISUAL_LAB_NOTEBOOKS:
        text = (notebooks_dir / notebook_name).read_text(encoding="utf-8")
        assert "USE_RECORDING" in text, notebook_name
        if notebook_name == "08_position_triangulation_three_receivers.ipynb":
            assert "RECEIVER_RECORDINGS" in text, notebook_name
        else:
            assert "RECORDING_CSV" in text, notebook_name
