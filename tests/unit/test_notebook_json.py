"""Lightweight checks that committed notebooks remain valid JSON."""

from __future__ import annotations

import json
from pathlib import Path


def test_notebooks_are_valid_json() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    notebooks = sorted((repo_root / "notebooks").glob("*.ipynb"))

    assert notebooks, "expected at least one notebook"

    for notebook in notebooks:
        data = json.loads(notebook.read_text(encoding="utf-8"))
        assert data["nbformat"] == 4
        assert isinstance(data["cells"], list), notebook
