"""Tests for viz.py — the lifted-label browser viewer."""

from __future__ import annotations

import json
import re

import pytest

from wato_semantic_lifting.config import ComponentConfig
from wato_semantic_lifting.pipeline import run
from wato_semantic_lifting.viz import write_html_viewer

from .test_pipeline import BAG, CHUNK, _stage_chunk


@pytest.fixture(autouse=True)
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")


def _payload(html: str) -> dict:
    return json.loads(re.search(r"const PAYLOAD = (\{.*?\});\n", html).group(1))


def test_viewer_shows_labeled_points_over_unlabeled_context(tmp_path):
    _stage_chunk(cams=(("CAM_A", 0.6), ("CAM_B", 0.9)))
    run(ComponentConfig(), bag_id=BAG)

    out = write_html_viewer(BAG, CHUNK, out_path=tmp_path)
    assert out.name == "lifted_labels_instance.html"
    payload = _payload(out.read_text(encoding="utf-8"))
    assert payload["counts"]["labeled"] == 3
    assert payload["counts"]["context"] == 2  # points 3 and 4
    assert payload["counts"]["distinct"] == 1  # one instance
    assert payload["static_label"] == "unlabeled"

    by_class = write_html_viewer(BAG, CHUNK, color_by="class", out_path=tmp_path)
    assert by_class.name == "lifted_labels_class.html"


def test_viewer_needs_lifted_labels(tmp_path):
    _stage_chunk()
    with pytest.raises(FileNotFoundError, match="run semantic_lifting first"):
        write_html_viewer(BAG, CHUNK, out_path=tmp_path)
