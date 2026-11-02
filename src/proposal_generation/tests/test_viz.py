"""Each viewer writes its files over the staged chunk."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from wato_proposal_generation.config import ComponentConfig, FramesConfig
from wato_proposal_generation.pipeline import process_chunk
from wato_proposal_generation.viz.bev import write_bev
from wato_proposal_generation.viz.browser import write_browser
from wato_proposal_generation.viz.camera import write_camera

from ._staging import BAG, CHUNK, N_FRAMES, stage_chunk

CFG = ComponentConfig(frames=FramesConfig(canonical_lidar="lidar_cc"))


@pytest.fixture(autouse=True)
def staged(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")
    stage_chunk(with_camera=True)
    process_chunk(CFG, BAG, CHUNK)


@pytest.mark.parametrize("show", ["sources", "fused"])
def test_browser_page_carries_points_and_boxes(show):
    page = write_browser(CFG, BAG, CHUNK, show=show, every=1)
    html = page.read_text(encoding="utf-8")
    assert '"boxes"' in html and "lineProgram" in html


def test_bev_writes_one_png_per_shown_frame():
    files = write_bev(CFG, BAG, CHUNK, every=1, size_px=200)
    assert len(files) == N_FRAMES
    img = np.asarray(Image.open(files[0]))
    assert img.shape == (200, 200, 3)
    # lidar_mos boxes are drawn in their source colour.
    assert (img == (255, 140, 0)).all(axis=2).any()


def test_camera_overlay_draws_the_box_in_front_of_the_camera():
    files = write_camera(CFG, BAG, CHUNK, every=1)
    assert len(files) == N_FRAMES
    img = np.asarray(Image.open(files[0]).convert("RGB")).astype(int)
    # Car A at x = 10 is in view: something other than the grey backdrop.
    assert (np.abs(img - 40).sum(axis=2) > 120).any()
