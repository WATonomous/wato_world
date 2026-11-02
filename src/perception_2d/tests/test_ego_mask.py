"""Tests for ego_mask: the body test, and the draft block, which must load as an
ego_mask config and rasterize back over the body."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from wato_perception_2d.config import ComponentConfig
from wato_perception_2d.ego_mask import draft_polygons, format_block, is_body
from wato_perception_2d.pipeline import _ego_mask


def _rack(shift: int = 0) -> np.ndarray:
    m = np.zeros((64, 80), dtype=bool)
    m[50:, 10 + shift : 70 + shift] = True  # along the bottom edge
    return m


@pytest.mark.parametrize(
    "masks, span, travel_m, body",
    [
        ([_rack()] * 8, 0.9, 200.0, True),
        ([_rack()] * 8, 0.9, 5.0, False),  # parked: everything holds still
        ([_rack()] * 8, 0.3, 200.0, False),  # a car alongside for part of the chunk
        ([_rack(), _rack(25)] * 4, 0.9, 200.0, False),  # moves across the image
        ([np.roll(_rack(), -20, axis=0)] * 8, 0.9, 200.0, False),  # off the border
    ],
)
def test_body_is_a_still_mask_at_the_border_while_driving(masks, span, travel_m, body):
    assert is_body(masks, span=span, travel_m=travel_m) == body


def test_draft_block_loads_and_covers_the_body_with_its_margin():
    pytest.importorskip("cv2")
    body = np.zeros((64, 80), dtype=bool)
    body[50:, 10:70] = True
    body[40:50, 36:40] = True  # an antenna above the rack
    block = format_block({"CAM_B": draft_polygons(body, 3)}, (80, 64), 0.5)
    cfg = ComponentConfig(**yaml.safe_load(block))
    drawn = _ego_mask(cfg, "CAM_B", 80, 64)
    assert not (body & ~drawn).any()
    assert drawn.sum() < 1.4 * body.sum()
    assert _ego_mask(cfg, "CAM_A", 80, 64) is None
    with pytest.raises(ValueError, match="80x64"):
        _ego_mask(cfg, "CAM_B", 160, 128)
