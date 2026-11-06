"""The shipped configs: perception_2d.wato.yaml is perception_2d.yaml plus the
WATO rig's ego mask."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from wato_perception_2d.config import (
    MaskRefinementConfig,
    SegmentationConfig,
    load_config,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def test_wato_config_is_the_default_plus_the_ego_mask():
    base = load_config(str(CONFIG_DIR / "perception_2d.yaml")).model_dump()
    wato = load_config(str(CONFIG_DIR / "perception_2d.wato.yaml")).model_dump()
    assert not base["ego_mask"]["polygons"]
    assert wato["ego_mask"]["polygons"]
    del base["ego_mask"], wato["ego_mask"]
    assert wato == base


def test_invalid_correction_range_is_rejected():
    with pytest.raises(ValidationError, match="must be below"):
        MaskRefinementConfig(correction_min_mask_iou=0.9, correction_below_mask_iou=0.5)


def test_mask_backend_is_explicit_and_validated():
    assert SegmentationConfig().backend == "dam4sam"
    assert SegmentationConfig(backend="sam2").backend == "sam2"
    with pytest.raises(ValidationError):
        SegmentationConfig(backend="dam4san")
