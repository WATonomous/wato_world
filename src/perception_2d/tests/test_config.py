"""The shipped configs: perception_2d.wato.yaml is perception_2d.yaml plus the
WATO rig's ego mask."""

from __future__ import annotations

from pathlib import Path

from wato_perception_2d.config import load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def test_wato_config_is_the_default_plus_the_ego_mask():
    base = load_config(str(CONFIG_DIR / "perception_2d.yaml")).model_dump()
    wato = load_config(str(CONFIG_DIR / "perception_2d.wato.yaml")).model_dump()
    assert not base["ego_mask"]["polygons"]
    assert wato["ego_mask"]["polygons"]
    del base["ego_mask"], wato["ego_mask"]
    assert wato == base
