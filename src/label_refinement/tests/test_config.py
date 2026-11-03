"""The shipped configs load, and the WATO one differs only where the rig does."""

from __future__ import annotations

from pathlib import Path

from wato_label_refinement.config import ComponentConfig, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def test_shipped_configs_load():
    for name in ("label_refinement.yaml", "label_refinement.wato.yaml"):
        load_config(str(CONFIG_DIR / name))


def test_wato_profile_is_the_default_with_the_rigs_canonical_lidar():
    default = load_config(str(CONFIG_DIR / "label_refinement.yaml"))
    wato = load_config(str(CONFIG_DIR / "label_refinement.wato.yaml"))
    assert wato.frames.canonical_lidar == "lidar_cc"
    assert wato.model_copy(update={"frames": default.frames}) == default
    assert default == ComponentConfig()
