from __future__ import annotations

from pathlib import Path

from wato_proposal_generation.config import load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def test_shipped_configs_load():
    for path in CONFIG_DIR.glob("*.yaml"):
        load_config(str(path))


def test_wato_profile_is_the_default_with_the_rigs_canonical_lidar():
    """lidar_preprocessing.wato.yaml groups frames on lidar_cc; nothing else
    may drift from the default profile."""
    default = load_config(str(CONFIG_DIR / "proposal_generation.yaml")).model_dump()
    wato = load_config(str(CONFIG_DIR / "proposal_generation.wato.yaml")).model_dump()
    assert wato["frames"]["canonical_lidar"] == "lidar_cc"
    assert default["frames"]["canonical_lidar"] is None
    wato["frames"]["canonical_lidar"] = None
    assert wato == default
