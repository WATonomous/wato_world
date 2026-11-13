from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from wato_proposal_generation.config import (
    LiftedConfig,
    ScanRefinementConfig,
    load_config,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def test_shipped_configs_load():
    for path in CONFIG_DIR.glob("*.yaml"):
        load_config(str(path))


def test_wato_profile_is_the_default_with_the_rigs_canonical_lidar():
    """WATO uses the centre LiDAR and enables supplemental BEVFusion proposals."""
    default = load_config(str(CONFIG_DIR / "proposal_generation.yaml")).model_dump()
    wato = load_config(str(CONFIG_DIR / "proposal_generation.wato.yaml")).model_dump()
    assert wato["frames"]["canonical_lidar"] == "lidar_cc"
    assert default["frames"]["canonical_lidar"] is None
    wato["frames"]["canonical_lidar"] = None
    assert wato["fusion_detector"]["enabled"] is True
    assert default["fusion_detector"]["enabled"] is False
    wato["fusion_detector"]["enabled"] = False
    assert wato == default


@pytest.mark.parametrize(
    "changes",
    [
        {"radii_m": ()},
        {"radii_m": (0,)},
        {"radii_m": (-1,)},
        {"radii_m": (float("nan"),)},
        {"radii_m": (float("inf"),)},
        {"min_samples": 0},
        {"min_iou": -0.1},
        {"min_iou": 1.1},
    ],
)
def test_invalid_refinement_settings_fail(changes):
    with pytest.raises(ValidationError):
        ScanRefinementConfig(**changes)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_coverage_thresholds_fail(threshold):
    with pytest.raises(ValidationError):
        LiftedConfig(max_support_iom=threshold)
