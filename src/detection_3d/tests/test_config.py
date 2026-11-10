from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from wato_detection_3d.config import MODEL_FILES, ComponentConfig, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def test_shipped_configs_load():
    for path in CONFIG_DIR.glob("*.yaml"):
        load_config(str(path))


def test_wato_profile_is_the_default_with_the_rigs_lidar_frame_and_cameras():
    """lidar_preprocessing.wato.yaml groups frames on lidar_cc; WATO has no
    LIDAR_TOP, so the model's frame is rebuilt; its six cameras are the car's.
    Nothing else may drift from the default profile."""
    default = load_config(str(CONFIG_DIR / "detection_3d.yaml")).model_dump()
    wato = load_config(str(CONFIG_DIR / "detection_3d.wato.yaml")).model_dump()
    assert wato["frames"]["canonical_lidar"] == "lidar_cc"
    assert wato["input_frame"]["kind"] == "nuscenes_lidar"
    assert default["input_frame"]["kind"] == "sensor"
    assert wato["cameras"] == [
        "PANO_NN",
        "PANO_NE",
        "PANO_NW",
        "PANO_SS",
        "PANO_SW",
        "PANO_SE",
    ]
    for key in ("frames", "input_frame", "cameras"):
        wato[key] = default[key]
    assert wato == default


def test_every_model_file_is_pinned():
    with pytest.raises(ValidationError, match="must pin"):
        ComponentConfig.model_validate(
            {"model": {"sha256": {MODEL_FILES[0]: "0" * 64}}, "cameras": list("abcdef")}
        )


def test_the_model_takes_six_distinct_cameras():
    model = {"sha256": {f: "0" * 64 for f in MODEL_FILES}}
    for cameras in (list("abcde"), list("abcdea")):
        with pytest.raises(ValidationError, match="6 distinct"):
            ComponentConfig.model_validate({"model": model, "cameras": cameras})
