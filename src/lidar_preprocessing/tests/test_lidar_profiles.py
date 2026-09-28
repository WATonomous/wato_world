from dataclasses import replace

import pytest
from pydantic import ValidationError

from wato_lidar_preprocessing import sensor_model as sensor_models
from wato_lidar_preprocessing.config import (
    ComponentConfig,
    LidarProfile,
    MFMosProfile,
    SensorModelParams,
)


def test_profile_for_uses_default_and_sensor_override():
    default = LidarProfile(sensor_model=SensorModelParams(profile="velodyne_vlp"))
    top = LidarProfile(
        sensor_model=SensorModelParams(profile="nuscenes"),
        lidar_sweep_duration_ms=50.0,
        point_time_unit="microseconds",
        patchwork_sensor_height=2.1,
        mf_mos=MFMosProfile(
            enabled=True,
            fusion_mode="mfmos_only",
            range_image_h=32,
            range_image_w=2048,
            fov_up_deg=12.0,
            fov_down_deg=-28.0,
        ),
    )
    cfg = ComponentConfig(
        default_lidar_profile=default,
        lidar_profiles={"LIDAR_TOP": top},
    )

    assert cfg.profile_for("LIDAR_LEFT") == default
    assert cfg.profile_for("LIDAR_TOP") == top
    assert cfg.profile_for("LIDAR_TOP").point_time_scale_to_ns() == 1e3
    assert cfg.patchwork_for("LIDAR_TOP").sensor_height == 2.1


def test_profile_for_rejects_unknown_sensor_without_default():
    cfg = ComponentConfig(lidar_profiles={"LIDAR_TOP": LidarProfile()})

    with pytest.raises(ValueError, match="unknown lidar_id.*LIDAR_LEFT"):
        cfg.profile_for("LIDAR_LEFT")


@pytest.mark.parametrize("lidar_id", ["../escape", "lidar/left", "", "left lidar"])
def test_lidar_profile_keys_must_be_safe(lidar_id):
    with pytest.raises(ValidationError, match="lidar_id"):
        ComponentConfig(lidar_profiles={lidar_id: LidarProfile()})


def test_profiles_on_one_aw_grid_must_share_inverse_occupancy_contract(monkeypatch):
    incompatible = replace(sensor_models.get_sensor_model("nuscenes"), p_hit=0.80)
    monkeypatch.setitem(sensor_models._PROFILES, "incompatible_test", incompatible)

    with pytest.raises(ValidationError, match="inverse-occupancy"):
        ComponentConfig(
            lidar_profiles={
                "LIDAR_LEFT": LidarProfile(
                    sensor_model=SensorModelParams(profile="velodyne_vlp")
                ),
                "LIDAR_RIGHT": LidarProfile(
                    sensor_model=SensorModelParams(profile="incompatible_test")
                ),
            }
        )


def test_legacy_global_sensor_settings_remain_the_implicit_default_profile():
    cfg = ComponentConfig(
        sensor_model=SensorModelParams(profile="nuscenes"),
        lidar_sweep_duration_ms=50.0,
        point_time_unit="nanoseconds",
    )

    profile = cfg.profile_for("LIDAR_TOP")

    assert profile.sensor_model.profile == "nuscenes"
    assert profile.lidar_sweep_duration_ms == 50.0
    assert profile.point_time_unit == "nanoseconds"
