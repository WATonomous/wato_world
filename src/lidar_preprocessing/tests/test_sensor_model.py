"""Tests for the sensor model — the single source of every derived constant.

These guard the property the whole config reduction rests on: picking a
scanner is enough. If a profile can be selected but yields nonsense geometry,
or if two profiles disagree on the decision rule, then the YAML would have to
compensate — which is exactly the situation this replaced.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from wato_lidar_preprocessing.config import ComponentConfig
from wato_lidar_preprocessing.mf_mos._core import (
    KITTI_SWEEP_RATE_HZ,
    residual_steps_for,
)
from wato_lidar_preprocessing.sensor_model import (
    available_profiles,
    get_sensor_model,
)


@pytest.mark.parametrize("profile", available_profiles())
def test_profile_datasheet_fields_are_physical(profile):
    """Every profile states a usable scanner, not a placeholder."""
    sm = get_sensor_model(profile)
    assert sm.beams > 0
    assert sm.fov_up_deg > sm.fov_down_deg
    assert sm.range_sigma_m > 0
    assert sm.beam_divergence_rad > 0
    assert sm.max_range_m > 0
    assert sm.sweep_duration_ms > 0
    assert sm.intensity_scale > 0
    assert sm.rotation_dir in ("cw", "ccw")


@pytest.mark.parametrize("profile", available_profiles())
def test_derived_log_odds_constants_are_ordered(profile):
    """An endpoint hit must outweigh a single pass-through, and both must sit
    inside the clamp — otherwise one observation could saturate a voxel."""
    sm = get_sensor_model(profile)
    assert sm.l_occ > sm.l_free > 0
    assert sm.log_odds_clamp > sm.l_occ
    assert 0 < sm.p_dynamic_threshold < 0.5 < sm.p_static_threshold < 1
    assert sm.p_static_threshold + sm.p_dynamic_threshold == pytest.approx(1.0)


def test_decision_rule_is_the_same_across_profiles():
    """Profiles differ in geometry, never in what counts as static or dynamic.

    classify reads the chunk-level thresholds from the rig's default profile,
    so a mixed rig must not end up judging one scanner's returns by a
    different rule than another's.
    """
    models = [get_sensor_model(p) for p in available_profiles()]
    assert len({m.p_static_threshold for m in models}) == 1
    assert len({m.p_dynamic_threshold for m in models}) == 1
    assert len({m.l_occ for m in models}) == 1
    assert len({m.l_free for m in models}) == 1


def test_unknown_profile_names_the_valid_ones():
    with pytest.raises(ValueError, match="valid:"):
        get_sensor_model("velodyne_vlp")  # the pre-split name


def test_per_lidar_profiles_resolve_and_fall_back():
    cfg = ComponentConfig(
        sensor_model={
            "profile": "vlp32c",
            "per_lidar": {"lidar_ne": "vlp16", "lidar_nw": "vlp16"},
        }
    )
    assert cfg.build_sensor_model("lidar_ne").beams == 16
    assert cfg.build_sensor_model("lidar_nw").beams == 16
    # Unlisted lidars — and the chunk-level call with no lidar at all — take
    # the rig default.
    assert cfg.build_sensor_model("lidar_cc").beams == 32
    assert cfg.build_sensor_model().name == "vlp32c"


def test_per_lidar_unknown_profile_is_rejected_by_name():
    with pytest.raises(ValidationError, match="lidar_ne"):
        ComponentConfig(sensor_model={"per_lidar": {"lidar_ne": "vlp99"}})


def test_residual_steps_scale_with_spin_rate():
    """Residual channel k means "k frames back". A 20 Hz scanner must step two
    frames to span the wall-clock motion KITTI's 10 Hz model was trained on."""
    sm = get_sensor_model("vlp32c")
    assert sm.sweep_rate_hz == pytest.approx(2 * KITTI_SWEEP_RATE_HZ)
    assert residual_steps_for(sm, 8) == [2, 4, 6, 8, 10, 12, 14, 16]


def test_residual_steps_never_collapse_to_zero():
    """A scanner slower than KITTI still gets consecutive-frame offsets rather
    than a degenerate all-zero window."""

    slow = get_sensor_model("vlp32c").__class__(
        **{
            **get_sensor_model("vlp32c").__dict__,
            "name": "slow",
            "sweep_duration_ms": 200.0,  # 5 Hz
        }
    )
    assert residual_steps_for(slow, 4) == [1, 2, 3, 4]
