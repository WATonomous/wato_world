"""The model's input frame and the vendor box convention (geometry.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from wato_detection_3d import geometry
from wato_detection_3d.geometry import WorldBox


def _ego_T_lidar(x=0.9, y=0.1, z=2.11, yaw=0.1):
    m = np.eye(4)
    c, s = math.cos(yaw), math.sin(yaw)
    m[:2, :2] = [[c, -s], [s, c]]
    m[:3, 3] = (x, y, z)
    return m


def test_nuscenes_lidar_frame_is_x_right_y_forward_at_training_height():
    T = geometry.ego_T_model("nuscenes_lidar", _ego_T_lidar(), 1.84)
    # model +y is ego forward, model +x is ego right (-y), z up.
    np.testing.assert_allclose(T[:3, :3] @ [0, 1, 0], [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(T[:3, :3] @ [1, 0, 0], [0, -1, 0], atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], [0.9, 0.1, 1.84])


def test_sensor_frame_is_the_lidar_as_the_car_runs():
    np.testing.assert_array_equal(
        geometry.ego_T_model("sensor", _ego_T_lidar(), 1.84), _ego_T_lidar()
    )
    with pytest.raises(ValueError):
        geometry.ego_T_model("bev", _ego_T_lidar(), 1.84)


def test_vendor_box_facing_forward():
    """mmdetection3d v0.x: a box facing the LiDAR frame's +y has rot = -pi
    (nuScenes yaw pi/2 = -rot - pi/2); z is its centre."""
    b = geometry.to_world(np.eye(4), 1.0, 2.0, -1.0, 1.9, 4.5, 1.6, -math.pi, 0.0, 3.0)
    assert b.heading == pytest.approx(math.pi / 2)
    assert (b.cx, b.cy, b.cz) == pytest.approx((1.0, 2.0, -1.0))
    assert (b.w, b.l, b.h) == (1.9, 4.5, 1.6)
    assert (b.vx, b.vy) == pytest.approx((0.0, 3.0))


def test_world_heading_follows_the_frame():
    ego = geometry.ego_T_model("nuscenes_lidar", _ego_T_lidar(), 1.84)
    world_T_ego = np.eye(4)
    world_T_ego[:2, :2] = [[0.0, -1.0], [1.0, 0.0]]  # ego faces world +y
    world_T_ego[:3, 3] = (100.0, 50.0, 3.0)
    T = world_T_ego @ ego
    # Facing model +y = ego forward = world +y.
    b = geometry.to_world(T, 0.0, 10.0, -1.04, 1.9, 4.5, 1.6, math.pi, 1.0, 0.0)
    assert b.heading == pytest.approx(math.pi / 2)
    # Model +x (ego right, -y) is world +x.
    assert (b.vx, b.vy) == pytest.approx((1.0, 0.0))
    assert b.cz == pytest.approx(3.0 + 0.8)


@pytest.mark.parametrize("heading", [-3.0, -1.2, 0.0, 0.4, 2.0, math.pi])
def test_round_trip(heading):
    T = geometry.ego_T_model("nuscenes_lidar", _ego_T_lidar(), 1.84)
    T[:3, 3] += (5.0, -7.0, 0.2)
    box = WorldBox(10.0, -4.0, 0.9, 1.8, 4.4, 1.5, heading, 2.0, -1.0)
    raw = geometry.from_world(T, box)
    back = geometry.to_world(T, *raw)
    for a, b in zip(
        (back.cx, back.cy, back.cz, back.w, back.l, back.h, back.vx, back.vy),
        (box.cx, box.cy, box.cz, box.w, box.l, box.h, box.vx, box.vy),
    ):
        assert a == pytest.approx(b)
    assert math.remainder(back.heading - heading, 2 * math.pi) == pytest.approx(
        0.0, abs=1e-9
    )
