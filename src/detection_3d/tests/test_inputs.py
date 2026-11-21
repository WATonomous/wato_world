"""Points, images and calibration for one frame (inputs.py)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from wato_common.artifacts.lidar_sweeps import WorldSweep
from wato_detection_3d import inputs
from wato_detection_3d.config import ImageConfig, PointsConfig

S = 1_000_000_000  # ns


def _sweep(xyz, intensity=None) -> WorldSweep:
    xyz = np.asarray(xyz, dtype=np.float64)
    i = np.full(len(xyz), 7.0, dtype=np.float32) if intensity is None else intensity
    return WorldSweep(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], intensity=i)


def test_pick_sweeps_newest_first_within_count_and_age():
    history = {"cc": [(t * S // 10, t) for t in range(20)]}  # 10 Hz
    cfg = PointsConfig(past_sweeps=9, max_time_lag_s=0.5)
    out = inputs.pick_sweeps(19 * S // 10, {"cc": (19, 19 * S // 10)}, history, cfg)
    assert out[0] == ("cc", 19, 19 * S // 10, True)
    # 0.5 s at 10 Hz: five earlier sweeps, newest first; the count doesn't bind.
    assert [sid for _, sid, _, cur in out[1:]] == [18, 17, 16, 15, 14]
    cfg = PointsConfig(past_sweeps=2, max_time_lag_s=0.5)
    out = inputs.pick_sweeps(19 * S // 10, {"cc": (19, 19 * S // 10)}, history, cfg)
    assert [sid for _, sid, _, _ in out[1:]] == [18, 17]


def test_assemble_points_ages_ranges_and_drops_close_points_of_earlier_sweeps():
    cfg = PointsConfig(remove_close_m=1.0)
    near = [0.5, 0.2, 0.0]  # within 1 m of a sensor at the origin
    far = [10.0, 0.0, 0.0]
    out_of_range = [80.0, 0.0, 0.0]
    current = inputs.SweepPoints(5, 10 * S, _sweep([near, far, out_of_range]), None)
    earlier = inputs.SweepPoints(4, 10 * S - S // 4, _sweep([near, far]), np.eye(4))
    points, used = inputs.assemble_points([current, earlier], np.eye(4), 10 * S, cfg)
    assert used == (4, 5)
    assert points.dtype == np.float32 and points.shape == (3, 5)
    # The current sweep keeps its close point; the earlier one loses it.
    np.testing.assert_allclose(points[:, :3], [near, far, far])
    np.testing.assert_allclose(points[:, 3], 7.0)
    np.testing.assert_allclose(points[:, 4], [0.0, 0.0, 0.25])


def test_assemble_points_caps_by_dropping_the_oldest(monkeypatch):
    monkeypatch.setattr(inputs, "MAX_POINTS", 3)
    cfg = PointsConfig(remove_close_m=0.0)
    a = inputs.SweepPoints(1, 9 * S, _sweep([[5, 0, 0], [6, 0, 0]]), np.eye(4))
    b = inputs.SweepPoints(2, 10 * S, _sweep([[7, 0, 0], [8, 0, 0]]), None)
    points, _ = inputs.assemble_points([a, b], np.eye(4), 10 * S, cfg)
    # The current sweep's two points stay; one of the earlier sweep's goes.
    assert len(points) == 3
    assert {7.0, 8.0} <= set(points[:, 0].tolist())
    assert (points[:, 4] == 0.0).sum() == 2


def test_assemble_points_needs_intensity():
    s = WorldSweep(x=np.zeros(1), y=np.zeros(1), z=np.zeros(1))
    with pytest.raises(ValueError, match="intensity"):
        inputs.assemble_points(
            [inputs.SweepPoints(1, 0, s, None)], np.eye(4), 0, PointsConfig()
        )


def test_img_aug_matrix_is_the_nuscenes_test_crop():
    # The car's test: 1600x900 at 0.48 → crop_x 32, crop_y 176.
    m = inputs.img_aug_matrix()
    np.testing.assert_allclose(m[:2, :2], np.eye(2) * 0.48)
    np.testing.assert_allclose(m[:2, 3], [-32.0, -176.0])


def test_band_passes_a_nuscenes_image_through():
    band = inputs.band_for(1600, 900, 491.5, ImageConfig())
    assert (band.scale, band.top, band.band_h) == (1.0, 0.0, 900.0)


def test_band_puts_the_principal_point_on_the_nuscenes_row():
    cfg = ImageConfig(principal_row_px=490.4)
    band = inputs.band_for(1280, 1024, 535.9, cfg)
    assert band.scale == 1.25 and band.band_h == 720.0
    assert (535.9 - band.top + 0.5) * 1.25 - 0.5 == pytest.approx(490.4)
    # Clamped inside the image.
    assert inputs.band_for(1280, 1024, 10.0, cfg).top == 0.0
    assert inputs.band_for(1280, 1024, 1020.0, cfg).top == pytest.approx(1024 - 720)
    with pytest.raises(ValueError, match="16:9"):
        inputs.band_for(1920, 900, 450.0, cfg)


def test_apply_band_intrinsics_follow_the_pixels():
    K = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 536.0], [0.0, 0.0, 1.0]])
    img = np.zeros((1024, 1280, 3), dtype=np.uint8)
    # A bright 8x8 block centred on a known ray.
    ray = np.array([0.1, 0.05, 1.0])
    u, v = (K @ ray)[:2]
    img[int(v) - 4 : int(v) + 4, int(u) - 4 : int(u) + 4] = 255
    band = inputs.band_for(1280, 1024, K[1, 2], ImageConfig())
    out, K2 = inputs.apply_band(Image.fromarray(img), K, band)
    assert out.shape == (900, 1600, 3) and out.dtype == np.uint8
    u2, v2 = (K2 @ ray)[:2]
    ys, xs = np.nonzero(out[:, :, 0] > 128)
    assert xs.mean() == pytest.approx(u2 - 0.5 * 1.25 + 0.5, abs=1.0)
    assert ys.mean() == pytest.approx(v2 - 0.5 * 1.25 + 0.5, abs=1.0)
