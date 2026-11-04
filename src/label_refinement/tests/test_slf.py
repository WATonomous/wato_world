"""slf: a box fit to silhouettes and LiDAR points, on a synthetic car.

A 4.5 x 1.8 x 1.5 m car centred at (10, 0) on the ground, heading +x; the ego
at the origin behind it. The LiDAR sees its rear face only, so the observed
box is 0.3 m long. Four cameras around its rear quarters give its
silhouette; the fit starts from a too-small Phase 3 box.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wato_label_refinement import refine, slf  # noqa: E402
from wato_label_refinement.config import SlfConfig  # noqa: E402
from wato_common.schemas import Box3D  # noqa: E402

TRUE = np.array([4.5, 1.8, 1.5])
CENTRE = np.array([10.0, 0.0, 0.75])
OBSERVED = Box3D(cx=7.9, cy=0.0, cz=0.75, w=1.8, l=0.3, h=1.5, heading=0.0)
F_PX, R = 400.0, 24


def _look_at(eye, at):
    """World rotation of a camera at eye looking at at (x right, y down, z ahead)."""
    z = (at - eye) / np.linalg.norm(at - eye)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def _hits(o, d, lo, hi):
    """Whether rays o + t d (t > 0) hit the axis-aligned box [lo, hi]."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (lo - o) / d, (hi - o) / d
    near = np.nanmax(np.minimum(t1, t2), axis=-1)
    far = np.nanmin(np.maximum(t1, t2), axis=-1)
    return (far >= np.maximum(near, 0)).astype(np.float32)


def _view(eye):
    eye = np.asarray(eye, float)
    rot = _look_at(eye, CENTRE)
    lo, hi = CENTRE - TRUE / 2, CENTRE + TRUE / 2
    # The crop views.problem would take: the car's projection grown by 20%.
    corners = np.array(
        [
            [x, y, z]
            for x in (lo[0], hi[0])
            for y in (lo[1], hi[1])
            for z in (lo[2], hi[2])
        ]
    )
    cam = (corners - eye) @ rot
    uv = cam[:, :2] / cam[:, 2:] * F_PX
    (u0, v0), (u1, v1) = uv.min(0), uv.max(0)
    pu, pv = 0.2 * (u1 - u0), 0.2 * (v1 - v0)
    grid = (np.arange(R) + 0.5) / R
    uu, vv = np.meshgrid(
        u0 - pu + grid * (u1 - u0 + 2 * pu), v0 - pv + grid * (v1 - v0 + 2 * pv)
    )
    d_cam = np.column_stack([uu.ravel() / F_PX, vv.ravel() / F_PX, np.ones(R * R)])
    d_cam /= np.linalg.norm(d_cam, axis=1, keepdims=True)
    dirs = d_cam @ rot.T
    target = _hits(eye, dirs, lo, hi)
    depth = float((CENTRE - eye) @ rot[:, 2])
    pitch = max(u1 - u0 + 2 * pu, v1 - v0 + 2 * pv) / R
    return eye, dirs, target, depth * pitch / F_PX


def _problem(size0, with_views=True, prior=None):
    eyes = [(2.0, -4.0, 1.6), (2.0, 4.0, 1.6), (4.0, -7.0, 1.6), (4.0, 7.0, 1.6)]
    views = [_view(e) for e in eyes] if with_views else []
    au, av = refine.anchors(OBSERVED, 0.0, (0.0, 0.0))
    gy, gz = np.meshgrid(np.linspace(-0.85, 0.85, 12), np.linspace(0.1, 1.45, 8))
    rear = np.column_stack([np.full(gy.size, 7.75), gy.ravel(), gz.ravel()])
    n = len(views)
    return slf.Problem(
        size0=np.asarray(size0, float),
        prior=prior,
        heading=np.zeros(1),
        centre_obs=np.array([[OBSERVED.cx, OBSERVED.cy]]),
        anchor_lo=np.array([[au.lo, av.lo]]),
        anchor_hi=np.array([[au.hi, av.hi]]),
        anchor_near_lo=np.array([[au.near_lo, av.near_lo]]),
        anchored=np.array([True]),
        ground_z=np.zeros(1),
        view_frame=np.zeros(n, int),
        view_shift=np.zeros((n, 2)),
        origin=np.array([v[0] for v in views]).reshape(n, 3),
        dirs=np.array([v[1] for v in views]).reshape(n, R * R, 3),
        target=np.array([v[2] for v in views]).reshape(n, R * R),
        valid=np.ones((n, R * R), bool),
        footprint=np.array([v[3] for v in views]),
        points=rear,
        point_frame=np.zeros(len(rear), int),
    )


def test_silhouettes_recover_the_length_the_lidar_never_saw():
    cfg = SlfConfig(iterations=200)
    fit = slf.fit(_problem([1.5, 1.5, 1.2]), cfg, torch.device("cpu"))
    assert fit.size == pytest.approx(TRUE, abs=0.35)
    assert fit.residual_silhouette < 0.2 < fit.start_residual_silhouette
    # The rear face stays on the points: the box grew away from the ego.
    assert fit.residual_lidar < 0.1
    near = refine.anchored_centre(OBSERVED, fit.size[0], fit.size[1], 0.0, (0, 0))
    assert near[0] - fit.size[0] / 2 + fit.offset[0] == pytest.approx(7.75, abs=0.15)


def test_points_alone_keep_the_size_where_nothing_says_otherwise():
    # No view: the rear face fixes width and height; the length is the prior's.
    cfg = SlfConfig(iterations=200)
    prior = np.array([4.63, 1.97, 1.74])
    fit = slf.fit(_problem([1.5, 1.5, 1.2], False, prior), cfg, torch.device("cpu"))
    assert fit.size[0] == pytest.approx(4.63, rel=0.2)
    assert np.isnan(fit.residual_silhouette)


def test_the_rounded_box_sdf_is_signed_distance():
    half = torch.tensor([2.0, 1.0, 0.5])
    p = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    d = slf.rounded_box_sdf(p, half, torch.tensor(0.0))
    assert d.tolist() == pytest.approx([-0.5, 1.0, 0.0], abs=2e-3)


def test_tracks_fit_together_as_they_fit_alone():
    cfg = SlfConfig(iterations=60)
    prior = np.array([4.63, 1.97, 1.74])
    probs = [_problem([1.5, 1.5, 1.2]), _problem([2.0, 1.2, 1.0], False, prior)]
    together = slf.fit_many(probs, cfg, torch.device("cpu"))
    for p, t in zip(probs, together):
        alone = slf.fit(p, cfg, torch.device("cpu"))
        assert t.size == pytest.approx(alone.size, abs=1e-3)
        assert t.offset == pytest.approx(alone.offset, abs=1e-3)


def test_fitting_is_independent_of_the_world_coordinate_origin():
    cfg = SlfConfig(iterations=60)
    original = _problem([1.5, 1.5, 1.2])
    translation = np.array([1_000_000.0, -2_000_000.0, 500_000.0])
    moved = replace(
        original,
        centre_obs=original.centre_obs + translation[:2],
        ground_z=original.ground_z + translation[2],
        origin=original.origin + translation,
        points=original.points + translation,
    )
    baseline, translated = slf.fit_many([original, moved], cfg, torch.device("cpu"))
    np.testing.assert_allclose(translated.size, baseline.size, atol=1e-3, rtol=0)
    np.testing.assert_allclose(translated.offset, baseline.offset, atol=1e-3, rtol=0)
    np.testing.assert_allclose(
        translated.centres - translation, baseline.centres, atol=1e-3, rtol=0
    )
