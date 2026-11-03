"""label_refinement's per-track refinement, and the pipeline on a staged chunk.

Staged scene, one LiDAR, 10 frames at 10 Hz, ego at the origin, ground at z 0:
  P  a parked 4.4 x 1.8 m car at (10, 5); each frame sees one face of it
     (its rear in the first half, its left side in the second), so each
     frame's box covers that face only
  M  a 4.4 x 1.8 m car driving +x at 10 m/s along y = -6; each frame sees its
     near (left) side, 0.3 m deep
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from wato_common.artifact_store import (
    lidar_proc_dir,
    lidar_proc_index_path,
    local_path,
    poses_path,
)
from wato_common.artifacts import ground, lidar_sweeps, refined_labels, tracks
from wato_common.artifacts.tracks import TrackFrame, TrackSummary
from wato_common.io.parquet_io import write_table
from wato_common.schemas import (
    POSES_SCHEMA,
    PROCESSED_SWEEPS_SCHEMA,
    Box3D,
    ProcessedSweepMeta,
)
from wato_common.tracking import fit_bev_box
from wato_label_refinement import refine
from wato_label_refinement.config import ComponentConfig, MovingConfig, ParkedConfig
from wato_label_refinement.pipeline import run

BAG, CHUNK = "b", "0000"
MS = 1_000_000
N = 10


@pytest.fixture(autouse=True)
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")


def _face(x0, x1, y0, y1, seed, n=60):
    rng = np.random.default_rng(seed)
    return rng.uniform([x0, y0, 0.3], [x1, y1, 1.5], size=(n, 3))


def _parked_face(k):
    # Rear face (x 7.8-8.1) first, then the left side (y 5.6-5.9).
    if k < N // 2:
        return _face(7.8, 8.1, 4.1, 5.9, k)
    return _face(7.8, 12.2, 5.6, 5.9, k)


def _mover_face(k):
    x = -10.0 + k  # 10 m/s at 10 Hz
    return _face(x - 2.2, x + 2.2, -5.4, -5.1, 100 + k)


def _t(k):
    return (100 + 100 * k) * MS


def _stage():
    os.makedirs(local_path(lidar_proc_dir(BAG, CHUNK)), exist_ok=True)
    normals = np.zeros((40, 40, 3), np.float32)
    normals[..., 2] = 1.0
    ground.write(
        BAG,
        CHUNK,
        ground.GroundGrid(
            np.zeros((40, 40), np.float32), normals, np.array([-50.0, -50.0]), 2.5
        ),
        np.empty((0, 3)),
    )
    meta, frames = [], []
    for k in range(N):
        p, m = _parked_face(k), _mover_face(k)
        xyz = np.concatenate([p, m])
        uri = lidar_sweeps.write_world(
            BAG,
            CHUNK,
            k,
            lidar_sweeps.WorldSweep(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                origin=np.zeros(3),
                ground_mask=np.zeros(len(xyz), bool),
            ),
        )
        meta.append(
            ProcessedSweepMeta(
                bag_id=BAG,
                chunk_id=CHUNK,
                sweep_id=k,
                lidar_id="lidar_top",
                frame_id=k,
                reference_timestamp_ns=_t(k),
                n_points_total=len(xyz),
                n_points_static=len(xyz),
                n_points_dynamic=0,
                world_path=uri,
                dynamic_mask_path="",
                has_intensity=False,
                deskewed=True,
            ).model_dump()
        )
        for tid, pts, owned in (("0000-0", p, True), ("0000-1", m, k < N - 1)):
            b = fit_bev_box(pts)
            frames.append(
                TrackFrame(
                    track_id=tid,
                    chunk_id=CHUNK,
                    sweep_id=k,
                    reference_timestamp_ns=_t(k),
                    proposal_id=f"fused-{k:06d}-{tid[-1]}",
                    cx=b.cx,
                    cy=b.cy,
                    cz=b.cz,
                    w=b.w,
                    l=b.l,
                    h=b.h,
                    heading=b.heading,
                    cls="car",
                    score=0.5,
                    owned=owned,
                )
            )
    write_table(meta, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(BAG, CHUNK))
    write_table(
        [
            dict(
                bag_id=BAG,
                chunk_id=CHUNK,
                timestamp_ns=t * MS,
                x=0.0,
                y=0.0,
                z=0.0,
                qx=0.0,
                qy=0.0,
                qz=0.0,
                qw=1.0,
                world_T_ego_flat=np.eye(4).flatten().tolist(),
                source="/odom",
                valid=True,
                interval_drop_reason=None,
            )
            for t in (0, 2000)
        ],
        POSES_SCHEMA,
        poses_path(BAG, CHUNK),
    )

    def summary(tid, state):
        return TrackSummary(
            track_id=tid,
            chunk_id=CHUNK,
            n_frames=N,
            first_timestamp_ns=_t(0),
            last_timestamp_ns=_t(N - 1),
            motion_state=state,
            cls="car",
            n_strong=N,
            frac_two_source=0.0,
            frac_lidar_mos=0.0,
            net_displacement_m=0.0,
            centre_var_m2=0.0,
            median_points=60.0,
        )

    tracks.write(
        BAG, frames, [summary("0000-0", "parked"), summary("0000-1", "moving")]
    )


def test_parked_car_gets_one_box_from_every_frames_points():
    _stage()
    run(ComponentConfig(), bag_id=BAG)
    out = refined_labels.read(BAG)
    parked = [b for b in out if b.track_id == "0000-0"]
    assert len(parked) == N and len({(b.cx, b.cy) for b in parked}) == 1
    p = parked[0]
    assert p.size_source == "aggregate" and not p.heading_directed
    # Rear and side together give the car's footprint, not one face.
    assert p.l == pytest.approx(4.4, abs=0.3) and p.w == pytest.approx(1.8, abs=0.3)
    assert (p.cx, p.cy) == pytest.approx((10.0, 5.0), abs=0.3)
    assert p.cz - p.h / 2 == pytest.approx(0.0, abs=1e-6)  # down to the ground


def test_moving_car_gets_one_size_its_heading_and_its_far_side_back():
    _stage()
    run(ComponentConfig(), bag_id=BAG)
    moving = sorted(
        (b for b in refined_labels.read(BAG) if b.track_id == "0000-1"),
        key=lambda b: b.sweep_id,
    )
    assert [b.owned for b in moving] == [True] * (N - 1) + [False]
    assert all(
        b.heading_directed and b.heading == pytest.approx(0.0, abs=0.05) for b in moving
    )
    # Every box is as long as the car. No frame saw more than the 0.3 m-deep
    # side, so the width is that and the box stays on the points.
    assert all(b.l == pytest.approx(4.4, abs=0.3) for b in moving)
    m = moving[N // 2]
    assert m.cx == pytest.approx(-10.0 + N // 2, abs=0.3)
    assert (m.cy - m.w / 2, m.cy + m.w / 2) == pytest.approx((-5.4, -5.1), abs=0.1)
    assert m.residual_smoothness is not None and m.residual_smoothness < 0.2


def test_anchored_centre_extends_away_from_the_ego():
    # Observed: the rear face of a car ahead (x 9.9-10.1, y -0.9-0.9); ego at
    # the origin. A 4 m box keeps the face and grows away: centre x = 11.9.
    face = Box3D(cx=10.0, cy=0.0, cz=0.75, w=0.2, l=1.8, heading=math.pi / 2, h=1.5)
    c = refine.anchored_centre(face, 4.0, 1.8, 0.0, (0.0, 0.0))
    assert c == pytest.approx((11.9, 0.0), abs=1e-6)
    # Ego on the far side: it grows the other way.
    c = refine.anchored_centre(face, 4.0, 1.8, 0.0, (20.0, 0.0))
    assert c == pytest.approx((8.1, 0.0), abs=1e-6)


def test_local_linear_recovers_a_constant_velocity():
    t = np.linspace(0, 2, 21)
    xy = np.column_stack([3.0 * t, -1.0 * t]) + np.random.default_rng(0).normal(
        0, 0.02, (21, 2)
    )
    fitted, vel = refine.local_linear(t, xy, 1.0)
    assert np.allclose(vel[5:-5], [3.0, -1.0], atol=0.15)
    assert np.abs(fitted - np.column_stack([3.0 * t, -t])).max() < 0.05


def test_too_few_points_leave_a_parked_track_to_the_moving_path():
    assert refine.refine_parked(np.zeros((5, 3)), None, ParkedConfig()) is None


def test_size_is_a_high_percentile_of_the_densest_frames():
    boxes = [
        Box3D(cx=float(i), cy=0, cz=0.8, w=1.0, l=2.0 + 0.2 * i, h=1.5, heading=0.0)
        for i in range(10)
    ]
    n_points = np.arange(10)  # later frames are denser and longer
    r = refine.refine_moving(
        np.arange(10) * 100 * MS,
        boxes,
        n_points,
        np.zeros((10, 2)),
        None,
        MovingConfig(),
    )
    lengths = [2.0 + 0.2 * i for i in range(5, 10)]
    assert r.boxes[0].l == pytest.approx(np.percentile(lengths, 90))
    assert len({b.l for b in r.boxes}) == 1


def test_a_stray_box_does_not_stretch_a_parked_region():
    car = [
        Box3D(cx=10.0 + 0.1 * (i % 2), cy=5.0, cz=0.8, w=1.8, l=4.4, h=1.5, heading=0.0)
        for i in range(9)
    ]
    stray = Box3D(cx=10.0, cy=8.5, cz=0.8, w=1.8, l=4.4, h=1.5, heading=0.0)
    region = refine.enclosing_region(car + [stray], 0.3, 1.0)
    assert region.w == pytest.approx(1.8 + 0.6, abs=0.05)
    assert refine.enclosing_region(car + [stray], 0.3, 10.0).w > 5.0
