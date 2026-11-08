"""label_refinement's per-track refinement, and the pipeline on staged chunks.

Staged scene, one LiDAR, 10 frames at 10 Hz, ego at the origin, ground at z 0
(as one chunk, or as two sharing frame 5):
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
from wato_label_refinement.config import (
    ComponentConfig,
    MovingConfig,
    ParkedConfig,
    SlfConfig,
)
from wato_label_refinement.pipeline import run

BAG, CHUNK = "b", "0000"
MS = 1_000_000
N = 10
PHASE3 = ComponentConfig(slf=SlfConfig(enabled=False))


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


def _stage_chunk(chunk, ks, owned, ground_z=0.0, masklets=False):
    """Chunk `chunk` sees frames ks (its sweeps numbered from 0); returns its
    track frames, owned where owned(k)."""
    os.makedirs(local_path(lidar_proc_dir(BAG, chunk)), exist_ok=True)
    normals = np.zeros((40, 40, 3), np.float32)
    normals[..., 2] = 1.0
    ground.write(
        BAG,
        chunk,
        ground.GroundGrid(
            np.full((40, 40), ground_z, np.float32),
            normals,
            np.array([-50.0, -50.0]),
            2.5,
        ),
        np.empty((0, 3)),
    )
    meta, frames = [], []
    for sid, k in enumerate(ks):
        p, m = _parked_face(k), _mover_face(k)
        xyz = np.concatenate([p, m])
        uri = lidar_sweeps.write_world(
            BAG,
            chunk,
            sid,
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
                chunk_id=chunk,
                sweep_id=sid,
                lidar_id="lidar_top",
                frame_id=sid,
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
        for tid, pts in (("0000-0", p), ("0000-1", m)):
            b = fit_bev_box(pts)
            frames.append(
                TrackFrame(
                    track_id=tid,
                    chunk_id=chunk,
                    sweep_id=sid,
                    reference_timestamp_ns=_t(k),
                    proposal_id=f"fused-{sid:06d}-{tid[-1]}",
                    cx=b.cx,
                    cy=b.cy,
                    cz=b.cz,
                    w=b.w,
                    l=b.l,
                    h=b.h,
                    heading=b.heading,
                    cls="car",
                    score=0.5,
                    owned=owned(tid, k),
                    supporting_cam_ids=("CAM_A",) if masklets else (),
                    supporting_masklet_ids=(f"m{tid[-1]}",) if masklets else (),
                )
            )
    write_table(meta, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(BAG, chunk))
    write_table(
        [
            dict(
                bag_id=BAG,
                chunk_id=chunk,
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
        poses_path(BAG, chunk),
    )
    return frames


def _write_tracks(frames):
    def summary(tid, state):
        return TrackSummary(
            track_id=tid,
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


def _stage():
    # The mover's last frame is another chunk's to label.
    _write_tracks(
        _stage_chunk(CHUNK, range(N), lambda tid, k: tid == "0000-0" or k < N - 1)
    )


def _stage_two_chunks():
    # Chunk 0000 sees frames 0-5, chunk 0001 frames 5-9 (as its sweeps 0-4),
    # standing on ground 0.5 m higher; 0001 labels frame 5 on.
    _write_tracks(
        _stage_chunk("0000", range(6), lambda tid, k: k < 5)
        + _stage_chunk("0001", range(5, N), lambda tid, k: True, ground_z=0.5)
    )


def test_parked_car_gets_one_box_from_every_frames_points():
    _stage()
    run(PHASE3, bag_id=BAG)
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
    run(PHASE3, bag_id=BAG)
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


def test_a_track_across_two_chunks_is_refined_whole():
    _stage_two_chunks()
    run(PHASE3, bag_id=BAG)
    out = refined_labels.read(BAG)
    parked = [b for b in out if b.track_id == "0000-0"]
    # Every row, both chunks' copies of frame 5 included, gets one box: the
    # rear seen by chunk 0000 and the side by 0001 make one footprint.
    assert len(parked) == N + 1 and len({(b.cx, b.cy, b.cz) for b in parked}) == 1
    assert parked[0].l == pytest.approx(4.4, abs=0.3)
    assert parked[0].w == pytest.approx(1.8, abs=0.3)
    moving = [b for b in out if b.track_id == "0000-1"]
    assert len({(b.w, b.l, b.h) for b in moving}) == 1
    at5 = [b for b in moving if b.reference_timestamp_ns == _t(5)]
    assert sorted((b.chunk_id, b.owned) for b in at5) == [
        ("0000", False),
        ("0001", True),
    ]
    assert at5[0].cx == at5[1].cx and at5[0].cz == at5[1].cz
    # Each frame sits on its own chunk's ground.
    bottom = {b.chunk_id: b.cz - b.h / 2 for b in moving if b.owned}
    assert bottom == pytest.approx({"0000": 0.0, "0001": 0.5}, abs=1e-6)


# A camera at the ego, 1.5 m up, looking at the mover's side (bearing ~243°).
CAM_POS = np.array([0.0, 0.0, 1.5])
CAM_K = np.array([[300.0, 0.0, 600.0], [0.0, 300.0, 400.0], [0.0, 0.0, 1.0]])
CAM_W, CAM_H = 1200, 800


def _ego_T_cam():
    z = np.array([-0.5, -1.0, 0.0]) / math.sqrt(1.25)
    y = np.array([0.0, 0.0, -1.0])
    x = np.cross(y, z)
    t = np.eye(4)
    t[:3, :3] = np.column_stack([x, y, z])
    t[:3, 3] = CAM_POS
    return t


def _true_box(tid, k):
    if tid == "0000-0":
        return np.array([7.8, 3.2, 0.0]), np.array([12.2, 6.8, 1.5])  # parked
    x = -10.0 + k
    return np.array([x - 2.2, -6.9, 0.0]), np.array([x + 2.2, -5.1, 1.5])


def _stage_camera():
    from wato_common.artifacts import calibration, frame_index, masklets
    from wato_common.artifacts.calibration import (
        Calibration,
        CameraCalibration,
        LidarCalibration,
    )
    from wato_common.schemas import FrameIndexRow

    calibration.write(
        BAG,
        Calibration(
            cameras={
                "CAM_A": CameraCalibration(
                    "CAM_A", CAM_K, _ego_T_cam(), width=CAM_W, height=CAM_H
                )
            },
            lidars={"lidar_top": LidarCalibration("lidar_top", np.eye(4))},
        ),
    )
    frame_index.write(
        BAG,
        CHUNK,
        [
            FrameIndexRow(
                frame_id=f"{k}",
                bag_id=BAG,
                chunk_id=CHUNK,
                sweep_id=k,
                lidar_id="lidar_top",
                lidar_path="",
                reference_timestamp_ns=_t(k),
                cam_id="CAM_A",
                image_path=f"/img/{k:06d}.jpg",
                camera_seq=k,
                camera_timestamp_ns=_t(k),
                valid_camera=True,
                valid_pose=True,
            )
            for k in range(N)
        ],
    )
    # Each car's silhouette, every pixel's ray tested against its true box.
    uu, vv = np.meshgrid(np.arange(CAM_W) + 0.5, np.arange(CAM_H) + 0.5)
    d = (
        np.stack(
            [
                (uu - CAM_K[0, 2]) / CAM_K[0, 0],
                (vv - CAM_K[1, 2]) / CAM_K[1, 1],
                np.ones_like(uu),
            ],
            -1,
        )
        @ _ego_T_cam()[:3, :3].T
    )
    for tid in ("0000-0", "0000-1"):
        for k in range(N):
            lo, hi = _true_box(tid, k)
            with np.errstate(divide="ignore", invalid="ignore"):
                t1, t2 = (lo - CAM_POS) / d, (hi - CAM_POS) / d
            near = np.nanmax(np.minimum(t1, t2), -1)
            far = np.nanmin(np.maximum(t1, t2), -1)
            masklets.write_mask(
                BAG, CHUNK, f"m{tid[-1]}", k, far >= np.maximum(near, 0)
            )
    masklets.write(
        BAG,
        CHUNK,
        [
            masklets.MaskletRecord(f"m{i}", "CAM_A", "car", 0.9, tuple(range(N)))
            for i in "01"
        ],
    )


def test_slf_gives_the_mover_the_width_its_lidar_never_saw():
    _write_tracks(_stage_chunk(CHUNK, range(N), lambda tid, k: True, masklets=True))
    _stage_camera()
    run(ComponentConfig(), bag_id=BAG)
    moving = [b for b in refined_labels.read(BAG) if b.track_id == "0000-1"]
    m = moving[N // 2]
    # Phase 3 left it 0.3 m wide (no frame saw more than the near side).
    assert m.size_source == "slf"
    assert m.w == pytest.approx(1.8, abs=0.4) and m.l == pytest.approx(4.4, abs=0.5)
    # Still on its near side's points, grown away from the ego.
    assert m.cy + m.w / 2 == pytest.approx(-5.1, abs=0.25)
    assert m.residual_silhouette is not None and m.residual_silhouette < 0.3


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
    fitted, vel, _ = refine.local_linear(t, xy, 1.0)
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
        [None] * 10,
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


def test_velocity_standard_error_measures_the_centres_jitter():
    t = np.linspace(0, 2, 41)
    line = np.column_stack([1.0 * t, np.zeros_like(t)])
    _, _, exact = refine.local_linear(t, line, 1.0)
    assert np.all(exact[1:-1] < 1e-9)
    noisy = line + np.random.default_rng(1).normal(0, 0.3, line.shape)
    _, _, se = refine.local_linear(t, noisy, 1.0)
    # sigma / sqrt(sum dt^2) for the 21 samples 0.05 s apart within ±0.5 s.
    dt = np.linspace(-0.5, 0.5, 21)
    assert np.median(se) == pytest.approx(0.3 / np.sqrt(np.sum(dt**2)), rel=0.1)
    _, _, two = refine.local_linear(np.array([0.0, 0.1]), line[:2], 1.0)
    assert np.all(np.isinf(two))


def _jittering(n, speed_mps, jitter_m, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 0.05
    return t, [
        Box3D(
            cx=float(speed_mps * ti + rng.normal(0, jitter_m)),
            cy=float(rng.normal(0, jitter_m)),
            cz=0.75,
            w=1.8,
            l=4.5,
            h=1.5,
            heading=0.0,
        )
        for ti in t
    ]


@pytest.mark.parametrize("z,expect_directed", [(0.0, True), (3.0, False)])
def test_a_parked_cars_jitter_gives_no_heading_once_judged_against_its_noise(
    z, expect_directed
):
    """A parked car whose visible faces change jitters its box centre; the
    local fit's speed then passes heading_min_speed_mps at many frames."""
    t, boxes = _jittering(80, 0.0, 0.3, 2)
    cfg = MovingConfig(heading_speed_z=z)
    r = refine.refine_moving(
        (t * 1e9).astype(np.int64),
        boxes,
        np.full(80, 50),
        np.zeros((80, 2)),
        [None] * 80,
        cfg,
    )
    assert any(r.heading_directed) is expect_directed


def test_a_slow_real_mover_keeps_its_velocity_heading():
    t, boxes = _jittering(80, 1.0, 0.05, 3)
    cfg = MovingConfig(heading_speed_z=3.0)
    r = refine.refine_moving(
        (t * 1e9).astype(np.int64),
        boxes,
        np.full(80, 50),
        np.zeros((80, 2)),
        [None] * 80,
        cfg,
    )
    assert np.mean(r.heading_directed) > 0.9
    assert np.median([abs(b.heading) for b in r.boxes]) < np.radians(5)
