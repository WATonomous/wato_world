"""Tests for mf_mos.py (Step A.5 — MF-MOS learned MOS).

All tests are CPU-runnable.  torch is never imported; MFMosModel is
monkeypatched with _StubModel that treats residual energy as the moving signal.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from wato_common.artifact_store import (
    calibration_path,
    chunks_index_path,
    dynamic_map_path,
    ensure_local_dir,
    lidar_proc_dir,
    lidar_proc_index_path,
    lidar_sweep_path,
    lidar_sweeps_path,
    lidar_world_path,
    local_path,
    mf_mos_mask_path,
    poses_path,
    static_map_path,
)
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import (
    CHUNK_SCHEMA,
    LIDAR_SWEEPS_SCHEMA,
    POSES_SCHEMA,
    PROCESSED_SWEEPS_SCHEMA,
)
from wato_lidar_preprocessing.mf_mos import _core as mf_mos_mod
from wato_lidar_preprocessing.classify import process_chunk as classify_chunk
from wato_lidar_preprocessing.config import ComponentConfig, MFMosParams
from wato_lidar_preprocessing.mf_mos import (
    _compute_residual,
    _range_project,
    _unproject_mask,
    process_chunk,
)

# ---------------------------------------------------------------------------
# Stub model — CPU-only; pixels where sum-of-|residuals| > 0.1 are moving.
# ---------------------------------------------------------------------------

H, W = 32, 1024
FOV_UP, FOV_DOWN = 10.0, -30.0


class _StubModel:
    H_model = H
    W_model = W
    n_input_scans = 4
    model_resolution = (H, W)

    def infer(
        self, range_image: np.ndarray, residual_images: list[np.ndarray]
    ) -> np.ndarray:
        h, w = range_image.shape[1], range_image.shape[2]
        if residual_images:
            res_energy = sum(np.abs(r) for r in residual_images)
        else:
            res_energy = np.zeros((h, w), dtype=np.float32)
        return (res_energy > 0.1).astype(np.float32)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def _clear_model_cache():
    mf_mos_mod._MODEL_CACHE.clear()
    yield
    mf_mos_mod._MODEL_CACHE.clear()


def _stub_load_model(_params):
    return _StubModel()


def _enabled_cfg(**kw) -> ComponentConfig:
    # Detection/projection tests use synthetic single-point movers; disable the
    # per-sweep 3D cluster denoise (min cluster = 1) so it doesn't drop them.
    kw.setdefault("moving_min_cluster_pts", 1)
    return ComponentConfig(
        mf_mos=MFMosParams(
            enabled=True,
            range_image_h=H,
            range_image_w=W,
            fov_up_deg=FOV_UP,
            fov_down_deg=FOV_DOWN,
            device="cpu",
            **kw,
        )
    )


def _write_calibration(bag_id: str) -> None:
    calib = {
        "calibration_version": "t",
        "ego_frame": "base_link",
        "cameras": {},
        "lidars": {
            "LIDAR_TOP": {
                "frame_id": "LIDAR_TOP",
                "ego_T_lidar": np.eye(4).tolist(),
            }
        },
        "static_transforms": {},
        "checks": {"sanity": "ok", "notes": ""},
    }
    p = local_path(calibration_path(bag_id))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as fh:
        json.dump(calib, fh)


def _write_poses(bag_id: str, chunk_id: str, timestamps_ns: list[int]) -> None:
    rows = [
        {
            "bag_id": bag_id,
            "chunk_id": chunk_id,
            "timestamp_ns": ts,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
            "world_T_ego_flat": np.eye(4).flatten().tolist(),
            "source": "odom",
            "valid": True,
        }
        for ts in timestamps_ns
    ]
    write_table(rows, POSES_SCHEMA, poses_path(bag_id, chunk_id))


def _write_raw_sweep(bag_id: str, chunk_id: str, sweep_id: int, xyz: np.ndarray) -> str:
    uri = lidar_sweep_path(bag_id, chunk_id, "LIDAR_TOP", sweep_id)
    path = local_path(uri)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2])
    return uri


def _write_lidar_sweeps(
    bag_id: str, chunk_id: str, sweeps: list[tuple[int, np.ndarray]]
) -> None:
    rows = []
    for sid, xyz in sweeps:
        raw_uri = _write_raw_sweep(bag_id, chunk_id, sid, xyz)
        n = xyz.shape[0]
        rows.append(
            {
                "bag_id": bag_id,
                "chunk_id": chunk_id,
                "lidar_id": "LIDAR_TOP",
                "sweep_id": sid,
                "lidar_path": raw_uri,
                "header_timestamp_ns": sid * 50_000_000,
                "record_timestamp_ns": sid * 50_000_000,
                "num_points": n,
                "has_ring": False,
                "has_intensity": False,
                "has_point_time": False,
                "min_range_m": 0.0,
                "max_range_m": float(np.linalg.norm(xyz, axis=1).max())
                if n > 0
                else 0.0,
                "valid": True,
                "drop_reason": None,
            }
        )
    write_table(rows, LIDAR_SWEEPS_SCHEMA, lidar_sweeps_path(bag_id, chunk_id))


def _write_proc_index(bag_id: str, chunk_id: str, sweep_ids: list[int]) -> None:
    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    rows = [
        {
            "bag_id": bag_id,
            "chunk_id": chunk_id,
            "sweep_id": sid,
            "lidar_id": "LIDAR_TOP",
            "reference_timestamp_ns": sid * 50_000_000,
            "n_points_total": 0,
            "n_points_static": 0,
            "n_points_dynamic": 0,
            "n_points_ground": 0,
            "world_path": "",
            "dynamic_mask_path": "",
            "has_intensity": False,
            "deskewed": True,
            "valid": True,
            "drop_reason": None,
            "world_xmin": None,
            "world_xmax": None,
            "world_ymin": None,
            "world_ymax": None,
            "world_zmin": None,
            "world_zmax": None,
            "frame_id": None,
            "mf_mos_mask_path": None,
        }
        for sid in sweep_ids
    ]
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))


def _make_in_fov_points(n: int) -> np.ndarray:
    """Return (n, 3) float32 points spread at radius=5 within the default FOV."""
    azimuths = np.linspace(0.05, 2 * np.pi - 0.05, max(n, 1))[:n]
    return np.stack(
        [5.0 * np.cos(azimuths), 5.0 * np.sin(azimuths), np.zeros(n)], axis=1
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Group 1: _range_project unit tests
# ---------------------------------------------------------------------------


def test_range_projection_round_trips_indices_match():
    """pixel_to_point_idx and point_to_pixel are consistent for in-FOV points."""
    xyz = _make_in_fov_points(8)
    _, p2p_idx, p2px = _range_project(xyz, None, H, W, FOV_UP, FOV_DOWN)
    for i, (row_px, col_px) in enumerate(p2px):
        if row_px < 0 or col_px < 0:
            continue
        assert p2p_idx[row_px, col_px] >= 0
        winner = p2p_idx[row_px, col_px]
        r_winner = float(np.linalg.norm(xyz[winner]))
        r_i = float(np.linalg.norm(xyz[i]))
        assert (
            r_winner <= r_i + 1e-4
        ), f"point {i}: winner range {r_winner:.4f} > self range {r_i:.4f}"


def test_range_projection_closest_point_wins():
    """When two points project to the same pixel, the closer one wins."""
    far_pt = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
    near_pt = np.array([[3.0, 0.0, 0.0]], dtype=np.float32)
    xyz = np.vstack([far_pt, near_pt])

    _, p2p_idx, p2px = _range_project(xyz, None, H, W, FOV_UP, FOV_DOWN)

    r0, c0 = p2px[0]
    r1, c1 = p2px[1]
    assert r0 >= 0 and c0 >= 0, "far point should be in FOV"
    assert r1 >= 0 and c1 >= 0, "near point should be in FOV"
    assert r0 == r1 and c0 == c1, "collinear points should hit same pixel"
    assert p2p_idx[r0, c0] == 1, "near point (index 1) must win the pixel"


# ---------------------------------------------------------------------------
# Group 2: _compute_residual unit tests
# ---------------------------------------------------------------------------


def test_residual_zero_for_static_scene():
    """Identical past/current xyz with identity ego motion → residual ≈ 0."""
    xyz = _make_in_fov_points(6)
    ri, _, _ = _range_project(xyz, None, H, W, FOV_UP, FOV_DOWN)
    identity = np.eye(4, dtype=np.float64)
    residual = _compute_residual(
        xyz, identity, identity, identity, H, W, FOV_UP, FOV_DOWN, ri[0]
    )
    valid = ri[0] >= 0
    assert np.allclose(residual[valid], 0.0, atol=1e-4)


def test_residual_nonzero_for_moving_object():
    """A car that moved along the radial direction produces a non-zero residual.

    cur has the car at range 5m; in the past scan (projected to current frame)
    the same car was at range 7m on the same azimuth.  Both scans return a valid
    pixel at that direction, so |5 - 7| = 2.0 appears in the residual image.
    """
    cur = np.array([[5.0, 0.0, 0.0]], dtype=np.float32)
    past = np.array([[7.0, 0.0, 0.0]], dtype=np.float32)
    ri_cur, _, _ = _range_project(cur, None, H, W, FOV_UP, FOV_DOWN)
    identity = np.eye(4, dtype=np.float64)
    residual = _compute_residual(
        past, identity, identity, identity, H, W, FOV_UP, FOV_DOWN, ri_cur[0]
    )
    assert residual.max() > 0.1


# ---------------------------------------------------------------------------
# Group 3: _unproject_mask / _unproject_scores unit tests
# ---------------------------------------------------------------------------


def test_unproject_mask_recovers_point_labels():
    """_unproject_mask maps pixel labels to correct per-point booleans."""
    H_px, W_px = 4, 8
    pixel_mask = np.zeros((H_px, W_px), dtype=bool)
    pixel_mask[1, 2] = True

    p2px = np.array([[1, 2], [0, 0], [-1, -1]], dtype=np.int32)
    out = _unproject_mask(pixel_mask, p2px, n_points=3)

    assert bool(out[0]) is True
    assert bool(out[1]) is False
    assert bool(out[2]) is False


def test_unproject_mask_occlusion_gate_excludes_background():
    """Range gate keeps the moving label on the front surface only.

    Two points project to the same moving pixel (1, 2): point 0 is the mover's
    front surface at range 5 m (the pixel winner), point 1 is occluded
    background (a wall) at range 9 m behind it. With the occlusion gate, only
    the front point inherits the moving label; the wall stays static.
    """
    H_px, W_px = 4, 8
    pixel_mask = np.zeros((H_px, W_px), dtype=bool)
    pixel_mask[1, 2] = True

    pixel_range = np.zeros((H_px, W_px), dtype=np.float32)
    pixel_range[1, 2] = 5.0  # winning (closest) range at the moving pixel

    p2px = np.array([[1, 2], [1, 2], [-1, -1]], dtype=np.int32)
    point_ranges = np.array([5.0, 9.0, 0.0], dtype=np.float32)

    out = _unproject_mask(
        pixel_mask,
        p2px,
        n_points=3,
        point_ranges=point_ranges,
        pixel_range=pixel_range,
        occlusion_range_tol_m=1.0,
    )
    assert bool(out[0]) is True, "front-surface point must keep the moving label"
    assert bool(out[1]) is False, "occluded background must not inherit moving label"
    assert bool(out[2]) is False, "out-of-FOV point stays False"

    # Without the gate (legacy), the background point bleeds dynamic.
    out_legacy = _unproject_mask(pixel_mask, p2px, n_points=3)
    assert bool(out_legacy[1]) is True


def test_unproject_mask_occlusion_gate_keeps_thick_object():
    """A point within tol of the winner (same object's depth) stays moving."""
    pixel_mask = np.zeros((4, 8), dtype=bool)
    pixel_mask[1, 2] = True
    pixel_range = np.zeros((4, 8), dtype=np.float32)
    pixel_range[1, 2] = 5.0

    p2px = np.array([[1, 2], [1, 2]], dtype=np.int32)
    # point 1 is 0.6 m behind the winner — within the 1.0 m tolerance.
    point_ranges = np.array([5.0, 5.6], dtype=np.float32)

    out = _unproject_mask(
        pixel_mask,
        p2px,
        n_points=2,
        point_ranges=point_ranges,
        pixel_range=pixel_range,
        occlusion_range_tol_m=1.0,
    )
    assert bool(out[0]) is True
    assert bool(out[1]) is True


# ---------------------------------------------------------------------------
# Group 4: process_chunk integration tests
# ---------------------------------------------------------------------------


def test_process_chunk_disabled_writes_nothing(tmp_env):
    """enabled=False → no _mf_mos_mask.npy files are written."""
    bag_id, chunk_id = "bag_dis", "chunk0"
    _write_calibration(bag_id)
    _write_poses(bag_id, chunk_id, [0])
    _write_lidar_sweeps(bag_id, chunk_id, [(0, _make_in_fov_points(4))])
    _write_proc_index(bag_id, chunk_id, [0])

    cfg = ComponentConfig()  # mf_mos.enabled defaults to False
    result = process_chunk(cfg, bag_id, chunk_id)

    proc_dir = local_path(lidar_proc_dir(bag_id, chunk_id))
    mask_files = [f for f in os.listdir(proc_dir) if "mf_mos_mask" in f]
    assert mask_files == []
    assert result.n_sweeps_skipped_disabled == 1


def test_process_chunk_lidar_id_allowlist_skips_others(tmp_env, monkeypatch):
    """LiDARs not in mf_mos.lidar_id_allowlist are skipped wholesale.

    The fov_up/fov_down/range_image_h/w params are global, so running MF-MOS
    on a LiDAR with a different mount geometry would project sweeps into a
    range image that doesn't resemble KITTI training data. Allowlist gives
    the user a knob to restrict inference to validated LiDARs without
    disabling MF-MOS entirely.

    Setup: single sweep on LIDAR_TOP, allowlist=["lidar_cc"] (LIDAR_TOP not
    listed). Expected: no mask file is written, n_sweeps_skipped_disabled
    counts the rejection (not skipped_invalid, since this is a deliberate
    opt-out — not a pipeline failure).
    """
    monkeypatch.setattr(mf_mos_mod, "_load_model", _stub_load_model)

    bag_id, chunk_id = "bag_allowlist", "chunk0"
    xyz = _make_in_fov_points(4)
    _write_calibration(bag_id)
    _write_poses(bag_id, chunk_id, [0, 1_000_000_000])
    _write_lidar_sweeps(bag_id, chunk_id, [(0, xyz)])
    _write_proc_index(bag_id, chunk_id, [0])

    # LIDAR_TOP is not in the allowlist — should be skipped entirely.
    cfg = _enabled_cfg(lidar_id_allowlist=["lidar_cc"])
    result = process_chunk(cfg, bag_id, chunk_id)

    assert result.n_sweeps_processed == 0
    assert result.n_sweeps_skipped_disabled == 1
    proc_dir = local_path(lidar_proc_dir(bag_id, chunk_id))
    mask_files = [f for f in os.listdir(proc_dir) if "mf_mos_mask" in f]
    assert (
        mask_files == []
    ), f"allowlisted-out LiDAR must write no mask files; got {mask_files}"


def test_process_chunk_lidar_id_allowlist_none_runs_all(tmp_env, monkeypatch):
    """lidar_id_allowlist=None (default) runs MF-MOS on every LiDAR seen."""
    monkeypatch.setattr(mf_mos_mod, "_load_model", _stub_load_model)

    bag_id, chunk_id = "bag_allowlist_none", "chunk0"
    xyz = _make_in_fov_points(4)
    _write_calibration(bag_id)
    _write_poses(bag_id, chunk_id, [0, 1_000_000_000])
    _write_lidar_sweeps(bag_id, chunk_id, [(0, xyz)])
    _write_proc_index(bag_id, chunk_id, [0])

    cfg = _enabled_cfg()  # lidar_id_allowlist defaults to None
    assert cfg.mf_mos.lidar_id_allowlist is None
    result = process_chunk(cfg, bag_id, chunk_id)

    assert result.n_sweeps_processed == 1
    assert result.n_sweeps_skipped_disabled == 0


def test_process_chunk_first_sweep_pads_zero_residuals(tmp_env, monkeypatch):
    """First sweep (no past scans) → zero residuals → stub outputs all-False mask."""
    monkeypatch.setattr(mf_mos_mod, "_load_model", _stub_load_model)

    bag_id, chunk_id = "bag_first", "chunk0"
    xyz = _make_in_fov_points(6)
    _write_calibration(bag_id)
    _write_poses(bag_id, chunk_id, [0, 1_000_000_000])
    _write_lidar_sweeps(bag_id, chunk_id, [(0, xyz)])
    _write_proc_index(bag_id, chunk_id, [0])

    cfg = _enabled_cfg(residual_steps=[1, 2])
    result = process_chunk(cfg, bag_id, chunk_id)

    assert result.n_sweeps_processed == 1
    mask = np.load(local_path(mf_mos_mask_path(bag_id, chunk_id, 0)))
    assert mask.dtype == bool
    assert mask.shape == (xyz.shape[0],)
    assert not mask.any()


def test_process_chunk_pose_gap_writes_zero_mask(tmp_env, monkeypatch):
    """Sweep beyond max_pose_gap_ms from last pose → zero-LENGTH sentinel mask.

    _write_zero_mask writes a (0,) array (not a full-length all-False mask) so
    the sweep is excluded from the chunk-wide vote denominator rather than
    diluting vote fractions for genuine movers in adjacent sweeps.
    """
    monkeypatch.setattr(mf_mos_mod, "_load_model", _stub_load_model)

    bag_id, chunk_id = "bag_gap", "chunk0"
    n_pts = 5
    xyz = _make_in_fov_points(n_pts)
    _write_calibration(bag_id)
    # Poses at t=0 and t=100ms.  sweep_id=8 → header_ts=400ms; gap=300ms > 200ms.
    _write_poses(bag_id, chunk_id, [0, 100_000_000])
    _write_lidar_sweeps(bag_id, chunk_id, [(8, xyz)])
    _write_proc_index(bag_id, chunk_id, [8])

    cfg = _enabled_cfg(max_pose_gap_ms=200)
    result = process_chunk(cfg, bag_id, chunk_id)

    assert result.n_sweeps_skipped_pose == 1
    mask = np.load(local_path(mf_mos_mask_path(bag_id, chunk_id, 8)))
    assert mask.shape == (0,)
    assert not mask.any()


def test_process_chunk_empty_pointcloud_writes_zero_length_mask(tmp_env, monkeypatch):
    """Empty raw cloud → (0,) mask written, inference not called."""
    monkeypatch.setattr(mf_mos_mod, "_load_model", _stub_load_model)

    bag_id, chunk_id = "bag_empty", "chunk0"
    _write_calibration(bag_id)
    _write_poses(bag_id, chunk_id, [0, 1_000_000_000])
    _write_lidar_sweeps(bag_id, chunk_id, [(0, np.empty((0, 3), dtype=np.float32))])
    _write_proc_index(bag_id, chunk_id, [0])

    cfg = _enabled_cfg()
    result = process_chunk(cfg, bag_id, chunk_id)

    assert result.n_sweeps_skipped_empty == 1
    mask = np.load(local_path(mf_mos_mask_path(bag_id, chunk_id, 0)))
    assert mask.shape == (0,)


def test_mask_length_equals_raw_point_count(tmp_env, monkeypatch):
    """For every output mask, length == raw NPZ x.shape[0]."""
    monkeypatch.setattr(mf_mos_mod, "_load_model", _stub_load_model)

    bag_id, chunk_id = "bag_align", "chunk0"
    sweeps = [
        (0, _make_in_fov_points(3)),
        (1, _make_in_fov_points(7)),
        (2, _make_in_fov_points(12)),
    ]
    _write_calibration(bag_id)
    _write_poses(bag_id, chunk_id, [0, 200_000_000])
    _write_lidar_sweeps(bag_id, chunk_id, sweeps)
    _write_proc_index(bag_id, chunk_id, [sid for sid, _ in sweeps])

    cfg = _enabled_cfg(residual_steps=[1])
    result = process_chunk(cfg, bag_id, chunk_id)

    assert result.n_sweeps_processed == len(sweeps)
    for sid, xyz in sweeps:
        mask = np.load(local_path(mf_mos_mask_path(bag_id, chunk_id, sid)))
        assert mask.shape == (
            xyz.shape[0],
        ), f"sweep {sid}: mask len {mask.shape} != raw len {xyz.shape[0]}"


def test_lidar_proc_index_carries_mf_mos_mask_path(tmp_env, monkeypatch):
    """After process_chunk, lidar_proc_index rows have mf_mos_mask_path populated."""
    monkeypatch.setattr(mf_mos_mod, "_load_model", _stub_load_model)

    bag_id, chunk_id = "bag_idx", "chunk0"
    _write_calibration(bag_id)
    _write_poses(bag_id, chunk_id, [0, 1_000_000_000])
    _write_lidar_sweeps(bag_id, chunk_id, [(0, _make_in_fov_points(5))])
    _write_proc_index(bag_id, chunk_id, [0])

    cfg = _enabled_cfg()
    process_chunk(cfg, bag_id, chunk_id)

    rows = read_rows(lidar_proc_index_path(bag_id, chunk_id))
    assert len(rows) == 1
    assert rows[0]["mf_mos_mask_path"] == mf_mos_mask_path(bag_id, chunk_id, 0)


# ---------------------------------------------------------------------------
# Group 5: classify fusion tests
# ---------------------------------------------------------------------------


def _write_world_sweep(
    bag_id: str, chunk_id: str, sweep_id: int, xyz: np.ndarray
) -> None:
    path = local_path(lidar_world_path(bag_id, chunk_id, sweep_id))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2])


def _write_world_sweep_with_origin(
    bag_id: str, chunk_id: str, sweep_id: int, xyz: np.ndarray, origin: np.ndarray
) -> None:
    """Write a world NPZ including the `origin` field required by the log_odds path."""
    path = local_path(lidar_world_path(bag_id, chunk_id, sweep_id))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        origin=origin.astype(np.float64),
    )


def _write_mf_mask(bag_id: str, chunk_id: str, sweep_id: int, mask: np.ndarray) -> str:
    uri = mf_mos_mask_path(bag_id, chunk_id, sweep_id)
    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    np.save(local_path(uri), mask)
    return uri


def _proc_row_with_mask(
    bag_id: str, chunk_id: str, sweep_id: int, xyz: np.ndarray, mf_uri: str | None
) -> dict:
    n = xyz.shape[0]
    return {
        "bag_id": bag_id,
        "chunk_id": chunk_id,
        "sweep_id": sweep_id,
        "lidar_id": "LIDAR_TOP",
        "reference_timestamp_ns": sweep_id * 100_000_000,
        "n_points_total": n,
        "n_points_static": 0,
        "n_points_dynamic": 0,
        "n_points_ground": 0,
        "world_path": lidar_world_path(bag_id, chunk_id, sweep_id),
        "dynamic_mask_path": "",
        "has_intensity": False,
        "deskewed": True,
        "valid": True,
        "drop_reason": None,
        "world_xmin": float(xyz[:, 0].min()) if n else None,
        "world_xmax": float(xyz[:, 0].max()) if n else None,
        "world_ymin": float(xyz[:, 1].min()) if n else None,
        "world_ymax": float(xyz[:, 1].max()) if n else None,
        "world_zmin": float(xyz[:, 2].min()) if n else None,
        "world_zmax": float(xyz[:, 2].max()) if n else None,
        "frame_id": None,
        "mf_mos_mask_path": mf_uri,
    }


def test_fusion_independent_preserves_voxel_mask(tmp_env):
    """fusion_mode='independent': an all-moving MF-MOS mask doesn't change classify."""
    bag_id, chunk_id = "bag_fus_ind", "chunk0"
    n_sweeps = 6
    sensor = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    # Three points seen in every sweep → static by ray-casting.
    static_xyz = np.array([[5.0, 0.0, 0.0], [5.0, 1.0, 0.0], [5.0, -1.0, 0.0]])

    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    rows = []
    for i in range(n_sweeps):
        _write_world_sweep_with_origin(bag_id, chunk_id, i, static_xyz, sensor)
        mf_uri = _write_mf_mask(bag_id, chunk_id, i, np.ones(3, dtype=bool))
        rows.append(_proc_row_with_mask(bag_id, chunk_id, i, static_xyz, mf_uri))
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    cfg = ComponentConfig(
        min_observations=2,
        mf_mos=MFMosParams(enabled=True, fusion_mode="independent"),
    )
    classify_chunk(cfg, bag_id, chunk_id)

    smap = np.load(local_path(static_map_path(bag_id, chunk_id)))
    dmap = np.load(local_path(dynamic_map_path(bag_id, chunk_id)))
    assert smap["xyz"].shape[0] > 0, "static_map should have points"
    assert dmap["xyz"].shape[0] == 0, "independent mode: no points should be dynamic"


def test_fusion_union_ors_signals(tmp_env):
    """fusion_mode='union': a voxel-static but MF-MOS-moving point lands in dynamic_map."""
    bag_id, chunk_id = "bag_fus_union", "chunk0"
    n_sweeps = 6
    sensor = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    static_xyz = np.array([[5.0, 0.0, 0.0], [5.0, 1.0, 0.0], [5.0, -1.0, 0.0]])

    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    rows = []
    for i in range(n_sweeps):
        _write_world_sweep_with_origin(bag_id, chunk_id, i, static_xyz, sensor)
        # MF-MOS says point 0 (5, 0, 0) is moving; points 1 and 2 are not.
        mf_uri = _write_mf_mask(bag_id, chunk_id, i, np.array([True, False, False]))
        rows.append(_proc_row_with_mask(bag_id, chunk_id, i, static_xyz, mf_uri))
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    cfg = ComponentConfig(
        min_observations=2,
        mf_mos=MFMosParams(enabled=True, fusion_mode="union"),
    )
    classify_chunk(cfg, bag_id, chunk_id)

    dmap = np.load(local_path(dynamic_map_path(bag_id, chunk_id)))
    assert dmap["xyz"].shape[0] > 0, "at least one point should be in dynamic_map"
    assert all(
        abs(x - 5.0) < 0.01 and abs(y) < 0.01
        for x, y in zip(dmap["xyz"][:, 0], dmap["xyz"][:, 1])
    ), "only the MF-MOS-moving point at (5,0,0) should be in dynamic_map"


def test_fusion_union_log_odds_path(tmp_env):
    """fusion_mode='union' with log_odds: a voxel static by ray-casting but voted dynamic
    by MF-MOS across enough sweeps must appear in dynamic_map.

    This is the production code path (log_odds + voxel-level vote aggregation in Pass 1,
    classify_from_log_odds, then masking.py).  The persistence-mode tests above don't
    exercise it.

    Setup: 8 sweeps, sensor at origin, single static point at (5, 0, 0).
      - log_odds after 8 hits: ~6.8 → p_occ ≈ 0.999 → classified static.
      - MF-MOS mask: True for that point in every sweep → 8/8 votes → passes thresholds.
      - Expected: union mode fuses the MF-MOS vote and the point lands in dynamic_map.
    """
    bag_id, chunk_id = "bag_fus_union_lo", "chunk0"
    n_sweeps = 8
    sensor_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    # Three distinct points; only point 0 (5, 0, 0) is flagged by MF-MOS.
    static_xyz = np.array([[5.0, 0.0, 0.0], [5.0, 1.0, 0.0], [5.0, -1.0, 0.0]])

    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    rows = []
    for i in range(n_sweeps):
        _write_world_sweep_with_origin(bag_id, chunk_id, i, static_xyz, sensor_origin)
        mf_uri = _write_mf_mask(bag_id, chunk_id, i, np.array([True, False, False]))
        rows.append(_proc_row_with_mask(bag_id, chunk_id, i, static_xyz, mf_uri))
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    cfg = ComponentConfig(
        mf_mos=MFMosParams(enabled=True, fusion_mode="union"),
    )
    classify_chunk(cfg, bag_id, chunk_id)

    dmap = np.load(local_path(dynamic_map_path(bag_id, chunk_id)))
    assert (
        dmap["xyz"].shape[0] > 0
    ), "union mode: MF-MOS-flagged point in a log_odds-static voxel must appear in dynamic_map"
    xs, ys = dmap["xyz"][:, 0], dmap["xyz"][:, 1]
    assert any(
        abs(x - 5.0) < 0.01 and abs(y) < 0.01 for x, y in zip(xs, ys)
    ), "point at (5.0, 0.0) must be in dynamic_map (MF-MOS union flipped it from static)"


def test_fusion_independent_log_odds_path(tmp_env):
    """fusion_mode='independent' with log_odds: MF-MOS all-dynamic vote must NOT override
    the voxel classifier.  This is the negative control for test_fusion_union_log_odds_path.
    """
    bag_id, chunk_id = "bag_fus_ind_lo", "chunk0"
    n_sweeps = 8
    sensor_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    static_xyz = np.array([[5.0, 0.0, 0.0], [5.0, 1.0, 0.0], [5.0, -1.0, 0.0]])

    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    rows = []
    for i in range(n_sweeps):
        _write_world_sweep_with_origin(bag_id, chunk_id, i, static_xyz, sensor_origin)
        mf_uri = _write_mf_mask(bag_id, chunk_id, i, np.ones(3, dtype=bool))
        rows.append(_proc_row_with_mask(bag_id, chunk_id, i, static_xyz, mf_uri))
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    cfg = ComponentConfig(
        mf_mos=MFMosParams(enabled=True, fusion_mode="independent"),
    )
    classify_chunk(cfg, bag_id, chunk_id)

    smap = np.load(local_path(static_map_path(bag_id, chunk_id)))
    dmap = np.load(local_path(dynamic_map_path(bag_id, chunk_id)))
    assert (
        smap["xyz"].shape[0] > 0
    ), "log_odds should classify repeatedly-seen points as static"
    assert (
        dmap["xyz"].shape[0] == 0
    ), "independent mode: MF-MOS all-dynamic vote must not override log_odds static label"


# ---------------------------------------------------------------------------
# Group 6: pipeline regression test
# ---------------------------------------------------------------------------


def test_pipeline_runs_with_mf_mos_disabled_unchanged(tmp_env):
    """_process_one_chunk with mf_mos disabled produces no mf_mos artifacts."""
    from wato_lidar_preprocessing.pipeline import _process_one_chunk

    bag_id, chunk_id = "bag_pipe", "chunk0"

    _write_calibration(bag_id)
    _write_poses(bag_id, chunk_id, [0])

    sw_path = local_path(lidar_sweep_path(bag_id, chunk_id, "LIDAR_TOP", 0))
    os.makedirs(os.path.dirname(sw_path), exist_ok=True)
    np.savez_compressed(
        sw_path,
        x=np.array([1.0, 2.0], dtype=np.float32),
        y=np.zeros(2, dtype=np.float32),
        z=np.zeros(2, dtype=np.float32),
    )

    write_table(
        [
            {
                "bag_id": bag_id,
                "chunk_id": chunk_id,
                "lidar_id": "LIDAR_TOP",
                "sweep_id": 0,
                "lidar_path": lidar_sweep_path(bag_id, chunk_id, "LIDAR_TOP", 0),
                "header_timestamp_ns": 0,
                "record_timestamp_ns": 0,
                "num_points": 2,
                "has_ring": False,
                "has_intensity": False,
                "has_point_time": False,
                "min_range_m": 1.0,
                "max_range_m": 2.0,
                "valid": True,
                "drop_reason": None,
            }
        ],
        LIDAR_SWEEPS_SCHEMA,
        lidar_sweeps_path(bag_id, chunk_id),
    )

    cfg = ComponentConfig()  # mf_mos.enabled defaults to False
    _, ok, err = _process_one_chunk(cfg, bag_id, chunk_id)
    assert ok, f"chunk failed: {err}"

    proc_dir = local_path(lidar_proc_dir(bag_id, chunk_id))
    mf_files = [f for f in os.listdir(proc_dir) if "mf_mos" in f]
    assert mf_files == [], f"unexpected mf_mos files: {mf_files}"


# ---------------------------------------------------------------------------
# Group 7: temporal-bleed regression (chunk-wide vote arr applied per-sweep)
# ---------------------------------------------------------------------------


def test_fusion_no_temporal_bleed_log_odds(tmp_env):
    """Per-sweep MF-MOS mask must not bleed across sweeps in the log-odds path.

    Regression: classify_from_log_odds returns mf_mos_dynamic_arr aggregated
    across the chunk (voxels passing vote thresholds at any time).  The old
    Pass-2 code applied that chunk-wide arr to every sweep, so a voxel a car
    passed through in one sweep would flag every static point landing in the
    same voxel in any other sweep — a temporal shadow of the moving object.

    Setup:
      - 2 sweeps, sensor at origin, same 3 points at (5, 0, 0), (5, 1, 0),
        (5, -1, 0).  Voxel V0 contains (5, 0, 0) in both sweeps.
      - Sweep 0: MF-MOS flags point 0 as moving; points 1, 2 static.
      - Sweep 1: MF-MOS flags NOTHING as moving.
      - V0 chunk-wide: 1 vote / 2 hits = 0.5 fraction (passes default
        thresholds: min_mf_mos_votes=1, vote_fraction>=0.5).

    Before fix (mfmos_only mode): mask = is_mf_mos_dyn where is_mf_mos_dyn
    is searchsort against chunk-wide arr; sweep 1's point 0 lands in V0,
    flips to dynamic.  Expected: 2 dynamic points (one per sweep, same xyz).
    After fix: sweep 1's per-sweep mask says no point is dynamic, so its
    sweep_mf_mos_dynamic_arr is empty.  Only sweep 0's flagged point
    survives — exactly 1 dynamic point.
    """
    bag_id, chunk_id = "bag_temporal_bleed", "chunk0"
    sensor_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    xyz = np.array([[5.0, 0.0, 0.0], [5.0, 1.0, 0.0], [5.0, -1.0, 0.0]])

    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    rows = []
    _write_world_sweep_with_origin(bag_id, chunk_id, 0, xyz, sensor_origin)
    mf_uri_0 = _write_mf_mask(bag_id, chunk_id, 0, np.array([True, False, False]))
    rows.append(_proc_row_with_mask(bag_id, chunk_id, 0, xyz, mf_uri_0))
    _write_world_sweep_with_origin(bag_id, chunk_id, 1, xyz, sensor_origin)
    mf_uri_1 = _write_mf_mask(bag_id, chunk_id, 1, np.zeros(3, dtype=bool))
    rows.append(_proc_row_with_mask(bag_id, chunk_id, 1, xyz, mf_uri_1))
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    cfg = ComponentConfig(
        min_observations=1,
        mf_mos=MFMosParams(enabled=True, fusion_mode="mfmos_only"),
    )
    classify_chunk(cfg, bag_id, chunk_id)

    dmap = np.load(local_path(dynamic_map_path(bag_id, chunk_id)))
    assert dmap["xyz"].shape[0] == 1, (
        f"expected 1 dynamic point (sweep 0's MF-MOS-flagged point only); "
        f"got {dmap['xyz'].shape[0]} — per-sweep mask must not bleed across sweeps"
    )
    assert int(dmap["sweep_id"][0]) == 0, (
        f"the surviving dynamic point must be from sweep 0 (the MF-MOS-flagged "
        f"one); got sweep_id={int(dmap['sweep_id'][0])}"
    )


def test_fusion_per_sweep_flag_flips_single_point(tmp_env):
    """A per-sweep MF-MOS flag flips exactly that sweep's point, no vote tier.

    4 sweeps; only sweep 0 flags the (5,0,0) point. The (denoised) per-sweep
    mask drives fusion directly — so exactly one dynamic point survives, from
    sweep 0. There is no chunk-wide vote/fraction threshold to satisfy: a fast
    mover seen briefly is not suppressed here (temporal confirmation is the
    tracker's job downstream).
    """
    bag_id, chunk_id = "bag_per_sweep_flag", "chunk0"
    sensor_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    xyz = np.array([[5.0, 0.0, 0.0], [5.0, 1.0, 0.0]])

    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    rows = []
    for i in range(4):
        _write_world_sweep_with_origin(bag_id, chunk_id, i, xyz, sensor_origin)
        mask = np.array([True, False]) if i == 0 else np.zeros(2, dtype=bool)
        mf_uri = _write_mf_mask(bag_id, chunk_id, i, mask)
        rows.append(_proc_row_with_mask(bag_id, chunk_id, i, xyz, mf_uri))
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    cfg = ComponentConfig(
        min_observations=1,
        mf_mos=MFMosParams(enabled=True, fusion_mode="mfmos_only"),
    )
    classify_chunk(cfg, bag_id, chunk_id)

    dmap = np.load(local_path(dynamic_map_path(bag_id, chunk_id)))
    assert dmap["xyz"].shape[0] == 1, (
        f"expected 1 dynamic point from sweep 0's per-sweep flag; "
        f"got {dmap['xyz'].shape[0]}"
    )
    assert int(dmap["sweep_id"][0]) == 0


# ---------------------------------------------------------------------------
# Group 8: per-sweep 3D cluster denoise (replaces the old chunk-wide vote tier)
# ---------------------------------------------------------------------------


def test_denoise_moving_mask_drops_isolated_points_keeps_blobs():
    """The 3D cluster filter removes isolated moving points and keeps dense blobs."""
    from wato_lidar_preprocessing.mf_mos._core import _denoise_moving_mask_3d

    # A 10-point dense blob near the origin, plus two far isolated speckle
    # points. cluster_voxel=0.5, min_cluster_pts=4.
    blob = np.random.RandomState(0).uniform(-0.2, 0.2, size=(10, 3))
    speckle = np.array([[50.0, 0.0, 0.0], [-50.0, 0.0, 0.0]])
    xyz = np.vstack([blob, speckle])
    mask = np.ones(xyz.shape[0], dtype=bool)

    out = _denoise_moving_mask_3d(
        mask, xyz, cluster_voxel_m=0.5, min_cluster_pts=4
    )

    assert out[:10].all(), "the 10-point blob must survive the size filter"
    assert not out[10] and not out[11], "isolated speckle points must be dropped"


def test_denoise_moving_mask_min_cluster_one_is_noop():
    """min_cluster_pts<=1 leaves the mask unchanged (denoise disabled)."""
    from wato_lidar_preprocessing.mf_mos._core import _denoise_moving_mask_3d

    xyz = np.array([[50.0, 0.0, 0.0], [-50.0, 0.0, 0.0]])
    mask = np.array([True, True])
    out = _denoise_moving_mask_3d(mask, xyz, cluster_voxel_m=0.5, min_cluster_pts=1)
    assert out.tolist() == [True, True]


# ---------------------------------------------------------------------------
# Group 9: residual-window priming from the prior chunk (cold-start fix)
# ---------------------------------------------------------------------------


def _write_chunks_index(bag_id: str, specs: list[tuple[str, int, int]]) -> None:
    """specs: list of (chunk_id, t_start_ns, t_end_ns)."""
    rows = [
        {
            "bag_id": bag_id,
            "chunk_id": cid,
            "t_start_ns": ts,
            "t_end_ns": te,
            "t_overlap_start_ns": ts,
            "t_overlap_end_ns": te,
        }
        for cid, ts, te in specs
    ]
    write_table(rows, CHUNK_SCHEMA, chunks_index_path(bag_id))


def _moving_scene_sweep(sid: int) -> np.ndarray:
    """A radially-moving point (2 m/sweep outward at azimuth 0) plus static
    filler. The mover keeps a stable pixel, so its range changes sweep-to-sweep
    produce a non-zero residual that the stub model reads as 'moving'."""
    mover = np.array([[5.0 + 2.0 * sid, 0.0, 0.0]], dtype=np.float32)
    return np.vstack([mover, _make_in_fov_points(6)])


def test_prime_window_recovers_first_sweep_residual(tmp_env, monkeypatch):
    """chunk1's first sweep gets a non-zero residual via prior-chunk priming.

    chunk0 holds sweeps 0..3, chunk1 holds sweeps 4..5, of a radially-moving
    object. With residual_steps=[1], chunk1's first sweep (sid=4) needs the
    sweep 1 step back (sid=3, in chunk0). Without priming that slot is a
    cold-start zero and the mover is missed; with priming the prior chunk's
    tail fills it and the mover is detected.
    """
    monkeypatch.setattr(mf_mos_mod, "_load_model", _stub_load_model)

    bag_id = "bag_prime"
    _write_calibration(bag_id)
    # Two chunks ordered by t_start_ns; sweep header ts = sid * 50 ms.
    _write_chunks_index(
        bag_id,
        [("chunk0", 0, 200_000_000), ("chunk1", 200_000_000, 400_000_000)],
    )

    chunk0_sweeps = [(sid, _moving_scene_sweep(sid)) for sid in range(4)]
    chunk1_sweeps = [(sid, _moving_scene_sweep(sid)) for sid in (4, 5)]

    # Poses (identity) at every sweep timestamp in each chunk.
    _write_poses(bag_id, "chunk0", [sid * 50_000_000 for sid in range(4)])
    _write_poses(bag_id, "chunk1", [sid * 50_000_000 for sid in (4, 5)])
    _write_lidar_sweeps(bag_id, "chunk0", chunk0_sweeps)
    _write_lidar_sweeps(bag_id, "chunk1", chunk1_sweeps)
    _write_proc_index(bag_id, "chunk0", [sid for sid, _ in chunk0_sweeps])
    _write_proc_index(bag_id, "chunk1", [sid for sid, _ in chunk1_sweeps])

    cfg = _enabled_cfg(residual_steps=[1], prime_window_from_prior_chunk=True)
    process_chunk(cfg, bag_id, "chunk1")

    # sid=4 is chunk1's first sweep; mover is point index 0.
    mask = np.load(local_path(mf_mos_mask_path(bag_id, "chunk1", 4)))
    assert bool(mask[0]), (
        "primed window: chunk1's first sweep should detect the mover using the "
        "prior chunk's tail to fill the residual channel"
    )


def test_no_prime_window_first_sweep_cold_starts(tmp_env, monkeypatch):
    """With priming disabled, chunk1's first sweep cold-starts (zero residual)
    and misses the mover — the negative control for the priming test."""
    monkeypatch.setattr(mf_mos_mod, "_load_model", _stub_load_model)

    bag_id = "bag_no_prime"
    _write_calibration(bag_id)
    _write_chunks_index(
        bag_id,
        [("chunk0", 0, 200_000_000), ("chunk1", 200_000_000, 400_000_000)],
    )
    chunk0_sweeps = [(sid, _moving_scene_sweep(sid)) for sid in range(4)]
    chunk1_sweeps = [(sid, _moving_scene_sweep(sid)) for sid in (4, 5)]
    _write_poses(bag_id, "chunk0", [sid * 50_000_000 for sid in range(4)])
    _write_poses(bag_id, "chunk1", [sid * 50_000_000 for sid in (4, 5)])
    _write_lidar_sweeps(bag_id, "chunk0", chunk0_sweeps)
    _write_lidar_sweeps(bag_id, "chunk1", chunk1_sweeps)
    _write_proc_index(bag_id, "chunk0", [sid for sid, _ in chunk0_sweeps])
    _write_proc_index(bag_id, "chunk1", [sid for sid, _ in chunk1_sweeps])

    cfg = _enabled_cfg(residual_steps=[1], prime_window_from_prior_chunk=False)
    process_chunk(cfg, bag_id, "chunk1")

    mask = np.load(local_path(mf_mos_mask_path(bag_id, "chunk1", 4)))
    assert not mask.any(), (
        "no priming: chunk1's first sweep has no past scan, so the residual is "
        "zero and the mover is missed"
    )
