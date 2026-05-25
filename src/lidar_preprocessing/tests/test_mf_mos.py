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
    _unproject_scores,
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

    def infer(self, range_image: np.ndarray, residual_images: list[np.ndarray]) -> np.ndarray:
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
            "x": 0.0, "y": 0.0, "z": 0.0,
            "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
            "world_T_ego_flat": np.eye(4).flatten().tolist(),
            "source": "odom",
            "valid": True,
        }
        for ts in timestamps_ns
    ]
    write_table(rows, POSES_SCHEMA, poses_path(bag_id, chunk_id))


def _write_raw_sweep(
    bag_id: str, chunk_id: str, sweep_id: int, xyz: np.ndarray
) -> str:
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
        rows.append({
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
            "max_range_m": float(np.linalg.norm(xyz, axis=1).max()) if n > 0 else 0.0,
            "valid": True,
            "drop_reason": None,
        })
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
        assert r_winner <= r_i + 1e-4, (
            f"point {i}: winner range {r_winner:.4f} > self range {r_i:.4f}"
        )


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
    assert mask_files == [], (
        f"allowlisted-out LiDAR must write no mask files; got {mask_files}"
    )


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
    """Sweep beyond max_pose_gap_ms from last pose → all-False mask, skipped_pose=1."""
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
    assert mask.shape == (n_pts,)
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
    sweeps = [(0, _make_in_fov_points(3)), (1, _make_in_fov_points(7)), (2, _make_in_fov_points(12))]
    _write_calibration(bag_id)
    _write_poses(bag_id, chunk_id, [0, 200_000_000])
    _write_lidar_sweeps(bag_id, chunk_id, sweeps)
    _write_proc_index(bag_id, chunk_id, [sid for sid, _ in sweeps])

    cfg = _enabled_cfg(residual_steps=[1])
    result = process_chunk(cfg, bag_id, chunk_id)

    assert result.n_sweeps_processed == len(sweeps)
    for sid, xyz in sweeps:
        mask = np.load(local_path(mf_mos_mask_path(bag_id, chunk_id, sid)))
        assert mask.shape == (xyz.shape[0],), (
            f"sweep {sid}: mask len {mask.shape} != raw len {xyz.shape[0]}"
        )


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


def _write_world_sweep(bag_id: str, chunk_id: str, sweep_id: int, xyz: np.ndarray) -> None:
    path = local_path(lidar_world_path(bag_id, chunk_id, sweep_id))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2])


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
    """fusion_mode='independent': MF-MOS mask saying all-moving doesn't change classify."""
    bag_id, chunk_id = "bag_fus_ind", "chunk0"
    n_sweeps = 6
    # Three points seen in every sweep → static by voxel.
    static_xyz = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    rows = []
    for i in range(n_sweeps):
        _write_world_sweep(bag_id, chunk_id, i, static_xyz)
        mf_uri = _write_mf_mask(bag_id, chunk_id, i, np.ones(3, dtype=bool))
        rows.append(_proc_row_with_mask(bag_id, chunk_id, i, static_xyz, mf_uri))
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    cfg = ComponentConfig(
        static_sweep_fraction=0.3,
        static_sweep_min=2,
        classification_method="persistence",
        mf_mos=MFMosParams(enabled=True, fusion_mode="independent"),
    )
    classify_chunk(cfg, bag_id, chunk_id)

    smap = np.load(local_path(static_map_path(bag_id, chunk_id)))
    dmap = np.load(local_path(dynamic_map_path(bag_id, chunk_id)))
    assert smap["xyz"].shape[0] > 0, "static_map should have points"
    assert dmap["xyz"].shape[0] == 0, "independent mode: no points should be dynamic"


def test_fusion_union_ors_signals(tmp_env):
    """fusion_mode='union': voxel-static but MF-MOS-moving point lands in dynamic_map."""
    bag_id, chunk_id = "bag_fus_union", "chunk0"
    n_sweeps = 6
    # Three points seen in every sweep → static by voxel.
    static_xyz = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    rows = []
    for i in range(n_sweeps):
        _write_world_sweep(bag_id, chunk_id, i, static_xyz)
        # MF-MOS says point 0 (x=1.0) is moving; points 1 and 2 are not.
        mf_uri = _write_mf_mask(bag_id, chunk_id, i, np.array([True, False, False]))
        rows.append(_proc_row_with_mask(bag_id, chunk_id, i, static_xyz, mf_uri))
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    cfg = ComponentConfig(
        static_sweep_fraction=0.3,
        static_sweep_min=2,
        classification_method="persistence",
        mf_mos=MFMosParams(enabled=True, fusion_mode="union"),
    )
    classify_chunk(cfg, bag_id, chunk_id)

    dmap = np.load(local_path(dynamic_map_path(bag_id, chunk_id)))
    assert dmap["xyz"].shape[0] > 0, "at least one point should be in dynamic_map"
    assert pytest.approx(float(dmap["xyz"][:, 0].min()), abs=0.01) == 1.0, (
        "point at x=1.0 (MF-MOS moving) should be in dynamic_map"
    )


# ---------------------------------------------------------------------------
# Group 6: pipeline regression test
# ---------------------------------------------------------------------------


def test_pipeline_runs_with_mf_mos_disabled_unchanged(tmp_env):
    """_process_one_chunk with mf_mos disabled produces no mf_mos artifacts."""
    from wato_common.artifact_store import chunks_index_path
    from wato_common.schemas import CHUNK_SCHEMA
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
                "bag_id": bag_id, "chunk_id": chunk_id,
                "lidar_id": "LIDAR_TOP", "sweep_id": 0,
                "lidar_path": lidar_sweep_path(bag_id, chunk_id, "LIDAR_TOP", 0),
                "header_timestamp_ns": 0, "record_timestamp_ns": 0,
                "num_points": 2, "has_ring": False, "has_intensity": False,
                "has_point_time": False, "min_range_m": 1.0, "max_range_m": 2.0,
                "valid": True, "drop_reason": None,
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
