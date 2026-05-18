"""Tests for ground.py (Step C — per-sweep ground mask aggregation + height grid).

Step C no longer runs Patchwork++ — that was moved to deskew.py to operate on
sensor-frame data per sweep.  Step C reads world_*.npz files written by
deskew (which include a `ground_mask` key when Patchwork++ was available)
and aggregates them into ground.npz.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from wato_common.artifact_store import (
    ground_path,
    lidar_proc_index_path,
    lidar_world_path,
    local_path,
    static_map_path,
)
from wato_common.io.parquet_io import write_table
from wato_common.schemas import PROCESSED_SWEEPS_SCHEMA
from wato_lidar_preprocessing.config import ComponentConfig
from wato_lidar_preprocessing.ground import _build_height_grid, process_chunk


def test_flat_plane_height_grid():
    """Flat z=0 ground → height grid should be approximately zero."""
    rng = np.random.default_rng(42)
    xy = rng.uniform(-10, 10, size=(500, 2))
    z = np.zeros(500)
    xyz = np.column_stack([xy, z])

    hg, ng, origin = _build_height_grid(xyz, cell_size=1.0)
    assert hg.shape[0] > 0 and hg.shape[1] > 0
    np.testing.assert_allclose(hg, 0.0, atol=1e-5)


def test_tilted_plane_height_grid():
    """Tilted plane z = 0.1*x → grid should reflect the slope."""
    xs = np.linspace(0, 9, 50)
    ys = np.zeros(50)
    zs = 0.1 * xs
    xyz = np.column_stack([xs, ys, zs])

    hg, ng, origin = _build_height_grid(xyz, cell_size=1.0)
    # Each column i should have z ≈ 0.1 * i (coarsely).
    for col in range(min(hg.shape[1], 5)):
        expected_z = 0.1 * col
        np.testing.assert_allclose(hg[0, col], expected_z, atol=0.15)


def test_empty_input_returns_valid_grid():
    hg, ng, origin = _build_height_grid(np.empty((0, 3)), cell_size=1.0)
    assert hg.shape == (1, 1)
    assert ng.shape == (1, 1, 3)


def test_height_grid_axis_convention():
    """Locks in (row=y, col=x) layout.

    A point at (x=3, y=0) must end up in column 3, row 0 (with cell_size=1.0
    and origin at (0, 0)).  This is the contract every downstream consumer
    of `height_grid` relies on; if the binned_statistic_2d transpose is ever
    accidentally removed, this test catches it.
    """
    pts = np.array(
        [
            [0.0, 0.0, 5.0],  # (x=0, y=0)  → cell (row=0, col=0)
            [3.0, 0.0, 7.0],  # (x=3, y=0)  → cell (row=0, col=3)
            [0.0, 4.0, 9.0],  # (x=0, y=4)  → cell (row=4, col=0)
        ]
    )
    hg, _, origin = _build_height_grid(pts, cell_size=1.0)
    # origin is the lower-left in (x, y) world units.
    np.testing.assert_allclose(origin, [0.0, 0.0])
    # Sample our seed points back out at their predicted (row, col).
    assert hg[0, 0] == np.float32(5.0)
    assert hg[0, 3] == np.float32(7.0)
    assert hg[4, 0] == np.float32(9.0)


def test_normals_unit_length():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 5, size=(200, 2))
    z = np.zeros(200)
    xyz = np.column_stack([xy, z])
    _, ng, _ = _build_height_grid(xyz, cell_size=1.0)
    norms = np.linalg.norm(ng.reshape(-1, 3), axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


@pytest.fixture()
def tmp_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", str(tmp_path))
    return tmp_path


def _write_world_sweep_with_mask(
    bag_id: str,
    chunk_id: str,
    sweep_id: int,
    xyz: np.ndarray,
    ground_mask: np.ndarray | None,
):
    path = local_path(lidar_world_path(bag_id, chunk_id, sweep_id))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    kwargs = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    if ground_mask is not None:
        kwargs["ground_mask"] = ground_mask
    np.savez_compressed(path, **kwargs)


def _write_static_map_covering_all(
    bag_id: str,
    chunk_id: str,
    all_xyz: np.ndarray,
    voxel_size: float = 0.15,
):
    """Write a static_map.npz whose voxel keys cover every point in `all_xyz`.

    Ground tests that don't care about the static intersection use this to
    pass every candidate ground point through the filter.
    """
    from wato_lidar_preprocessing.voxel import voxel_indices

    origin = np.zeros(3, dtype=np.float64) if all_xyz.size == 0 else all_xyz.min(axis=0)
    if all_xyz.shape[0] > 0:
        keys = np.unique(voxel_indices(all_xyz, origin, voxel_size, chunk_id=chunk_id))
    else:
        keys = np.empty(0, dtype=np.int64)
    path = local_path(static_map_path(bag_id, chunk_id))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        xyz=all_xyz,
        origin=origin,
        voxel_size=np.float32(voxel_size),
        static_voxel_keys=keys,
    )


def _write_proc_index(bag_id: str, chunk_id: str, sweep_ids: list[int]):
    rows = []
    for sid in sweep_ids:
        rows.append(
            {
                "bag_id": bag_id,
                "chunk_id": chunk_id,
                "sweep_id": sid,
                "lidar_id": "LIDAR_TOP",
                "reference_timestamp_ns": sid * 100_000_000,
                "n_points_total": 0,
                "n_points_static": 0,
                "n_points_dynamic": 0,
                "world_path": lidar_world_path(bag_id, chunk_id, sid),
                "dynamic_mask_path": "",
                "has_intensity": False,
                "deskewed": True,
            }
        )
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))


def test_aggregates_per_sweep_ground_masks(tmp_env):
    """Sweeps with ground_mask → ground.npz contains masked points + height grid."""
    bag_id, chunk_id = "bag_g0", "chunk0"
    rng = np.random.default_rng(7)
    # Each sweep: 100 ground points at z≈0, 50 obstacle points at z=2.
    n_per_sweep = 3
    all_pts: list[np.ndarray] = []
    for sid in range(n_per_sweep):
        ground_xy = rng.uniform(-10, 10, size=(100, 2))
        ground_z = rng.normal(0.0, 0.02, size=100)
        ground = np.column_stack([ground_xy, ground_z])
        obstacle_xy = rng.uniform(-5, 5, size=(50, 2))
        obstacle = np.column_stack([obstacle_xy, np.full(50, 2.0)])
        xyz = np.concatenate([ground, obstacle])
        mask = np.concatenate([np.ones(100, bool), np.zeros(50, bool)])
        _write_world_sweep_with_mask(bag_id, chunk_id, sid, xyz, mask)
        all_pts.append(
            xyz[mask]
        )  # ground points only — those need to pass intersection
    _write_proc_index(bag_id, chunk_id, list(range(n_per_sweep)))
    _write_static_map_covering_all(bag_id, chunk_id, np.concatenate(all_pts))

    cfg = ComponentConfig()
    result = process_chunk(cfg, bag_id, chunk_id)
    assert result.status == "ok"
    assert result.n_ground == 300
    assert result.n_nonground == 150

    data = np.load(local_path(ground_path(bag_id, chunk_id)))
    assert data["height_grid"].shape[0] > 0
    assert data["height_grid"].shape[1] > 0
    # Ground is at z≈0, so most height_grid cells should be near zero.
    np.testing.assert_allclose(np.median(data["height_grid"]), 0.0, atol=0.1)
    assert str(data["status"]) == "ok"


def test_no_masks_writes_sentinel(tmp_env):
    """World NPZs without ground_mask → sentinel ground.npz with status flag."""
    bag_id, chunk_id = "bag_g_no_mask", "chunk0"
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    _write_world_sweep_with_mask(bag_id, chunk_id, 0, xyz, ground_mask=None)
    _write_proc_index(bag_id, chunk_id, [0])
    _write_static_map_covering_all(bag_id, chunk_id, xyz)

    cfg = ComponentConfig()
    result = process_chunk(cfg, bag_id, chunk_id)
    assert result.status == "skipped_no_ground_mask"
    assert result.n_ground == 0

    data = np.load(local_path(ground_path(bag_id, chunk_id)))
    assert data["ground_xyz"].shape == (0, 3)
    assert str(data["status"]) == "skipped_no_ground_mask"


def test_empty_proc_index_writes_sentinel(tmp_env):
    """No proc_index rows → 'empty' status sentinel."""
    bag_id, chunk_id = "bag_g_empty", "chunk0"
    write_table([], PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    cfg = ComponentConfig()
    result = process_chunk(cfg, bag_id, chunk_id)
    assert result.status == "empty"
    assert result.n_ground == 0


def _write_static_voxel_set(
    bag_id: str, chunk_id: str, static_xyz: np.ndarray, voxel_size: float
):
    """Write a static_map.npz with static_voxel_keys derived from static_xyz."""
    from wato_lidar_preprocessing.voxel import voxel_indices

    origin = (
        np.zeros(3, dtype=np.float64)
        if static_xyz.size == 0
        else static_xyz.min(axis=0)
    )
    if static_xyz.shape[0] > 0:
        keys = voxel_indices(static_xyz, origin, voxel_size, chunk_id=chunk_id)
        keys = np.unique(keys)
    else:
        keys = np.empty(0, dtype=np.int64)
    path = local_path(static_map_path(bag_id, chunk_id))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        xyz=static_xyz,
        origin=origin,
        voxel_size=np.float32(voxel_size),
        static_voxel_keys=keys,
    )


def test_intersection_drops_dynamic_ground_points(tmp_env):
    """Ground points whose voxel ended up dynamic must be dropped from ground.npz.

    Setup: one sweep with two ground-flagged points.  Only one of them
    falls in a voxel that classify marked static.  The other is filtered
    out, n_dropped_dynamic reports it, and ground.npz contains only the
    survivor.
    """
    bag_id, chunk_id = "bag_g_inter", "chunk0"
    # Two ground points; only the first is in the static voxel set.
    static_pt = np.array([[10.0, 10.0, 0.0]])
    dynamic_pt = np.array([[100.0, 100.0, 0.0]])
    sweep_xyz = np.concatenate([static_pt, dynamic_pt], axis=0)
    ground_mask = np.array([True, True])  # both flagged ground by Patchwork++
    _write_world_sweep_with_mask(bag_id, chunk_id, 0, sweep_xyz, ground_mask)
    _write_proc_index(bag_id, chunk_id, [0])

    # static_map.npz contains only static_pt.  Voxel size 1 m so the two
    # points land in clearly distinct voxels.
    _write_static_voxel_set(bag_id, chunk_id, static_pt, voxel_size=1.0)

    cfg = ComponentConfig(voxel_size_m=1.0)
    result = process_chunk(cfg, bag_id, chunk_id)
    assert result.status == "ok"
    assert result.n_ground == 1, "only the static-voxel point should survive"
    assert result.n_dropped_dynamic == 1

    data = np.load(local_path(ground_path(bag_id, chunk_id)))
    np.testing.assert_allclose(data["ground_xyz"], static_pt)


def test_intersection_raises_on_legacy_static_map(tmp_env):
    """Legacy static_map.npz without static_voxel_keys → loud KeyError
    pointing at the re-run command, not silent passthrough."""
    bag_id, chunk_id = "bag_g_legacy", "chunk0"
    rng = np.random.default_rng(3)
    xy = rng.uniform(-5, 5, size=(50, 2))
    z = np.zeros(50)
    xyz = np.column_stack([xy, z])
    mask = np.ones(50, dtype=bool)
    _write_world_sweep_with_mask(bag_id, chunk_id, 0, xyz, mask)
    _write_proc_index(bag_id, chunk_id, [0])

    path = local_path(static_map_path(bag_id, chunk_id))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, xyz=xyz, origin=np.zeros(3), voxel_size=np.float32(0.15))

    cfg = ComponentConfig()
    with pytest.raises(KeyError, match="re-run lidar_preprocessing with --force"):
        process_chunk(cfg, bag_id, chunk_id)


def test_intersection_raises_on_missing_static_map(tmp_env):
    """No static_map.npz at all → FileNotFoundError, not a silent skip."""
    bag_id, chunk_id = "bag_g_no_static", "chunk0"
    rng = np.random.default_rng(3)
    xy = rng.uniform(-5, 5, size=(10, 2))
    z = np.zeros(10)
    xyz = np.column_stack([xy, z])
    mask = np.ones(10, dtype=bool)
    _write_world_sweep_with_mask(bag_id, chunk_id, 0, xyz, mask)
    _write_proc_index(bag_id, chunk_id, [0])

    cfg = ComponentConfig()
    with pytest.raises(FileNotFoundError, match="run Step B"):
        process_chunk(cfg, bag_id, chunk_id)


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("pypatchworkpp"),
    reason="pypatchworkpp not installed",
)
def test_deskew_estimates_ground_in_sensor_frame(tmp_env):
    """End-to-end: deskew runs Patchwork++ on a synthetic flat plane and
    ground.process_chunk aggregates the resulting masks."""
    import json
    from wato_common.artifact_store import (
        calibration_path,
        ensure_local_dir,
        lidar_proc_dir,
        lidar_sweep_path,
        lidar_sweeps_path,
        poses_path,
    )
    from wato_common.schemas import LIDAR_SWEEPS_SCHEMA, POSES_SCHEMA
    from wato_lidar_preprocessing.deskew import process_chunk as deskew_chunk

    bag_id, chunk_id = "bag_e2e", "chunk0"
    # Calibration: identity ego_T_lidar.
    calib = {
        "calibration_version": "test",
        "ego_frame": "base_link",
        "cameras": {},
        "lidars": {
            "LIDAR_TOP": {
                "frame_id": "v",
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

    # Identity pose.
    pose_row = {
        "bag_id": bag_id,
        "chunk_id": chunk_id,
        "timestamp_ns": 0,
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
    write_table([pose_row], POSES_SCHEMA, poses_path(bag_id, chunk_id))

    # Synthetic flat plane: 2000 ground points + 100 obstacle points.
    rng = np.random.default_rng(7)
    xy = rng.uniform(-20, 20, size=(2000, 2)).astype(np.float32)
    z = rng.normal(-1.8, 0.02, size=2000).astype(
        np.float32
    )  # ground at sensor_height below sensor
    xy2 = rng.uniform(-5, 5, size=(100, 2)).astype(np.float32)
    z2 = np.full(100, 0.0, dtype=np.float32)
    x = np.concatenate([xy[:, 0], xy2[:, 0]])
    y = np.concatenate([xy[:, 1], xy2[:, 1]])
    z_all = np.concatenate([z, z2])

    sw_path = local_path(lidar_sweep_path(bag_id, chunk_id, "LIDAR_TOP", 0))
    os.makedirs(os.path.dirname(sw_path), exist_ok=True)
    np.savez_compressed(sw_path, x=x, y=y, z=z_all)

    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
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
                "num_points": x.shape[0],
                "has_ring": False,
                "has_intensity": False,
                "has_point_time": False,
                "min_range_m": 1.0,
                "max_range_m": 25.0,
                "valid": True,
                "drop_reason": None,
            }
        ],
        LIDAR_SWEEPS_SCHEMA,
        lidar_sweeps_path(bag_id, chunk_id),
    )

    cfg = ComponentConfig()
    deskew_chunk(cfg, bag_id, chunk_id)

    # World NPZ should contain a ground_mask (Patchwork++ ran).
    world = np.load(local_path(lidar_world_path(bag_id, chunk_id, 0)))
    assert "ground_mask" in world
    assert world["ground_mask"].sum() > 1000  # most ground points detected
    world_xyz = np.column_stack([world["x"], world["y"], world["z"]])
    _write_static_map_covering_all(bag_id, chunk_id, world_xyz)

    result = process_chunk(cfg, bag_id, chunk_id)
    assert result.status == "ok"
    assert result.n_ground > 1000
