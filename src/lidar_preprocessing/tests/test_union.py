"""Tests for the `union` segmentation method (union/segment.py).

All CPU-runnable: union.classify_chunk reads AW's static_map.npz and the
per-sweep MF-MOS masks off disk, so no numba (ray traversal) or torch
(inference) is needed. The tests stage those two artifacts directly and assert
the fused dynamic cloud.

Fusion contract under test:
    dynamic = mf_mos_moving & ~ground & ~aw_static            (default)
    dynamic = (mf_mos_moving | aw_dynamic) & ~ground & ~aw_static
                                                     (keep_aw_dynamic=True)
static_map.npz is kept verbatim (AW owns the static cloud).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from wato_common.artifact_store import (
    dynamic_map_path,
    dynamic_mask_path,
    ensure_local_dir,
    lidar_proc_dir,
    lidar_proc_index_path,
    lidar_world_path,
    local_path,
    mf_mos_mask_path,
    static_map_path,
)
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import PROCESSED_SWEEPS_SCHEMA
from wato_lidar_preprocessing.config import ComponentConfig, UnionParams
from wato_lidar_preprocessing.union import classify_chunk
from wato_lidar_preprocessing.voxel import voxel_indices

VOXEL = 1.0
ORIGIN = np.zeros(3, dtype=np.float64)


@pytest.fixture()
def tmp_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", str(tmp_path))
    return tmp_path


def _union_cfg(**union_kw) -> ComponentConfig:
    return ComponentConfig(
        segmentation="union",
        voxel_size_m=VOXEL,
        union=UnionParams(**union_kw),
    )


def _write_world(
    bag_id: str,
    chunk_id: str,
    sweep_id: int,
    xyz: np.ndarray,
    *,
    ground_mask: np.ndarray | None = None,
    intensity: np.ndarray | None = None,
) -> None:
    path = local_path(lidar_world_path(bag_id, chunk_id, sweep_id))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    kwargs = dict(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], origin=ORIGIN)
    if ground_mask is not None:
        kwargs["ground_mask"] = ground_mask
    if intensity is not None:
        kwargs["intensity"] = intensity.astype(np.float32)
    np.savez_compressed(path, **kwargs)


def _write_static_map(
    bag_id: str, chunk_id: str, static_xyz: np.ndarray | None
) -> None:
    """Stage an AW-style static_map.npz. static_voxel_keys are derived from
    static_xyz under the same origin/voxel_size union will use."""
    if static_xyz is None or static_xyz.shape[0] == 0:
        keys = np.empty(0, dtype=np.int64)
        xyz = np.empty((0, 3), dtype=np.float64)
    else:
        keys = np.unique(voxel_indices(static_xyz, ORIGIN, VOXEL))
        xyz = static_xyz
    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    np.savez_compressed(
        local_path(static_map_path(bag_id, chunk_id)),
        xyz=xyz,
        voxel_size=np.float32(VOXEL),
        origin=ORIGIN,
        static_voxel_keys=keys,
    )


def _write_mf_mask(
    bag_id: str, chunk_id: str, sweep_id: int, mask: np.ndarray
) -> str:
    uri = mf_mos_mask_path(bag_id, chunk_id, sweep_id)
    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    np.save(local_path(uri), mask)
    return uri


def _proc_row(
    bag_id: str,
    chunk_id: str,
    sweep_id: int,
    xyz: np.ndarray,
    mf_uri: str | None,
    *,
    n_static: int = 0,
    has_intensity: bool = False,
) -> dict:
    n = xyz.shape[0]
    return {
        "bag_id": bag_id,
        "chunk_id": chunk_id,
        "sweep_id": sweep_id,
        "lidar_id": "LIDAR_TOP",
        "reference_timestamp_ns": sweep_id * 100_000_000,
        "n_points_total": n,
        "n_points_static": n_static,
        "n_points_dynamic": 0,
        "n_points_ground": 0,
        "world_path": lidar_world_path(bag_id, chunk_id, sweep_id),
        "dynamic_mask_path": "",
        "has_intensity": has_intensity,
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


# One point per voxel along +x: P_i sits in voxel (i, 0, 0).
_XYZ = np.array(
    [[0.5, 0.0, 0.0], [1.5, 0.0, 0.0], [2.5, 0.0, 0.0], [3.5, 0.0, 0.0]]
)


def test_union_veto_drops_mfmos_on_static(tmp_env):
    """Core contract: a MF-MOS-moving point whose voxel AW called static is
    vetoed; moving points off the static map survive."""
    bag_id, chunk_id = "bag_union_veto", "chunk0"
    # AW static = voxels of P1 and P3.
    _write_static_map(bag_id, chunk_id, _XYZ[[1, 3]])
    # MF-MOS says P0, P1, P2 moving (P1 is on static → must be vetoed).
    mf_uri = _write_mf_mask(
        bag_id, chunk_id, 0, np.array([True, True, True, False])
    )
    _write_world(bag_id, chunk_id, 0, _XYZ)
    write_table(
        [_proc_row(bag_id, chunk_id, 0, _XYZ, mf_uri, n_static=2)],
        PROCESSED_SWEEPS_SCHEMA,
        lidar_proc_index_path(bag_id, chunk_id),
    )

    result = classify_chunk(_union_cfg(), bag_id, chunk_id)

    assert result.n_dynamic == 2  # P0, P2
    assert result.n_vetoed == 1  # P1 dropped by AW-static veto
    assert result.n_static == 2  # preserved from index
    dmap = np.load(local_path(dynamic_map_path(bag_id, chunk_id)))
    assert sorted(dmap["xyz"][:, 0].tolist()) == [0.5, 2.5]
    mask = np.load(local_path(dynamic_mask_path(bag_id, chunk_id, 0)))
    assert mask.tolist() == [True, False, True, False]


def test_union_no_veto_equals_raw_mfmos(tmp_env):
    """aw_static_veto=False → union == raw MF-MOS dynamic (ground-removed)."""
    bag_id, chunk_id = "bag_union_noveto", "chunk0"
    _write_static_map(bag_id, chunk_id, _XYZ[[1, 3]])
    mf_uri = _write_mf_mask(
        bag_id, chunk_id, 0, np.array([True, True, True, False])
    )
    _write_world(bag_id, chunk_id, 0, _XYZ)
    write_table(
        [_proc_row(bag_id, chunk_id, 0, _XYZ, mf_uri)],
        PROCESSED_SWEEPS_SCHEMA,
        lidar_proc_index_path(bag_id, chunk_id),
    )

    result = classify_chunk(_union_cfg(aw_static_veto=False), bag_id, chunk_id)

    assert result.n_dynamic == 3  # P0, P1, P2 — no veto
    assert result.n_vetoed == 0


def test_union_ground_never_dynamic(tmp_env):
    """A ground point flagged moving by MF-MOS is never dynamic, even off the
    static map."""
    bag_id, chunk_id = "bag_union_ground", "chunk0"
    _write_static_map(bag_id, chunk_id, _XYZ[[3]])
    ground = np.array([True, False, False, False])  # P0 ground
    mf_uri = _write_mf_mask(
        bag_id, chunk_id, 0, np.array([True, True, False, False])
    )
    _write_world(bag_id, chunk_id, 0, _XYZ, ground_mask=ground)
    write_table(
        [_proc_row(bag_id, chunk_id, 0, _XYZ, mf_uri)],
        PROCESSED_SWEEPS_SCHEMA,
        lidar_proc_index_path(bag_id, chunk_id),
    )

    result = classify_chunk(_union_cfg(), bag_id, chunk_id)

    assert result.n_dynamic == 1  # only P1 (P0 ground-suppressed)
    mask = np.load(local_path(dynamic_mask_path(bag_id, chunk_id, 0)))
    assert mask.tolist() == [False, True, False, False]


def test_union_keep_aw_dynamic_unions_with_existing_mask(tmp_env):
    """keep_aw_dynamic=True unions AW's own dynamic_mask.npy (written by the AW
    pass) with the MF-MOS verdict, still ground- and static-vetoed."""
    bag_id, chunk_id = "bag_union_keepaw", "chunk0"
    _write_static_map(bag_id, chunk_id, _XYZ[[3]])  # P3 static
    # AW dynamic mask (pre-existing): P2 dynamic. MF-MOS: P0 moving.
    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    np.save(
        local_path(dynamic_mask_path(bag_id, chunk_id, 0)),
        np.array([False, False, True, True]),  # P2, P3 — P3 must be vetoed
    )
    mf_uri = _write_mf_mask(
        bag_id, chunk_id, 0, np.array([True, False, False, False])
    )
    _write_world(bag_id, chunk_id, 0, _XYZ)
    write_table(
        [_proc_row(bag_id, chunk_id, 0, _XYZ, mf_uri)],
        PROCESSED_SWEEPS_SCHEMA,
        lidar_proc_index_path(bag_id, chunk_id),
    )

    result = classify_chunk(_union_cfg(keep_aw_dynamic=True), bag_id, chunk_id)

    # union(P0 from MF-MOS, {P2,P3} from AW) minus AW-static(P3) = {P0, P2}.
    assert result.n_dynamic == 2
    mask = np.load(local_path(dynamic_mask_path(bag_id, chunk_id, 0)))
    assert mask.tolist() == [True, False, True, False]


def test_union_missing_mfmos_mask_no_dynamics(tmp_env):
    """A sweep with no MF-MOS mask fabricates no dynamics (default mode)."""
    bag_id, chunk_id = "bag_union_nomask", "chunk0"
    _write_static_map(bag_id, chunk_id, _XYZ[[3]])
    _write_world(bag_id, chunk_id, 0, _XYZ)
    write_table(
        [_proc_row(bag_id, chunk_id, 0, _XYZ, None)],  # mf_mos_mask_path=None
        PROCESSED_SWEEPS_SCHEMA,
        lidar_proc_index_path(bag_id, chunk_id),
    )

    result = classify_chunk(_union_cfg(), bag_id, chunk_id)

    assert result.n_dynamic == 0
    assert result.n_sweeps_no_mask == 1


def test_union_preserves_static_map(tmp_env):
    """union rewrites only the dynamic side — static_map.npz is byte-identical
    before and after, and n_static is summed from the index."""
    bag_id, chunk_id = "bag_union_static", "chunk0"
    _write_static_map(bag_id, chunk_id, _XYZ[[1, 3]])
    sm_path = local_path(static_map_path(bag_id, chunk_id))
    before = open(sm_path, "rb").read()

    mf_uri = _write_mf_mask(
        bag_id, chunk_id, 0, np.array([True, False, False, False])
    )
    _write_world(bag_id, chunk_id, 0, _XYZ)
    write_table(
        [_proc_row(bag_id, chunk_id, 0, _XYZ, mf_uri, n_static=2)],
        PROCESSED_SWEEPS_SCHEMA,
        lidar_proc_index_path(bag_id, chunk_id),
    )

    result = classify_chunk(_union_cfg(), bag_id, chunk_id)

    assert open(sm_path, "rb").read() == before  # untouched
    assert result.n_static == 2
    # the index row's n_points_dynamic reflects the fused count
    rows = read_rows(lidar_proc_index_path(bag_id, chunk_id))
    assert rows[0]["n_points_dynamic"] == 1
    assert rows[0]["n_points_static"] == 2


def test_union_multi_sweep_totals(tmp_env):
    """Totals accumulate across sweeps; intensity rides along when present."""
    bag_id, chunk_id = "bag_union_multi", "chunk0"
    _write_static_map(bag_id, chunk_id, _XYZ[[3]])
    rows = []
    for sid in range(3):
        intens = np.arange(4, dtype=np.float32)
        _write_world(bag_id, chunk_id, sid, _XYZ, intensity=intens)
        mf_uri = _write_mf_mask(
            bag_id, chunk_id, sid, np.array([True, False, True, True])
        )
        rows.append(
            _proc_row(bag_id, chunk_id, sid, _XYZ, mf_uri, has_intensity=True)
        )
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    result = classify_chunk(_union_cfg(), bag_id, chunk_id)

    # per sweep: moving {P0,P2,P3} minus static {P3} = {P0,P2} → 2 × 3 = 6
    assert result.n_dynamic == 6
    assert result.n_vetoed == 3
    dmap = np.load(local_path(dynamic_map_path(bag_id, chunk_id)))
    assert dmap["xyz"].shape[0] == 6
    assert "intensity" in dmap
    assert dmap["intensity"].shape[0] == 6
