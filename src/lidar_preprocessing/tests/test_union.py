"""Tests for the `union` segmentation method (union/segment.py).

All CPU-runnable: union.classify_chunk reads AW's static_map.npz and the
per-sweep MF-MOS masks off disk, so no numba (ray traversal) or torch
(inference) is needed. The tests stage those two artifacts directly and assert
the fused dynamic cloud.

Fusion contract under test:
    dynamic = mf_mos_moving & ~ground & ~near_ego & ~aw_static   (default)
    dynamic = (mf_mos_moving | aw_dynamic) & ~ground & ~near_ego & ~aw_static
                                                     (keep_aw_dynamic=True)
static_map.npz is kept verbatim (AW owns the static cloud).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from wato_common.artifact_store import (
    aw_dynamic_mask_path,
    dynamic_map_path,
    dynamic_mask_path,
    ensure_local_dir,
    ground_path,
    lidar_proc_dir,
    lidar_proc_index_path,
    lidar_sweep_path,
    lidar_world_path,
    local_path,
    mf_mos_mask_path,
    mf_mos_score_path,
    static_map_path,
)
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import PROCESSED_SWEEPS_SCHEMA
from wato_lidar_preprocessing.config import (
    ComponentConfig,
    MotionFilterParams,
    UnionParams,
)
from wato_lidar_preprocessing.union import classify_chunk
from wato_lidar_preprocessing.voxel import voxel_indices

VOXEL = 1.0
ORIGIN = np.zeros(3, dtype=np.float64)


@pytest.fixture()
def tmp_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", str(tmp_path))
    return tmp_path


def _union_cfg(*, dynamic_min_range_m: float = 0.0, **union_kw) -> ComponentConfig:
    # The staged points sit 0.5–3.5 m from the sweep origin and on z=0, so
    # the near-ego gate, the veto dilation, and the ground-height veto are
    # disabled here and exercised by their dedicated tests below. The post-veto
    # motion filter is likewise disabled so these tests isolate the veto logic —
    # it has its own coverage in test_motion_filter.py (the few-sweep synthetic
    # clouds here would otherwise be wiped by the persistence/coherence gates).
    union_kw.setdefault("veto_dilation_voxels", 0)
    union_kw.setdefault("ground_height_veto_m", 0.0)
    union_kw.setdefault("motion_filter", MotionFilterParams(enabled=False))
    return ComponentConfig(
        segmentation="union",
        voxel_size_m=VOXEL,
        dynamic_min_range_m=dynamic_min_range_m,
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
    bag_id: str,
    chunk_id: str,
    static_xyz: np.ndarray | None,
    *,
    dynamic_xyz: np.ndarray | None = None,
) -> None:
    """Stage an AW-style static_map.npz. static_voxel_keys are derived from
    static_xyz under the same origin/voxel_size union will use. When
    dynamic_xyz is None the dynamic_voxel_keys field is omitted entirely,
    exercising the legacy (pre-export) fallback."""
    if static_xyz is None or static_xyz.shape[0] == 0:
        keys = np.empty(0, dtype=np.int64)
        xyz = np.empty((0, 3), dtype=np.float64)
    else:
        keys = np.unique(voxel_indices(static_xyz, ORIGIN, VOXEL))
        xyz = static_xyz
    kwargs = {}
    if dynamic_xyz is not None:
        kwargs["dynamic_voxel_keys"] = np.unique(
            voxel_indices(dynamic_xyz, ORIGIN, VOXEL)
        )
    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    np.savez_compressed(
        local_path(static_map_path(bag_id, chunk_id)),
        xyz=xyz,
        voxel_size=np.float32(VOXEL),
        origin=ORIGIN,
        static_voxel_keys=keys,
        **kwargs,
    )


def _write_ground_grid(bag_id: str, chunk_id: str, z: float = 0.0) -> None:
    """Stage a flat Step-C ground.npz covering x,y ∈ [-8, 8] at height z."""
    cell = 0.5
    hw = int(16 / cell)
    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    np.savez_compressed(
        local_path(ground_path(bag_id, chunk_id)),
        height_grid=np.full((hw, hw), z, dtype=np.float32),  # (H=y, W=x)
        grid_origin=np.array([-8.0, -8.0]),
        cell_size=np.float32(cell),
        ground_xyz=np.empty((0, 3), dtype=np.float64),
        status=np.array("ok"),
    )


def _write_mf_mask(bag_id: str, chunk_id: str, sweep_id: int, mask: np.ndarray) -> str:
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
_XYZ = np.array([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0], [2.5, 0.0, 0.0], [3.5, 0.0, 0.0]])


def test_union_veto_drops_mfmos_on_static(tmp_env):
    """Core contract: a MF-MOS-moving point whose voxel AW called static is
    vetoed; moving points off the static map survive."""
    bag_id, chunk_id = "bag_union_veto", "chunk0"
    # AW static = voxels of P1 and P3.
    _write_static_map(bag_id, chunk_id, _XYZ[[1, 3]])
    # MF-MOS says P0, P1, P2 moving (P1 is on static → must be vetoed).
    mf_uri = _write_mf_mask(bag_id, chunk_id, 0, np.array([True, True, True, False]))
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
    mf_uri = _write_mf_mask(bag_id, chunk_id, 0, np.array([True, True, True, False]))
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
    mf_uri = _write_mf_mask(bag_id, chunk_id, 0, np.array([True, True, False, False]))
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
    """keep_aw_dynamic=True unions AW's dynamic verdict with the MF-MOS one,
    still ground- and static-vetoed. No aw_dynamic_mask.npy snapshot is staged
    here, so this also covers the legacy fallback to dynamic_mask.npy."""
    bag_id, chunk_id = "bag_union_keepaw", "chunk0"
    _write_static_map(bag_id, chunk_id, _XYZ[[3]])  # P3 static
    # AW dynamic mask (pre-existing): P2 dynamic. MF-MOS: P0 moving.
    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    np.save(
        local_path(dynamic_mask_path(bag_id, chunk_id, 0)),
        np.array([False, False, True, True]),  # P2, P3 — P3 must be vetoed
    )
    mf_uri = _write_mf_mask(bag_id, chunk_id, 0, np.array([True, False, False, False]))
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

    mf_uri = _write_mf_mask(bag_id, chunk_id, 0, np.array([True, False, False, False]))
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
        rows.append(_proc_row(bag_id, chunk_id, sid, _XYZ, mf_uri, has_intensity=True))
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    result = classify_chunk(_union_cfg(), bag_id, chunk_id)

    # per sweep: moving {P0,P2,P3} minus static {P3} = {P0,P2} → 2 × 3 = 6
    assert result.n_dynamic == 6
    assert result.n_vetoed == 3
    dmap = np.load(local_path(dynamic_map_path(bag_id, chunk_id)))
    assert dmap["xyz"].shape[0] == 6
    assert "intensity" in dmap
    assert dmap["intensity"].shape[0] == 6


def test_union_near_ego_gate_suppresses_close_movers(tmp_env):
    """MF-MOS movers within dynamic_min_range_m of the sweep origin are
    suppressed — AW's static map can't veto ego self-returns (near-ego voxels
    are carved, never static), so union must apply the gate itself."""
    bag_id, chunk_id = "bag_union_nearego", "chunk0"
    _write_static_map(bag_id, chunk_id, None)  # empty static map: no veto
    # All four points moving; P0 (0.5 m) and P1 (1.5 m) are inside the gate.
    mf_uri = _write_mf_mask(bag_id, chunk_id, 0, np.array([True, True, True, True]))
    _write_world(bag_id, chunk_id, 0, _XYZ)  # world origin == ORIGIN (zeros)
    write_table(
        [_proc_row(bag_id, chunk_id, 0, _XYZ, mf_uri)],
        PROCESSED_SWEEPS_SCHEMA,
        lidar_proc_index_path(bag_id, chunk_id),
    )

    result = classify_chunk(_union_cfg(dynamic_min_range_m=2.0), bag_id, chunk_id)

    assert result.n_dynamic == 2  # P2 (2.5 m), P3 (3.5 m)
    mask = np.load(local_path(dynamic_mask_path(bag_id, chunk_id, 0)))
    assert mask.tolist() == [False, False, True, True]


def test_union_veto_score_exempt_keeps_confident_movers(tmp_env):
    """veto_score_exempt: a static-voxel point whose MF-MOS moving probability
    clears the threshold survives the veto (parked-then-moving case); one
    below it is still vetoed."""
    bag_id, chunk_id = "bag_union_exempt", "chunk0"
    _write_static_map(bag_id, chunk_id, _XYZ[[1, 3]])  # P1, P3 voxel-static
    mf_uri = _write_mf_mask(bag_id, chunk_id, 0, np.array([True, True, False, True]))
    np.save(
        local_path(mf_mos_score_path(bag_id, chunk_id, 0)),
        np.array([0.1, 0.95, 0.0, 0.2], dtype=np.float32),
    )
    _write_world(bag_id, chunk_id, 0, _XYZ)
    write_table(
        [_proc_row(bag_id, chunk_id, 0, _XYZ, mf_uri)],
        PROCESSED_SWEEPS_SCHEMA,
        lidar_proc_index_path(bag_id, chunk_id),
    )

    result = classify_chunk(_union_cfg(veto_score_exempt=0.9), bag_id, chunk_id)

    # P0 moving off-static → kept. P1 on-static, score 0.95 ≥ 0.9 → exempt.
    # P3 on-static, score 0.2 < 0.9 → vetoed.
    assert result.n_dynamic == 2
    assert result.n_vetoed == 1
    mask = np.load(local_path(dynamic_mask_path(bag_id, chunk_id, 0)))
    assert mask.tolist() == [True, True, False, False]


def test_union_keep_aw_dynamic_prefers_snapshot(tmp_env):
    """keep_aw_dynamic reads the aw_dynamic_mask.npy snapshot, NOT
    dynamic_mask.npy — the latter holds the fused verdict after a union run,
    so re-fusing must not feed union's own output back in."""
    bag_id, chunk_id = "bag_union_snapshot", "chunk0"
    _write_static_map(bag_id, chunk_id, None)
    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    # AW snapshot: only P2 dynamic. Poisoned dynamic_mask.npy (a stale fused
    # verdict): everything dynamic — must be ignored.
    np.save(
        local_path(aw_dynamic_mask_path(bag_id, chunk_id, 0)),
        np.array([False, False, True, False]),
    )
    np.save(
        local_path(dynamic_mask_path(bag_id, chunk_id, 0)),
        np.array([True, True, True, True]),
    )
    mf_uri = _write_mf_mask(bag_id, chunk_id, 0, np.array([True, False, False, False]))
    _write_world(bag_id, chunk_id, 0, _XYZ)
    write_table(
        [_proc_row(bag_id, chunk_id, 0, _XYZ, mf_uri)],
        PROCESSED_SWEEPS_SCHEMA,
        lidar_proc_index_path(bag_id, chunk_id),
    )

    result = classify_chunk(_union_cfg(keep_aw_dynamic=True), bag_id, chunk_id)

    # MF-MOS {P0} ∪ AW snapshot {P2} — not the poisoned all-True mask.
    assert result.n_dynamic == 2
    mask = np.load(local_path(dynamic_mask_path(bag_id, chunk_id, 0)))
    assert mask.tolist() == [True, False, True, False]


def test_union_ground_height_veto_drops_road_skimmers(tmp_env):
    """Candidates below ground_height_veto_m over the Step C ground grid are
    dropped — MF-MOS road false positives (including below-grade artifacts)
    that the AW-static veto can never reach (road voxels are never static)."""
    bag_id, chunk_id = "bag_union_groundh", "chunk0"
    _write_static_map(bag_id, chunk_id, None)
    _write_ground_grid(bag_id, chunk_id, z=0.0)
    # P0 skims the road (0.1 m), P1 is below grade (-0.2 m), P2 is at object
    # height (1.0 m). All flagged moving by MF-MOS.
    xyz = np.array([[3.0, 0.0, 0.1], [3.5, 0.0, -0.2], [3.0, 1.0, 1.0]])
    mf_uri = _write_mf_mask(bag_id, chunk_id, 0, np.array([True, True, True]))
    _write_world(bag_id, chunk_id, 0, xyz)
    write_table(
        [_proc_row(bag_id, chunk_id, 0, xyz, mf_uri)],
        PROCESSED_SWEEPS_SCHEMA,
        lidar_proc_index_path(bag_id, chunk_id),
    )

    result = classify_chunk(_union_cfg(ground_height_veto_m=0.25), bag_id, chunk_id)

    assert result.n_dynamic == 1
    assert result.n_ground_vetoed == 2
    mask = np.load(local_path(dynamic_mask_path(bag_id, chunk_id, 0)))
    assert mask.tolist() == [False, False, True]


def test_union_ground_height_veto_skipped_without_ground_npz(tmp_env):
    """No ground.npz on disk → the ground-height veto is skipped (warned),
    never fabricated — all movers survive."""
    bag_id, chunk_id = "bag_union_nogrid", "chunk0"
    _write_static_map(bag_id, chunk_id, None)
    xyz = np.array([[3.0, 0.0, 0.05], [3.0, 1.0, 1.0]])
    mf_uri = _write_mf_mask(bag_id, chunk_id, 0, np.array([True, True]))
    _write_world(bag_id, chunk_id, 0, xyz)
    write_table(
        [_proc_row(bag_id, chunk_id, 0, xyz, mf_uri)],
        PROCESSED_SWEEPS_SCHEMA,
        lidar_proc_index_path(bag_id, chunk_id),
    )

    result = classify_chunk(_union_cfg(ground_height_veto_m=0.25), bag_id, chunk_id)

    assert result.n_dynamic == 2
    assert result.n_ground_vetoed == 0


def test_union_dilated_veto_catches_adjacent_voxel(tmp_env):
    """veto_dilation_voxels=1 vetoes a mover one voxel off AW-static — the
    leakage shell straddling the voxel boundary. A mover three voxels away
    survives. static_map has no dynamic_voxel_keys here, also covering the
    legacy (pre-export) fallback."""
    bag_id, chunk_id = "bag_union_dilate", "chunk0"
    _write_static_map(bag_id, chunk_id, _XYZ[[3]])  # static voxel (3,0,0)
    # P0 (voxel 0): far from static. P2 (voxel 2): adjacent to static.
    mf_uri = _write_mf_mask(bag_id, chunk_id, 0, np.array([True, False, True, False]))
    _write_world(bag_id, chunk_id, 0, _XYZ)
    write_table(
        [_proc_row(bag_id, chunk_id, 0, _XYZ, mf_uri)],
        PROCESSED_SWEEPS_SCHEMA,
        lidar_proc_index_path(bag_id, chunk_id),
    )

    result = classify_chunk(_union_cfg(veto_dilation_voxels=1), bag_id, chunk_id)

    assert result.n_dynamic == 1  # only P0
    assert result.n_vetoed == 1  # P2, via the dilated shell
    mask = np.load(local_path(dynamic_mask_path(bag_id, chunk_id, 0)))
    assert mask.tolist() == [True, False, False, False]


def test_union_dilated_veto_exempts_aw_dynamic_voxels(tmp_env):
    """A candidate adjacent to AW-static whose own voxel AW classed dynamic
    (dynamic_voxel_keys) is exempt from the dilated veto — AW corroborates
    the motion, so the neighbour must not delete it."""
    bag_id, chunk_id = "bag_union_exempt_dyn", "chunk0"
    # Static voxel (3,0,0); P2's voxel (2,0,0) is in AW's dynamic set.
    _write_static_map(bag_id, chunk_id, _XYZ[[3]], dynamic_xyz=_XYZ[[2]])
    mf_uri = _write_mf_mask(bag_id, chunk_id, 0, np.array([False, False, True, False]))
    _write_world(bag_id, chunk_id, 0, _XYZ)
    write_table(
        [_proc_row(bag_id, chunk_id, 0, _XYZ, mf_uri)],
        PROCESSED_SWEEPS_SCHEMA,
        lidar_proc_index_path(bag_id, chunk_id),
    )

    result = classify_chunk(_union_cfg(veto_dilation_voxels=1), bag_id, chunk_id)

    assert result.n_dynamic == 1  # P2 survives via the AW-dynamic exemption
    assert result.n_vetoed == 0
    mask = np.load(local_path(dynamic_mask_path(bag_id, chunk_id, 0)))
    assert mask.tolist() == [False, False, True, False]


def test_union_mask_realigned_when_raw_longer_than_world(tmp_env):
    """An MF-MOS mask aligned to the raw sweep (pre-nonfinite-filter) is
    realigned to the world cloud by re-applying the filter."""
    bag_id, chunk_id = "bag_union_realign", "chunk0"
    _write_static_map(bag_id, chunk_id, None)

    # Raw sweep: 5 points, index 1 is non-finite → world keeps raw [0,2,3,4].
    raw_x = np.array([0.5, np.nan, 1.5, 2.5, 3.5], dtype=np.float32)
    raw_path = local_path(lidar_sweep_path(bag_id, chunk_id, "LIDAR_TOP", 0))
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    np.savez_compressed(
        raw_path,
        x=raw_x,
        y=np.zeros(5, dtype=np.float32),
        z=np.zeros(5, dtype=np.float32),
    )
    # Raw-length mask: raw {0, 2} moving → world {0, 1} after the filter.
    mf_uri = _write_mf_mask(
        bag_id, chunk_id, 0, np.array([True, False, True, False, False])
    )
    _write_world(bag_id, chunk_id, 0, _XYZ)  # 4-point world cloud
    write_table(
        [_proc_row(bag_id, chunk_id, 0, _XYZ, mf_uri)],
        PROCESSED_SWEEPS_SCHEMA,
        lidar_proc_index_path(bag_id, chunk_id),
    )

    result = classify_chunk(_union_cfg(), bag_id, chunk_id)

    assert result.n_sweeps_no_mask == 0  # mask was reconciled, not dropped
    assert result.n_dynamic == 2
    mask = np.load(local_path(dynamic_mask_path(bag_id, chunk_id, 0)))
    assert mask.tolist() == [True, True, False, False]


def test_union_motion_filter_drops_persistent_blob(tmp_env):
    """Integration: with motion_filter enabled, classify_chunk drops a blob that
    dwells in one voxel across many sweeps (which the AW-static veto can't catch
    because the blob is never AW-static) while keeping a translating mover."""
    bag_id, chunk_id = "bag_union_mf", "chunk0"
    n_sweeps = 6
    # Static map deliberately excludes both the blob and the mover, so the veto
    # passes them through and the motion filter is the only thing that can act.
    _write_static_map(bag_id, chunk_id, _XYZ[[1, 3]])
    rows = []
    for s in range(n_sweeps):
        mover = [s * 1.0 + 0.5, 5.0, 0.0]  # distinct voxel each sweep
        blob = [10.5, 10.5, 0.0]  # same voxel every sweep
        xyz = np.array([mover, blob])
        _write_world(bag_id, chunk_id, s, xyz)
        mf_uri = _write_mf_mask(bag_id, chunk_id, s, np.array([True, True]))  # moving
        rows.append(_proc_row(bag_id, chunk_id, s, xyz, mf_uri))
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    cfg = _union_cfg()
    # Explicit threshold (6-sweep blob must exceed it) — the shipped default is
    # looser and recall-biased; this test pins the wiring, not the tuning.
    cfg.union.motion_filter = MotionFilterParams(
        enabled=True, persistence_max_sweeps=5
    )
    result = classify_chunk(cfg, bag_id, chunk_id)

    # Mover (6 pts, one per sweep) survives; blob (6 pts, one voxel) is dropped.
    assert result.n_dynamic == n_sweeps
    assert result.n_persistence_dropped == n_sweeps
    assert result.n_coherence_dropped == 0
    dm = np.load(local_path(dynamic_map_path(bag_id, chunk_id)))
    assert dm["xyz"].shape[0] == n_sweeps
    assert np.all(dm["xyz"][:, 1] == 5.0), "survivors are the mover, not the blob"
