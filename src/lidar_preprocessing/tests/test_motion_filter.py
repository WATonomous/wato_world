"""Tests for union/motion_filter.py (post-veto persistence + coherence gates).

The filter rewrites a chunk's already-written union artifacts in place, so the
end-to-end tests build a minimal dynamic_map.npz + per-sweep dynamic_mask.npy +
lidar_proc_index.parquet (the exact set union.classify_chunk produces) and
assert the rewrite is internally consistent: dynamic_map, per-sweep masks, and
index n_points_dynamic all agree afterwards.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from wato_common.artifact_store import (
    dynamic_map_path,
    dynamic_mask_path,
    lidar_proc_index_path,
    local_path,
)
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import PROCESSED_SWEEPS_SCHEMA
from wato_lidar_preprocessing.config import ComponentConfig, MotionFilterParams
from wato_lidar_preprocessing.union.motion_filter import (
    coherence_keep,
    compute_keep,
    filter_dynamic_artifacts,
    persistence_keep,
)


@pytest.fixture()
def tmp_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", str(tmp_path))
    return tmp_path


# --------------------------------------------------------------------------- #
# Pure-function gates
# --------------------------------------------------------------------------- #


def test_persistence_keeps_movers_drops_dwellers():
    """A point translating through voxels survives; a voxel-dwelling blob drops."""
    mover = np.array([[s * 1.0, 0.0, 0.5] for s in range(10)])
    mover_sw = np.arange(10)
    blob = np.array([[50.0, 50.0, 1.0]] * 10)  # same voxel, every sweep
    blob_sw = np.arange(10)
    xyz = np.vstack([mover, blob])
    sw = np.concatenate([mover_sw, blob_sw]).astype(np.int32)

    keep = persistence_keep(xyz, sw, voxel_m=0.5, max_sweeps=5)
    assert keep[:10].all(), "mover sweeps through voxels — must survive"
    assert not keep[10:].any(), "blob occupies one voxel across 10 sweeps — must drop"


def test_persistence_disabled_is_noop():
    xyz = np.array([[50.0, 50.0, 1.0]] * 10)
    sw = np.arange(10).astype(np.int32)
    keep = persistence_keep(xyz, sw, voxel_m=0.5, max_sweeps=0)
    assert keep.all()


def test_coherence_keeps_track_drops_speck():
    """Points forming a multi-sweep track survive; a one-sweep speck drops."""
    # A coherent track: one cluster per sweep, translating < link_gate per step.
    track = np.array([[s * 0.5, 0.0, 0.5] for s in range(6)])
    track_sw = np.arange(6)
    # An isolated speck only present in a single sweep, far away.
    speck = np.array([[200.0, 200.0, 0.5]])
    speck_sw = np.array([99])
    xyz = np.vstack([track, speck])
    sw = np.concatenate([track_sw, speck_sw]).astype(np.int32)

    keep = coherence_keep(
        xyz, sw, cell_m=0.4, link_gate_m=3.0, min_life=3, max_object_m=7.0
    )
    assert keep[:6].all(), "6-sweep track must survive the coherence gate"
    assert not keep[6], "single-sweep speck belongs to no track — must drop"


def test_coherence_size_cap_drops_large_structure():
    """A cluster wider than max_object_m never seeds a track, even if persistent."""
    # A dense 12 m wall present every sweep — too big to be an object. Points
    # are spaced well under coherence_cell_m so the wall is one connected
    # component (a sparse wall would fragment into object-sized pieces).
    span = np.linspace(0, 12, 200)
    wall = np.array([[x, 0.0, 1.0] for _ in range(8) for x in span])
    wall_sw = np.array([s for s in range(8) for _ in span])
    keep = coherence_keep(
        wall, wall_sw.astype(np.int32), cell_m=0.4, link_gate_m=3.0,
        min_life=3, max_object_m=7.0,
    )
    assert not keep.any(), "12 m structure exceeds the size cap — must drop"


def test_compute_keep_counts_split_between_gates():
    mover = np.array([[s * 1.0, 0.0, 0.5] for s in range(10)])
    blob = np.array([[50.0, 50.0, 1.0]] * 10)  # persistence drop
    speck = np.array([[200.0, 200.0, 0.5]])  # coherence drop
    xyz = np.vstack([mover, blob, speck])
    sw = np.concatenate([np.arange(10), np.arange(10), [99]]).astype(np.int32)

    # Both gates explicitly on, so the count split is exercised regardless of
    # the (recall-biased) defaults, which leave coherence off.
    mf = MotionFilterParams(persistence_max_sweeps=5, coherence_min_life=3)
    keep, n_persist, n_coh = compute_keep(mf, xyz, sw)
    assert keep[:10].all()
    assert not keep[10:20].any()
    assert not keep[20]
    assert n_persist == 10  # the blob
    assert n_coh == 1  # the speck (a persistence survivor)


def test_defaults_are_recall_biased():
    """Pin the shipped defaults: loose persistence, coherence off."""
    mf = MotionFilterParams()
    assert mf.enabled is True
    assert mf.persistence_max_sweeps == 20
    assert mf.coherence_min_life == 0  # off — it over-cuts movers on sparse LiDAR


# --------------------------------------------------------------------------- #
# End-to-end artifact rewrite
# --------------------------------------------------------------------------- #


def _write_chunk(bag_id, chunk_id, per_sweep, *, with_intensity=False):
    """Write masks + dynamic_map + index for {sweep_id: (world_xyz, dyn_mask)}.

    Builds dynamic_map exactly as union does — concat of world_xyz[mask] per
    sweep in world order — so the filter's keep-mask mapping is exercised. When
    with_intensity, the dynamic_map gets an `intensity` channel (one value per
    dynamic point, = its x coord) so the filter's intensity[keep] path is hit.
    """
    dyn_xyz, dyn_sw, dyn_int, rows = [], [], [], []
    for sid, (xyz, mask) in sorted(per_sweep.items()):
        n = xyz.shape[0]
        dyn_uri = dynamic_mask_path(bag_id, chunk_id, sid)
        os.makedirs(os.path.dirname(local_path(dyn_uri)), exist_ok=True)
        np.save(local_path(dyn_uri), mask)
        sel = xyz[mask]
        dyn_xyz.append(sel)
        dyn_sw.append(np.full(sel.shape[0], sid, dtype=np.int32))
        dyn_int.append(sel[:, 0].astype(np.float32))  # intensity == x, a unique tag
        rows.append(
            {
                "bag_id": bag_id,
                "chunk_id": chunk_id,
                "sweep_id": sid,
                "lidar_id": "LIDAR_TOP",
                "reference_timestamp_ns": sid * 100_000_000,
                "n_points_total": n,
                "n_points_static": n - int(mask.sum()),
                "n_points_dynamic": int(mask.sum()),
                "n_points_ground": 0,
                "world_path": f"world/{sid}.npz",
                "dynamic_mask_path": dyn_uri,
                "has_intensity": False,
                "deskewed": True,
                "valid": True,
                "world_xmin": float(xyz[:, 0].min()),
                "world_xmax": float(xyz[:, 0].max()),
                "world_ymin": float(xyz[:, 1].min()),
                "world_ymax": float(xyz[:, 1].max()),
                "world_zmin": float(xyz[:, 2].min()),
                "world_zmax": float(xyz[:, 2].max()),
            }
        )
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))
    dm_uri = dynamic_map_path(bag_id, chunk_id)
    os.makedirs(os.path.dirname(local_path(dm_uri)), exist_ok=True)
    kwargs = {
        "xyz": np.vstack(dyn_xyz) if dyn_xyz else np.empty((0, 3)),
        "sweep_id": np.concatenate(dyn_sw) if dyn_sw else np.empty(0, np.int32),
    }
    if with_intensity:
        kwargs["intensity"] = (
            np.concatenate(dyn_int) if dyn_int else np.empty(0, np.float32)
        )
    np.savez_compressed(local_path(dm_uri), **kwargs)


def _assert_consistent(bag_id, chunk_id):
    """dynamic_map, per-sweep masks, and index must agree after filtering."""
    dm = np.load(local_path(dynamic_map_path(bag_id, chunk_id)))
    map_per_sweep = {
        int(s): int((dm["sweep_id"] == s).sum()) for s in np.unique(dm["sweep_id"])
    }
    for row in read_rows(lidar_proc_index_path(bag_id, chunk_id)):
        sid = int(row["sweep_id"])
        mask = np.load(local_path(dynamic_mask_path(bag_id, chunk_id, sid)))
        n_mask = int(mask.sum())
        assert n_mask == int(row["n_points_dynamic"]), f"sweep {sid}: index vs mask"
        assert n_mask == map_per_sweep.get(sid, 0), f"sweep {sid}: map vs mask"


def test_filter_removes_structure_keeps_mover(tmp_env):
    bag_id, chunk_id = "bag0", "chunk0"
    # 8 sweeps. Each sweep's world cloud = [mover point, 4 static-blob points].
    # Mover translates 1 m/sweep (distinct voxels, coherent track).
    # Blob sits in the same voxel every sweep (persistence target).
    per_sweep = {}
    for s in range(8):
        mover = np.array([[s * 1.0, 0.0, 0.5]])
        blob = np.array(
            [[50.0, 50.0, 1.0], [50.1, 50.0, 1.0], [50.0, 50.1, 1.0], [50.1, 50.1, 1.0]]
        )
        xyz = np.vstack([mover, blob])
        mask = np.ones(xyz.shape[0], dtype=bool)  # all flagged dynamic pre-filter
        per_sweep[s] = (xyz, mask)
    _write_chunk(bag_id, chunk_id, per_sweep)

    cfg = ComponentConfig(segmentation="union")
    # Explicit threshold (the 8-sweep blob must exceed it) — independent of the
    # shipped default, which is looser.
    cfg.union.motion_filter.persistence_max_sweeps = 5
    res = filter_dynamic_artifacts(cfg, bag_id, chunk_id)

    assert res.applied
    assert res.n_before == 8 * 5
    assert res.n_after == 8, "only the 8 mover points (1/sweep) should survive"
    assert res.n_persistence_dropped == 8 * 4, "all blob points dropped by persistence"
    _assert_consistent(bag_id, chunk_id)

    # The surviving points are the movers, not the blob.
    dm = np.load(local_path(dynamic_map_path(bag_id, chunk_id)))
    assert dm["xyz"].shape[0] == 8
    assert np.all(dm["xyz"][:, 0] < 10), "survivors are the translating mover"


def test_filter_disabled_is_noop(tmp_env):
    bag_id, chunk_id = "bag1", "chunk0"
    blob = np.array([[50.0, 50.0, 1.0]] * 4)
    per_sweep = {s: (blob, np.ones(4, bool)) for s in range(8)}
    _write_chunk(bag_id, chunk_id, per_sweep)

    cfg = ComponentConfig(segmentation="union")
    cfg.union.motion_filter.enabled = False
    res = filter_dynamic_artifacts(cfg, bag_id, chunk_id)

    assert not res.applied
    dm = np.load(local_path(dynamic_map_path(bag_id, chunk_id)))
    assert dm["xyz"].shape[0] == 8 * 4, "disabled filter must not drop anything"


def test_filter_empty_cloud_is_safe(tmp_env):
    bag_id, chunk_id = "bag2", "chunk0"
    per_sweep = {s: (np.array([[1.0, 1.0, 1.0]]), np.zeros(1, bool)) for s in range(3)}
    _write_chunk(bag_id, chunk_id, per_sweep)

    cfg = ComponentConfig(segmentation="union")
    res = filter_dynamic_artifacts(cfg, bag_id, chunk_id)
    assert res.n_after == 0
    _assert_consistent(bag_id, chunk_id)


def test_filter_both_gates_off_is_noop(tmp_env):
    """Enabled but both thresholds 0 → applied=False, nothing rewritten."""
    bag_id, chunk_id = "bag3", "chunk0"
    blob = np.array([[50.0, 50.0, 1.0]] * 4)
    per_sweep = {s: (blob, np.ones(4, bool)) for s in range(8)}
    _write_chunk(bag_id, chunk_id, per_sweep)

    cfg = ComponentConfig(segmentation="union")
    cfg.union.motion_filter.persistence_max_sweeps = 0
    cfg.union.motion_filter.coherence_min_life = 0
    res = filter_dynamic_artifacts(cfg, bag_id, chunk_id)

    assert not res.applied
    dm = np.load(local_path(dynamic_map_path(bag_id, chunk_id)))
    assert dm["xyz"].shape[0] == 8 * 4


def test_filter_preserves_intensity(tmp_env):
    """The intensity channel is filtered point-for-point with xyz."""
    bag_id, chunk_id = "bag4", "chunk0"
    per_sweep = {}
    for s in range(8):
        mover = np.array([[s * 1.0, 0.0, 0.5]])  # survives
        blob = np.array([[50.0, 50.0, 1.0]] * 4)  # dropped by persistence
        xyz = np.vstack([mover, blob])
        per_sweep[s] = (xyz, np.ones(xyz.shape[0], bool))
    _write_chunk(bag_id, chunk_id, per_sweep, with_intensity=True)

    cfg = ComponentConfig(segmentation="union")
    cfg.union.motion_filter.persistence_max_sweeps = 5  # drop the 8-sweep blob
    filter_dynamic_artifacts(cfg, bag_id, chunk_id)

    dm = np.load(local_path(dynamic_map_path(bag_id, chunk_id)))
    assert "intensity" in dm.files
    assert dm["intensity"].shape[0] == dm["xyz"].shape[0]
    # _write_chunk set intensity == x, so the channel must still track x exactly.
    assert np.allclose(dm["intensity"], dm["xyz"][:, 0].astype(np.float32))


def test_filter_persistence_only(tmp_env):
    """coherence_min_life=0 → only persistence acts: an isolated speck survives
    (no coherence gate to drop it) while the voxel-dwelling blob is dropped."""
    bag_id, chunk_id = "bag5", "chunk0"
    per_sweep = {}
    for s in range(8):
        blob = np.array([[50.0, 50.0, 1.0]] * 2)  # dwells → persistence drop
        per_sweep[s] = (blob, np.ones(2, bool))
    per_sweep[0] = (
        np.vstack([per_sweep[0][0], [[5.0, 5.0, 0.5]]]),  # one-sweep speck
        np.ones(3, bool),
    )
    _write_chunk(bag_id, chunk_id, per_sweep)

    cfg = ComponentConfig(segmentation="union")
    cfg.union.motion_filter.persistence_max_sweeps = 5  # drop the 8-sweep blob
    cfg.union.motion_filter.coherence_min_life = 0  # persistence only
    res = filter_dynamic_artifacts(cfg, bag_id, chunk_id)

    assert res.n_coherence_dropped == 0
    assert res.n_after == 1, "only the speck survives; the dwelling blob is dropped"
    _assert_consistent(bag_id, chunk_id)


def test_filter_coherence_only(tmp_env):
    """persistence_max_sweeps=0 → only coherence acts: a coherent dweller-track
    survives while the one-sweep speck (no track) is dropped."""
    bag_id, chunk_id = "bag6", "chunk0"
    per_sweep = {}
    for s in range(8):
        blob = np.array([[50.0, 50.0, 1.0]])  # same spot → 8-sweep track
        per_sweep[s] = (blob, np.ones(1, bool))
    per_sweep[0] = (
        np.vstack([per_sweep[0][0], [[5.0, 5.0, 0.5]]]),  # one-sweep speck
        np.ones(2, bool),
    )
    _write_chunk(bag_id, chunk_id, per_sweep)

    cfg = ComponentConfig(segmentation="union")
    cfg.union.motion_filter.persistence_max_sweeps = 0  # persistence off
    cfg.union.motion_filter.coherence_min_life = 3  # coherence only (off by default)
    res = filter_dynamic_artifacts(cfg, bag_id, chunk_id)

    assert res.n_persistence_dropped == 0
    assert res.n_after == 8, "the 8-sweep track survives; the lone speck is dropped"
    _assert_consistent(bag_id, chunk_id)
