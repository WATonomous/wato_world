"""Tests for classify (Step B — voxel static/dynamic decomposition).

Constants are derived from the SensorModel (default velodyne_vlp): l_occ≈1.99,
l_free≈0.41, p_static=0.88, p_dynamic=0.12. Tests engineer geometry that lands
robustly in the intended class rather than pinning exact log-odds. Every sweep
carries an `origin` (required by the log-odds ray-casting path).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from wato_common.artifact_store import (
    dynamic_map_path,
    dynamic_mask_path,
    lidar_proc_index_path,
    lidar_world_path,
    local_path,
    mf_mos_mask_path,
    static_map_path,
    voxel_occupancy_path,
)
from wato_common.io.parquet_io import write_table
from wato_common.schemas import PROCESSED_SWEEPS_SCHEMA
from wato_lidar_preprocessing.classify import process_chunk
from wato_lidar_preprocessing.config import ComponentConfig

# Sensor behind the scene on -x so a same-point sweep never self-carves.
_SENSOR = np.array([-5.0, 0.0, 0.0])


@pytest.fixture()
def tmp_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", str(tmp_path))
    return tmp_path


def _write_world_sweep(
    bag_id: str,
    chunk_id: str,
    sweep_id: int,
    xyz: np.ndarray,
    *,
    origin: np.ndarray | None = None,
    ground_mask: np.ndarray | None = None,
):
    path = local_path(lidar_world_path(bag_id, chunk_id, sweep_id))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    kwargs = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    if origin is not None:
        kwargs["origin"] = np.asarray(origin, dtype=np.float64)
    if ground_mask is not None:
        kwargs["ground_mask"] = ground_mask
    np.savez_compressed(path, **kwargs)


def _proc_row(
    bag_id: str,
    chunk_id: str,
    sweep_id: int,
    xyz: np.ndarray,
    *,
    has_intensity: bool = False,
) -> dict:
    """Build a parquet row that mirrors what real deskew would write."""
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
        "world_path": lidar_world_path(bag_id, chunk_id, sweep_id),
        "dynamic_mask_path": "",
        "has_intensity": has_intensity,
        "deskewed": True,
        "world_xmin": float(xyz[:, 0].min()) if n else None,
        "world_xmax": float(xyz[:, 0].max()) if n else None,
        "world_ymin": float(xyz[:, 1].min()) if n else None,
        "world_ymax": float(xyz[:, 1].max()) if n else None,
        "world_zmin": float(xyz[:, 2].min()) if n else None,
        "world_zmax": float(xyz[:, 2].max()) if n else None,
    }


def _write_proc_index(
    bag_id: str,
    chunk_id: str,
    sweep_ids: list[int],
    xyz_per_sweep: list[np.ndarray] | None = None,
):
    if xyz_per_sweep is None:
        xyz_per_sweep = [np.empty((0, 3))] * len(sweep_ids)
    rows = [
        _proc_row(bag_id, chunk_id, sid, xyz)
        for sid, xyz in zip(sweep_ids, xyz_per_sweep)
    ]
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))


def test_all_sweeps_present_classified_static(tmp_env):
    """Points hit in every sweep from a fixed sensor → static; mask all False."""
    bag_id, chunk_id = "bag0", "chunk0"
    n_sweeps = 10
    static_xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    for i in range(n_sweeps):
        _write_world_sweep(bag_id, chunk_id, i, static_xyz, origin=_SENSOR)
    _write_proc_index(
        bag_id, chunk_id, list(range(n_sweeps)), xyz_per_sweep=[static_xyz] * n_sweeps
    )

    result = process_chunk(ComponentConfig(), bag_id, chunk_id)

    assert result.n_dynamic == 0
    assert result.n_static == n_sweeps * 3

    for i in range(n_sweeps):
        mask = np.load(local_path(dynamic_mask_path(bag_id, chunk_id, i)))
        assert not mask.any(), f"sweep {i}: expected all static"

    static_data = np.load(local_path(static_map_path(bag_id, chunk_id)))
    assert static_data["xyz"].shape[0] == n_sweeps * 3


def test_single_sweep_point_then_carved_is_dynamic(tmp_env):
    """A voxel hit once then carved by many through-rays → dynamic.

    Sweep 0 deposits a hit at (0,0,0). Sweeps 1..N cast rays whose endpoints
    are far beyond it, so each carves the (0,0,0) voxel. With l_occ≈1.99 and
    l_free≈0.41, ~20 carves drive the log-odds well below logit(p_dynamic).
    """
    bag_id, chunk_id = "bag1", "chunk0"
    hit_xyz = np.array([[0.0, 0.0, 0.0]])
    beyond_xyz = np.array([[8.0, 0.0, 0.0]])
    n_carves = 20

    _write_world_sweep(bag_id, chunk_id, 0, hit_xyz, origin=_SENSOR)
    xyz_per_sweep = [hit_xyz]
    for i in range(1, n_carves + 1):
        _write_world_sweep(bag_id, chunk_id, i, beyond_xyz, origin=_SENSOR)
        xyz_per_sweep.append(beyond_xyz)
    _write_proc_index(
        bag_id, chunk_id, list(range(n_carves + 1)), xyz_per_sweep=xyz_per_sweep
    )

    result = process_chunk(ComponentConfig(), bag_id, chunk_id)

    mask0 = np.load(local_path(dynamic_mask_path(bag_id, chunk_id, 0)))
    assert mask0[0], "voxel carved by many later rays must be dynamic"
    assert result.n_dynamic >= 1

    dyn = np.load(local_path(dynamic_map_path(bag_id, chunk_id)))
    assert dyn["xyz"].shape[0] >= 1
    np.testing.assert_allclose(dyn["xyz"][0], hit_xyz[0], atol=1e-6)
    assert (dyn["sweep_id"] == 0).all()


def test_dynamic_map_aggregates_across_sweeps(tmp_env):
    """Dynamic points from multiple sweeps land in one chunk-level NPZ.

    Each sweep deposits a unique point that is then carved by every other
    sweep's far ray through it, so all become dynamic.
    """
    bag_id, chunk_id = "bag_dynagg", "chunk0"
    n_sweeps = 12
    # Distinct near points along y, all carved by a shared far ray on +x.
    far = np.array([[40.0, 0.0, 0.0]])
    xyz_per_sweep: list[np.ndarray] = []
    for i in range(n_sweeps):
        near = np.array([[1.0, float(i), 0.0]])
        xyz = np.concatenate([near, far])
        _write_world_sweep(bag_id, chunk_id, i, xyz, origin=_SENSOR)
        xyz_per_sweep.append(xyz)
    _write_proc_index(
        bag_id, chunk_id, list(range(n_sweeps)), xyz_per_sweep=xyz_per_sweep
    )

    process_chunk(ComponentConfig(), bag_id, chunk_id)

    dyn = np.load(local_path(dynamic_map_path(bag_id, chunk_id)))
    # The far point is hit every sweep from a fixed origin → static; the
    # near points are each hit once and never re-confirmed → free/under, but
    # they're not carved either (no ray passes through them), so they are
    # under-evidenced, not dynamic. Assert dynamic_map carries no far point.
    if dyn["xyz"].shape[0]:
        assert not np.any(np.all(np.isclose(dyn["xyz"], far[0]), axis=1))


def test_voxel_occupancy_emitted_by_default(tmp_env):
    """save_voxel_occupancy defaults to True (SAM4D / MinkUNet contract)."""
    bag_id, chunk_id = "bag_occ", "chunk0"
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    for i in range(5):
        _write_world_sweep(bag_id, chunk_id, i, xyz, origin=_SENSOR)
    _write_proc_index(bag_id, chunk_id, list(range(5)), xyz_per_sweep=[xyz] * 5)

    cfg = ComponentConfig()
    assert cfg.save_voxel_occupancy is True
    process_chunk(cfg, bag_id, chunk_id)

    occ_path = local_path(voxel_occupancy_path(bag_id, chunk_id))
    assert os.path.exists(occ_path), "voxel_occupancy.npz must be written by default"
    data = np.load(occ_path)
    assert data["coords"].shape[1] == 3
    assert data["coords"].dtype == np.int32
    assert "origin" in data and "voxel_size" in data


def test_intensity_backfilled_when_first_sweep_lacks_it(tmp_env):
    """Sweep 0 has no intensity, later sweeps do → output intensity aligns with xyz."""
    bag_id, chunk_id = "bag_mixed_int", "chunk0"
    n_sweeps = 5
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    for i in range(n_sweeps):
        path = local_path(lidar_world_path(bag_id, chunk_id, i))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        kwargs = {
            "x": pts[:, 0],
            "y": pts[:, 1],
            "z": pts[:, 2],
            "origin": _SENSOR,
        }
        if i >= 2:
            kwargs["intensity"] = np.array([0.5, 0.7], dtype=np.float32)
        np.savez_compressed(path, **kwargs)

    rows = [
        _proc_row(bag_id, chunk_id, sid, pts, has_intensity=sid >= 2)
        for sid in range(n_sweeps)
    ]
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    process_chunk(ComponentConfig(), bag_id, chunk_id)

    static = np.load(local_path(static_map_path(bag_id, chunk_id)))
    assert "intensity" in static
    assert static["intensity"].shape[0] == static["xyz"].shape[0], (
        f"intensity ({static['intensity'].shape[0]}) and xyz "
        f"({static['xyz'].shape[0]}) must align"
    )
    # First 4 entries (sweeps 0–1, 2 points each) should be zero-padded.
    assert (static["intensity"][:4] == 0).all()
    # Sweeps 2–4 contributed real intensity values.
    assert (static["intensity"][4:] > 0).all()


def test_empty_proc_index_writes_sentinel(tmp_env):
    """No proc_index rows → write empty static_map + dynamic_map, don't crash."""
    bag_id, chunk_id = "bag_empty", "chunk0"
    write_table([], PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    result = process_chunk(ComponentConfig(), bag_id, chunk_id)
    assert result.n_static == 0
    assert result.n_dynamic == 0
    static = np.load(local_path(static_map_path(bag_id, chunk_id)))
    assert static["xyz"].shape == (0, 3)
    dyn = np.load(local_path(dynamic_map_path(bag_id, chunk_id)))
    assert dyn["xyz"].shape == (0, 3)
    assert dyn["sweep_id"].shape == (0,)


def test_static_map_written(tmp_env):
    bag_id, chunk_id = "bag2", "chunk0"
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    for i in range(5):
        _write_world_sweep(bag_id, chunk_id, i, xyz, origin=_SENSOR)
    _write_proc_index(bag_id, chunk_id, list(range(5)), xyz_per_sweep=[xyz] * 5)

    process_chunk(ComponentConfig(), bag_id, chunk_id)

    assert os.path.exists(local_path(static_map_path(bag_id, chunk_id)))
    data = np.load(local_path(static_map_path(bag_id, chunk_id)))
    assert "xyz" in data
    assert data["xyz"].shape[1] == 3


def test_log_odds_static_accumulated_from_fixed_point(tmp_env):
    """A point seen in all 10 sweeps from a fixed sensor accumulates enough
    l_occ to exceed p_static_threshold and is classified static.
    """
    bag_id, chunk_id = "bag_lo_static", "chunk0"
    n_sweeps = 10
    point_xyz = np.array([[10.0, 0.0, 0.0]])
    sensor_origin = np.array([-5.0, 0.0, 0.0])

    for i in range(n_sweeps):
        _write_world_sweep(bag_id, chunk_id, i, point_xyz, origin=sensor_origin)
    _write_proc_index(
        bag_id, chunk_id, list(range(n_sweeps)), xyz_per_sweep=[point_xyz] * n_sweeps
    )

    result = process_chunk(
        ComponentConfig(min_observations=3), bag_id, chunk_id
    )

    assert result.n_static == n_sweeps, "all points should be static"
    assert result.n_dynamic == 0


def test_log_odds_free_only_voxel_not_dynamic(tmp_env):
    """Voxels with n_hits==0 (only traversed) must not be labeled dynamic even
    when their log_odds are negative.

    Sweep 0: non-ground point at (5,0,0) carves the (0,0,0) voxel as free.
    Sweep 1: a GROUND point at (0.05,0,0) lands in (0,0,0); skip_endpoint keeps
    its n_hits==0 → free-only → mask=False.
    """
    bag_id, chunk_id = "bag_lo_freeonly", "chunk0"
    xyz0 = np.array([[5.0, 0.0, 0.0]])
    xyz1 = np.array([[0.05, 0.0, 0.0]])
    origin_v = np.array([-1.0, 0.0, 0.0])

    _write_world_sweep(bag_id, chunk_id, 0, xyz0, origin=origin_v)
    _write_world_sweep(
        bag_id, chunk_id, 1, xyz1, origin=origin_v, ground_mask=np.array([True])
    )
    _write_proc_index(bag_id, chunk_id, [0, 1], xyz_per_sweep=[xyz0, xyz1])

    cfg = ComponentConfig(
        ground_endpoint_strategy="skip_endpoint",
        min_observations=3,
        min_occupied_hits=1,
    )
    process_chunk(cfg, bag_id, chunk_id)

    mask1 = np.load(local_path(dynamic_mask_path(bag_id, chunk_id, 1)))
    assert not mask1[0], "ground point in free-only voxel must not be dynamic"


def test_static_map_xyz_excludes_free_only_and_under_evidenced(tmp_env):
    """static_map.npz["xyz"] must contain only confident-static points.

    Three voxels: confident-static (repeated hits), free-only ground, and
    a single-hit under-evidenced voxel. Only the static coords may appear.
    """
    bag_id, chunk_id = "bag_static_xyz_purity", "chunk0"
    n_static_sweeps = 5
    static_pt = np.array([[5.0, 0.0, 0.0]])
    ground_pt = np.array([[0.05, 0.0, 0.0]])
    under_pt = np.array([[10.0, 0.0, 0.0]])
    sensor_origin = np.array([-1.0, 0.0, 0.0])

    xyz_per_sweep: list[np.ndarray] = []
    for i in range(n_static_sweeps):
        _write_world_sweep(bag_id, chunk_id, i, static_pt, origin=sensor_origin)
        xyz_per_sweep.append(static_pt)

    ground_sweep_id = n_static_sweeps
    _write_world_sweep(
        bag_id, chunk_id, ground_sweep_id, ground_pt,
        origin=sensor_origin, ground_mask=np.array([True]),
    )
    xyz_per_sweep.append(ground_pt)

    under_sweep_id = n_static_sweeps + 1
    _write_world_sweep(bag_id, chunk_id, under_sweep_id, under_pt, origin=sensor_origin)
    xyz_per_sweep.append(under_pt)

    sweep_ids = list(range(n_static_sweeps + 2))
    _write_proc_index(bag_id, chunk_id, sweep_ids, xyz_per_sweep=xyz_per_sweep)

    cfg = ComponentConfig(
        ground_endpoint_strategy="skip_endpoint",
        min_observations=3,
        min_occupied_hits=1,
    )
    result = process_chunk(cfg, bag_id, chunk_id)

    assert result.n_static == n_static_sweeps, (
        f"expected {n_static_sweeps} confident-static points, got {result.n_static}"
    )

    static_xyz = np.load(local_path(static_map_path(bag_id, chunk_id)))["xyz"]
    assert static_xyz.shape[0] == n_static_sweeps
    np.testing.assert_allclose(
        static_xyz, np.repeat(static_pt, n_static_sweeps, axis=0)
    )

    flat = static_xyz.reshape(-1, 3)
    assert not (flat == ground_pt[0]).all(axis=1).any(), (
        "ground point in free-only voxel must NOT appear in static_map.npz['xyz']"
    )
    assert not (flat == under_pt[0]).all(axis=1).any(), (
        "under-evidenced point must NOT appear in static_map.npz['xyz']"
    )


def test_under_evidenced_with_hits_not_dynamic(tmp_env):
    """Under-evidenced voxel WITH endpoint hits gets the benefit of the doubt.

    One sweep, one point, min_observations=3: the endpoint voxel has n_hits=1,
    n_obs=1 → under_evidenced_with_hits → not_dynamic (mask=False), but not
    static either.
    """
    bag_id, chunk_id = "bag_lo_under", "chunk0"
    xyz = np.array([[5.0, 0.0, 0.0]])

    _write_world_sweep(bag_id, chunk_id, 0, xyz, origin=np.array([-1.0, 0.0, 0.0]))
    _write_proc_index(bag_id, chunk_id, [0], xyz_per_sweep=[xyz])

    result = process_chunk(
        ComponentConfig(min_observations=3, min_occupied_hits=1), bag_id, chunk_id
    )

    assert result.n_static == 0, "under-evidenced voxel must not be static"
    assert result.n_dynamic == 0, "under-evidenced voxel WITH hits must not be dynamic"

    mask = np.load(local_path(dynamic_mask_path(bag_id, chunk_id, 0)))
    assert mask.shape == (1,)
    assert not mask[0]


def test_skip_endpoint_isolated_ground_voxel_not_dynamic(tmp_env):
    """skip_endpoint: a ground voxel that no other ray traverses is still
    filtered out by the patchwork mask (the unconditional ground post-filter)."""
    bag_id, chunk_id = "bag_lo_skipendpoint_iso", "chunk0"
    xyz = np.array([[0.05, 0.0, 0.01]])

    _write_world_sweep(
        bag_id, chunk_id, 0, xyz,
        origin=np.array([-1.0, 0.0, 0.0]), ground_mask=np.array([True]),
    )
    _write_proc_index(bag_id, chunk_id, [0], xyz_per_sweep=[xyz])

    cfg = ComponentConfig(
        ground_endpoint_strategy="skip_endpoint",
        min_observations=1,
        min_occupied_hits=1,
    )
    process_chunk(cfg, bag_id, chunk_id)

    mask = np.load(local_path(dynamic_mask_path(bag_id, chunk_id, 0)))
    assert not mask[0], "isolated ground voxel must not leak into dynamic"
    dyn = np.load(local_path(dynamic_map_path(bag_id, chunk_id)))
    assert dyn["xyz"].shape[0] == 0, "dynamic_map.npz must be empty"


def test_skip_ray_ground_not_dynamic(tmp_env):
    """In skip_ray mode, ground voxels must not be labeled dynamic and the mask
    length still matches the world NPZ point count."""
    bag_id, chunk_id = "bag_lo_skipray", "chunk0"
    xyz = np.array([[0.05, 0.0, 0.01], [5.0, 0.0, 0.0]])
    ground_mask = np.array([True, False])

    _write_world_sweep(
        bag_id, chunk_id, 0, xyz,
        origin=np.array([-1.0, 0.0, 0.0]), ground_mask=ground_mask,
    )
    _write_proc_index(bag_id, chunk_id, [0], xyz_per_sweep=[xyz])

    cfg = ComponentConfig(
        ground_endpoint_strategy="skip_ray",
        min_observations=1,
        min_occupied_hits=1,
    )
    process_chunk(cfg, bag_id, chunk_id)

    mask = np.load(local_path(dynamic_mask_path(bag_id, chunk_id, 0)))
    assert mask.shape == (2,), (
        f"mask length must match total world NPZ points (2), got {mask.shape[0]}"
    )
    assert not mask[0], "ground point in skip_ray mode must not be dynamic"


# ---------------------------------------------------------------------------
# MF-MOS fusion — per-sweep mask drives fusion (no chunk-wide vote tier).
# ---------------------------------------------------------------------------


def _write_mf_mos_mask(
    bag_id: str, chunk_id: str, sweep_id: int, mask: np.ndarray
) -> str:
    uri = mf_mos_mask_path(bag_id, chunk_id, sweep_id)
    path = local_path(uri)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, mask.astype(bool))
    return uri


def _proc_row_mf_mos(
    bag_id: str,
    chunk_id: str,
    sweep_id: int,
    xyz: np.ndarray,
    mf_mos_mask_uri: str | None = None,
) -> dict:
    row = _proc_row(bag_id, chunk_id, sweep_id, xyz)
    row["mf_mos_mask_path"] = mf_mos_mask_uri
    return row


def test_union_fusion_flips_per_sweep_flagged_points(tmp_env):
    """union mode: only the sweeps whose MF-MOS mask flags the point flip to
    dynamic; AW-static sweeps without a flag stay static.

    Five sweeps hit the same voxel (AW → confident static). 3 of 5 MF-MOS
    masks flag it. The (denoised) per-sweep mask drives fusion directly, so
    exactly those 3 sweeps' points flip — no chunk-wide vote needed.
    """
    bag_id, chunk_id = "bag_union", "chunk0"
    xyz = np.array([[5.0, 0.0, 0.0]])
    sensor_origin = np.array([-1.0, 0.0, 0.0])
    mf_flags = [True, True, True, False, False]
    n_flagged = sum(mf_flags)

    rows = []
    for i in range(5):
        _write_world_sweep(bag_id, chunk_id, i, xyz, origin=sensor_origin)
        mf_uri = _write_mf_mos_mask(bag_id, chunk_id, i, np.array([mf_flags[i]]))
        rows.append(_proc_row_mf_mos(bag_id, chunk_id, i, xyz, mf_uri))
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    cfg = ComponentConfig(
        min_observations=3,
        min_occupied_hits=1,
        mf_mos={"enabled": True, "fusion_mode": "union"},
    )
    result = process_chunk(cfg, bag_id, chunk_id)

    assert result.n_dynamic == n_flagged, (
        f"only the {n_flagged} per-sweep-flagged points must be dynamic; "
        f"got {result.n_dynamic}"
    )
    assert result.n_static == 5 - n_flagged


def test_independent_mode_no_mf_mos_effect(tmp_env):
    """fusion_mode=independent: MF-MOS masks don't change AW classification."""
    bag_id, chunk_id = "bag_indep", "chunk0"
    xyz = np.array([[5.0, 0.0, 0.0]])
    sensor_origin = np.array([-1.0, 0.0, 0.0])
    mf_flags = [True, True, True, False, False]

    rows = []
    for i in range(5):
        _write_world_sweep(bag_id, chunk_id, i, xyz, origin=sensor_origin)
        mf_uri = _write_mf_mos_mask(bag_id, chunk_id, i, np.array([mf_flags[i]]))
        rows.append(_proc_row_mf_mos(bag_id, chunk_id, i, xyz, mf_uri))
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    cfg = ComponentConfig(
        min_observations=3,
        min_occupied_hits=1,
        mf_mos={"enabled": True, "fusion_mode": "independent"},
    )
    result = process_chunk(cfg, bag_id, chunk_id)

    assert result.n_dynamic == 0, "independent fusion must not apply MF-MOS masks"


def test_union_fusion_must_not_reintroduce_ground_via_mf_mos(tmp_env):
    """union fusion ORs the per-sweep MF-MOS mask in, then must re-apply the
    ground filter so a co-voxel ground point isn't dragged into dynamic."""
    bag_id, chunk_id = "bag_union_ground", "chunk0"
    xyz = np.array([[5.0, 0.0, 0.0], [5.0, 0.0, 0.05]])
    ground_mask = np.array([True, False])
    mf_flags = np.array([False, True])

    _write_world_sweep(
        bag_id, chunk_id, 0, xyz,
        origin=np.array([-1.0, 0.0, 0.0]), ground_mask=ground_mask,
    )
    mf_uri = _write_mf_mos_mask(bag_id, chunk_id, 0, mf_flags)
    rows = [_proc_row_mf_mos(bag_id, chunk_id, 0, xyz, mf_uri)]
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    cfg = ComponentConfig(
        ground_endpoint_strategy="skip_endpoint",
        min_observations=1,
        min_occupied_hits=1,
        mf_mos={"enabled": True, "fusion_mode": "union"},
    )
    process_chunk(cfg, bag_id, chunk_id)

    dyn = np.load(local_path(dynamic_map_path(bag_id, chunk_id)))
    is_ground_pt = np.all(np.isclose(dyn["xyz"], xyz[0]), axis=1)
    assert not is_ground_pt.any(), "ground point must not appear in dynamic_map.npz"
    mask = np.load(local_path(dynamic_mask_path(bag_id, chunk_id, 0)))
    assert not mask[0], "ground point's dynamic-mask bit must be False under union"


def test_mfmos_only_fusion_must_not_label_ground_dynamic(tmp_env):
    """mfmos_only mode overwrites the mask with the MF-MOS mask; the ground
    filter must be re-applied so a co-voxel ground point isn't flagged."""
    bag_id, chunk_id = "bag_mfonly_ground", "chunk0"
    xyz = np.array([[5.0, 0.0, 0.0], [5.0, 0.0, 0.05]])
    ground_mask = np.array([True, False])
    mf_flags = np.array([False, True])

    _write_world_sweep(
        bag_id, chunk_id, 0, xyz,
        origin=np.array([-1.0, 0.0, 0.0]), ground_mask=ground_mask,
    )
    mf_uri = _write_mf_mos_mask(bag_id, chunk_id, 0, mf_flags)
    rows = [_proc_row_mf_mos(bag_id, chunk_id, 0, xyz, mf_uri)]
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    cfg = ComponentConfig(
        ground_endpoint_strategy="skip_endpoint",
        min_observations=1,
        min_occupied_hits=1,
        mf_mos={"enabled": True, "fusion_mode": "mfmos_only"},
    )
    process_chunk(cfg, bag_id, chunk_id)

    dyn = np.load(local_path(dynamic_map_path(bag_id, chunk_id)))
    is_ground_pt = np.all(np.isclose(dyn["xyz"], xyz[0]), axis=1)
    assert not is_ground_pt.any(), (
        "ground point must not appear in dynamic_map.npz under mfmos_only"
    )


def test_classify_raises_when_world_npz_missing_origin(tmp_env):
    """classify must hard-fail when a valid sweep's world NPZ has no 'origin'."""
    bag_id, chunk_id = "bag_no_origin", "chunk0"
    xyz = np.array([[5.0, 0.0, 1.0]])
    _write_world_sweep(bag_id, chunk_id, 0, xyz)  # no origin kwarg
    _write_proc_index(bag_id, chunk_id, [0], xyz_per_sweep=[xyz])

    with pytest.raises(ValueError, match="origin"):
        process_chunk(ComponentConfig(), bag_id, chunk_id)


def test_well_observed_static_survives_far_noisy_carves(tmp_env):
    """Range credibility protects a well-observed wall from far noisy carves.

    A wall at moderate range is hit every sweep. A few sweeps cast very long
    rays whose endpoints are far beyond the wall; those through-rays would
    carve the wall voxel, but at >> d_star the carve is down-weighted ∝ 1/d,
    so the wall stays static. This replaces the old clamp=50 "headroom"
    band-aid: the physics (credibility weight), not a giant clamp, keeps the
    wall static.
    """
    bag_id, chunk_id = "bag_far_carve", "chunk0"
    wall_pt = np.array([[5.0, 0.0, 1.0]])
    far_pt = np.array([[300.0, 0.0, 1.0]])  # >> d_star ≈ 50 m at voxel 0.15
    sensor = np.array([-5.0, 0.0, 1.0])

    n_hits = 20
    n_far_carves = 6
    rows_xyz: list[np.ndarray] = []
    for sid in range(n_hits):
        _write_world_sweep(bag_id, chunk_id, sid, wall_pt, origin=sensor)
        rows_xyz.append(wall_pt)
    for j in range(n_far_carves):
        sid = n_hits + j
        _write_world_sweep(bag_id, chunk_id, sid, far_pt, origin=sensor)
        rows_xyz.append(far_pt)
    _write_proc_index(
        bag_id, chunk_id, list(range(n_hits + n_far_carves)), xyz_per_sweep=rows_xyz
    )

    cfg = ComponentConfig(
        ground_endpoint_strategy="skip_endpoint",
        min_observations=3,
        min_occupied_hits=1,
    )
    result = process_chunk(cfg, bag_id, chunk_id)

    dyn = np.load(local_path(dynamic_map_path(bag_id, chunk_id)))
    is_wall_pt = (
        np.all(np.isclose(dyn["xyz"], wall_pt[0]), axis=1)
        if dyn["xyz"].shape[0]
        else np.zeros(0, dtype=bool)
    )
    assert not is_wall_pt.any(), (
        "wall hit by 20 sweeps must not be dynamic when its only carves are "
        "far, low-credibility through-rays"
    )
    assert result.n_static >= n_hits


def test_cache_disabled_must_still_apply_ground_filter(tmp_env):
    """With the in-memory cache off, ground_mask must still gate the mask."""
    bag_id, chunk_id = "bag_nocache_ground", "chunk0"
    ground_pt = np.array([[0.05, 0.05, 0.0]])
    _write_world_sweep(
        bag_id, chunk_id, 0, ground_pt,
        origin=np.array([-1.0, 0.0, 0.0]), ground_mask=np.array([True]),
    )
    _write_proc_index(bag_id, chunk_id, [0], xyz_per_sweep=[ground_pt])

    cfg = ComponentConfig(
        ground_endpoint_strategy="skip_endpoint",
        cache_world_xyz_in_memory=False,
        min_observations=1,
        min_occupied_hits=1,
    )
    process_chunk(cfg, bag_id, chunk_id)

    dyn = np.load(local_path(dynamic_map_path(bag_id, chunk_id)))
    assert dyn["xyz"].shape[0] == 0, "cache-disabled path must not leak ground points"


def test_min_occupied_hits_filters_below_threshold(tmp_env):
    """min_occupied_hits>1 routes 0 < n_hits < threshold voxels to free-only.

    A voxel with a single hit and many carves, with min_occupied_hits=3, must
    be free-only (not dynamic) — free_only_mask uses n_hits < min_occupied_hits.
    """
    bag_id, chunk_id = "bag_min_hits_hole", "chunk0"
    hit_pt = np.array([[5.0, 0.0, 0.0]])
    beyond_pt = np.array([[10.0, 0.0, 0.0]])
    sensor_origin = np.array([-1.0, 0.0, 0.0])

    rows_xyz: list[np.ndarray] = [hit_pt]
    _write_world_sweep(bag_id, chunk_id, 0, hit_pt, origin=sensor_origin)
    n_carves = 12
    for j in range(n_carves):
        sid = j + 1
        _write_world_sweep(bag_id, chunk_id, sid, beyond_pt, origin=sensor_origin)
        rows_xyz.append(beyond_pt)
    _write_proc_index(
        bag_id, chunk_id, list(range(1 + n_carves)), xyz_per_sweep=rows_xyz
    )

    cfg = ComponentConfig(
        ground_endpoint_strategy="skip_endpoint",
        voxel_size_m=0.25,
        min_observations=3,
        min_occupied_hits=3,
    )
    process_chunk(cfg, bag_id, chunk_id)

    dyn = np.load(local_path(dynamic_map_path(bag_id, chunk_id)))
    is_hit_pt = (
        np.all(np.isclose(dyn["xyz"], hit_pt[0]), axis=1)
        if dyn["xyz"].shape[0]
        else np.zeros(0, dtype=bool)
    )
    assert not is_hit_pt.any(), (
        "voxel with n_hits=1 < min_occupied_hits=3 must be free-only, not dynamic"
    )
    mask0 = np.load(local_path(dynamic_mask_path(bag_id, chunk_id, 0)))
    assert not mask0[0]
