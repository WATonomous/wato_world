"""Tests for classify.py (Step B — voxel static/dynamic decomposition)."""

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
    """Points seen in every sweep → static; dynamic mask all False."""
    bag_id, chunk_id = "bag0", "chunk0"
    n_sweeps = 10
    # Same 3 points across all sweeps (definitely static).
    static_xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    for i in range(n_sweeps):
        _write_world_sweep(bag_id, chunk_id, i, static_xyz)
    _write_proc_index(
        bag_id,
        chunk_id,
        list(range(n_sweeps)),
        xyz_per_sweep=[static_xyz] * n_sweeps,
    )

    cfg = ComponentConfig(
        static_sweep_fraction=0.3, static_sweep_min=2, classification_method="persistence"
    )
    result = process_chunk(cfg, bag_id, chunk_id)

    assert result.n_dynamic == 0
    assert result.n_static == n_sweeps * 3

    for i in range(n_sweeps):
        mask = np.load(local_path(dynamic_mask_path(bag_id, chunk_id, i)))
        assert not mask.any(), f"sweep {i}: expected all static"

    static_data = np.load(local_path(static_map_path(bag_id, chunk_id)))
    assert static_data["xyz"].shape[0] == n_sweeps * 3


def test_single_sweep_classified_dynamic(tmp_env):
    """Points seen in only 1 out of 10 sweeps → dynamic.

    Also asserts the new chunk-level dynamic_map.npz: it must carry exactly
    the union of per-sweep dynamic points, with sweep_id-per-point matching
    the originating sweep.  This is the contract proposal_generation reads.
    """
    bag_id, chunk_id = "bag1", "chunk0"
    n_sweeps = 10
    # Background static points present in all sweeps.
    static_xyz = np.array([[100.0, 0.0, 0.0], [101.0, 0.0, 0.0]])
    # Dynamic points present only in sweep 5.
    dynamic_xyz = np.array([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]])

    xyz_per_sweep: list[np.ndarray] = []
    for i in range(n_sweeps):
        if i == 5:
            xyz = np.concatenate([static_xyz, dynamic_xyz])
        else:
            xyz = static_xyz
        _write_world_sweep(bag_id, chunk_id, i, xyz)
        xyz_per_sweep.append(xyz)
    _write_proc_index(
        bag_id, chunk_id, list(range(n_sweeps)), xyz_per_sweep=xyz_per_sweep
    )

    cfg = ComponentConfig(
        static_sweep_fraction=0.3, static_sweep_min=2, classification_method="persistence"
    )
    process_chunk(cfg, bag_id, chunk_id)

    mask_5 = np.load(local_path(dynamic_mask_path(bag_id, chunk_id, 5)))
    # First 2 points are static (background), last 2 are dynamic.
    assert not mask_5[0] and not mask_5[1], "background should be static"
    assert mask_5[2] and mask_5[3], "one-off points should be dynamic"

    # dynamic_map.npz contract: exactly the dynamic points across the chunk,
    # all sweep_ids pointing at sweep 5.
    dyn = np.load(local_path(dynamic_map_path(bag_id, chunk_id)))
    assert dyn["xyz"].shape == (2, 3)
    assert dyn["sweep_id"].shape == (2,)
    assert (dyn["sweep_id"] == 5).all()
    # Coordinates round-trip.
    np.testing.assert_allclose(np.sort(dyn["xyz"][:, 0]), [0.0, 0.15])


def test_dynamic_map_aggregates_across_sweeps(tmp_env):
    """Dynamic points from multiple sweeps land in one chunk-level NPZ."""
    bag_id, chunk_id = "bag_dynagg", "chunk0"
    n_sweeps = 10
    static_xyz = np.array([[100.0, 0.0, 0.0]])
    # Each sweep has its own unique dynamic point.
    xyz_per_sweep: list[np.ndarray] = []
    for i in range(n_sweeps):
        dyn = np.array([[float(i), 50.0, 0.0]])  # unique per sweep
        xyz = np.concatenate([static_xyz, dyn])
        _write_world_sweep(bag_id, chunk_id, i, xyz)
        xyz_per_sweep.append(xyz)
    _write_proc_index(
        bag_id, chunk_id, list(range(n_sweeps)), xyz_per_sweep=xyz_per_sweep
    )

    cfg = ComponentConfig(
        static_sweep_fraction=0.5, static_sweep_min=2, classification_method="persistence"
    )
    process_chunk(cfg, bag_id, chunk_id)

    dyn = np.load(local_path(dynamic_map_path(bag_id, chunk_id)))
    # All 10 unique dynamic points should be in dynamic_map.
    assert dyn["xyz"].shape[0] == n_sweeps
    # sweep_id-per-point must match the originating sweep.
    sort_idx = np.argsort(dyn["sweep_id"])
    np.testing.assert_array_equal(
        dyn["sweep_id"][sort_idx], np.arange(n_sweeps, dtype=np.int32)
    )
    np.testing.assert_allclose(
        dyn["xyz"][sort_idx, 0], np.arange(n_sweeps, dtype=float)
    )


def test_voxel_occupancy_emitted_by_default(tmp_env):
    """save_voxel_occupancy defaults to True (SAM4D / MinkUNet contract)."""
    bag_id, chunk_id = "bag_occ", "chunk0"
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    for i in range(5):
        _write_world_sweep(bag_id, chunk_id, i, xyz)
    _write_proc_index(bag_id, chunk_id, list(range(5)), xyz_per_sweep=[xyz] * 5)

    cfg = ComponentConfig(classification_method="persistence")
    assert cfg.save_voxel_occupancy is True
    process_chunk(cfg, bag_id, chunk_id)

    occ_path = local_path(voxel_occupancy_path(bag_id, chunk_id))
    assert os.path.exists(occ_path), "voxel_occupancy.npz must be written by default"
    data = np.load(occ_path)
    assert data["coords"].shape[1] == 3
    assert data["coords"].dtype == np.int32
    assert "origin" in data and "voxel_size" in data


def test_intensity_backfilled_when_first_sweep_lacks_it(tmp_env):
    """Sweep 0 has no intensity, sweep 1 does → output intensity aligns with xyz."""
    bag_id, chunk_id = "bag_mixed_int", "chunk0"
    n_sweeps = 5
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    # All sweeps share the same xyz so they're all classified static.
    for i in range(n_sweeps):
        path = local_path(lidar_world_path(bag_id, chunk_id, i))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        kwargs = {"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2]}
        # Sweeps 0,1 have no intensity; 2,3,4 do.
        if i >= 2:
            kwargs["intensity"] = np.array([0.5, 0.7], dtype=np.float32)
        np.savez_compressed(path, **kwargs)

    rows = [
        _proc_row(bag_id, chunk_id, sid, pts, has_intensity=sid >= 2)
        for sid in range(n_sweeps)
    ]
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    cfg = ComponentConfig(
        static_sweep_fraction=0.3, static_sweep_min=2, classification_method="persistence"
    )
    process_chunk(cfg, bag_id, chunk_id)

    static = np.load(local_path(static_map_path(bag_id, chunk_id)))
    assert "intensity" in static
    assert static["intensity"].shape[0] == static["xyz"].shape[0], (
        f"intensity ({static['intensity'].shape[0]}) and xyz ({static['xyz'].shape[0]}) "
        "must align"
    )
    # First 4 entries (sweeps 0–1, 2 points each) should be zero-padded.
    assert (static["intensity"][:4] == 0).all()
    # Sweeps 2–4 contributed real intensity values.
    assert (static["intensity"][4:] > 0).all()


def test_empty_proc_index_writes_sentinel(tmp_env):
    """No proc_index rows → write empty static_map + dynamic_map, don't crash."""
    bag_id, chunk_id = "bag_empty", "chunk0"
    write_table([], PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    cfg = ComponentConfig(classification_method="persistence")
    result = process_chunk(cfg, bag_id, chunk_id)
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
        _write_world_sweep(bag_id, chunk_id, i, xyz)
    _write_proc_index(bag_id, chunk_id, list(range(5)), xyz_per_sweep=[xyz] * 5)

    cfg = ComponentConfig(
        static_sweep_fraction=0.1, static_sweep_min=1, classification_method="persistence"
    )
    process_chunk(cfg, bag_id, chunk_id)

    assert os.path.exists(local_path(static_map_path(bag_id, chunk_id)))
    data = np.load(local_path(static_map_path(bag_id, chunk_id)))
    assert "xyz" in data
    assert data["xyz"].shape[1] == 3


def test_log_odds_static_accumulated_from_fixed_point(tmp_env):
    """A point seen in all 10 sweeps from a fixed sensor accumulates enough
    l_occ to exceed p_static_threshold and is classified static.

    Geometry: sensor stays behind the point on the x-axis so no sweep casts a
    through-ray that would carve the endpoint voxel as free.
    """
    bag_id, chunk_id = "bag_lo_static", "chunk0"
    n_sweeps = 10
    # Point at (10, 0, 0).  Sensor well behind at (-5, 0, 0).
    # chunk_origin = (10, 0, 0) (only points at x=10).
    # Voxel of (10,0,0) in chunk space = (0,0,0).
    # Ray length = 15 m; stop_t = 14.85 m; sensor enters chunk at t=15 m → no carve.
    point_xyz = np.array([[10.0, 0.0, 0.0]])
    sensor_origin = np.array([-5.0, 0.0, 0.0])

    for i in range(n_sweeps):
        _write_world_sweep(bag_id, chunk_id, i, point_xyz, origin=sensor_origin)
    _write_proc_index(
        bag_id, chunk_id, list(range(n_sweeps)), xyz_per_sweep=[point_xyz] * n_sweeps
    )

    cfg = ComponentConfig(
        classification_method="log_odds",
        l_occ=0.85,
        l_free=0.40,
        log_odds_clamp=5.0,
        p_static_threshold=0.70,
        p_dynamic_threshold=0.30,
        min_observations=3,
    )
    result = process_chunk(cfg, bag_id, chunk_id)

    assert result.n_static == n_sweeps, "all points should be static"
    assert result.n_dynamic == 0


def test_log_odds_dynamic_via_ray_carving(tmp_env):
    """A voxel hit once (occupied) then passed through by later rays ends up
    with p_occ < p_dynamic_threshold and is classified dynamic.

    Sweep 0: sensor at (-1, 0, 0), point at (0, 0, 0) → l_occ hit on v0.
    Sweeps 1-5: sensor at (-10, 0, 0), point at (5, 0, 0).  The ray enters
    chunk-space voxel (0,0,0) at t=10 m < stop_t=14.85 m → 5× l_free hits.

    Final log_odds[v0] = 0.85 − 5×0.40 = −1.15 → p_occ ≈ 0.24 → dynamic.
    """
    bag_id, chunk_id = "bag_lo_dyn", "chunk0"
    xyz_sweep0 = np.array([[0.0, 0.0, 0.0]])
    origin_sweep0 = np.array([-1.0, 0.0, 0.0])
    xyz_later = np.array([[5.0, 0.0, 0.0]])
    origin_later = np.array([-10.0, 0.0, 0.0])

    _write_world_sweep(bag_id, chunk_id, 0, xyz_sweep0, origin=origin_sweep0)
    for i in range(1, 6):
        _write_world_sweep(bag_id, chunk_id, i, xyz_later, origin=origin_later)

    xyz_per_sweep = [xyz_sweep0] + [xyz_later] * 5
    _write_proc_index(bag_id, chunk_id, list(range(6)), xyz_per_sweep=xyz_per_sweep)

    cfg = ComponentConfig(
        classification_method="log_odds",
        l_occ=0.85,
        l_free=0.40,
        log_odds_clamp=5.0,
        p_static_threshold=0.70,
        p_dynamic_threshold=0.30,
        min_observations=3,
    )
    result = process_chunk(cfg, bag_id, chunk_id)

    # The point at (0,0,0) from sweep 0 should be classified dynamic.
    mask0 = np.load(local_path(dynamic_mask_path(bag_id, chunk_id, 0)))
    assert mask0[0], "voxel carved by later rays must be dynamic"
    assert result.n_static < result.n_dynamic + result.n_static  # at least one dynamic


def test_log_odds_free_only_voxel_not_dynamic(tmp_env):
    """Voxels with n_hits==0 (only traversed, never an endpoint) must not be
    labeled dynamic even when their log_odds are negative.

    Sweep 0: sensor at (-1,0,0), NON-ground point at (5,0,0).  The ray
    traverses chunk-space voxels (0,0,0)-(31,0,0) as free space → n_hits=0.
    Sweep 1: sensor at (-1,0,0), GROUND point at (0.05,0,0) = chunk-space
    voxel (0,0,0).  skip_endpoint keeps n_hits==0 for that voxel.

    Without the n_hits gate the voxel has log_odds=-0.40 → p_occ<p_dyn →
    dynamic.  With the fix: n_hits==0 → free-only → mask=False.
    """
    bag_id, chunk_id = "bag_lo_freeonly", "chunk0"
    xyz0 = np.array([[5.0, 0.0, 0.0]])
    xyz1 = np.array([[0.05, 0.0, 0.0]])
    origin_v = np.array([-1.0, 0.0, 0.0])
    ground_mask1 = np.array([True])

    _write_world_sweep(bag_id, chunk_id, 0, xyz0, origin=origin_v)
    _write_world_sweep(bag_id, chunk_id, 1, xyz1, origin=origin_v, ground_mask=ground_mask1)
    _write_proc_index(bag_id, chunk_id, [0, 1], xyz_per_sweep=[xyz0, xyz1])

    cfg = ComponentConfig(
        classification_method="log_odds",
        ground_endpoint_strategy="skip_endpoint",
        l_occ=0.85,
        l_free=0.40,
        min_observations=3,
        min_occupied_hits=1,
    )
    process_chunk(cfg, bag_id, chunk_id)

    mask1 = np.load(local_path(dynamic_mask_path(bag_id, chunk_id, 1)))
    assert not mask1[0], "ground point in free-only voxel (n_hits==0) must not be labeled dynamic"


def test_static_map_xyz_excludes_free_only_and_under_evidenced(tmp_env):
    """static_map.npz["xyz"] must contain only confident-static points.

    Regression for the masking bug where `static_mask = ~mask` pulled every
    not-dynamic point (including free-only ground voxels and
    under-evidenced-with-hits voxels) into static_map.npz["xyz"], polluting
    the global static cloud with ground returns.

    Build three voxels in one chunk:
      - confident-static: a non-ground point repeated across enough sweeps
        to clear min_observations + p_static_threshold.
      - free-only (ground): voxel only ever traversed; tagged ground in one
        sweep so skip_endpoint suppresses l_occ → n_hits=0.
      - under-evidenced: a single endpoint hit, n_obs<min_observations.

    Post-fix:
      - n_static == n_sweeps (one static point per static-sweep).
      - static_map.npz["xyz"] contains ONLY the confident-static coords;
        neither the ground point nor the under-evidenced point appear.
    """
    bag_id, chunk_id = "bag_static_xyz_purity", "chunk0"
    n_static_sweeps = 5
    static_pt = np.array([[5.0, 0.0, 0.0]])
    ground_pt = np.array([[0.05, 0.0, 0.0]])
    under_pt = np.array([[10.0, 0.0, 0.0]])
    sensor_origin = np.array([-1.0, 0.0, 0.0])

    # Sweeps 0..n_static_sweeps-1: just the static point, all from same origin.
    # Repeated hits on (5,0,0) drive n_obs and l_occ up → confident static.
    xyz_per_sweep: list[np.ndarray] = []
    for i in range(n_static_sweeps):
        _write_world_sweep(bag_id, chunk_id, i, static_pt, origin=sensor_origin)
        xyz_per_sweep.append(static_pt)

    # Sweep n_static_sweeps: ground point, tagged so skip_endpoint skips l_occ.
    # Voxel (0,0,0) was carved by the earlier rays → already negative log_odds,
    # n_hits stays 0 → free-only.
    ground_sweep_id = n_static_sweeps
    _write_world_sweep(
        bag_id,
        chunk_id,
        ground_sweep_id,
        ground_pt,
        origin=sensor_origin,
        ground_mask=np.array([True]),
    )
    xyz_per_sweep.append(ground_pt)

    # Sweep n_static_sweeps+1: single hit on (10,0,0) → n_obs=1<3 (under-evidenced),
    # n_hits=1 → falls into under_evidenced_with_hits bucket (not_dynamic but
    # not static).
    under_sweep_id = n_static_sweeps + 1
    _write_world_sweep(bag_id, chunk_id, under_sweep_id, under_pt, origin=sensor_origin)
    xyz_per_sweep.append(under_pt)

    sweep_ids = list(range(n_static_sweeps + 2))
    _write_proc_index(bag_id, chunk_id, sweep_ids, xyz_per_sweep=xyz_per_sweep)

    cfg = ComponentConfig(
        classification_method="log_odds",
        ground_endpoint_strategy="skip_endpoint",
        l_occ=0.85,
        l_free=0.40,
        p_static_threshold=0.70,
        min_observations=3,
        min_occupied_hits=1,
    )
    result = process_chunk(cfg, bag_id, chunk_id)

    # Sanity: the static point made it through the static gates.
    assert result.n_static == n_static_sweeps, (
        f"expected {n_static_sweeps} confident-static points, got {result.n_static}"
    )

    static_data = np.load(local_path(static_map_path(bag_id, chunk_id)))
    static_xyz = static_data["xyz"]

    # Every saved point must be the static point — no ground, no under-evidenced.
    assert static_xyz.shape[0] == n_static_sweeps
    np.testing.assert_allclose(static_xyz, np.repeat(static_pt, n_static_sweeps, axis=0))

    flat = static_xyz.reshape(-1, 3)
    assert not (flat == ground_pt[0]).all(axis=1).any(), (
        "ground point in free-only voxel must NOT appear in static_map.npz['xyz']"
    )
    assert not (flat == under_pt[0]).all(axis=1).any(), (
        "under-evidenced point must NOT appear in static_map.npz['xyz']"
    )


def test_under_evidenced_with_hits_not_dynamic(tmp_env):
    """Under-evidenced voxel WITH endpoint hits gets the benefit of the doubt.

    One sweep, one point at (5, 0, 0), sensor at (-1, 0, 0), min_observations=3,
    min_occupied_hits=1.  The endpoint voxel has n_hits=1, n_obs=1:
      - evidenced = False (1 < 3)
      - has_hits  = True (1 >= 1)
      - → under_evidenced_with_hits bucket → not_dynamic_arr → mask=False
      - but NOT in static_arr (n_static stays 0)

    This is exactly the false-positive-dynamic case bug fix #2 addresses.
    Pre-fix this test would have asserted `n_dynamic == 1`; post-fix it must
    be 0 because under-evidenced-with-hits is no longer labeled dynamic.
    """
    bag_id, chunk_id = "bag_lo_under", "chunk0"
    xyz = np.array([[5.0, 0.0, 0.0]])
    sensor_origin = np.array([-1.0, 0.0, 0.0])

    _write_world_sweep(bag_id, chunk_id, 0, xyz, origin=sensor_origin)
    _write_proc_index(bag_id, chunk_id, [0], xyz_per_sweep=[xyz])

    cfg = ComponentConfig(
        classification_method="log_odds",
        min_observations=3,
        min_occupied_hits=1,
    )
    result = process_chunk(cfg, bag_id, chunk_id)

    assert result.n_static == 0, "under-evidenced voxel must not be static"
    assert result.n_dynamic == 0, (
        "under-evidenced voxel WITH hits must not be labeled dynamic "
        "(bug fix #2)"
    )

    mask = np.load(local_path(dynamic_mask_path(bag_id, chunk_id, 0)))
    assert mask.shape == (1,)
    assert not mask[0], "the single point must have mask=False"


def test_skip_endpoint_isolated_ground_voxel_not_dynamic(tmp_env):
    """skip_endpoint: a ground voxel that no other ray traverses must still
    be filtered out by the patchwork mask.

    Single sweep, single ground point.  No other sweep's ray ever crosses the
    endpoint voxel, so it never enters unique_keys.  Without the unconditional
    ground-mask post-filter the Pass 2 searchsorted reports "not found" →
    mask=True → ground point ends up in dynamic_map.npz.  Locks in the fix
    that always honours patchwork's per-point ground call regardless of
    ground_endpoint_strategy.
    """
    bag_id, chunk_id = "bag_lo_skipendpoint_iso", "chunk0"
    xyz = np.array([[0.05, 0.0, 0.01]])
    ground_mask = np.array([True])
    sensor_origin = np.array([-1.0, 0.0, 0.0])

    _write_world_sweep(
        bag_id, chunk_id, 0, xyz, origin=sensor_origin, ground_mask=ground_mask
    )
    _write_proc_index(bag_id, chunk_id, [0], xyz_per_sweep=[xyz])

    cfg = ComponentConfig(
        classification_method="log_odds",
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
    """In skip_ray mode, ground voxels must not be labeled dynamic.

    Two-point sweep: ground point at (0.05, 0, 0.01), non-ground at (5, 0, 0).
    skip_ray means ground rays are never traversed → ground voxels have no
    entry in log_odds_dict → would fall into the dynamic bucket without fix #4.

    Bug fix #4 post-filters the mask by ground_mask in skip_ray mode so:
      - mask length == total world NPZ point count (hard contract)
      - mask[ground positions] == False
    """
    bag_id, chunk_id = "bag_lo_skipray", "chunk0"
    xyz = np.array([[0.05, 0.0, 0.01], [5.0, 0.0, 0.0]])
    ground_mask = np.array([True, False])
    sensor_origin = np.array([-1.0, 0.0, 0.0])

    _write_world_sweep(
        bag_id, chunk_id, 0, xyz, origin=sensor_origin, ground_mask=ground_mask
    )
    _write_proc_index(bag_id, chunk_id, [0], xyz_per_sweep=[xyz])

    cfg = ComponentConfig(
        classification_method="log_odds",
        ground_endpoint_strategy="skip_ray",
        min_observations=1,
        min_occupied_hits=1,
    )
    process_chunk(cfg, bag_id, chunk_id)

    mask = np.load(local_path(dynamic_mask_path(bag_id, chunk_id, 0)))
    # Contract 1: mask length matches world NPZ length (NOT the filtered count).
    assert mask.shape == (2,), (
        f"mask length must match total world NPZ points (2), got {mask.shape[0]}. "
        "A buggy impl that filters at sweep_keys level would give length 1."
    )
    # Contract 2: the ground point at index 0 must be False (bug fix #4).
    assert not mask[0], (
        "ground point in skip_ray mode must not be labeled dynamic (bug fix #4)"
    )


def _write_mf_mos_mask(
    bag_id: str, chunk_id: str, sweep_id: int, mask: np.ndarray
) -> str:
    """Write a MF-MOS bool mask .npy and return the URI."""
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
    """Variant of _proc_row that includes mf_mos_mask_path."""
    row = _proc_row(bag_id, chunk_id, sweep_id, xyz)
    row["mf_mos_mask_path"] = mf_mos_mask_uri
    return row


def test_voxel_vote_fusion_requires_min_votes(tmp_env):
    """A single-sweep MF-MOS vote is insufficient when min_mf_mos_votes=2.

    One sweep, one point, MF-MOS flags it moving.  With min_mf_mos_votes=2,
    a single vote is below the threshold → no mf_mos_dynamic_arr entry →
    voxel must not be labeled dynamic.
    """
    bag_id, chunk_id = "bag_vote_min", "chunk0"
    xyz = np.array([[5.0, 0.0, 0.0]])
    sensor_origin = np.array([-1.0, 0.0, 0.0])

    _write_world_sweep(bag_id, chunk_id, 0, xyz, origin=sensor_origin)
    mf_uri = _write_mf_mos_mask(bag_id, chunk_id, 0, np.array([True]))

    row = _proc_row_mf_mos(bag_id, chunk_id, 0, xyz, mf_uri)
    write_table([row], PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    cfg = ComponentConfig(
        classification_method="log_odds",
        min_observations=3,
        min_occupied_hits=1,
        min_mf_mos_votes=2,
        mf_mos_vote_fraction_threshold=0.5,
        mf_mos={"enabled": True, "fusion_mode": "union"},
    )
    result = process_chunk(cfg, bag_id, chunk_id)

    assert result.n_dynamic == 0, (
        "single-sweep vote below min_mf_mos_votes=2 must not produce dynamic points"
    )
    mask = np.load(local_path(dynamic_mask_path(bag_id, chunk_id, 0)))
    assert not mask[0], "point in below-threshold-vote voxel must be mask=False"


def test_voxel_vote_fusion_union_threshold(tmp_env):
    """MF-MOS votes meeting threshold override AW static in union mode.

    Five sweeps hit the same voxel; 3 of 5 MF-MOS masks flag it moving
    (vote_fraction = 0.6 >= 0.5, votes = 3 >= 1).  AW sees 5 hits with
    min_observations=3 → confident static.  Union mode must override: all
    5 sweep points become dynamic, none static.
    """
    bag_id, chunk_id = "bag_vote_union", "chunk0"
    xyz = np.array([[5.0, 0.0, 0.0]])
    sensor_origin = np.array([-1.0, 0.0, 0.0])
    n_sweeps = 5
    mf_flags = [True, True, True, False, False]

    rows = []
    for i in range(n_sweeps):
        _write_world_sweep(bag_id, chunk_id, i, xyz, origin=sensor_origin)
        mf_uri = _write_mf_mos_mask(bag_id, chunk_id, i, np.array([mf_flags[i]]))
        rows.append(_proc_row_mf_mos(bag_id, chunk_id, i, xyz, mf_uri))
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    cfg = ComponentConfig(
        classification_method="log_odds",
        l_occ=0.85,
        l_free=0.40,
        min_observations=3,
        min_occupied_hits=1,
        min_mf_mos_votes=1,
        mf_mos_vote_fraction_threshold=0.5,
        mf_mos={"enabled": True, "fusion_mode": "union"},
    )
    result = process_chunk(cfg, bag_id, chunk_id)

    assert result.n_dynamic == n_sweeps, (
        f"all {n_sweeps} points must be dynamic via MF-MOS union; got {result.n_dynamic}"
    )
    assert result.n_static == 0, (
        "voxel in mf_mos_dynamic_arr must be removed from static cloud"
    )


def test_voxel_vote_fusion_independent_mode_no_effect(tmp_env):
    """fusion_mode=independent: MF-MOS votes have no effect on AW classification.

    Same five-sweep setup as test_voxel_vote_fusion_union_threshold, but with
    fusion_mode=independent.  Vote accumulation is skipped entirely → AW result
    stands alone → confident-static voxel → n_dynamic == 0.
    """
    bag_id, chunk_id = "bag_vote_indep", "chunk0"
    xyz = np.array([[5.0, 0.0, 0.0]])
    sensor_origin = np.array([-1.0, 0.0, 0.0])
    n_sweeps = 5
    mf_flags = [True, True, True, False, False]

    rows = []
    for i in range(n_sweeps):
        _write_world_sweep(bag_id, chunk_id, i, xyz, origin=sensor_origin)
        mf_uri = _write_mf_mos_mask(bag_id, chunk_id, i, np.array([mf_flags[i]]))
        rows.append(_proc_row_mf_mos(bag_id, chunk_id, i, xyz, mf_uri))
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    cfg = ComponentConfig(
        classification_method="log_odds",
        l_occ=0.85,
        l_free=0.40,
        min_observations=3,
        min_occupied_hits=1,
        min_mf_mos_votes=1,
        mf_mos_vote_fraction_threshold=0.5,
        mf_mos={"enabled": True, "fusion_mode": "independent"},
    )
    result = process_chunk(cfg, bag_id, chunk_id)

    assert result.n_dynamic == 0, (
        "independent fusion_mode must not apply MF-MOS votes to AW classification"
    )


# ---------------------------------------------------------------------------
# Ground-leak tests — ground points must NEVER appear in dynamic_map.npz
# ---------------------------------------------------------------------------


def test_union_fusion_must_not_reintroduce_ground_via_mf_mos_vote(tmp_env):
    """BUG: MF-MOS union fusion ORs ground voxels back into the dynamic mask.

    masking.py pipeline order:
        mask = ~is_not_dynamic                # AW classification
        mask &= ~ground_mask_cache_i           # ground filter (correct)
        mask = mask | is_mf_mos_dyn            # union OR — RE-ADDS GROUND

    Setup: two points in the same voxel — a ground point (z=0) and a non-
    ground point (z=0.05) that MF-MOS flags moving. The non-ground point's
    return puts the voxel in unique_keys (n_hits=1), the vote populates
    mf_mos_dynamic_arr, and the union OR drags the ground point along into
    dynamic_map.npz.
    """
    bag_id, chunk_id = "bag_union_ground", "chunk0"
    # Two co-voxel points (default voxel_size_m=0.15 → both at voxel (33, 0, 0)):
    # index 0 is ground, index 1 is a non-ground moving point.
    xyz = np.array(
        [
            [5.0, 0.0, 0.0],   # ground point — must NOT be dynamic
            [5.0, 0.0, 0.05],  # non-ground moving point — MAY be dynamic
        ]
    )
    ground_mask = np.array([True, False])
    mf_flags = np.array([False, True])  # only the non-ground point votes

    _write_world_sweep(
        bag_id,
        chunk_id,
        0,
        xyz,
        origin=np.array([-1.0, 0.0, 0.0]),
        ground_mask=ground_mask,
    )
    mf_uri = _write_mf_mos_mask(bag_id, chunk_id, 0, mf_flags)
    rows = [_proc_row_mf_mos(bag_id, chunk_id, 0, xyz, mf_uri)]
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    cfg = ComponentConfig(
        classification_method="log_odds",
        ground_endpoint_strategy="skip_endpoint",
        min_observations=1,
        min_occupied_hits=1,
        min_mf_mos_votes=1,
        mf_mos_vote_fraction_threshold=0.5,
        mf_mos={"enabled": True, "fusion_mode": "union"},
    )
    process_chunk(cfg, bag_id, chunk_id)

    dyn = np.load(local_path(dynamic_map_path(bag_id, chunk_id)))
    is_ground_pt = np.all(np.isclose(dyn["xyz"], xyz[0]), axis=1)
    assert not is_ground_pt.any(), (
        f"ground point {xyz[0].tolist()} must not appear in dynamic_map.npz; "
        f"found it {int(is_ground_pt.sum())} time(s). MF-MOS union fusion "
        f"OR-ed the voxel back into the mask without re-applying ~ground_mask "
        f"after the OR — see masking.py:117."
    )
    mask = np.load(local_path(dynamic_mask_path(bag_id, chunk_id, 0)))
    assert not mask[0], (
        "ground point's dynamic-mask bit must be False even when its voxel "
        "is in mf_mos_dynamic_arr under union fusion"
    )


def test_mfmos_only_fusion_must_not_label_ground_dynamic(tmp_env):
    """BUG: mfmos_only fusion overwrites the ground-filtered mask entirely.

    masking.py:119 in mfmos_only mode does ``mask = is_mf_mos_dyn`` —
    discards the ground filter applied moments earlier. Any ground voxel
    that MF-MOS flags (typically via a co-voxel non-ground moving point)
    becomes dynamic.
    """
    bag_id, chunk_id = "bag_mfonly_ground", "chunk0"
    xyz = np.array([[5.0, 0.0, 0.0], [5.0, 0.0, 0.05]])
    ground_mask = np.array([True, False])
    mf_flags = np.array([False, True])

    _write_world_sweep(
        bag_id,
        chunk_id,
        0,
        xyz,
        origin=np.array([-1.0, 0.0, 0.0]),
        ground_mask=ground_mask,
    )
    mf_uri = _write_mf_mos_mask(bag_id, chunk_id, 0, mf_flags)
    rows = [_proc_row_mf_mos(bag_id, chunk_id, 0, xyz, mf_uri)]
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    cfg = ComponentConfig(
        classification_method="log_odds",
        ground_endpoint_strategy="skip_endpoint",
        min_observations=1,
        min_occupied_hits=1,
        min_mf_mos_votes=1,
        mf_mos_vote_fraction_threshold=0.5,
        mf_mos={"enabled": True, "fusion_mode": "mfmos_only"},
    )
    process_chunk(cfg, bag_id, chunk_id)

    dyn = np.load(local_path(dynamic_map_path(bag_id, chunk_id)))
    is_ground_pt = np.all(np.isclose(dyn["xyz"], xyz[0]), axis=1)
    assert not is_ground_pt.any(), (
        f"ground point {xyz[0].tolist()} must not appear in dynamic_map.npz "
        f"under mfmos_only; found it {int(is_ground_pt.sum())} time(s). "
        f"masking.py:119 overwrites the ground-filtered mask with "
        f"is_mf_mos_dyn — the ground filter needs to be re-applied after."
    )


def test_persistence_mode_must_apply_ground_filter(tmp_env):
    """BUG: persistence path never applies the per-sweep ground filter.

    masking.py:87 gates the ground AND-filter on
    `classification_method == 'log_odds'`, and _run_pass_1_persistence in
    classify/pipeline.py initialises ground_mask_cache to all-None.  So in
    persistence mode, any ground voxel whose static-sweep-fraction falls
    short of the threshold ends up in dynamic_map.npz.

    Two sweeps, single ground point each.  static_sweep_min defaults to 5
    so 2 hits < 5 → voxel never gets marked static → AW thinks it's dynamic
    → no ground filter intercedes → ground point ends up dynamic.
    """
    bag_id, chunk_id = "bag_pers_ground", "chunk0"
    ground_pt = np.array([[0.05, 0.05, 0.0]])
    n_sweeps = 2
    for sid in range(n_sweeps):
        _write_world_sweep(
            bag_id,
            chunk_id,
            sid,
            ground_pt,
            origin=np.array([-1.0, 0.0, 0.0]),
            ground_mask=np.array([True]),
        )
    _write_proc_index(
        bag_id, chunk_id, list(range(n_sweeps)), xyz_per_sweep=[ground_pt] * n_sweeps
    )

    cfg = ComponentConfig(classification_method="persistence")
    process_chunk(cfg, bag_id, chunk_id)

    dyn = np.load(local_path(dynamic_map_path(bag_id, chunk_id)))
    assert dyn["xyz"].shape[0] == 0, (
        f"persistence mode leaks ground points into dynamic_map.npz; "
        f"got {dyn['xyz'].shape[0]} dynamic points (all ground-flagged). "
        f"The ground AND-filter in masking.py:87 is gated on log_odds "
        f"but the same filtering is required for persistence."
    )


def test_classify_raises_when_world_npz_missing_origin(tmp_env):
    """classify must hard-fail when a valid sweep's world NPZ has no 'origin'.

    Silently skipping this case was the previous behavior; it left the
    sweep's voxels out of unique_keys, which made every point in those
    voxels classify as dynamic-by-default (via the masking.py
    "not in not_dynamic_arr" branch).

    Re-running deskew always writes 'origin' — a missing field is artifact
    corruption and the user needs to know.
    """
    bag_id, chunk_id = "bag_no_origin", "chunk0"
    xyz = np.array([[5.0, 0.0, 1.0]])
    # _write_world_sweep with origin=None deliberately omits the field.
    _write_world_sweep(bag_id, chunk_id, 0, xyz)  # no origin kwarg
    _write_proc_index(bag_id, chunk_id, [0], xyz_per_sweep=[xyz])

    cfg = ComponentConfig(classification_method="log_odds")
    with pytest.raises(ValueError, match="origin"):
        process_chunk(cfg, bag_id, chunk_id)


def test_clamp_asymmetry_does_not_carve_heavily_observed_static(tmp_env):
    """Heavily-observed static voxels must survive a moderate number of carves.

    Bug: log_odds_clamp is applied symmetrically (+clamp on hits, −clamp on
    carves).  With l_occ=1.20 and clamp=5.0, a voxel reaches the +5.0
    ceiling after only ~4 hits — additional hits add NOTHING.  Carves
    keep subtracting at -l_free=0.25 each.  A wall observed by N>>4
    sweeps with M carving through-rays ends up at log_odds = +5 − 0.25·M,
    which goes deeply negative as M grows even though N hits >> M carves
    in an unclamped world.

    This is the "buildings/trees as red dynamic" leak: well-observed
    statics that are nonetheless carved by a moderate number of through-
    rays end up confidently dynamic when they should still be static.

    Setup: 1 wall voxel at world (5, 0, 1.0). 20 hit-sweeps endpoint here;
    30 carving-sweeps endpoint behind it at (15, 0, 1.0) so their rays
    pass through. With unclamped math: log_odds = 20·1.20 − 30·0.25 =
    +16.5 → solidly static.  With clamp=5.0: log_odds = +5.0 − 7.5 =
    −2.5 → p_occ ≈ 0.08 → classified DYNAMIC.  Bug.
    """
    bag_id, chunk_id = "bag_clamp_asym", "chunk0"
    wall_pt = np.array([[5.0, 0.0, 1.0]])
    beyond_pt = np.array([[15.0, 0.0, 1.0]])
    sensor = np.array([-5.0, 0.0, 1.0])

    n_hits = 20
    n_carves = 30
    rows_xyz: list[np.ndarray] = []
    for sid in range(n_hits):
        _write_world_sweep(bag_id, chunk_id, sid, wall_pt, origin=sensor)
        rows_xyz.append(wall_pt)
    for j in range(n_carves):
        sid = n_hits + j
        _write_world_sweep(bag_id, chunk_id, sid, beyond_pt, origin=sensor)
        rows_xyz.append(beyond_pt)
    _write_proc_index(
        bag_id,
        chunk_id,
        list(range(n_hits + n_carves)),
        xyz_per_sweep=rows_xyz,
    )

    cfg = ComponentConfig(
        classification_method="log_odds",
        ground_endpoint_strategy="skip_endpoint",
        l_occ=1.20,
        l_free=0.25,
        # Bug-fix value: clamp must be MUCH larger than l_occ so well-
        # observed statics build healthy headroom.  With the pre-fix 5.0
        # default, this test fails — 20 hits cap at +5 (after ~4) and 30
        # carves bring it to −2.5 (deep red).
        log_odds_clamp=50.0,
        voxel_size_m=0.25,
        free_space_margin_voxels=2.0,
        max_ray_length_m=50.0,
        min_observations=3,
        min_occupied_hits=1,
        p_static_threshold=0.70,
        p_dynamic_threshold=0.30,
    )
    result = process_chunk(cfg, bag_id, chunk_id)

    dyn = np.load(local_path(dynamic_map_path(bag_id, chunk_id)))
    is_wall_pt = np.all(np.isclose(dyn["xyz"], wall_pt[0]), axis=1)
    assert not is_wall_pt.any(), (
        f"wall voxel observed by {n_hits} hit-sweeps must not be labeled "
        f"dynamic just because {n_carves} other sweeps carved through it; "
        f"got {int(is_wall_pt.sum())} occurrences in dynamic_map.npz. "
        f"With log_odds_clamp=50.0, log_odds = 20·1.20 − 30·0.25 = +16.5 "
        f"→ p_occ ≈ 1.0 → solidly static."
    )
    assert result.n_static >= n_hits, (
        f"wall voxel observed by {n_hits} sweeps should produce >= {n_hits} "
        f"static points; got {result.n_static}. The hit-sweep endpoints "
        f"must remain in static_map.npz."
    )


def test_ambiguous_log_odds_voxel_not_dynamic(tmp_env):
    """AW must not classify "ambiguous" voxels as dynamic.

    Bug: cfg.p_dynamic_threshold is configured (default 0.30) and validated
    but never consulted in classify_from_log_odds.  The only static gate is
    p_occ >= p_static_threshold; anything below that with evidence and hits
    falls through to the implicit dynamic bucket.  This leaks textured
    statics (brick walls, tree canopies, sparse foliage) whose endpoint
    hits get partially carved by through-ray beam noise.

    Reproduce: one sweep, one point.  log_odds = l_occ = 0.85; p_occ = 0.70.
    With p_static_threshold = 0.85 the voxel is NOT static.  Without the
    p_dynamic_threshold gate (= 0.10 here), it falls through to dynamic.

    Correct behaviour: only voxels with p_occ < p_dynamic_threshold AND
    hits are dynamic.  Ambiguous voxels go to not_dynamic (excluded from
    BOTH static and dynamic clouds, same as under-evidenced-with-hits).
    """
    bag_id, chunk_id = "bag_ambig", "chunk0"
    voxel_pt = np.array([[5.0, 0.0, 0.0]])

    _write_world_sweep(
        bag_id, chunk_id, 0, voxel_pt, origin=np.array([-1.0, 0.0, 0.0])
    )
    _write_proc_index(bag_id, chunk_id, [0], xyz_per_sweep=[voxel_pt])

    cfg = ComponentConfig(
        classification_method="log_odds",
        ground_endpoint_strategy="skip_endpoint",
        l_occ=0.85,
        l_free=0.40,
        min_observations=1,
        min_occupied_hits=1,
        # Threshold band engineered so single-hit (p_occ ≈ 0.70) lands in
        # the ambiguous range [0.10, 0.85).
        p_static_threshold=0.85,
        p_dynamic_threshold=0.10,
    )
    process_chunk(cfg, bag_id, chunk_id)

    dyn = np.load(local_path(dynamic_map_path(bag_id, chunk_id)))
    assert dyn["xyz"].shape[0] == 0, (
        f"voxel with sigmoid(log_odds) ≈ 0.70 (in ambiguous range "
        f"[p_dynamic_threshold=0.10, p_static_threshold=0.85)) must NOT "
        f"appear in dynamic_map.npz; got {dyn['xyz'].shape[0]} dynamic "
        f"points. Fix: respect cfg.p_dynamic_threshold in "
        f"classify_from_log_odds — only p_occ < p_dynamic_threshold AND "
        f"has_hits is confidently dynamic."
    )


def test_cache_disabled_must_still_apply_ground_filter(tmp_env):
    """BUG: when in-memory cache is off, ground_mask is dropped and not re-loaded.

    log_odds.py:113 only appends ground_mask to ground_mask_cache when
    cache_xyz is True.  Pass 2's masking.py re-loads xyz/intensity from disk
    via load_world_xyz_intensity but NOT ground_mask, so the ground filter
    at masking.py:87 silently no-ops on big chunks where cache auto-disabled.

    Force the bug with cache_world_xyz_in_memory=False on a tiny chunk so
    the failure cause is the missing reload, not cache size.
    """
    bag_id, chunk_id = "bag_nocache_ground", "chunk0"
    # One ground point + one tiny "moving" voxel that AW labels dynamic, so
    # the ground point's dynamic-mask outcome is determined by the ground
    # filter (not by happening to land in not_dynamic_arr).
    ground_pt = np.array([[0.05, 0.05, 0.0]])
    _write_world_sweep(
        bag_id,
        chunk_id,
        0,
        ground_pt,
        origin=np.array([-1.0, 0.0, 0.0]),
        ground_mask=np.array([True]),
    )
    _write_proc_index(bag_id, chunk_id, [0], xyz_per_sweep=[ground_pt])

    cfg = ComponentConfig(
        classification_method="log_odds",
        ground_endpoint_strategy="skip_endpoint",
        cache_world_xyz_in_memory=False,
        min_observations=1,
        min_occupied_hits=1,
    )
    process_chunk(cfg, bag_id, chunk_id)

    dyn = np.load(local_path(dynamic_map_path(bag_id, chunk_id)))
    assert dyn["xyz"].shape[0] == 0, (
        f"cache-disabled path leaks ground points into dynamic_map.npz; "
        f"got {dyn['xyz'].shape[0]} dynamic points. log_odds.py only caches "
        f"ground_mask when cache_xyz is True, and masking.py never re-reads "
        f"it from disk."
    )
