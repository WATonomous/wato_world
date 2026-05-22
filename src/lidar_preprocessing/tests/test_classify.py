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
