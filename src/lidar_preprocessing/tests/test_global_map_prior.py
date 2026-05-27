"""Tests for the two-pass global static map prior (UniLiPs IWU).

Covers:
  * GlobalMapPrior.query_sweep returns the expected (map_hit, r_star) for
    empty maps and for synthetic 3-point maps with known sweep matches.
  * _apply_global_map_boost only mutates log_odds_dict — never n_hits_dict
    or n_obs_dict (the prior must not bypass the has_hits gate).
  * Running the log-odds pass with a prior produces strictly higher log-odds
    on map-matched voxels, with identical n_hits to the no-prior run.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from wato_common.artifact_store import (
    lidar_proc_index_path,
    lidar_world_path,
    local_path,
)
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import PROCESSED_SWEEPS_SCHEMA
from wato_lidar_preprocessing.classify import GlobalMapPrior, process_chunk
from wato_lidar_preprocessing.classify.io_helpers import origin_from_index
from wato_lidar_preprocessing.classify.log_odds import build_log_odds_grid
from wato_lidar_preprocessing.config import ComponentConfig
from wato_lidar_preprocessing.ray_traversal import (
    apply_global_map_boost,
    make_log_odds_dicts,
)
from wato_lidar_preprocessing.voxel import voxel_indices


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
    origin: np.ndarray,
):
    path = local_path(lidar_world_path(bag_id, chunk_id, sweep_id))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        origin=np.asarray(origin, dtype=np.float64),
    )


def _proc_row(bag_id: str, chunk_id: str, sweep_id: int, xyz: np.ndarray) -> dict:
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
        "has_intensity": False,
        "deskewed": True,
        "world_xmin": float(xyz[:, 0].min()) if n else None,
        "world_xmax": float(xyz[:, 0].max()) if n else None,
        "world_ymin": float(xyz[:, 1].min()) if n else None,
        "world_ymax": float(xyz[:, 1].max()) if n else None,
        "world_zmin": float(xyz[:, 2].min()) if n else None,
        "world_zmax": float(xyz[:, 2].max()) if n else None,
    }


# ---------------------------------------------------------------------------
# 1. Empty global map → query_sweep is a safe no-op
# ---------------------------------------------------------------------------


def test_global_map_prior_empty_map():
    prior = GlobalMapPrior(np.empty((0, 3), dtype=np.float64), match_radius_m=0.3)
    xyz = np.array([[1.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    sensor = np.zeros(3)
    map_hit, r_star = prior.query_sweep(xyz, sensor)

    assert map_hit.shape == (2,)
    assert not map_hit.any(), "empty map cannot match anything"
    assert r_star.shape == (2,)
    # r_max default 200 → both points are within range → r_star == 1.0
    np.testing.assert_allclose(r_star, [1.0, 1.0])


def test_global_map_prior_handles_zero_length_sweep():
    """Zero-point sweep should not crash."""
    map_xyz = np.array([[0.0, 0.0, 0.0]])
    prior = GlobalMapPrior(map_xyz, match_radius_m=0.3)
    map_hit, r_star = prior.query_sweep(np.empty((0, 3)), np.zeros(3))
    assert map_hit.shape == (0,)
    assert r_star.shape == (0,)


# ---------------------------------------------------------------------------
# 2. Synthetic 3-point map, 5-point sweep — verify map_hit + r_star math
# ---------------------------------------------------------------------------


def test_global_map_prior_query_sweep_matches_within_radius():
    # Three known static points along the +x axis at 1m, 5m, 100m.
    map_xyz = np.array(
        [
            [1.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ]
    )
    prior = GlobalMapPrior(map_xyz, match_radius_m=0.30, r_max_credibility_m=200.0)

    sweep_xyz = np.array(
        [
            [1.10, 0.0, 0.0],   # 0.10 m from map[0] → HIT
            [5.40, 0.0, 0.0],   # 0.40 m from map[1] → MISS (radius is 0.30)
            [100.0, 0.0, 0.0],  # exact match map[2] → HIT
            [50.0, 0.0, 0.0],   # 45 m from map[1], 50 m from map[2] → MISS
            [0.99, 0.0, 0.0],   # 0.01 m from map[0] → HIT
        ]
    )
    sensor = np.zeros(3)

    map_hit, r_star = prior.query_sweep(sweep_xyz, sensor)

    expected_hit = np.array([True, False, True, False, True])
    np.testing.assert_array_equal(map_hit, expected_hit)

    # r_star = min(1, 200/range).  For all points here range < 200 so r_star = 1.0.
    np.testing.assert_allclose(r_star, [1.0, 1.0, 1.0, 1.0, 1.0])


def test_global_map_prior_r_star_attenuates_past_r_max():
    """Points beyond r_max get r_star < 1 (range-credibility decay)."""
    map_xyz = np.array([[300.0, 0.0, 0.0]])
    prior = GlobalMapPrior(map_xyz, match_radius_m=0.3, r_max_credibility_m=100.0)

    sweep_xyz = np.array(
        [
            [50.0, 0.0, 0.0],    # range 50 → r_star = min(1, 100/50) = 1.0
            [100.0, 0.0, 0.0],   # range 100 → r_star = 1.0 exactly
            [200.0, 0.0, 0.0],   # range 200 → r_star = 100/200 = 0.5
            [400.0, 0.0, 0.0],   # range 400 → r_star = 100/400 = 0.25
        ]
    )
    sensor = np.zeros(3)
    _map_hit, r_star = prior.query_sweep(sweep_xyz, sensor)
    np.testing.assert_allclose(r_star, [1.0, 1.0, 0.5, 0.25], rtol=1e-6)


# ---------------------------------------------------------------------------
# 3. apply_global_map_boost: only mutates log_odds (never n_obs / n_hits)
# ---------------------------------------------------------------------------


def test_apply_global_map_boost_injects_log_odds_only():
    """Boost adds to log_odds.  n_obs / n_hits dicts must NOT be touched.

    Bypassing n_hits via a synthetic prior would let voxels the current chunk
    never observed pass the has_hits gate in classify_from_log_odds — the prior
    is allowed to nudge p_occ up, but real sweep returns must back the
    voxel before it can become static.
    """
    log_odds, n_obs, n_hits = make_log_odds_dicts()
    # Pre-seed a voxel so we can verify additive (not overwrite) behaviour.
    log_odds[42] = np.float32(0.2)

    hit_keys = np.array([42, 42, 7, 100], dtype=np.int64)  # voxel 42 hit twice
    hit_r_star = np.array([0.5, 0.8, 1.0, 0.25], dtype=np.float32)
    l_occ_boost = 0.10
    clamp = 50.0

    apply_global_map_boost(hit_keys, hit_r_star, l_occ_boost, clamp, log_odds)

    # Voxel 42: max r_star across its two hits is 0.8 → boost = 0.1 * 0.8 = 0.08.
    #          Previous lo = 0.2 → new lo = 0.28.
    np.testing.assert_allclose(float(log_odds[42]), 0.2 + 0.10 * 0.8, atol=1e-6)
    # Voxel 7: boost = 0.1 * 1.0 = 0.1 (no prior value → starts at 0).
    np.testing.assert_allclose(float(log_odds[7]), 0.10 * 1.0, atol=1e-6)
    # Voxel 100: boost = 0.1 * 0.25 = 0.025.
    np.testing.assert_allclose(float(log_odds[100]), 0.10 * 0.25, atol=1e-6)

    # n_hits / n_obs must be untouched — this is the safety contract.
    assert len(n_hits) == 0, "boost must not increment n_hits (bypasses has_hits gate)"
    assert len(n_obs) == 0, "boost must not increment n_obs"


def test_apply_global_map_boost_respects_clamp():
    """Boost saturates at log_odds_clamp, doesn't overshoot."""
    log_odds, _, _ = make_log_odds_dicts()
    log_odds[1] = np.float32(49.5)
    hit_keys = np.array([1], dtype=np.int64)
    hit_r_star = np.array([1.0], dtype=np.float32)
    apply_global_map_boost(hit_keys, hit_r_star, l_occ_boost=2.0, clamp=50.0, log_odds=log_odds)
    # 49.5 + 2.0 = 51.5 → clamped to 50.0.
    assert float(log_odds[1]) == pytest.approx(50.0)


def test_apply_global_map_boost_empty_hits_is_noop():
    log_odds, _, _ = make_log_odds_dicts()
    log_odds[5] = np.float32(1.0)
    apply_global_map_boost(
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float32),
        l_occ_boost=0.5,
        clamp=50.0,
        log_odds=log_odds,
    )
    assert float(log_odds[5]) == pytest.approx(1.0)
    assert len(log_odds) == 1


# ---------------------------------------------------------------------------
# 4. End-to-end: build_log_odds_grid with vs. without prior
# ---------------------------------------------------------------------------


def test_build_log_odds_grid_with_prior_increases_lo(tmp_env):
    """Prior increases log_odds on map-matched voxels; n_hits stays identical.

    Builds a synthetic chunk with 3 sweeps all hitting the same voxel from
    the same sensor origin.  Runs build_log_odds_grid twice:
      (a) no prior → baseline lo + n_hits per voxel
      (b) with a prior containing that voxel's centre → lo strictly increases,
          n_hits stays identical (since prior must NOT add synthetic hits).
    """
    bag_id, chunk_id = "bag_prior", "chunk0"
    n_sweeps = 3
    # One static point at world (1.0, 0.0, 0.0); sensor at the world origin.
    static_pt = np.array([[1.0, 0.0, 0.0]])
    chunk_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    for sid in range(n_sweeps):
        _write_world_sweep(bag_id, chunk_id, sid, static_pt, origin=chunk_origin)
    rows = [_proc_row(bag_id, chunk_id, sid, static_pt) for sid in range(n_sweeps)]
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    cfg = ComponentConfig(
        classification_method="log_odds",
        voxel_size_m=0.15,
        ground_endpoint_strategy="skip_ray",
        l_occ_global_map=0.10,
    )

    meta_rows = read_rows(lidar_proc_index_path(bag_id, chunk_id))
    origin = origin_from_index(meta_rows)

    # (a) Baseline: no prior.
    *_, lo_arrays_no_prior = build_log_odds_grid(
        meta_rows, cfg, origin, chunk_id, cache_xyz=False, bag_id=bag_id,
    )
    keys_no, lo_no, n_obs_no, n_hits_no, *_ = lo_arrays_no_prior

    # Find the voxel index for our static point so we can assert on it.
    pt_key = int(voxel_indices(static_pt, origin, cfg.voxel_size_m)[0])
    idx_no = np.searchsorted(keys_no, pt_key)
    assert idx_no < len(keys_no) and keys_no[idx_no] == pt_key
    baseline_lo = float(lo_no[idx_no])
    baseline_n_hits = int(n_hits_no[idx_no])

    # (b) With a prior: place the global map exactly at the static point's
    # world location.  Every sweep's endpoint will fall within match_radius_m.
    prior = GlobalMapPrior(static_pt, match_radius_m=0.30, r_max_credibility_m=200.0)
    *_, lo_arrays_prior = build_log_odds_grid(
        meta_rows, cfg, origin, chunk_id,
        cache_xyz=False, bag_id=bag_id, global_map_prior=prior,
    )
    keys_p, lo_p, n_obs_p, n_hits_p, *_ = lo_arrays_prior
    idx_p = np.searchsorted(keys_p, pt_key)
    assert idx_p < len(keys_p) and keys_p[idx_p] == pt_key

    # Prior must strictly increase log_odds on the map-matched voxel.
    assert float(lo_p[idx_p]) > baseline_lo, (
        f"prior failed to boost log_odds: baseline={baseline_lo}, with_prior={float(lo_p[idx_p])}"
    )

    # n_hits must be unchanged — prior is NOT allowed to add synthetic hits.
    assert int(n_hits_p[idx_p]) == baseline_n_hits, (
        f"prior leaked into n_hits: baseline={baseline_n_hits}, with_prior={int(n_hits_p[idx_p])}"
    )

    # n_obs must also be unchanged — prior is NOT allowed to add synthetic observations.
    assert int(n_obs_p[idx_p]) == int(n_obs_no[idx_no]), (
        "prior leaked into n_obs"
    )

    # Expected boost magnitude: 3 sweeps × l_occ_global_map × r_star.
    # The single static point at (1,0,0) seen from origin (0,0,0) → range 1m,
    # r_star = min(1, 200/1) = 1.0.  So 3 × 0.10 × 1.0 = 0.30 added.
    np.testing.assert_allclose(
        float(lo_p[idx_p]) - baseline_lo, 3 * 0.10, atol=1e-5
    )


def _write_world_sweep_with_ground(
    bag_id: str,
    chunk_id: str,
    sweep_id: int,
    xyz: np.ndarray,
    ground_mask: np.ndarray,
    *,
    origin: np.ndarray,
) -> None:
    path = local_path(lidar_world_path(bag_id, chunk_id, sweep_id))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        origin=np.asarray(origin, dtype=np.float64),
        ground_mask=ground_mask.astype(bool),
    )


def test_iwu_boost_excludes_ground_points(tmp_env):
    """Ground points are filtered from the IWU boost query.

    Regression: the boost previously used the full sweep xyz, so a ground
    voxel matching the global static map got a phantom log_odds entry with
    no n_obs / n_hits backing.  Harmless for correctness (free_only →
    not_dynamic_arr) but it inflated unique_keys.

    Geometry: non-ground endpoint sits on the +x axis at (10, 0, 0); ground
    endpoint is at (0, 5, 0).  In skip_ray mode the ground ray is never
    traversed, and the non-ground ray (sensor at origin) goes along +x and
    doesn't touch the ground voxel at vy≈33.  So ground voxel only enters
    the dict if the boost is querying it — which is exactly the bug.
    """
    bag_id, chunk_id = "bag_iwu_ground", "chunk0"
    chunk_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    xyz = np.array([[10.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    ground = np.array([False, True])

    for sid in range(2):
        _write_world_sweep_with_ground(
            bag_id, chunk_id, sid, xyz, ground, origin=chunk_origin
        )
    rows = [_proc_row(bag_id, chunk_id, sid, xyz) for sid in range(2)]
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))

    cfg = ComponentConfig(
        classification_method="log_odds",
        voxel_size_m=0.15,
        ground_endpoint_strategy="skip_ray",
        l_occ_global_map=0.10,
    )
    meta_rows = read_rows(lidar_proc_index_path(bag_id, chunk_id))
    origin = origin_from_index(meta_rows)
    prior = GlobalMapPrior(xyz, match_radius_m=0.30, r_max_credibility_m=200.0)

    *_, lo_arrays = build_log_odds_grid(
        meta_rows, cfg, origin, chunk_id,
        cache_xyz=False, bag_id=bag_id, global_map_prior=prior,
    )
    keys, lo_vals, _n_obs, n_hits, *_ = lo_arrays

    non_ground_key = int(voxel_indices(xyz[0:1], origin, cfg.voxel_size_m)[0])
    ground_key = int(voxel_indices(xyz[1:2], origin, cfg.voxel_size_m)[0])

    # Non-ground voxel must be present with both traversal hits and boost.
    ng_idx = np.searchsorted(keys, non_ground_key)
    assert ng_idx < len(keys) and keys[ng_idx] == non_ground_key, (
        "non-ground voxel must appear in unique_keys"
    )
    assert int(n_hits[ng_idx]) == 2, "non-ground voxel got both sweep hits"
    assert float(lo_vals[ng_idx]) > 0, "non-ground voxel should have positive log_odds"

    # Ground voxel must NOT be in unique_keys: skip_ray skips its ray, the
    # non-ground ray doesn't traverse it, and the boost no longer queries
    # it.  Pre-fix, the boost created a phantom entry.
    gnd_idx = np.searchsorted(keys, ground_key)
    found_ground = gnd_idx < len(keys) and keys[gnd_idx] == ground_key
    assert not found_ground, (
        f"ground voxel (key={ground_key}) leaked into unique_keys via IWU "
        f"boost — Bug 4 regression"
    )
