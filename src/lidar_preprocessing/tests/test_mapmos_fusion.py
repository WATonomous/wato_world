"""Tests for MapMOS fusion into the log-odds classifier.

The regression contract (plan §1 / non-negotiable #1): all-zero logits +
ANY alpha must produce static/dynamic counts identical to the no-MapMOS
run. The neutral-prior test is the merge gate for Step 1.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from wato_common.artifact_store import (
    lidar_proc_index_path,
    lidar_world_path,
    local_path,
    mapmos_logit_path,
)
from wato_common.io.parquet_io import write_table
from wato_common.schemas import PROCESSED_SWEEPS_SCHEMA
from wato_lidar_preprocessing.classify import process_chunk
from wato_lidar_preprocessing.config import ComponentConfig, MapMOSFusionParams, MapMOSParams
from wato_lidar_preprocessing.ray_traversal import _NUMBA_AVAILABLE

# Log-odds + kernel-level fusion all require Numba. importorskip is not
# enough here because the env's numba imports can raise non-ImportError
# (e.g. coverage version mismatch); use the dispatch-layer probe instead.
requires_numba = pytest.mark.skipif(
    not _NUMBA_AVAILABLE, reason="numba unavailable"
)

# Apply to every test in the module.
pytestmark = requires_numba


@pytest.fixture()
def tmp_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# Synthetic chunk helpers
# ---------------------------------------------------------------------------


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
        "n_points_ground": 0,
        "world_path": lidar_world_path(bag_id, chunk_id, sweep_id),
        "dynamic_mask_path": "",
        "mapmos_logit_path": None,
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
    }


def _build_synthetic_chunk(
    bag_id: str,
    chunk_id: str,
    n_sweeps: int = 5,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Two static points + one extra dynamic point present in sweep 2 only.

    The static cluster persists in every sweep; the dynamic point only
    shows up once. Used to exercise both the static and dynamic decisions
    under the log-odds path with sufficient evidence.
    """
    static_xyz = np.array(
        [[5.0, 0.0, 0.0], [5.0, 0.5, 0.0]], dtype=np.float64
    )
    dynamic_xyz = np.array([[0.0, 5.0, 0.0]], dtype=np.float64)

    xyz_per_sweep: list[np.ndarray] = []
    origins: list[np.ndarray] = []
    for i in range(n_sweeps):
        if i == 2:
            xyz = np.concatenate([static_xyz, dynamic_xyz])
        else:
            xyz = static_xyz
        # Sensor at origin so the DDA carves a straight, non-degenerate path.
        origins.append(np.array([0.0, 0.0, 0.0], dtype=np.float64))
        _write_world_sweep(
            bag_id,
            chunk_id,
            i,
            xyz,
            origin=origins[i],
            ground_mask=np.zeros(xyz.shape[0], dtype=np.bool_),
        )
        xyz_per_sweep.append(xyz)

    rows = [_proc_row(bag_id, chunk_id, i, xyz_per_sweep[i]) for i in range(n_sweeps)]
    write_table(rows, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id))
    return xyz_per_sweep, origins


def _write_sidecars(bag_id: str, chunk_id: str, xyz_per_sweep, fill: float | None):
    """Write per-sweep MapMOS sidecars. fill=None -> no sidecars."""
    if fill is None:
        return
    for i, xyz in enumerate(xyz_per_sweep):
        logits = np.full(xyz.shape[0], fill, dtype=np.float32)
        np.save(local_path(mapmos_logit_path(bag_id, chunk_id, i)), logits)


def _cfg(enabled: bool, alpha: float = 0.0, **fusion_kwargs) -> ComponentConfig:
    return ComponentConfig(
        voxel_size_m=0.15,
        classification_method="log_odds",
        ground_endpoint_strategy="skip_endpoint",
        mapmos=MapMOSParams(
            enabled=enabled,
            fusion=MapMOSFusionParams(alpha=alpha, **fusion_kwargs),
        ),
    )


# ---------------------------------------------------------------------------
# Regression contract — neutral prior must be bit-identical
# ---------------------------------------------------------------------------


def test_neutral_prior_no_change_alpha_zero(tmp_env):
    """Baseline (mapmos disabled) vs (enabled, all-zero logits, alpha=0)."""
    bag_id = "bagN0"
    xyz_per_sweep, _ = _build_synthetic_chunk(bag_id, "c0")
    baseline = process_chunk(_cfg(enabled=False), bag_id, "c0")

    _build_synthetic_chunk(bag_id, "c1")
    _write_sidecars(bag_id, "c1", xyz_per_sweep, fill=0.0)
    candidate = process_chunk(_cfg(enabled=True, alpha=0.0), bag_id, "c1")

    assert candidate.n_static == baseline.n_static
    assert candidate.n_dynamic == baseline.n_dynamic


def test_neutral_prior_no_change_alpha_one(tmp_env):
    """Same regression contract at alpha=1.0 (multiplied by 0 == 0).

    This is the canonical additive-prior invariant: when priors are zero,
    the kernel update collapses to `+= l_occ` regardless of alpha. If
    this fails, priors are leaking non-zero into the kernel somewhere.
    """
    bag_id = "bagN1"
    xyz_per_sweep, _ = _build_synthetic_chunk(bag_id, "c0")
    baseline = process_chunk(_cfg(enabled=False), bag_id, "c0")

    _build_synthetic_chunk(bag_id, "c1")
    _write_sidecars(bag_id, "c1", xyz_per_sweep, fill=0.0)
    candidate = process_chunk(_cfg(enabled=True, alpha=1.0), bag_id, "c1")

    assert candidate.n_static == baseline.n_static
    assert candidate.n_dynamic == baseline.n_dynamic


def test_missing_logit_file_falls_back(tmp_env):
    """No sidecars on disk -> classify path identical to disabled mode."""
    bag_id = "bagMissing"
    _build_synthetic_chunk(bag_id, "c0")
    baseline = process_chunk(_cfg(enabled=False), bag_id, "c0")

    _build_synthetic_chunk(bag_id, "c1")
    # Deliberately no sidecars.
    candidate = process_chunk(_cfg(enabled=True, alpha=1.0), bag_id, "c1")

    assert candidate.n_static == baseline.n_static
    assert candidate.n_dynamic == baseline.n_dynamic


def test_logit_length_mismatch_falls_back(tmp_env, caplog):
    """Wrong-length sidecar -> warning + fall through to geometry-only."""
    bag_id = "bagLen"
    xyz_per_sweep, _ = _build_synthetic_chunk(bag_id, "c0")
    baseline = process_chunk(_cfg(enabled=False), bag_id, "c0")

    _build_synthetic_chunk(bag_id, "c1")
    # Write a sidecar with the WRONG length on sweep 0 only.
    for i, xyz in enumerate(xyz_per_sweep):
        if i == 0:
            logits = np.zeros(xyz.shape[0] + 7, dtype=np.float32)
        else:
            logits = np.zeros(xyz.shape[0], dtype=np.float32)
        np.save(local_path(mapmos_logit_path(bag_id, "c1", i)), logits)
    candidate = process_chunk(_cfg(enabled=True, alpha=1.0), bag_id, "c1")

    # All-zero on the well-formed sweeps -> total result still matches baseline.
    assert candidate.n_static == baseline.n_static
    assert candidate.n_dynamic == baseline.n_dynamic


# ---------------------------------------------------------------------------
# Direct kernel-level fusion behavior
# ---------------------------------------------------------------------------


def test_max_logit_stores_raw_not_scaled():
    """max_logit accumulator stores RAW prior[i], independent of alpha.

    Plan non-negotiable #4 / #18.
    """
    from wato_lidar_preprocessing.ray_traversal import (
        extract_log_odds_arrays,
        extract_max_logit_array,
        make_log_odds_dicts,
        make_max_logit_dict,
        update_sweep_log_odds,
    )

    lo, no, nh = make_log_odds_dicts()
    ml = make_max_logit_dict()

    sweep_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    endpoints = np.array([[5.0, 0.0, 0.0]], dtype=np.float64)
    priors = np.array([1.0], dtype=np.float32)
    chunk_origin = np.array([-1.0, -1.0, -1.0], dtype=np.float64)

    # Run with alpha=2.0; the RAW logit stored in max_logit must be 1.0.
    update_sweep_log_odds(
        sweep_origin,
        endpoints,
        None,
        chunk_origin,
        voxel_size=0.15,
        margin_voxels=1.0,
        max_length_m=80.0,
        log_odds=lo,
        n_obs=no,
        n_hits=nh,
        l_occ=0.85,
        l_free=0.40,
        log_odds_clamp=5.0,
        endpoint_priors=priors,
        alpha=2.0,
        max_logit=ml,
    )

    unique_keys, _, _, _ = extract_log_odds_arrays(lo, no, nh)
    max_vals = extract_max_logit_array(unique_keys, ml)
    # The endpoint voxel must contain the RAW logit 1.0 (not 2.0).
    assert np.any(np.isclose(max_vals, 1.0, atol=1e-5)), (
        f"expected max_logit==1.0 at the endpoint voxel, "
        f"saw {sorted(set(max_vals.tolist()))}"
    )
    # And no entry was 2.0 (which would be alpha*prior).
    assert not np.any(np.isclose(max_vals, 2.0, atol=1e-5)), (
        "max_logit was alpha-scaled — plan non-negotiable #4 broken"
    )


def test_max_logit_default_is_neg_inf_not_zero():
    """A voxel whose only prior is -0.5 must store -0.5, NOT 0.0.

    Default-init bug: if max_logit defaulted to 0.0 instead of -inf, the
    `max(prev, prior)` update with a negative prior would keep 0.0 and
    silently corrupt the rescue threshold comparison.
    """
    from wato_lidar_preprocessing.ray_traversal import (
        extract_log_odds_arrays,
        extract_max_logit_array,
        make_log_odds_dicts,
        make_max_logit_dict,
        update_sweep_log_odds,
    )

    lo, no, nh = make_log_odds_dicts()
    ml = make_max_logit_dict()

    sweep_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    endpoints = np.array([[5.0, 0.0, 0.0]], dtype=np.float64)
    priors = np.array([-0.5], dtype=np.float32)
    chunk_origin = np.array([-1.0, -1.0, -1.0], dtype=np.float64)

    update_sweep_log_odds(
        sweep_origin,
        endpoints,
        None,
        chunk_origin,
        voxel_size=0.15,
        margin_voxels=1.0,
        max_length_m=80.0,
        log_odds=lo,
        n_obs=no,
        n_hits=nh,
        l_occ=0.85,
        l_free=0.40,
        log_odds_clamp=5.0,
        endpoint_priors=priors,
        alpha=1.0,
        max_logit=ml,
    )

    unique_keys, _, _, _ = extract_log_odds_arrays(lo, no, nh)
    max_vals = extract_max_logit_array(unique_keys, ml)
    # The endpoint voxel must contain -0.5 (or close), NOT 0.0.
    assert np.any(np.isclose(max_vals, -0.5, atol=1e-5)), (
        f"expected -0.5 in max_logit, saw {sorted(set(max_vals.tolist()))}"
    )
    assert not np.any(np.isclose(max_vals, 0.0, atol=1e-5)), (
        "max_logit defaulted to 0.0 — plan non-negotiable #18 broken"
    )


def test_max_logit_array_aligned_with_unique_keys():
    """Per-voxel max logit array must remain aligned through sort permutation."""
    from wato_lidar_preprocessing.ray_traversal import (
        extract_log_odds_arrays,
        extract_max_logit_array,
        make_log_odds_dicts,
        make_max_logit_dict,
        update_sweep_log_odds,
    )

    lo, no, nh = make_log_odds_dicts()
    ml = make_max_logit_dict()

    sweep_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    # Three endpoint voxels with distinct priors.
    endpoints = np.array(
        [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]], dtype=np.float64
    )
    priors = np.array([3.0, -1.0, 1.5], dtype=np.float32)
    chunk_origin = np.array([-1.0, -1.0, -1.0], dtype=np.float64)

    update_sweep_log_odds(
        sweep_origin,
        endpoints,
        None,
        chunk_origin,
        voxel_size=0.15,
        margin_voxels=1.0,
        max_length_m=80.0,
        log_odds=lo,
        n_obs=no,
        n_hits=nh,
        l_occ=0.85,
        l_free=0.40,
        log_odds_clamp=5.0,
        endpoint_priors=priors,
        alpha=1.0,
        max_logit=ml,
    )

    unique_keys, lo_vals, _, n_hits_vals = extract_log_odds_arrays(lo, no, nh)
    max_vals = extract_max_logit_array(unique_keys, ml)

    # Alignment contract: each unique_key's max_vals entry sits at the
    # same index as its lo_vals entry. Filtering with one mask must
    # filter the other consistently.
    assert unique_keys.shape == max_vals.shape
    hit_mask = n_hits_vals > 0
    filtered_keys = unique_keys[hit_mask]
    filtered_max = max_vals[hit_mask]
    assert filtered_keys.shape == filtered_max.shape
    # Each filtered max must be one of the three raw priors.
    for v in filtered_max:
        assert any(np.isclose(v, p, atol=1e-5) for p in priors.tolist())


def test_under_evidenced_rescue_independent_of_alpha(tmp_env):
    """Rescue threshold is on RAW logits — same fires under alpha=0.5 vs 2.0.

    We use a minimal hand-rigged log-odds grid and call classify_from_log_odds
    directly so we can compare the diagnostic dicts.
    """
    from wato_lidar_preprocessing.classify.log_odds import classify_from_log_odds

    unique_keys = np.array([0, 1, 2], dtype=np.int64)
    # Voxel 0: confident static. Voxel 1: under-evidenced with hits + strong
    # moving signal (raw 2.5 > threshold 2.0). Voxel 2: same evidence but
    # weak signal (raw 1.0 < threshold) -> should NOT be rescued.
    lo_vals = np.array([3.0, 0.5, 0.5], dtype=np.float32)
    n_obs_vals = np.array([10, 1, 1], dtype=np.int32)
    n_hits_vals = np.array([10, 1, 1], dtype=np.int32)
    max_logit_vals = np.array([0.0, 2.5, 1.0], dtype=np.float32)

    diag_a = None
    diag_b = None
    for alpha in (0.5, 2.0):
        cfg = _cfg(enabled=True, alpha=alpha)
        static_arr, not_dynamic, diag = classify_from_log_odds(
            unique_keys,
            lo_vals,
            n_obs_vals,
            n_hits_vals,
            cfg,
            max_logit_vals=max_logit_vals,
        )
        if diag_a is None:
            diag_a = diag
            static_a, not_dyn_a = static_arr, not_dynamic
        else:
            diag_b = diag
            static_b, not_dyn_b = static_arr, not_dynamic

    assert diag_a["n_under_evidenced_rescued"] == 1
    assert diag_b["n_under_evidenced_rescued"] == 1
    np.testing.assert_array_equal(static_a, static_b)
    np.testing.assert_array_equal(not_dyn_a, not_dyn_b)
