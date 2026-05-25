"""Unit tests for the Amanatides-Woo DDA kernels.

Four cases:
  1. Same-voxel ray (origin and endpoint inside one voxel).
  2. Numba/Python parity (bit-identical f32 across both implementations).
  3. Missing-numba dispatch hard-fail (monkeypatched).
  4. Missing-numba real-import hard-fail (subprocess with sys.modules['numba']=None).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from wato_lidar_preprocessing.ray_traversal import (
    _NUMBA_AVAILABLE,
    extract_log_odds_arrays,
    make_log_odds_dicts,
    update_sweep_log_odds,
)
from wato_lidar_preprocessing.ray_traversal._keys import SHIFT_X, SHIFT_Y

requires_numba = pytest.mark.skipif(
    not _NUMBA_AVAILABLE, reason="numba unavailable"
)


def _pack(vx: int, vy: int, vz: int) -> int:
    return (vx << SHIFT_X) | (vy << SHIFT_Y) | vz


@requires_numba
def test_same_voxel_ray():
    """Origin and endpoint inside the same voxel: one endpoint hit, no carving."""
    log_odds, n_obs, n_hits = make_log_odds_dicts()
    sweep_origin = np.array([0.05, 0.05, 0.05])
    endpoints = np.array([[0.10, 0.05, 0.05]])  # both inside voxel (0,0,0)
    chunk_origin = np.zeros(3)

    update_sweep_log_odds(
        sweep_origin,
        endpoints,
        None,
        chunk_origin,
        voxel_size=0.15,
        margin_voxels=1.0,
        max_length_m=80.0,
        log_odds=log_odds,
        n_obs=n_obs,
        n_hits=n_hits,
        l_occ=0.85,
        l_free=0.40,
        log_odds_clamp=5.0,
    )

    keys, lo_vals, n_obs_vals, n_hits_vals = extract_log_odds_arrays(
        log_odds, n_obs, n_hits
    )
    assert keys.size == 1, f"expected 1 voxel touched, got {keys.size}"
    expected_key = _pack(0, 0, 0)
    assert int(keys[0]) == expected_key
    assert int(n_hits_vals[0]) == 1
    assert int(n_obs_vals[0]) == 1
    np.testing.assert_array_equal(lo_vals, np.array([0.85], dtype=np.float32))


@requires_numba
def test_numba_python_parity():
    """Numba kernel must produce bit-identical f32 to the Python reference.

    Random rays spanning all 8 octants so every sign combination of
    (dxn, dyn, dzn) is exercised. We use np.array_equal (not allclose) for
    the float32 array because identical arithmetic on identical inputs has
    no parallelism — any drift would be a real divergence.
    """
    from wato_lidar_preprocessing.ray_traversal._python_kernel import (
        _update_sweep_python,
    )

    rng = np.random.default_rng(seed=42)
    n_rays = 50
    # Mix positive and negative components to cover all 8 octants.
    endpoints = rng.uniform(-3.0, 3.0, size=(n_rays, 3))
    is_ground = rng.choice([True, False], size=n_rays)
    sweep_origin = np.array([0.001, -0.002, 0.003])  # not on a voxel boundary
    chunk_origin = np.array([-5.0, -5.0, -5.0])
    voxel_size = 0.15
    margin_voxels = 1.0
    max_length_m = 80.0
    l_occ, l_free, log_odds_clamp = 0.85, 0.40, 5.0

    # Numba path
    nlo, nno, nnh = make_log_odds_dicts()
    update_sweep_log_odds(
        sweep_origin,
        endpoints.astype(np.float64),
        is_ground,
        chunk_origin,
        voxel_size,
        margin_voxels,
        max_length_m,
        nlo, nno, nnh,
        l_occ, l_free, log_odds_clamp,
    )
    n_keys, n_lo, n_no, n_nh = extract_log_odds_arrays(nlo, nno, nnh)

    # Python reference path
    plo: dict[int, np.float32] = {}
    pno: dict[int, np.int32] = {}
    pnh: dict[int, np.int32] = {}
    _update_sweep_python(
        sweep_origin,
        endpoints.astype(np.float64),
        is_ground,
        chunk_origin,
        voxel_size,
        margin_voxels,
        max_length_m,
        plo, pno, pnh,
        l_occ, l_free, log_odds_clamp,
    )
    p_keys = np.array(sorted(plo.keys()), dtype=np.int64)
    p_lo = np.array([plo[k] for k in p_keys], dtype=np.float32)
    p_no = np.array([int(pno[k]) for k in p_keys], dtype=np.int32)
    p_nh = np.array([int(pnh.get(k, 0)) for k in p_keys], dtype=np.int32)

    # Bit-identical comparisons. np.array_equal is the strict check.
    assert np.array_equal(n_keys, p_keys), "voxel-key sets differ between kernels"
    assert np.array_equal(n_no, p_no), "n_obs arrays differ"
    assert np.array_equal(n_nh, p_nh), "n_hits arrays differ"
    assert np.array_equal(n_lo, p_lo), (
        "log_odds arrays differ between Numba and Python kernels — "
        "the DDA implementations have diverged (likely a t_max init bug)"
    )


@requires_numba
def test_range_weight_disabled_endpoint_unweighted():
    """With use_range_weight=False, a 500m endpoint gets +l_occ unchanged.

    Locks in the reproducibility contract: when the flag is off, the kernel
    must produce bit-identical log_odds to the pre-feature behaviour
    regardless of endpoint distance.
    """
    log_odds, n_obs, n_hits = make_log_odds_dicts()
    sweep_origin = np.array([0.001, 0.0, 0.0])
    endpoints = np.array([[500.001, 0.0, 0.0]])  # ~500m endpoint
    chunk_origin = np.array([0.0, 0.0, 0.0])

    update_sweep_log_odds(
        sweep_origin,
        endpoints,
        None,
        chunk_origin,
        voxel_size=1.0,
        margin_voxels=1.0,
        max_length_m=1000.0,
        log_odds=log_odds,
        n_obs=n_obs,
        n_hits=n_hits,
        l_occ=0.85,
        l_free=0.40,
        log_odds_clamp=50.0,
        # use_range_weight defaults to False; r_max irrelevant
    )

    # Find the endpoint voxel's log_odds.
    keys, lo_vals, _, n_hits_vals = extract_log_odds_arrays(log_odds, n_obs, n_hits)
    endpoint_key = _pack(500, 0, 0)
    idx = np.searchsorted(keys, endpoint_key)
    assert keys[idx] == endpoint_key
    assert int(n_hits_vals[idx]) == 1
    # Endpoint must carry exactly +l_occ — no range scaling.
    np.testing.assert_allclose(float(lo_vals[idx]), 0.85, atol=1e-6)


@requires_numba
def test_range_weight_endpoint_scaled_by_r_max_over_length():
    """With use_range_weight=True, endpoint l_occ scales by min(1, r_max/length)."""
    log_odds, n_obs, n_hits = make_log_odds_dicts()
    sweep_origin = np.array([0.001, 0.0, 0.0])
    # 500m endpoint, r_max=200 → r*_j = 200/500 = 0.4
    endpoints = np.array([[500.001, 0.0, 0.0]])
    chunk_origin = np.array([0.0, 0.0, 0.0])

    update_sweep_log_odds(
        sweep_origin,
        endpoints,
        None,
        chunk_origin,
        voxel_size=1.0,
        margin_voxels=1.0,
        max_length_m=1000.0,
        log_odds=log_odds,
        n_obs=n_obs,
        n_hits=n_hits,
        l_occ=0.85,
        l_free=0.40,
        log_odds_clamp=50.0,
        r_max=200.0,
        use_range_weight=True,
    )

    keys, lo_vals, _, n_hits_vals = extract_log_odds_arrays(log_odds, n_obs, n_hits)
    endpoint_key = _pack(500, 0, 0)
    idx = np.searchsorted(keys, endpoint_key)
    assert keys[idx] == endpoint_key
    assert int(n_hits_vals[idx]) == 1
    # Expected: l_occ * (r_max / length) = 0.85 * (200 / 500.0) = 0.34
    # Length is sqrt(500^2 + 0 + 0) but with the 0.001 offset slightly larger.
    length = np.sqrt(500.0**2)  # endpoint - origin along x = 500 exactly
    expected = 0.85 * min(1.0, 200.0 / length)
    np.testing.assert_allclose(float(lo_vals[idx]), expected, rtol=1e-5)


@requires_numba
def test_range_weight_endpoint_inside_r_max_is_unweighted():
    """Endpoints at d <= r_max get full l_occ (the min() clamp pins r*_j to 1.0)."""
    log_odds, n_obs, n_hits = make_log_odds_dicts()
    sweep_origin = np.array([0.001, 0.0, 0.0])
    # 50m endpoint, r_max=200 → r*_j = min(1, 200/50) = 1.0
    endpoints = np.array([[50.001, 0.0, 0.0]])
    chunk_origin = np.array([0.0, 0.0, 0.0])

    update_sweep_log_odds(
        sweep_origin,
        endpoints,
        None,
        chunk_origin,
        voxel_size=1.0,
        margin_voxels=1.0,
        max_length_m=200.0,
        log_odds=log_odds,
        n_obs=n_obs,
        n_hits=n_hits,
        l_occ=0.85,
        l_free=0.40,
        log_odds_clamp=50.0,
        r_max=200.0,
        use_range_weight=True,
    )

    keys, lo_vals, _, n_hits_vals = extract_log_odds_arrays(log_odds, n_obs, n_hits)
    endpoint_key = _pack(50, 0, 0)
    idx = np.searchsorted(keys, endpoint_key)
    assert keys[idx] == endpoint_key
    assert int(n_hits_vals[idx]) == 1
    # 50 < 200 → clamp to 1.0 → endpoint carries full l_occ.
    np.testing.assert_allclose(float(lo_vals[idx]), 0.85, atol=1e-6)


@requires_numba
def test_range_weight_per_voxel_free_space_uses_t_entry():
    """Traversed voxels are weighted by their own near-edge distance, not the endpoint's.

    Long ray (20 voxels at voxel_size=1 → 20m endpoint), r_max=5. Near voxels
    (t_entry <= 5) keep full -l_free; far voxels (t_entry > 5) get
    -l_free * (5 / t_entry).

    This is the Option B contract: a 20m ray's near-field carving at voxel
    x=1 must be at full weight even though the endpoint at x=20 is itself
    down-weighted to r*_j = 5/20 = 0.25. Option A would (wrongly) scale all
    carving along the ray by the same 0.25.
    """
    log_odds, n_obs, n_hits = make_log_odds_dicts()
    sweep_origin = np.array([0.001, 0.0, 0.0])
    endpoints = np.array([[20.001, 0.0, 0.0]])
    chunk_origin = np.array([0.0, 0.0, 0.0])

    update_sweep_log_odds(
        sweep_origin,
        endpoints,
        None,
        chunk_origin,
        voxel_size=1.0,
        margin_voxels=1.0,
        max_length_m=50.0,
        log_odds=log_odds,
        n_obs=n_obs,
        n_hits=n_hits,
        l_occ=0.85,
        l_free=0.40,
        log_odds_clamp=50.0,
        r_max=5.0,
        use_range_weight=True,
    )

    keys, lo_vals, _, n_hits_vals = extract_log_odds_arrays(log_odds, n_obs, n_hits)

    # Voxel at cx=1: entered at t_entry ≈ 0.999 → r*_t = min(1, 5/0.999) = 1.0
    # → log_odds = -0.40 * 1.0 = -0.40
    idx_near = np.searchsorted(keys, _pack(1, 0, 0))
    assert keys[idx_near] == _pack(1, 0, 0)
    assert int(n_hits_vals[idx_near]) == 0, "free-space voxel must have n_hits=0"
    np.testing.assert_allclose(float(lo_vals[idx_near]), -0.40, atol=1e-5)

    # Voxel at cx=10: entered at t_entry ≈ 9.999 → r*_t ≈ 5/9.999 ≈ 0.5000500...
    # → log_odds ≈ -0.40 * 0.50005 ≈ -0.20002
    idx_mid = np.searchsorted(keys, _pack(10, 0, 0))
    assert keys[idx_mid] == _pack(10, 0, 0)
    assert int(n_hits_vals[idx_mid]) == 0
    expected_mid = -0.40 * (5.0 / 9.999)
    np.testing.assert_allclose(float(lo_vals[idx_mid]), expected_mid, rtol=1e-4)

    # Voxel at cx=15: entered at t_entry ≈ 14.999 → r*_t ≈ 5/14.999 ≈ 0.33336
    idx_far = np.searchsorted(keys, _pack(15, 0, 0))
    assert keys[idx_far] == _pack(15, 0, 0)
    expected_far = -0.40 * (5.0 / 14.999)
    np.testing.assert_allclose(float(lo_vals[idx_far]), expected_far, rtol=1e-4)

    # Sanity: near voxel must be more strongly carved than far voxel.
    # (|near| > |mid| > |far| in absolute terms; all negative.)
    assert float(lo_vals[idx_near]) < float(lo_vals[idx_mid]) < float(lo_vals[idx_far]) < 0.0


@requires_numba
def test_range_weight_numba_python_parity():
    """Parity test with use_range_weight=True on both kernels.

    Same setup as test_numba_python_parity but exercises the weighted code
    path. Range weighting changes the float arithmetic at every update site,
    so this guards against the weighting math diverging between the two
    implementations (e.g., if the f32 cast position drifts).
    """
    from wato_lidar_preprocessing.ray_traversal._python_kernel import (
        _update_sweep_python,
    )

    rng = np.random.default_rng(seed=43)
    n_rays = 50
    # Spread endpoints across distances so r*_j varies across rays
    # (some inside r_max, some outside).
    endpoints = rng.uniform(-150.0, 150.0, size=(n_rays, 3))
    is_ground = rng.choice([True, False], size=n_rays)
    sweep_origin = np.array([0.001, -0.002, 0.003])
    chunk_origin = np.array([-200.0, -200.0, -200.0])
    voxel_size = 0.5
    margin_voxels = 1.0
    max_length_m = 300.0
    l_occ, l_free, log_odds_clamp = 1.20, 0.25, 50.0
    r_max = 50.0  # small enough that many endpoints are down-weighted

    nlo, nno, nnh = make_log_odds_dicts()
    update_sweep_log_odds(
        sweep_origin,
        endpoints.astype(np.float64),
        is_ground,
        chunk_origin,
        voxel_size,
        margin_voxels,
        max_length_m,
        nlo, nno, nnh,
        l_occ, l_free, log_odds_clamp,
        r_max=r_max,
        use_range_weight=True,
    )
    n_keys, n_lo, n_no, n_nh = extract_log_odds_arrays(nlo, nno, nnh)

    plo: dict[int, np.float32] = {}
    pno: dict[int, np.int32] = {}
    pnh: dict[int, np.int32] = {}
    _update_sweep_python(
        sweep_origin,
        endpoints.astype(np.float64),
        is_ground,
        chunk_origin,
        voxel_size,
        margin_voxels,
        max_length_m,
        plo, pno, pnh,
        l_occ, l_free, log_odds_clamp,
        r_max=r_max,
        use_range_weight=True,
    )
    p_keys = np.array(sorted(plo.keys()), dtype=np.int64)
    p_lo = np.array([plo[k] for k in p_keys], dtype=np.float32)
    p_no = np.array([int(pno[k]) for k in p_keys], dtype=np.int32)
    p_nh = np.array([int(pnh.get(k, 0)) for k in p_keys], dtype=np.int32)

    assert np.array_equal(n_keys, p_keys)
    assert np.array_equal(n_no, p_no)
    assert np.array_equal(n_nh, p_nh)
    assert np.array_equal(n_lo, p_lo), (
        "weighted log_odds diverged between Numba and Python kernels — "
        "check that f32 cast position matches in both (l_occ * r_star, "
        "then cast to f32)"
    )


def test_range_weight_per_voxel_free_space_python_kernel():
    """Pure-Python equivalent of test_range_weight_per_voxel_free_space_uses_t_entry.

    Runs without numba so the per-voxel weighting math can be validated in
    any environment (the @requires_numba variant above exercises the same
    contract via the JIT kernel). Both kernels are required to compute the
    same numbers; if this passes but the JIT variant fails, the divergence
    is in the JIT kernel.
    """
    from wato_lidar_preprocessing.ray_traversal._python_kernel import (
        _update_sweep_python,
    )

    sweep_origin = np.array([0.001, 0.0, 0.0])
    endpoints = np.array([[20.001, 0.0, 0.0]])
    chunk_origin = np.array([0.0, 0.0, 0.0])

    log_odds: dict[int, np.float32] = {}
    n_obs: dict[int, np.int32] = {}
    n_hits: dict[int, np.int32] = {}
    _update_sweep_python(
        sweep_origin,
        endpoints,
        None,
        chunk_origin,
        voxel_size=1.0,
        margin_voxels=1.0,
        max_length_m=50.0,
        log_odds=log_odds,
        n_obs=n_obs,
        n_hits=n_hits,
        l_occ=0.85,
        l_free=0.40,
        log_odds_clamp=50.0,
        r_max=5.0,
        use_range_weight=True,
    )

    near_key = _pack(1, 0, 0)
    mid_key = _pack(10, 0, 0)
    far_key = _pack(15, 0, 0)

    assert near_key in log_odds, "near voxel must be carved"
    assert mid_key in log_odds, "mid voxel must be carved"
    assert far_key in log_odds, "far voxel must be carved"

    # Near voxel (t_entry ≈ 0.999): r*_t pinned to 1.0 → full -l_free.
    np.testing.assert_allclose(float(log_odds[near_key]), -0.40, atol=1e-5)
    # Mid voxel (t_entry ≈ 9.999): r*_t ≈ 5/9.999 → -l_free * 0.50005.
    np.testing.assert_allclose(
        float(log_odds[mid_key]), -0.40 * (5.0 / 9.999), rtol=1e-4
    )
    # Far voxel (t_entry ≈ 14.999): r*_t ≈ 5/14.999 → -l_free * 0.33336.
    np.testing.assert_allclose(
        float(log_odds[far_key]), -0.40 * (5.0 / 14.999), rtol=1e-4
    )

    # Monotonicity contract: near carved more heavily than far.
    assert log_odds[near_key] < log_odds[mid_key] < log_odds[far_key] < 0.0


def test_range_weight_endpoint_python_kernel():
    """Pure-Python endpoint scaling test (mirror of the numba-gated variants)."""
    from wato_lidar_preprocessing.ray_traversal._python_kernel import (
        _update_sweep_python,
    )

    sweep_origin = np.array([0.001, 0.0, 0.0])
    chunk_origin = np.array([0.0, 0.0, 0.0])

    # Case 1: 50m endpoint, r_max=200 → r*_j pinned to 1.0 → endpoint = +l_occ.
    log_odds: dict[int, np.float32] = {}
    n_obs: dict[int, np.int32] = {}
    n_hits: dict[int, np.int32] = {}
    _update_sweep_python(
        sweep_origin,
        np.array([[50.001, 0.0, 0.0]]),
        None,
        chunk_origin,
        1.0, 1.0, 200.0,
        log_odds, n_obs, n_hits,
        0.85, 0.40, 50.0,
        r_max=200.0,
        use_range_weight=True,
    )
    assert _pack(50, 0, 0) in log_odds
    np.testing.assert_allclose(
        float(log_odds[_pack(50, 0, 0)]), 0.85, atol=1e-6,
        err_msg="endpoint inside r_max must keep full l_occ"
    )

    # Case 2: 500m endpoint, r_max=200 → r*_j = 200/500 = 0.4 → endpoint = 0.34.
    log_odds = {}
    n_obs = {}
    n_hits = {}
    _update_sweep_python(
        sweep_origin,
        np.array([[500.001, 0.0, 0.0]]),
        None,
        chunk_origin,
        1.0, 1.0, 1000.0,
        log_odds, n_obs, n_hits,
        0.85, 0.40, 50.0,
        r_max=200.0,
        use_range_weight=True,
    )
    assert _pack(500, 0, 0) in log_odds
    np.testing.assert_allclose(
        float(log_odds[_pack(500, 0, 0)]),
        0.85 * (200.0 / 500.0),
        rtol=1e-5,
        err_msg="endpoint at 2.5*r_max must scale to r_max/length"
    )

    # Case 3: flag off → 500m endpoint still gets +l_occ unchanged.
    log_odds = {}
    n_obs = {}
    n_hits = {}
    _update_sweep_python(
        sweep_origin,
        np.array([[500.001, 0.0, 0.0]]),
        None,
        chunk_origin,
        1.0, 1.0, 1000.0,
        log_odds, n_obs, n_hits,
        0.85, 0.40, 50.0,
        # default r_max=200.0, use_range_weight=False
    )
    assert _pack(500, 0, 0) in log_odds
    np.testing.assert_allclose(
        float(log_odds[_pack(500, 0, 0)]), 0.85, atol=1e-6,
        err_msg="flag-off reproducibility: endpoint must carry full l_occ"
    )


def test_missing_numba_hard_fail_dispatch(monkeypatch):
    """When _NUMBA_AVAILABLE is False, make_log_odds_dicts must raise ImportError
    with the expected remediation message. Tests the guard logic directly.
    """
    import wato_lidar_preprocessing.ray_traversal.dispatch as dispatch_mod

    monkeypatch.setattr(dispatch_mod, "_NUMBA_AVAILABLE", False)
    monkeypatch.setattr(dispatch_mod, "_NUMBA_IMPORT_ERROR", None)

    with pytest.raises(ImportError) as exc_info:
        dispatch_mod.make_log_odds_dicts()
    msg = str(exc_info.value)
    assert "install" in msg.lower() and "numba" in msg.lower()
    assert "classification_method=persistence" in msg


def test_missing_numba_hard_fail_real_import(tmp_path):
    """Subprocess test: simulate a clean import with numba blocked.

    Hides numba via sys.modules before the first import of ray_traversal.
    This exercises the actual import-time path in dispatch.py — the
    monkeypatch test above only exercises the runtime _require_numba()
    guard.

    Skipped if subprocess invocation fails for environmental reasons
    (e.g. sandboxed CI without subprocess permissions).
    """
    script = textwrap.dedent(
        """
        import sys
        # Block numba BEFORE the first import attempt so the try/except in
        # dispatch.py falls into the _NUMBA_AVAILABLE = False branch.
        sys.modules['numba'] = None
        sys.modules['numba.types'] = None
        sys.modules['numba.typed'] = None

        from wato_lidar_preprocessing.ray_traversal import (
            make_log_odds_dicts, _NUMBA_AVAILABLE,
        )
        assert _NUMBA_AVAILABLE is False, "numba should be hidden in subprocess"

        try:
            make_log_odds_dicts()
        except ImportError as e:
            msg = str(e)
            assert "install" in msg.lower() and "numba" in msg.lower(), msg
            assert "classification_method=persistence" in msg, msg
            print("OK")
            sys.exit(0)
        sys.exit(2)
        """
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            env={
                "PYTHONPATH": "src/common/src:src/lidar_preprocessing/src",
                "PATH": "/usr/bin:/bin",
            },
            cwd=".",
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as e:
        pytest.skip(f"subprocess unsupported: {e}")

    assert result.returncode == 0, (
        f"subprocess failed (rc={result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "OK" in result.stdout
