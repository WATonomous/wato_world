"""Unit tests for the Amanatides-Woo DDA kernels.

The kernel takes the carve margin directly in metres (margin_m) and a
beam-footprint credibility crossover d_star (range weighting is always on,
scaling every update by min(1, d_star / d)). Passing a very large d_star
recovers unweighted behaviour for the geometry-only tests.
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

requires_numba = pytest.mark.skipif(not _NUMBA_AVAILABLE, reason="numba unavailable")

# Large enough that min(1, d_star/d) == 1 for every endpoint in these tests,
# i.e. range weighting is a no-op.
_NO_WEIGHT = 1e12


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
        margin_m=0.15,
        max_length_m=80.0,
        log_odds=log_odds,
        n_obs=n_obs,
        n_hits=n_hits,
        l_occ=0.85,
        l_free=0.40,
        log_odds_clamp=5.0,
        d_star=_NO_WEIGHT,
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
    (dxn, dyn, dzn) is exercised.
    """
    from wato_lidar_preprocessing.ray_traversal._python_kernel import (
        _update_sweep_python,
    )

    rng = np.random.default_rng(seed=42)
    n_rays = 50
    endpoints = rng.uniform(-3.0, 3.0, size=(n_rays, 3))
    is_ground = rng.choice([True, False], size=n_rays)
    sweep_origin = np.array([0.001, -0.002, 0.003])  # not on a voxel boundary
    chunk_origin = np.array([-5.0, -5.0, -5.0])
    voxel_size = 0.15
    margin_m = 0.15
    max_length_m = 80.0
    l_occ, l_free, log_odds_clamp = 0.85, 0.40, 5.0
    d_star = _NO_WEIGHT

    # Numba path
    nlo, nno, nnh = make_log_odds_dicts()
    update_sweep_log_odds(
        sweep_origin,
        endpoints.astype(np.float64),
        is_ground,
        chunk_origin,
        voxel_size,
        margin_m,
        max_length_m,
        nlo,
        nno,
        nnh,
        l_occ,
        l_free,
        log_odds_clamp,
        d_star,
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
        margin_m,
        max_length_m,
        plo,
        pno,
        pnh,
        l_occ,
        l_free,
        log_odds_clamp,
        d_star,
    )
    p_keys = np.array(sorted(plo.keys()), dtype=np.int64)
    p_lo = np.array([plo[k] for k in p_keys], dtype=np.float32)
    p_no = np.array([int(pno[k]) for k in p_keys], dtype=np.int32)
    p_nh = np.array([int(pnh.get(k, 0)) for k in p_keys], dtype=np.int32)

    assert np.array_equal(n_keys, p_keys), "voxel-key sets differ between kernels"
    assert np.array_equal(n_no, p_no), "n_obs arrays differ"
    assert np.array_equal(n_nh, p_nh), "n_hits arrays differ"
    assert np.array_equal(n_lo, p_lo), (
        "log_odds arrays differ between Numba and Python kernels — "
        "the DDA implementations have diverged (likely a t_max init bug)"
    )


@requires_numba
def test_range_weight_large_d_star_endpoint_unweighted():
    """With d_star >> length, a 500m endpoint gets +l_occ unchanged."""
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
        margin_m=1.0,
        max_length_m=1000.0,
        log_odds=log_odds,
        n_obs=n_obs,
        n_hits=n_hits,
        l_occ=0.85,
        l_free=0.40,
        log_odds_clamp=50.0,
        d_star=_NO_WEIGHT,
    )

    keys, lo_vals, _, n_hits_vals = extract_log_odds_arrays(log_odds, n_obs, n_hits)
    endpoint_key = _pack(500, 0, 0)
    idx = np.searchsorted(keys, endpoint_key)
    assert keys[idx] == endpoint_key
    assert int(n_hits_vals[idx]) == 1
    np.testing.assert_allclose(float(lo_vals[idx]), 0.85, atol=1e-6)


@requires_numba
def test_range_weight_endpoint_scaled_by_d_star_over_length():
    """Endpoint l_occ scales by min(1, d_star/length) for length > d_star."""
    log_odds, n_obs, n_hits = make_log_odds_dicts()
    sweep_origin = np.array([0.001, 0.0, 0.0])
    # 500m endpoint, d_star=200 → r* = 200/500 = 0.4
    endpoints = np.array([[500.001, 0.0, 0.0]])
    chunk_origin = np.array([0.0, 0.0, 0.0])

    update_sweep_log_odds(
        sweep_origin,
        endpoints,
        None,
        chunk_origin,
        voxel_size=1.0,
        margin_m=1.0,
        max_length_m=1000.0,
        log_odds=log_odds,
        n_obs=n_obs,
        n_hits=n_hits,
        l_occ=0.85,
        l_free=0.40,
        log_odds_clamp=50.0,
        d_star=200.0,
    )

    keys, lo_vals, _, n_hits_vals = extract_log_odds_arrays(log_odds, n_obs, n_hits)
    endpoint_key = _pack(500, 0, 0)
    idx = np.searchsorted(keys, endpoint_key)
    assert keys[idx] == endpoint_key
    assert int(n_hits_vals[idx]) == 1
    length = 500.0  # endpoint - origin along x ≈ 500
    expected = 0.85 * min(1.0, 200.0 / length)
    np.testing.assert_allclose(float(lo_vals[idx]), expected, rtol=1e-5)


@requires_numba
def test_range_weight_endpoint_inside_d_star_is_unweighted():
    """Endpoints at d <= d_star get full l_occ (the min() clamp pins r* to 1.0)."""
    log_odds, n_obs, n_hits = make_log_odds_dicts()
    sweep_origin = np.array([0.001, 0.0, 0.0])
    # 50m endpoint, d_star=200 → r* = min(1, 200/50) = 1.0
    endpoints = np.array([[50.001, 0.0, 0.0]])
    chunk_origin = np.array([0.0, 0.0, 0.0])

    update_sweep_log_odds(
        sweep_origin,
        endpoints,
        None,
        chunk_origin,
        voxel_size=1.0,
        margin_m=1.0,
        max_length_m=200.0,
        log_odds=log_odds,
        n_obs=n_obs,
        n_hits=n_hits,
        l_occ=0.85,
        l_free=0.40,
        log_odds_clamp=50.0,
        d_star=200.0,
    )

    keys, lo_vals, _, n_hits_vals = extract_log_odds_arrays(log_odds, n_obs, n_hits)
    endpoint_key = _pack(50, 0, 0)
    idx = np.searchsorted(keys, endpoint_key)
    assert keys[idx] == endpoint_key
    assert int(n_hits_vals[idx]) == 1
    np.testing.assert_allclose(float(lo_vals[idx]), 0.85, atol=1e-6)


@requires_numba
def test_range_weight_per_voxel_free_space_uses_t_entry():
    """Traversed voxels are weighted by their own near-edge distance, not the endpoint's.

    Long ray (20 voxels at voxel_size=1 → 20m endpoint), d_star=5. Near voxels
    (t_entry <= 5) keep full -l_free; far voxels (t_entry > 5) get
    -l_free * (5 / t_entry). A 20m ray's near-field carving at voxel x=1 stays
    at full weight even though the endpoint at x=20 is down-weighted to 0.25.
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
        margin_m=1.0,
        max_length_m=50.0,
        log_odds=log_odds,
        n_obs=n_obs,
        n_hits=n_hits,
        l_occ=0.85,
        l_free=0.40,
        log_odds_clamp=50.0,
        d_star=5.0,
    )

    keys, lo_vals, _, n_hits_vals = extract_log_odds_arrays(log_odds, n_obs, n_hits)

    # Voxel at cx=1: entered at t_entry ≈ 0.999 → r*_t = min(1, 5/0.999) = 1.0
    idx_near = np.searchsorted(keys, _pack(1, 0, 0))
    assert keys[idx_near] == _pack(1, 0, 0)
    assert int(n_hits_vals[idx_near]) == 0, "free-space voxel must have n_hits=0"
    np.testing.assert_allclose(float(lo_vals[idx_near]), -0.40, atol=1e-5)

    # Voxel at cx=10: t_entry ≈ 9.999 → r*_t ≈ 5/9.999
    idx_mid = np.searchsorted(keys, _pack(10, 0, 0))
    assert keys[idx_mid] == _pack(10, 0, 0)
    assert int(n_hits_vals[idx_mid]) == 0
    expected_mid = -0.40 * (5.0 / 9.999)
    np.testing.assert_allclose(float(lo_vals[idx_mid]), expected_mid, rtol=1e-4)

    # Voxel at cx=15: t_entry ≈ 14.999 → r*_t ≈ 5/14.999
    idx_far = np.searchsorted(keys, _pack(15, 0, 0))
    assert keys[idx_far] == _pack(15, 0, 0)
    expected_far = -0.40 * (5.0 / 14.999)
    np.testing.assert_allclose(float(lo_vals[idx_far]), expected_far, rtol=1e-4)

    # Near voxel must be more strongly carved than far voxel.
    assert (
        float(lo_vals[idx_near])
        < float(lo_vals[idx_mid])
        < float(lo_vals[idx_far])
        < 0.0
    )


@requires_numba
def test_range_weight_numba_python_parity():
    """Parity test with range weighting active (small d_star) on both kernels."""
    from wato_lidar_preprocessing.ray_traversal._python_kernel import (
        _update_sweep_python,
    )

    rng = np.random.default_rng(seed=43)
    n_rays = 50
    endpoints = rng.uniform(-150.0, 150.0, size=(n_rays, 3))
    is_ground = rng.choice([True, False], size=n_rays)
    sweep_origin = np.array([0.001, -0.002, 0.003])
    chunk_origin = np.array([-200.0, -200.0, -200.0])
    voxel_size = 0.5
    margin_m = 0.5
    max_length_m = 300.0
    l_occ, l_free, log_odds_clamp = 1.20, 0.25, 50.0
    d_star = 50.0  # small enough that many endpoints are down-weighted

    nlo, nno, nnh = make_log_odds_dicts()
    update_sweep_log_odds(
        sweep_origin,
        endpoints.astype(np.float64),
        is_ground,
        chunk_origin,
        voxel_size,
        margin_m,
        max_length_m,
        nlo,
        nno,
        nnh,
        l_occ,
        l_free,
        log_odds_clamp,
        d_star,
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
        margin_m,
        max_length_m,
        plo,
        pno,
        pnh,
        l_occ,
        l_free,
        log_odds_clamp,
        d_star,
    )
    p_keys = np.array(sorted(plo.keys()), dtype=np.int64)
    p_lo = np.array([plo[k] for k in p_keys], dtype=np.float32)
    p_no = np.array([int(pno[k]) for k in p_keys], dtype=np.int32)
    p_nh = np.array([int(pnh.get(k, 0)) for k in p_keys], dtype=np.int32)

    assert np.array_equal(n_keys, p_keys)
    assert np.array_equal(n_no, p_no)
    assert np.array_equal(n_nh, p_nh)
    assert np.array_equal(n_lo, p_lo), (
        "weighted log_odds diverged between Numba and Python kernels"
    )


def test_range_weight_per_voxel_free_space_python_kernel():
    """Pure-Python equivalent of the per-voxel weighting contract (no numba)."""
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
        margin_m=1.0,
        max_length_m=50.0,
        log_odds=log_odds,
        n_obs=n_obs,
        n_hits=n_hits,
        l_occ=0.85,
        l_free=0.40,
        log_odds_clamp=50.0,
        d_star=5.0,
    )

    near_key = _pack(1, 0, 0)
    mid_key = _pack(10, 0, 0)
    far_key = _pack(15, 0, 0)

    assert near_key in log_odds, "near voxel must be carved"
    assert mid_key in log_odds, "mid voxel must be carved"
    assert far_key in log_odds, "far voxel must be carved"

    np.testing.assert_allclose(float(log_odds[near_key]), -0.40, atol=1e-5)
    np.testing.assert_allclose(
        float(log_odds[mid_key]), -0.40 * (5.0 / 9.999), rtol=1e-4
    )
    np.testing.assert_allclose(
        float(log_odds[far_key]), -0.40 * (5.0 / 14.999), rtol=1e-4
    )
    assert log_odds[near_key] < log_odds[mid_key] < log_odds[far_key] < 0.0


def test_range_weight_endpoint_python_kernel():
    """Pure-Python endpoint scaling test (mirror of the numba-gated variants)."""
    from wato_lidar_preprocessing.ray_traversal._python_kernel import (
        _update_sweep_python,
    )

    sweep_origin = np.array([0.001, 0.0, 0.0])
    chunk_origin = np.array([0.0, 0.0, 0.0])

    # Case 1: 50m endpoint, d_star=200 → r* pinned to 1.0 → endpoint = +l_occ.
    log_odds: dict[int, np.float32] = {}
    n_obs: dict[int, np.int32] = {}
    n_hits: dict[int, np.int32] = {}
    _update_sweep_python(
        sweep_origin,
        np.array([[50.001, 0.0, 0.0]]),
        None,
        chunk_origin,
        1.0,
        1.0,
        200.0,
        log_odds,
        n_obs,
        n_hits,
        0.85,
        0.40,
        50.0,
        200.0,
    )
    assert _pack(50, 0, 0) in log_odds
    np.testing.assert_allclose(
        float(log_odds[_pack(50, 0, 0)]),
        0.85,
        atol=1e-6,
        err_msg="endpoint inside d_star must keep full l_occ",
    )

    # Case 2: 500m endpoint, d_star=200 → r* = 200/500 = 0.4 → endpoint = 0.34.
    log_odds = {}
    n_obs = {}
    n_hits = {}
    _update_sweep_python(
        sweep_origin,
        np.array([[500.001, 0.0, 0.0]]),
        None,
        chunk_origin,
        1.0,
        1.0,
        1000.0,
        log_odds,
        n_obs,
        n_hits,
        0.85,
        0.40,
        50.0,
        200.0,
    )
    assert _pack(500, 0, 0) in log_odds
    np.testing.assert_allclose(
        float(log_odds[_pack(500, 0, 0)]),
        0.85 * (200.0 / 500.0),
        rtol=1e-5,
        err_msg="endpoint at 2.5*d_star must scale to d_star/length",
    )

    # Case 3: large d_star → 500m endpoint gets +l_occ unchanged.
    log_odds = {}
    n_obs = {}
    n_hits = {}
    _update_sweep_python(
        sweep_origin,
        np.array([[500.001, 0.0, 0.0]]),
        None,
        chunk_origin,
        1.0,
        1.0,
        1000.0,
        log_odds,
        n_obs,
        n_hits,
        0.85,
        0.40,
        50.0,
        _NO_WEIGHT,
    )
    assert _pack(500, 0, 0) in log_odds
    np.testing.assert_allclose(
        float(log_odds[_pack(500, 0, 0)]),
        0.85,
        atol=1e-6,
        err_msg="large d_star: endpoint must carry full l_occ",
    )


@requires_numba
def test_compute_normals_recovers_plane_normal():
    """A voxel filled with points on a y-z plane yields a normal along ±x."""
    from wato_lidar_preprocessing.ray_traversal import (
        accumulate_cov,
        compute_normals,
        make_cov_dicts,
    )

    rng = np.random.default_rng(0)
    # Points spread in y,z within voxel (0,0,0), nearly constant x → normal ≈ x.
    n = 40
    pts = np.empty((n, 3))
    pts[:, 0] = 0.5 + rng.normal(0, 0.001, n)  # thin in x
    pts[:, 1] = rng.uniform(0.0, 1.0, n)
    pts[:, 2] = rng.uniform(0.0, 1.0, n)
    is_ground = np.zeros(n, dtype=bool)

    cov = make_cov_dicts()
    accumulate_cov(pts, is_ground, np.zeros(3), 1.0, cov)
    nx, ny, nz = compute_normals(cov, min_pts=3)

    key = _pack(0, 0, 0)
    assert key in nx, "a planar, well-populated voxel must get a normal"
    normal = np.array([nx[key], ny[key], nz[key]])
    assert abs(abs(normal[0]) - 1.0) < 0.05, f"normal should align with x, got {normal}"


def test_incidence_gate_skips_grazing_carve_python_kernel():
    """A through-ray grazing an occupied voxel (|ray·n| < grazing_cos) must not
    carve it; a head-on ray through the same voxel must carve it.

    Voxel (5,0,0) holds a wall whose normal points +x. A ray travelling +y
    skims the wall (dot=0) → no carve. A ray travelling +x hits it head-on
    (dot=1) → carve.
    """
    from wato_lidar_preprocessing.ray_traversal._python_kernel import (
        _update_sweep_python,
    )

    wall = _pack(5, 0, 0)
    nx = {wall: 1.0}
    ny = {wall: 0.0}
    nz = {wall: 0.0}
    chunk_origin = np.zeros(3)

    # Grazing ray: travels +y through column x-voxel 5.
    lo_g: dict[int, np.float32] = {}
    no_g: dict[int, np.int32] = {}
    nh_g: dict[int, np.int32] = {}
    _update_sweep_python(
        np.array([5.5, -3.0, 0.5]),
        np.array([[5.5, 3.0, 0.5]]),
        None,
        chunk_origin,
        1.0,
        0.5,
        50.0,
        lo_g,
        no_g,
        nh_g,
        0.85,
        0.40,
        50.0,
        _NO_WEIGHT,
        nx,
        ny,
        nz,
        0.5,
    )
    assert wall not in lo_g, "grazing ray must not carve the occupied wall voxel"

    # Head-on ray: travels +x through the same voxel.
    lo_h: dict[int, np.float32] = {}
    no_h: dict[int, np.int32] = {}
    nh_h: dict[int, np.int32] = {}
    _update_sweep_python(
        np.array([0.5, 0.5, 0.5]),
        np.array([[9.5, 0.5, 0.5]]),
        None,
        chunk_origin,
        1.0,
        0.5,
        50.0,
        lo_h,
        no_h,
        nh_h,
        0.85,
        0.40,
        50.0,
        _NO_WEIGHT,
        nx,
        ny,
        nz,
        0.5,
    )
    assert float(lo_h.get(wall, 0.0)) < 0.0, "head-on ray must carve the wall voxel"


def test_missing_numba_hard_fail_dispatch(monkeypatch):
    """When _NUMBA_AVAILABLE is False, make_log_odds_dicts must raise ImportError."""
    import wato_lidar_preprocessing.ray_traversal.dispatch as dispatch_mod

    monkeypatch.setattr(dispatch_mod, "_NUMBA_AVAILABLE", False)
    monkeypatch.setattr(dispatch_mod, "_NUMBA_IMPORT_ERROR", None)

    with pytest.raises(ImportError) as exc_info:
        dispatch_mod.make_log_odds_dicts()
    msg = str(exc_info.value)
    assert "install" in msg.lower() and "numba" in msg.lower()


def test_missing_numba_hard_fail_real_import(tmp_path):
    """Subprocess test: simulate a clean import with numba blocked."""
    script = textwrap.dedent(
        """
        import sys
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
