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


# ---------------------------------------------------------------------------
# Defense-in-depth: endpoint_priors length mismatch tripwire
# ---------------------------------------------------------------------------


def _length_mismatch_inputs():
    """3 endpoints + 2 priors -> kernel must raise (off-by-one tripwire)."""
    sweep_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    chunk_origin = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
    endpoints = np.array(
        [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]], dtype=np.float64
    )
    # Deliberately one short -- the silent-disable case before the fix.
    priors = np.array([1.0, -1.0], dtype=np.float32)
    return sweep_origin, endpoints, chunk_origin, priors


def test_python_kernel_endpoint_priors_length_mismatch_raises():
    """The Python parity kernel must raise on a length-mismatched priors array.

    Pre-fix: `use_priors = endpoint_priors is not None and len(...) == len(endpoints)`
    silently set use_priors=False on mismatch and ran geometry-only -- exactly the
    failure mode the tripwire closes for future Step-3 refactors.
    """
    from wato_lidar_preprocessing.ray_traversal._python_kernel import (
        _update_sweep_python,
    )

    sweep_origin, endpoints, chunk_origin, priors = _length_mismatch_inputs()
    log_odds, n_obs, n_hits = {}, {}, {}
    max_logit: dict = {}

    with pytest.raises(ValueError, match="endpoint_priors length"):
        _update_sweep_python(
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
            endpoint_priors=priors,
            alpha=1.0,
            max_logit=max_logit,
            track_max_logit=True,
        )


def test_python_kernel_zero_length_priors_still_geometry_only():
    """Length-0 priors array is the 'no priors' sentinel; must NOT raise.

    Mirrors the numba kernel's convention where the typed-array signature
    can't accept None.
    """
    from wato_lidar_preprocessing.ray_traversal._python_kernel import (
        _update_sweep_python,
    )

    sweep_origin, endpoints, chunk_origin, _ = _length_mismatch_inputs()
    log_odds, n_obs, n_hits = {}, {}, {}
    max_logit: dict = {}
    empty_priors = np.empty(0, dtype=np.float32)

    # No raise expected -- geometry-only path with zero-length sentinel.
    _update_sweep_python(
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
        endpoint_priors=empty_priors,
        alpha=1.0,
        max_logit=max_logit,
        track_max_logit=False,
    )
    # Sanity: at least one endpoint voxel registered an l_occ hit.
    assert any(v > 0 for v in log_odds.values()), "endpoints should accumulate l_occ"


@requires_numba
def test_numba_kernel_endpoint_priors_length_mismatch_raises():
    """Numba mirror of the Python tripwire.

    Numba supports `raise` in nopython mode but not f-strings; the message
    in the kernel is a literal, so we only match the prefix.
    """
    from wato_lidar_preprocessing.ray_traversal import (
        make_log_odds_dicts,
        make_max_logit_dict,
        update_sweep_log_odds,
    )

    sweep_origin, endpoints, chunk_origin, priors = _length_mismatch_inputs()
    log_odds, n_obs, n_hits = make_log_odds_dicts()
    max_logit = make_max_logit_dict()

    with pytest.raises(ValueError, match="endpoint_priors length"):
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
            endpoint_priors=priors,
            alpha=1.0,
            max_logit=max_logit,
        )


def test_import_emits_no_warning_when_numba_missing():
    """Importing the package without numba must NOT log a warning.

    Pre-fix: dispatch.py warned at module load, polluting any tooling
    (ground, viz, deskew) that imports the package for non-log-odds work.
    Post-fix: silent at import; warning/raise happens only on first
    _require_numba() call.

    Skipped if subprocess invocation fails for environmental reasons.
    """
    script = textwrap.dedent(
        """
        import sys
        import io
        import logging

        # Capture root-logger output to stderr-like buffer BEFORE imports.
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.WARNING)
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.DEBUG)

        # Block numba so dispatch.py falls into the unavailable branch.
        sys.modules['numba'] = None
        sys.modules['numba.types'] = None
        sys.modules['numba.typed'] = None

        from wato_lidar_preprocessing.ray_traversal import _NUMBA_AVAILABLE
        assert _NUMBA_AVAILABLE is False

        handler.flush()
        captured = buf.getvalue()
        # The import path must not log a WARNING-level message about numba.
        if 'numba unavailable' in captured.lower() or 'numba is missing' in captured.lower():
            print('FAIL: import emitted a warning:\\n' + captured)
            sys.exit(2)
        print('OK')
        sys.exit(0)
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
