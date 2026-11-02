"""Tests for viz.py pure helpers (no matplotlib / no GUI)."""

from __future__ import annotations

import numpy as np

from wato_perception_2d.viz import _depth_to_norm


def test_invalid_pixels_excluded_and_zeroed():
    """Non-finite / non-positive pixels are marked invalid and forced to norm 0."""
    depth = np.full((20, 20), 10.0, dtype=np.float32)
    depth[0, 0] = np.nan
    depth[0, 1] = 0.0
    depth[0, 2] = -5.0
    depth[1, :] = 25.0  # give the valid region some spread

    norm, valid, vmin, vmax = _depth_to_norm(depth)

    assert not valid[0, 0] and not valid[0, 1] and not valid[0, 2]
    assert norm[0, 0] == 0.0 and norm[0, 1] == 0.0 and norm[0, 2] == 0.0
    assert valid[1, 0]
    assert np.isfinite(vmin) and np.isfinite(vmax)


def test_constant_depth_does_not_divide_by_zero():
    """A flat valid map widens vmax by an epsilon rather than dividing by zero."""
    depth = np.full((8, 8), 12.0, dtype=np.float32)

    norm, valid, vmin, vmax = _depth_to_norm(depth)

    assert valid.all()
    assert vmax > vmin
    assert np.isfinite(norm).all()
