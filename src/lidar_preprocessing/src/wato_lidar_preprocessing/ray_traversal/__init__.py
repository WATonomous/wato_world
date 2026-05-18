"""Amanatides-Woo 3D-DDA ray traversal for voxel log-odds updates.

Public surface mirrors what the old monolithic ray_traversal.py exported,
so callers (classify, tests) can keep their imports unchanged.
"""

from .dispatch import (
    _MISSING_NUMBA_MSG,
    _NUMBA_AVAILABLE,
    _NUMBA_IMPORT_ERROR,
    extract_log_odds_arrays,
    make_log_odds_dicts,
    update_sweep_log_odds,
)

__all__ = [
    "extract_log_odds_arrays",
    "make_log_odds_dicts",
    "update_sweep_log_odds",
    "_NUMBA_AVAILABLE",
    "_MISSING_NUMBA_MSG",
    "_NUMBA_IMPORT_ERROR",
]
