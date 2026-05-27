"""Legacy persistence-counting classifier (kept for A/B comparison).

Counts how many distinct sweeps each voxel appears in and thresholds on
`max(static_sweep_min, static_sweep_fraction * n_sweeps)`. Biased with a
moving ego (per-voxel hit count is dominated by ego dwell time, not object
motion), but cheap and useful as a baseline.
"""

from __future__ import annotations

import numpy as np

from wato_lidar_preprocessing.config import ComponentConfig


def count_unique_sweeps_per_voxel(
    sweep_keys: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Count distinct sweeps each voxel appears in.

    Builds a single (key, sweep_idx) array, dedupes per (voxel, sweep) so a
    voxel hit twice within one sweep counts once, then groups by voxel key
    and counts sweeps.

    Returns:
        unique_keys: sorted int64 array of voxel keys
        sweep_counts: int64 array, parallel to unique_keys
    """
    parts_keys: list[np.ndarray] = []
    for keys in sweep_keys:
        if keys.size == 0:
            continue
        parts_keys.append(np.unique(keys))

    if not parts_keys:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    all_keys = np.concatenate(parts_keys)
    unique_keys, sweep_counts = np.unique(all_keys, return_counts=True)
    return unique_keys, sweep_counts.astype(np.int64)


def classify_persistence(
    sweep_keys: list[np.ndarray],
    n_sweeps: int,
    cfg: ComponentConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Persistence-counting classifier.

    Returns (static_arr, not_dynamic_arr, threshold). In the persistence
    path every voxel that isn't static is dynamic, so not_dynamic_arr ==
    static_arr (kept as a separate return so the orchestration code can
    treat the log-odds and persistence paths uniformly).
    """
    unique_keys, sweep_counts = count_unique_sweeps_per_voxel(sweep_keys)
    threshold = max(cfg.static_sweep_min, int(cfg.static_sweep_fraction * n_sweeps))
    if unique_keys.size > 0:
        static_arr = unique_keys[sweep_counts >= threshold]
    else:
        static_arr = np.empty(0, dtype=np.int64)
    return static_arr, static_arr, threshold
