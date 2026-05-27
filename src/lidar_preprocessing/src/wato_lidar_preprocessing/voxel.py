"""Voxel-key packing shared by classify (Step B) and ground (Step C).

Both steps must produce the same int64 key for the same world-frame point
under the same `origin` / `voxel_size`. Lives here to avoid a private
cross-module import between classify and ground.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

# Voxel index bit-width: supports ±(2^20 * voxel_size) m from origin per axis.
# At 0.15 m that's ≈157 km; anything beyond is corrupt input and we raise.
AXIS_BITS = 20
AXIS_RANGE = 1 << AXIS_BITS  # 1 048 576


class VoxelOverflowError(ValueError):
    """Raised when world-frame points exceed the voxel-key bit budget.

    Should never trip at realistic drive lengths/voxel sizes. When it does,
    upstream coordinates are corrupted (wrong calibration sign, NaN-leaked
    pose, etc.). Silently clipping would alias every offender to the same key.
    """


def pack_voxel_key(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    *,
    chunk_id: str | None = None,
) -> np.ndarray:
    """Encode three int32 voxel indices into a single int64 key.

    Raises VoxelOverflowError if any index falls outside [0, AXIS_RANGE).
    """
    out_of_range = (
        (vx < 0)
        | (vx >= AXIS_RANGE)
        | (vy < 0)
        | (vy >= AXIS_RANGE)
        | (vz < 0)
        | (vz >= AXIS_RANGE)
    )
    n_bad = int(out_of_range.sum())
    if n_bad > 0:
        bad_idx = np.argmax(out_of_range)
        raise VoxelOverflowError(
            f"chunk {chunk_id!r}: {n_bad} voxel indices outside ±{AXIS_RANGE} range "
            f"(first offender: vx={int(vx[bad_idx])} vy={int(vy[bad_idx])} vz={int(vz[bad_idx])}). "
            "Check upstream world-frame coordinates for corruption (calibration, pose interp, NaN leak)."
        )
    vx64 = vx.astype(np.int64)
    vy64 = vy.astype(np.int64)
    vz64 = vz.astype(np.int64)
    return (vx64 << (2 * AXIS_BITS)) | (vy64 << AXIS_BITS) | vz64


def voxel_indices(
    xyz: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    *,
    chunk_id: str | None = None,
) -> np.ndarray:
    """Return packed int64 voxel key per point."""
    rel = xyz - origin  # (N, 3) float64
    idx = np.floor(rel / voxel_size).astype(np.int64)
    return pack_voxel_key(idx[:, 0], idx[:, 1], idx[:, 2], chunk_id=chunk_id)
