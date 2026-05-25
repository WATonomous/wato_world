"""Voxel-occupancy NPZ writers (chunk + per-frame)."""

from __future__ import annotations

import logging

import numpy as np

from wato_common.artifact_store import (
    local_path,
    voxel_diag_path,
    voxel_occupancy_frame_path,
    voxel_occupancy_path,
)
from wato_lidar_preprocessing.voxel import AXIS_BITS

from .io_helpers import sigmoid

log = logging.getLogger(__name__)

# Shared bit-mask for unpacking packed int64 keys → (vx, vy, vz) int32.
_AXIS_MASK = np.int64((1 << AXIS_BITS) - 1)


def _coords_from_keys(keys: np.ndarray) -> np.ndarray:
    vz = (keys & _AXIS_MASK).astype(np.int32)
    vy = ((keys >> np.int64(AXIS_BITS)) & _AXIS_MASK).astype(np.int32)
    vx = ((keys >> np.int64(2 * AXIS_BITS)) & _AXIS_MASK).astype(np.int32)
    return np.stack([vx, vy, vz], axis=1)


def write_chunk_voxel_occupancy(
    bag_id: str,
    chunk_id: str,
    unique_keys: np.ndarray,
    voxel_size: float,
    origin: np.ndarray,
    *,
    log_odds: np.ndarray | None = None,
    n_obs: np.ndarray | None = None,
    n_hits: np.ndarray | None = None,
) -> None:
    """Write voxel_occupancy.npz aggregating all occupied voxels in the chunk.

    In log_odds mode unique_keys also includes free-space-only voxels;
    filter to log_odds >= 0 (net-occupied evidence) before saving.
    """
    if log_odds is not None and log_odds.size > 0:
        occ_mask = log_odds >= 0
        occ_keys = unique_keys[occ_mask]
        occ_lo = log_odds[occ_mask]
        occ_n_obs = n_obs[occ_mask] if n_obs is not None else None
        occ_n_hits = n_hits[occ_mask] if n_hits is not None else None
    else:
        occ_keys = unique_keys
        occ_lo = None
        occ_n_obs = None
        occ_n_hits = None

    coords = _coords_from_keys(occ_keys)
    save_kwargs: dict[str, np.ndarray] = {
        "coords": coords,
        "origin": origin,
        "voxel_size": np.float32(voxel_size),
    }
    if occ_lo is not None:
        save_kwargs["log_odds"] = occ_lo
        if occ_n_obs is not None:
            save_kwargs["n_obs"] = occ_n_obs
        if occ_n_hits is not None:
            save_kwargs["n_hits"] = occ_n_hits
        save_kwargs["p_occ"] = sigmoid(occ_lo).astype(np.float32)

    np.savez_compressed(local_path(voxel_occupancy_path(bag_id, chunk_id)), **save_kwargs)
    log.info(
        "chunk %s: wrote voxel_occupancy.npz (%d occupied voxels)",
        chunk_id,
        coords.shape[0],
    )


# Classification label codes for voxel_diag.npz.  int8 storage keeps the
# artifact small; viz code maps these back to category names.
CLASS_STATIC = 0
CLASS_AMBIGUOUS = 1          # evidenced + has_hits + p_dynamic ≤ p_occ < p_static
CLASS_UNDER_EVIDENCED = 2    # has_hits but n_obs < min_observations
CLASS_FREE_ONLY = 3          # n_hits == 0 (only ever traversed)
CLASS_DYNAMIC = 4            # evidenced + has_hits + p_occ < p_dynamic_threshold


def write_chunk_voxel_diagnostics(
    bag_id: str,
    chunk_id: str,
    unique_keys: np.ndarray,
    log_odds: np.ndarray,
    n_obs: np.ndarray,
    n_hits: np.ndarray,
    voxel_size: float,
    origin: np.ndarray,
    *,
    p_static_threshold: float,
    p_dynamic_threshold: float,
    min_observations: int,
    min_occupied_hits: int,
) -> None:
    """Write voxel_diag.npz: full per-voxel diagnostics including carved voxels.

    Unlike voxel_occupancy.npz which filters to log_odds >= 0 for MinkUNet,
    this artifact keeps every classified voxel so the viz can colour
    dynamic points by their voxel's p_occ.  Adds a `classification` column
    (int8) using the CLASS_* codes so post-hoc analysis doesn't have to
    re-derive the bucketing logic.
    """
    if unique_keys.size == 0:
        log.info("chunk %s: no voxels to diagnose, skipping voxel_diag.npz", chunk_id)
        return

    p_occ = sigmoid(log_odds).astype(np.float32)
    evidenced = n_obs >= min_observations
    has_hits = n_hits >= min_occupied_hits

    classification = np.full(unique_keys.shape[0], CLASS_FREE_ONLY, dtype=np.int8)
    # Order matters: each later assignment overrides earlier ones where
    # masks overlap, so we go least- to most-specific.
    classification[has_hits & ~evidenced] = CLASS_UNDER_EVIDENCED
    classification[evidenced & has_hits & (p_occ < p_dynamic_threshold)] = CLASS_DYNAMIC
    classification[
        evidenced
        & has_hits
        & (p_occ >= p_dynamic_threshold)
        & (p_occ < p_static_threshold)
    ] = CLASS_AMBIGUOUS
    classification[evidenced & has_hits & (p_occ >= p_static_threshold)] = CLASS_STATIC

    coords = _coords_from_keys(unique_keys)
    np.savez_compressed(
        local_path(voxel_diag_path(bag_id, chunk_id)),
        keys=unique_keys.astype(np.int64),
        coords=coords,
        origin=np.asarray(origin, dtype=np.float64),
        voxel_size=np.float32(voxel_size),
        log_odds=log_odds.astype(np.float32),
        p_occ=p_occ,
        n_obs=n_obs.astype(np.int32),
        n_hits=n_hits.astype(np.int32),
        classification=classification,
    )

    n_per_class = {
        "static": int((classification == CLASS_STATIC).sum()),
        "ambiguous": int((classification == CLASS_AMBIGUOUS).sum()),
        "under_evidenced": int((classification == CLASS_UNDER_EVIDENCED).sum()),
        "free_only": int((classification == CLASS_FREE_ONLY).sum()),
        "dynamic": int((classification == CLASS_DYNAMIC).sum()),
    }
    log.info(
        "chunk %s: wrote voxel_diag.npz (%d voxels: %s)",
        chunk_id,
        unique_keys.shape[0],
        ", ".join(f"{k}={v}" for k, v in n_per_class.items()),
    )


def write_per_frame_voxel_occupancy(
    bag_id: str,
    chunk_id: str,
    frame_keys: dict[int, list[np.ndarray]],
    voxel_size: float,
    origin: np.ndarray,
) -> int:
    """Write one voxel_occupancy_frame_NNNN.npz per frame_id; return count."""
    n_frames = 0
    for fid, key_list in frame_keys.items():
        merged = np.unique(np.concatenate(key_list))
        if merged.size == 0:
            continue
        coords = _coords_from_keys(merged)
        np.savez_compressed(
            local_path(voxel_occupancy_frame_path(bag_id, chunk_id, fid)),
            coords=coords,
            origin=origin,
            voxel_size=np.float32(voxel_size),
        )
        n_frames += 1
    log.info("chunk %s: wrote %d per-frame voxel_occupancy files", chunk_id, n_frames)
    return n_frames
