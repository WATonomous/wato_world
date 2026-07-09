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

    np.savez_compressed(
        local_path(voxel_occupancy_path(bag_id, chunk_id)), **save_kwargs
    )
    log.info(
        "chunk %s: wrote voxel_occupancy.npz (%d occupied voxels)",
        chunk_id,
        coords.shape[0],
    )


def write_chunk_voxel_diagnostics(
    bag_id: str,
    chunk_id: str,
    unique_keys: np.ndarray,
    log_odds: np.ndarray,
    n_obs: np.ndarray,
    n_hits: np.ndarray,
    classification: np.ndarray,
    voxel_size: float,
    origin: np.ndarray,
) -> None:
    """Write voxel_diag.npz: full per-voxel diagnostics including carved voxels.

    `classification` is produced by classify_from_log_odds — passing it in
    keeps the bucketing rules in one place. CLASS_* codes live in
    classify/log_odds.py.
    """
    if unique_keys.size == 0:
        log.info("chunk %s: no voxels to diagnose, skipping voxel_diag.npz", chunk_id)
        return

    p_occ = sigmoid(log_odds).astype(np.float32)

    coords = _coords_from_keys(unique_keys)
    save_kwargs: dict[str, np.ndarray] = {
        "keys": unique_keys.astype(np.int64),
        "coords": coords,
        "origin": np.asarray(origin, dtype=np.float64),
        "voxel_size": np.float32(voxel_size),
        "log_odds": log_odds.astype(np.float32),
        "p_occ": p_occ,
        "n_obs": n_obs.astype(np.int32),
        "n_hits": n_hits.astype(np.int32),
        "classification": classification.astype(np.int8),
    }

    np.savez_compressed(local_path(voxel_diag_path(bag_id, chunk_id)), **save_kwargs)

    from .log_odds import (
        CLASS_AMBIGUOUS,
        CLASS_CARVED_NOISE,
        CLASS_DYNAMIC,
        CLASS_FREE_ONLY,
        CLASS_STATIC,
        CLASS_UNDER_EVIDENCED,
    )

    n_per_class = {
        "static": int((classification == CLASS_STATIC).sum()),
        "ambiguous": int((classification == CLASS_AMBIGUOUS).sum()),
        "under_evidenced": int((classification == CLASS_UNDER_EVIDENCED).sum()),
        "free_only": int((classification == CLASS_FREE_ONLY).sum()),
        "carved_noise": int((classification == CLASS_CARVED_NOISE).sum()),
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
