"""Step A — Deskew and project LiDAR sweeps into world frame.

Reads raw per-sweep NPZ files + poses.parquet + calibration.json written by
ingest.  For each sweep, applies per-point pose interpolation so that every
point is expressed in the SLAM world frame at its own sensor timestamp
(motion-compensated deskewing).  Output is one float64 world-frame NPZ per
sweep plus a lidar_proc_index.parquet summary.

Per-sweep ground extraction (Patchwork++) runs on the raw sensor-frame xyz
BEFORE deskewing, matching the wato_monorepo C++ patchwork node that consumes
sensor-frame PointCloud2 messages.  The resulting boolean mask is stored as
the `ground_mask` array inside the world NPZ so Step C (ground.py) can
aggregate ground points across sweeps without re-running Patchwork++.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from tqdm import tqdm

from wato_common.artifact_store import (
    calibration_path,
    ensure_local_dir,
    lidar_proc_dir,
    lidar_proc_index_path,
    lidar_sweeps_path,
    lidar_world_path,
    local_path,
    poses_path,
)
from wato_common.geometry import PoseSample, batch_interpolate_poses, unflatten_se3
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import PROCESSED_SWEEPS_SCHEMA, ProcessedSweepMeta
from wato_lidar_preprocessing.config import (
    ComponentConfig,
    FrameSyncParams,
    PatchworkParams,
)

log = logging.getLogger(__name__)

# Sanity bound for per-point timestamp offsets, in nanoseconds.  A LiDAR sweep
# at 5 Hz lasts 200 ms = 2e8 ns; we accept up to 1 s as a comfortable upper
# bound.  Anything bigger means `point_time_unit` is mis-configured (e.g.
# microseconds being treated as seconds) and the resulting deskew would
# wildly displace points.
_MAX_POINT_TIME_OFFSET_NS = 1_000_000_000  # 1 second


@dataclass
class DeskewResult:
    sweep_id: int
    lidar_id: str
    n_points: int
    deskewed: bool
    world_path: str
    has_ground_mask: bool = False


def _load_pose_samples(bag_id: str, chunk_id: str) -> list[PoseSample]:
    rows = read_rows(poses_path(bag_id, chunk_id))
    samples: list[PoseSample] = []
    for r in rows:
        if not r.get("valid", True):
            continue
        flat = r["world_T_ego_flat"]
        T = unflatten_se3(flat)
        # Sanity check: matrix translation must agree with the (x,y,z) columns.
        # If they disagree the upstream writer is broken — warn but trust matrix.
        xyz = np.array([r["x"], r["y"], r["z"]], dtype=np.float64)
        if not np.allclose(T[:3, 3], xyz, atol=1e-6):
            log.warning(
                "pose row at t=%d: world_T_ego translation %s != (x,y,z)=%s",
                int(r["timestamp_ns"]),
                T[:3, 3].tolist(),
                xyz.tolist(),
            )
        samples.append(
            PoseSample(
                timestamp_ns=int(r["timestamp_ns"]),
                translation=T[:3, 3].copy(),
                quat_xyzw=np.array(
                    [r["qx"], r["qy"], r["qz"], r["qw"]], dtype=np.float64
                ),
            )
        )
    samples.sort(key=lambda s: s.timestamp_ns)
    return samples


def _load_ego_T_lidar(bag_id: str, lidar_id: str) -> np.ndarray:
    """Return the 4×4 ego_T_lidar extrinsic from calibration.json."""
    calib_file = local_path(calibration_path(bag_id))
    with open(calib_file, "r", encoding="utf-8") as fh:
        calib = json.load(fh)
    lidar_entry = calib.get("lidars", {}).get(lidar_id)
    if lidar_entry is None:
        raise KeyError(f"lidar {lidar_id!r} not found in calibration.json")
    if lidar_entry.get("ego_T_lidar") is None:
        raise ValueError(
            f"lidar {lidar_id!r} has null ego_T_lidar in calibration.json "
            "(extrinsic unresolved — check /tf_static coverage)"
        )
    return np.asarray(lidar_entry["ego_T_lidar"], dtype=np.float64)


def _make_patchwork(params: PatchworkParams) -> Optional[Any]:
    """Lazy-import + construct one Patchwork++ instance, or return None."""
    try:
        import pypatchworkpp  # type: ignore[import]
    except ImportError:
        log.warning(
            "pypatchworkpp not installed — per-sweep ground extraction skipped. "
            "Install via: uv pip install pypatchworkpp==1.0.4"
        )
        return None
    p = pypatchworkpp.Parameters()
    for k, v in params.to_patchwork_dict().items():
        setattr(p, k, v)
    return pypatchworkpp.patchworkpp(p)


def _estimate_ground_mask(
    xyz_lidar: np.ndarray,
    pw: Optional[Any],
) -> Optional[np.ndarray]:
    """Run Patchwork++ on sensor-frame points and return per-point ground mask.

    Returns None if pw is None (Patchwork++ unavailable).  Otherwise returns
    a bool array of shape (N,) where True == ground.
    """
    if pw is None or xyz_lidar.shape[0] == 0:
        return None

    pw.estimateGround(xyz_lidar.astype(np.float32))
    n = xyz_lidar.shape[0]
    mask = np.zeros(n, dtype=bool)
    try:
        idx = np.asarray(pw.getGroundIndices(), dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < n)]
        mask[idx] = True
    except AttributeError:
        # Older bindings without getGroundIndices: fall back to point-set match.
        # This is O(N log N) and slightly less precise due to fp comparison.
        ground_pts = np.asarray(pw.getGround(), dtype=np.float32)
        if ground_pts.size == 0:
            return mask
        # Build a hashable view of input points and look up ground points.
        view_in = xyz_lidar.astype(np.float32).view([("", np.float32)] * 3).reshape(-1)
        view_g = ground_pts[:, :3].view([("", np.float32)] * 3).reshape(-1)
        mask = np.isin(view_in, view_g)
    return mask


def _assign_frame_ids(
    meta_rows: list[dict],
    frame_sync: FrameSyncParams,
) -> None:
    """Populate the ``frame_id`` field on each meta row in place.

    Two modes:

    1. ``frame_sync.canonical_lidar`` is None / not in the chunk → each lidar's
       sweeps get sequential frame_ids (0, 1, 2, ...) ordered by timestamp.
       This is the right behaviour for single-lidar bags (NuScenes mini) and
       the only sensible fallback when the configured canonical lidar isn't
       present in this chunk.

    2. ``frame_sync.canonical_lidar`` matches sweeps in the chunk → the
       canonical sweeps anchor the frames (frame_id = 0..N-1 by sorted
       timestamp); non-canonical sweeps whose timestamp falls within
       ±tolerance_ns of a canonical sweep inherit that canonical frame_id.
       Non-canonical sweeps outside any window get frame_id=None — downstream
       can drop them or treat them as standalone.

    Rows with ``valid is False`` always get ``frame_id=None``.
    """
    tol_ns = frame_sync.tolerance_ns()
    canonical = frame_sync.canonical_lidar

    valid_rows = [r for r in meta_rows if r.get("valid", True) is not False]
    canonical_rows = (
        [r for r in valid_rows if r["lidar_id"] == canonical] if canonical else []
    )

    if not canonical_rows:
        # Mode 1: per-lidar sequential.  Group valid rows by lidar_id and
        # assign frame_id by sorted timestamp.
        if canonical:
            log.warning(
                "frame_sync.canonical_lidar=%r not present in chunk; "
                "falling back to per-lidar sequential frame_id",
                canonical,
            )
        by_lidar: dict[str, list[dict]] = {}
        for r in valid_rows:
            by_lidar.setdefault(r["lidar_id"], []).append(r)
        for rows in by_lidar.values():
            rows.sort(key=lambda r: int(r["reference_timestamp_ns"]))
            for i, r in enumerate(rows):
                r["frame_id"] = i
        for r in meta_rows:
            if r.get("valid", True) is False:
                r["frame_id"] = None
            elif "frame_id" not in r:
                r["frame_id"] = None
        return

    # Mode 2: canonical lidar present.  Sort canonical sweeps, assign 0..N-1,
    # then for every non-canonical valid sweep look up the nearest canonical
    # timestamp via searchsorted (both left/right neighbour candidates).
    canonical_rows.sort(key=lambda r: int(r["reference_timestamp_ns"]))
    can_ts = np.array(
        [int(r["reference_timestamp_ns"]) for r in canonical_rows], dtype=np.int64
    )
    for i, r in enumerate(canonical_rows):
        r["frame_id"] = i

    can_id_by_ts_id = {id(r): i for i, r in enumerate(canonical_rows)}

    for r in meta_rows:
        if r.get("valid", True) is False:
            r["frame_id"] = None
            continue
        if r["lidar_id"] == canonical:
            # Already assigned above.  (Guard against rows that snuck past
            # valid_rows because of None-typed `valid` columns.)
            if "frame_id" not in r:
                r["frame_id"] = can_id_by_ts_id.get(id(r))
            continue
        ts = int(r["reference_timestamp_ns"])
        i = int(np.searchsorted(can_ts, ts))
        best = -1
        best_delta = tol_ns + 1
        for j in (i - 1, i):
            if 0 <= j < can_ts.size:
                d = abs(int(can_ts[j]) - ts)
                if d <= tol_ns and d < best_delta:
                    best = j
                    best_delta = d
        r["frame_id"] = int(best) if best >= 0 else None


def _deskew_sweep(
    *,
    raw_path: str,
    header_timestamp_ns: int,
    has_point_time: bool,
    unit_scale: float,
    pose_samples: list[PoseSample],
    ego_T_lidar: np.ndarray,
    filter_nonfinite: bool,
    pw: Optional[Any],
) -> dict[str, np.ndarray]:
    """Transform one sweep from sensor frame to world frame (float64 xyz).

    Returns a dict of arrays ready for np.savez_compressed.  May include a
    `ground_mask` boolean array if Patchwork++ was available.
    """
    data = np.load(local_path(raw_path))
    x = data["x"].astype(np.float64)
    y = data["y"].astype(np.float64)
    z = data["z"].astype(np.float64)
    intensity = data["intensity"] if "intensity" in data else None
    ring = data["ring"] if "ring" in data else None
    t_offset = data["t_offset_us"] if "t_offset_us" in data else None

    if filter_nonfinite:
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        n_drop = int((~finite).sum())
        if n_drop > 0:
            log.info("dropped %d non-finite points from %s", n_drop, raw_path)
            x = x[finite]
            y = y[finite]
            z = z[finite]
            if intensity is not None:
                intensity = intensity[finite]
            if ring is not None:
                ring = ring[finite]
            if t_offset is not None:
                t_offset = t_offset[finite]

    n = x.shape[0]
    xyz_lidar = np.stack([x, y, z], axis=1)  # (N, 3)

    # Per-sweep ground extraction in SENSOR frame.
    ground_mask = _estimate_ground_mask(xyz_lidar, pw)

    if has_point_time and t_offset is not None:
        offset_ns = t_offset.astype(np.float64) * unit_scale
        # Sanity check: per-point offsets should fit inside ~one sweep.  If
        # `point_time_unit` is mis-configured the offsets explode by 1e3 or
        # 1e9 and deskewed points end up kilometres away from where they
        # should be — fail loudly instead of writing garbage to disk.
        max_abs = float(np.max(np.abs(offset_ns))) if offset_ns.size else 0.0
        if max_abs > _MAX_POINT_TIME_OFFSET_NS:
            raise ValueError(
                f"per-point time offsets exceed {_MAX_POINT_TIME_OFFSET_NS} ns "
                f"(max |offset|={max_abs:.0f} ns) — `point_time_unit` is likely "
                "mis-configured for this lidar."
            )
        t_ns = header_timestamp_ns + offset_ns
    else:
        t_ns = np.full(n, float(header_timestamp_ns), dtype=np.float64)

    # Unique timestamps to avoid redundant interpolation.
    unique_ts, inv = np.unique(t_ns.astype(np.int64), return_inverse=True)

    # world_T_ego at each unique timestamp → (U, 4, 4).
    world_T_ego_u = batch_interpolate_poses(pose_samples, unique_ts)

    # world_T_lidar per unique timestamp.
    world_T_lidar_u = world_T_ego_u @ ego_T_lidar  # broadcast: (U,4,4) @ (4,4)

    # Map back to per-point and apply.
    R = world_T_lidar_u[inv, :3, :3]  # (N, 3, 3)
    t = world_T_lidar_u[inv, :3, 3]  # (N, 3)

    # Batched: xyz_world[i] = R[i] @ xyz_lidar[i] + t[i]
    xyz_world = np.einsum("nij,nj->ni", R, xyz_lidar) + t  # (N, 3) float64

    # Sensor origin in world frame at the sweep midpoint timestamp.  Using the
    # midpoint halves the worst-case positional error compared to the sweep-start
    # or sweep-end timestamps (e.g. 3 m/s ego × 0.1 s/2 ≈ 0.15 m error vs 0.30 m).
    mid_idx = len(unique_ts) // 2
    sensor_origin = world_T_lidar_u[mid_idx, :3, 3].copy()  # (3,) float64

    out: dict[str, np.ndarray] = {
        "x": xyz_world[:, 0],
        "y": xyz_world[:, 1],
        "z": xyz_world[:, 2],
        "origin": sensor_origin,
    }
    if intensity is not None:
        out["intensity"] = intensity.astype(np.float32)
    if ring is not None:
        out["ring"] = ring
    if ground_mask is not None:
        out["ground_mask"] = ground_mask
    return out


def process_chunk(
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str,
) -> list[DeskewResult]:
    """Deskew all valid sweeps in a chunk and write world-frame NPZ files."""
    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))
    index_uri = lidar_proc_index_path(bag_id, chunk_id)

    pose_samples = _load_pose_samples(bag_id, chunk_id)
    if not pose_samples:
        log.warning("chunk %s has no valid poses; skipping deskew", chunk_id)
        write_table([], PROCESSED_SWEEPS_SCHEMA, index_uri)
        return []

    unit_scale = cfg.point_time_scale_to_ns()

    sweep_rows = read_rows(lidar_sweeps_path(bag_id, chunk_id))

    # Group by lidar_id so we load calibration once per lidar.
    lidar_ids = {r["lidar_id"] for r in sweep_rows if r.get("valid", True)}
    ego_T_lidar_by_id: dict[str, np.ndarray] = {}
    for lid in lidar_ids:
        try:
            ego_T_lidar_by_id[lid] = _load_ego_T_lidar(bag_id, lid)
        except (KeyError, ValueError) as exc:
            log.error("lidar %s: %s", lid, exc)

    # One Patchwork++ instance per chunk (re-used across sweeps).
    pw = _make_patchwork(cfg.patchwork)

    results: list[DeskewResult] = []
    meta_rows: list[dict] = []

    for row in tqdm(
        sweep_rows,
        desc=f"deskew chunk {chunk_id}",
        unit="sweep",
    ):
        if not row.get("valid", True):
            continue
        lid = row["lidar_id"]
        if lid not in ego_T_lidar_by_id:
            continue

        sweep_id = int(row["sweep_id"])
        header_ts = int(row["header_timestamp_ns"])
        has_pt = bool(row.get("has_point_time", False))
        raw_path = row["lidar_path"]
        ego_T_lidar = ego_T_lidar_by_id[lid]

        try:
            arrays = _deskew_sweep(
                raw_path=raw_path,
                header_timestamp_ns=header_ts,
                has_point_time=has_pt,
                unit_scale=unit_scale,
                pose_samples=pose_samples,
                ego_T_lidar=ego_T_lidar,
                filter_nonfinite=cfg.filter_nonfinite_points,
                pw=pw,
            )
        except Exception as exc:  # noqa: BLE001 — record failure, keep going
            log.exception("sweep %d deskew failed", sweep_id)
            # Write an explicit valid=False row so downstream stages can
            # tell "sweep failed" apart from "sweep never existed".
            meta_rows.append(
                ProcessedSweepMeta(
                    bag_id=bag_id,
                    chunk_id=chunk_id,
                    sweep_id=sweep_id,
                    lidar_id=lid,
                    reference_timestamp_ns=header_ts,
                    n_points_total=0,
                    n_points_static=0,
                    n_points_dynamic=0,
                    n_points_ground=0,
                    world_path="",
                    dynamic_mask_path="",
                    has_intensity=False,
                    deskewed=False,
                    valid=False,
                    drop_reason=f"deskew_failed: {type(exc).__name__}: {exc}",
                ).model_dump()
            )
            continue

        out_uri = lidar_world_path(bag_id, chunk_id, sweep_id)
        np.savez_compressed(local_path(out_uri), **arrays)

        n = arrays["x"].shape[0]

        # Cache per-sweep world bbox + ground count so classify and ground
        # don't have to re-scan every NPZ to plan their work.
        if n > 0:
            xmin, xmax = float(arrays["x"].min()), float(arrays["x"].max())
            ymin, ymax = float(arrays["y"].min()), float(arrays["y"].max())
            zmin, zmax = float(arrays["z"].min()), float(arrays["z"].max())
        else:
            xmin = xmax = ymin = ymax = zmin = zmax = None

        n_ground = int(arrays["ground_mask"].sum()) if "ground_mask" in arrays else 0

        result = DeskewResult(
            sweep_id=sweep_id,
            lidar_id=lid,
            n_points=n,
            deskewed=has_pt,
            world_path=out_uri,
            has_ground_mask="ground_mask" in arrays,
        )
        results.append(result)

        meta = ProcessedSweepMeta(
            bag_id=bag_id,
            chunk_id=chunk_id,
            sweep_id=sweep_id,
            lidar_id=lid,
            reference_timestamp_ns=header_ts,
            n_points_total=n,
            n_points_static=0,
            n_points_dynamic=0,
            n_points_ground=n_ground,
            world_path=out_uri,
            dynamic_mask_path="",
            has_intensity="intensity" in arrays,
            deskewed=has_pt,
            world_xmin=xmin,
            world_xmax=xmax,
            world_ymin=ymin,
            world_ymax=ymax,
            world_zmin=zmin,
            world_zmax=zmax,
        )
        meta_rows.append(meta.model_dump())

    # Assign frame_id across all sweeps in this chunk before we persist the
    # parquet.  Done here (rather than in pipeline.py post-pass) so the index
    # is correct on disk after deskew completes — classify reads it back and
    # must preserve the column.
    _assign_frame_ids(meta_rows, cfg.frame_sync)

    write_table(meta_rows, PROCESSED_SWEEPS_SCHEMA, index_uri)
    log.info("deskewed %d sweeps for chunk %s", len(results), chunk_id)
    return results
