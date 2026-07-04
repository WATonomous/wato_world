"""Step A — Deskew and project LiDAR sweeps into world frame.

Reads raw per-sweep NPZ + poses.parquet + calibration.json from ingest.
For each sweep, applies per-point pose interpolation so every point is
expressed in the SLAM world frame at its own sensor timestamp. Writes one
float64 world-frame NPZ per sweep plus a lidar_proc_index.parquet summary.

Patchwork++ runs on the raw sensor-frame xyz BEFORE deskewing (mirrors the
wato_monorepo C++ node). The resulting boolean mask rides along in the
world NPZ as `ground_mask` so downstream stages don't re-run Patchwork++.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from tqdm import tqdm

from wato_common.artifact_store import (
    ensure_local_dir,
    lidar_proc_dir,
    lidar_proc_index_path,
    lidar_sweeps_path,
    lidar_world_path,
    local_path,
)
from wato_common.geometry import PoseSample, batch_interpolate_poses
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import PROCESSED_SWEEPS_SCHEMA, ProcessedSweepMeta
from wato_lidar_preprocessing._inputs import (
    load_ego_T_lidar,
    load_pose_samples,
    load_pose_valid_sweep_ids,
)
from wato_lidar_preprocessing.config import (
    ComponentConfig,
    FrameSyncParams,
    PatchworkParams,
)

log = logging.getLogger(__name__)

# Cap on per-point offsets (ns). A 5 Hz sweep is 200 ms = 2e8 ns; we accept
# up to 1 s. Anything bigger means `point_time_unit` is mis-configured.
_MAX_POINT_TIME_OFFSET_NS = 1_000_000_000


@dataclass
class DeskewResult:
    sweep_id: int
    lidar_id: str
    n_points: int
    deskewed: bool
    world_path: str


def _make_patchwork(params: PatchworkParams, *, required: bool = True) -> Optional[Any]:
    """Lazy-import + construct one Patchwork++ instance.

    Raises ImportError when ``required=True`` and pypatchworkpp isn't
    installed. Without ground extraction, ground points pollute both
    static_map.npz and dynamic_map.npz.
    """
    try:
        import pypatchworkpp  # type: ignore[import]
    except ImportError as exc:
        if required:
            raise ImportError(
                "pypatchworkpp is required for per-sweep ground extraction. "
                "Without it, ground points pollute both static_map.npz and "
                "dynamic_map.npz. Install via "
                "'uv pip install pypatchworkpp==1.0.4'. If you really need "
                "to run without ground extraction, set 'require_patchwork: "
                "false' in lidar_preprocessing.yaml."
            ) from exc
        log.warning(
            "pypatchworkpp not installed and require_patchwork=False — "
            "per-sweep ground extraction skipped. Ground points WILL "
            "pollute static_map.npz and dynamic_map.npz."
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
        # Older bindings without getGroundIndices — fall back to point-set
        # match. O(N log N), slightly less precise due to fp comparison.
        ground_pts = np.asarray(pw.getGround(), dtype=np.float32)
        if ground_pts.size == 0:
            return mask
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

    1. canonical_lidar is None / absent from the chunk → per-lidar sequential
       frame_ids (0, 1, 2, ...) ordered by timestamp. Correct for single-
       lidar bags and the only safe fallback when the configured canonical
       is missing.

    2. canonical_lidar present → canonical sweeps anchor frames (0..N-1 by
       sorted timestamp); non-canonical sweeps within ±tolerance_ns of a
       canonical sweep inherit its frame_id, otherwise frame_id=None.

    Rows with valid=False always get frame_id=None.
    """
    tol_ns = frame_sync.tolerance_ns()
    canonical = frame_sync.canonical_lidar

    valid_rows = [r for r in meta_rows if r.get("valid", True) is not False]
    canonical_rows = (
        [r for r in valid_rows if r["lidar_id"] == canonical] if canonical else []
    )

    if not canonical_rows:
        # Mode 1: per-lidar sequential.
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

    # Mode 2: canonical lidar present.
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
            # Already assigned above; guard against None-typed `valid` rows.
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


def _synthesize_t_offset_ns_from_azimuth(
    xyz_lidar: np.ndarray,
    sweep_duration_ns: float,
    rotation_dir: str,
) -> np.ndarray:
    """Compute per-point time offsets [ns] from azimuth for a rotating LiDAR.

    Sweep start is anchored at phi_start = phi[0] — the raw NPZ preserves
    the bag's point order, which is firing order for standard rosbag
    conversions. NuScenes LIDAR_TOP starts at the back of the vehicle
    (phi ≈ -π), NOT phi = 0, so anchoring on phi[0] (not 0) is required.

    Args:
        xyz_lidar: (N, 3) float — sensor-frame points in firing order.
        sweep_duration_ns: full rotation duration in nanoseconds.
        rotation_dir: "cw" / "ccw" — the scanner's spin direction, from the
            sensor_model profile (a fixed hardware property, not probed).

    Returns:
        (N,) float64 — per-point offsets in [0, sweep_duration_ns); offset[0]==0.
    """
    n = xyz_lidar.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    phi = np.arctan2(xyz_lidar[:, 1], xyz_lidar[:, 0])  # (-π, π]
    phi_start = phi[0]

    if rotation_dir == "ccw":
        delta = (phi - phi_start) % (2.0 * np.pi)
    elif rotation_dir == "cw":
        delta = (phi_start - phi) % (2.0 * np.pi)
    else:
        raise ValueError(
            f"rotation_dir must be 'ccw' or 'cw'; got {rotation_dir!r}"
        )
    frac = delta / (2.0 * np.pi)
    return (frac * sweep_duration_ns).astype(np.float64)


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
    synthesize_per_point_times: bool = False,
    sweep_duration_ns: float = 0.0,
    rotation_dir: str = "ccw",
    allow_uncompensated_motion: bool = False,
) -> dict[str, np.ndarray]:
    """Transform one sweep from sensor frame to world frame (float64 xyz).

    Returns a dict of arrays ready for np.savez_compressed. Includes a
    `ground_mask` bool array when Patchwork++ is available.
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
            log.debug("dropped %d non-finite points from %s", n_drop, raw_path)
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

    # Per-sweep ground extraction runs in SENSOR frame.
    ground_mask = _estimate_ground_mask(xyz_lidar, pw)

    if has_point_time and t_offset is not None:
        offset_ns = t_offset.astype(np.float64) * unit_scale
        # Fail loudly on mis-configured point_time_unit (offsets explode by
        # 1e3 or 1e9 → kilometre-scale displacement).
        max_abs = float(np.max(np.abs(offset_ns))) if offset_ns.size else 0.0
        if max_abs > _MAX_POINT_TIME_OFFSET_NS:
            raise ValueError(
                f"per-point time offsets exceed {_MAX_POINT_TIME_OFFSET_NS} ns "
                f"(max |offset|={max_abs:.0f} ns) — `point_time_unit` is likely "
                "mis-configured for this lidar."
            )
        t_ns = header_timestamp_ns + offset_ns
    elif synthesize_per_point_times and n > 0 and sweep_duration_ns > 0:
        # Raw NPZ has no per-point timestamps — synthesize from azimuth.
        # Without compensation, all points share the header pose, producing
        # ~ego_speed × sweep_duration / 2 of intra-sweep smear that spreads
        # statics across voxels (the "buildings-as-dynamic" failure mode).
        offset_ns = _synthesize_t_offset_ns_from_azimuth(
            xyz_lidar, sweep_duration_ns, rotation_dir
        )
        t_ns = header_timestamp_ns + offset_ns
    elif n == 0 or allow_uncompensated_motion:
        t_ns = np.full(n, float(header_timestamp_ns), dtype=np.float64)
    else:
        raise ValueError(
            "deskew has no way to compensate intra-sweep ego motion: the raw "
            "NPZ has no per-point timestamps AND synthesize_per_point_times "
            "is False. Either (a) re-ingest with a bag that has per-point "
            "times, (b) set synthesize_per_point_times: true to synthesize "
            "from azimuth (recommended for rotating LiDARs), or (c) set "
            "allow_uncompensated_motion: true if you accept smeared statics "
            "potentially leaking into dynamic_map.npz."
        )

    unique_ts, inv = np.unique(t_ns.astype(np.int64), return_inverse=True)

    world_T_ego_u = batch_interpolate_poses(pose_samples, unique_ts)
    world_T_lidar_u = world_T_ego_u @ ego_T_lidar  # (U,4,4) @ (4,4)

    R = world_T_lidar_u[inv, :3, :3]
    t = world_T_lidar_u[inv, :3, 3]

    xyz_world = np.einsum("nij,nj->ni", R, xyz_lidar) + t

    # Sensor origin at the sweep midpoint timestamp — halves worst-case
    # positional error vs picking sweep-start or sweep-end.
    mid_idx = len(unique_ts) // 2
    sensor_origin = world_T_lidar_u[mid_idx, :3, 3].copy()

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

    pose_samples = load_pose_samples(bag_id, chunk_id)
    if not pose_samples:
        log.warning("chunk %s has no valid poses; skipping deskew", chunk_id)
        write_table([], PROCESSED_SWEEPS_SCHEMA, index_uri)
        return []

    # Sweeps ingest flagged as pose-invalid (no interpolatable pose within
    # max_pose_gap_ms — e.g. before SLAM converges at bag start) are skipped
    # rather than clamped to a stale pose. None = no frame_index → process all.
    pose_valid_sweep_ids = load_pose_valid_sweep_ids(bag_id, chunk_id)

    unit_scale = cfg.point_time_scale_to_ns()

    sweep_rows = read_rows(lidar_sweeps_path(bag_id, chunk_id))

    lidar_ids = {r["lidar_id"] for r in sweep_rows if r.get("valid", True)}
    ego_T_lidar_by_id: dict[str, np.ndarray] = {}
    for lid in lidar_ids:
        try:
            ego_T_lidar_by_id[lid] = load_ego_T_lidar(bag_id, lid)
        except (KeyError, ValueError) as exc:
            log.error("lidar %s: %s", lid, exc)

    pw = _make_patchwork(cfg.patchwork, required=cfg.require_patchwork)

    # Scan rotation direction is a fixed hardware property — read it from the
    # datasheet sensor profile rather than probing azimuth signs per sweep.
    rotation_dir = cfg.build_sensor_model().rotation_dir

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

        # Honor ingest's per-sweep valid_pose: skip sweeps with no interpolatable
        # pose instead of clamping them to a stale one. Record an explicit
        # valid=False row so downstream distinguishes this from a missing sweep.
        if pose_valid_sweep_ids is not None and sweep_id not in pose_valid_sweep_ids:
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
                    drop_reason="pose_invalid: no interpolatable ego pose "
                    "(valid_pose=False in frame_index)",
                ).model_dump()
            )
            continue

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
                synthesize_per_point_times=cfg.synthesize_per_point_times,
                sweep_duration_ns=cfg.lidar_sweep_duration_ms * 1_000_000.0,
                rotation_dir=rotation_dir,
                allow_uncompensated_motion=cfg.allow_uncompensated_motion,
            )
        except Exception as exc:  # noqa: BLE001 — record failure, keep going
            log.exception("sweep %d deskew failed", sweep_id)
            # Explicit valid=False row distinguishes "sweep failed" from
            # "sweep never existed" for downstream stages.
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

        # Cache world bbox + ground count so classify/ground don't re-scan
        # every NPZ to plan their work.
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

    # Assign frame_id before persisting the parquet so the on-disk index
    # already carries it (classify reads the column back).
    _assign_frame_ids(meta_rows, cfg.frame_sync)

    write_table(meta_rows, PROCESSED_SWEEPS_SCHEMA, index_uri)
    log.info("deskewed %d sweeps for chunk %s", len(results), chunk_id)
    return results
