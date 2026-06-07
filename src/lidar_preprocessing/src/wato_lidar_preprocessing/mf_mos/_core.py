"""Step A.5 — MF-MOS learned moving-object segmentation.

Range-image-based deep MOS. The inference half of the `--seg mos` method;
the segmentation half (masks → static/dynamic clouds) lives in segment.py.
Runs after deskew on the mos path only — never on `--seg aw`.

Outputs per sweep:
  <sweep_id:06d>_mf_mos_mask.npy   (n_raw,) bool   True == moving
  <sweep_id:06d>_mf_mos_score.npy  (n_raw,) float32 [optional]

Mask length matches the RAW sweep (before deskew's nonfinite filter) so
consumers loading raw NPZs get index-aligned arrays.

Skipped sweeps (valid=False, pose gap, empty, inference error) leave
mf_mos_mask_path=None in the index. enabled=False makes the whole step
a no-op.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from wato_common.artifact_store import (
    ensure_local_dir,
    lidar_proc_dir,
    lidar_proc_index_path,
    lidar_sweeps_path,
    local_path,
    mf_mos_mask_path,
    mf_mos_score_path,
)
from wato_common.geometry import PoseSample, batch_interpolate_poses
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import PROCESSED_SWEEPS_SCHEMA
from wato_lidar_preprocessing._inputs import load_ego_T_lidar_dict, load_pose_samples
from wato_lidar_preprocessing.config import ComponentConfig, MFMosParams

if TYPE_CHECKING:
    from ._runtime import MFMosModel

log = logging.getLogger(__name__)

# Cache key: (checkpoint_path, arch_cfg, data_cfg, device).
_MODEL_CACHE: dict[tuple[str, str, str, str], "MFMosModel"] = {}


@dataclass
class MFMosResult:
    n_sweeps_processed: int = 0
    n_sweeps_skipped_disabled: int = 0
    n_sweeps_skipped_invalid: int = 0
    n_sweeps_skipped_pose: int = 0
    n_sweeps_skipped_empty: int = 0
    n_sweeps_skipped_inference_error: int = 0
    n_points_moving: int = 0
    n_points_total: int = 0
    skip_reasons: list[tuple[int, str]] = field(default_factory=list)

    @property
    def n_skipped(self) -> int:
        return (
            self.n_sweeps_skipped_invalid
            + self.n_sweeps_skipped_pose
            + self.n_sweeps_skipped_empty
            + self.n_sweeps_skipped_inference_error
        )


def process_chunk(
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str,
) -> MFMosResult:
    """Run MF-MOS inference for every valid sweep in a chunk.

    Called only on the `mos` segmentation path (`--seg mos`); the orchestrator
    never invokes it under `--seg aw`. Writes per-sweep moving masks consumed
    by mf_mos.segment.classify_chunk.
    """
    params = cfg.mf_mos

    sweep_rows = read_rows(lidar_sweeps_path(bag_id, chunk_id))
    meta_rows = read_rows(lidar_proc_index_path(bag_id, chunk_id))
    meta_by_sid: dict[int, dict] = {int(r["sweep_id"]): r for r in meta_rows}

    pose_samples = load_pose_samples(bag_id, chunk_id)
    if not pose_samples:
        log.warning(
            "chunk %s: no valid poses; MF-MOS skipped for entire chunk", chunk_id
        )
        return MFMosResult()

    lidar_ids = {r["lidar_id"] for r in sweep_rows if r.get("valid", True) is not False}
    ego_T_lidar_by_id = load_ego_T_lidar_dict(bag_id, lidar_ids)

    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))

    model = _load_model(params)
    max_gap_ns = int(params.max_pose_gap_ms * 1_000_000)
    # Separate cap for the residual time baseline. A residual at step k spans
    # k * sweep_dt and legitimately exceeds the pose-gap cap for the longer
    # steps; gating it on max_gap_ns (the old bug) zeroed most channels.
    max_residual_gap_ns = int(params.max_residual_gap_ms * 1_000_000)

    # Group by lidar_id + sort by timestamp to build the residual sliding window.
    rows_by_lidar: dict[str, list[dict]] = {}
    for r in sweep_rows:
        if r.get("valid", True) is False:
            continue
        rows_by_lidar.setdefault(r["lidar_id"], []).append(r)
    for rows in rows_by_lidar.values():
        rows.sort(key=lambda r: int(r["header_timestamp_ns"]))

    result = MFMosResult()
    max_k = max(params.residual_steps) if params.residual_steps else 0

    for lid, lid_rows in rows_by_lidar.items():
        # Allowlist filter must come BEFORE the calibration check: fov_up/down
        # and H/W are global, so running on a LiDAR with different mount
        # geometry would project into a non-KITTI-like range image and the
        # model mispredicts. Allowlist rejections count as disabled (not
        # invalid) so the result reflects "deliberately skipped" vs. "failed".
        if (
            params.lidar_id_allowlist is not None
            and lid not in params.lidar_id_allowlist
        ):
            log.info(
                "lidar %s: not in mf_mos.lidar_id_allowlist=%s; skipping its %d sweeps",
                lid,
                params.lidar_id_allowlist,
                len(lid_rows),
            )
            for r in lid_rows:
                result.n_sweeps_skipped_disabled += 1
            continue

        if lid not in ego_T_lidar_by_id:
            log.warning("lidar %s: no calibration; skipping MF-MOS for its sweeps", lid)
            for r in lid_rows:
                result.n_sweeps_skipped_invalid += 1
                result.skip_reasons.append(
                    (int(r["sweep_id"]), f"no calibration for lidar {lid}")
                )
            continue

        ego_T_lidar = ego_T_lidar_by_id[lid]

        # Sliding window of (header_ts_ns, xyz_raw_float32) — last max_k sweeps.
        past_window: list[tuple[int, np.ndarray]] = []

        for row in lid_rows:
            sid = int(row["sweep_id"])
            raw_path = row["lidar_path"]
            cur_ts = int(row["header_timestamp_ns"])

            try:
                raw_data = np.load(local_path(raw_path))
                x_r = raw_data["x"].astype(np.float32)
                y_r = raw_data["y"].astype(np.float32)
                z_r = raw_data["z"].astype(np.float32)
                n_raw = x_r.shape[0]
                intensity_r = (
                    raw_data["intensity"].astype(np.float32)
                    if "intensity" in raw_data.files
                    else None
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "sweep %d: failed to load raw NPZ (%s); skipping MF-MOS", sid, exc
                )
                _record_meta_path(meta_by_sid, sid, None)
                result.n_sweeps_skipped_invalid += 1
                result.skip_reasons.append((sid, f"raw load: {exc}"))
                continue

            if n_raw == 0:
                np.save(
                    local_path(mf_mos_mask_path(bag_id, chunk_id, sid)),
                    np.zeros(0, dtype=bool),
                )
                if params.save_scores:
                    np.save(
                        local_path(mf_mos_score_path(bag_id, chunk_id, sid)),
                        np.zeros(0, dtype=np.float32),
                    )
                _record_meta_path(
                    meta_by_sid, sid, mf_mos_mask_path(bag_id, chunk_id, sid)
                )
                result.n_sweeps_skipped_empty += 1
                past_window = (
                    past_window + [(cur_ts, np.empty((0, 3), dtype=np.float32))]
                )[-max_k:]
                continue

            # Match deskew's nonfinite filter so the mask is index-aligned.
            if cfg.filter_nonfinite_points:
                finite = np.isfinite(x_r) & np.isfinite(y_r) & np.isfinite(z_r)
            else:
                finite = np.ones(n_raw, dtype=bool)
            xyz_cur = np.stack([x_r[finite], y_r[finite], z_r[finite]], axis=1)
            intensity_cur = intensity_r[finite] if intensity_r is not None else None
            # KITTI remission is [0, 1]; NuScenes raw is [0, 255]. Rescale to
            # match training distribution — img_means/img_stds otherwise send
            # NuScenes intensity ~1000× out of range.
            if intensity_cur is not None and params.intensity_scale != 1.0:
                intensity_cur = intensity_cur / np.float32(params.intensity_scale)
            n_finite = xyz_cur.shape[0]

            try:
                pose_cur = _interpolate_pose(pose_samples, cur_ts, max_gap_ns)
            except _PoseGapError as exc:
                log.warning("sweep %d: %s; MF-MOS writing zero mask", sid, exc)
                _write_zero_mask(bag_id, chunk_id, sid, n_raw, params.save_scores)
                _record_meta_path(
                    meta_by_sid, sid, mf_mos_mask_path(bag_id, chunk_id, sid)
                )
                result.n_sweeps_skipped_pose += 1
                result.skip_reasons.append((sid, str(exc)))
                past_window = (past_window + [(cur_ts, xyz_cur)])[-max_k:]
                continue

            range_img, pixel_to_point_idx, point_to_pixel = _range_project(
                xyz_cur,
                intensity_cur,
                params.range_image_h,
                params.range_image_w,
                params.fov_up_deg,
                params.fov_down_deg,
                min_range_m=params.min_range_m,
                max_range_m=params.max_range_m,
            )

            # One residual per configured step (zero image when unavailable).
            residuals: list[np.ndarray] = []
            for k in params.residual_steps:
                j = len(past_window) - k
                if j < 0:
                    residuals.append(
                        np.zeros(
                            (params.range_image_h, params.range_image_w),
                            dtype=np.float32,
                        )
                    )
                    continue
                past_ts, past_xyz = past_window[j]
                if abs(cur_ts - past_ts) > max_residual_gap_ns or past_xyz.shape[0] == 0:
                    residuals.append(
                        np.zeros(
                            (params.range_image_h, params.range_image_w),
                            dtype=np.float32,
                        )
                    )
                    continue
                try:
                    pose_past = _interpolate_pose(pose_samples, past_ts, max_gap_ns)
                except _PoseGapError:
                    residuals.append(
                        np.zeros(
                            (params.range_image_h, params.range_image_w),
                            dtype=np.float32,
                        )
                    )
                    continue
                residuals.append(
                    _compute_residual(
                        past_xyz,
                        pose_cur,
                        pose_past,
                        ego_T_lidar,
                        params.range_image_h,
                        params.range_image_w,
                        params.fov_up_deg,
                        params.fov_down_deg,
                        range_img[0],
                        min_range_m=params.min_range_m,
                        max_range_m=params.max_range_m,
                    )
                )

            try:
                score_img = model.infer(
                    range_image=range_img, residual_images=residuals
                )  # (H, W) float32
            except Exception as exc:  # noqa: BLE001
                log.exception(
                    "sweep %d: MF-MOS inference failed; writing zero mask", sid
                )
                _write_zero_mask(bag_id, chunk_id, sid, n_raw, params.save_scores)
                _record_meta_path(
                    meta_by_sid, sid, mf_mos_mask_path(bag_id, chunk_id, sid)
                )
                result.n_sweeps_skipped_inference_error += 1
                result.skip_reasons.append((sid, f"infer: {type(exc).__name__}: {exc}"))
                past_window = (past_window + [(cur_ts, xyz_cur)])[-max_k:]
                continue

            pixel_mask = score_img >= params.score_threshold
            mf_finite_mask = _unproject_mask(pixel_mask, point_to_pixel, n_finite)

            # Re-expand to n_raw length so the saved mask is index-aligned
            # with the raw NPZ. NaN/inf points stay False.
            full_mask = np.zeros(n_raw, dtype=bool)
            full_mask[finite] = mf_finite_mask

            assert (
                full_mask.shape == (n_raw,)
            ), f"mf_mos mask len {full_mask.shape} != raw sweep len {n_raw} for sweep {sid}"

            np.save(local_path(mf_mos_mask_path(bag_id, chunk_id, sid)), full_mask)

            if params.save_scores:
                mf_finite_scores = _unproject_scores(
                    score_img, point_to_pixel, n_finite
                )
                full_scores = np.zeros(n_raw, dtype=np.float32)
                full_scores[finite] = mf_finite_scores
                np.save(
                    local_path(mf_mos_score_path(bag_id, chunk_id, sid)), full_scores
                )

            _record_meta_path(meta_by_sid, sid, mf_mos_mask_path(bag_id, chunk_id, sid))
            result.n_sweeps_processed += 1
            result.n_points_total += n_raw
            result.n_points_moving += int(full_mask.sum())

            past_window = (past_window + [(cur_ts, xyz_cur)])[-max_k:]

    updated = [meta_by_sid[int(r["sweep_id"])] for r in meta_rows]
    write_table(
        updated, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(bag_id, chunk_id)
    )

    log.info(
        "chunk %s: mf_mos processed=%d skipped(invalid=%d pose=%d empty=%d infer=%d) "
        "moving=%d/%d",
        chunk_id,
        result.n_sweeps_processed,
        result.n_sweeps_skipped_invalid,
        result.n_sweeps_skipped_pose,
        result.n_sweeps_skipped_empty,
        result.n_sweeps_skipped_inference_error,
        result.n_points_moving,
        result.n_points_total,
    )
    return result


def _load_model(params: MFMosParams) -> "MFMosModel":
    """Lazy load + cache the model. torch is imported inside _runtime."""
    key = (
        params.checkpoint_path,
        params.arch_config,
        params.data_config,
        params.device,
    )
    if key not in _MODEL_CACHE:
        from ._runtime import MFMosModel  # noqa: PLC0415

        _MODEL_CACHE[key] = MFMosModel(
            checkpoint_path=params.checkpoint_path,
            arch_cfg=params.arch_config,
            data_cfg=params.data_config,
            device=params.device,
        )
    return _MODEL_CACHE[key]


def _range_project(
    points_xyz_sensor: np.ndarray,
    intensity: np.ndarray | None,
    h: int,
    w: int,
    fov_up_deg: float,
    fov_down_deg: float,
    min_range_m: float = 0.0,
    max_range_m: float = float("inf"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spherical range-image projection.

    Args:
        points_xyz_sensor: (N, 3) float32, sensor frame.
        intensity: (N,) float32 or None.
        h, w: output image dimensions.
        fov_up_deg, fov_down_deg: vertical field of view (fov_down is negative).

    Returns:
        range_image: (5, H, W) float32, channels [range, x, y, z, intensity].
            Empty pixels have range=-1.0 and xyz/intensity=0.0.
        pixel_to_point_idx: (H, W) int32. Index of the closest-range point that
            won each pixel; -1 for empty pixels.
        point_to_pixel: (N, 2) int32, [row, col] for each input point.
            [-1, -1] for points outside the FOV or with zero/NaN range.
    """
    n = points_xyz_sensor.shape[0]

    range_image = np.full((5, h, w), 0.0, dtype=np.float32)
    range_image[0] = -1.0  # sentinel for empty pixels in range channel
    pixel_to_point_idx = np.full((h, w), -1, dtype=np.int32)
    point_to_pixel = np.full((n, 2), -1, dtype=np.int32)

    if n == 0:
        return range_image, pixel_to_point_idx, point_to_pixel

    x = points_xyz_sensor[:, 0].astype(np.float64)
    y = points_xyz_sensor[:, 1].astype(np.float64)
    z = points_xyz_sensor[:, 2].astype(np.float64)
    r = np.sqrt(x**2 + y**2 + z**2)

    valid = (r >= min_range_m) & (r <= max_range_m)
    r_safe = np.where(r > 1e-6, r, 1.0)

    yaw = -np.arctan2(y, x)
    pitch = np.arcsin(np.clip(z / r_safe, -1.0, 1.0))

    fov_up = np.deg2rad(fov_up_deg)
    fov_down = np.deg2rad(fov_down_deg)
    fov = fov_up - fov_down

    proj_x = 0.5 * (yaw / np.pi + 1.0)  # [0, 1]
    proj_y = 1.0 - (pitch - fov_down) / fov  # [0, 1], top=0

    col = np.clip(np.floor(proj_x * w).astype(np.int32), 0, w - 1)
    row = np.clip(np.floor(proj_y * h).astype(np.int32), 0, h - 1)

    in_fov = valid & (proj_y >= 0.0) & (proj_y <= 1.0)

    # Sort descending so closer-range points overwrite farther ones.
    order = np.argsort(r)[::-1]
    col_s = col[order]
    row_s = row[order]
    r_s = r[order].astype(np.float32)
    x_s = x[order].astype(np.float32)
    y_s = y[order].astype(np.float32)
    z_s = z[order].astype(np.float32)
    infov_s = in_fov[order]

    write_mask = infov_s
    range_image[0][row_s[write_mask], col_s[write_mask]] = r_s[write_mask]
    range_image[1][row_s[write_mask], col_s[write_mask]] = x_s[write_mask]
    range_image[2][row_s[write_mask], col_s[write_mask]] = y_s[write_mask]
    range_image[3][row_s[write_mask], col_s[write_mask]] = z_s[write_mask]

    written_orig_idx = order[write_mask]
    pixel_to_point_idx[row_s[write_mask], col_s[write_mask]] = written_orig_idx.astype(
        np.int32
    )

    if intensity is not None:
        intens_s = intensity[order].astype(np.float32)
        range_image[4][row_s[write_mask], col_s[write_mask]] = intens_s[write_mask]

    # point_to_pixel records, per input point, the pixel its projection
    # landed in. Points that lost the closest-range tiebreak are still
    # recorded here even though pixel_to_point_idx no longer points at them.
    row_orig = np.full(n, -1, dtype=np.int32)
    col_orig = np.full(n, -1, dtype=np.int32)
    in_fov_idx = np.where(in_fov)[0]
    row_orig[in_fov_idx] = row[in_fov_idx]
    col_orig[in_fov_idx] = col[in_fov_idx]
    point_to_pixel[:, 0] = row_orig
    point_to_pixel[:, 1] = col_orig

    return range_image, pixel_to_point_idx, point_to_pixel


def _compute_residual(
    past_xyz_sensor: np.ndarray,
    world_T_ego_current: np.ndarray,
    world_T_ego_past: np.ndarray,
    ego_T_lidar: np.ndarray,
    h: int,
    w: int,
    fov_up_deg: float,
    fov_down_deg: float,
    current_range_channel: np.ndarray,
    min_range_m: float = 0.0,
    max_range_m: float = float("inf"),
) -> np.ndarray:
    """Project past scan into current sensor frame and return normalized range residual.

    Matches the upstream MF-MOS preprocessing (utils/gen_residual_images.py with
    `normalize: True`): residual = |range_cur - range_past_in_cur| / range_cur,
    valid only where BOTH scans have returns within [min_range, max_range].
    Zero elsewhere — same sentinel as the training data.
    """
    if past_xyz_sensor.shape[0] == 0:
        return np.zeros((h, w), dtype=np.float32)

    # past sensor frame → world → current sensor frame.
    lidar_T_ego = np.linalg.inv(ego_T_lidar)
    cur_lidar_T_past_lidar = (
        lidar_T_ego
        @ np.linalg.inv(world_T_ego_current)
        @ world_T_ego_past
        @ ego_T_lidar
    )
    R = cur_lidar_T_past_lidar[:3, :3].astype(np.float32)
    t = cur_lidar_T_past_lidar[:3, 3].astype(np.float32)
    past_in_current = (R @ past_xyz_sensor.T).T + t  # (N_past, 3) float32

    past_range_img, _, _ = _range_project(
        past_in_current,
        None,
        h,
        w,
        fov_up_deg,
        fov_down_deg,
        min_range_m=min_range_m,
        max_range_m=max_range_m,
    )
    past_range = past_range_img[0]  # (H, W)

    valid = (
        (current_range_channel >= min_range_m)
        & (current_range_channel <= max_range_m)
        & (past_range >= min_range_m)
        & (past_range <= max_range_m)
    )
    residual = np.zeros((h, w), dtype=np.float32)
    residual[valid] = (
        np.abs(current_range_channel[valid] - past_range[valid])
        / current_range_channel[valid]
    )
    return residual


def _unproject_mask(
    pixel_mask: np.ndarray,
    point_to_pixel: np.ndarray,
    n_points: int,
) -> np.ndarray:
    """Map (H, W) bool pixel mask back to (N,) per-point bool.

    Points outside the FOV (point_to_pixel == -1) default to False.

    OCCLUSION CAVEAT: every in-FOV point inherits its pixel's label,
    including points that lost the closest-range tiebreak. A static
    surface directly behind a mover projecting to the same pixels will
    inherit the mover's label. Intrinsic to range-image MOS — relevant
    when investigating static structure showing up immediately behind a
    confirmed mover in the dynamic cloud.
    """
    out = np.zeros(n_points, dtype=bool)
    in_image = (point_to_pixel[:, 0] >= 0) & (point_to_pixel[:, 1] >= 0)
    idx_h = point_to_pixel[in_image, 0]
    idx_w = point_to_pixel[in_image, 1]
    out[in_image] = pixel_mask[idx_h, idx_w]
    return out


def _unproject_scores(
    score_image: np.ndarray,
    point_to_pixel: np.ndarray,
    n_points: int,
) -> np.ndarray:
    """Map (H, W) float32 score image back to (N,) float32 per-point scores.

    Points outside the FOV default to 0.0.
    """
    out = np.zeros(n_points, dtype=np.float32)
    in_image = (point_to_pixel[:, 0] >= 0) & (point_to_pixel[:, 1] >= 0)
    idx_h = point_to_pixel[in_image, 0]
    idx_w = point_to_pixel[in_image, 1]
    out[in_image] = score_image[idx_h, idx_w]
    return out


class _PoseGapError(RuntimeError):
    pass


def _interpolate_pose(
    samples: list[PoseSample],
    ts_ns: int,
    max_gap_ns: int,
) -> np.ndarray:
    """Return (4,4) world_T_ego at ts_ns, or raise _PoseGapError if the gap exceeds max_gap_ns."""
    try:
        poses = batch_interpolate_poses(samples, np.array([ts_ns], dtype=np.int64))
    except Exception as exc:
        raise _PoseGapError(f"pose interp at t={ts_ns}: {exc}") from exc
    closest_gap = min(abs(s.timestamp_ns - ts_ns) for s in samples)
    if closest_gap > max_gap_ns:
        raise _PoseGapError(
            f"pose gap {closest_gap / 1_000_000:.0f} ms > "
            f"{max_gap_ns / 1_000_000:.0f} ms at t={ts_ns}"
        )
    return poses[0]


def _write_zero_mask(
    bag_id: str,
    chunk_id: str,
    sweep_id: int,
    n_raw: int,
    save_scores: bool,
) -> None:
    np.save(
        local_path(mf_mos_mask_path(bag_id, chunk_id, sweep_id)),
        np.zeros(n_raw, dtype=bool),
    )
    if save_scores:
        np.save(
            local_path(mf_mos_score_path(bag_id, chunk_id, sweep_id)),
            np.zeros(n_raw, dtype=np.float32),
        )


def _record_meta_path(
    meta_by_sid: dict[int, dict],
    sweep_id: int,
    path: str | None,
) -> None:
    if sweep_id in meta_by_sid:
        meta_by_sid[sweep_id]["mf_mos_mask_path"] = path
