"""LiDAR-anchored metric alignment for Depth Anything V2 disparity maps.

Depth Anything V2's relative checkpoints output affine-invariant *disparity*
(inverse depth: larger = closer), not depth. Metric depth is therefore
recovered in inverse-depth space,

    1 / d_metric = scale * disparity + shift

solved via RANSAC against LiDAR depths, then inverted to a metric depth map.
Fitting depth itself as an affine of disparity is the wrong model: the true
relation has a negative, hyperbolic slope, so no positive-scale affine fits it.

IMPORTANT: only static LiDAR points should be passed as anchors.  Dynamic
points cause temporal-offset errors because cameras and LiDARs are not
hardware-synchronized (~25 ms offset at 12 Hz camera / 20 Hz LiDAR).
A car moving at 30 m/s travels 75 cm in 25 ms — using its points as depth
anchors would corrupt the fit.  Use dynamic_mask from
lidar_preprocessing to filter them out before calling build_anchor_pairs().
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from wato_common.geometry import project_lidar_to_image

log = logging.getLogger(__name__)


def build_anchor_pairs(
    lidar_world_pts: np.ndarray,
    disparity: np.ndarray,
    K: np.ndarray,
    cam_T_world: np.ndarray,
    image_size: tuple[int, int],
    sky_mask_top_fraction: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Project static LiDAR points into the image and collect (d_lidar, disp) pairs.

    Args:
        lidar_world_pts: (N, 3) static-only world-frame LiDAR points.
        disparity: (H, W) float32 DA-V2 relative disparity map.
        K: (3, 3) intrinsic matrix.
        cam_T_world: (4, 4) SE3 — world → camera transform.
        image_size: (width, height).
        sky_mask_top_fraction: discard points in the top X% of the image
            (DA-V2 is unreliable for sky pixels).

    Returns:
        d_lidar: (M,) float32 — camera-frame depths of projected LiDAR points.
        disp: (M,) float32 — DA-V2 disparity at the same pixels.
    """
    if lidar_world_pts is None or lidar_world_pts.shape[0] == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    W, H = image_size
    sky_cutoff = int(H * sky_mask_top_fraction)

    uv, depth_cam, valid = project_lidar_to_image(
        lidar_world_pts, K, cam_T_world, image_size
    )

    if not valid.any():
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    uv_valid = uv[valid].astype(np.int32)
    d_lidar = depth_cam[valid].astype(np.float32)

    # Sky filter: remove points whose projected pixel is in the upper band.
    not_sky = uv_valid[:, 1] >= sky_cutoff
    uv_valid = uv_valid[not_sky]
    d_lidar = d_lidar[not_sky]

    if uv_valid.shape[0] == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    # Clamp to image bounds (projection may leave edge pixels at exactly W or H).
    u_idx = np.clip(uv_valid[:, 0], 0, W - 1)
    v_idx = np.clip(uv_valid[:, 1], 0, H - 1)
    disp = disparity[v_idx, u_idx].astype(np.float32)

    # Zero / negative disparity is sky or a model artifact — no usable depth.
    good = (disp > 0) & (d_lidar > 0)
    return d_lidar[good], disp[good]


def _depth_residuals(
    d_lidar: np.ndarray, disp: np.ndarray, scale: float, shift: float
) -> np.ndarray:
    """|d_lidar - 1/(scale*disp + shift)| in metres; inf where the model has no
    positive depth (non-positive inverse depth)."""
    inv = scale * disp + shift
    pred = np.divide(1.0, inv, out=np.full_like(inv, np.inf), where=inv > 0)
    return np.abs(d_lidar - pred)


def _failed() -> dict:
    return {
        "scale": 1.0,
        "shift": 0.0,
        "n_inliers": 0,
        "rmse_inliers_m": float("inf"),
        "fit_status": 2,
    }


def ransac_disparity_fit(
    d_lidar: np.ndarray,
    disp: np.ndarray,
    n_iter: int = 200,
    inlier_threshold_m: float = 0.5,
    min_depth_range_m: float = 5.0,
    min_anchors: int = 2,
    fallback: Optional[tuple[float, float]] = None,
) -> dict:
    """Fit 1/d_lidar = scale * disp + shift via RANSAC.

    Hypotheses come from 2-point samples in inverse-depth space; inliers are
    judged by the residual in *metres* after inverting, so the threshold means
    the same thing near and far. The final refit is least squares on the
    inliers in inverse-depth space with each row weighted by d_lidar**2, which
    to first order minimises the metric residual (d(1/z) = -dz / z**2).

    Args:
        d_lidar: (M,) float32 LiDAR camera-frame depths.
        disp: (M,) float32 DA-V2 disparities (same pixel, same order).
        n_iter: number of RANSAC iterations.
        inlier_threshold_m: max |d_lidar - 1/(scale*disp + shift)| to be inlier.
        min_depth_range_m: if LiDAR depth range < this, fit is underdetermined;
            use fallback or fail.
        min_anchors: require at least this many anchor pairs to attempt a fit
            (clamped to >= 2, the minimum for the 2-point RANSAC sample);
            fewer → use fallback or fail.
        fallback: (scale, shift) from a prior successful frame, or None.

    Returns:
        dict with keys: scale, shift, n_inliers, rmse_inliers_m, fit_status.
        fit_status: 0=ok, 1=used fallback, 2=failed entirely.
    """
    M = d_lidar.shape[0]

    def _use_fallback(reason: str) -> dict:
        if fallback is None:
            log.debug("ransac_disparity_fit: %s and no fallback", reason)
            return _failed()
        scale, shift = fallback
        log.debug(
            "ransac_disparity_fit: %s — using fallback (scale=%.4g, shift=%.4g)",
            reason,
            scale,
            shift,
        )
        return _eval_fit(scale, shift, d_lidar, disp, inlier_threshold_m, fit_status=1)

    if M < max(2, min_anchors) or float(d_lidar.max() - d_lidar.min()) < (
        min_depth_range_m
    ):
        return _use_fallback("insufficient anchors or depth range")

    inv_lidar = 1.0 / d_lidar.astype(np.float64)
    disp64 = disp.astype(np.float64)

    best_n_inliers = 0
    best_scale, best_shift = 1.0, 0.0
    rng = np.random.default_rng(0)

    for _ in range(n_iter):
        i0, i1 = rng.choice(M, size=2, replace=False)
        dx = disp64[i1] - disp64[i0]
        if abs(dx) < 1e-9:
            continue
        scale = (inv_lidar[i1] - inv_lidar[i0]) / dx
        # Disparity grows as depth shrinks: inverse depth must rise with it.
        if scale <= 0:
            continue
        shift = inv_lidar[i0] - scale * disp64[i0]
        n_inliers = int(
            (_depth_residuals(d_lidar, disp64, scale, shift) < inlier_threshold_m).sum()
        )
        if n_inliers > best_n_inliers:
            best_n_inliers = n_inliers
            best_scale, best_shift = scale, shift

    if best_n_inliers < 4:
        return _use_fallback("fewer than 4 RANSAC inliers")

    # Refit on the inliers: depth-squared-weighted LS in inverse-depth space.
    inl = _depth_residuals(d_lidar, disp64, best_scale, best_shift) < inlier_threshold_m
    w = d_lidar[inl].astype(np.float64) ** 2
    A = np.column_stack([disp64[inl], np.ones(int(inl.sum()))]) * w[:, None]
    (scale, shift), *_ = np.linalg.lstsq(A, inv_lidar[inl] * w, rcond=None)
    if scale <= 0:
        return _use_fallback("refit produced a non-positive scale")

    return _eval_fit(
        float(scale), float(shift), d_lidar, disp, inlier_threshold_m, fit_status=0
    )


def _eval_fit(
    scale: float,
    shift: float,
    d_lidar: np.ndarray,
    disp: np.ndarray,
    inlier_threshold_m: float,
    fit_status: int,
) -> dict:
    """Score (scale, shift) on this frame's anchors: inlier count + inlier RMSE."""
    res = _depth_residuals(
        d_lidar.astype(np.float64), disp.astype(np.float64), scale, shift
    )
    inl = res < inlier_threshold_m
    n_inliers = int(inl.sum())
    rmse = float(np.sqrt(np.mean(res[inl] ** 2))) if n_inliers else float("inf")
    return {
        "scale": scale,
        "shift": shift,
        "n_inliers": n_inliers,
        "rmse_inliers_m": rmse,
        "fit_status": fit_status,
    }


def disparity_to_depth(
    disparity: np.ndarray,
    scale: float,
    shift: float,
    out_dtype: str = "float16",
    max_depth_m: float = 250.0,
) -> np.ndarray:
    """Metric depth d = 1 / (scale * disparity + shift) over the whole map.

    Args:
        disparity: (H, W) float32 DA-V2 disparity.
        scale, shift: the inverse-depth fit from ransac_disparity_fit.
        out_dtype: numpy dtype name for the stored depth map (default float16).
            Unknown names fall back to float16.
        max_depth_m: depths beyond this (incl. sky, where disparity -> 0) are
            clamped to it.

    Returns:
        metric_depth: (H, W) out_dtype array in [0, max_depth_m]. Pixels whose
        fitted inverse depth is non-positive get 0, the depth artifact's
        "no measurement" value.
    """
    try:
        dtype = np.dtype(out_dtype)
    except TypeError:
        dtype = np.dtype(np.float16)
    inv = scale * disparity.astype(np.float32) + shift
    metric = np.zeros_like(inv)
    pos = inv > 0
    metric[pos] = np.minimum(1.0 / inv[pos], max_depth_m)
    return metric.astype(dtype)
