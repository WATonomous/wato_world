"""LiDAR-anchored affine alignment for Depth Anything V2 relative depth maps.

Solves d_lidar = a * d_da + b via RANSAC, then applies the fit to produce
a metric depth map aligned to LiDAR scale.

IMPORTANT: only static LiDAR points should be passed as anchors.  Dynamic
points cause temporal-offset errors because cameras and LiDARs are not
hardware-synchronized (~25 ms offset at 12 Hz camera / 20 Hz LiDAR).
A car moving at 30 m/s travels 75 cm in 25 ms — using its points as depth
anchors would corrupt the affine fit.  Use dynamic_mask from
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
    relative_depth: np.ndarray,
    K: np.ndarray,
    cam_T_world: np.ndarray,
    image_size: tuple[int, int],
    sky_mask_top_fraction: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Project static LiDAR points into the image and collect (d_lidar, d_da) pairs.

    Args:
        lidar_world_pts: (N, 3) static-only world-frame LiDAR points.
        relative_depth: (H, W) float32 DA relative depth map.
        K: (3, 3) intrinsic matrix.
        cam_T_world: (4, 4) SE3 — world → camera transform.
        image_size: (width, height).
        sky_mask_top_fraction: discard points in the top X% of the image
            (DA-V2 is unreliable for sky pixels).

    Returns:
        d_lidar: (M,) float32 — camera-frame depths of projected LiDAR points.
        d_da: (M,) float32 — DA relative depth at the same pixels.
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
    d_da = relative_depth[v_idx, u_idx].astype(np.float32)

    # Filter out DA values that are zero or negative (model artifacts).
    good = (d_da > 0) & (d_lidar > 0)
    return d_lidar[good], d_da[good]


def ransac_affine_fit(
    d_lidar: np.ndarray,
    d_da: np.ndarray,
    n_iter: int = 200,
    inlier_threshold_m: float = 0.5,
    min_depth_range_m: float = 5.0,
    fallback: Optional[tuple[float, float]] = None,
    min_anchors: int = 2,
) -> dict:
    """Fit d_lidar = a * d_da + b via RANSAC.

    Args:
        d_lidar: (M,) float32 LiDAR camera-frame depths.
        d_da: (M,) float32 DA relative depths (same pixel, same order).
        n_iter: number of RANSAC iterations.
        inlier_threshold_m: max residual |d_lidar - (a*d_da + b)| to be inlier.
        min_depth_range_m: if LiDAR depth range < this, fit is underdetermined;
            use fallback or fail.
        fallback: (a, b) from a prior successful frame, or None.
        min_anchors: minimum number of anchor pairs required to attempt a fit;
            below this the fit is treated as underdetermined (fallback or fail).
            2 is the absolute floor for a 2-point fit; the pipeline passes the
            configured depth.min_lidar_anchors.

    Returns:
        dict with keys: a, b, n_inliers, rmse_inliers_m, fit_status.
        fit_status: 0=ok, 1=used fallback, 2=failed entirely.
    """
    M = d_lidar.shape[0]

    # Need enough anchors AND enough depth range for a reliable fit.
    if (
        M < max(2, min_anchors)
        or float(d_lidar.max() - d_lidar.min()) < min_depth_range_m
    ):
        if fallback is not None:
            a, b = fallback
            log.debug(
                "ransac_affine_fit: insufficient depth range — using fallback (a=%.3f, b=%.3f)",
                a,
                b,
            )
            return _eval_fit(a, b, d_lidar, d_da, fit_status=1)
        log.debug("ransac_affine_fit: insufficient depth range and no fallback")
        return {
            "a": 1.0,
            "b": 0.0,
            "n_inliers": 0,
            "rmse_inliers_m": float("inf"),
            "fit_status": 2,
        }

    best_n_inliers = 0
    best_a, best_b = 1.0, 0.0
    rng = np.random.default_rng(0)

    for _ in range(n_iter):
        idx = rng.choice(M, size=2, replace=False)
        da0, da1 = d_da[idx[0]], d_da[idx[1]]
        dl0, dl1 = d_lidar[idx[0]], d_lidar[idx[1]]
        if abs(da1 - da0) < 1e-6:
            continue
        a = (dl1 - dl0) / (da1 - da0)
        b = dl0 - a * da0
        if a <= 0:
            continue
        residuals = np.abs(d_lidar - (a * d_da + b))
        n_inliers = int((residuals < inlier_threshold_m).sum())
        if n_inliers > best_n_inliers:
            best_n_inliers = n_inliers
            best_a, best_b = a, b

    if best_n_inliers < 4:
        if fallback is not None:
            a, b = fallback
            return _eval_fit(a, b, d_lidar, d_da, fit_status=1)
        return {
            "a": 1.0,
            "b": 0.0,
            "n_inliers": 0,
            "rmse_inliers_m": float("inf"),
            "fit_status": 2,
        }

    # Refit on all inliers via least-squares for best accuracy.
    inlier_mask = np.abs(d_lidar - (best_a * d_da + best_b)) < inlier_threshold_m
    A = np.column_stack([d_da[inlier_mask], np.ones(inlier_mask.sum())])
    result = np.linalg.lstsq(A, d_lidar[inlier_mask], rcond=None)
    best_a, best_b = float(result[0][0]), float(result[0][1])

    return _eval_fit(best_a, best_b, d_lidar, d_da, fit_status=0)


def _eval_fit(
    a: float, b: float, d_lidar: np.ndarray, d_da: np.ndarray, fit_status: int
) -> dict:
    if d_lidar.shape[0] == 0:
        return {
            "a": a,
            "b": b,
            "n_inliers": 0,
            "rmse_inliers_m": float("inf"),
            "fit_status": fit_status,
        }
    residuals = d_lidar - (a * d_da + b)
    n_inliers = int(d_lidar.shape[0])
    rmse = float(np.sqrt(np.mean(residuals**2)))
    return {
        "a": a,
        "b": b,
        "n_inliers": n_inliers,
        "rmse_inliers_m": rmse,
        "fit_status": fit_status,
    }


def apply_affine(
    relative_depth: np.ndarray,
    a: float,
    b: float,
) -> np.ndarray:
    """Apply affine d_metric = a * d_da + b to the full depth map.

    Args:
        relative_depth: (H, W) float32 DA relative depth.
        a: scale factor.
        b: offset.

    Returns:
        metric_depth: (H, W) float16, clamped to [0, 250] m.
    """
    metric = (a * relative_depth.astype(np.float32) + b).clip(0, 250)
    return metric.astype(np.float16)
