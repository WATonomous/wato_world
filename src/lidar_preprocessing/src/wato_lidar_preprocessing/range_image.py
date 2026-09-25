"""Spherical range-image projection shared by MF-MOS and IWU.

Lives at the package top level, not under mf_mos/, because two independent
steps need it: MF-MOS (the learned `mos` half) builds its residual images
with it, and IWU (bag-level geometry, Step E) uses it to decide whether a map
point was seen, seen through, or occluded in a sweep. The aw/mos module rule
forbids geometry code importing from mf_mos/, so the projection is shared
from here instead.

Image geometry (rows, vertical FoV, azimuth width) is the caller's to supply
from the scanner's SensorModel — this module holds no sensor constants.
"""

from __future__ import annotations

import numpy as np


def spherical_pixels(
    points_xyz: np.ndarray,
    h: int,
    w: int,
    fov_up_deg: float,
    fov_down_deg: float,
    min_range_m: float = 0.0,
    max_range_m: float = float("inf"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-point spherical pixel coordinates.

    Args:
        points_xyz: (N, 3) coordinates relative to the projection centre
            (sensor frame for MF-MOS; origin-centred, world-axis-aligned for
            IWU).
        h, w: image rows / azimuth columns.
        fov_up_deg, fov_down_deg: vertical field of view (fov_down negative).
        min_range_m, max_range_m: points outside this range are not in FoV.

    Returns:
        row, col: (N,) int32 pixel indices (clipped into the image; only
            meaningful where in_fov).
        r: (N,) float64 range.
        in_fov: (N,) bool — inside the range window and the vertical FoV.
    """
    x = points_xyz[:, 0].astype(np.float64)
    y = points_xyz[:, 1].astype(np.float64)
    z = points_xyz[:, 2].astype(np.float64)
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
    return row, col, r, in_fov


def range_project(
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

    row, col, r, in_fov = spherical_pixels(
        points_xyz_sensor, h, w, fov_up_deg, fov_down_deg, min_range_m, max_range_m
    )
    x = points_xyz_sensor[:, 0]
    y = points_xyz_sensor[:, 1]
    z = points_xyz_sensor[:, 2]

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
    in_fov_idx = np.where(in_fov)[0]
    point_to_pixel[in_fov_idx, 0] = row[in_fov_idx]
    point_to_pixel[in_fov_idx, 1] = col[in_fov_idx]

    return range_image, pixel_to_point_idx, point_to_pixel


def min_range_image(
    points_xyz: np.ndarray,
    h: int,
    w: int,
    fov_up_deg: float,
    fov_down_deg: float,
    max_range_m: float = float("inf"),
) -> np.ndarray:
    """(H, W) float64 image of the closest return per pixel; +inf where empty.

    The occlusion-test primitive: a scene point at range r behind pixel range
    R (r > R + tol) is occluded; in front of it (r < R - tol) it was seen
    through.
    """
    img = np.full((h, w), np.inf, dtype=np.float64)
    if points_xyz.shape[0] == 0:
        return img
    row, col, r, in_fov = spherical_pixels(
        points_xyz, h, w, fov_up_deg, fov_down_deg, 0.0, max_range_m
    )
    np.minimum.at(img, (row[in_fov], col[in_fov]), r[in_fov])
    return img
