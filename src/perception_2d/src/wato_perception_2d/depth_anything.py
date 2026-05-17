"""Depth Anything V2 monocular depth estimator (and LiDAR-based metric alignment).

Per docs/research/depth_anything_guidance.md, this module runs DA V2 Large on
each camera frame and emits a (H, W) float32 relative-depth map.  A second
helper rescales that relative depth into metric units by least-squares-fitting
(scale, shift) against LiDAR points that project into the same image.  The
metric depth feeds proposal_generation's pseudo-LiDAR lift step.

Lazy-imports the DA V2 model code so the module can be imported (and tests
that don't need the model can pass) even when the depth-anything-v2 package
is absent.  Checkpoint is loaded from MODELS_ROOT/depth_anything_v2/.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from wato_common.artifact_store import detector_checkpoint_path

log = logging.getLogger(__name__)

_warned_missing = False


@dataclass
class DepthAlignmentResult:
    """Output of `align_depth_to_lidar` — what gets persisted to depth_index.parquet."""

    scale: float
    shift: float
    n_overlap_pts: int
    residual_rmse: Optional[float]
    scale_method: str   # "lidar_aligned" | "metric_native" | "uncalibrated"


class DepthAnythingV2Estimator:
    """Depth Anything V2 wrapper with lazy model load.

    Default checkpoint: $MODELS_ROOT/depth_anything_v2/depth_anything_v2_vitl.pth
    (download with `watod fetch-models`).  Model is loaded on first
    predict() call and reused thereafter — build one estimator per process,
    not per frame.

    Output is *relative* depth.  Pair with `align_depth_to_lidar` per frame
    to convert into metric depth.
    """

    DEFAULT_FILENAME = "depth_anything_v2_vitl.pth"

    # ViT-L config from the upstream DA V2 repo.  Hardcoded because we only
    # ship the Large variant; switching variants is a separate decision.
    _CONFIG_LARGE = {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    }

    def __init__(
        self,
        model_size: str = "large",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        if model_size != "large":
            raise ValueError(
                f"only model_size='large' is supported in v1 (got {model_size!r})"
            )
        self._model_size = model_size
        self._checkpoint_path = checkpoint_path or detector_checkpoint_path(
            "depth_anything_v2", self.DEFAULT_FILENAME
        )
        self._device = device or self._default_device()
        self._model = None     # lazy-loaded on first predict() call

    @staticmethod
    def _default_device() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load(self) -> bool:
        """Lazy-load model.  Returns False if unavailable."""
        global _warned_missing
        if self._model is not None:
            return True
        try:
            import torch
            from depth_anything_v2.dpt import DepthAnythingV2
            if not os.path.exists(self._checkpoint_path):
                raise FileNotFoundError(
                    f"DA V2 checkpoint not found at {self._checkpoint_path}; "
                    "run `watod fetch-models` to populate MODELS_ROOT."
                )
            self._model = DepthAnythingV2(**self._CONFIG_LARGE)
            state = torch.load(self._checkpoint_path, map_location="cpu")
            self._model.load_state_dict(state)
            self._model = self._model.to(self._device).eval()
            log.info("Depth Anything V2 (%s) loaded on %s", self._model_size, self._device)
            return True
        except Exception as exc:  # noqa: BLE001
            if not _warned_missing:
                log.warning(
                    "Depth Anything V2 unavailable (%s) — depth maps will be "
                    "skipped for this run. Install: depth-anything-v2 + run "
                    "`watod fetch-models`.",
                    exc,
                )
                _warned_missing = True
            return False

    def predict(self, image_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Run DA V2 on an (H, W, 3) uint8 RGB image.

        Returns (H, W) float32 relative depth, or None if the model is not
        available (caller should skip writing a depth artifact for this frame).
        """
        if not self._load():
            return None
        import torch  # noqa: F401 — needed by the model's forward, also confirms torch is present
        # DA V2's `infer_image` accepts an HWC BGR uint8 array (the upstream
        # entrypoint applies its own preprocessing internally).  Convert RGB→BGR.
        bgr = image_rgb[:, :, ::-1].copy()
        depth = self._model.infer_image(bgr)
        return np.asarray(depth, dtype=np.float32)


def align_depth_to_lidar(
    depth_relative: np.ndarray,
    lidar_world_pts: np.ndarray,
    K: np.ndarray,
    cam_T_world: np.ndarray,
    min_overlap_pts: int = 50,
    inlier_thresh_m: float = 0.5,
    max_iters: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> DepthAlignmentResult:
    """Least-squares fit ``z_metric = scale * depth_relative + shift`` against LiDAR.

    Projects ``lidar_world_pts`` into the image using ``K`` and ``cam_T_world``,
    samples the predicted relative depth at each projected pixel, then runs a
    simple RANSAC to drop outliers (LiDAR points hitting sky / cars in front /
    other camera-disagreement pixels) before a final least-squares fit on the
    inlier set.

    Conventions:
      * `cam_T_world` is the 4x4 transform that takes world → camera.
      * Pinhole, no distortion — distorted images should have been rectified
        upstream (perception_2d's calibration uses rectified frames).
      * Returns a `DepthAlignmentResult` with scale=1.0 / shift=0.0 and
        `scale_method="uncalibrated"` when overlap is too small to fit.

    The diagnostic fields (`residual_rmse`, `n_overlap_pts`) end up in
    `depth_index.parquet` for the downstream pseudo-LiDAR lift step and the
    cross-modal uncertainty score.
    """
    if depth_relative.ndim != 2:
        raise ValueError(f"depth_relative must be (H, W); got shape {depth_relative.shape}")
    H, W = depth_relative.shape

    if lidar_world_pts.shape[0] == 0:
        return DepthAlignmentResult(1.0, 0.0, 0, None, "uncalibrated")

    # Project world points into camera frame.
    pts_hom = np.concatenate(
        [lidar_world_pts, np.ones((lidar_world_pts.shape[0], 1), dtype=lidar_world_pts.dtype)],
        axis=1,
    )
    pts_cam = (cam_T_world @ pts_hom.T).T[:, :3]
    z_cam = pts_cam[:, 2]
    in_front = z_cam > 0.1
    if not in_front.any():
        return DepthAlignmentResult(1.0, 0.0, 0, None, "uncalibrated")
    pts_cam = pts_cam[in_front]

    pix = (K @ pts_cam.T).T
    pix[:, 0] /= pix[:, 2]
    pix[:, 1] /= pix[:, 2]
    u = pix[:, 0].astype(np.int32)
    v = pix[:, 1].astype(np.int32)
    in_image = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if in_image.sum() < min_overlap_pts:
        return DepthAlignmentResult(1.0, 0.0, int(in_image.sum()), None, "uncalibrated")

    u = u[in_image]
    v = v[in_image]
    z_lidar = pts_cam[in_image, 2].astype(np.float64)
    d_rel = depth_relative[v, u].astype(np.float64)

    # Drop zero/negative relative depth samples (DA can produce them at sky pixels).
    keep = d_rel > 0
    if keep.sum() < min_overlap_pts:
        return DepthAlignmentResult(1.0, 0.0, int(keep.sum()), None, "uncalibrated")
    z_lidar = z_lidar[keep]
    d_rel = d_rel[keep]
    n = z_lidar.shape[0]

    # RANSAC over (scale, shift) pairs.  Two-sample minimum set.
    rng = rng if rng is not None else np.random.default_rng(0)
    best_inliers = np.zeros(n, dtype=bool)
    best_count = 0
    for _ in range(max_iters):
        i, j = rng.choice(n, size=2, replace=False)
        if abs(d_rel[i] - d_rel[j]) < 1e-6:
            continue
        s = (z_lidar[i] - z_lidar[j]) / (d_rel[i] - d_rel[j])
        b = z_lidar[i] - s * d_rel[i]
        residuals = np.abs(z_lidar - (s * d_rel + b))
        inliers = residuals < inlier_thresh_m
        n_in = int(inliers.sum())
        if n_in > best_count:
            best_count = n_in
            best_inliers = inliers

    if best_count < min_overlap_pts:
        # Fall back to a least-squares fit over all samples — better than nothing.
        best_inliers = np.ones(n, dtype=bool)
        best_count = n

    A = np.vstack([d_rel[best_inliers], np.ones(best_count)]).T
    coeffs, *_ = np.linalg.lstsq(A, z_lidar[best_inliers], rcond=None)
    scale, shift = float(coeffs[0]), float(coeffs[1])

    residuals = z_lidar - (scale * d_rel + shift)
    rmse = float(np.sqrt(np.mean(residuals[best_inliers] ** 2)))

    method = "lidar_aligned" if best_count >= min_overlap_pts else "uncalibrated"
    return DepthAlignmentResult(
        scale=scale,
        shift=shift,
        n_overlap_pts=int(best_count),
        residual_rmse=rmse,
        scale_method=method,
    )
