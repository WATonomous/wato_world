"""BEVFusion through the fork's pybind11 module (libpybev).

libpybev is built into the image (docker/detection_3d.Dockerfile) and
imported only here, so the rest of the package and its tests run without a
GPU. One instance per process: the vendor core is not thread-safe.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from wato_detection_3d.engines import Engines
from wato_detection_3d.inputs import CameraInputs

# Columns of the (M, 11) array libpybev's forward returns.
X, Y, Z, W, L, H, ROT, VX, VY, CLASS_ID, SCORE = range(11)


class Detector(Protocol):
    def infer(self, points: np.ndarray, cameras: CameraInputs) -> np.ndarray:
        """(M, 11) float32 boxes (columns above) for one frame."""


class Bevfusion:
    def __init__(self, engines: Engines, precision: str, num_cameras: int = 6):
        import libpybev

        self._num_cameras = num_cameras
        self._core = libpybev.load_bevfusion(
            engines.camera_backbone,
            engines.camera_vtransform,
            engines.lidar_backbone,
            engines.fuser,
            engines.head_bbox,
            precision,
            num_cameras,
        )
        if self._core is None:
            raise RuntimeError(
                f"libpybev failed to load the engines in {engines.directory}"
            )

    def infer(self, points: np.ndarray, cameras: CameraInputs) -> np.ndarray:
        if cameras.images.shape != (self._num_cameras, 900, 1600, 3):
            raise ValueError(f"expected {self._num_cameras} camera images")
        for matrix in (
            cameras.camera2lidar,
            cameras.intrinsics,
            cameras.lidar2image,
            cameras.img_aug,
        ):
            if matrix.shape != (self._num_cameras, 4, 4):
                raise ValueError(
                    f"expected calibration for {self._num_cameras} cameras"
                )
        # The calibration moves with the ego between each image's time and
        # the frame's, so it is uploaded per frame.
        self._core.update(
            cameras.camera2lidar[None],
            cameras.intrinsics[None],
            cameras.lidar2image[None],
            cameras.img_aug[None],
        )
        out = self._core.forward(
            np.ascontiguousarray(cameras.images[None]),
            np.ascontiguousarray(points.astype(np.float16)),
            True,
            False,
        )
        return np.asarray(out, dtype=np.float32).reshape(-1, 11)
