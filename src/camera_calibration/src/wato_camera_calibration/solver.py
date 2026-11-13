"""One file-based Koide backend, with bounded execution and reproducible starts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from wato_common.artifacts.calibration import CameraCalibration
from wato_common.artifacts.calibration_refinement import atomic_json, sha256
from wato_common.geometry.camera_model import validate_transform

from wato_camera_calibration.config import SolverConfig
from wato_camera_calibration.inputs import Sample

REVISION = "02a0dc039f5509708f384be4ff3228e0ae09352d"


def backend_identity(cfg: SolverConfig) -> dict:
    command = cfg.executable or os.environ.get(
        "WATO_KOIDE_EXECUTABLE", "wato_koide_calibrate"
    )
    path = shutil.which(command)
    if path is None:
        raise RuntimeError(
            f"Koide runner {command!r} is missing; build the camera_calibration image"
        )
    version = subprocess.run(
        [path, "--version"], capture_output=True, text=True, check=True, timeout=10
    ).stdout.strip()
    if version != f"koide-nid-headless {REVISION}":
        raise RuntimeError(f"unexpected calibration backend: {version!r}")
    packages = Path("/opt/watonomous/calibration_build/apt-packages.txt")
    return {
        "executable": path,
        "version": version,
        "sha256": sha256(path),
        "native_packages_sha256": sha256(packages) if packages.exists() else None,
    }


def camera_parameters(cam: CameraCalibration) -> dict:
    model = cam.model
    if not np.allclose(model.K[[0, 1], [1, 0]], 0, atol=1e-12, rtol=0):
        raise ValueError("Koide's camera projection cannot represent nonzero K skew")
    name = model.distortion_model
    distortion = list(model.distortion)
    if name in ("", "none", "plumb_bob"):
        name = "plumb_bob"
        distortion += [0.0] * (5 - len(distortion))
    return {
        "model": name,
        "intrinsics": [
            float(model.K[0, 0]),
            float(model.K[1, 1]),
            float(model.K[0, 2]),
            float(model.K[1, 2]),
        ],
        "distortion": distortion,
    }


def prepare(samples: list[Sample], directory: Path) -> list[dict]:
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, sample in enumerate(samples):
        image_path = (directory / f"{i:04d}.gray").resolve()
        points_path = (directory / f"{i:04d}.bin").resolve()
        sample.gray.astype("u1").tofile(image_path)
        np.column_stack((sample.xyz, sample.intensity)).astype("<f8").tofile(
            points_path
        )
        rows.append(
            {
                "image": str(image_path),
                "width": sample.gray.shape[1],
                "height": sample.gray.shape[0],
                "points": str(points_path),
                "timestamp_ns": sample.timestamp_ns,
                "block": sample.block,
                "split": sample.split,
            }
        )
    atomic_json(directory / "samples.json", {"samples": rows})
    return rows


def fit(
    cam: CameraCalibration,
    pairs: list[dict],
    initial: np.ndarray,
    cfg: SolverConfig,
    identity: dict,
    directory: Path,
) -> tuple[np.ndarray, dict]:
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        **camera_parameters(cam),
        "pairs": pairs,
        "initial_ego_T_cam": initial.tolist(),
        "reference_ego_T_cam": cam.ego_T_cam.tolist(),
        **cfg.model_dump(
            exclude={"executable", "starts", "subsets", "perturb_rotation_deg"}
        ),
    }
    atomic_json(directory / "input.json", config)
    output = directory / "result.json"
    output.unlink(missing_ok=True)
    with (directory / "backend.log").open("w") as log:
        subprocess.run(
            [identity["executable"], str(directory / "input.json"), str(output)],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=cfg.timeout_s + 15,
        )
    result = json.loads(output.read_text())
    if result.get("revision") != REVISION:
        raise ValueError("solver result revision mismatch")
    if result.get("motion_model") != cfg.motion_model:
        raise ValueError(
            "backend motion model mismatch; rebuild the camera_calibration image"
        )
    transform = validate_transform(np.asarray(result["ego_T_cam"], dtype=np.float64))
    if cfg.motion_model == "rotation_only" and not np.allclose(
        transform[:3, 3], cam.ego_T_cam[:3, 3], atol=1e-12, rtol=0
    ):
        raise ValueError("rotation-only backend changed the fixed camera centre")
    return transform, result


def run_fits(
    cam: CameraCalibration,
    pairs: list[dict],
    cfg: SolverConfig,
    identity: dict,
    directory: Path,
):
    train = [r for r in pairs if r["split"] == "train"]
    if not train:
        raise ValueError("no training pairs")
    transforms, results = [], []
    for i in range(cfg.starts):
        initial = cam.ego_T_cam.copy()
        if i:
            rotation = np.zeros(3)
            rotation[(i - 1) % 3] = cfg.perturb_rotation_deg * (-1 if i % 2 else 1)
            initial[:3, :3] = (
                initial[:3, :3] @ Rotation.from_rotvec(np.deg2rad(rotation)).as_matrix()
            )
        T, result = fit(cam, train, initial, cfg, identity, directory / f"start_{i}")
        transforms.append(T)
        results.append(result)
    # Select by training objective only. Validation/test never choose the fit.
    selected = int(np.argmin([r["history"][-1]["final_cost"] for r in results]))
    blocks = sorted({r["block"] for r in train})
    if len(blocks) < cfg.subsets:
        raise ValueError("too few independent training blocks for subset stability")
    for i in range(cfg.subsets):
        subset = [r for r in train if blocks.index(r["block"]) % cfg.subsets == i]
        T, result = fit(
            cam, subset, cam.ego_T_cam, cfg, identity, directory / f"subset_{i}"
        )
        transforms.append(T)
        results.append(result)
    return transforms[selected], transforms, results
